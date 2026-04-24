"""
OMTF Stage 4 -- Edge-compatibility network.

Architecture
------------
1. Per-stub encoder        MLP(7 -> H)
2. Per-edge scorer         MLP(2H + 6 -> H) -> edge embedding + scalar logit
3. Per-node aggregation    mean-pool incident edge embeddings (both directions)
4. Updated node repr       concat(stub_emb, node_update)  ->  (2H,)
5. Node head               MLP(2H -> 1)  --  signal/noise logit per stub
6. Slot-query attention    K learned queries (K, 2H) attend over updated stubs
                           -> K slot contexts (B, K, 2H)
7. K candidate heads       per-slot MLP(2H -> 1) for candidate score, pT, charge, dxy

The slot-query stage (6-7) replaces the original global mean-pool candidate heads.
Each slot independently attends to updated stubs via scaled dot-product attention,
giving real slot specialization and competition for stubs between candidates.

Edge policy: cross-layer pairs only (same-layer excluded).

Inputs  (per sample, assembled by collate_omtf)
-------
  stubs      : (B, Nmax, 7)   float32
  valid_mask : (B, Nmax)      bool
  edge_index : list[Tensor]   each (2, E_i) int64
  edge_attr  : list[Tensor]   each (E_i, 6) float32

Outputs (dict)
-------
  node_logits      : (B, Nmax)
  edge_logits      : list[Tensor] each (E_i,)
  attn             : (B, K, Nmax)  -- slot attention weights
  candidate_logits : (B, K)
  pt_pred          : (B, K)
  charge_pred      : (B, K)
  dxy_pred         : (B, K)

Node feature order : [phi, phiB, eta, r, quality, type, layer]
Edge feature order : [delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, phiB_diff]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NODE_DIM = 7
EDGE_DIM = 6


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2,
         dropout: float = 0.0) -> nn.Sequential:
    mods: list[nn.Module] = [
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        mods.append(nn.Dropout(dropout))
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            mods.append(nn.Dropout(dropout))
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


# ---------------------------------------------------------------------------
# Edge-compatibility network
# ---------------------------------------------------------------------------

class EdgeCompatNet(nn.Module):
    """
    Parameters
    ----------
    hidden   : embedding dimension (default 64)
    K        : number of output candidate slots (default 3)
    dropout  : dropout rate (default 0.1)
    """

    def __init__(self, hidden: int = 64, K: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.K = K
        self._attn_scale = math.sqrt(2 * hidden)

        # 1. Per-stub encoder
        self.stub_encoder = _mlp(NODE_DIM, hidden, hidden, layers=3, dropout=dropout)

        # 2. Edge scorer: takes concat(emb_i, emb_j, edge_attr) -> edge embedding
        self.edge_encoder   = _mlp(2 * hidden + EDGE_DIM, hidden, hidden,
                                   layers=3, dropout=dropout)
        self.edge_score_head = nn.Linear(hidden, 1)

        # 5. Node output head (from updated 2H repr)
        self.node_head = _mlp(2 * hidden, hidden // 2, 1, layers=2)

        # 6. Slot queries for candidate attention over updated stubs
        self.slot_queries = nn.Parameter(torch.randn(K, 2 * hidden) * 0.02)

        # 7. Per-slot candidate heads (input dim = 2H, one head per output type)
        self.candidate_head = _mlp(2 * hidden, hidden // 2, 1, layers=2)
        self.pt_head        = _mlp(2 * hidden, hidden // 2, 1, layers=2)
        self.charge_head    = _mlp(2 * hidden, hidden // 2, 1, layers=2)
        self.dxy_head       = _mlp(2 * hidden, hidden // 2, 1, layers=2)

    # ------------------------------------------------------------------

    def forward(
        self,
        stubs: Tensor,              # (B, Nmax, 7)
        valid_mask: Tensor,         # (B, Nmax) bool
        edge_index: list[Tensor],   # list of (2, E_i)
        edge_attr:  list[Tensor],   # list of (E_i, 6)
    ) -> dict[str, Tensor | list[Tensor]]:
        B, Nmax, _ = stubs.shape
        device = stubs.device

        # 1. Per-stub embedding for whole batch
        stub_emb  = self.stub_encoder(stubs)         # (B, Nmax, H)
        stub_flat = stub_emb.view(B * Nmax, -1)      # (B*Nmax, H)

        # 2-4. Batched edge processing
        edge_counts = [ei.shape[1] for ei in edge_index]
        has_edges   = [c > 0 for c in edge_counts]

        if any(has_edges):
            offset_edges = [
                edge_index[b].to(device) + b * Nmax
                for b in range(B) if has_edges[b]
            ]
            cat_edge_index = torch.cat(offset_edges, dim=1)          # (2, E_total)
            cat_edge_attr  = torch.cat(
                [edge_attr[b].to(device) for b in range(B) if has_edges[b]], dim=0
            )                                                          # (E_total, 6)

            src, dst  = cat_edge_index[0], cat_edge_index[1]
            edge_input = torch.cat([stub_flat[src], stub_flat[dst], cat_edge_attr], dim=-1)
            edge_emb   = self.edge_encoder(edge_input)                # (E_total, H)
            all_edge_logit_flat = self.edge_score_head(edge_emb).squeeze(-1)  # (E_total,)

            node_updates_flat = torch.zeros(B * Nmax, self.hidden, device=device)
            count_flat        = torch.zeros(B * Nmax, 1, device=device)
            ones = torch.ones(src.shape[0], 1, device=device)
            node_updates_flat.index_add_(0, src, edge_emb)
            node_updates_flat.index_add_(0, dst, edge_emb)
            count_flat.index_add_(0, src, ones)
            count_flat.index_add_(0, dst, ones)
            node_updates_flat = node_updates_flat / count_flat.clamp(min=1)
        else:
            all_edge_logit_flat = None
            node_updates_flat   = torch.zeros(B * Nmax, self.hidden, device=device)

        # Split edge logits back per sample
        all_edge_logits: list[Tensor] = []
        flat_idx = 0
        for b in range(B):
            c = edge_counts[b]
            if c > 0 and all_edge_logit_flat is not None:
                all_edge_logits.append(all_edge_logit_flat[flat_idx:flat_idx + c])
                flat_idx += c
            else:
                all_edge_logits.append(torch.zeros(0, device=device))

        node_updates = node_updates_flat.view(B, Nmax, self.hidden)  # (B, Nmax, H)

        # 4. Updated node representation
        updated = torch.cat([stub_emb, node_updates], dim=-1)  # (B, Nmax, 2H)

        # 5. Per-stub signal/noise logit
        node_logits = self.node_head(updated).squeeze(-1)  # (B, Nmax)

        # 6. Slot-query attention over updated stubs (scaled dot-product)
        #    slot_queries: (K, 2H) -- treated as (1, K, 2H) for broadcasting
        #    updated:      (B, Nmax, 2H)
        #    attn_raw:     (B, K, Nmax) = queries @ updated^T / sqrt(2H)
        queries  = self.slot_queries.unsqueeze(0)                        # (1, K, 2H)
        attn_raw = torch.bmm(
            queries.expand(B, self.K, 2 * self.hidden),
            updated.transpose(1, 2),                                     # (B, 2H, Nmax)
        ) / self._attn_scale                                             # (B, K, Nmax)

        pad_mask = ~valid_mask.unsqueeze(1).expand(B, self.K, Nmax)     # True = pad
        attn_raw = attn_raw.masked_fill(pad_mask, -1e9)
        attn = F.softmax(attn_raw, dim=-1)                              # (B, K, Nmax)

        # Slot contexts: weighted sum of updated stubs
        slot_ctx = torch.bmm(attn, updated)                             # (B, K, 2H)

        # 7. Per-slot candidate heads
        candidate_logits = self.candidate_head(slot_ctx).squeeze(-1)   # (B, K)
        pt_pred          = self.pt_head(slot_ctx).squeeze(-1)           # (B, K)
        charge_pred      = self.charge_head(slot_ctx).squeeze(-1)       # (B, K)
        dxy_pred         = self.dxy_head(slot_ctx).squeeze(-1)          # (B, K)

        return {
            "node_logits":       node_logits,
            "edge_logits":       all_edge_logits,
            "attn":              attn,
            "candidate_logits":  candidate_logits,
            "pt_pred":           pt_pred,
            "charge_pred":       charge_pred,
            "dxy_pred":          dxy_pred,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_edge_compat(
    hidden: int = 64,
    K: int = 3,
    dropout: float = 0.1,
) -> EdgeCompatNet:
    return EdgeCompatNet(hidden=hidden, K=K, dropout=dropout)
