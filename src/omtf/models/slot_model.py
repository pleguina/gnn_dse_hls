"""
OMTF Stage 5 -- Fixed-K slot model.

Architecture
------------
1. Per-stub encoder        MLP(7 -> H)
2. K learnable slot queries  nn.Parameter(K, H)  -- fixed at hardware time
3. Stub-to-slot compat       MLP(2H -> 1) per (slot, stub) pair
                             -> softmax over valid stubs -> attention a (K, Nmax)
4. Slot aggregation          slot_emb[k] = sum_i a[k,i] * stub_emb[i]   -> (K, H)
5. Node update               node_upd[i]  = sum_k a[k,i] * slot_emb[k]  -> (Nmax, H)
6. Combined node repr        concat(stub_emb, node_upd)                  -> (2H,)
7. Node head                 MLP(2H -> 1)  --  signal/noise logit per stub
8. Slot output heads         MLP(H -> 1) per head, per slot
                               candidate_logit  -- is this slot occupied?
                               pt_pred          -- log(pT) per slot
                               charge_pred      -- charge logit per slot
                               dxy_pred         -- dxy prediction per slot (cm)

Candidate supervision
---------------------
A proxy target for slot BCE when gen_pt is unavailable:
  slot_signal_mass[k] = sum_i a[k,i] * node_label[i]   (attention detached)
Computed externally by compute_slot_signal_mass() so that forward() is
completely label-free and safe for inference / export.

Inputs (per batch)
------------------
  stubs      : (B, Nmax, 7)  float32
  valid_mask : (B, Nmax)     bool

Outputs (dict)
--------------
  node_logits      : (B, Nmax)
  attn             : (B, K, Nmax)
  candidate_logits : (B, K)
  pt_pred          : (B, K)
  charge_pred      : (B, K)
  dxy_pred         : (B, K)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NODE_DIM = 7


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
# Slot model
# ---------------------------------------------------------------------------

class SlotModel(nn.Module):
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

        # 1. Per-stub encoder
        self.stub_encoder = _mlp(NODE_DIM, hidden, hidden, layers=3, dropout=dropout)

        # 2. Learnable slot queries
        self.slot_queries = nn.Parameter(torch.randn(K, hidden) * 0.02)

        # 3. Stub-to-slot compatibility scorer
        self.compat_mlp = _mlp(2 * hidden, hidden // 2, 1, layers=2, dropout=dropout)

        # 7. Node output head (from combined 2H repr)
        self.node_head = _mlp(2 * hidden, hidden // 2, 1, layers=2)

        # 8. Slot output heads
        self.candidate_head = _mlp(hidden, hidden // 2, 1, layers=2)
        self.pt_head        = _mlp(hidden, hidden // 2, 1, layers=2)
        self.charge_head    = _mlp(hidden, hidden // 2, 1, layers=2)
        self.dxy_head       = _mlp(hidden, hidden // 2, 1, layers=2)

    # ------------------------------------------------------------------

    def forward(
        self,
        stubs: Tensor,       # (B, Nmax, 7)
        valid_mask: Tensor,  # (B, Nmax) bool
    ) -> dict[str, Tensor]:
        B, Nmax, _ = stubs.shape

        # 1. Per-stub embedding
        stub_emb = self.stub_encoder(stubs)  # (B, Nmax, H)

        # 3. Stub-to-slot compatibility scores
        queries  = self.slot_queries.unsqueeze(0).unsqueeze(2)          # (1, K, 1, H)
        queries  = queries.expand(B, self.K, Nmax, self.hidden)
        stub_exp = stub_emb.unsqueeze(1).expand(B, self.K, Nmax, self.hidden)

        compat_input = torch.cat([queries, stub_exp], dim=-1)           # (B, K, Nmax, 2H)
        attn_raw = self.compat_mlp(compat_input).squeeze(-1)            # (B, K, Nmax)

        # Mask padding positions before softmax
        pad_mask = ~valid_mask.unsqueeze(1).expand(B, self.K, Nmax)    # True = pad
        attn_raw = attn_raw.masked_fill(pad_mask, -1e9)
        attn = F.softmax(attn_raw, dim=-1)  # (B, K, Nmax)

        # 4. Slot aggregation
        slot_emb = torch.bmm(attn, stub_emb)  # (B, K, H)

        # 5. Node update
        node_upd = torch.bmm(attn.transpose(1, 2), slot_emb)  # (B, Nmax, H)

        # 6. Combined node repr
        combined = torch.cat([stub_emb, node_upd], dim=-1)  # (B, Nmax, 2H)

        # 7. Per-stub signal/noise logit
        node_logits = self.node_head(combined).squeeze(-1)  # (B, Nmax)

        # 8. Slot output heads
        candidate_logits = self.candidate_head(slot_emb).squeeze(-1)  # (B, K)
        pt_pred          = self.pt_head(slot_emb).squeeze(-1)          # (B, K)
        charge_pred      = self.charge_head(slot_emb).squeeze(-1)      # (B, K)
        dxy_pred         = self.dxy_head(slot_emb).squeeze(-1)         # (B, K)

        return {
            "node_logits":       node_logits,
            "attn":              attn,
            "candidate_logits":  candidate_logits,
            "pt_pred":           pt_pred,
            "charge_pred":       charge_pred,
            "dxy_pred":          dxy_pred,
        }


# ---------------------------------------------------------------------------
# Training helper: proxy supervision target when gen_pt is unavailable
# ---------------------------------------------------------------------------

def compute_slot_signal_mass(
    attn: Tensor,       # (B, K, Nmax) -- detached attention weights
    node_label: Tensor, # (B, Nmax) float32
) -> Tensor:
    """
    slot_signal_mass[b,k] = sum_i attn[b,k,i] * node_label[b,i]
    Returns (B, K) clamped to [0, 1].
    Attention must be detached before calling so the proxy does not create
    a second gradient path through the attention weights.
    """
    return torch.bmm(
        attn.detach(),
        node_label.unsqueeze(-1),
    ).squeeze(-1).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_slot_model(
    hidden: int = 64,
    K: int = 3,
    dropout: float = 0.1,
) -> SlotModel:
    return SlotModel(hidden=hidden, K=K, dropout=dropout)
