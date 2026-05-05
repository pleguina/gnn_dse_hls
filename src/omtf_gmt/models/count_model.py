"""
GMT count-then-build model — Branch B, Phase B2.

Architecture
------------
EdgeCompat encoder (identical to GMTEdgeCompat) produces context-enriched
stub embeddings, followed by two independent heads on the global context:

  Encoder (shared with EdgeCompat):
    1. Node encoder:   MLP(F → H)
    2. Edge encoder:   MLP(2H → 1) → σ      pairwise compatibility score
    3. Score-weighted stub aggregation
    4. Node updater:   MLP(2H → H)
    5. Node head:      Linear(H → 1)         signal/noise logit per stub
    6. Global pool:    masked mean of updated embeddings → (H,)

  Count head:       MLP(H → 4)              logits for n_cands ∈ {0,1,2,3}
  Candidate heads:  MLP(H → K)              candidate logits (as EdgeCompat)
  pT heads:         MLP(H → K)
  Charge heads:     MLP(H → K)

At inference the predicted count n = argmax(count_logits) suppresses slots
n..K-1 (set their candidate logits to -inf).  Training uses all K slots with
standard BCE targets (slot k positive iff gen_pt[k] > 0) plus a cross-entropy
count loss that directly supervises the count head.

This splits the multiplicity problem cleanly:
  - count head answers "how many muons?"
  - candidate heads answer "where is each muon?"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES

K_MAX = 3


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GMTCountModel(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        H = hidden

        # EdgeCompat encoder
        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)
        self.edge_encoder = _mlp([2 * H, H, 1],      dropout)
        self.node_updater = _mlp([2 * H, H, H],       dropout)
        self.node_head    = nn.Linear(H, 1)

        # Heads on global context
        self.count_head = _mlp([H, H, K + 1], dropout)   # logits for 0..K candidates
        self.cand_head  = _mlp([H, H, K],     dropout)
        self.pt_head    = _mlp([H, H, K],     dropout)
        self.chg_head   = _mlp([H, H, K],     dropout)

    def forward(
        self,
        stubs:      torch.Tensor,   # (N, Nmax, F)
        valid_mask: torch.Tensor,   # (N, Nmax) bool
        **_,
    ) -> dict[str, torch.Tensor]:
        N, Nmax, _ = stubs.shape
        device     = stubs.device

        # ---- EdgeCompat encoder ----
        node_emb = self.node_encoder(stubs)                        # (N, Nmax, H)

        ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)
        ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)
        edge_score = torch.sigmoid(
            self.edge_encoder(torch.cat([ei, ej], dim=-1)).squeeze(-1)
        )                                                          # (N, Nmax, Nmax)

        vm = valid_mask
        pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)
        no_self    = ~torch.eye(Nmax, dtype=torch.bool,
                                device=device).unsqueeze(0)
        edge_mask  = pair_valid & no_self

        w     = edge_score * edge_mask.float()
        denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
        w_n   = (w / denom).unsqueeze(-1)
        ctx   = (w_n * ej).sum(dim=2)                             # (N, Nmax, H)

        node_upd   = self.node_updater(
            torch.cat([node_emb, ctx], dim=-1)
        )                                                          # (N, Nmax, H)
        node_logit = self.node_head(node_upd).squeeze(-1)         # (N, Nmax)

        # ---- global pool ----
        vmf      = valid_mask.unsqueeze(-1).float()
        denom_g  = vmf.sum(dim=1).clamp(min=1.0)
        global_ctx = (node_upd * vmf).sum(dim=1) / denom_g        # (N, H)

        # ---- heads ----
        count_logits = self.count_head(global_ctx)                 # (N, K+1)
        cand_logits  = self.cand_head(global_ctx)                  # (N, K)
        pt_pred      = F.softplus(self.pt_head(global_ctx))        # (N, K)
        chg_pred     = torch.tanh(self.chg_head(global_ctx))       # (N, K)

        return {
            "node_logit":       node_logit,
            "count_logits":     count_logits,
            "candidate_logits": cand_logits,
            "pt_pred":          pt_pred,
            "charge_pred":      chg_pred,
        }


def build_count_model(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTCountModel:
    return GMTCountModel(hidden=hidden, K=K, dropout=dropout)
