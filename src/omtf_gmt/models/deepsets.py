"""
GMT DeepSets baseline — Branch B smoke-test model.

Architecture:
  1. Per-stub encoder:   MLP(N_FEATURES → H)
  2. Valid-masked pool:  mean over valid stubs → global context (N, H)
  3. Per-node update:   concat(stub_emb, global) → MLP(2H → H) → node logit
  4. Candidate heads:   MLP(H → K) from global → candidate_logits (N, K)
  5. pT head:           MLP(H → K) from global → pt_pred (N, K) [GeV]
  6. Charge head:       MLP(H → K) from global → charge_pred (N, K) [±1]

Structurally mirrors the OMTF-internal DeepSets baseline so Branch A / Branch B
comparisons are architecture-controlled.
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


class GMTDeepSets(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        self.encoder   = _mlp([N_FEATURES, hidden, hidden], dropout)
        self.node_head = _mlp([2 * hidden, hidden, 1], dropout)
        self.cand_head = _mlp([hidden, hidden, K], dropout)
        self.pt_head   = _mlp([hidden, hidden, K], dropout)
        self.chg_head  = _mlp([hidden, hidden, K], dropout)

    def forward(
        self,
        stubs:      torch.Tensor,   # (N, Nmax, F)
        valid_mask: torch.Tensor,   # (N, Nmax) bool
        **_,
    ) -> dict[str, torch.Tensor]:
        N, Nmax, _ = stubs.shape

        stub_emb = self.encoder(stubs)                        # (N, Nmax, H)

        # masked mean pool
        vm = valid_mask.unsqueeze(-1).float()                 # (N, Nmax, 1)
        denom = vm.sum(dim=1).clamp(min=1.0)                  # (N, 1)
        global_ctx = (stub_emb * vm).sum(dim=1) / denom       # (N, H)

        # per-node logits
        global_exp = global_ctx.unsqueeze(1).expand(-1, Nmax, -1)
        node_in    = torch.cat([stub_emb, global_exp], dim=-1)
        node_logit = self.node_head(node_in).squeeze(-1)      # (N, Nmax)

        # candidate / pT / charge from global context
        cand_logits = self.cand_head(global_ctx)              # (N, K)
        pt_pred     = F.softplus(self.pt_head(global_ctx))    # (N, K)  [> 0]
        chg_pred    = torch.tanh(self.chg_head(global_ctx))   # (N, K)  [-1,1]

        return {
            "node_logit":      node_logit,
            "candidate_logits": cand_logits,
            "pt_pred":         pt_pred,
            "charge_pred":     chg_pred,
        }


def build_deepsets(hidden: int = 64, K: int = K_MAX, dropout: float = 0.0) -> GMTDeepSets:
    return GMTDeepSets(hidden=hidden, K=K, dropout=dropout)
