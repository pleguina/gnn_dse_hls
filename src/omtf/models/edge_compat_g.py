"""
EdgeCompatNet for the OMTF-internal G-dataset study.

Architecture identical to src/omtf_gmt/models/edge_compat.py (GMT Phase B5
winner) but with a parametric input dimension so it can accept any number of
stub features.  The default n_features=14 reproduces the GMT checkpoint
interface; set n_features=11 for the OMTF-internal feature schema.

No graph pre-computation needed — edges are formed on-the-fly as all valid
stub pairs inside each window (cross-layer masking done in forward()).

Inputs
------
  stubs      : (N, Nmax, F)  float32
  valid_mask : (N, Nmax)     bool

Outputs (dict)
--------------
  node_logit       : (N, Nmax)
  candidate_logits : (N, K)
  pt_pred          : (N, K)   softplus
  charge_pred      : (N, K)   tanh
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX: int = 3


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeCompatG(nn.Module):
    """
    Parameters
    ----------
    n_features : input feature dimension per stub (default 14 for GMT compat)
    hidden     : embedding dimension
    K          : candidate output slots
    dropout    : dropout rate
    """

    def __init__(
        self,
        n_features: int = 14,
        hidden: int = 64,
        K: int = K_MAX,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.K = K
        H = hidden

        self.node_encoder = _mlp([n_features, H, H], dropout)
        self.edge_encoder = _mlp([2 * H, H, 1],      dropout)
        self.node_updater = _mlp([2 * H, H, H],       dropout)
        self.node_head    = nn.Linear(H, 1)
        self.cand_head    = _mlp([H, H, K], dropout)
        self.pt_head      = _mlp([H, H, K], dropout)
        self.chg_head     = _mlp([H, H, K], dropout)

    def forward(
        self,
        stubs:      torch.Tensor,   # (N, Nmax, F)
        valid_mask: torch.Tensor,   # (N, Nmax) bool
        **_,
    ) -> dict[str, torch.Tensor]:
        N, Nmax, _ = stubs.shape
        device = stubs.device

        node_emb = self.node_encoder(stubs)                        # (N, Nmax, H)

        ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)       # (N, Nmax, Nmax, H)
        ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)       # (N, Nmax, Nmax, H)
        edge_score = torch.sigmoid(
            self.edge_encoder(torch.cat([ei, ej], dim=-1)).squeeze(-1)
        )                                                          # (N, Nmax, Nmax)

        vm = valid_mask
        pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)
        no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask  = pair_valid & no_self

        w     = edge_score * edge_mask.float()
        denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
        w_n   = (w / denom).unsqueeze(-1)                         # (N, Nmax, Nmax, 1)
        ctx   = (w_n * ej).sum(dim=2)                             # (N, Nmax, H)

        node_upd = self.node_updater(torch.cat([node_emb, ctx], dim=-1))

        vmf      = valid_mask.unsqueeze(-1).float()
        denom_g  = vmf.sum(dim=1).clamp(min=1.0)
        global_ctx = (node_upd * vmf).sum(dim=1) / denom_g        # (N, H)

        node_logit  = self.node_head(node_upd).squeeze(-1)        # (N, Nmax)
        cand_logits = self.cand_head(global_ctx)                   # (N, K)
        pt_pred     = F.softplus(self.pt_head(global_ctx))        # (N, K)
        chg_pred    = torch.tanh(self.chg_head(global_ctx))       # (N, K)

        return {
            "node_logit":       node_logit,
            "candidate_logits": cand_logits,
            "pt_pred":          pt_pred,
            "charge_pred":      chg_pred,
        }


def build_edge_compat_g(
    n_features: int = 14,
    hidden: int = 64,
    K: int = K_MAX,
    dropout: float = 0.0,
) -> EdgeCompatG:
    return EdgeCompatG(n_features=n_features, hidden=hidden, K=K, dropout=dropout)
