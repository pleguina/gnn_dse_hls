"""
OMTF Stage 3 Baselines.

Baseline A — DeepSets
---------------------
  Per-stub MLP embed → masked global mean pool → node + candidate heads.
  Tests whether any muon information is recoverable without relational reasoning.

  Inputs : stubs (B, Nmax, 7), valid_mask (B, Nmax)
  Outputs: node_logits (B, Nmax), pt_pred (B, Nmax), candidate_logits (B, K)

Baseline B — Pure Edge MLP
--------------------------
  Per-pair MLP on derived edge features → edge binary score.
  Tests whether pair features alone are discriminating before adding aggregation.

  Inputs : edge_attr (E, 6)
  Outputs: edge_logits (E,)

Node feature order: [phi, phiB, eta, r, quality, type, layer]  (bx excluded)
Edge feature order: [delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, phiB_diff]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

NODE_DIM = 7   # phi, phiB, eta, r, quality, type, layer
EDGE_DIM = 6   # delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, phiB_diff


# --------------------------------------------------------------------------
# Shared building block
# --------------------------------------------------------------------------

def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2,
         dropout: float = 0.0) -> nn.Sequential:
    """Small MLP with LayerNorm on the first hidden layer."""
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


# --------------------------------------------------------------------------
# Baseline A — DeepSets
# --------------------------------------------------------------------------

class DeepSetsBaseline(nn.Module):
    """
    Baseline A: per-stub embedding + masked mean pool.

    Architecture
    ------------
    - phi_stub: MLP(7 → H → H) per stub
    - pool: valid-masked mean over stubs → global_ctx (H,)
    - node_head: MLP(2H → 1) per stub (concat stub_emb + broadcast global)
    - pt_head: MLP(2H → 1) per stub — predicts log(pT) for signal stubs
    - candidate_head: MLP(H → K) → K candidate scores (for efficiency proxy)

    Parameters
    ----------
    hidden    : embedding dimension (default 64)
    K         : number of output candidate slots (default 3)
    dropout   : dropout rate (default 0.1)
    """

    def __init__(self, hidden: int = 64, K: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.K = K

        # Per-stub encoder
        self.stub_encoder = _mlp(NODE_DIM, hidden, hidden, layers=3, dropout=dropout)

        # Per-stub heads (input: stub_emb concat global)
        self.node_head = _mlp(2 * hidden, hidden // 2, 1, layers=2)
        self.pt_head   = _mlp(2 * hidden, hidden // 2, 1, layers=2)

        # Candidate head (from global only)
        self.candidate_head = _mlp(hidden, hidden // 2, K, layers=2)

    def forward(
        self,
        stubs: Tensor,       # (B, Nmax, 7)
        valid_mask: Tensor,  # (B, Nmax) bool
    ) -> dict[str, Tensor]:
        """
        Returns
        -------
        node_logits     : (B, Nmax)   — signal/noise per stub
        pt_pred         : (B, Nmax)   — log(pT) prediction per stub
        candidate_logits: (B, K)      — K candidate scores from global
        """
        B, Nmax, _ = stubs.shape

        # Per-stub embed: (B, Nmax, H)
        stub_emb = self.stub_encoder(stubs)

        # Masked mean pool: (B, H)
        mask = valid_mask.unsqueeze(-1).float()  # (B, Nmax, 1)
        n_valid = mask.sum(dim=1).clamp(min=1)   # (B, 1)
        global_ctx = (stub_emb * mask).sum(dim=1) / n_valid  # (B, H)

        # Broadcast global to each stub: (B, Nmax, H)
        global_bcast = global_ctx.unsqueeze(1).expand(-1, Nmax, -1)

        # Concatenate stub + global: (B, Nmax, 2H)
        combined = torch.cat([stub_emb, global_bcast], dim=-1)

        # Per-stub outputs
        node_logits = self.node_head(combined).squeeze(-1)     # (B, Nmax)
        pt_pred     = self.pt_head(combined).squeeze(-1)       # (B, Nmax)

        # Candidate output from global
        candidate_logits = self.candidate_head(global_ctx)     # (B, K)

        return {
            "node_logits":      node_logits,
            "pt_pred":          pt_pred,
            "candidate_logits": candidate_logits,
        }


# --------------------------------------------------------------------------
# Baseline B — Pure edge MLP
# --------------------------------------------------------------------------

class EdgeMLPBaseline(nn.Module):
    """
    Baseline B: per-pair MLP on derived edge features → binary same-track score.

    No stub embedding, no global aggregation. Edge features only.
    Tests whether kappa_hat, phiB_diff, delta_r etc. are discriminating alone.

    Parameters
    ----------
    hidden  : MLP hidden dimension (default 64)
    layers  : number of MLP layers (default 3)
    dropout : dropout rate (default 0.1)
    """

    def __init__(self, hidden: int = 64, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = _mlp(EDGE_DIM, hidden, 1, layers=layers, dropout=dropout)

    def forward(self, edge_attr: Tensor) -> Tensor:
        """
        Parameters
        ----------
        edge_attr : (E, 6) — derived pair features

        Returns
        -------
        edge_logits : (E,) — raw logit per pair
        """
        return self.edge_mlp(edge_attr).squeeze(-1)

    def forward_batch(self, edge_attr_list: list[Tensor]) -> list[Tensor]:
        """Process a list of variable-length edge_attr tensors per sample."""
        return [self.forward(ea) for ea in edge_attr_list]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def build_baseline(
    name: str,
    hidden: int = 64,
    K: int = 3,
    dropout: float = 0.1,
    edge_layers: int = 3,
) -> nn.Module:
    """
    name : "deepsets" (Baseline A) or "edge_mlp" (Baseline B)
    """
    if name == "deepsets":
        return DeepSetsBaseline(hidden=hidden, K=K, dropout=dropout)
    elif name == "edge_mlp":
        return EdgeMLPBaseline(hidden=hidden, layers=edge_layers, dropout=dropout)
    else:
        raise ValueError(f"Unknown baseline: {name!r}. Choose 'deepsets' or 'edge_mlp'.")
