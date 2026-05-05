"""
GMT sequential competitive slot model — Branch B, Phase B2.

Architecture
------------
EdgeCompat encoder (identical to GMTEdgeCompat) produces context-enriched
stub embeddings, followed by a sequential slot decoder:

  Encoder (shared with EdgeCompat):
    1. Node encoder:   MLP(F → H)
    2. Edge encoder:   MLP(2H → 1) → σ      pairwise compatibility score
    3. Score-weighted stub aggregation
    4. Node updater:   MLP(2H → H)
    5. Node head:      Linear(H → 1)         signal/noise logit per stub

  Sequential slot decoder (K slots, processed in order):
    For slot k = 0, 1, 2:
      scores_k  = query_k · h_i + log(remaining_i)
      attn_k    = softmax(scores_k)           over valid stubs
      ctx_k     = Σ_i attn_k_i · h_i
      logit_k   = cand_head_k(ctx_k)
      claim_k   = σ(logit_k)                 confidence this slot fires
      remaining_i ← remaining_i · (1 − claim_k · attn_k_i)

The log(remaining) masking in logit space means: stubs claimed by earlier
slots appear with very negative scores and are effectively excluded from
later slot attention.  If a slot is not confident (low claim_k) it barely
modifies remaining, so later slots still see the full stub set.
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


class GMTSeqSlot(nn.Module):
    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()
        self.K = K
        H = hidden

        # EdgeCompat encoder
        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)
        self.edge_encoder = _mlp([2 * H, H, 1],      dropout)
        self.node_updater = _mlp([2 * H, H, H],       dropout)
        self.node_head    = nn.Linear(H, 1)

        # Per-slot learned queries and heads
        self.slot_queries = nn.Parameter(torch.randn(K, H) * 0.02)
        self.cand_heads   = nn.ModuleList([_mlp([H, H, 1], dropout) for _ in range(K)])
        self.pt_heads     = nn.ModuleList([_mlp([H, H, 1], dropout) for _ in range(K)])
        self.chg_heads    = nn.ModuleList([_mlp([H, H, 1], dropout) for _ in range(K)])

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

        ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)       # (N, Nmax, Nmax, H)
        ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)       # (N, Nmax, Nmax, H)
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

        # ---- sequential slot decoder ----
        remaining = valid_mask.float().clone()                     # (N, Nmax) in [0, 1]

        cand_logits_list: list[torch.Tensor] = []
        pt_pred_list:     list[torch.Tensor] = []
        chg_pred_list:    list[torch.Tensor] = []

        for k in range(self.K):
            query = self.slot_queries[k]                           # (H,)

            # scores: dot product + log-space remaining mask
            scores  = (node_upd * query).sum(dim=-1)              # (N, Nmax)
            scores  = scores + torch.log(remaining.clamp(min=1e-9))
            scores  = scores.masked_fill(~valid_mask, float("-inf"))

            attn  = torch.softmax(scores, dim=-1)                  # (N, Nmax)
            ctx_k = (attn.unsqueeze(-1) * node_upd).sum(dim=1)    # (N, H)

            cand_logit_k = self.cand_heads[k](ctx_k).squeeze(-1)  # (N,)
            pt_k   = F.softplus(self.pt_heads[k](ctx_k)).squeeze(-1)   # (N,)
            chg_k  = torch.tanh(self.chg_heads[k](ctx_k)).squeeze(-1)  # (N,)

            cand_logits_list.append(cand_logit_k)
            pt_pred_list.append(pt_k)
            chg_pred_list.append(chg_k)

            # soft claiming: down-weight stubs proportional to confidence × attention
            claim_k   = torch.sigmoid(cand_logit_k)               # (N,)
            remaining = remaining * (1.0 - claim_k.unsqueeze(-1) * attn)

        cand_logits = torch.stack(cand_logits_list, dim=-1)        # (N, K)
        pt_pred     = torch.stack(pt_pred_list,     dim=-1)        # (N, K)
        chg_pred    = torch.stack(chg_pred_list,    dim=-1)        # (N, K)

        return {
            "node_logit":       node_logit,
            "candidate_logits": cand_logits,
            "pt_pred":          pt_pred,
            "charge_pred":      chg_pred,
        }


def build_seq_slot(
    hidden: int = 64, K: int = K_MAX, dropout: float = 0.0
) -> GMTSeqSlot:
    return GMTSeqSlot(hidden=hidden, K=K, dropout=dropout)
