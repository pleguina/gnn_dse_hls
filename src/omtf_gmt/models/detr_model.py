"""
GMT DETR-style model — Branch B, Phase B2.

This is the corrected DETR-style candidate model for the GMT-visible-stub branch.

Purpose
-------
The earlier DeepSets, EdgeCompat, and independent slot models showed a persistent
multiplicity failure:

  * S2/B2 are true 1-candidate samples.
  * The models frequently predict 2 candidates.
  * Diagnostics showed that S2 is almost pure duplicate attention collapse:
    two slots attend to the same true muon.
  * B2 contains both duplicate-like failures and PU/noise fake candidates.

This model keeps the EdgeCompat encoder because it is already good at pairwise
compatibility and B4 fake rejection, but replaces the independent candidate head
with a DETR-style learned-query decoder. The matching loss is implemented outside
this file, in the training script, using Hungarian assignment.

Architecture
------------
Input:
  stubs      : Tensor [B, Nmax, F]
  valid_mask : BoolTensor [B, Nmax]

Encoder:
  1. Node encoder MLP:      F -> H -> H
  2. Pair scorer MLP:       [h_i, h_j] -> H -> 1
  3. Pair-score aggregation over neighbouring stubs
  4. Node updater MLP:      [h_i, aggregated_context_i] -> H -> H
  5. Node head:             H -> 1, per-stub signal/noise logit

DETR-style decoder:
  1. K learned object queries, one per possible output candidate.
  2. Cross-attention from queries to encoded stubs.
  3. Slot updater MLP receives both query identity and attended context.
  4. Per-slot heads produce:
       * candidate/no-object logit
       * pT prediction
       * charge prediction

Important implementation details
--------------------------------
1. The slot updater receives concat(query, context). Without this, two different
   queries that attend to the same stub cluster produce almost identical slot
   contexts, weakening slot identity.

2. The model returns attention weights. This is essential for the false-slot
   attribution diagnostic.

3. The masked softmax is protected with torch.nan_to_num. This avoids NaNs if an
   all-padding row ever appears.

4. This file intentionally does not implement the Hungarian loss. The model only
   produces predictions and diagnostic tensors. The training script should compute
   the set-based matching loss.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES

K_MAX = 3


# --------------------------------------------------------------------------- #
# Small helper MLP                                                            #
# --------------------------------------------------------------------------- #

def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    """
    Build a simple fully-connected ReLU MLP.

    Parameters
    ----------
    dims:
        Layer dimensions. Example: [F, H, H, 1].
    dropout:
        Dropout probability inserted after hidden ReLU layers only.

    Returns
    -------
    nn.Sequential
        MLP with Linear/ReLU/(Dropout) blocks.
    """
    layers: list[nn.Module] = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        is_hidden = i < len(dims) - 2

        layers.append(nn.Linear(in_dim, out_dim))

        if is_hidden:
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

class GMTDetrModel(nn.Module):
    """
    DETR-style GMT model with an EdgeCompat encoder.

    Parameters
    ----------
    hidden:
        Hidden dimension H used throughout the encoder and decoder.
    K:
        Maximum number of candidate slots. For the OMTF/GMT study this is 3.
    dropout:
        Dropout probability used inside MLP hidden layers.

    Forward input
    -------------
    stubs:
        FloatTensor [B, Nmax, F]. Padded node/stub features.
    valid_mask:
        BoolTensor [B, Nmax]. True for real stubs, False for padding.

    Forward output
    --------------
    Dictionary containing:

      node_logit:
        FloatTensor [B, Nmax]. Per-stub signal/noise logit.

      candidate_logits:
        FloatTensor [B, K]. Per-candidate muon/no-object logit.

      pt_pred:
        FloatTensor [B, K]. Positive pT prediction in GeV-like scale.

      charge_pred:
        FloatTensor [B, K]. Charge prediction in [-1, +1].

      attn_weights:
        FloatTensor [B, K, Nmax]. Decoder query attention over stubs.

      edge_score:
        FloatTensor [B, Nmax, Nmax]. Pairwise compatibility score before masking.

      edge_mask:
        BoolTensor [B, Nmax, Nmax]. Valid non-self pair mask.
    """

    def __init__(self, hidden: int = 64, K: int = K_MAX, dropout: float = 0.0):
        super().__init__()

        self.K = K
        self.H = hidden
        H = hidden

        # ------------------------------------------------------------------ #
        # EdgeCompat-style encoder                                            #
        # ------------------------------------------------------------------ #
        # Node embedding. Each stub is encoded independently first.
        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)

        # Pair compatibility scorer. This version uses learned node embeddings
        # only. If explicit pair features are later added, this input can become
        # [2*H + F_pair, H, 1].
        self.edge_encoder = _mlp([2 * H, H, 1], dropout)

        # Node updater after score-weighted neighbour aggregation.
        self.node_updater = _mlp([2 * H, H, H], dropout)

        # Per-node signal/noise classification head.
        self.node_head = nn.Linear(H, 1)

        # ------------------------------------------------------------------ #
        # DETR-style object-query decoder                                     #
        # ------------------------------------------------------------------ #
        # Learned candidate queries. They do not correspond to fixed physical
        # candidates. Hungarian matching in the loss makes their ordering
        # permutation-invariant.
        self.queries = nn.Parameter(torch.randn(K, H) * 0.02)

        # Important: the slot updater receives both the query identity and the
        # attended context. Without this, two queries attending the same stubs
        # produce nearly identical slot contexts.
        self.slot_updater = _mlp([2 * H, H, H], dropout)

        # Per-candidate heads.
        self.cand_head = _mlp([H, H, 1], dropout)
        self.pt_head = _mlp([H, H, 1], dropout)
        self.chg_head = _mlp([H, H, 1], dropout)

    def forward(
        self,
        stubs: torch.Tensor,
        valid_mask: torch.Tensor,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        """
        Run one forward pass.

        Parameters
        ----------
        stubs:
            FloatTensor [B, Nmax, F].
        valid_mask:
            BoolTensor [B, Nmax].

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs and diagnostics.
        """
        B, Nmax, _ = stubs.shape
        device = stubs.device
        H = self.H

        # Ensure mask is boolean. This avoids surprises if a cached dataset
        # stores valid_mask as uint8/float.
        valid_mask = valid_mask.bool()

        # ------------------------------------------------------------------ #
        # 1. Encode stubs                                                     #
        # ------------------------------------------------------------------ #
        node_emb = self.node_encoder(stubs)  # [B, Nmax, H]

        # ------------------------------------------------------------------ #
        # 2. Pairwise EdgeCompat score matrix                                 #
        # ------------------------------------------------------------------ #
        # ei[b, i, j, :] = h_i
        # ej[b, i, j, :] = h_j
        ei = node_emb.unsqueeze(2).expand(-1, -1, Nmax, -1)
        ej = node_emb.unsqueeze(1).expand(-1, Nmax, -1, -1)

        pair_input = torch.cat([ei, ej], dim=-1)  # [B, Nmax, Nmax, 2H]
        edge_score = torch.sigmoid(
            self.edge_encoder(pair_input).squeeze(-1)
        )  # [B, Nmax, Nmax]

        # Valid non-self pair mask.
        pair_valid = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        no_self = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask = pair_valid & no_self

        # ------------------------------------------------------------------ #
        # 3. Score-weighted neighbour aggregation                             #
        # ------------------------------------------------------------------ #
        # w[b, i, j] = compatibility from i to j, zero for invalid pairs.
        w = edge_score * edge_mask.float()

        # Normalise over j for each receiver i. If a node has no neighbours,
        # denom clamps to avoid divide-by-zero and context becomes zero.
        denom = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
        w_norm = (w / denom).unsqueeze(-1)  # [B, Nmax, Nmax, 1]

        # ctx_i = sum_j w_ij * h_j
        neighbour_ctx = (w_norm * ej).sum(dim=2)  # [B, Nmax, H]

        # ------------------------------------------------------------------ #
        # 4. Node update and node head                                        #
        # ------------------------------------------------------------------ #
        node_upd = self.node_updater(
            torch.cat([node_emb, neighbour_ctx], dim=-1)
        )  # [B, Nmax, H]

        node_logit = self.node_head(node_upd).squeeze(-1)  # [B, Nmax]

        # ------------------------------------------------------------------ #
        # 5. DETR-style cross-attention decoder                               #
        # ------------------------------------------------------------------ #
        # queries: [K, H] -> [B, K, H]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        # scores[b, k, i] = q_k dot h_i / sqrt(H)
        scores = torch.einsum("bkh,bih->bki", queries, node_upd) / math.sqrt(H)

        # Padding stubs cannot be attended.
        scores = scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)  # [B, K, Nmax]

        # Guard against all-padding windows. Ideally these should not occur,
        # but this keeps the model numerically safe.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # slot_ctx[b, k] = sum_i attn[b, k, i] * h_i
        slot_ctx = torch.einsum("bki,bih->bkh", attn_weights, node_upd)  # [B, K, H]

        # Fuse query identity and attended context.
        slot_emb = self.slot_updater(
            torch.cat([queries, slot_ctx], dim=-1)
        )  # [B, K, H]

        # ------------------------------------------------------------------ #
        # 6. Per-slot heads                                                   #
        # ------------------------------------------------------------------ #
        candidate_logits = self.cand_head(slot_emb).squeeze(-1)  # [B, K]

        # pT is positive. softplus is smooth and avoids negative predictions.
        pt_pred = F.softplus(self.pt_head(slot_emb)).squeeze(-1)  # [B, K]

        # Charge is represented as a continuous value in [-1, +1].
        charge_pred = torch.tanh(self.chg_head(slot_emb)).squeeze(-1)  # [B, K]

        return {
            "node_logit": node_logit,
            "candidate_logits": candidate_logits,
            "pt_pred": pt_pred,
            "charge_pred": charge_pred,
            "attn_weights": attn_weights,
            "edge_score": edge_score,
            "edge_mask": edge_mask,
        }


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #

def build_detr_model(
    hidden: int = 64,
    K: int = K_MAX,
    dropout: float = 0.0,
) -> GMTDetrModel:
    """
    Factory used by the training/evaluation scripts.
    """
    return GMTDetrModel(hidden=hidden, K=K, dropout=dropout)
