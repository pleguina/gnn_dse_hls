"""
GMT fixed-K slot model — Branch B, Phase B3, v2.

Main changes vs v1
------------------
1. Learned NULL token:
   Each slot can attend to a learned "empty candidate" token instead of being
   forced to attend to real stubs. This is essential for 0-candidate and
   1-candidate windows.

2. Slot-specific learned queries:
   One query per output candidate slot. Each query attends independently over
   the real stubs plus the NULL token.

3. Node head uses only real-stub attention:
   The node classifier receives the maximum attention assigned to each real
   stub by any candidate slot. NULL attention is excluded from the node head.

4. Training helpers included at bottom:
   - attention_diversity_loss(...)
   - candidate_count_loss(...)

The helper losses are training-only. They do not add firmware cost.

Expected behavior
-----------------
- B4 / 0-candidate windows:
    all slots should learn high NULL attention and low candidate logits.

- S2/B2 / 1-candidate windows:
    slot 0 should attend to the real muon stubs;
    slots 1 and 2 should prefer the NULL token.

- S4 / 3-candidate windows:
    all three slots should attend to distinct real-stub clusters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from omtf_gmt.features import N_FEATURES

K_MAX = 3


def _mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    """
    Build a simple ReLU MLP.

    Example:
        _mlp([9, 64, 64]) means:
            Linear(9 -> 64) -> ReLU -> Linear(64 -> 64)
    """
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class GMTSlotModel(nn.Module):
    """
    Fixed-K slot model for GMT-visible-stub candidate reconstruction.

    Parameters
    ----------
    hidden:
        Internal embedding width H.

    K:
        Number of candidate output slots. For OMTF/GMT studies this is usually 3.

    dropout:
        Dropout probability inside MLP blocks.

    use_null_token:
        If True, append one learned NULL token to the attention memory. Empty
        slots can attend to this token. Keep this True unless doing an ablation.
    """

    def __init__(
        self,
        hidden: int = 64,
        K: int = K_MAX,
        dropout: float = 0.0,
        use_null_token: bool = True,
    ):
        super().__init__()

        self.K = K
        self.hidden = hidden
        self.use_null_token = use_null_token
        H = hidden

        # ------------------------------------------------------------------
        # 1. Encode each stub independently.
        # ------------------------------------------------------------------
        self.node_encoder = _mlp([N_FEATURES, H, H], dropout)

        # ------------------------------------------------------------------
        # 2. One learned query per candidate slot.
        #
        # Small random initialization prevents all slots from starting exactly
        # identical while keeping the initial dot products small.
        # ------------------------------------------------------------------
        self.slot_queries = nn.Parameter(torch.randn(K, H) * 0.02)

        # ------------------------------------------------------------------
        # 3. Learned NULL token.
        #
        # This represents "no real candidate for this slot". With plain
        # softmax attention, every slot must place probability mass somewhere.
        # The NULL token gives empty slots a meaningful place to attend.
        # ------------------------------------------------------------------
        if use_null_token:
            self.null_token = nn.Parameter(torch.zeros(1, 1, H))
        else:
            self.register_parameter("null_token", None)

        # ------------------------------------------------------------------
        # 4. Convert query + attended context into final slot embedding.
        # ------------------------------------------------------------------
        self.slot_updater = _mlp([2 * H, H, H], dropout)

        # ------------------------------------------------------------------
        # 5. Per-slot heads.
        #
        # candidate_logits: raw logits for BCEWithLogitsLoss.
        # pt_pred: positive pT prediction through softplus.
        # charge_pred: charge-like signed output in [-1, +1].
        # ------------------------------------------------------------------
        self.cand_head = nn.Linear(H, 1)
        self.pt_head = nn.Linear(H, 1)
        self.chg_head = nn.Linear(H, 1)

        # ------------------------------------------------------------------
        # 6. Per-node head.
        #
        # The extra scalar is max attention received by that stub from any
        # candidate slot. This links node-level signal/noise prediction to the
        # candidate assignment mechanism.
        # ------------------------------------------------------------------
        self.node_head = _mlp([H + 1, max(1, H // 2), 1], dropout)

    def forward(
        self,
        stubs: torch.Tensor,       # [B, Nmax, F]
        valid_mask: torch.Tensor,  # [B, Nmax], bool
        **_,
    ) -> dict[str, torch.Tensor]:
        B, Nmax, _ = stubs.shape
        H = self.hidden

        # --------------------------------------------------------------
        # Encode real stubs.
        # node_emb: [B, Nmax, H]
        # --------------------------------------------------------------
        node_emb = self.node_encoder(stubs)

        # --------------------------------------------------------------
        # Build attention memory.
        #
        # If use_null_token=True:
        #   memory = [real stubs, NULL]
        #   memory_valid = [valid real stubs, True]
        #
        # The NULL token is always valid.
        # --------------------------------------------------------------
        if self.use_null_token:
            null_emb = self.null_token.expand(B, 1, H)  # [B, 1, H]
            memory = torch.cat([node_emb, null_emb], dim=1)  # [B, Nmax+1, H]

            null_valid = torch.ones(B, 1, dtype=torch.bool, device=valid_mask.device)
            memory_valid = torch.cat([valid_mask.bool(), null_valid], dim=1)  # [B, Nmax+1]
        else:
            memory = node_emb
            memory_valid = valid_mask.bool()

        Nmem = memory.shape[1]

        # --------------------------------------------------------------
        # Expand learned queries over batch.
        # queries: [B, K, H]
        # --------------------------------------------------------------
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)

        # --------------------------------------------------------------
        # Scaled dot-product attention over memory.
        # attn_raw: [B, K, Nmem]
        # --------------------------------------------------------------
        attn_raw = torch.bmm(queries, memory.transpose(1, 2)) / (H ** 0.5)

        # Mask invalid real-stub padding positions.
        # The NULL token, if present, remains valid.
        pad_mask = ~memory_valid.unsqueeze(1)  # [B, 1, Nmem]
        attn_raw = attn_raw.masked_fill(pad_mask, float("-inf"))

        # Softmax normalizes attention for each slot.
        # Guard nan in pathological all-padding cases.
        attn_all = torch.softmax(attn_raw, dim=-1)
        attn_all = torch.nan_to_num(attn_all, nan=0.0, posinf=0.0, neginf=0.0)

        # --------------------------------------------------------------
        # Split real-stub attention from NULL attention.
        #
        # attn_real is what we use for node diagnostics and diversity.
        # null_attn tells us whether each slot chose the empty token.
        # --------------------------------------------------------------
        if self.use_null_token:
            attn_real = attn_all[:, :, :Nmax]      # [B, K, Nmax]
            null_attn = attn_all[:, :, Nmax]       # [B, K]
        else:
            attn_real = attn_all                   # [B, K, Nmax]
            null_attn = torch.zeros(B, self.K, dtype=attn_all.dtype, device=attn_all.device)

        # --------------------------------------------------------------
        # Slot contexts and slot embeddings.
        # slot_ctx: [B, K, H]
        # slot_emb: [B, K, H]
        # --------------------------------------------------------------
        slot_ctx = torch.bmm(attn_all, memory)
        slot_emb = self.slot_updater(torch.cat([queries, slot_ctx], dim=-1))

        # --------------------------------------------------------------
        # Candidate heads.
        # --------------------------------------------------------------
        candidate_logits = self.cand_head(slot_emb).squeeze(-1)      # [B, K]
        pt_pred = F.softplus(self.pt_head(slot_emb)).squeeze(-1)     # [B, K]
        charge_pred = torch.tanh(self.chg_head(slot_emb)).squeeze(-1)  # [B, K]

        # --------------------------------------------------------------
        # Node head.
        #
        # Only real-stub attention is used. NULL attention is not a node.
        # Padding stubs are masked by multiplying max_attn with valid_mask.
        # --------------------------------------------------------------
        max_attn = attn_real.max(dim=1).values  # [B, Nmax]
        max_attn = max_attn * valid_mask.float()

        node_in = torch.cat([node_emb, max_attn.unsqueeze(-1)], dim=-1)
        node_logit = self.node_head(node_in).squeeze(-1)  # [B, Nmax]

        return {
            "node_logit": node_logit,
            "candidate_logits": candidate_logits,
            "pt_pred": pt_pred,
            "charge_pred": charge_pred,

            # Diagnostics / optional training losses.
            "attn_weights": attn_real,  # [B, K, Nmax], real stubs only
            "null_attn": null_attn,     # [B, K]
            "slot_emb": slot_emb,       # [B, K, H]
        }


# ---------------------------------------------------------------------------
# Training-only helper losses
# ---------------------------------------------------------------------------

def candidate_count_loss(
    candidate_logits: torch.Tensor,  # [B, K]
    gen_pt: torch.Tensor,            # [B, K], 0 for empty slots
) -> torch.Tensor:
    """
    Penalize wrong predicted candidate multiplicity.

    This directly attacks the observed Phase B1 failure mode:
        true 1-candidate windows were often predicted as 2-candidate windows.

    The predicted count is the sum of candidate probabilities, not a hard
    thresholded count, so the loss is differentiable.
    """
    pred_count = torch.sigmoid(candidate_logits).sum(dim=1)  # [B]
    true_count = (gen_pt > 0).float().sum(dim=1)              # [B]
    return F.mse_loss(pred_count, true_count)


def attention_diversity_loss(
    attn_weights: torch.Tensor,  # [B, K, Nmax], real stubs only
    valid_mask: torch.Tensor,    # [B, Nmax]
) -> torch.Tensor:
    """
    Penalize different slots attending to the same real stubs.

    For each slot pair (k,l), compute the dot product between their attention
    distributions over real stubs. If two slots attend to the same cluster, the
    overlap is high. If they attend to different stubs or one goes to NULL, the
    real-stub overlap is small.

    Important:
        This should be used with a small weight, e.g. 0.05–0.1. Too much
        diversity pressure can push empty slots to attend to random noise.
    """
    B, K, Nmax = attn_weights.shape

    # Padding should not contribute to overlap.
    attn = attn_weights * valid_mask.unsqueeze(1).float()

    loss = attn_weights.new_tensor(0.0)
    n_pairs = 0

    for k in range(K):
        for l in range(k + 1, K):
            overlap = (attn[:, k, :] * attn[:, l, :]).sum(dim=1)  # [B]
            loss = loss + overlap.mean()
            n_pairs += 1

    if n_pairs == 0:
        return loss
    return loss / n_pairs


def null_attention_empty_slot_loss(
    null_attn: torch.Tensor,          # [B, K]
    gen_pt: torch.Tensor,             # [B, K]
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Optional auxiliary loss: encourage empty slots to use the NULL token.

    This is intentionally optional. The candidate BCE and count loss may already
    be enough. If empty slots still attend to real stubs and create false
    candidates, add this loss with a small weight, e.g. 0.05.

    For empty target slots, we encourage null_attn > margin.
    For positive target slots, we do not force null_attn low here because the
    candidate/regression losses already do that indirectly.
    """
    empty = gen_pt <= 0
    if not empty.any():
        return null_attn.new_tensor(0.0)

    # Penalize only when empty slots have too little NULL attention.
    penalty = F.relu(margin - null_attn[empty])
    return penalty.mean()


def attention_supervision_loss(
    attn_weights: torch.Tensor,  # [B, K, Nmax] real-stub attention
    null_attn:    torch.Tensor,  # [B, K]
    track_id:     torch.Tensor,  # [B, Nmax] int; 0=noise, k+1=gen muon k
    valid_mask:   torch.Tensor,  # [B, Nmax] bool
    gen_pt:       torch.Tensor,  # [B, K] float; 0 if slot empty
) -> torch.Tensor:
    """
    Directly supervise where each slot should attend using track_id ground truth.

    Occupied slot k  →  concentrate real-stub attention uniformly over stubs of
                         track_id == k+1; push null_attn toward 0.
    Empty slot k     →  push null_attn toward 1 (attend to NULL, not to any real stub).

    This is the strongest anti-duplication signal available: slot 1 in an S2 window
    is explicitly told its target distribution has all mass on the NULL token.

    Uses MSE over attention distributions (simpler than KL, avoids log instability).
    Recommended weight range: 0.1–1.0.  Start at 0.5.
    """
    B, K, Nmax = attn_weights.shape
    tid    = track_id.long()
    losses: list[torch.Tensor] = []

    for k in range(K):
        occupied = gen_pt[:, k] > 0   # [B]
        empty    = ~occupied

        # occupied: uniform target over muon-k stubs + null_attn → 0
        if occupied.any():
            muon_mask = (tid == k + 1) & valid_mask               # [B, Nmax]
            n_muon    = muon_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            tgt_real  = muon_mask.float() / n_muon                # [B, Nmax]

            losses.append(F.mse_loss(attn_weights[occupied, k], tgt_real[occupied]))
            losses.append(F.mse_loss(
                null_attn[occupied, k],
                torch.zeros(int(occupied.sum()), device=null_attn.device),
            ))

        # empty: all attention should go to NULL
        if empty.any():
            losses.append(F.mse_loss(
                null_attn[empty, k],
                torch.ones(int(empty.sum()), device=null_attn.device),
            ))

    if not losses:
        return attn_weights.new_tensor(0.0)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_slot_model(
    hidden: int = 64,
    K: int = K_MAX,
    dropout: float = 0.0,
    use_null_token: bool = True,
) -> GMTSlotModel:
    return GMTSlotModel(
        hidden=hidden,
        K=K,
        dropout=dropout,
        use_null_token=use_null_token,
    )
