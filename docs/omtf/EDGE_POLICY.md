# OMTF Edge Sparsity Policy

**Status**: Decided  
**Date**: 2026-04-19  
**Applies to**: Stage 4 (edge_compat), Stage 5 (slot_model), Stage 7 (HLS)

---

## 1. Empirical basis

Analysis on 10 S1 files (50k same-track pairs, 5k windows):

### 1.1 Active layer-type mapping (observed in data)

| Layer | Type | r (HW units) | Notes |
|---|---|---|---|
| 0 | DT | 431 | DT barrel station 1, fixed r |
| 2 | DT | 512 | DT barrel station 2, fixed r |
| 4 | DT | 620 | DT barrel station 3, fixed r |
| 6 | CSC | 511–668 | CSC endcap, variable r (ring-dependent) |
| 7 | CSC | 376–680 | CSC endcap, variable r |
| 8 | CSC | 397–680 | CSC endcap, variable r |
| 9 | CSC | 287–451 | CSC endcap, variable r |
| 10 | RPC | 414 | RPC barrel, fixed r |
| 11 | RPC | 449 | RPC barrel, fixed r |
| 12 | RPC | 495 | RPC barrel, fixed r |
| 13 | RPC | 530 | RPC barrel, fixed r |
| 14 | RPC | 602 | RPC barrel, fixed r |
| 15 | RPC | 535–664 | RPC endcap, variable r |
| 16 | RPC | 534–670 | RPC endcap, variable r |
| 17 | RPC | 534–670 | RPC endcap, variable r |

Layers 1, 3, 5 are absent in positive-eta samples (likely reserved for negative-eta sub-sectors or DT inner chambers not populated in this geometry).

### 1.2 Same-track edge distribution

| Metric | Value |
|---|---|
| Total unique (layer_i, layer_j) pairs seen | 102 / 105 possible |
| Same-layer same-track pairs | 0.8% |
| Near-zero Δr (<20 HW) same-track pairs | 14.0% |
| Pairs to cover 80% of same-track edges | 45 |
| Pairs to cover 90% of same-track edges | 56 |
| Pairs to cover 95% of same-track edges | 64 |

**Key conclusion**: The same-track pair distribution is nearly flat (~1–3.5% per pair). 102 out of 105 possible layer pairs appear. There is no natural physics-based subset to exploit for layer-pair sparsity without incurring meaningful signal loss (>5% loss requires excluding ~38 pairs).

### 1.3 Practical edge counts

| Dataset | Mean valid stubs/window | Mean edges/window | p99 edges/window |
|---|---|---|---|
| S1 (clean signal) | 5.8 | 17 | 91 |
| B4 (PU200 noise) | 3.0 | 4 | 36 |

The theoretical worst case (Nmax=24, all-to-all) of 276 edges is never observed. The realistic p99 is 91 (S1). With Nmax padding, training processes up to 276 potential pairs per window, but most involve at least one padded (invalid) stub and are masked.

---

## 2. Stage 4 / Stage 5 edge policy: cross-layer only

**Rule**: exclude pairs where both stubs are in the same layer (`layer_i == layer_j`).

**Rationale**:
- Removes 0.8% of same-track signal edges — negligible signal loss
- Eliminates kappa_hat singularity (Δr = 0 exactly for fixed-r same-layer pairs)
- Removes 2–5% of all edges depending on dataset; minor training speedup
- Does not require layer identity at hardware routing time

**Implementation**: in `features.py`, add `layer_i != layer_j` guard to the pair loop, or mask out same-layer pairs in `compute_pair_features_np`.

---

## 3. Stage 7 firmware policy: layer-slot ordered + Δr threshold

At Stage 7, when building HLS kernels, the sparse policy tightens to enable static hardware instantiation.

### 3.1 Stub slot assignment (preprocessing step)

Sort stubs into fixed-position slots by layer number before feeding to the model. The 15 active layers map to 15 slot indices:

| Slot | Layer | Type | Fixed r |
|---|---|---|---|
| 0 | 0 | DT | 431 |
| 1 | 2 | DT | 512 |
| 2 | 4 | DT | 620 |
| 3 | 6 | CSC | variable |
| 4 | 7 | CSC | variable |
| 5 | 8 | CSC | variable |
| 6 | 9 | CSC | variable |
| 7 | 10 | RPC | 414 |
| 8 | 11 | RPC | 449 |
| 9 | 12 | RPC | 495 |
| 10 | 13 | RPC | 530 |
| 11 | 14 | RPC | 602 |
| 12 | 15 | RPC | variable |
| 13 | 16 | RPC | variable |
| 14 | 17 | RPC | variable |

Multiple stubs in the same layer: **initial firmware-oriented per-layer cap = 2** (top-2 by quality), with a global Nmax cap applied after. Strict one-stub-per-layer discards ambiguity information that is precisely what the model needs to resolve in S3/S4/S5/B3 — same-layer multiplicity often signals competing candidates or one-real-plus-one-fake situations. A tighter or detector-dependent per-layer cap (e.g. top-1 for certain low-value RPC barrel layers vs top-2 for DT/CSC) may be introduced later if compression demands it, but that is a future refinement, not the initial policy.

### 3.2 Static pair mask

**Do not choose the mask by pair frequency alone.**

Pair frequency in S1 reflects detector acceptance and generator kinematics for prompt single muons. It is biased toward common prompt geometry and under-represents pairs that are rare but critical for displaced reconstruction (S2/S5/B2), close dimuon separation (S3/S4/B3), and incomplete-layer boundary events. A rare pair can be more informative per occurrence than a common redundant one.

**Correct mixed criterion (in priority order):**

| Priority | Criterion | Operational definition |
|---|---|---|
| 1 | Geometric legitimacy | Non-negligible co-occurrence in truth-matched signal windows across **all nine datasets**; physically plausible radial/subsystem combination for one through-going muon in the OMTF eta/phi acceptance; not dominated by pathological same-object combinations. This must be turned into a reproducible threshold (e.g. ≥0.1% of same-track edges in ≥3 datasets) before the mask is frozen. |
| 2 | Information value | Candidate-level performance drop when this pair class is masked out — specifically: candidate recovery efficiency (S1/S3), close-dimuon separation (S3), displaced efficiency (S5/B2), B4 false positive rate. **Edge AUC is a diagnostic only**, not the decision metric; a pair that helps edge AUC but does not move candidate-level numbers is low priority. |
| 3 | Frequency | Tie-breaker after the above two criteria are applied. |

**Initial conservative target: 80–90 pairs retained** (out of 105). Reducing to 56 is a large step that cannot be assumed harmless before architecture-level evidence from the mask comparison below.

### 3.3 Edge classes for hardware

Short lever-arm pairs (e.g. adjacent RPC barrel layers with small |Δr|) have limited kappa_hat signal but can still carry useful information for pattern consistency, local support, noise suppression, and ambiguity resolution in dense windows. The correct hardware response is a **cheaper scoring path**, not unconditional removal.

**Two hardware edge classes:**

**Class A — full scorer** (large lever arm; primary curvature estimators)
- Full 6-feature input: Δphi, Δr, Δr², kappa_hat, |Δeta|, phiB_diff
- Full-cost MLP compute unit
- Candidate pairs: cross-station DT–DT, DT–CSC, DT–far-RPC, CSC–CSC with meaningful radial separation

**Class B — support scorer** (short lever arm or coarse detector combination)
- **Default**: reduced feature input omitting kappa_hat (unstable for near-zero Δr)
- **Validated experimentally**: whether a quantized, coarsened, or sign-only kappa_hat still contributes for a specific Class B subset. The reduced set is the starting point, not a permanent decision.
- Cheaper MLP or linear scorer
- Candidate pairs: adjacent RPC barrel, local CSC/RPC combinations
- Same-layer pairs still excluded even in this class

Class assignment per (slot_i, slot_j) is determined at Stage 7 by geometric legitimacy criterion (§3.2), not frequency.

### 3.4 What to measure before freezing the firmware mask

Run the following five mask variants. **Evaluate on all nine datasets.** Primary metrics are candidate-level, not edge AUC:

| Mask | Description |
|---|---|
| A | Cross-layer only, no further pruning — full software baseline |
| B | Cross-layer + Class A edges only (drop all short-lever support edges) |
| C | Top-56 by S1 frequency |
| D | Geometry-informed Class A + Class B, 80–90 pairs |
| E | Geometry-informed Class A + Class B, 80–90 pairs, top-2-per-layer input |

**Primary decision metrics** (in order): candidate recovery (S1/S3), dimuon separation (S3), displaced efficiency (S5/B2), B4 false positives, synthesis cost estimate.  
**Diagnostic only**: edge AUC.

Only after this comparison is the hardware mask frozen.

---

## 4. Summary table

| Stage | Policy | Signal loss | Notes |
|---|---|---|---|
| Stage 4–5 (software) | Cross-layer only | ~0% (0.8% same-layer pairs) | Simple, honest, keeps full relational view |
| Stage 7 (firmware) | Layer-slot ordered + Class A/B mask (80–90 pairs) + top-2/layer | TBD from mask comparison | Must be validated on all 9 datasets before freezing |

The firmware edge loss is not assumed to be harmless. It must be quantified by the five-mask comparison (§3.4) on candidate-level metrics across all nine datasets before any policy is frozen. Edge AUC alone is not sufficient evidence.

---

## 5. What this means for `edge_compat.py` (Stage 4)

The only change needed in `features.py` / `dataset.py` for Stage 4 is the same-layer exclusion. No other sparse changes required.

The firmware slot-ordering, Class A/B split, and static pair mask are Stage 7 decisions. This separation keeps Stage 4 honest and allows Stage 7 sparsification to be evaluated against the full-graph baseline.
