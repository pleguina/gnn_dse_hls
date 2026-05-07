# OMTF-Internal Input View Comparison Plan

## 1. Goal

We already compared two GMT-visible input views:

* **GMT-KMTF view**: barrel KMTF stubs from NanoAOD, truth transferred from OMTF stubs.
* **GMT-TPS view**: TPS / EMTF-hybrid stubs from NanoAOD, truth transferred from OMTF stubs.

The next useful ablation is:

* **OMTF-internal view**: use the stubs directly from `OMTFAllInputTree`, i.e. the native OMTF trigger input representation.

The objective is to answer:

> If the model sees the same internal OMTF variables that the OMTF firmware already uses, can it match or beat the TPS model on signal efficiency, displaced muons, hard-negative rejection, and multi-candidate recovery?

This is not yet the FPGA implementation step. This is still a **physics/input-view comparison** before QAT and hardware deployment.

---

## 2. Why this comparison is important

The TPS results are very strong:

* Better G1/G3/G4 signal efficiency than KMTF.
* Much lower G8 PU hard-negative false-positive rate.
* Good displaced-muon efficiency.
* B4 pure-noise FP remains zero.

However, TPS uses a different upstream stub source. Before committing to it as the final trigger input, we should compare it against the native OMTF representation.

The OMTF-internal view has several advantages:

* It uses the actual OMTF processor-window input.
* It already contains `reg_stub_trackId`, so no truth-transfer approximation is needed.
* It uses variables closer to the firmware implementation: local phi, phiB, etaHw, r, quality, type, layer.
* It may be easier to deploy in the trigger if the model is placed inside or near the OMTF processing chain.

So the comparison should be:

| Input view    | Source                        | Truth labels              | Purpose                      |
| ------------- | ----------------------------- | ------------------------- | ---------------------------- |
| KMTF          | NanoAOD `MuonStubKmtf_*`      | transferred from OMTF     | barrel-stub baseline         |
| TPS           | NanoAOD `MuonStubTps_*`       | transferred from OMTF     | endcap/hybrid-stub candidate |
| OMTF-internal | `OMTFAllInputTree reg_stub_*` | native `reg_stub_trackId` | closest-to-firmware ablation |

---

## 3. Important point: do we need regioning?

For the OMTF-internal cache, probably **no extra regioning is needed**.

Reason:

* `OMTFAllInputTree` already has one entry per OMTF processor window.
* Each entry already corresponds to a specific `reg_iProcessor`.
* The `reg_stub_phiHw` values are already in the local processor frame.
* The stubs in one entry are already the stubs selected for that OMTF processor region.

So unlike KMTF/TPS, we do **not** need to do:

```text
select NanoAOD stubs inside the processor phi window
```

because OMTF already did that upstream.

For KMTF/TPS we needed regioning because the NanoAOD stub tables are event-level collections. We had to select the stubs whose global phi fell into the relevant OMTF processor window.

For OMTF-internal:

```text
ROOT entry = already one processor window
```

Therefore the cache builder should simply process each `OMTFAllInputTree` entry directly.

However, for fair comparison with KMTF/TPS, we must keep the **same logical sample definition**:

* same G datasets,
* same train/validation split seed,
* same candidate target convention,
* same hard-negative treatment,
* same threshold scans,
* same metrics.

So: **no extra phi-window regioning**, but **same evaluation protocol**.

---

## 4. What must match TPS/KMTF for a fair comparison

The comparison must be fair at the level of the training and evaluation setup.

### Must match

* Same physical datasets:

  * `G1_pos`, `G1_neg`
  * `G2_pos`, `G2_neg`
  * `G3_pos`, `G3_neg`
  * `G4_pos`, `G4_neg`
  * `G5_pos`, `G5_neg`
  * `G6_pos`, `G6_neg`
  * `G7`, `G8`, `B4`

* Same train/validation split:

  * 85% train
  * 15% validation
  * random seed `42`

* Same model family:

  * start with `EdgeCompat h64`
  * optionally compare `EdgeCompat h128`

* Same operating threshold:

  * first compare at logit threshold `0.0`
  * then include threshold scan

* Same metrics:

  * G1/G2 prompt efficiency
  * G3/G4 displaced efficiency
  * G5 two-target recovery
  * G6 three-target recovery
  * G7 hard-negative FP
  * G8 PU hard-negative FP
  * B4 pure-noise FP
  * pT regression metrics
  * efficiency vs pT
  * efficiency vs |d0|

* Same hard-negative training strategy:

  * use the B5 result as reference
  * `hn025` was the best Pareto move for TPS
  * test OMTF-internal with the same hard-negative loss weight

### Does not need to match exactly

The feature dimensions do not need to be identical to TPS/KMTF because the input views are physically different.

But the **training task** must be identical:

```text
Given one processor-window graph, predict the overlap-region muon candidates and reject non-overlap / noise windows.
```

---

## 5. OMTF-internal input variables

The base OMTF-internal variables are:

| Branch               | Meaning                                               |
| -------------------- | ----------------------------------------------------- |
| `reg_stub_phiHw`     | OMTF local phi, 5400-bin scale, processor-local frame |
| `reg_stub_phiBHw`    | bending-angle proxy, mostly DT-relevant               |
| `reg_stub_etaHw`     | OMTF eta hardware code                                |
| `reg_stub_r`         | radial position in cm                                 |
| `reg_stub_quality`   | trigger primitive quality                             |
| `reg_stub_type`      | detector/stub type                                    |
| `reg_stub_layer`     | OMTF logic layer                                      |
| `reg_stub_bx`        | BX offset                                             |
| `reg_stub_trackId`   | native truth ID, 0=noise, >0=gen muon                 |
| `reg_stub_ambiguous` | truth ambiguity flag                                  |
| `reg_iProcessor`     | OMTF processor index 0–2                              |

These are already closer to firmware than KMTF/TPS NanoAOD features.

---

## 6. Proposed OMTF-internal feature schema

The old OMTF cache used only 7 raw features:

```text
phi, phiB, eta, r, quality, type, layer
```

For the new comparison, we should use a schema closer in spirit to the GMT v2 cache.

Recommended features:

| Index | Feature               | Definition                                   |
| ----- | --------------------- | -------------------------------------------- |
| 0     | `phi_norm`            | normalized `reg_stub_phiHw`                  |
| 1     | `phiB_norm`           | normalized `reg_stub_phiBHw`                 |
| 2     | `eta_hw_norm`         | normalized `reg_stub_etaHw`                  |
| 3     | `r_norm`              | normalized `reg_stub_r`                      |
| 4     | `quality_norm`        | `reg_stub_quality / 15`                      |
| 5     | `type_norm`           | normalized `reg_stub_type`                   |
| 6     | `layer_norm`          | `reg_stub_layer / 17`                        |
| 7     | `bx_norm`             | normalized `reg_stub_bx`                     |
| 8     | `stub_in_overlap`     | 1 if `0.83 <= abs(eta_phys) <= 1.24`, else 0 |
| 9     | `abs_eta`             | `abs(reg_stub_etaHw * 0.010875)`             |
| 10    | `eta_dist_to_overlap` | signed distance to overlap eta band          |

Where:

```python
eta_phys = reg_stub_etaHw * 0.010875
abs_eta = abs(eta_phys)
```

and:

```python
if abs_eta < 0.83:
    eta_dist_to_overlap = abs_eta - 0.83      # negative: barrel side
elif abs_eta <= 1.24:
    eta_dist_to_overlap = 0.0                 # inside overlap
else:
    eta_dist_to_overlap = abs_eta - 1.24      # positive: too far endcap side
```

This keeps the key domain features that solved the old KMTF ambiguity:

* `stub_in_overlap`
* `abs_eta`
* `eta_dist_to_overlap`

---

## 7. Truth labels

For OMTF-internal, do not use truth transfer.

Use native OMTF labels:

```python
track_id = reg_stub_trackId
ambiguous = reg_stub_ambiguous
node_label = track_id != 0
```

This is cleaner than KMTF/TPS because the labels are already attached to the OMTF stubs.

Recommended `truth_source` encoding for compatibility:

| Value | Meaning                                              |
| ----- | ---------------------------------------------------- |
| 2     | signal OMTF stub, `track_id > 0`                     |
| 1     | confirmed OMTF noise, `track_id == 0` and valid stub |
| 0     | padding                                              |

So:

```python
truth_source = 2 if valid and track_id > 0
truth_source = 1 if valid and track_id == 0
truth_source = 0 if padding
```

---

## 8. Candidate target convention

This is the most important part to get right.

The old OMTF cache used:

```text
gen_pt[k] = NanoAOD GenMuon k
```

But for the G datasets, this can be misleading because an event can contain multiple generated muons while a specific processor window may only contain stubs from one of them.

For fair window-level training, candidate targets should follow the **actual OMTF window content**.

Recommended rule:

```text
Window target multiplicity = number of unique positive track_id values in this OMTFAllInputTree entry.
```

Example:

```python
unique_ids = sorted(set(track_id[valid_mask] values where track_id > 0))
```

Then fill candidate targets from these IDs:

```python
for slot, tid in enumerate(unique_ids[:K_MAX]):
    muon_index = tid - 1
    gen_pt[slot] = GenMuon_pt[muon_index]
    gen_charge[slot] = GenMuon_charge[muon_index]
    gen_dxy[slot] = GenMuon_dXY[muon_index]
    gen_phi[slot] = GenMuon_phi[muon_index]
```

This makes the OMTF-internal cache comparable to the actual per-window learning problem.

Store both counters:

| Field                   | Meaning                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `meta_n_gen_event`      | number of gen muons in the full event                        |
| `meta_n_targets_window` | number of unique positive track IDs in this processor window |

For training and evaluation, use `meta_n_targets_window` / filled `gen_pt` slots, not just the full-event `nGenMuon`.

---

## 9. Hard negatives

For G7/G8:

```python
meta_is_hard_neg = 1
```

For all other datasets:

```python
meta_is_hard_neg = 0
```

This allows us to reuse the hard-negative loss used in B5.

The aim is:

```text
If a real barrel / non-overlap muon appears, the model should output zero overlap candidates.
```

This is especially important because G7/G8 were one of the major improvements in the new G-dataset campaign.

---

## 10. Proposed cache output

Use a separate cache directory:

```bash
build/omtf_internal/cache_g_v1/
```

Do not overwrite the old OMTF cache.

Each shard should contain:

```text
stubs                 (N, Nmax, F)
valid_mask            (N, Nmax)
track_id              (N, Nmax)
ambiguous             (N, Nmax)
node_label            (N, Nmax)
truth_source          (N, Nmax)
gen_pt                (N, K)
gen_charge            (N, K)
gen_dxy               (N, K)
gen_phi               (N, K)
meta_event_num        (N,)
meta_i_proc           (N,)
meta_n_stubs          (N,)
meta_n_gen_event      (N,)
meta_n_targets_window (N,)
meta_is_hard_neg      (N,)
```

Recommended constants:

```python
NMAX = 24
K_MAX = 3
SCHEMA_VERSION = 1
```

---

## 11. Validation / audit before training

Before training, run a small audit similar to the GMT feature audit.

Checks to perform:

* Feature dimension is correct.
* No NaNs or infinities.
* `stub_in_overlap` is binary.
* `abs_eta` and `eta_dist_to_overlap` are sane.
* `track_id` values are non-negative.
* `truth_source` only contains `{0, 1, 2}`.
* `meta_is_hard_neg = 1` only for G7/G8.
* `meta_n_targets_window <= 3`.
* G1/G2/G3/G4 mostly have 1 target in signal windows.
* G5 has some 2-target windows.
* G6 has some 3-target windows.
* G7/G8 mostly have 0 overlap targets.
* B4 has 0 targets.
* Pos/neg pairs are balanced.

Important: the audit should distinguish:

```text
full-event gen multiplicity
```

from:

```text
processor-window target multiplicity
```

This avoids confusion for G5/G6.

---

## 12. Training plan

### First run: OMTF-internal h64 baseline

Use the same dataset mix as the TPS B5 baseline:

```text
G1_pos G1_neg
G2_pos G2_neg
G3_pos G3_neg
G4_pos G4_neg
G5_pos G5_neg
G6_pos G6_neg
G7
G8
B4
```

Start with:

```text
EdgeCompat h64
hard-negative loss = hn025
threshold = 0.0
```

This is the most important comparison because TPS-h64 is currently the best stable TPS model.

### Optional second run: OMTF-internal h128

Run only if h64 is promising.

Purpose:

* check whether larger capacity improves G5/G6 recovery,
* check whether it hurts hard-negative rejection,
* compare with KMTF-h128 and TPS-h128.

---

## 13. Evaluation plan

Evaluate with the same script structure as KMTF/TPS.

Main metrics:

| Metric           | Why it matters                                 |   |                         |
| ---------------- | ---------------------------------------------- | - | ----------------------- |
| G1 efficiency    | clean prompt overlap baseline                  |   |                         |
| G2 efficiency    | prompt overlap + PU200                         |   |                         |
| G3 efficiency    | clean displaced overlap                        |   |                         |
| G4 efficiency    | displaced overlap + PU200                      |   |                         |
| G5 recovery      | two-displaced-muon recovery                    |   |                         |
| G6 recovery      | three-prompt-muon recovery                     |   |                         |
| G7 FP            | clean hard negative: real muon outside overlap |   |                         |
| G8 FP            | PU hard negative                               |   |                         |
| B4 FP            | pure noise rejection                           |   |                         |
| pT sigma68 / MAE | regression quality                             |   |                         |
| efficiency vs pT | low-pT behavior                                |   |                         |
| efficiency vs    | d0                                             |   | displaced-muon behavior |

Final comparison table:

| Metric           | KMTF-h128 | TPS-h64 | TPS-h128 | OMTF-internal h64 | OMTF-internal h128 |
| ---------------- | --------: | ------: | -------: | ----------------: | -----------------: |
| G1 eff           |           |         |          |                   |                    |
| G2 eff           |           |         |          |                   |                    |
| G3 eff           |           |         |          |                   |                    |
| G4 eff           |           |         |          |                   |                    |
| G5 eff           |           |         |          |                   |                    |
| G6 eff           |           |         |          |                   |                    |
| G7 FP            |           |         |          |                   |                    |
| G8 FP            |           |         |          |                   |                    |
| B4 FP            |           |         |          |                   |                    |
| pT sigma68 G1    |           |         |          |                   |                    |
| pT sigma68 G3/G4 |           |         |          |                   |                    |

---

## 14. Expected outcomes

Possible outcomes:

### Case A — OMTF-internal matches TPS

If OMTF-internal gives similar efficiency and lower/equal FP, it becomes very attractive.

Reason:

* It uses native firmware-like variables.
* It avoids truth-transfer uncertainty.
* It may be easier to integrate into the OMTF trigger pipeline.

### Case B — TPS remains clearly better

If TPS keeps higher displaced efficiency and lower G8 FP, TPS remains the best input view.

Reason:

* TPS provides cleaner endcap/overlap-oriented information.
* It may naturally suppress barrel-only hard negatives.

### Case C — OMTF-internal is better for pT but worse for efficiency

Then we may consider hybrid training ideas later, but not yet.

For now, the comparison should stay simple:

```text
same model, same datasets, different input view
```

---

## 15. Immediate next steps

**Status: complete (2026-05-07).** All steps executed.

* ~~Create `scripts/omtf/build_cache_g_internal.py` from the old OMTF cache builder.~~ Done.
* ~~Add support for G1–G6 pos/neg, G7, G8, B4.~~ Done.
* ~~Remove phi-window regioning, use each entry directly as one window.~~ Done.
* ~~Build OMTF-internal feature schema (11 features).~~ Done — see `src/omtf/features_g.py`.
* ~~Use native `reg_stub_trackId` labels.~~ Done.
* ~~Fill candidate targets from unique positive track_ids.~~ Done.
* ~~Add `meta_is_hard_neg` for G7/G8.~~ Done.
* ~~Build cache to `build/omtf/cache_internal_g_v1/`.~~ Done — 15 datasets, ~2.3M samples.
* ~~Train EdgeCompat h64 with hn025.~~ Done — HTCondor job 1075280.
* ~~Evaluate against TPS-h64-hn025.~~ Done — see section 17.

---

## 16. Bottom-line decision rule

Do not move OMTF-internal to QAT unless it is competitive with TPS.

Minimum useful target:

| Metric                        | Target                      | Result      |
| ----------------------------- | --------------------------- | ----------- |
| G1/G3 clean signal efficiency | close to TPS-h64            | **worse** (−4 to −5 pp) |
| G2/G4 PU signal efficiency    | not much worse than TPS-h64 | **better** (+5 pp) |
| G7 FP                         | ≤ TPS-h64 or close          | **worse** (6.9% vs 3.6%) |
| G8 FP                         | close to TPS-h64/TPS-h128   | **much worse** (5.8% vs 2.2%) |
| B4 FP                         | 0%                          | **pass** (0.0%) |
| G5/G6 recovery                | similar to TPS/KMTF         | **better** (+3–4 pp) |

If OMTF-internal is clearly worse than TPS, keep TPS as the main path before QAT.

If OMTF-internal is comparable, it becomes a strong candidate because it is much closer to the actual trigger firmware representation.

**Decision: TPS remains the primary path. See section 17.**

---

## 17. Phase C results (2026-05-07)

### Implementation

| Component | Location |
| --- | --- |
| Feature schema (11 features) | `src/omtf/features_g.py` |
| Model (EdgeCompatG, parametric n_features) | `src/omtf/models/edge_compat_g.py` |
| Training script | `src/omtf/train_g.py` |
| Cache builder | `scripts/omtf/build_cache_g_internal.py` |
| Eval wrapper | `scripts/omtf/eval_internal.py` |

### Cache

Built to `build/omtf/cache_internal_g_v1/`, schema_version=2, 11 features.
~2.3M total samples across all 15 datasets. Sample counts are ~70% larger than TPS
because no phi-window filtering is applied — all OMTFAllInputTree entries are used.

G8 in particular: 352k samples (internal) vs 47k (TPS) — 7.5× more because TPS
rejects many PU windows where no TPS stub falls inside the phi window.

### Training

| Parameter | Value |
| --- | --- |
| Model | EdgeCompatG h64, n_features=11 |
| Hard-negative loss | w_hard_neg=0.25 (same as TPS B5) |
| Best epoch | 68 |
| Best val_loss | 0.4904 |
| HTCondor job | 1075280.0 |

Training note: NaN explosion at epoch 89 due to cosine LR tail and missing gradient
clipping. Best checkpoint at epoch 68 is valid. `train_g.py` now includes
`clip_grad_norm_(max_norm=1.0)`; a cleaner rerun would likely push best epoch later.

### Evaluation — threshold 0.0

| Model | G1% | G2% | G3% | G4% | G5% | G6% | G7 FP% | G8 FP% | B4 FP% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TPS-h64 baseline | 94.1 | 84.9 | 91.7 | 83.2 | 82.2 | 94.8 | 6.2 | 2.9 | 0.0 |
| **TPS-h64-hn025** | 93.3 | 82.9 | 90.3 | 80.5 | 80.0 | 94.9 | **3.6** | **2.2** | 0.0 |
| OMTF-internal-h64-hn025 | 88.5 | **88.1** | 86.3 | **85.2** | **83.4** | **96.3** | 6.9 | 5.8 | 0.0 |

### Threshold scan

| Model | Thr | G2% | G4% | G5% | G6% | G7 FP% | G8 FP% | B4 FP% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TPS-h64-hn025 | −0.50 | 87.7 | 85.5 | 85.0 | 97.2 | 6.3 | 3.1 | 0.0 |
| TPS-h64-hn025 | +0.00 | 82.9 | 80.5 | 80.0 | 94.9 | 3.6 | 2.2 | 0.0 |
| TPS-h64-hn025 | +0.25 | 80.1 | 77.8 | 77.5 | 93.6 | 2.9 | 1.9 | 0.0 |
| TPS-h64-hn025 | +0.50 | 74.6 | 72.7 | 73.2 | 91.2 | 2.0 | 1.5 | 0.0 |
| OMTF-internal | −0.50 | 90.0 | 87.7 | 86.4 | 97.4 | 10.6 | 7.5 | 0.2 |
| OMTF-internal | +0.00 | 88.1 | 85.2 | 83.4 | 96.3 | 6.9 | 5.8 | 0.0 |
| OMTF-internal | +0.25 | 87.1 | 84.2 | 82.1 | 95.8 | 5.4 | 5.2 | 0.0 |
| OMTF-internal | +0.50 | 85.4 | 82.1 | 79.9 | 95.0 | 3.6 | 4.4 | 0.0 |
| OMTF-internal | +0.75 | 83.9 | 80.5 | 78.3 | 94.3 | 2.8 | 3.9 | 0.0 |
| OMTF-internal | +1.00 | 81.4 | 77.8 | 75.4 | 93.1 | 1.8 | 3.3 | 0.0 |

### Key findings

**Signal efficiency:** OMTF-internal is better on G2/G4/G5/G6 (PU200 and multi-muon
recovery) by +3–5 pp at threshold 0.0. This is likely because native `reg_stub_trackId`
labels avoid truth-transfer uncertainty and the phi/r/layer features give finer spatial
resolution than TPS's depthRegion/tfLayer.

**Clean-muon efficiency:** OMTF-internal is worse on G1/G3 by 4–5 pp. Single-muon
prompt/displaced efficiency is lower, possibly because the uncentred phiHw range and
absence of `eta2` reduce the model's ability to disambiguate low-multiplicity windows.

**Hard-negative rejection (G8/PU200): OMTF-internal cannot match TPS at any threshold.**
At the best G8 operating point (+1.0), OMTF-internal achieves G8=3.3% vs TPS 0.9%.
The G8 gap persists through the full threshold scan. This is the decisive metric.

Equal-FP analysis at G7=3.6%:
- TPS @ +0.00: G2=82.9%, G4=80.5%, G8=2.2%
- OMTF-internal @ +0.50: G2=85.4%, G4=82.1%, G8=4.4%

At equal G7 FP, OMTF-internal recovers more signal but at 2× the G8 FP.

### Phase C decision

**Case B confirmed:** TPS remains clearly better for PU200 hard-negative rejection (G8).

**TPS-h64-hn025 at threshold 0.0 remains the selected floating-point baseline for QAT.**

OMTF-internal is not carried forward. Reason: G8 FP 5.8% vs 2.2% at threshold 0.0,
and no threshold exists where OMTF-internal matches TPS G8 FP while maintaining
competitive signal efficiency.

The signal efficiency advantage of OMTF-internal (G2/G4/G5/G6) is interesting and
suggests the native representation contains complementary information. A potential
future direction would be a hybrid model, but this is outside the current QAT scope.

### Output documents

| Document | Path |
| --- | --- |
| Eval report | `build/omtf/eval/internal_h64_hn025_best_eval.md` |
| Eval JSON | `build/omtf/eval/internal_h64_hn025_best_eval.json` |
| Checkpoint | `build/omtf/checkpoints/internal_h64_hn025/omtf_internal_best.pt` |
