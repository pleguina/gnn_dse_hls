# OMTF Migration Status

**Branch**: `omtf-migration`  
**Plan reference**: `docs/omtf/MIGRATION_PLAN.md`

---

## Stage Tracker

| Stage | Description | Status |
|---|---|---|
| 0 | Additive repository preparation | **Complete** |
| 0.5 | Import graph audit | **Complete** |
| 1 | Dataset audit + DATASET_SIGNOFF | **Complete** |
| 1.5 | Physical variable validation | **Complete** |
| 2 | Minimal OMTF Python pipeline | **Complete** |
| V1 | Vertical Slice V1 milestone | **Complete** (item 8 deferred to Stage 7) |
| 3 | First baselines (A + B) — code | **Complete** |
| 3 | First baselines (A + B) — full training runs | **Complete** (2026-04-19) |
| 4 | Edge-compatibility network | **Complete** (2026-04-19) |
| 5 | Fixed-K slot model | **Complete** (2026-04-19) |
| ADG | Architecture Decision Gate | **Complete** (2026-04-21) — see results below |
| ADG-R | Full-scale retrain (2M+ samples, 4-checkpoint strategy) | **Complete** (2026-04-22–23) |
| ADG-E | Full-scale eval — pT, dXY, all 9 datasets | **Complete** (2026-04-23) — see results below |
| ADG-V | Checkpoint variant eval (best_trig vs best_combined) | **Complete** (2026-04-24) |
| ADG-D | Architecture decision | Pending |
| 6 | Full HECIN reference (optional) | Pending |
| 7 | Quantization and HLS path | Pending |
| 8 | HECIN-Lite (conditional) | Pending |

---

## Stage 0.5 — Import Graph Audit Findings

Completed before any structural changes.

### What calls what

| Entrypoint | Imports from src/ |
|---|---|
| `explore_design_space.py` | `train.train_reduced_model` (dynamic), `parse_hls_report.parse_csynth_xml` (via scripts/ path) |
| `train.py` | `model_base`, `config`, `visualization` |
| `quantization_ptq.py` | `model_base`, `config` |
| `quantization_qat.py` | `config`, `model_qat`, `subgraph_extraction` |
| `brevitas_quantization.py` | `brevitas_models`, `config`, `model_base` |
| `analyze_models.py` | `config`, `model_base`, `model_qat`, `model_qat_v2`, `quantization_ptq`, `train`, `train_qat`, `visualization` |
| `evaluate_qformat.py` | `config`, `model_base`, `train.load_cora_dataset` |

### Safety conclusions

- `explore_design_space.py` uses `sys.path.insert(0, str(PROJECT_ROOT / "src"))` — flat layout assumed, but adding `src/omtf/` as a subpackage does **not** conflict because Python only imports `omtf.*` on explicit request.
- `config.py` `get()` returns `None` for missing keys — safe to add `omtf:` section to YAML.
- No existing import references `omtf` or `audit` — zero collision risk.
- All existing Cora/GraphSAGE code stays in place at `src/` root — no moves yet.

---

## Stage 1 — Dataset Audit

**Signed off**: 2026-04-16. See `docs/omtf/DATASET_SIGNOFF.md`.

Key decisions locked in:
- `Nmax = 24` (covers p99 across all datasets, clean hardware boundary)
- Padding: zero-pad + `valid_mask`
- Truncation: by quality (descending) when stubs > 24
- `delta_bx` excluded from model inputs (identically 0 in current dataset)
- `kappa_hat` requires clipping or tanh normalization before model input

---

## Stage 1.5 — Physical Variable Validation

**Signed off**: 2026-04-16. See `docs/omtf/VARIABLE_SIGNOFF.md` and `docs/omtf/FEATURE_READINESS.md`.

All 9 datasets validated for physical variable correctness. GenMuon displacement
available as `dXY` (not `d0`). All raw and derived features verified against
expected physical ranges.

---

## Stage 2 — Minimal Pipeline

**Complete**: `src/omtf/dataset.py`, `features.py`, `splits.py`, `losses.py`.

Validated during Stage 1 audit runs: joins 100%, no NaN in derived features,
Nmax=24 padding + valid_mask working.

---

## Vertical Slice V1 — Status

| Criterion | Status |
|---|---|
| 1. Load one ROOT file from S1 without errors | **Pass** (validated in audit) |
| 2. Join to NanoAOD ≥ 95% entries | **Pass** — 100% across all datasets |
| 3. Produce padded sample (Nmax=24, valid_mask, track_ids) | **Pass** |
| 4. Derive pair features with no NaN | **Pass** (kappa_hat tails clipped) |
| 5. Forward pass through Baseline A without errors | **Pass** |
| 6. Edge-label AUC on S1 val well above 0.5 | **Pass** — edge_auc=0.66 (smoke-test run) |
| 7. Forward pass on B4 without crashing | **Pass** |
| 8. Export batch of test vectors (flat text) | Not yet implemented |

**V1 is functionally complete.** Item 8 requires `tests/generate_test_vectors_omtf.py` (Stage 7 deliverable).

AUC in criterion 6 was from a 20-epoch smoke-test (3 files, 200 entries/file).
Full training run (2026-04-19) confirmed node_auc=0.934 — V1 criterion 6 fully satisfied.

---

## Stage 3 — Baseline Runs

### Code status: Complete
- `src/omtf/models/baselines.py` — DeepSets (A) and EdgeMLP (B)
- `src/omtf/train.py`, `eval.py`, `metrics.py`

### Smoke-test run (2026-04-17): 3 files × 200 entries/file, 20 epochs

| Model | node_auc | edge_auc | stub_eff |
|---|---|---|---|
| Baseline A (DeepSets) | 0.539 | — | 0.922 |
| Baseline B (EdgeMLP) | — | 0.664 | — |

Note: near-random results expected at this scale; pipeline integrity confirmed.

---

### Full run (2026-04-19): 50 files/dataset, 50 epochs

**Infrastructure fix applied**: `OMTFDataset` now pre-loads all entries at
construction time (eager loading). Previously opened the ROOT file per-sample
causing ~330× slowdown on CPU. Now ~0.1s/epoch (deepsets) and ~25s/epoch
(edge_mlp with graph).

**Baseline A — DeepSets (S1 + B1 + B4, hidden=64)**

| Metric | Value |
|---|---|
| best node_auc (val) | **0.934** |
| best stub_eff (val) | **0.983** |
| best val_loss | 0.233 |
| train samples | 103,626 |
| val samples | 51,711 |

Checkpoint: `build/omtf/checkpoints/deepsets_best.pt`

**Baseline B — EdgeMLP (S1 only, hidden=64)**

| Metric | Value |
|---|---|
| best edge_auc (val) | **0.838** |
| best val_loss | 0.287 |
| train samples | ~37,000 (S1 only) |

Checkpoint: `build/omtf/checkpoints/edge_mlp_best.pt`

### Stage 3 go/no-go decision

Per the plan's decision table:

| Criterion | Result | Implication |
|---|---|---|
| Baseline A node_auc | 0.934 — well above random | Pipeline healthy; data labels trustworthy |
| Baseline B edge_auc | 0.838 — between 0.7 and 0.85 | Pair features discriminating; Stage 4 well-motivated |

**Decision: PROCEED to Stage 4.**

Baseline A node_auc=0.934 confirms the data pipeline and labels are sound. Baseline B
edge_auc=0.838 is below the 0.85 plan target but well above 0.7 and still trending up at
epoch 50 on 50 files. Pair features (kappa_hat, phiB_diff, Δphi, Δr) carry discriminating
power. The edge AUC gap is expected at this data scale (50 files S1 only); Stage 4 with
mixed datasets will provide a fuller picture. No data quality issue warranting a return to
Stage 1.

---

## Stage 4 — Edge-compatibility network

### Code: Complete
- `src/omtf/models/edge_compat.py` — EdgeCompatNet (43k parameters)
- Wired into `src/omtf/train.py`: `--model edge_compat` flag

### Architecture
- Per-stub encoder MLP(7→H)
- Edge scorer MLP(2H+6→H) → edge embedding + scalar logit
- Per-node mean-pool of incident edge embeddings → node update
- Updated repr concat(stub_emb, node_update) → per-node and global heads
- K=3 candidate heads: candidate score, pT, charge

### Full run (2026-04-19): S1+S3+B1+B4, 50 files/dataset, 50 epochs

| Metric | Value |
|---|---|
| best node_auc (val) | **0.938** |
| best edge_auc (val) | **0.902** |
| best stub_eff (val) | **0.983** |
| best val_loss | 0.536 |

Checkpoint: `build/omtf/checkpoints/edge_compat_best.pt`

### Comparison vs baselines

| Model | Training data | node_auc | edge_auc | stub_eff |
|---|---|---|---|---|
| Baseline A (DeepSets) | S1+B1+B4, 50 files | 0.934 | — | 0.983 |
| Baseline B (EdgeMLP) | S1, 50 files | — | 0.838 | — |
| **Stage 4 (EdgeCompat)** | **S1+S3+B1+B4, 50 files** | **0.938** | **0.902** | **0.983** |

**Key gains from Stage 4:**
- edge_auc: 0.838 → **0.902** — node aggregation after edge scoring substantially improves edge discrimination
- node_auc: 0.934 → **0.938** — marginal improvement (DeepSets already strong at node level)
- First model trained on S3 (multi-track): demonstrates model handles multi-track windows without collapsing

Both metrics still trending upward at epoch 50. More data or epochs likely push edge_auc further.

---

## Stage 5 — Fixed-K Slot Model

### Code: Complete
- `src/omtf/models/slot_model.py` — SlotModel (24,133 parameters)
- Wired into `src/omtf/train.py`: `--model slot_model` flag

### Architecture
- Per-stub encoder MLP(7→H)
- K=3 learnable slot queries (nn.Parameter)
- Stub-to-slot compatibility MLP(2H→1) → softmax attention α (K, Nmax)
- Slot aggregation: α-weighted sum of stub embeddings → slot_emb (K, H)
- Node update: α-weighted sum of slot embeddings back to stubs (Nmax, H)
- Combined repr concat(stub_emb, node_upd) → per-node and per-slot heads
- No edge features — pure attention-based candidate hypothesis model

### Full run (2026-04-19): S1+S3+B1+B4, 50 files/dataset, 50 epochs

| Metric | Value |
|---|---|
| best node_auc (val) | **0.907** |
| best stub_eff (val) | **0.976** |
| best val_loss | 0.3848 (epoch 47) |

Checkpoint: `build/omtf/checkpoints/slot_model_best.pt`

### Comparison vs Stage 4

| Model | Training data | node_auc | edge_auc | stub_eff | params |
|---|---|---|---|---|---|
| **Stage 4 (EdgeCompat)** | S1+S3+B1+B4, 50 files | **0.938** | **0.902** | **0.983** | 43k |
| **Stage 5 (SlotModel)** | S1+S3+B1+B4, 50 files | 0.907 | — | 0.976 | 24k |

**Key observations:**
- SlotModel node_auc (0.907) is 3 points below EdgeCompatNet (0.938) despite similar training setup
- SlotModel stub_eff (0.976) is slightly below EdgeCompatNet (0.983)
- SlotModel is 44% smaller (24k vs 43k params) — meaningful for firmware
- SlotModel does not use pair features — a pure attention-based ablation
- Both models still trending at epoch 50; gap may narrow with more data/epochs

---

## Architecture Decision Gate — Status: Blocked (retrain required)

**Eval run**: 2026-04-19, S1+S3+B1+B4, 50 files/dataset  
**Eval code**: `src/omtf/eval.py --compare`  
**Results**: `build/omtf/eval/edge_compat_20260419_125422.json`, `slot_model_20260419_125422.json`

### Full trigger-level comparison

| Metric | EdgeCompatNet | SlotModel | Notes |
|---|---|---|---|
| node_auc (S1) | **0.867** | 0.770 | EdgeCompat leads |
| stub_eff (S1) | **0.987** | 0.984 | EdgeCompat leads |
| cand_recovery (S1) | 0.690 | **0.957** | Slot leads (fixed-index metric) |
| trig_eff@5 (S1) | **0.998** | 0.961 | EdgeCompat leads |
| trig_eff@10 (S1) | **0.998** | 0.963 | EdgeCompat leads |
| trig_eff@10 (B1/PU200) | **0.985** | 0.565 | EdgeCompat strongly leads |
| dimuon_sep (S3) | 0.364 | **0.894** | Slot strongly leads |
| bg_accept (B4) | 0.975 | **0.025** | **Slot strongly leads — critical** |
| edge_auc (S3) | **0.923** | — | EdgeCompat only |
| params | 43k | **24k** | Slot 44% smaller |

### Root cause analysis

**EdgeCompatNet bg_accept = 97.5% (B4)** — critical failure.  
Root cause: the candidate heads (`MLP(2H → K)` from global context) were never supervised
with explicit negative examples during training. The candidate loss was absent from
`_compute_loss_edge_compat` — only node BCE and edge BCE were used. The K output scores
are therefore uninformative: they fire ~always regardless of window content. This is a
training gap, not an architectural defect. Fix: add NanoAOD-supervised candidate loss
to EdgeCompatNet training (gen_pt now available from Gap 2 implementation).

**SlotModel trig_eff@10 (B1) = 56.5%** — weak under PU200.  
Root cause: the fixed-index slot assignment (slot k ↔ gen muon track_id=k+1) is fragile
under PU200 pileup. In B1, the signal muon has track_id=1, but the slot model may attend
to noise stubs at slot 0 when PU stubs dominate the window. Additionally, SlotModel was
trained with slot_signal_mass proxy supervision (attention-weighted labels) — under PU200,
the attentionweights spread across noise stubs and the proxy target for slot 0 is diluted.

**pT resolution is meaningless for both**: both checkpoints predate the NanoAOD join
(Gap 2 implemented 2026-04-19). The pt_pred heads have never seen real pT targets.
pt_mae ~44 GeV, pt_sigma68 is artificially low (constant near-zero prediction).

**cand_recovery discrepancy (Edge=0.69 vs Slot=0.96)**: the fixed-index metric favors
SlotModel by construction — slot k is supervised by gen muon k, so candidate_logit[k]
is correlated with gen muon k presence. For EdgeCompatNet, the K global-context heads
are not slot-ordered, so the fixed-index metric is unfair. The `trig_eff` scan (window-level,
model-agnostic) is the honest comparison: EdgeCompat leads there.

### What must change before the ADG decision can be made

1. **Add candidate head supervision to EdgeCompatNet** — wire gen_pt into
   `_compute_loss_edge_compat` using the same fixed-index BCE as SlotModel.
   Retrain for 50 epochs. Expected to collapse bg_accept substantially.

2. **Retrain SlotModel with pT/charge supervision** — the NanoAOD join is now
   wired in. Retraining will calibrate pt_pred heads for the efficiency-vs-pT scan.
   Also expected to improve B1 performance since pT supervision reinforces correct
   slot assignment.

3. **Re-run ADG eval** on both retrained checkpoints across all 9 datasets.

4. **Decide**: proceed with the better model to Stage 7 (quantization + HLS).
   If both are viable, may carry both forward to Stage 7 as parallel paths.

---

## ADG Retrain Runs (2026-04-20)

Both models retrained with ADG fixes applied, trained on all 8 datasets (S1+S2+S3+S5+B1+B2+B3+B4), 50 files/dataset, 50 epochs. Condor cluster jobs.

### Job history

| Job ID | Process | Model | Status | Notes |
|---|---|---|---|---|
| 1061689 | 0+1 | both | Failed | `--epochs` arg parsing error — datasets list consumed the value |
| 1061690 | 0+1 | both | Failed | Same arg parsing error |
| 1061691 | 0+1 | both | Failed | PyTorch shared memory "too many open files" (num_workers) |
| 1061705 | 0 | edge_compat | Evicted | Killed at epoch 7 |
| 1061705 | 1 | slot_model | Complete | Superseded by 1061717.1 |
| **1061717** | **0** | **edge_compat** | **Complete** | Final retrain ✓ |
| **1061717** | **1** | **slot_model** | **Complete** | Final retrain ✓ |

### EdgeCompatNet retrain (all 8 datasets, 50 epochs, + candidate head supervision)

| Metric | Value |
|---|---|
| best node_auc (val) | **0.9232** |
| best edge_auc (val) | **0.8929** |
| best stub_eff (val) | **0.9795** |
| best val_loss | 2.0350 (epoch 47) |
| train samples | 248,552 |
| val samples | 124,763 |
| parameters | 47,502 |

Checkpoint: `build/omtf/checkpoints/edge_compat_best.pt` (updated 2026-04-20 02:11)

### SlotModel retrain (all 8 datasets, 50 epochs, + pT/charge supervision)

| Metric | Value |
|---|---|
| best node_auc (val) | **0.8925** |
| best stub_eff (val) | **0.9814** |
| best val_loss | 1.8930 (epoch 43) |
| train samples | 248,552 |
| val samples | 124,763 |
| parameters | 26,310 |

Checkpoint: `build/omtf/checkpoints/slot_model_best.pt` (updated 2026-04-20 00:35)

---

## Architecture Decision Gate — Final Eval (2026-04-21)

**Eval run**: 2026-04-21  
**Checkpoints**: edge_compat epoch 47, slot_model epoch 43  
**All 9 datasets**, cache-backed (`.pt` shards, `schema_v1_graph`)  
**Results**: `build/omtf/eval/edge_compat_20260421_222454.json`, `slot_model_20260421_222454.json`

### Signal efficiency (trig_eff@10)

| Dataset | edge_compat | slot_model | Notes |
|---|---|---|---|
| S1 (single µ, prompt) | **0.9989** | 0.9647 | +3.4 pp |
| S2 (single µ, displaced) | **0.9986** | 0.9116 | +8.7 pp |
| S3 (multi-µ, prompt) | **0.9991** | 0.9686 | +3.1 pp |
| S4 (multi-µ, displaced low-pt) | **0.9998** | 0.9943 | +0.6 pp |
| S5 (multi-µ, displaced high-pt) | **0.9973** | 0.9101 | +8.7 pp |

### Background / pile-up efficiency (trig_eff@10)

| Dataset | edge_compat | slot_model | Notes |
|---|---|---|---|
| B1 (PU200) | **0.9884** | 0.6456 | +34.3 pp — major improvement |
| B2 (PU200 displaced) | **0.9827** | 0.4775 | +50.5 pp |
| B3 (PU200 multi-µ) | **0.9915** | 0.7429 | +24.9 pp |
| B4 bg_accept (min-bias) | 0.8489 | **0.1565** | edge_compat much higher — see below |

### Full metric table (ADG comparison)

| Metric | edge_compat | slot_model | Notes |
|---|---|---|---|
| node_auc (S1) | **0.8608** | 0.7733 | |
| stub_eff (S1) | 0.9844 | **0.9860** | near-equal |
| cand_recovery (S1) | **0.9948** | 0.9576 | |
| trig_eff@10 (S1) | **0.9989** | 0.9647 | |
| edge_auc (S1) | 0.8406 | — | edge_compat only |
| node_auc (B1) | **0.9447** | 0.9117 | |
| trig_eff@10 (B1/PU200) | **0.9884** | 0.6456 | slot_model still weak on PU200 |
| dimuon_sep (B3) | 0.7306 | **0.6525** | edge_compat leads |
| bg_accept (B4) | 0.8489 | **0.1565** | **edge_compat concern — see below** |
| params | 47,502 | **26,310** | slot_model 45% smaller |

### B4 background acceptance — analysis

edge_compat bg_accept dropped from **0.975 → 0.849** after adding candidate supervision
to the retrain. The metric is comparable between models: both use
`background_acceptance_slots(candidate_logits > 0)` with K=3 slots and the same threshold.

**Root cause of edge_compat's high bg_accept (structural, not a eval bug):**

edge_compat's K=3 candidate heads are all conditioned on the same mean-pooled global
context `global_ctx = mean(updated_stubs)`. This makes the K outputs correlated: if the
global window looks like signal, all three heads tend to fire together. The
`.any(dim=-1)` test in `background_acceptance_slots` then fires if any slot exceeds
threshold, which for a globally-conditioned model is essentially asking "does this window
look like signal at all?" — very sensitive.

slot_model's slot-attention mechanism has K slots that compete for stubs via softmax
attention, making them selective: a slot fires only when its dedicated attention region
aligns with real track stubs.

B4 (minimum-bias) contains real low-pT muon stubs from the underlying event. edge_compat
finds real stub pairs in these windows and fires its candidate heads. slot_model's slot
competition suppresses this.

**Implication for trigger rate**: edge_compat as currently trained would accept ~85% of
min-bias windows. That translates to a very high L1 rate (×5–6 above slot_model's ~16%).
This is the key risk of selecting edge_compat for Stage 7.

### ADG decision (2026-04-21 — superseded)

edge_compat won on every signal and PU200 metric. bg_accept=0.849 was the blocking concern.
Options 1–3 listed above were deferred pending full-scale retrain. See ADG-R/ADG-E below.

---

## ADG-R: Full-scale Retrain (2026-04-22–23)

Both models retrained on all 8 datasets at full data scale with the 4-checkpoint strategy
(best val_loss, best trig_eff_proxy, best bg_accept_proxy, best combined_score).

### Condor job history

| Job ID | Model | Status | Notes |
|---|---|---|---|
| 1065997.1 | slot_model | **Complete** | All 8 datasets, 50 epochs |
| 1069115.0 | edge_compat | **Complete** | All 8 datasets, 50 epochs |

### EdgeCompatNet full-scale retrain (job 1069115, 2026-04-23)

| Metric | Value |
|---|---|
| train samples | 2,073,252 |
| val samples | 414,646 |
| parameters | 47,622 |
| best val_loss | 1.7414 (epoch 49) → `edge_compat_best.pt` |
| best trig_eff_proxy | 0.8387 (epoch 46) → `edge_compat_best_trig.pt` |
| best bg_accept_proxy | 0.0953 (epoch 41) → `edge_compat_best_rate.pt` |
| best combined_score | 0.7225 (epoch 47) → `edge_compat_best_combined.pt` |

### SlotModel full-scale retrain (job 1065997.1, 2026-04-22)

| Metric | Value |
|---|---|
| train samples | 2,073,252 |
| val samples | 414,646 |
| parameters | 26,310 |
| best val_loss | 1.5737 (epoch 44) → `slot_model_best.pt` |
| best trig_eff_proxy | 0.8286 → `slot_model_best_trig.pt` |
| best bg_accept_proxy | 0.1724 → `slot_model_best_rate.pt` |
| best combined_score | 0.6261 → `slot_model_best_combined.pt` |

---

## ADG-E: Full-scale Eval (2026-04-23)

**Eval job**: 1070773  
**Checkpoints**: `edge_compat_best.pt` (epoch 49), `slot_model_best.pt` (epoch 44)  
**All 9 datasets**, cache-backed (`schema_v1_graph`)  
**Results**: `build/omtf/eval/edge_compat_20260423_234428.json`, `slot_model_20260423_234428.json`

### Signal efficiency (trig_eff@10)

| Dataset | edge_compat | slot_model | Delta |
|---|---|---|---|
| S1 (single µ, prompt) | **0.9794** | 0.9738 | +0.6 pp |
| S2 (single µ, displaced) | **0.9411** | 0.9274 | +1.4 pp |
| S3 (multi-µ, prompt) | 0.9824 | **0.9851** | −0.3 pp |
| S4 (multi-µ, displaced low-pt) | 0.9968 | **0.9977** | −0.1 pp |
| S5 (multi-µ, displaced high-pt) | **0.9217** | 0.9126 | +0.9 pp |

### PU200 efficiency + background rate

| Dataset | edge_compat | slot_model | Notes |
|---|---|---|---|
| B1 (PU200) trig_eff@10 | **0.6873** | 0.6826 | Both weak — see analysis |
| B2 (PU200 displaced) trig_eff@10 | **0.5355** | 0.5284 | Both weak |
| B3 (PU200 multi-µ) trig_eff@10 | 0.7742 | **0.7780** | Near-equal |
| B4 bg_accept (min-bias) | **0.1079** | 0.2014 | edge_compat now leads on rate |

### pT resolution

| Dataset | edge_compat pt_mae | slot_model pt_mae | edge_compat pt_σ68 | slot_model pt_σ68 |
|---|---|---|---|---|
| S1 (prompt) | 27.4 GeV | 28.4 GeV | 0.83 | 0.84 |
| S2 (displaced) | 8.4 GeV | 9.0 GeV | 0.68 | 0.82 |
| S3 (multi-µ prompt) | 5.7 GeV | 5.9 GeV | 0.96 | 1.02 |
| S5 (displaced high-pt) | 9.1 GeV | 9.6 GeV | 1.13 | 1.23 |

pT resolution remains poor. pt_mae 27 GeV on prompt muons indicates near-constant predictions;
pt_σ68 ~0.83–1.1 confirms this. The regression head is not dominating training.

### dXY resolution

| Dataset | edge_compat dxy_mae | slot_model dxy_mae | Notes |
|---|---|---|---|
| S1 (prompt) | 0.68 cm | 0.55 cm | Prompt — small true dxy |
| S2 (displaced) | 43.1 cm | 53.1 cm | Completely wrong — see analysis |
| B2 (PU200 displaced) | 53.2 cm | 55.8 cm | Completely wrong |
| S4 (displaced low-pt) | 0.11 cm | 0.33 cm | edge_compat leads |

dxy reconstruction fails completely for high-displacement datasets (S2, B2, S5). The model
predicts near-zero dxy for all windows; displaced muons with large true dxy drive the mae.

### dimuon separation (multi-µ datasets)

| Dataset | edge_compat | slot_model |
|---|---|---|
| S3 | **0.9555** | 0.9589 |
| S4 | **0.9959** | 0.9971 |
| S5 | 0.8123 | **0.8464** |
| B3 | 0.6868 | **0.7027** |

Near-equal on S3/S4. SlotModel leads on displaced multi-µ (S5, B3).

### Analysis

**bg_accept reversal**: EdgeCompatNet bg_accept collapsed from 0.849 (ADG-21) to 0.108 —
now better than SlotModel (0.201). Training on 8× more data with candidate supervision
properly learned to suppress min-bias windows. The ADG-21 structural concern is resolved.

**Apr-21 B1/B2 numbers were inflated**: The Apr-21 edge_compat had B1 trig_eff@10=0.9884
with bg_accept=0.849. A model that accepts 85% of windows has near-trivially high PU200
efficiency. The honest operating point at selective threshold is ~0.69 (B1), visible here.

**PU200 efficiency ~0.69 is the open problem**: Both models land at ~0.68–0.69 on B1.
This is the genuine limit at this training scale / architecture. Reaching >0.95 (typical
L1 trigger target) requires either more PU200 training data, architecture changes, or
threshold-optimised checkpoints (see ADG-V below).

**pT and dXY heads not learning**: Regression losses are not dominating relative to the
classification losses. Both heads need either loss reweighting or separate training phases.
Displaced dxy (S2/B2) fails entirely — the model cannot discriminate large displacement.

### Checkpoint summary (full-scale, edge_compat)

| Checkpoint | epoch | trig_eff_proxy | bg_accept_proxy | combined |
|---|---|---|---|---|
| best (val_loss) | 49 | 0.818 | 0.111 | 0.707 |
| best_trig | 46 | **0.839** | 0.146 | 0.693 |
| best_rate | 41 | 0.814 | **0.095** | 0.718 |
| best_combined | 47 | 0.819 | 0.096 | **0.723** |

ADG-V eval (job 1070777) covers best_trig vs best_combined to quantify the
trig_eff vs bg_accept trade-off at threshold.

---

## ADG-V: Checkpoint Variant Eval (2026-04-24)

**Eval job**: 1070777  
**Checkpoints**: `edge_compat_best_trig.pt` (epoch 46) vs `edge_compat_best_combined.pt` (epoch 47)  
**Results**: `build/omtf/eval/edge_compat_20260424_*.json` (both variants)

### Three-checkpoint comparison (EdgeCompatNet only)

| Dataset | best_trig (ep46) | best / val_loss (ep49) | best_combined (ep47) |
|---|---|---|---|
| S1 trig_eff@10 | **0.9894** | 0.9794 | 0.9807 |
| S2 trig_eff@10 | **0.9679** | 0.9411 | 0.9489 |
| S3 trig_eff@10 | **0.9904** | 0.9824 | 0.9825 |
| S4 trig_eff@10 | **0.9992** | 0.9968 | 0.9968 |
| S5 trig_eff@10 | **0.9506** | 0.9217 | 0.9245 |
| B1 trig_eff@10 | **0.7097** | 0.6873 | 0.6879 |
| B2 trig_eff@10 | **0.5675** | 0.5355 | 0.5384 |
| B3 trig_eff@10 | **0.7929** | 0.7742 | 0.7754 |
| B4 bg_accept | 0.1439 | 0.1079 | **0.0935** |

### Analysis

`best_trig` wins on every efficiency metric (signal and PU200) by 1–3 pp.
Cost: bg_accept rises from 0.094–0.108 to 0.144.

All three checkpoints sit within a narrow band — the model behaviour is stable.
Choosing a checkpoint is an operating-point decision, not an architectural one.

**Recommended deployment checkpoint**: `edge_compat_best_trig.pt` for maximum signal
efficiency; `edge_compat_best_combined.pt` if rate budget is tight.

---

## Data Adequacy Analysis (2026-04-24)

Analysis performed directly on `schema_v1_graph` cache shards to understand whether
the PU200 efficiency (~0.69–0.79 on B1/B2/B3) is limited by data volume or physics.

### Finding 1: Nmax=24 truncation never fires

| Dataset | mean raw stubs | mean used stubs | % truncated |
|---|---|---|---|
| S1 | 4.9 | 4.9 | 0% |
| B1 (PU200) | 3.5 | 3.5 | 0% |
| B4 (min-bias) | 1.5 | 1.5 | 0% |

Average stub counts are well below the Nmax=24 cap across all datasets.
Signal stubs are never truncated. This is not the bottleneck.

### Finding 2: Signal stub fraction per dataset

| Dataset | mean stubs/window | mean signal stubs | signal% | % zero-signal windows |
|---|---|---|---|---|
| S1 | 4.9 | 4.5 | 93.9% | 0.7% |
| S2 | 4.1 | 3.9 | 94.9% | 0.7% |
| S3 | 6.6 | 6.2 | 94.5% | 0.4% |
| S5 | 4.3 | 4.1 | 95.6% | 0.5% |
| B1 | 3.5 | 2.5 | 48.8% | **45.3%** |
| B2 | 2.3 | 1.0 | 22.7% | **74.8%** |
| B3 | 4.9 | 3.9 | 56.0% | **38.3%** |
| B4 | 1.5 | 0.0 | 0.0% | 100% |

### Finding 3: Physics ceiling on PU200 datasets

OMTF uses 12 processors covering different φ sectors. A gen muon in a given window
may produce no reconstructed stubs in that processor (it passed through a different
sector, or was too low-pT). These windows are unrecoverable by any reconstruction algorithm.

| Dataset | % gen_pt>10 windows with zero sig stubs | Physics ceiling | Model trig_eff@10 |
|---|---|---|---|
| B1 | 44.9% | **55.1%** | 68.7–70.9% |
| B2 | 65.9% | **34.1%** | 53.5–56.8% |
| B3 | 31.6% | **68.4%** | 77.4–79.3% |

The model consistently exceeds the visible-stub-only ceiling. The extra percentage comes
from the model firing on some invisible-muon windows at a rate consistent with bg_accept (~10–14%),
and since those windows do contain a gen muon, they count as correct triggers.

### Conclusion

**Adding more training data will not improve B1/B2/B3 trigger efficiency.**
The 45% (B1) / 66% (B2) / 32% (B3) of windows with zero signal stubs provide no
learnable gradient — there is no information in those windows that distinguishes them
from min-bias. The efficiency ceiling is set by OMTF processor coverage, not by the model.

The model is performing at or near the physical limit for single-processor reconstruction.
Improving beyond ~70% on B1 requires either:
- Multi-processor coordination (outside OMTF scope)
- A different detector geometry / processing scheme
- Accepting that single-processor OMTF reconstruction cannot tag all PU200 muons

**pT and dXY heads are the remaining trainable gap.** Both still show poor resolution
(pt_mae ~27 GeV on prompt muons) indicating the regression losses are not dominating
training. Loss reweighting or a separate regression fine-tuning phase is needed for Stage 7.

---

## Legacy path

The existing Cora/GraphSAGE code is the **legacy path**. It remains fully functional and untouched.

Files marked as legacy (do not extend for OMTF):
- `src/train.py`
- `src/model_base.py`
- `src/model_base_QAT.py`
- `src/model_qat.py`
- `src/model_qat_v2.py`
- `src/subgraph_extraction.py`

These will be moved to `src/legacy/` only after V1 is fully signed off.

---

## Key structural facts discovered

- Tree path: `simOmtfPhase2Digis/OMTFAllInputTree` (not at ROOT file root level)
- `reg_eventNum`: `UInt_t` — join key to NanoAOD `event` (`ULong64_t`) via uint32 cast
- `reg_iProcessor`: `UChar_t`
- `reg_stub_trackId`: `vector<signed char>` — must call `int()` to convert from char
- `reg_stub_ambiguous`: `vector<unsigned char>`
- `reg_stub_phiHw`, `reg_stub_phiBHw`, `reg_stub_r`: `vector<short>` (int16)
- `reg_stub_layer`, `reg_stub_etaHw`, `reg_stub_quality`, `reg_stub_type`, `reg_stub_bx`: `vector<signed char>`
- NanoAOD `Events` tree is at root level of the nano file
- 500 events per nano file, ~938 processor-window entries per hits file (B1 sample)
- GenMuon displacement: use `dXY` (not `d0`)
