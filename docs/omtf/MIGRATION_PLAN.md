# OMTF ML-to-Firmware Repository Migration Plan

**Status**: Approved, pending implementation  
**Last updated**: 2026-04-16  
**Primary reference**: `docs/OMTF_MODEL_STUDY.md`

---

## What This Document Is

This is the agreed implementation plan for migrating this repository from a
GraphSAGE/Cora FPGA benchmark toward a full OMTF trigger ML-to-firmware
development repository.

It must be read before any code changes are made.

---

## Key Constraints (non-negotiable)

1. **No file moves until the first OMTF vertical slice is working.** Existing
   Cora/GraphSAGE code stays in place. New OMTF code is purely additive.
2. **Do an import graph audit before any structural change.** Stage 0.5 is not
   optional.
3. **`DATASET_SIGNOFF.md` is a hard gate.** No serious model stage (Stage 4+)
   begins without it being filled and signed off.
4. **HECIN does not block the deployment path.** Stage 6 is optional and runs
   in parallel with the firmware path after the Architecture Decision Gate.
5. **The Architecture Decision Gate after Stage 5 prevents endless parallel
   growth.** One main deployment candidate must be chosen before Stage 7.

---

## Changes from First Draft

1. Stage 0 is now additive-only — no file moves
2. New Stage 0.5 — import graph audit before any structural changes
3. Vertical Slice V1 is a named milestone with explicit success criteria
4. Formal `DATASET_SIGNOFF.md` gate gates all serious model stages
5. Efficiency/rate language split into four distinct metrics with explicit scope
6. Baseline B (pure edge BCE, no global aggregation) added before Stage 4
7. HECIN stage is explicitly optional and non-blocking
8. Fixed-Nmax policy decision added as a formal checkpoint within Stage 1
9. Architecture Decision Gate added after Stage 5
10. Exact first files listed

---

## Stage Dependency Map

```
Stage 0  (additive stubs)
    └─→ Stage 0.5 (import audit)
            └─→ Stage 1 (dataset audit + Nmax policy)
                    │
                    └─→ DATASET_SIGNOFF.md  ← HARD GATE
                              │
                              └─→ Stage 2 (minimal pipeline)
                                        └─→ Vertical Slice V1  ← NAMED MILESTONE
                                                  └─→ Stage 3 (Baseline A + Baseline B)
                                                            │
                                                   [Baseline checks pass?]
                                                            │
                                                            ├─→ Stage 4 (edge-compat)
                                                            └─→ Stage 5 (slot model)
                                                                      │
                                                         Architecture Decision Gate
                                                                      │
                                                          ┌───────────┴──────────────┐
                                                          │                          │
                                                     Stage 7                   Stage 6 (optional)
                                                  (quant + HLS)              (HECIN reference)
                                                          │                          │
                                                     Stage 8                [gap meaningful?]
                                                  (HECIN-Lite,
                                                   only if needed)
```

---

## Stage 0 — Additive repository preparation

**Goal**: Create new OMTF directory structure alongside existing code. No
existing file is moved or modified.

**Deliverables**:

```
src/omtf/__init__.py                    empty stub
src/omtf/models/__init__.py             empty stub
src/audit/__init__.py                   empty stub
docs/omtf/MIGRATION_STATUS.md           migration progress tracker
docs/omtf/DATASET_SIGNOFF.md           stub with required sections
configs/model_config.yaml               extend with omtf: section (stubs only)
```

**Constraint**: Every existing import continues to work unchanged. File moves
happen only after the first OMTF vertical slice is working.

---

## Stage 0.5 — Import graph and code audit

**Goal**: Map what calls what before making any structural change.

**Produce a short written note answering**:

- Which scripts import from which `src/` modules
  (e.g. `explore_design_space.py` → `train.py`, `quantization_ptq.py`,
  `prepare_ptq_int8_parameters.py`, `parse_hls_report.py`)
- Whether `explore_design_space.py` assumes a flat `src/` layout
  (it does — uses `sys.path.insert(0, str(PROJECT_ROOT / "src"))`)
- Whether adding `src/omtf/` as a subpackage conflicts with any existing import
- Which config keys are currently read by which entrypoints
- Whether the new `omtf:` config section can coexist with existing keys without
  breaking `get_config()`

This is not optional. It prevents accidental breakage in Stage 1 onward.

---

## Stage 1 — Dataset audit

**Goal**: Answer definitively whether the produced data is correct, consistent,
and sufficient for the intended OMTF study.

### Scripts to create

**`src/audit/check_branches.py`**
- All `reg_stub_*` arrays equal-length per entry
- No NaN, no sentinel overflow values
- `reg_iProcessor` in expected range [0, 11]
- `reg_eventNum` non-duplicate within file
- Schema consistency across multiple files per dataset bucket

**`src/audit/check_labels.py`**
- `reg_stub_trackId == 0` behaves as noise/PU stub — measure rate by trackId value
- `reg_stub_trackId > 0` maps to expected GenMuon index range per dataset
- `reg_stub_ambiguous` is not trivially all-zero or all-one; measure fraction per dataset
- Multi-muon events (S3/S4/S5): confirm track IDs non-colliding within each processor window
- Measure same-track vs cross-track legal pair ratio — needed to assess edge label balance

**`src/audit/check_joins.py`**
- `reg_eventNum` uint32-cast matches `Events.event` in NanoAOD
- `track_id - 1` indexing agreement with `GenMuon_pt` etc.
- No systematic off-by-one issues
- Join success rate per dataset bucket

**`src/audit/occupancy.py`**
- Stubs per processor window: histogram + mean/p95/p99/max per dataset
- Per-type and per-layer stub distributions
- BX distribution per stub type
- PU vs no-PU occupancy comparison
- Output: `occupancy_summary.csv`

**`src/audit/feature_ranges.py`**
- Min/max/mean/std per raw stub feature
- Per derived pair feature: Δphi, Δr, Δr², kappa_hat, |Δeta|, Δbx, phiB_diff
- Outlier detection (values beyond 3σ)
- Output: `data/prod/feature_ranges.json` — used later for quantization range-setting

**`scripts/audit/run_dataset_audit.py`**
- Master runner: calls all checks, produces `audit_report.md` and plots
- Runs on a sample of files per dataset bucket (B1–B4, S1–S5)

### Fixed-Nmax policy decision (checkpoint within Stage 1)

After `occupancy.py` runs, a formal decision is required before Stage 2 proceeds.

| Decision | Options |
|---|---|
| Padding strategy | Zero-pad stubs to `Nmax`; provide `valid_mask` |
| Truncation strategy (stubs > Nmax) | By quality (descending), by BX (in-time first), by detector priority, or random |
| Nmax value | Set from p99 occupancy across all datasets; likely 32, 48, or 64 |
| Stub ordering within window | Sorted by layer, by type, or no ordering (padded at end) |

**Nmax is not just a statistic. For firmware it becomes architecture.** It is
fixed here and does not change without an explicit decision.

Record the chosen values in `docs/omtf/DATASET_SIGNOFF.md`.

### Required sign-off artifact

**`docs/omtf/DATASET_SIGNOFF.md`** must answer all of the following before
Stage 2 proceeds:

- Are joins correct? (yes/no + evidence)
- Is `trackId` trustworthy? (yes/no + fractions)
- What is the recommended `Nmax`? (with occupancy table)
- Are ambiguous labels usable? (ambiguous fraction per dataset)
- Are pair features numerically stable? (range table)
- Are S1–S5 / B1–B4 behaving as intended? (per-dataset stub count and
  trackId statistics)

**This document is the hard gate. No Stage 4 or later begins without it.**

---

## Stage 2 — Minimal OMTF Python pipeline

**Goal**: Read ROOT files, produce tensors, run a trivial forward pass. No
serious model yet. Establishes the data pipeline foundation.

**Deliverables**:

**`src/omtf/dataset.py`** — ROOT → PyTorch Dataset
- Uses `uproot` for reading (pure Python, no CMSSW dependency)
- Per entry: stub arrays → pad to `Nmax` using policy from Stage 1 → sample tensor
- Output per sample: `{stubs: (Nmax, 8), valid_mask: (Nmax,), track_ids: (Nmax,), ambiguous: (Nmax,)}`
- Lazy loading with file-level caching
- Supports reading from multiple dataset buckets (S1–S5, B1–B4)

**`src/omtf/features.py`** — derived pair features
- Computes `(edge_index, edge_features, edge_labels)` per window
- Derived features: Δphi, Δr, Δr², kappa_hat, |Δeta|, Δbx, phiB_diff,
  layer-pair category, type-pair category
- `edge_labels` from `track_ids` (same-track = 1, cross-track = 0)
- Output: `(Nmax*(Nmax-1), F_edge)` edge feature tensor

**`src/omtf/splits.py`** — file-level train/val/test split per dataset bucket
- Outputs `splits.json` with file lists per split and per dataset
- Stratification by dataset type (S/B) and optional pT range

**`src/omtf/losses.py`** — loss stubs
- Node-level BCE (signal vs noise)
- Edge-level BCE (same-track vs cross-track)
- pT regression loss (MSE on 1/pT or direct pT)

---

## Vertical Slice V1 — named milestone

**This is the first "it works" moment. All criteria must pass.**

1. Load one ROOT file from S1 using `src/omtf/dataset.py` without errors
2. Validate join to NanoAOD for at least 95% of entries in that file
3. Produce one padded processor-window sample (`Nmax` stubs, `valid_mask`, `track_ids`)
4. Compute derived pair features from `src/omtf/features.py` with no NaN
5. Run one forward pass through the trivial DeepSets baseline (Baseline A) without errors
6. Compute edge-label AUC on S1 validation split — confirm well above 0.5
7. Run one forward pass on B4 — confirm model produces output without crashing
8. Export one batch of test vectors in flat text format matching the existing
   HLS testbench pattern (`network_input.txt` / `weights_layer1.txt` convention)

**V1 is not about accuracy. It is about end-to-end pipeline integrity.**

---

## Stage 3 — First baselines

**Goal**: Establish a performance floor before any serious model. Two baselines.

### Baseline A — DeepSets / pooled MLP

- Embed each stub with a small per-stub MLP
- Global mean pool over valid stubs (respecting `valid_mask`)
- Decode K=3 candidate outputs (pT/charge/phi per slot)
- Also decode per-node signal/noise classification

**What this tests**: Whether any muon information can be recovered at all from
this data with a simple non-relational model.

### Baseline B — Pure edge MLP (no global aggregation)

- For each stub pair, compute derived features from `src/omtf/features.py`
- Small per-edge MLP
- Predict edge-level binary label (same-track or not)
- Loss: edge BCE only. No candidate output, no regression.

**What this tests**:
- Whether the pair supervision is healthy before building full candidate output logic
- Whether kappa_hat, phiB_diff etc. are actually discriminating

### Decision from Baseline A and B

| Result | Implication |
|---|---|
| Baseline A near random AUC | Data pipeline or label problem; return to Stage 1 |
| Baseline A reasonable | Relation modeling is potentially beneficial |
| Baseline B edge AUC > 0.85 on S1 | Pair features are strong; Stage 4 is well-motivated |
| Baseline B edge AUC < 0.7 | Investigate derived features and label quality before proceeding |

### Scripts to create

```
src/omtf/models/baselines.py    Baseline A (DeepSets) and Baseline B (pure edge BCE)
src/omtf/metrics.py             four-metric evaluation utilities
src/omtf/eval.py                evaluation entrypoint, summary tables + plots
src/omtf/train.py               OMTF training entrypoint
```

### Standard evaluation metric split (applies to all models from here on)

**All evaluation tables must use this exact split:**

| Metric | Definition |
|---|---|
| Stub-to-track recovery efficiency | Fraction of signal stubs correctly classified as signal at node level |
| Candidate recovery efficiency | Fraction of gen-muons matched to a model output candidate (proxy — not full GMT efficiency) |
| Background acceptance proxy | Fraction of B4 processor windows where model outputs at least one candidate above threshold |
| Zero-track false positive rate | Same as background acceptance; B4 is a proxy only |

**Explicit scope statement (must appear in every evaluation table)**:
> Full trigger rate requires minimum-bias full events through GMT emulation.
> B4 gives a background-acceptance proxy only, not absolute CMS L1 rate.

### Initial evaluation datasets

- S1 — clean single-track regression and classification
- B4 — zero-track false positive rate

---

## Stage 4 — Edge-compatibility network (first serious deployment candidate)

**Prerequisites**:
- `DATASET_SIGNOFF.md` signed off
- Baseline B edge AUC confirms pair features are discriminating

**`src/omtf/models/edge_compat.py`**
- Input: stub node features + derived pairwise features
- Per-edge MLP → edge compatibility score
- Per-node aggregation of edge scores → node embedding
- K=3 output heads (pT/charge/phi regression + candidate score per slot)
- Loss: edge BCE + pT regression

Training: mixed S1 + S3 + B1 + B4

Evaluation: full four-metric split on S1, S3, B1, B4

---

## Stage 5 — Fixed-K slot model (second deployment candidate)

**`src/omtf/models/slot_model.py`**
- K=3 fixed output slots
- Stub-to-slot compatibility scores
- Per-slot stub aggregation → slot embedding → K output heads
- Loss: Hungarian matching or fixed-assignment loss on slot outputs

Evaluation: side-by-side comparison table vs edge-compat on all four metrics
across S1, S3, B1, B4, B4 threshold scan

---

## Architecture Decision Gate

**This gate is required after Stages 3–5. It prevents endless parallel growth.**

After baselines, edge-compat model, and slot model are evaluated, explicitly assign:

| Role | Assigned to |
|---|---|
| Main deployment candidate | TBD from Stage 5 comparison |
| Secondary deployment candidate | TBD from Stage 5 comparison |
| Software teacher path | Moves to Stage 6 — only after main candidate identified |

**Decision criteria**:

| Criterion | Tests |
|---|---|
| Candidate recovery efficiency on S3 | Multi-track disambiguation |
| Background acceptance proxy on B4 | Fake rate behavior |
| Edge AUC vs compute cost | Whether relation modeling pays off |
| Quick PTQ sensitivity check | Firmware readiness |

The architecture that wins on {candidate recovery + background proxy +
quantization robustness} becomes the main deployment candidate.

**Both main and secondary candidates move to Stage 7. Stage 6 is parallel, not blocking.**

---

## Stage 6 — Full HECIN software reference (optional, non-blocking)

**Constraint**: Stage 6 does not block Stages 7–8. It runs in parallel with
the firmware path once the Architecture Decision Gate is passed.

**Goal**: Software performance ceiling. Optional teacher model.

**`src/omtf/models/hecin.py`**
- Full heterogeneous edge-conditioned interaction network
- Multi-round message passing over legal sparse graph
- Optional OC-style multi-track readout

**Questions Stage 6 answers**:
- Does richer multi-round interaction materially improve close-muon separation
  over the deployment-oriented models?
- Does it improve displaced muon reconstruction (S2/S5/B2)?
- Is it worth distilling into Stage 8?

**If the HECIN ceiling is not meaningfully above the Stage 5 winner on
close-muon and displaced tasks, Stage 8 is skipped entirely.**

---

## Stage 7 — Quantization and HLS path

**Prerequisites**: Architecture Decision Gate passed. Main deployment candidate identified.

**Deliverables**:

- PTQ flow for chosen model using `quantization_ptq.py`
  - Calibration using `feature_ranges.json` from Stage 1
  - Export INT8 weights via `export_weights_as_c_header`
- `tests/generate_test_vectors_omtf.py` — OMTF test vector generator (flat text, matching existing convention)
- `hls/kernels/edge_feature_calc.h` — pairwise feature computation kernel (ap_fixed)
- `hls/kernels/edge_scorer.h` — edge compatibility scoring kernel (INT8 or ap_fixed)
- `hls/kernels/slot_assign.h` or `hls/kernels/candidate_accum.h` (depending on winning architecture)
- `hls/kernels/regression_head.h` — compact pT/charge/phi decoder per candidate
- `configs/omtf_design_space.yaml` — DSE config for OMTF kernel parameters
- DSE run: latency × DSP × efficiency Pareto for OMTF deployment candidates

The existing `explore_design_space.py`, `analyze_pareto.py`, TCL Jinja2
templates, and `parse_hls_report.py` are reused as-is.

---

## Stage 8 — HECIN-Lite (conditional)

**Triggered only if**: Stage 6 shows a meaningful accuracy gap on close-muon
or displaced tasks that cannot be closed by tuning Stage 4–5 models.

**`src/omtf/models/hecin_lite.py`**
- Static sparse HECIN with fixed K=3 slot readout
- One or two edge-conditioned interaction rounds maximum
- Tiny hidden dimensions; no dynamic graph construction; no OC clustering
- Knowledge distillation from Stage 6 HECIN teacher

---

## Exact First Files to Create

In order of creation:

### Stage 0 — additive stubs
```
src/omtf/__init__.py
src/omtf/models/__init__.py
src/audit/__init__.py
docs/omtf/MIGRATION_STATUS.md
docs/omtf/DATASET_SIGNOFF.md          (stub with required sections)
configs/model_config.yaml              (extend with omtf: section, stubs only)
```

### Stage 0.5 — import audit (internal note, no new file required)

### Stage 1 — dataset audit
```
src/audit/check_branches.py
src/audit/check_labels.py
src/audit/check_joins.py
src/audit/occupancy.py
src/audit/feature_ranges.py
scripts/audit/run_dataset_audit.py
```

### Stage 2 — minimal pipeline
```
src/omtf/dataset.py
src/omtf/features.py
src/omtf/splits.py
src/omtf/losses.py
```

### Stage 3 — baselines
```
src/omtf/models/baselines.py          (Baseline A: DeepSets; Baseline B: pure edge BCE)
src/omtf/metrics.py
src/omtf/eval.py
src/omtf/train.py
```

### Stage 4+
```
src/omtf/models/edge_compat.py
src/omtf/models/slot_model.py
src/omtf/models/hecin.py              (Stage 6, optional)
src/omtf/models/hecin_lite.py         (Stage 8, conditional)
hls/kernels/edge_feature_calc.h
hls/kernels/edge_scorer.h
hls/kernels/slot_assign.h
hls/kernels/regression_head.h
tests/generate_test_vectors_omtf.py
configs/omtf_design_space.yaml
```

---

## What Stays from the Existing Codebase

| Component | Status |
|---|---|
| `src/explore_design_space.py` | Keep, extend for OMTF design points |
| `src/analyze_pareto.py` | Keep as-is |
| `src/quantization_ptq.py` | Keep as-is |
| `src/brevitas_*.py` | Keep, extend for OMTF models in Stage 7 |
| `src/optimize_bitwidths_int8.py` | Keep, refactor parameters for OMTF |
| `hls/graphsage_layer_*.{h,cpp}` | Keep as legacy reference |
| `hls/tcl_example/` | Keep as-is |
| `scripts/parse_hls_report.py` | Keep as-is |
| `src/visualization.py` | Keep, extend with trigger-specific plots |
| `src/train.py`, `src/model_base.py` | Keep in place (legacy path, not moved yet) |

---

## What is Not Yet Available

For reference: these metrics and studies are **not possible** with the current
9-dataset production. They require additional samples or infrastructure.

| Study | What is missing |
|---|---|
| Full L1 trigger rate | Min-bias full events + GMT emulation |
| Absolute background acceptance | Rate normalization (cross-section × luminosity) |
| Full detector fake-rate study | Full event processing, not just processor-window |
| LLP exotic stress-test (D1) | Not in current production |

B4 provides a background-acceptance proxy and a zero-track false positive rate
at the processor-window level. These are useful for threshold setting but should
not be described as final CMS L1 rates.
