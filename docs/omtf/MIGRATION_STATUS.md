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
| 3 | First baselines (A + B) — full training runs | **Pending** (smoke-test done) |
| 4 | Edge-compatibility network | Pending |
| 5 | Fixed-K slot model | Pending |
| ADG | Architecture Decision Gate | Pending |
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

Note: AUC in criterion 6 is from a 20-epoch smoke-test run (3 files, 200 entries/file).
Full training runs needed before making Stage 3 go/no-go decision.

---

## Stage 3 — Baseline Runs

### Code status: Complete
- `src/omtf/models/baselines.py` — DeepSets (A) and EdgeMLP (B)
- `src/omtf/train.py`, `eval.py`, `metrics.py`

### Smoke-test run (2026-04-17): 3 files × 200 entries/file, 20 epochs

**Baseline A — DeepSets (S1 + B1 + B4, hidden=64)**

| Dataset | node_auc | stub_eff | cand_rec | bg_accept |
|---|---|---|---|---|
| S1 | 0.539 | 0.922 | 1.000 | — |
| B4 | — | — | — | 0.315 |

**Baseline B — EdgeMLP (S1 only, hidden=64)**

| Dataset | edge_auc | bg_accept |
|---|---|---|
| S1 | 0.664 | — |
| B4 | — | 0.917 |

### Interpretation (smoke-test, not final)

Per the plan's decision criteria:
- Baseline A node_auc=0.54 is near-random — expected with 1800 samples and 20 epochs.
  Pipeline integrity confirmed; model has not converged.
- Baseline B edge_auc=0.66 on S1 is above random but below the 0.85 target.
  Not a concern at this data scale; needs full-data run.
- B4 bg_accept=0.915 for Baseline B is high — logit threshold of 0.0 is too loose
  for an untrained edge model; threshold tuning required at full scale.

**Full training runs required before making the Stage 3 go/no-go decision.**
Recommended: `--max-files 50 --epochs 50` for an intermediate run.

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
