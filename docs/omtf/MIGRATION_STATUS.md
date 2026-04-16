# OMTF Migration Status

**Branch**: `omtf-migration`  
**Plan reference**: `docs/omtf/MIGRATION_PLAN.md`

---

## Stage Tracker

| Stage | Description | Status |
|---|---|---|
| 0 | Additive repository preparation | In progress |
| 0.5 | Import graph audit | Complete (see below) |
| 1 | Dataset audit | In progress |
| 2 | Minimal OMTF Python pipeline | Pending |
| V1 | Vertical Slice V1 milestone | Pending |
| 3 | First baselines (A + B) | Pending |
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

## Legacy path

The existing Cora/GraphSAGE code is the **legacy path**. It remains fully functional and untouched.

Files marked as legacy (do not extend for OMTF):
- `src/train.py`
- `src/model_base.py`
- `src/model_base_QAT.py`
- `src/model_qat.py`
- `src/model_qat_v2.py`
- `src/subgraph_extraction.py`

These will be moved to `src/legacy/` only after the first OMTF vertical slice (V1) is working.

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
