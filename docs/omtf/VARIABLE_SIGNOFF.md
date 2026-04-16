# OMTF Variable Physical Validation Sign-Off

**Status**: SIGNED OFF  
**Date**: 2026-04-16  
**Produced by**: `src/audit/check_variables.py --all --max-files 3 --max-entries 300`  
**Artifact**: `build/audit/variable_check.json`

---

## Summary

All variables are physically consistent across all 9 datasets. The 99 "issues" reported by the first run were false positives from incorrect assumed bounds (wrong OMTF hardware encoding). After correcting the bounds to match actual OMTF Phase-2 encoding, zero issues remain.

**Critical finding**: GenMuon pT is accessible via NanoAOD join in all signal datasets (S1–S5, B1–B3). GenMuon_d0 (displacement) is NOT available in these files.

---

## 1. Variable encoding — corrected understanding

### phi (reg_stub_phiHw)
- **Range**: approximately [−463, +1854] across all datasets
- **Interpretation**: hardware phi within processor window, 12-bit signed integer
- **Status**: CLEAN. phi=0 is one edge of the processor window; full range is ~2300 units wide, consistent with 60° processor sectors in OMTF phi units.

### phiB (reg_stub_phiBHw)
- **Range**: approximately [−987, +1078]; ~99% of values within [−512, +511]
- **Interpretation**: phi bending at the muon station. DT and CSC have non-zero phiB. **RPC always has phiB=0** (100.0% of type-5 stubs) — RPC has no directional bending measurement.
- **Status**: CLEAN. The ~0.3–1.6% tail beyond ±512 is from high-curvature tracks. Confirmed consistent with Stage 1 feature range audit.
- **Action for models**: phiB for RPC (type=5) is always 0 and carries no information. Consider masking phiB for RPC stubs or using it as-is (it will learn this).

### eta (reg_stub_etaHw)
- **Range**: [58, 127] — all datasets use positive-eta region (processors 0–2)
- **Interpretation**: OMTF hardware eta code (unsigned integer). NOT a signed small index. The range 58–127 corresponds to the positive overlap muon region.
- **Unique values**: 43–69 per dataset (chamber-level granularity)
- **Status**: CLEAN. Previous assumption of [−16, 16] was wrong.

### r (reg_stub_r)
- **Range**: [281, 680] in hardware units (NOT metres or cm)
- **Interpretation**: hardware radial position code. Constant per detector ring.
- **Layer-r structure** (critical for understanding):
  - Layers 0, 2, 4, 10, 11, 12, 13, 14: **exactly one unique r value** → single-ring RPC barrel layers. Geometrically fixed.
  - Layers 6, 7, 8, 9: **multiple r values per layer** (std=30–110) → CSC chambers from multiple rings (ME1/1, ME1/2, ME1/3 etc.) grouped into the same OMTF layer number.
  - Layers 15, 16, 17: **multiple r values** → RPC endcap layers, multiple rings.
- **Status**: CLEAN. Multi-r layers are expected physics, not data corruption.
- **Action for models**: `r` should be used as-is — it encodes both layer and ring position simultaneously. It is a useful geometric feature.

### quality (reg_stub_quality)
- **Range**: [1, 6] across all datasets (not 0–3 as sometimes assumed)
- **Distribution**: varies per stub type
  - Type 3 (DT): quality [2, 3, 4, 5, 6]
  - Type 5 (RPC): quality [1, 2, 3]
  - Type 9 (CSC): quality [1, 2, 3]
- **Status**: CLEAN. Quality=0 stubs not present (likely filtered at production).

### type (reg_stub_type)
- **Values observed**: {3, 5, 9} only — exactly 3 stub types in all datasets
- **Encoding** (Phase-2 OMTF):
  - **3 = DT** (Drift Tube): quality [2–6], phiB ≠ 0, ~20–30% of stubs
  - **5 = RPC** (Resistive Plate Chamber): quality [1–3], phiB = 0 always, ~53–66% of stubs
  - **9 = CSC** (Cathode Strip Chamber): quality [1–3], phiB ≠ 0, ~17–25% of stubs
- **Status**: CLEAN. Previous assumption of [0, 7] was wrong.
- **Composition is stable across datasets** (signal and background):
  - RPC dominates (~54–66%) due to RPC barrel coverage
  - DT: 9–30% (lower in PU-heavy datasets as PU noise is RPC-dominated)
  - CSC: 17–25%

### layer (reg_stub_layer)
- **Range**: [0, 17] — exactly 18 OMTF layers as expected
- **Unique per dataset**: 14–15 layers populated (not all 18 hit in every sample)
- **Status**: CLEAN.

### bx (reg_stub_bx)
- **Range**: [0, 0] — identically 0 in all 9 datasets
- **Status**: Confirmed uninformative in current samples. See DATASET_SIGNOFF §5.
- **Action for models**: exclude from model inputs. Confirmed against real mixed-BX production before enabling.

---

## 2. Stub type composition per dataset

| Dataset | DT (type=3) | RPC (type=5) | CSC (type=9) |
|---|---|---|---|
| S1 | 29.6% | 53.0% | 17.5% |
| S2 | 27.3% | 54.6% | 18.1% |
| S3 | 21.4% | 56.3% | 22.3% |
| S4 | 24.0% | 56.3% | 19.6% |
| S5 | 28.0% | 52.9% | 19.1% |
| B1 | 25.4% | 54.7% | 20.0% |
| B2 | 18.4% | 57.3% | 24.3% |
| B3 | 20.1% | 56.2% | 23.7% |
| B4 | 9.5% | 65.7% | 24.8% |

**Observation**: B4 (noise-only PU200) has significantly fewer DT stubs (9.5%) and more RPC stubs (65.7%). PU noise is dominated by RPC hits, as expected. This is a potential discriminating feature for noise rejection.

---

## 3. iProcessor distribution

- Only processors 0, 1, 2 appear (positive-eta side)
- Distribution roughly uniform across the 3 processors (~33% each)
- S3/S4 (two/three muons same window): slight imbalance (proc 0 and 2 ~50% each, proc 1 near-absent in S4) — this is a generator gun artifact, not a data quality issue.
- **Status**: CLEAN.

---

## 4. GenMuon pT — accessible and correct

GenMuon pT is accessible via the NanoAOD join (reg_eventNum ↔ event, uint32 cast) for all signal datasets.

| Dataset | N matched | pT p5 | pT p50 | pT p95 | max | Notes |
|---|---|---|---|---|---|---|
| S1 | 1379 | 4.8 GeV | 22.6 GeV | 161.5 GeV | 198.5 GeV | Prompt, flat gun — wide pT range |
| S2 | 657 | 3.0 GeV | 5.8 GeV | 28.0 GeV | 194.5 GeV | Displaced, softer spectrum |
| S3 | 1859 | 3.6 GeV | 7.4 GeV | 40.4 GeV | 97.0 GeV | Two prompt muons |
| S4 | 3519 | 5.3 GeV | 9.7 GeV | 48.4 GeV | 79.2 GeV | Three prompt muons, narrow gun |
| S5 | 1323 | 3.0 GeV | 6.4 GeV | 41.1 GeV | 189.0 GeV | Two displaced muons |
| B1 | 758 | 5.4 GeV | 21.5 GeV | 130.0 GeV | 198.5 GeV | Prompt + PU200 |
| B2 | 262 | 2.8 GeV | 7.0 GeV | 36.8 GeV | 61.5 GeV | Displaced + PU200 |
| B3 | 1241 | 3.5 GeV | 7.8 GeV | 41.1 GeV | 97.0 GeV | Two prompt + PU200 |

**S1 vs S2**: S1 (prompt) has p50=22.6 GeV vs S2 (displaced) p50=5.8 GeV — displaced muons have a softer pT spectrum, as expected.

**GenMuon_charge**: Available. Roughly balanced (+1/−1 ~50/50) across all datasets.

**GenMuon_d0 (displacement)**: **NOT AVAILABLE** in these files. Displacement regression is not possible without it. If needed, must be confirmed to exist in production NanoAOD files.

---

## 5. Layer-r structure — model implications

The multi-ring structure of CSC and RPC endcap layers means:
- `r` is NOT redundant with `layer` — two stubs on the same `layer` can have very different `r` (different rings)
- `(layer, r)` together uniquely identify the detector ring
- For models: using `r` as a continuous feature is correct. Do not attempt to embed `layer` alone as if it fully specifies detector geometry.

Fixed-r layers (stubs spread=0): layers 0, 2, 4, 10, 11, 12, 13, 14 → these are RPC barrel layers, single ring.

---

## 6. What we have for models

| Target/Feature | Available | Notes |
|---|---|---|
| Node signal/noise classification | ✓ | trackId != 0 |
| Edge same-track classification | ✓ | trackId matching |
| pT regression | ✓ | GenMuon_pt via NanoAOD join |
| Charge classification | ✓ | GenMuon_charge via join |
| Displacement (d0) | ✗ | GenMuon_d0 not in files |
| Ambiguity masking | ✓ | reg_stub_ambiguous |
| Multi-muon scenarios (S3/S4/S5) | ✓ | Multiple trackId values per window |

---

## Sign-off Checklist

- [x] phi: correct hardware encoding, range [−463, +1854], no anomalies
- [x] phiB: DT/CSC non-zero, RPC always 0 — physically correct; ~1% tails expected
- [x] eta: hardware code [58, 127] for positive-eta region — correct
- [x] r: hardware distance code [281, 680]; constant per ring, multi-r per CSC/RPC-endcap layer — correct
- [x] quality: range [1, 6]; DT uses [2–6], RPC/CSC use [1–3] — correct
- [x] type: DT=3, RPC=5, CSC=9 — documented and consistent across all datasets
- [x] layer: [0, 17], 14–15 layers populated per sample — correct
- [x] bx: all 0 — uninformative in current samples, exclude from models
- [x] GenMuon pT: accessible via join, distributions match dataset intent
- [x] GenMuon_d0: NOT available — displacement regression requires production NanoAOD confirmation

**Signed off by**: pleguina  
**Date**: 2026-04-16
