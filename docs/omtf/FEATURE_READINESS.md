# OMTF Feature Readiness Assessment

**Status**: COMPLETE  
**Date**: 2026-04-16  
**Branch**: omtf-migration

This document answers: what is available, what is in good shape, and what additional
variables could help the models.

---

## 1. Model input features — ready

| Feature | Branch | Range | Notes |
|---|---|---|---|
| phi | reg_stub_phiHw | [−463, +1854] | HW units; clean |
| phiB | reg_stub_phiBHw | [−987, +1078] | DT/CSC non-zero; RPC always 0; ~1% tails |
| eta | reg_stub_etaHw | [58, 127] | HW code, positive-eta region; clean |
| r | reg_stub_r | [281, 680] | HW radial code; constant per ring; clean |
| quality | reg_stub_quality | [1, 6] | DT [2–6], RPC/CSC [1–3]; clean |
| type | reg_stub_type | {3, 5, 9} | DT=3, RPC=5, CSC=9; clean |
| layer | reg_stub_layer | [0, 17] | 18 layers; clean |

**bx is excluded**: identically 0 in all current samples → uninformative.  
**Effective node feature dimension: 7** (not 8 as originally planned).

---

## 2. Training targets — ready

| Target | Branch | Available | Notes |
|---|---|---|---|
| Node label (signal/noise) | reg_stub_trackId | ✓ | trackId != 0 |
| Edge label (same-track) | reg_stub_trackId | ✓ | pair-wise matching |
| Ambiguity mask | reg_stub_ambiguous | ✓ | 1–5% of stubs |
| pT regression | GenMuon_pt | ✓ | [2–200 GeV]; distributions validated |
| Charge classification | GenMuon_charge | ✓ | balanced ±1 |
| Displacement (dXY) | GenMuon_dXY | ✓ | S2/S5: 96–97% have \|dXY\|>1 cm |
| Decay length | GenMuon_lXY | ✓ | S2/S5 median ~130–160 cm |

**Note**: GenMuon_dXY branch is named `dXY`, not `d0`.

---

## 3. Additional variables that could help

### A. GenMuon extrapolated station positions (high value)

Branches: `GenMuon_etaSt1`, `GenMuon_phiSt1`, `GenMuon_etaSt2`, `GenMuon_phiSt2`

These are the **extrapolated muon trajectory positions at OMTF stations 1 and 2**,
in floating-point eta/phi (radians). Sentinel value: -9.0 = no extrapolation.

- Valid for all signal muons (S1–S5, B1–B3). B4 is all -9.0 (noise-only, correct).
- `etaSt1` valid range: [−1.4, +1.7]; `phiSt1`: [−π, +π]

**Use cases**:
1. **Auxiliary regression targets**: train the model to predict where the muon goes
   at station 1 and 2. This teaches the model the trajectory concept implicitly.
2. **Additional supervision for edge labels**: a stub at layer L is a better positive
   pair candidate if its eta/phi is consistent with the extrapolated position. Could
   produce soft or weighted edge labels.
3. **Evaluation**: compare predicted track direction with truth extrapolation.

**Requires**: HW eta/phi → float coordinate transformation to align with stub units.

### B. Existing OMTF algorithm output (performance baseline)

Branches: `omtf_hwPt`, `omtf_hwQual`, `omtf_hwEta`, `omtf_hwPhi`, `omtf_hwDXY`,
`omtf_processor`, `nomtf`, `omtf_muIdx`

The NanoAOD contains the output of the **current production OMTF algorithm**.
This is the baseline our GNN is benchmarked against.

Key observations:
- `omtf_hwPt` range: [1, 401] HW units — the existing pT assignment
- `omtf_hwQual`: 4 values {0, 1, 8, 12}
- **`omtf_hwDXY` is always 0** — the current OMTF has no displacement output.
  Our model's dXY regression is a new capability, not a reimplementation.
- **`omtf_muIdx` is always −1** — truth matching between OMTF candidates and
  GenMuons was not run in this production. Cannot use OMTF output as per-stub
  supervision, but can use as system-level efficiency/resolution baseline.
- ~1.1 OMTF candidates per window for S1 (signal); ~0.2 for B4 (noise).

**Use in evaluation**: compare GNN pT resolution (GeV scale via regression) against
OMTF hwPt (HW scale, needs LUT inversion or comparison on rank).

### C. reg_mtfType and iProcessor (window-level context)

- `reg_mtfType` ∈ {1, 2}: ~50/50 split. Likely encodes processor sub-type
  (e.g., positive-eta sub-sector). Usable as a 1-dimensional window feature.
- `reg_iProcessor` ∈ {0, 1, 2}: which of the 12 OMTF processors (only 3 positive-eta
  sectors in current data). Encodes the phi sector. Could embed as sector index.

**Low priority**: models should be sector-agnostic. Use only if sector-dependent
bias is observed in evaluation.

### D. MuonStubKmtf / MuonStubTps (do NOT use directly)

These are stub collections from the **KMTF (Kalman Muon Track Finder)** and **TPS**
algorithms — parallel trigger chains, not the OMTF. Their stub counts differ from
OMTFAllInputTree (overlap only 9% of windows). They are NOT an alternative
representation of OMTF stubs and cannot be used as additional features for OMTF
stubs without establishing an explicit correspondence, which is non-trivial.

The `offlineCoord1/2` fields contain float-precision eta/phi, but these are KMTF-
specific coordinates on KMTF-selected stubs.

**Decision**: exclude from OMTF model inputs.

---

## 4. What is missing

| Item | Status | Notes |
|---|---|---|
| Stub position uncertainty | Not available | No per-stub error/covariance in hits tree |
| Sub-BX timing | Not available | All BX=0 in current samples |
| OMTF-to-GenMuon matching | Not available | omtf_muIdx=-1 everywhere |
| Displacement regression from existing OMTF | Not available | omtf_hwDXY=0 always |

The absence of stub-level uncertainties means we cannot do Kalman-style
propagation weighting inside the GNN. Use quality as a proxy.

---

## 5. Recommended feature set for Stage 3+ models

**Node features (7-dim)**: `[phi, phiB, eta, r, quality, type, layer]`

**Edge features (6-dim)**: `[delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, phiB_diff]`
(delta_bx excluded — always 0)

**Training targets (primary)**:
- Node BCE: `trackId != 0`
- Edge BCE: same `trackId` and `trackId != 0`
- pT regression: `log(GenMuon_pt)` for signal stubs

**Training targets (optional extensions)**:
- Displacement: `GenMuon_dXY` (sign-aware — not just |dXY|)
- Station regression: `GenMuon_etaSt1`, `GenMuon_phiSt1` as auxiliary targets

**Performance reference**: `omtf_hwPt` / `omtf_hwQual` for per-event comparison in evaluation.

---

## Sign-off

- [x] All 7 model input features validated and in good shape
- [x] bx excluded (uninformative)
- [x] All primary training targets accessible (node label, edge label, pT, dXY, charge)
- [x] Station extrapolation branches identified and available for auxiliary use
- [x] Existing OMTF output available as performance reference
- [x] MuonStubKmtf/Tps assessed — not directly usable, excluded
- [x] No critical missing variables for Stage 3 baseline models

**Signed off by**: pleguina  
**Date**: 2026-04-16
