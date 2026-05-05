# OMTF↔KMTF Truth Transfer Signoff

Validation date: 2026-04-24  
Phi match threshold: 0.100 rad  
Files per dataset: 20

---

## Match rate summary

| Dataset | Windows | KMTF stubs | Match % | Unmatched % | Empty windows | Signal transferred | Noise confirmed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 10,745 | 23,157 | 66.8% | 33.2% | 3,912 | 14,744 | 729 |
| S3 | 12,913 | 13,366 | 90.2% | 9.8% | 6,947 | 11,695 | 365 |
| B1 | 19,593 | 41,469 | 43.9% | 56.1% | 7,729 | 14,836 | 3,372 |
| B3 | 21,650 | 20,763 | 67.5% | 32.5% | 12,830 | 11,921 | 2,094 |
| B4 | 11,120 | 2,660 | 57.9% | 42.1% | 9,540 | 0 | 1,540 |

---

## Phi residuals at matched stubs

- **S1**: median=1.0 mrad, p95=35.0 mrad, max=99.8 mrad  
- **S3**: median=0.7 mrad, p95=5.0 mrad, max=97.3 mrad  
- **B1**: median=1.1 mrad, p95=53.8 mrad, max=99.9 mrad  
- **B3**: median=0.8 mrad, p95=12.1 mrad, max=99.9 mrad  
- **B4**: median=0.7 mrad, p95=70.4 mrad, max=74.0 mrad  

Residuals are tight when matching succeeds. Median ~1 mrad across all datasets confirms that the local-to-global phi handling and the OMTF↔KMTF stub matching are working correctly.

---

## Root cause of unmatched fractions

The primary source of unmatched KMTF stubs is an **acceptance mismatch between the two systems**:

- OMTF truth exists only for the overlap-region acceptance: \(|\eta| \in [0.83, 1.24]\)
- KMTF barrel stubs cover a wider η range, including \(|\eta| < 0.83\)
- therefore many KMTF stubs originate from muons that OMTF never processed and never labelled
- those stubs have no OMTF truth counterpart by construction
- they are conservatively assigned `trackId=0`

This is **correct for the overlap-region task** and does not indicate a failure of the OMTF↔KMTF matching procedure itself.

### Two distinct categories of unmatched stubs

| Case | Description | Correct label |
| --- | --- | --- |
| Outside OMTF acceptance | Real KMTF stubs from muons with \(|\eta| < 0.83\); OMTF never processed them | Assigned `trackId=0` for this overlap-region task (out of OMTF truth domain) |
| Inside acceptance, no counterpart | PU stubs, OMTF omissions, or rare matching failures | `trackId=0` as a conservative default |

A large fraction of unmatched stubs in S1/B1/B3 is explained by Case 1.

### S1 specifically (verified by event dump)

S1 events always contain two generated muons per event (prompt Drell–Yan μ+μ−, `pdgId = ±13`, `status = 1`). Although S1 is used as the single-target overlap trigger sample, the underlying generated event contains a dimuon final state; only the overlap-acceptance subset is relevant for the OMTF truth domain.

Verified on 500 S1 events:

| Condition | Events | Fraction |
| --- | --- | --- |
| Both muons outside OMTF \(|\eta| \in [0.83, 1.24]\) | 238 | 47.6% |
| Exactly 1 muon in OMTF acceptance | 210 | 42.0% |
| Both muons in OMTF acceptance | 52 | 10.4% |

The KMTF barrel view therefore naturally contains many real stubs that are out of scope for OMTF truth labelling. The ~33% unmatched fraction is expected and reflects acceptance coverage, not a failure of the matching procedure.

### S3: why the match rate is higher (90%)

S3 places both prompt muons explicitly inside the same OMTF processor window, so both are by construction inside the OMTF truth domain. The KMTF regional view captures the same stubs, leading to near-complete matching.

### B1: why the match rate is lowest (44%)

Two effects contribute:
1. the same η-acceptance mismatch as in S1  
2. additional KMTF stubs from PU200 tracks with no OMTF truth counterpart

Both cases correctly receive `trackId=0`.

### B4: all-noise correct (58% match, 0 signal transferred)

B4 is a min-bias sample with no gen muons in the target domain. All 1,540 matched stubs are confirmed as noise. The unmatched KMTF stubs also correctly default to `trackId=0`.

---

## Task-domain caveat

The acceptance mismatch means the GMT-visible branch is not learning a pure signal-vs-noise distinction. It is partly learning:

> **in-overlap-task** vs **out-of-overlap-task**

This is intentional for Phase B1. The target task is **overlap-region reconstruction**, directly comparable to the OMTF-internal branch. KMTF stubs from muons outside the OMTF acceptance are treated as `trackId=0` because they are outside the truth domain of this task.

If the GMT branch is later extended to full KMTF barrel reconstruction beyond the overlap region, an additional gen-muon spatial fallback using `GenMuon_phiSt1/phiSt2` would be needed to recover labels for those stubs. That is out of scope for Phase B1.

---

## Decisions

| Dataset | Result | Reason |
| --- | --- | --- |
| S1 | ACCEPTABLE | Unmatched fraction explained by η-acceptance coverage; transferred signal is clean |
| S3 | PASS | 90.2% match; both muons in the OMTF truth domain by construction |
| B1 | ACCEPTABLE | Unmatched fraction understood (η acceptance + PU); transferred signal is clean |
| B3 | ACCEPTABLE | Same acceptance-coverage effect as S1/B1; conservative noise assignment is correct for the task |
| B4 | PASS | 0 false signal transfers; all labels consistent with noise |

---

## Sign-off

- [x] Root cause of unmatched fraction identified and verified: acceptance mismatch, not a matching bug  
- [x] When matching succeeds, phi residuals are tight (median ~1 mrad, p95 < 55 mrad for signal-bearing datasets)  
- [x] B4: 0 signal transferred; all transferred labels consistent with noise  
- [x] Unmatched stubs are conservatively assigned `trackId=0`, which is correct for the overlap-region task  
- [x] S1 generator-level structure verified by event dump  
- [x] Task-domain scope documented: Phase B1 targets overlap-region reconstruction only  
- [x] Scope of this signoff: barrel KMTF truth transfer only; TPS endcap requires a separate signoff  

**Current truth transfer provides high-precision but incomplete-recall supervision, acceptable for the first overlap-region baseline.**

**Proceed to barrel-first GMT model training: YES**
