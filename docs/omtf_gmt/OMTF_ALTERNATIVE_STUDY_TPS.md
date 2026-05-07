# New idea: compare KMTF vs TPS as GMT-visible stub sources for the OMTF overlap task

## Objective

So far, the GMT-visible branch has been trained using **KMTF** stubs. In practice, this means we take barrel/KMTF stubs from the NanoAOD table, project them into OMTF-like processor windows, transfer OMTF truth labels onto them, and train a model to reconstruct overlap-region muon candidates.

The new idea is to build a second equivalent branch, but using **TPS** stubs instead of KMTF stubs.

The main question is:

```text
Which GMT-visible trigger-primitive source is more useful for OMTF-overlap muon reconstruction?
```

This would let us compare three different views of the same physics problem:

```text
1. OMTF-internal branch
   - Native OMTF processing view.
   - Closest reference to the real OMTF trigger path.

2. GMT/KMTF branch
   - Uses KMTF barrel stubs visible in NanoAOD.
   - Provides useful barrel-side information.
   - Has a domain ambiguity problem: it sees real barrel muons that are outside the OMTF overlap target.

3. GMT/TPS branch
   - Uses TPS / EMTF-hybrid stubs visible in NanoAOD.
   - Potentially closer to the endcap/overlap side of the detector.
   - May reduce barrel hard-negative ambiguity, but may also have lower useful coverage for some OMTF-overlap muons.
```

This comparison should **not** be interpreted as an object-by-object comparison, because the graphs are not the same. The fair comparison is at the physics-response level:

```text
efficiency vs pT,
efficiency vs |d0|,
multi-candidate recovery,
hard-negative false-positive rate,
pure-noise false-positive rate,
event-level trigger response.
```

---

## Why this is worth doing

The KMTF branch already showed that GMT-visible information can be useful for the OMTF overlap task, but it also exposed a specific domain issue:

```text
KMTF sees many real barrel muons that are not valid OMTF-overlap candidates.
```

This is why we needed:

```text
stub_in_overlap,
abs_eta,
eta_dist_to_overlap,
G7/G8 hard-negative datasets.
```

TPS stubs are endcap-oriented and may be closer to the OMTF overlap boundary. Therefore, the TPS branch could show one of three outcomes:

```text
Case A — TPS is better than KMTF:
  - lower G7/G8 false-positive rate,
  - similar or better G1/G2 efficiency,
  - less need for hard-negative oversampling.

Case B — KMTF is better than TPS:
  - higher signal efficiency,
  - better multi-target recovery,
  - TPS has too few useful stubs in the overlap region.

Case C — KMTF and TPS are complementary:
  - KMTF gives better signal coverage,
  - TPS gives better hard-negative rejection,
  - a future KMTF+TPS fusion model could be justified.
```

The most interesting scientific result would be Case C, because it would motivate a combined GMT-visible model using both barrel-side and endcap-side primitive information.

---

## Important technical point

The KMTF and TPS NanoAOD tables have similar branch names, but their integer coordinate scales are not fully interchangeable.

For both sources, the safe quantities to use are the offline physical coordinates:

```text
MuonStubKmtf_offlineCoord1  → global phi [rad]
MuonStubTps_offlineCoord1   → global phi [rad]

MuonStubKmtf_offlineEta1    → physical eta
MuonStubTps_offlineEta1     → physical eta
```

For TPS, use:

```text
MuonStubTps_offlineCoord1
MuonStubTps_offlineCoord2
MuonStubTps_offlineEta1
MuonStubTps_offlineEta2
MuonStubTps_quality
MuonStubTps_etaQuality
MuonStubTps_bxNum
MuonStubTps_tfLayer
MuonStubTps_depthRegion
MuonStubTps_stubType
```

Do **not** directly reuse the KMTF interpretation of `coord2`.

For KMTF:

```text
offlineCoord2 ≈ DT phiBend / bending proxy.
```

For TPS:

```text
offlineCoord2 ≈ second phi coordinate for RPC-matched or RPC-only stubs, and 0 for CSC-only stubs.
```

So in the TPS feature builder this should be treated as a generic second-coordinate feature, not as DT bending.

---

## Proposed TPS branch name

Use a separate phase name:

```text
Phase B4 — TPS / EMTF-hybrid GMT-visible branch
```

Do not overwrite the current KMTF cache or results.

Recommended cache layout:

```text
build/omtf_gmt/cache_v2_kmtf/
build/omtf_gmt/cache_v2_tps/
```

or:

```text
build/omtf_gmt/cache_v2/
  kmtf/
  tps/
```

The important rule is: keep KMTF and TPS caches separate.

---

## Step 1 — audit TPS inputs before training

Before building a full TPS cache, write an audit script:

```text
scripts/omtf_gmt/audit_tps_inputs.py
```

The goal is to answer whether TPS has enough useful information for the OMTF-overlap task.

The audit should report, for each dataset:

```text
number of events,
number of OMTF windows,
number of TPS stubs,
mean TPS stubs per event,
mean TPS stubs per OMTF-like window,
fraction of TPS stubs inside the OMTF overlap eta band,
fraction of TPS stubs outside the OMTF overlap eta band,
BX distribution,
eta distribution,
phi distribution,
quality distribution,
stubType distribution,
tfLayer distribution,
depthRegion distribution.
```

The most important first table is:

```text
Dataset | events | TPS stubs | mean stubs/event | mean stubs/window | frac in overlap
```

If TPS has very low occupancy in G1–G6, it may not be useful for this task.

---

## Step 2 — implement TPS truth transfer

For KMTF, truth labels were transferred from OMTF stubs to KMTF stubs using phi/BX/layer-style matching.

For TPS, the equivalent matching should start from:

```text
TPS global phi  = MuonStubTps_offlineCoord1
TPS eta         = MuonStubTps_offlineEta1
TPS BX          = MuonStubTps_bxNum
TPS layer info  = MuonStubTps_tfLayer / depthRegion / stubType
```

The OMTF phi must be globalized before matching:

```text
phi_global = (reg_stub_phiHw + phiZero(proc)) * 2π / 5400
phiZero(proc) = 1800 * proc + 225
```

Then compare with:

```text
MuonStubTps_offlineCoord1
```

using angular wrapping.

Start with a threshold scan:

```text
Δphi < 5 mrad
Δphi < 10 mrad
Δphi < 20 mrad
Δphi < 40 mrad
```

Choose the smallest threshold that gives stable matching without obvious fake matches.

The TPS truth-transfer audit should report:

```text
match percentage,
phi residual median,
phi residual p95,
number of signal-transferred stubs,
number of confirmed-noise stubs,
unmatched fraction,
target multiplicity distribution.
```

---

## Step 3 — build a TPS cache with the same output schema

Create a TPS cache builder, for example:

```text
scripts/omtf_gmt/make_gmt_dataset_tps.py
```

The output schema should match the KMTF cache as much as possible:

```text
stubs
valid_mask
track_id
ambiguous
node_label
truth_source
gen_pt
gen_charge
gen_dxy
meta_event_num
meta_i_proc
meta_n_stubs
meta_n_gen
meta_is_hard_neg
```

This allows the existing `train.py` and `eval_gmt.py` scripts to be reused with minimal changes.

The TPS feature vector should be semantically similar to the KMTF one:

```text
phi_rel
coord2_feature
eta1
eta2
quality_norm
eta_quality_norm
bx_norm
tf_layer_norm
depth_region_norm
stub_type_norm
has_coord2
stub_in_overlap
abs_eta
eta_dist_to_overlap
```

The exact feature names should be documented separately as:

```text
FEATURE_NAMES_TPS
N_FEATURES_TPS
```

If we want to reuse the same model code without changes, then KMTF and TPS should expose the same number of input features. But the documentation must clearly state that some features have source-dependent meaning.

---

## Step 4 — validate the TPS cache

Before training, run a TPS cache audit similar to the KMTF feature audit.

Checks:

```text
feature dimension is correct,
stub_in_overlap is binary,
abs_eta and eta_dist_to_overlap are sane,
truth_source encoding is valid,
meta_is_hard_neg is correct for G7/G8,
no target multiplicity exceeds the expected maximum,
G1–G6 have usable signal statistics,
G7/G8 behave as hard negatives,
B4 behaves as pure noise.
```

Only proceed to training if the TPS cache passes these sanity checks.

---

## Step 5 — train only the relevant models first

Do not repeat the full architecture sweep.

Start with:

```text
TPS EdgeCompat h64
TPS EdgeCompat h128
```

Use EdgeCompat because it is the selected model family from the KMTF study.

Initial training runs:

```text
edge_compat_h64_tps_B4
edge_compat_h128_tps_B4
```

Do not train DETR initially. DETR is only worth revisiting if TPS shows a specific multi-candidate weakness that EdgeCompat cannot solve.

---

## Step 6 — compare KMTF vs TPS vs OMTF

The first comparison table should be:

```text
Input view      | Model             | G1 eff | G2 eff | G5 rec | G6 rec | G7 FP | G8 FP | B4 FP
--------------- | ----------------- | ------ | ------ | ------ | ------ | ----- | ----- | -----
OMTF-internal   | existing model     | ...    | ...    | ...    | ...    | ...   | ...   | ...
GMT-KMTF        | EdgeCompat h128    | 85.7   | 84.5   | 69.5   | 80.8   | 6.4   | 9.8   | 0.0
GMT-TPS         | EdgeCompat h64     | ...    | ...    | ...    | ...    | ...   | ...   | ...
GMT-TPS         | EdgeCompat h128    | ...    | ...    | ...    | ...    | ...   | ...   | ...
```

This comparison should be described as an **input-view comparison**, not an object-level equivalence test.

Correct interpretation:

```text
The goal is not to prove that KMTF and TPS reconstruct the exact same graph.
The goal is to test which GMT-visible primitive source gives the best physics response for overlap-region muon reconstruction.
```

---

## Expected outcomes

### If TPS is better

Expected pattern:

```text
G7/G8 FP lower than KMTF,
B4 FP near zero,
G1/G2 efficiency comparable to KMTF,
G5/G6 recovery comparable or better.
```

Interpretation:

```text
TPS/EMTF-hybrid stubs are a cleaner GMT-visible input source for the overlap task than KMTF barrel stubs.
```

### If KMTF is better

Expected pattern:

```text
KMTF has higher G1/G2 efficiency,
KMTF has better G5/G6 recovery,
TPS has too few useful stubs or weaker overlap coverage.
```

Interpretation:

```text
KMTF provides more useful low-level overlap information despite its barrel-domain ambiguity.
```

### If they are complementary

Expected pattern:

```text
KMTF better signal efficiency,
TPS better hard-negative rejection,
combined information likely beneficial.
```

Interpretation:

```text
A future KMTF+TPS fusion model is justified.
```

---

## Possible final thesis/paper framing

The final study could be framed like this:

```text
We compare three input views for overlap-region muon reconstruction:

1. native OMTF internal primitives,
2. GMT-visible KMTF barrel stubs,
3. GMT-visible TPS/EMTF-hybrid stubs.

Because these inputs produce different graphs, the comparison is performed at the physics-response level rather than at node-level equivalence. The goal is to determine which trigger-primitive source available near the GMT interface carries sufficient information for robust overlap-region candidate reconstruction.
```

This would make the study stronger because it moves from:

```text
Can KMTF stubs be used?
```

to:

```text
Which GMT-visible stub source is best for the overlap task?
```

---

## Immediate next actions

* Write `scripts/omtf_gmt/audit_tps_inputs.py`.
* Check TPS occupancy on G1–G8 and B4.
* Implement TPS truth-transfer using globalized OMTF phi and `MuonStubTps_offlineCoord1`.
* Validate TPS target multiplicity.
* Build `cache_v2_tps` only if the audit is healthy.
* Train `EdgeCompat h64` on TPS as a quick baseline.
* Train `EdgeCompat h128` only if h64 is promising.
* Compare TPS against the selected KMTF baseline: `EdgeCompat h128 B3d epoch_050`.
