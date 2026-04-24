# GMT-Visible-Stub Model with Offline Truth Transfer

## Scope

This document defines a **parallel development track** to the main OMTF-internal model line.

The purpose of this track is to study a regional trigger model whose **inference inputs are only GMT-visible stubs**, while using richer internal information **only during offline dataset construction** to attach truth labels.

This branch is especially valuable when barrel RPC information is unavailable, unreliable, or intentionally excluded from the model design.

It should be developed **in parallel** to the main OMTF-like GNN line, not as a replacement for it at the start.

---

# 1. Project role

This branch should be treated as:

* a **parallel R&D line**
* a **reduced-RPC / RPC-robustness line**
* a **regional GMT-visible-stub model study**, not yet a full global GMT replacement effort

It should not replace the main OMTF-internal line as the primary first deployment target.

## Recommended project split

### Branch A — OMTF-internal model

Goal:

* strongest overlap-region reconstruction
* full OMTF stub information
* best supervision quality
* primary high-information reference and primary deployment-oriented path

### Branch B — GMT-visible-stub model with offline truth transfer

Goal:

* investigate a trigger model that uses only GMT-visible stubs at inference
* study performance when barrel RPC information is absent or intentionally not used
* evaluate an alternative detector-facing representation on the same events and same regional trigger problem

---

# 2. Correct conceptual boundary

This branch must be described precisely.

It is **not** “NanoAOD-only” in the dataset-production sense.

It is instead:

> a **GMT-visible-stub inference model** with **offline truth transfer** during dataset generation.

## Training time

The dataset generator may use:

* `omtf_hits_*.root`
* `omtf_nano_*.root`
* OMTF internal truth carriers such as `reg_stub_trackId` and `reg_stub_ambiguous`
* NanoAOD `GenMuon_*` truth

## Inference time

The trained model must consume only:

* GMT-visible stubs (KMTF barrel and, later, TPS endcap stubs)
* no OMTF internal state
* no OMTF internal truth fields

This is the correct setup.

---

# 3. Why this branch is worth doing

The original GMT/NanoAOD-only idea looked weaker than the OMTF-internal line because it loses information, especially barrel RPC layers.

However, if barrel RPC information is currently leaking or operationally problematic, then the scientific question changes from:

> “What is the strongest possible overlap-region representation?”

to:

> “What trigger representation still works well when barrel RPC information is absent or unusable?”

Under that question, this branch becomes strategically valuable.

## Practical motivations

* reduced dependence on problematic barrel RPC input
* regional trigger reconstruction using only GMT-visible primitives at inference
* possible fallback or robustness strategy under detector degradation
* direct comparison of two input representations built from the same EOS production campaign

---

# 4. First-scope recommendation

This branch should start as a **barrel-first, region-based study**.

Do **not** begin with a whole-event all-phi graph.

## Recommended initial scope

* use the same local trigger-style regioning as the OMTF comparison line
* use KMTF barrel stubs as the primary barrel input
* add TPS endcap stubs only after the barrel dataset path is validated
* validate truth transfer before any model training

This keeps the problem:

* local
* trigger-shaped
* directly comparable with the OMTF-internal line

---

# 5. Data-source policy

## Barrel source

Use:

* `MuonStubKmtf`

### Why

* it preserves the best available barrel DT representation in the GMT-visible branch
* its reliable phi quantity is `MuonStubKmtf_offlineCoord1`
* its reliable bend quantity is `MuonStubKmtf_offlineCoord2`

### Important warning

Do **not** use `MuonStubKmtf_coord1` as the main phi feature or as a cross-collection matching coordinate.

Reason:

* the producer uses an internal 18-bit barrel phi representation
* NanoAOD stores `coord1` as `Short_t`
* the original internal integer cannot be reconstructed reliably from the stored NanoAOD value
* `offlineCoord1` is the reliable phi quantity for ML and matching

## Endcap source (later stage)

Use:

* `MuonStubTps` with `isEndcap == True`

### Important note

For TPS endcap stubs, `offlineCoord1` is also the reliable phi quantity. Integer `coord1` exists on a different scale from KMTF and must not be mixed with KMTF integer `coord1` values.

## Do not use initially

* TPS barrel hybrid stubs as the main barrel representation

Reason:

* they appear deliberately compressed for downstream interface/hardware convenience
* they are not the best starting barrel representation for this branch

---

# 6. Region policy: eta coverage vs phi coverage

The eta side of the problem is naturally constrained by the detector acceptance and by the specific regional trigger question.

The harder design choice is phi.

## Recommendation

The first version should be **regional in phi**, not global over all phi.

That means:

* do **not** build one all-phi graph per event as the first approach
* do **not** train a single model over all stubs in the full 2π event
* instead, keep the same local trigger-style regioning used for the OMTF comparison

## Correct distinction

* **dataset coverage** should span all phi across all regional samples
* each **training/inference sample** should still cover only one local phi region

## Recommended first choice

Use the same region unit as the OMTF-internal comparison line:

* `reg_iProcessor` from `OMTFAllInputTree` is the authoritative region ID
* for each `(event, reg_iProcessor)` instance, collect the corresponding GMT-visible stubs from NanoAOD in that same local region

This ensures:

* same events
* same regional decision problem
* same comparison basis between the two branches

### Important OMTF processor note

In the current production, `reg_iProcessor` for OMTF should be treated as **0–2** for the three 120° sectors per eta side. This is the processor index that is consistent with `phiZero(proc)` in the OMTF local-to-global phi conversion.

---

# 7. Coordinate policy for dataset construction

This section defines which coordinates are reliable and how they should be used.

## OMTF internal phi

For OMTF internal stubs:

* `reg_stub_phiHw` is on the OMTF 5400-bin scale
* it is stored in the **processor-local frame**
* to obtain global phi, add `phiZero(proc)` before converting to radians

This is relevant mainly for truth transfer and cross-checks, not for the GMT-visible model input.

## GMT-visible phi

For GMT-visible stubs, always prefer the float offline coordinates:

* `MuonStubKmtf_offlineCoord1`
* `MuonStubTps_offlineCoord1`

These are the correct phi quantities for:

* ML feature construction
* cross-collection geometric comparisons
* OMTF↔KMTF matching checks

## Phi normalization for ML

Use processor- or region-centered phi:

* `phi_rel = angle_diff(offlineCoord1, phi_region_center)`

This enforces local phi-translation invariance.

## Coord2 policy

Use `offlineCoord2`, but do **not** assume identical semantics across all sources.

* for KMTF barrel stubs, `offlineCoord2` behaves as local DT bend information
* for TPS endcap stubs, `offlineCoord2` behaves as a second coordinate with different semantics

Therefore, the dataset must carry an explicit source/type semantic indicator.

## Eta policy

Use:

* `offlineEta1`
* optionally `offlineEta2`, but only with an explicit presence mask

## Fields to avoid as primary ML geometry features

Do not use the following as the main geometry representation:

* `MuonStubKmtf_coord1`
* `MuonStubTps_coord1` as a shared phi feature with KMTF integer coordinates
* `phiRegion` as the primary phi feature
* `etaRegion` as the primary eta feature

These may still be kept for QA, debugging, or matching constraints.

---

# 8. Training dataset generation strategy

A dedicated generator is required.

The right training unit is:

> one `(event, region)` object containing only the GMT-visible stubs in that region, together with offline-attached truth labels.

This should not be trained directly from raw NanoAOD tables without preprocessing.

## Proposed module

* `src/omtf_gmt/make_gmt_graphs.py`

## Inputs per sample/job pair

* `omtf_hits_<ID>_<N>.root`
* `omtf_nano_<ID>_<N>.root`

## Output

One processed object per:

* `(event, region)`

## Initial scope

Only barrel KMTF stubs.

Do **not** add TPS endcap in the first implementation.

---

# 9. Truth-transfer policy

This is the critical dataset-construction step.

## Barrel truth

For barrel KMTF stubs, transfer truth from `OMTFAllInputTree`.

Available truth carriers:

* `reg_stub_trackId`
* `reg_stub_ambiguous`

## Endcap truth

For TPS endcap stubs, strong OMTF-based truth is not available outside OMTF coverage.

The first endcap version should therefore use:

* propagated `GenMuon_*` coordinates
* spatial matching rules
* explicit lower-confidence labeling

## Required truth-source tracking

At minimum, each stub should carry:

* `truth_source = {omtf_transfer, gen_spatial_match}`
* optional `truth_confidence`

This allows:

* clean ablations
* training masks
* honest interpretation of results

---

# 10. Barrel truth-transfer matching policy

The barrel OMTF↔KMTF match must not rely on phi alone.

## Do not use only nearest-phi matching

In busy events, nearest phi alone may be fragile.

## Recommended composite matching key

Use a composite match using as many as possible of:

* event number
* compatible region / processor
* station / `depthRegion`
* sector / `phiRegion`
* wheel / `etaRegion`
* BX
* nearest `offlineCoord1` as final disambiguator

`offlineCoord1` should be the final geometric tie-breaker, not the only identity anchor.

## New module

* `src/omtf_gmt/truth_transfer.py`

---

# 11. Hard validation gate before training

No model training should start before the truth-transfer validation looks good.

## New validation module

* `src/omtf_gmt/validate_truth_transfer.py`

## Required checks

* match rate
* unmatched fraction
* duplicate-match fraction
* collision / ambiguous-match fraction
* transferred `track_id` consistency
* transferred `is_ambiguous` consistency
* dataset dependence on representative samples:

  * `S1`
  * `S3`
  * `B1`
  * `B3`
  * `B4`

## Required artifact

* `docs/omtf_gmt/TRUTH_TRANSFER_SIGNOFF.md`

### Important scope note

The first signoff should explicitly state that it validates the **barrel KMTF truth-transfer path only**.

---

# 12. Node feature policy for the first barrel model

Freeze the node feature schema only after truth transfer is validated.

## Continuous features

* `phi_rel`
* `offlineCoord2`
* `offlineEta1`
* `offlineEta2_or_zero`
* `quality_norm`
* `etaQuality_norm`
* `bx_norm`

## Mask features

* `has_eta2`
* `has_coord2`

## Categorical features

* `tfLayer`
* `depthRegion`
* `source_kind`

## First-stage source kind

Initially only:

* `kmtf_barrel_dt`

### Important encoding rules

* `tfLayer` must be categorical, not ordinal
* `depthRegion` should be treated as categorical or at least evaluated as categorical
* `phi_rel` must be computed from the actual region center used in extraction

## New module

* `src/omtf_gmt/features.py`

---

# 13. Edge-construction policy for this branch

This branch should start with a conservative software edge policy.

## Recommended first policy

* cross-layer only
* region-local stubs only
* no aggressive hardware sparsification in the first software dataset version

This keeps the branch comparable to the OMTF-internal software baseline.

Hardware-oriented sparsity for this branch should come later.

---

# 14. Dataset export

After regioning, truth transfer, and feature building are working, export the first processed GMT dataset.

## Output format

One serialized sample per `(event, region)`.

This may be:

* PyG `Data` objects
* or an intermediate tensor format first, if that is easier for debugging

## Minimum required contents

* node features
* truth labels
* masks
* region metadata
* optional derived pair features

## Output location

For example:

* `build/datasets/omtf_gmt_barrel/`

## Required metadata

Each sample should carry:

* `event`
* `region_id`
* `sample_id`
* `truth_source`
* `n_true_tracks`

---

# 15. Sample suite policy

The GMT-visible-stub line must use the same sample families as the OMTF-internal line:

* `S1`
* `S2`
* `S3`
* `S4`
* `S5`
* `B1`
* `B2`
* `B3`
* `B4`

## Why this matters

This makes the comparison scientifically meaningful:

* same physics
* same occupancy conditions
* same region concept
* different input representation

That is the main value of this branch.

---

# 16. Model families to test

The GMT-visible-stub line should test the same broad model families as the OMTF-internal branch, but with GMT-aware feature encoders and GMT-specific source semantics.

## Recommended first order

1. DeepSets / pooled baseline
2. Edge-compatibility model
3. Fixed-K slot model
4. Edge scorer + deterministic builder
5. Rich software teacher (HECIN-like)

## Why this order

* the baseline validates the dataset and features quickly
* the edge-compatibility model is still the most natural relation-based formulation
* the slot model remains attractive because the output is still bounded and trigger-like
* the richer teacher should come only after the simpler branches are stable

## Important note

Do not simply reuse the OMTF-internal model implementation unchanged. The GMT branch requires source-aware encoders and careful treatment of heterogeneous stub semantics.

---

# 17. Evaluation policy

To compare Branch A and Branch B fairly, use the same evaluation philosophy and, where possible, the same metric definitions.

## Required metrics

* candidate recovery efficiency
* efficiency vs pT
* efficiency vs d0
* close-dimuon separation
* fake candidate multiplicity
* B4 zero-track false positives
* PU robustness

## Important rule

Do not silently change metric definitions between the two branches.

---

# 18. First comparison milestone

Once both branches have at least one stable baseline model, run the first direct comparison.

## Compare

* Branch A: OMTF-internal representation
* Branch B: GMT-visible KMTF barrel representation

## Under identical conditions

* same samples
* same regions
* same model family where possible
* same metrics

## Main scientific question answered here

How much performance is lost or retained when switching from the OMTF-internal regional stub view to the GMT-visible regional stub view?

This is the central value of the branch.

---

# 19. Only then extend to TPS endcap

Do not start with a mixed KMTF+TPS branch.

After the barrel-first path is validated and compared:

## Add

* TPS endcap stubs in the same region
* weaker spatial truth assignment from `GenMuon_*`
* explicit `truth_source = gen_spatial_match`
* explicit label-confidence tracking

## Required caution

This remains a lower-confidence regime and must initially be treated as an extension, not as supervision of equal quality to the barrel truth-transfer path.

---

# 20. Later hardware-oriented studies

Only after the GMT barrel-first branch works in software should this line move toward hardware-oriented simplifications.

## Later studies

* fixed `Nmax`
* per-layer or per-source caps
* edge sparsification
* KMTF-vs-TPS-barrel ablation
* quantization
* HLS-friendly kernels

The branch should not start from these compressions.

---

# 21. Implementation sequence summary

### Step 1

Define the branch as a **parallel representation study**.

### Step 2

Freeze the source policy:

* barrel = `MuonStubKmtf`
* endcap later = `MuonStubTps isEndcap=True`

### Step 3

Use the **same EOS samples** as the OMTF line.

### Step 4

Use the **same regional problem definition** as the OMTF line.

### Step 5

Build a **barrel-first GMT dataset** using KMTF stubs.

### Step 6

Transfer truth offline from `OMTFAllInputTree`.

### Step 7

Validate truth transfer before training.

### Step 8

Freeze the barrel feature schema.

### Step 9

Train the same broad model families, with GMT-aware encoders.

### Step 10

Compare Branch A vs Branch B on the same metrics.

### Step 11

Only later add TPS endcap and hardware-oriented simplifications.

---

# 22. Final recommendation

Approve this branch as a **parallel development line**, with the following constraints:

1. start barrel-first
2. use `MuonStubKmtf_offlineCoord1` as the reliable barrel phi quantity
3. do not use `MuonStubKmtf_coord1` as the main phi feature or cross-collection matching coordinate
4. validate OMTF↔KMTF truth transfer before model training
5. keep region extraction tied to the same local trigger geometry used for comparison with the OMTF-internal branch
6. treat TPS/endcap truth as a weaker-label extension until proven otherwise
7. use the same sample suite and the same evaluation metrics as the OMTF-internal line
8. explicitly motivate this branch as an RPC-reduced / RPC-robustness study using the same EOS production but a different data view

This makes the branch scientifically useful, practically feasible with current EOS data, and cleanly comparable to the main OMTF-internal path.
