# OMTF ML-to-Firmware Dataset Decision Action Plan

**Date**: 2026-04-24

## Purpose

This plan defines the dataset analysis needed before deciding whether to:

1. continue with **EdgeCompatNet** as the main architecture,
2. try alternative models,
3. generate new samples,
4. simplify the architecture for firmware deployment.

The goal is not to choose a model by preference. The goal is to decide from evidence:

> Does the current dataset contain enough information for a fixed sparse edge model to learn candidate finding and pT/rank, or do we need another architecture, better losses, better features, or new samples?

---

# 1. Phase 0 — Dataset trustworthiness

Before comparing architectures, verify that the dataset and labels are reliable.

## 1.1 Branch integrity check

For every entry in `OMTFAllInputTree`, verify that all stub-level vectors have the same length:

```text
reg_stub_layer
reg_stub_phiHw
reg_stub_phiBHw
reg_stub_etaHw
reg_stub_r
reg_stub_quality
reg_stub_type
reg_stub_bx
reg_stub_trackId
reg_stub_ambiguous
```

Expected output:

| Dataset | Entries | Bad vector-length entries | Bad percentage |
| ------- | ------: | ------------------------: | -------------: |
| S1      |         |                           |                |
| S2      |         |                           |                |
| S3      |         |                           |                |
| S4      |         |                           |                |
| S5      |         |                           |                |
| B1      |         |                           |                |
| B2      |         |                           |                |
| B3      |         |                           |                |
| B4      |         |                           |                |

Decision rule:

```text
If bad percentage > 0:
    fix dataset reading / production before training more models.
else:
    continue.
```

---

## 1.2 NanoAOD join validation

Validate the join between `OMTFAllInputTree` and NanoAOD truth.

Check:

```text
reg_eventNum cast to uint32 matches NanoAOD Events.event
trackId = 1 maps to GenMuon index 0
trackId = 2 maps to GenMuon index 1
trackId = 3 maps to GenMuon index 2
```

Expected output:

| Dataset | OMTF entries | NanoAOD matched | Unmatched | Duplicated event IDs |
| ------- | -----------: | --------------: | --------: | -------------------: |
| S1      |              |                 |           |                      |
| S2      |              |                 |           |                      |
| S3      |              |                 |           |                      |
| S4      |              |                 |           |                      |
| S5      |              |                 |           |                      |
| B1      |              |                 |           |                      |
| B2      |              |                 |           |                      |
| B3      |              |                 |           |                      |
| B4      |              |                 |           |                      |

Manual spot-check for random entries:

```text
eventNum
processor
trackId values in stubs
GenMuon_pt
GenMuon_charge
GenMuon_dXY
```

Decision rule:

```text
If trackId-to-GenMuon mapping is not stable:
    do not train pT / charge / dXY heads yet.
    fix join or truth convention first.
else:
    continue.
```

---

# 2. Phase 1 — Occupancy and firmware bounds

This phase decides whether the fixed-input firmware assumption is realistic.

Current working assumption:

```text
Nmax = 24 stubs per processor window
K = 3 output candidates
```

## 2.1 Raw stub occupancy

For each processor-window entry, count the number of stubs before padding/truncation.

Expected output:

| Dataset | Mean | p50 | p90 | p95 | p99 | Max | % > 24 |
| ------- | ---: | --: | --: | --: | --: | --: | -----: |
| S1      |      |     |     |     |     |     |        |
| S2      |      |     |     |     |     |     |        |
| S3      |      |     |     |     |     |     |        |
| S4      |      |     |     |     |     |     |        |
| S5      |      |     |     |     |     |     |        |
| B1      |      |     |     |     |     |     |        |
| B2      |      |     |     |     |     |     |        |
| B3      |      |     |     |     |     |     |        |
| B4      |      |     |     |     |     |     |        |

Decision rule:

```text
If % > 24 is approximately zero:
    keep Nmax = 24.

If % > 24 is non-negligible:
    study whether truncation removes signal stubs.
    if truncation removes signal:
        increase Nmax or change truncation policy.
```

---

## 2.2 Signal, noise, and track multiplicity

For every processor window, compute:

```text
n_total_stubs
n_signal_stubs = count(trackId != 0)
n_noise_stubs  = count(trackId == 0)
n_tracks_present = number of unique nonzero trackId values
```

Expected output:

| Dataset | Mean total | Mean signal | Mean noise | Zero-signal % | 1-track % | 2-track % | 3-track % |
| ------- | ---------: | ----------: | ---------: | ------------: | --------: | --------: | --------: |
| S1      |            |             |            |               |           |           |           |
| S2      |            |             |            |               |           |           |           |
| S3      |            |             |            |               |           |           |           |
| S4      |            |             |            |               |           |           |           |
| S5      |            |             |            |               |           |           |           |
| B1      |            |             |            |               |           |           |           |
| B2      |            |             |            |               |           |           |           |
| B3      |            |             |            |               |           |           |           |
| B4      |            |             |            |               |           |           |           |

Decision rule:

```text
If most realistic windows have <= 1 true track:
    a simpler edge scorer + candidate builder may be enough.

If many realistic windows have 2 or 3 true tracks:
    keep EdgeCompatNet or SlotModel-v2 as serious candidates.
```

---

# 3. Phase 2 — Is the edge problem strong enough to justify EdgeCompatNet?

This phase decides whether relation-based learning is truly necessary or whether a simpler model could be enough.

## 3.1 Legal edge statistics

Define the candidate legal edge mask.

Start with:

```text
valid stubs only
different layers only
optional allowed layer-pair table
optional allowed detector-type pair table
```

For each window, compute:

```text
number of legal edges
number of same-track legal edges
number of different-track legal edges
same-track edge fraction
windows with no positive edge
```

Expected output:

| Dataset | Mean legal edges | Mean positive edges | Positive edge % | Windows with no positive edge |
| ------- | ---------------: | ------------------: | --------------: | ----------------------------: |
| S1      |                  |                     |                 |                               |
| S2      |                  |                     |                 |                               |
| S3      |                  |                     |                 |                               |
| S4      |                  |                     |                 |                               |
| S5      |                  |                     |                 |                               |
| B1      |                  |                     |                 |                               |
| B2      |                  |                     |                 |                               |
| B3      |                  |                     |                 |                               |
| B4      |                  |                     |                 |                               |

Decision rule:

```text
If positive edge fraction is extremely tiny:
    edge loss needs class weighting or focal loss.

If many signal windows have no positive legal edges:
    legal edge mask is too restrictive.

If positive edges are common and clean:
    EdgeCompatNet is well motivated.
```

---

## 3.2 Edge feature separability

Compare same-track and different-track legal pairs using the planned edge features.

Analyze:

```text
delta_phi
abs_delta_phi
delta_r
delta_r_sq
kappa_hat
abs_delta_eta
phiB_i
phiB_j
phiB consistency
layer-pair category
type-pair category
```

Useful plots:

```text
same-track vs different-track kappa_hat
same-track vs different-track abs_delta_eta
same-track vs different-track abs_delta_phi
same-track vs different-track phiB residual
same-track vs different-track layer-pair frequencies
```

Expected summary:

| Feature               | Separates same/different? | Issue         |
| --------------------- | ------------------------- | ------------- |
| `delta_phi`           | strong / medium / weak    |               |
| `abs_delta_phi`       | strong / medium / weak    |               |
| `kappa_hat`           | strong / medium / weak    | clipping?     |
| `abs_delta_eta`       | strong / medium / weak    |               |
| `phiB residual`       | strong / medium / weak    | maybe missing |
| `layer-pair category` | strong / medium / weak    |               |

Decision rule:

```text
If pair features strongly separate same-track edges:
    continue EdgeCompatNet.

If pair features are weak:
    try HECIN-Lite or add better physics features.

If one or two features dominate strongly:
    consider a simpler deterministic/ML hybrid.
```

---

# 4. Phase 3 — Is pT/rank learnable from the current inputs?

This is the most important phase.

Current observation:

```text
The pT regression head predicts almost constant values.
```

Before changing architecture or generating data, prove whether pT information is actually visible in the input features.

---

## 4.1 pT distribution per dataset

For every true muon associated with signal stubs, analyze:

```text
GenMuon_pt
log(GenMuon_pt)
1 / GenMuon_pt
charge / GenMuon_pt
```

Expected output:

| Dataset | pT mean | pT p10 | pT p50 | pT p90 | pT p99 | % > 3 | % > 5 | % > 10 | % > 20 | % > 22 |
| ------- | ------: | -----: | -----: | -----: | -----: | ----: | ----: | -----: | -----: | -----: |
| S1      |         |        |        |        |        |       |       |        |        |        |
| S2      |         |        |        |        |        |       |       |        |        |        |
| S3      |         |        |        |        |        |       |       |        |        |        |
| S4      |         |        |        |        |        |       |       |        |        |        |
| S5      |         |        |        |        |        |       |       |        |        |        |
| B1      |         |        |        |        |        |       |       |        |        |        |
| B2      |         |        |        |        |        |       |       |        |        |        |
| B3      |         |        |        |        |        |       |       |        |        |        |

Decision rule:

```text
If pT distribution is heavily imbalanced:
    use pT reweighting or generate pT-flat samples.

If threshold regions have enough statistics:
    do not generate new samples yet.
    fix target formulation and loss weighting first.
```

---

## 4.2 pT versus curvature proxies

For signal stub pairs, compute curvature-like quantities:

```text
abs(delta_phi)
signed delta_phi
kappa_hat
charge * kappa_hat
phiB
phiB residual
layer-pair-specific kappa_hat
```

Compare these against:

```text
GenMuon_pt
1 / GenMuon_pt
charge / GenMuon_pt
```

Required plots:

```text
kappa_hat vs 1/pT
kappa_hat vs charge/pT
delta_phi vs 1/pT
signed delta_phi vs charge/pT
phiB vs 1/pT
best pair kappa_hat vs 1/pT
median pair kappa_hat vs 1/pT
layer-pair-specific kappa_hat vs charge/pT
```

Expected correlation table:

| Feature             | Target | Pearson | Spearman |
| ------------------- | ------ | ------: | -------: |
| `kappa_hat`         | `1/pT` |         |          |
| `signed kappa_hat`  | `q/pT` |         |          |
| `phiB`              | `q/pT` |         |          |
| `best_pair_kappa`   | `q/pT` |         |          |
| `median_pair_kappa` | `q/pT` |         |          |
| `layer_pair_kappa`  | `q/pT` |         |          |

Decision rule:

```text
If pT / curvature correlation is visible:
    do not generate samples yet.
    the pT failure is probably loss/head/training related.

If there is almost no correlation:
    current features are insufficient, incorrectly normalized, or the geometry target is misaligned.
    inspect feature construction and consider new engineered features.
```

---

## 4.3 Tiny classical pT baseline

Before using a large neural model, test whether simple models can learn pT/rank from derived features.

Train small baselines on per-candidate or per-window summary features.

Possible input features:

```text
best absolute kappa_hat
mean kappa_hat
median kappa_hat
signed kappa_hat summary
phiB summary
phiB residual summary
layer count
fired layer pattern
eta summary
r summary
quality summary
number of signal-like edges
```

Targets:

```text
log(pT)
q/pT
pT threshold labels:
    pT > 3
    pT > 5
    pT > 10
    pT > 15
    pT > 20
    pT > 22
```

Baselines:

```text
linear regression on q/pT
small MLP on pair summary features
small classifier for pT threshold labels
```

Decision rule:

```text
If a tiny baseline predicts pT thresholds:
    the information is in the data.
    EdgeCompat pT failure is a training/loss/head problem.

If even simple baselines fail:
    either features are insufficient, labels are wrong, or new pT-controlled samples are needed.
```

---

# 5. Phase 4 — Decide whether new pT samples are needed

Do not generate new samples blindly.

Only generate new pT samples if Phase 3 shows that the existing pT distribution is insufficient or badly imbalanced.

## 5.1 Threshold-bin statistics

Count true muons near the relevant L1 thresholds:

```text
2–4 GeV
4–6 GeV
8–12 GeV
13–17 GeV
18–24 GeV
>24 GeV
```

Expected output:

| Dataset | 2–4 | 4–6 | 8–12 | 13–17 | 18–24 | >24 |
| ------- | --: | --: | ---: | ----: | ----: | --: |
| S1      |     |     |      |       |       |     |
| S2      |     |     |      |       |       |     |
| S3      |     |     |      |       |       |     |
| S4      |     |     |      |       |       |     |
| S5      |     |     |      |       |       |     |
| B1      |     |     |      |       |       |     |
| B2      |     |     |      |       |       |     |
| B3      |     |     |      |       |       |     |

Decision rule:

```text
If threshold regions are underpopulated:
    generate pT-flat or threshold-enriched samples.

If threshold regions are populated:
    use reweighting first.
```

Recommended targeted samples if needed:

```text
S1_ptflat
S2_ptflat
B1_ptflat
B2_ptflat
B3_ptflat
```

Do not generate more generic S1/B1/B4 unless a specific analysis shows they are needed.

---

# 6. Phase 5 — B4 and false-positive analysis

This phase decides whether B4 fake rate requires new hard-negative samples or only better threshold calibration.

## 6.1 Accepted B4 window audit

Take all B4 windows where the model fires.

Compare accepted B4 windows against rejected B4 windows.

Analyze:

```text
stub count
layer pattern
phi spread
eta spread
phiB pattern
quality pattern
max edge score
mean edge score
candidate score
number of high-score edges
predicted pT/rank distribution
```

Expected output:

| Quantity               | Rejected B4 | Accepted B4 |
| ---------------------- | ----------: | ----------: |
| Mean stubs             |             |             |
| Mean layers            |             |             |
| Mean max edge score    |             |             |
| Mean candidate score   |             |             |
| Mean predicted pT/rank |             |             |
| Common layer pattern   |             |             |

Decision rule:

```text
If accepted B4 windows are rare high-occupancy track-like noise:
    mine hard negatives from existing B4.
    if not enough, generate hard B4 samples.

If accepted B4 windows look ordinary:
    candidate threshold/loss calibration is the issue.
```

---

## 6.2 B4 fake multiplicity

Do not only measure whether a B4 window is accepted.

Also measure:

```text
number of accepted candidates per processor window
number of accepted candidates per event
candidate score distribution
predicted pT/rank distribution
```

Expected output:

| Dataset | Accept/window | Candidates/window | Accept/event | Candidates/event |
| ------- | ------------: | ----------------: | -----------: | ---------------: |
| B4      |               |                   |              |                  |

Decision rule:

```text
If B4 produces many candidates per event:
    need stronger candidate suppression or candidate-level negative loss.

If B4 produces rare single candidates:
    threshold/rate tuning may be enough.
```

---

# 7. Phase 6 — Does SlotModel deserve another chance?

This phase decides whether the SlotModel failed because of the architecture or because of unfair fixed-index supervision.

## 7.1 TrackId ordering stability

For multi-track windows, check whether `trackId` ordering has a meaningful physical interpretation.

Questions:

```text
Is trackId 1 usually the highest-pT muon?
Is trackId 1 usually the first generated muon?
Is trackId order correlated with eta or phi?
Is trackId order random-looking?
```

Expected output:

| Dataset | TrackId order correlated with pT? | Correlated with eta/phi? | Random-looking? |
| ------- | --------------------------------- | ------------------------ | --------------- |
| S3      |                                   |                          |                 |
| S4      |                                   |                          |                 |
| S5      |                                   |                          |                 |
| B3      |                                   |                          |                 |

Decision rule:

```text
If trackId ordering is arbitrary:
    the old SlotModel supervision is unfair.
    try SlotModel-v2 with permutation-invariant matching.

If trackId ordering is stable and meaningful:
    SlotModel weakness is more likely architectural.
```

---

## 7.2 Multi-track confusion analysis

For S3, S4, S5, and B3, inspect:

```text
same-track edge purity
track separation in phi
track separation in eta
shared or ambiguous stubs
number of stubs per track
layer conflicts between tracks
```

Expected output:

| Dataset | Mean Δphi between tracks | Close-track % | Ambiguous stub % | Shared-layer conflict % |
| ------- | -----------------------: | ------------: | ---------------: | ----------------------: |
| S3      |                          |               |                  |                         |
| S4      |                          |               |                  |                         |
| S5      |                          |               |                  |                         |
| B3      |                          |               |                  |                         |

Decision rule:

```text
If close-track cases are common:
    keep relation/edge model as main path.

If close-track cases are rare:
    simpler candidate builder may be enough.
```

---

# 8. Phase 7 — dXY analysis as a separate branch

The prompt pT/rank path should not be blocked by dXY.

Analyze dXY only if a displaced-muon trigger path is a serious target.

## 8.1 dXY visibility analysis

For S2, S5, and B2, analyze:

```text
GenMuon_dXY distribution
dXY vs phiB
dXY vs phiB residual
dXY vs kappa_hat residual
dXY vs layer pattern
dXY vs eta
dXY vs number of stubs
```

Expected output:

| Feature           | Correlation with dXY |
| ----------------- | -------------------: |
| `phiB`            |                      |
| `phiB residual`   |                      |
| `kappa residual`  |                      |
| `layer pattern`   |                      |
| `eta`             |                      |
| `number of stubs` |                      |

Decision rule:

```text
If dXY is weakly visible:
    redesign displacement-sensitive features.

If dXY is visible but model fails:
    use separate dXY loss/head/training.
```

Recommended handling:

```text
Stage 7A: prompt pT/rank trigger path
Stage 7B: displaced/dXY-specialized path
```

---

# 9. Final decision matrix

Use the results above to decide the next architecture and data-production steps.

| Evidence                                               | Decision                                                           |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| Edge features clearly separate same-track pairs        | Keep EdgeCompatNet mainline                                        |
| Simpler relation baseline matches EdgeCompatNet        | Prefer simpler RelationNet / EdgeMLP++ for HLS                     |
| Deterministic builder is close to EdgeCompatNet        | Use edge scorer + deterministic builder as deployment architecture |
| pT visible in curvature features                       | Fix loss/head; do not generate samples yet                         |
| pT not visible but labels are correct                  | Add better features or generate pT-controlled samples              |
| pT bins around thresholds are underpopulated           | Generate pT-flat / threshold-enriched samples                      |
| B4 false positives are rare hard topologies            | Mine or generate hard B4 negatives                                 |
| SlotModel failed because trackId ordering is arbitrary | Try SlotModel-v2 with permutation matching                         |
| dXY not visible in current features                    | Do not pursue dXY until feature redesign                           |

---

# 10. Recommended execution order

Run the analysis in this order:

```text
1. Dataset integrity and NanoAOD join validation
2. Full occupancy table for all 9 datasets
3. Signal/noise/track multiplicity table
4. Legal edge statistics and edge class balance
5. Edge feature separability plots
6. pT distribution and threshold-bin statistics
7. pT versus curvature-proxy analysis
8. Tiny classical pT baseline
9. B4 false-positive audit
10. TrackId ordering / SlotModel fairness audit
11. Decide:
      keep EdgeCompatNet?
      try simpler relation model?
      try deterministic builder?
      generate targeted samples?
```

---

# 11. Expected likely outcome

The most likely outcome is:

1. **EdgeCompatNet remains the correct mainline.**
2. The pT problem is probably not lack of model capacity.
3. The pT issue is likely one of:

   * wrong loss balance,
   * wrong target formulation,
   * not using `q/pT` or ordinal threshold labels,
   * poor pT sampling around thresholds,
   * missing layer-pair-specific curvature features.
4. Generic new samples are probably unnecessary.
5. Targeted samples may be useful if pT threshold bins are underpopulated.
6. A simpler relation model and an edge-scorer-plus-deterministic-builder variant should be tested because either may be easier to deploy in firmware than the full current EdgeCompatNet.

---

# 12. Concrete next milestone

The next milestone should be:

> **Dataset Decision Report v1**

It should contain:

```text
- dataset integrity table
- NanoAOD join validation table
- occupancy and Nmax validation
- signal/noise/track multiplicity table
- legal edge statistics
- edge feature separability plots
- pT distribution and threshold-bin analysis
- pT-curvature correlation plots
- tiny pT baseline result
- B4 false-positive audit
- SlotModel trackId-ordering audit
- final recommendation:
      keep EdgeCompatNet?
      add simpler relation baseline?
      add deterministic candidate builder?
      generate new samples?
```

Only after this report should the project decide whether to:

```text
1. continue directly to EdgeCompatNet-pT-v1,
2. train challenger models,
3. generate targeted new samples,
4. redesign features for pT or dXY.
```
