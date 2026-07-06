# AI Instructions: Displaced-Muon Variable Analysis for an OMTF-Parallel ML Line

## Mission

Build a variable-level analysis to identify which trigger-primitive and derived features are most useful for detecting displaced muons in the CMS OMTF overlap region.

The final goal is not yet to train the full GNN. The immediate goal is to answer:

> Which variables separate displaced muons from prompt low-pT muons, while remaining decorrelated from pT, eta, ordinary curvature, pileup, and OMTF quality?

This analysis will decide which node features and edge features should later be used in a GNN/ML line running in parallel to the standard OMTF algorithm.

You have access to:

- the muon-trigger variable audit;
- the dataset summary;
- ROOT outputs of the form `omtf_hits_<DATASET>_<PROCID>.root`;
- ROOT outputs of the form `omtf_nano_<DATASET>_<PROCID>.root`.

Use the audit as the source of truth for which variables are physically meaningful and which ones are firmware-realistic.

Do not treat this as a generic ML classification task. Treat it as a controlled trigger-physics study.

---

## Core Physics Question

A displaced muon can look similar to a low-pT prompt muon because both may have:

- large bending;
- poor pointing to the beamline;
- unusual station-to-station consistency;
- degraded trigger quality;
- low reconstructed OMTF pT.

Therefore, the central comparison is not simply:

```text
prompt vs displaced
```

The central comparison is:

```text
displaced muons vs prompt low-pT muons
```

in the same pT bin, eta region, detector acceptance, and pileup condition.

The useful variables are those that still separate the two after this matching.

---

## Dataset Strategy

### Use the C-series first

The C-series is the primary dataset campaign for the variable-ranking study because it is:

- pT-binned;
- overlap-filtered;
- available with and without PU200;
- separated into prompt, displaced, and mild-displaced samples.

Start with the C-series before using S/B or G samples.

### First no-PU comparisons

Use these to discover clean displacement-sensitive variables:

| Prompt sample | Displaced sample | pT range | Pileup | Purpose |
|---|---|---:|---|---|
| `C1` | `C7` | 2-5 GeV | no PU | hardest low-pT comparison |
| `C2` | `C8` | 5-10 GeV | no PU | low-pT comparison |
| `C3` | `C9` | 10-20 GeV | no PU | medium pT |
| `C4` | `C10` | 20-50 GeV | no PU | medium-high pT |
| `C5` | `C11` | 50-100 GeV | no PU | high pT |
| `C6` | `C12` | 100-200 GeV | no PU | very high pT |

The most important first plots are:

```text
C1 vs C7
C2 vs C8
```

because low-pT prompt muons are the most dangerous background for a displaced trigger.

### PU200 comparisons

After no-PU studies, repeat the same analysis under pileup:

| Prompt sample | Displaced sample | pT range | Pileup |
|---|---|---:|---|
| `C13` | `C15` | 2-5 GeV | PU200 |
| `C14` | `C16` | 5-10 GeV | PU200 |
| `C17` | `C21` | 10-20 GeV | PU200 |
| `C18` | `C22` | 20-50 GeV | PU200 |
| `C19` | `C23` | 50-100 GeV | PU200 |
| `C20` | `C24` | 100-200 GeV | PU200 |

The important question is whether variables that work in no-PU still work in PU200.

### Mild-displacement ambiguity samples

Use these after the prompt-vs-displaced separation is understood:

| Sample | Meaning |
|---|---|
| `C25` | mild-displaced, 2-5 GeV, no PU |
| `C26` | mild-displaced, 5-10 GeV, no PU |
| `C27` | mild-displaced, 2-5 GeV, PU200 |
| `C28` | mild-displaced, 5-10 GeV, PU200 |

These samples occupy the ambiguity region:

```text
0.05 cm < |d0| < 0.2 cm
```

Use them to test whether the selected variables create an overly aggressive displaced tag.

### G-series and B4 are for later GNN training and final validation

Do not use the G-series first for basic variable ranking.

Use them later for realistic model training and stress tests:

| Dataset | Role |
|---|---|
| `G2` | prompt overlap + PU200 |
| `G4` | displaced overlap + PU200 |
| `G5` | two displaced muons + PU200 |
| `G6` | three prompt muons + PU200 |
| `G8` | barrel hard negative + PU200 |
| `G10_pos`, `G10_neg` | endcap hard negatives + PU200 |
| `B4` | pure PU200 noise |

Use `B4` as a pure no-candidate noise sample. Do not replace it with G hard-negative samples. G hard negatives contain real out-of-domain muons; B4 contains pileup-only occupancy.

---

## Labels

Define labels from generator truth only.

### Binary label

Use:

```text
label = 0 for prompt
label = 1 for displaced
```

Prompt:

```text
|gen_dxy| < 0.05 cm
```

Displaced:

```text
|gen_dxy| > 0.2 cm
```

Mild-displaced ambiguity:

```text
0.05 cm < |gen_dxy| < 0.2 cm
```

Keep mild-displaced samples out of the first binary training. Use them as a separate validation category.

### Do not use truth as input

The following truth variables may be used for labels, binning, matching, or validation:

```text
gen_pt
gen_eta
gen_phi
gen_dxy
gen_lxy
gen_vx
gen_vy
gen_vz
```

They must not be used as input features.

---

## First Analysis Product

Create a flat analysis table, preferably one row per generated muon or one row per matched trigger candidate.

The first table should be simple and robust. Use a format that can be read easily by Python:

```text
Parquet, HDF5, ROOT RNTuple, or flat ROOT TTree
```

Recommended output:

```text
analysis_tables/variable_study_<DATASET>.parquet
```

Each row should contain:

```text
dataset
event
muon_index
label
category
has_pu
pt_bin

gen_pt
gen_eta
gen_phi
gen_dxy
gen_lxy

omtf_hwPt
omtf_hwPtUnconstrained
omtf_hwQual
omtf_hwEta
omtf_hwPhi

n_dt
n_csc
n_rpc
n_tps_stubs
n_omtf_hits

feature columns...
```

If matching to an OMTF candidate is ambiguous, store both:

```text
candidate_matched = true/false
candidate_delta_phi
candidate_delta_eta
```

Do not silently drop unmatched generated muons. They are important for efficiency studies.

---

## Required Feature Families

Build both raw summary features and derived physics features.

### 1. DT phi-bending features

From DT phi primitives:

```text
phi
phiB or phiBend
quality
station
wheel
sector
bx
```

Create:

```text
dt_phiB_mean
dt_phiB_std
dt_phiB_abs_mean
dt_phiB_abs_max
dt_phiB_sign_changes
dt_phiB_station_slope
dt_phiB_residual_mean
dt_phiB_residual_std
dt_phiB_residual_abs_max
```

The raw `phiB` may correlate with pT. The residual form is more important.

Approximate residual:

```text
dt_phiB_residual = measured_local_bend - expected_local_bend_from_global_stub_pattern
```

If exact geometry is not available yet, implement a first approximate version using inter-station phi differences and station radius lookup tables.

### 2. DT theta features

From Phase-2 DT theta primitives:

```text
z
k
quality
t0
chi2
station
wheel
sector
bx
```

Create:

```text
dt_z_mean
dt_z_std
dt_k_mean
dt_k_std
dt_k_abs_mean
dt_k_residual_mean
dt_k_residual_std
dt_theta_consistency
```

Use `k` as the local theta-slope variable.

### 3. DT timing features

From Phase-2 DT primitives:

```text
t0
bx
```

Important caveat from the audit:

```text
DT t0 has a large raw offset and must be reference-subtracted before use.
```

Do not use raw `t0` directly as a final physics feature.

Create:

```text
dt_t0_raw_mean
dt_t0_raw_std
dt_t0_corrected_mean
dt_t0_corrected_std
dt_t0_station_slope
dt_bx_mean
dt_bx_std
```

Reference subtraction options:

1. subtract the prompt mean per station, wheel, and sector;
2. subtract the prompt mean per station and eta bin;
3. subtract the event-level prompt-like reference if available.

Start with option 1.

### 4. CSC local-direction features

From CSC LCTs:

```text
strip
keywire
pattern
run3Pattern
slope
bend
quality
bx
station
ring
chamber
```

Create signed slope:

```text
signed_csc_slope = slope with sign from bend
```

The exact sign convention must be verified. Until then, store both raw and signed variants.

Create:

```text
csc_slope_mean
csc_slope_std
csc_slope_abs_mean
csc_slope_abs_max
csc_signed_slope_mean
csc_signed_slope_std
csc_pattern_mean
csc_pattern_std
csc_bend_balance
csc_slope_residual_mean
csc_slope_residual_std
csc_bx_mean
csc_bx_std
```

Important audit note:

```text
CSC local slope is present in raw CSCLctDigi tables, but CSC coord2 is not filled in TPS stubs in the checked samples.
```

Therefore, do not rely only on TPS stubs for CSC local direction.

### 5. RPC features

RPC has less local-direction information, but it is still useful for timing and confirmation.

Create:

```text
rpc_n_hits
rpc_strip_mean
rpc_strip_std
rpc_bx_mean
rpc_bx_std
rpc_station_occupancy
```

Do not treat RPC as a primary displaced-direction detector.

### 6. Pairwise inter-station geometry features

For all relevant primitive pairs or OMTF/TPS stubs, compute:

```text
delta_phi_ij
delta_eta_ij
delta_r_ij
delta_station_ij
delta_bx_ij
detector_pair_type_ij
```

Use station-radius lookup tables where exact geometry is not yet available.

Create:

```text
curvature_proxy_ij = delta_phi_ij / max(abs(delta_r_ij), epsilon)
```

Then summarize:

```text
curvature_proxy_mean
curvature_proxy_std
curvature_proxy_abs_max
curvature_proxy_sign_changes
```

### 7. Beamline-intercept proxy

For each pair:

```text
phi0_proxy_ij = phi_i - curvature_proxy_ij * r_i
```

Then compute:

```text
phi0_proxy_mean
phi0_proxy_std
phi0_proxy_range
```

This is one of the most important derived features.

For prompt muons, different station pairs should extrapolate to a consistent beamline phi.

For displaced muons, this consistency should degrade.

### 8. Local-vs-global bend consistency

For stubs with local bend information:

```text
sign_product_i = sign(local_bend_i) * sign(global_delta_phi_near_i)
```

Create:

```text
local_global_bend_agreement_fraction
local_global_bend_disagreement_fraction
local_global_bend_sign_entropy
```

This can help separate ordinary low-pT bending from displaced non-pointing.

### 9. OMTF output features

Use OMTF output variables only as diagnostic or auxiliary features at first:

```text
omtf_hwPt
omtf_hwPtUnconstrained
omtf_hwQual
omtf_hwEta
omtf_hwPhi
```

Create:

```text
omtf_pt_unc_minus_pt = omtf_hwPtUnconstrained - omtf_hwPt
omtf_pt_unc_over_pt = omtf_hwPtUnconstrained / max(omtf_hwPt, 1)
```

Important audit note:

```text
omtf_hwDXY is uniformly zero in checked samples.
```

Therefore:

```text
Do not use omtf_hwDXY as an input, label, or performance target in the current samples.
```

---

## Variables to Avoid as Inputs

Never use these as model input features:

```text
gen_pt
gen_eta
gen_phi
gen_dxy
gen_lxy
gen_vx
gen_vy
gen_vz
offline reconstructed muon variables
offline-only MuonStub coordinates
simulation-only RPC coordinateX/coordinateY
simulation-only CSC type labels
raw truth matching flags
```

These may be used for plotting, labels, binning, validation, and matching diagnostics only.

---

## Data Quality Checks

Before ranking features, produce a data-quality report.

For each dataset, print:

```text
number of files opened
number of events
number of generated muons
number of matched candidates
mean number of DT primitives
mean number of CSC primitives
mean number of RPC primitives
mean number of OMTF hits
fraction with at least one OMTF candidate
```

Also check:

```text
missing branches
empty branches
NaN or inf values
unexpected all-zero features
negative DT phi chi2 values
CSC slope availability
DT t0 availability
TPS coord2 availability by detector type
```

Known caveats from the audit:

```text
1. omtf_hwDXY is uniformly zero.
2. DT phi chi2 may contain negative overflow-wrapped values.
3. DT t0 has a large raw offset and needs reference subtraction.
4. CSC coord2 is not filled in TPS stubs, so raw CSC LCT slope must be used.
```

Handle DT phi chi2 as:

```text
if chi2 < 0:
    chi2_overflow = 1
    chi2_clean = high_chi2_sentinel
else:
    chi2_overflow = 0
    chi2_clean = chi2
```

---

## Feature Ranking Procedure

### Step 1: single-variable histograms

For every feature, plot normalized histograms:

```text
prompt vs displaced
```

Do this separately for each matched pT pair:

```text
C1 vs C7
C2 vs C8
C3 vs C9
C4 vs C10
C5 vs C11
C6 vs C12
```

Repeat for PU200:

```text
C13 vs C15
C14 vs C16
C17 vs C21
C18 vs C22
C19 vs C23
C20 vs C24
```

### Step 2: AUC per variable

For each feature and each matched pair, compute:

```python
auc = roc_auc_score(label, feature)
auc = max(auc, 1 - auc)
```

Store:

```text
feature
sample_pair
auc
auc_direction
mean_prompt
mean_displaced
std_prompt
std_displaced
```

### Step 3: correlation checks

For each feature, compute correlation with:

```text
gen_pt
gen_eta
abs(gen_eta)
omtf_hwPt
omtf_hwQual
n_omtf_hits
n_dt
n_csc
n_rpc
```

Use both Pearson and Spearman correlations.

The best variables have:

```text
high AUC with label
low correlation with gen_pt
low correlation with omtf_hwPt
low correlation with eta
stable behavior under PU200
```

### Step 4: pT stability score

For each feature, compute:

```text
mean_auc_no_pu = mean AUC over C1-C6 vs C7-C12
min_auc_no_pu
std_auc_no_pu

mean_auc_pu200 = mean AUC over C13-C20 vs C15-C24
min_auc_pu200
std_auc_pu200
```

Good variables should not work only in one pT bin.

### Step 5: pileup robustness score

For each matched pT bin:

```text
delta_auc_pu = auc_no_pu - auc_pu200
```

Good variables should have small degradation from no-PU to PU200.

### Step 6: decorrelated ranking score

Define a simple ranking score:

```text
score =
    mean_auc_pu200
    - 0.20 * abs(corr_with_gen_pt)
    - 0.20 * abs(corr_with_omtf_hwPt)
    - 0.10 * abs(corr_with_abs_eta)
    - 0.10 * auc_std_across_pt_bins
```

This is not a final physics metric. It is a practical sorting tool.

Produce a ranked table:

```text
feature
mean_auc_no_pu
mean_auc_pu200
min_auc_pu200
delta_auc_pu
corr_gen_pt
corr_omtf_hwPt
corr_abs_eta
score
recommendation
```

Recommendations:

```text
KEEP
KEEP_AS_AUXILIARY
REJECT_CORRELATED_WITH_PT
REJECT_UNSTABLE_UNDER_PU
REJECT_TOO_WEAK
REJECT_NOT_FIRMWARE_REALISTIC
```

---

## Simple Baseline Classifiers

After single-variable ranking, train simple classifiers before any GNN.

Use:

```text
logistic regression
shallow random forest
shallow gradient-boosted decision tree
```

Do not over-optimize. These are diagnostic tools.

Suggested Random Forest:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    class_weight="balanced",
    random_state=1,
)
```

Suggested checks:

```text
ROC AUC
efficiency at fixed fake rate
feature permutation importance
feature ablation
```

### Feature ablation groups

Train the classifier with:

```text
all features
without DT phiB features
without DT theta k/z features
without DT timing features
without CSC slope/pattern features
without pairwise geometry features
without OMTF output features
```

This tells which feature family is genuinely important.

---

## Required Plots

Produce at least:

```text
1. Histogram overlays for top 20 variables, C1 vs C7.
2. Histogram overlays for top 20 variables, C2 vs C8.
3. Histogram overlays for top 20 variables, C13 vs C15.
4. Histogram overlays for top 20 variables, C14 vs C16.
5. AUC heatmap: feature vs pT bin, no PU.
6. AUC heatmap: feature vs pT bin, PU200.
7. Correlation heatmap for top 30 features.
8. Feature score ranking table.
9. AUC vs pT bin for top 10 features.
10. No-PU vs PU200 AUC comparison for top 10 features.
11. Mild-displacement response for C25-C28.
12. Feature-ablation performance plot.
```

Also produce diagnostic plots:

```text
number of primitives per event
DT t0 distributions before and after correction
DT chi2 negative-overflow fraction
CSC slope availability
TPS coord2 availability by detector type
OMTF hwDXY distribution showing it is unusable
```

---

## Expected Best Variables to Test Carefully

The audit suggests these are likely to be strongest:

```text
phi0_proxy_std
phi0_proxy_range
dt_phiB_residual_mean
dt_phiB_residual_std
dt_phiB_residual_abs_max
csc_signed_slope_residual_mean
csc_signed_slope_residual_std
dt_k_residual_mean
dt_k_residual_std
dt_t0_corrected_mean
dt_t0_corrected_std
local_global_bend_disagreement_fraction
curvature_proxy_std
omtf_pt_unc_minus_pt
omtf_pt_unc_over_pt
bx_std
```

Be skeptical of raw bend variables:

```text
raw dt_phiB
raw csc_slope
raw curvature_proxy
```

They may be useful, but they may also be strongly correlated with pT.

Prefer residuals and consistency variables.

---

## Final Deliverables

Create the following outputs:

```text
outputs/variable_analysis_report.md
outputs/feature_ranking.csv
outputs/feature_ranking_pu200.csv
outputs/feature_correlations.csv
outputs/feature_ablation.csv
outputs/top_features.json
plots/*.pdf
plots/*.png
analysis_tables/*.parquet
```

The report must include:

```text
1. Dataset list used.
2. Branches read from each ROOT tree.
3. Feature definitions.
4. Data-quality checks.
5. Single-variable ranking.
6. Correlation/decorrelation analysis.
7. PU200 robustness.
8. Mild-displacement ambiguity behavior.
9. Recommended variables for the GNN node schema.
10. Recommended variables for GNN edge features.
11. Variables rejected and why.
12. Remaining uncertainties requiring detector-expert input.
```

The final recommended feature list should be split into:

```text
node_features
edge_features
candidate_summary_features
diagnostic_only_features
do_not_use_features
```

---

## Decision Criteria for GNN Inputs

A variable should be accepted as a GNN input only if it satisfies most of:

```text
1. It is available in the trigger/emulator path.
2. It does not use generator truth.
3. It has useful AUC against displaced labels.
4. It works against low-pT prompt muons.
5. It remains useful in PU200.
6. It is not just a pT proxy.
7. It is not just an eta proxy.
8. It is stable across pT bins.
9. It has a physical interpretation.
10. It can be implemented or approximated in firmware later.
```

The final answer should not simply say:

```text
these features have the highest ML importance
```

It should say:

```text
these features carry displacement information beyond ordinary prompt low-pT curvature
```

That distinction is the whole point of the study.

---

## Suggested Implementation Structure

Use a small, modular Python package:

```text
analysis/
  config/
    datasets.yaml
    features.yaml
  src/
    load_root.py
    build_tables.py
    feature_engineering.py
    quality_checks.py
    rank_features.py
    train_baselines.py
    make_plots.py
  notebooks/
    01_branch_inventory.ipynb
    02_feature_distributions.ipynb
    03_feature_ranking.ipynb
    04_pu200_validation.ipynb
    05_mild_displacement.ipynb
  outputs/
  plots/
  analysis_tables/
```

Prefer `uproot`, `awkward`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

Do not start with PyTorch Geometric. First produce the feature-ranking and decorrelation results.

---

## Minimal First Milestone

The first milestone is complete when the following exists:

```text
1. A table for C1, C2, C7, C8.
2. Histograms for all candidate features.
3. AUC ranking for C1 vs C7 and C2 vs C8.
4. Correlation with gen_pt, omtf_hwPt, and abs(gen_eta).
5. A first ranked list of KEEP / REJECT variables.
```

Only after this milestone should the analysis move to PU200 and later to GNN training.

