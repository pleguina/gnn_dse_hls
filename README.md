# OMTF GMT — FP32 Validation Baseline

**TPS EdgeCompat h64 + G7/G8/G9/G10 hard negatives**
Run A · w_hard_neg = 0.25 · threshold logit > 0.0

This branch is a self-contained reconstruction kit for the frozen FP32 baseline model
for the CMS OMTF GMT ML trigger candidate study.
It contains only the source code, scripts, models, and plots needed to reproduce the result.
The full development history lives on the `omtf-migration` branch.

---

## What is in this branch

```
src/omtf_gmt/           Python package  (dataset, features, train, models)
scripts/                Standalone Python pipeline scripts
pipeline/               Shell scripts, one per stage
condor/                 HTCondor submit files
docs/omtf/              Documentation
artifacts/
  models/
    hn025_baseline/     TPS EdgeCompat h64, G7/G8 only  (pre-G9/G10)
    g9g10_runa_frozen/  Frozen FP32 baseline  (Run A)
  plots/
    presentation/       43 study plots  (hn025 baseline)
    das_validation/     7 DAS external validation plots  (Run A)
requirements.txt
README.md
```

---

## Frozen model identity

| | |
|---|---|
| Architecture | EdgeCompat h64 · 38,987 params |
| Stub source | MuonStubTps (barrel + endcap) |
| Hard negatives | G7/G8 low-eta + G9/G10 high-eta |
| w_hard_neg | 0.25 (Run A) |
| Threshold | logit > 0.0 |
| Checkpoint | `artifacts/models/g9g10_runa_frozen/model.pt` |

See `artifacts/models/g9g10_runa_frozen/README.md` for full metrics.

---

## How to reconstruct from scratch

### 0. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Build the TPS cache (G1–G8 + B4)

```bash
bash pipeline/build_cache.sh
# HTCondor: condor_submit condor/build_cache.sub
```

### 2. Validate G1–G8 datasets

```bash
bash pipeline/validate.sh G1
# repeat for G2–G8
# HTCondor: condor_submit condor/validate.sub
```

### 3. Append G9/G10 to the cache

```bash
bash pipeline/append_g9g10.sh
# HTCondor: condor_submit condor/append_g9g10.sub
```

### 4. Validate G9/G10 datasets

```bash
bash pipeline/validate.sh G9
bash pipeline/validate.sh G10
```

### 5. Train Run A (the selected model)

```bash
bash pipeline/train_runa.sh
# HTCondor: condor_submit condor/train.sub
```

Run B (w_hard_neg=0.50) and Run C (w_hard_neg=0.10) are available as reference.
Run A was selected: best G9/G10 rejection without regressing G7/G8 or B4.

### 6. Evaluate on controlled datasets (G1–G10, B4)

Included in the train script above. To re-run standalone:

```bash
python scripts/eval.py \
    --checkpoint build/omtf_gmt/checkpoints/edge_compat_h64_tps_g9g10_runa/gmt_edge_compat_best.pt \
    --cache-dir  build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --output     build/omtf_gmt/eval/runa_eval.md
```

### 7. Freeze the checkpoint

```bash
bash pipeline/freeze.sh
# writes build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/
```

### 8. Build DAS external validation cache

```bash
bash pipeline/build_das_cache.sh
# HTCondor: condor_submit condor/build_das_cache.sub
```

### 9. Evaluate on DAS samples (overlap eta-filtered)

```bash
bash pipeline/eval_das_filtered.sh
# HTCondor: condor_submit condor/eval_das_filtered.sub
```

### 10. Generate validation plots

```bash
bash pipeline/plots.sh --device cpu
```

Produces 7 plots in `build/omtf_gmt/plots/das_validation/`:
1. Efficiency before/after eta filter
2. Efficiency vs gen pT (overlap-filtered)
3. MinBias accept vs logit threshold (pT-gated curves)
4. Efficiency vs |dxy| — displaced/LLP
5. Efficiency vs TPS stub count — displaced/LLP
6. MinBias accept vs predicted-pT cut at logit > 0.0
7. ML model vs current OMTF comparison

---

## Key results

| Dataset | ML evt-eff | OMTF (any) |
|---------|-----------|------------|
| Single μ flat pT | 0.915 | 0.989 |
| Displaced low pT | 0.681 | 0.885 |
| Displaced mid pT | 0.721 | 0.935 |
| DY→μμ | 0.792 | 0.973 |
| LLP H→4μ | 0.787 | 0.904 |
| **MinBias bg accept** | **0.036** | **0.066** |

MinBias window FP drops to 0.001 with predicted-pT > 10 GeV cut.

---

## Data

Raw ROOT files (omtf_hits + omtf_nano pairs) are expected at:
- `data/prod/` — controlled G-datasets (G1–G10, B4)
- `data/das_prod/` — DAS external validation samples

See `docs/omtf/ROOT_BRANCHES.md` for the NanoAOD branch schema.
