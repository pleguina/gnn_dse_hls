# FP32 Validation Baseline

**TPS EdgeCompat h64 + G7/G8/G9/G10 hard negatives**
Run A, w_hard_neg = 0.25
Threshold: logit > 0.0
Plus predicted-pT operating scans

---

## Identity

| Field | Value |
|-------|-------|
| Model | EdgeCompat h64 (38,987 params) |
| Stub source | MuonStubTps (TPS, barrel + endcap) |
| Hard negatives | G7/G8 (low-eta barrel) + G9/G10 (high-eta endcap) |
| Hard-neg loss weight | w_hard_neg = 0.25 (Run A) |
| Operating threshold | logit > 0.0 |
| Best epoch | 94 |
| Best val_loss | 0.3988 |
| Precision | FP32 (not quantized) |

---

## Training configuration

```
Datasets:  G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4
Repeats:   B4x4  G7x4  G8x3  G9x4  G10x1
LR:        5e-4  cosine annealing  100 epochs
Batch:     4096
AMP:       yes  grad_clip_norm=1.0
Cache:     cache_v2_tps (TPS stubs, schema v2)
```

---

## Controlled dataset metrics (logit > 0.0)

| Dataset | Role | cand_eff | zero_fp | evt_trig@10 |
|---------|------|----------|---------|-------------|
| G1 (clean prompt) | signal | 0.908 | 0.48 | 0.993 |
| G2 (prompt PU200) | signal | 0.803 | 0.07 | 0.959 |
| G3 (displaced) | signal | 0.886 | 0.61 | 0.975 |
| G4 (displaced PU200) | signal | 0.789 | 0.08 | 0.936 |
| G5 (2-displaced PU200) | signal | 0.775 | 0.11 | 0.931 |
| G6 (3-prompt PU200) | signal | 0.944 | 0.15 | 0.993 |
| G7 (low-eta HN) | hard neg | — | 0.038 | win_bg=0.036 |
| G8 (low-eta HN PU200) | hard neg | — | 0.021 | win_bg=0.021 |
| G9 (high-eta HN) | hard neg | — | 0.002 | win_bg=0.003 |
| G10 (high-eta HN PU200) | hard neg | — | 0.015 | win_bg=0.015 |
| B4 (pure noise PU200) | background | — | 0.000 | win_bg=0.000 |

---

## DAS external validation metrics (overlap-filtered, logit > 0.0)

Overlap filter: gen muon |eta| in [0.82, 1.24] via signal-stub proxy.

| Dataset | ML evt-eff | OMTF (any) | OMTF (pT>10) |
|---------|-----------|------------|--------------|
| Single mu flat pT | 0.915 | 0.989 | 0.809 |
| Displaced low pT | 0.681 | 0.885 | 0.248 |
| Displaced mid pT | 0.721 | 0.935 | 0.300 |
| DY->mumu | 0.792 | 0.973 | 0.723 |
| LLP H->4mu | 0.787 | 0.904 | 0.619 |
| MinBias bg accept | 0.036 | 0.066 | 0.017 |

---

## Predicted-pT operating scan (MinBias, logit > 0.0)

| Predicted-pT cut | Window FP | Event accept |
|-----------------|-----------|--------------|
| none | 0.028 | 0.036 |
| > 5 GeV | 0.014 | 0.017 |
| > 10 GeV | 0.001 | 0.002 |
| > 15 GeV | ~0.000 | ~0.000 |
| > 20 GeV | ~0.000 | ~0.000 |

Full threshold x pT-cut scan: minbias_pt_operating_scan.json

---

## Files in this directory

| File | Contents |
|------|----------|
| model.pt | PyTorch checkpoint (state_dict + training args) |
| training_mix.json | Effective dataset weights used during training |
| history.json | Per-epoch train/val loss and metrics |
| feature_config.json | TPS feature schema (14 features, Nmax=24, K=3) |
| dataset_manifest.json | Cache manifest for cache_v2_tps |
| threshold_scan.json | Per-dataset logit threshold scan (G1-G10, B4) |
| eval.md / eval.json | Full controlled-dataset evaluation report |
| das_eval_unfiltered.json/md | DAS eval without eta filter |
| das_eval_filtered.json/md | DAS eval with overlap eta filter |
| minbias_pt_operating_scan.json | MinBias window/event accept vs logit thr x pT cut |
| omtf_comparison_stats.json | Current OMTF candidate efficiency from nano files |
| plots/ | Seven CMS-facing validation figures |

---

## What comes next

This checkpoint is frozen for:
1. QAT (Quantization-Aware Training) — input to the quantization pass
2. HLS/RTL implementation — reference for accuracy vs latency trade-offs
3. CMS collaboration validation — plots in plots/ are presentation-ready

Do not modify this directory. Create a new frozen_qat_* directory for any
quantized variants.
