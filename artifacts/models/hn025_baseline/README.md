# Baseline model: TPS EdgeCompat h64, w_hard_neg=0.25, G7/G8 only

This is the selected TPS baseline **before** G9/G10 high-eta hard negatives were added.
It is the direct predecessor of the frozen FP32 baseline in `g9g10_runa_frozen/`.

## Identity

| Field | Value |
|-------|-------|
| Model | EdgeCompat h64 (38,987 params) |
| Stub source | MuonStubTps |
| Hard negatives | G7/G8 (low-eta barrel only) |
| Hard-neg loss weight | w_hard_neg = 0.25 |
| Operating threshold | logit > 0.0 |
| Cache | cache_v2_tps (G1-G8, B4) |

## Key metrics at logit > 0.0

| Dataset | cand_eff | zero_fp |
|---------|----------|---------|
| G1 (clean prompt)    | 0.932 | 0.55 |
| G2 (prompt PU200)    | 0.829 | 0.08 |
| G3 (displaced)       | 0.901 | 0.59 |
| G4 (displaced PU200) | 0.804 | 0.08 |
| G5 (2-displaced PU)  | 0.797 | 0.11 |
| G6 (3-prompt PU)     | 0.950 | 0.15 |
| G7 (low-eta HN)      | —     | 0.036 |
| G8 (low-eta HN PU)   | —     | 0.022 |
| B4 (pure noise PU)   | —     | 0.000 |

## Why this was superseded

G9/G10 hard negatives (high-eta endcap muons) were added to teach the model
that muons above the overlap band (|eta| > 1.24) should also produce zero candidates.
See `g9g10_runa_frozen/README.md` for the updated model.

## Presentation plots

The 43 plots in `artifacts/plots/presentation/` were generated with this model.
