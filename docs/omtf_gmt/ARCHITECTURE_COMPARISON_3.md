# Phase B3 Action Plan — GMT/OMTF overlap with `cache_v2`

The goal of Phase B3 is to isolate the effect of the new G-datasets and `cache_v2` schema before doing more architecture work.

In Phase B1/B2, the best stable baseline was `EdgeCompat h64`, but the old `S/B` dataset mixture had two problems:

1. It did not cleanly separate overlap muons from out-of-overlap barrel/KMTF stubs.
2. It did not include explicit hard negatives where a real muon exists but should produce zero OMTF-overlap candidates.

The new G-dataset production and cache schema v2 were built exactly to address that.

So the next step is **not** to immediately invent another model. The next step is:

```text
Train the same baseline models on cache_v2 and compare against the old Phase B1 results.
```

---

## 0. Freeze the validated starting point

Before training, consider the following state frozen:

```text
Cache:
  build/omtf_gmt/cache_v2/

Validation reports:
  build/omtf_gmt/eval/new_dataset_validation_full_G{1–8}.md
  build/omtf_gmt/eval/FEATURE_AUDIT_v2.md

Schema:
  SCHEMA_VERSION = 2
  N_FEATURES = 14
```

Accepted dataset interpretation:

| Dataset    | Role                                                |
| ---------- | --------------------------------------------------- |
| `G1`, `G3` | clean 1-overlap-target samples, prompt/displaced    |
| `G2`, `G4` | 1-overlap-target samples with PU200                 |
| `G5`       | 2-overlap-target displaced sample, sparse but valid |
| `G6`       | 3-overlap-target prompt sample, sparse but valid    |
| `G7`       | low-eta real-muon hard negative, no PU              |
| `G8`       | low-eta real-muon hard negative with PU200          |
| `B4`       | pure-noise background                               |

Important accepted warnings:

```text
G5: frac_target=2 ≈ 3.3%, accepted
G6: frac_target=3 ≈ 6.5%, accepted
```

Reason: multi-muon events do not always place all generated muons inside the same OMTF processor window with transferable KMTF stubs. Zero windows exceed the expected maximum multiplicity, so there is no PartID contamination.

---

## 1. Minimal model set to train first

Train only the minimal set first:

| Priority | Model             | Purpose                                                                   |
| -------- | ----------------- | ------------------------------------------------------------------------- |
| 1        | `edge_compat h64` | Main old winner; must become the new `cache_v2` baseline                  |
| 2        | `deepsets h64`    | Simple non-edge control baseline                                          |
| 3        | `detr_model h64`  | Tests whether Hungarian/permutation matching fixes duplicated slot firing |

Do **not** repeat the full old sweep immediately.

No need yet for:

```text
DeepSets h128
DeepSets h256
EdgeCompat h128
large hyperparameter sweeps
```

Those only make sense after the first `cache_v2` comparison is understood.

---

## 2. Training dataset mix

Use the full `cache_v2` dataset:

```text
G1_pos, G1_neg
G2_pos, G2_neg
G3_pos, G3_neg
G4_pos, G4_neg
G5_pos, G5_neg
G6_pos, G6_neg
G7, G8
B4
```

Do not physically merge the cache folders. Keep them separate on disk and combine them logically in the training dataloader/sampler.

Recommended first repeat factors:

| Dataset | Repeat | Why                                         |
| ------- | -----: | ------------------------------------------- |
| `B4`    |     ×8 | preserve pure-noise rejection from Phase B1 |
| `G7`    |     ×2 | teach clean hard-negative rejection         |
| `G8`    |     ×4 | teach hard-negative + PU rejection          |
| `G1–G6` |     ×1 | normal signal supervision                   |

Rationale:

```text
B4 teaches: no real muon, no candidate.
G7 teaches: real low-eta muon, no overlap candidate.
G8 teaches: real low-eta muon + PU, no overlap candidate.
G1–G6 teach: real overlap candidates with multiplicity 1/2/3.
```

Do not oversample `G7/G8` too aggressively at first. If hard negatives dominate too much, the model may become overly conservative and lose signal efficiency.

---

## 3. Dataset aliases

Add aliases in the training/eval scripts so the user-facing command can stay clean.

```python
DATASET_ALIASES = {
    "G1": ["G1_pos", "G1_neg"],
    "G2": ["G2_pos", "G2_neg"],
    "G3": ["G3_pos", "G3_neg"],
    "G4": ["G4_pos", "G4_neg"],
    "G5": ["G5_pos", "G5_neg"],
    "G6": ["G6_pos", "G6_neg"],
}
```

Then this command:

```bash
--datasets G1 G2 G3 G4 G5 G6 G7 G8 B4
```

expands internally to:

```text
G1_pos G1_neg G2_pos G2_neg ... G6_pos G6_neg G7 G8 B4
```

The dataloader or `WeightedRandomSampler` must shuffle/mix samples across all physical datasets.

---

## 4. Training commands

### 4.1 EdgeCompat h64 — main baseline

```bash
python src/omtf_gmt/train.py \
  --cache-dir build/omtf_gmt/cache_v2 \
  --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
  --repeat B4:8 G7:2 G8:4 \
  --model edge_compat \
  --hidden 64 \
  --epochs 50 \
  --batch-size 4096 \
  --output-dir build/omtf_gmt/checkpoints/edge_compat_B3a_cachev2
```

This is the most important run.

It answers:

```text
Did the new dataset/schema solve a large fraction of the old problem without changing the architecture?
```

---

### 4.2 DeepSets h64 — simple control

```bash
python src/omtf_gmt/train.py \
  --cache-dir build/omtf_gmt/cache_v2 \
  --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
  --repeat B4:8 G7:2 G8:4 \
  --model deepsets \
  --hidden 64 \
  --epochs 50 \
  --batch-size 4096 \
  --output-dir build/omtf_gmt/checkpoints/deepsets_B3a_cachev2
```

This checks whether the improvement comes mainly from the data/schema or from edge-based relational reasoning.

---

### 4.3 DETR h64 — slot competition test

```bash
python src/omtf_gmt/train.py \
  --cache-dir build/omtf_gmt/cache_v2 \
  --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
  --repeat B4:8 G7:2 G8:4 \
  --model detr_model \
  --hidden 64 \
  --epochs 50 \
  --batch-size 4096 \
  --output-dir build/omtf_gmt/checkpoints/detr_B3a_cachev2
```

This tests the architectural fix for the clean one-muon duplication problem.

Expected:

```text
DETR should help G1/G3 duplicate-slot collapse.
DETR may only partially help G2/G4 PU-noise firing.
```

---

## 5. Evaluation plan

Evaluation must be done in two modes:

1. Combined logical datasets.
2. Separate eta-side datasets.

---

### 5.1 Combined headline evaluation

Evaluate:

```text
G1 = G1_pos + G1_neg
G2 = G2_pos + G2_neg
G3 = G3_pos + G3_neg
G4 = G4_pos + G4_neg
G5 = G5_pos + G5_neg
G6 = G6_pos + G6_neg
G7
G8
B4
```

This gives the main physics table.

---

### 5.2 Pos/neg eta-side evaluation

Also evaluate separately:

```text
G1_pos vs G1_neg
G2_pos vs G2_neg
G3_pos vs G3_neg
G4_pos vs G4_neg
G5_pos vs G5_neg
G6_pos vs G6_neg
```

Reason: combined metrics can hide eta-side bugs.

Possible hidden issues:

```text
phi centering sign bug
processor mapping bug
eta-sign asymmetry
charge/sign convention issue
truth-transfer issue on one side
```

---

## 6. Metrics to report

The critical metrics are not just global loss.

Report at least:

| Metric                        | Meaning                                 |
| ----------------------------- | --------------------------------------- |
| `G1/G3 slot-1 fake`           | clean one-target duplicate-slot problem |
| `G2/G4 slot-1 fake`           | one-target + PU noise-firing problem    |
| `G5 two-target recovery`      | two-candidate reconstruction            |
| `G6 slot-2 efficiency`        | three-candidate recovery                |
| `G7 zero-win FP`              | clean hard-negative rejection           |
| `G8 zero-win FP`              | PU hard-negative rejection              |
| `B4 zero-win FP`              | pure-noise rejection                    |
| pos/neg efficiency difference | eta-side symmetry                       |
| `sigma68` pT error            | pT regression quality                   |

For the first comparison, the most important table is:

| Metric                     | Old EdgeCompat on S/B | New EdgeCompat on G/cache_v2 | Interpretation                              |
| -------------------------- | --------------------: | ---------------------------: | ------------------------------------------- |
| clean 1-target slot-1 fake |             S2: 57.4% |                     G1/G3: ? | Did duplicate collapse improve?             |
| PU 1-target slot-1 fake    |             B2: 71.7% |                     G2/G4: ? | Did domain features/hard negatives help?    |
| 3-target efficiency        |             S4: 69.3% |                        G6: ? | Did G6 improve slot-2 learning?             |
| pure-noise FP              |              B4: 0.0% |                        B4: ? | Did we preserve noise rejection?            |
| hard-negative FP           |                   n/a |                     G7/G8: ? | New critical metric                         |
| pT sigma68                 |            old values |                   new values | Did target construction improve regression? |

---

## 7. Decision logic after EdgeCompat B3a

After `edge_compat h64` on `cache_v2`, decide which branch to follow.

---

### Case A — Big improvement on G2/G4 and G7/G8

If:

```text
G2/G4 slot-1 fake drops strongly
G7/G8 FP ≈ 0
B4 FP ≈ 0
```

then the old problem was largely caused by dataset/domain ambiguity.

Next move:

```text
Keep EdgeCompat as baseline.
Tune threshold/loss/repeat factors.
Only then consider larger hidden sizes.
```

---

### Case B — G7/G8 fixed, but G1/G3 still overcount

If:

```text
G7/G8 FP ≈ 0
G2/G4 improves
G1/G3 slot-1 fake remains high
```

then the hard-negative/domain issue is improved, but the clean duplicate-slot problem remains architectural.

Next move:

```text
Train/evaluate DETR or another hard slot-competition model.
```

---

### Case C — G8 remains bad

If:

```text
G8 false positives remain high
```

then the PU hard-negative problem is still not solved.

Next options:

```text
increase G8 repeat factor
add explicit hard-negative loss
inspect attention on G8 false positives
use truth_source-aware loss weighting
try stricter stub filtering or attention masking
```

---

### Case D — B4 worsens

If:

```text
B4 zero-win FP increases
```

then the training mix diluted pure-noise supervision.

Next options:

```text
increase B4 from ×8 to ×10 or ×12
verify B4 is still included in every training run
check threshold scan
```

---

## 8. Optional repeat-factor sweep

Only do this **after** the first EdgeCompat run.

| Run             | B4 | G7 | G8 | Purpose         |
| --------------- | -: | -: | -: | --------------- |
| default         |  8 |  2 |  4 | baseline        |
| more hard-neg   |  8 |  4 |  8 | reduce G7/G8 FP |
| more pure-noise | 12 |  2 |  4 | protect B4      |
| balanced        | 10 |  3 |  6 | compromise      |

Do not start with this sweep. First get one clean baseline.

---

## 9. Documentation update after each run

Append a new section to the architecture comparison document:

```text
Phase B3 — cache_v2 / G-dataset results
```

Suggested structure:

```text
1. Training setup
2. Dataset mix
3. EdgeCompat h64 result
4. Comparison to old S/B baseline
5. Pos/neg eta symmetry
6. Hard-negative performance
7. Decision: dataset fixed problem or architecture still needed?
```

---

## 10. Minimal next moves

The shortest useful sequence is:

```text
1. Train EdgeCompat h64 on cache_v2 with B4×8, G7×2, G8×4.
2. Evaluate on G1–G8+B4, both combined and pos/neg split.
3. Compare against old EdgeCompat S/B results.
4. Train DETR h64 on the same cache/mix.
5. Decide whether the remaining issue is slot competition or PU/hard-negative rejection.
```

This gives a clean answer to the main question:

```text
Was the old GMT model failing because of architecture, because of dataset/domain ambiguity, or both?
```

Most likely answer:

```text
Both.
```

But Phase B3 will tell us how much each part matters.

---

# Immediate Next Instructions — after B3a/B3b/B3c/B3d/B3e

* **Document the findings first.**

  * Update `docs/omtf_gmt/ARCHITECTURE_COMPARISON.md` with a new section:

    ```text
    Phase B3 — cache_v2 repeat-factor sweep
    ```
  * Include the five-way comparison table:

    ```text
    B3a = B4×8, G7×2, G8×4
    B3d = B4×6, G7×4, G8×4
    B3e = B4×6, G7×4, G8×3
    B3c = B4×6, G7×4, G8×2
    B3b = B4×6, G7×4, G8×1
    ```
  * State the main conclusion clearly:

    ```text
    No repeat-factor point satisfies all provisional targets simultaneously.
    B3d is selected as the current working baseline because it gives the best overall balance:
    strong G5/G6 recovery, B4 FP = 0%, near-zero slot overcounting, and only moderate G7/G8 FP excess.
    ```
  * Update the interpretation of the old Phase B1 overcounting:

    ```text
    The dominant 1-target overcounting was mostly a dataset/domain-definition issue, not an intrinsic architecture limit.
    ```
  * Add the new working targets:

    ```text
    G7 FP monitoring target: ≲ 8%
    G8 FP monitoring target: ≲ 13%
    B4 FP hard target: 0%
    ```

* **Freeze B3d as the current EdgeCompat h64 baseline.**

  * Baseline name:

    ```text
    edge_compat_h64_B3d_cachev2
    ```
  * Training mix:

    ```text
    B4×6, G7×4, G8×4
    ```
  * Reason:

    ```text
    B3d gives much better multi-target recovery than B3a while only slightly missing the hard-negative FP targets.
    ```

* **Do not keep tuning repeat factors immediately.**

  * The sweep already shows the trend:

    ```text
    More G8 → lower hard-negative FP, lower signal/multi-target efficiency.
    Less G8 → higher signal/multi-target efficiency, worse hard-negative FP.
    ```
  * Further repeat-factor tuning alone is unlikely to remove the trade-off.

* **Add periodic checkpoint saving to `train.py`.**

  * Keep saving best validation loss as now.
  * Also save fixed epochs:

    ```text
    epoch_050.pt
    epoch_075.pt
    epoch_100.pt
    last.pt
    ```
  * Reason:

    ```text
    Best val_loss is not necessarily the best physics operating point.
    ```

* **Add a learning-rate scheduler.**

  * Preferred first option:

    ```python
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    ```
  * Call after each epoch:

    ```python
    scheduler.step()
    ```
  * Add CLI flag if convenient:

    ```text
    --scheduler cosine
    ```

* **Run EdgeCompat h64 B3d for 100 epochs.**

  * Purpose:

    ```text
    Check whether the 50-epoch B3d model was undertrained.
    ```
  * Command:

    ```bash
    python src/omtf_gmt/train.py \
      --cache-dir build/omtf_gmt/cache_v2 \
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
      --repeat B4:6 G7:4 G8:4 \
      --model edge_compat \
      --hidden 64 \
      --epochs 100 \
      --batch-size 4096 \
      --amp \
      --output-dir build/omtf_gmt/checkpoints/edge_compat_h64_B3d_100ep_cachev2
    ```

* **Evaluate several checkpoints from the 100-epoch run.**

  * Evaluate:

    ```text
    best_val_loss checkpoint
    epoch_050 checkpoint
    epoch_075 checkpoint
    epoch_100 checkpoint
    last checkpoint
    ```
  * Use the same evaluation datasets:

    ```text
    G1 G2 G3 G4 G5 G6 G7 G8 B4
    ```
  * Compare physics metrics, not only validation loss:

    ```text
    G1/G2/G3/G4 efficiency
    G5 slot-1 efficiency
    G6 slot-2 efficiency
    G7 FP
    G8 FP
    B4 FP
    slot-1 fake / overcounting
    pT sigma68
    ```

* **Run EdgeCompat h128 with the B3d mix.**

  * Purpose:

    ```text
    Test whether more capacity improves the B3d trade-off without changing the dataset mix.
    ```
  * Command:

    ```bash
    python src/omtf_gmt/train.py \
      --cache-dir build/omtf_gmt/cache_v2 \
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
      --repeat B4:6 G7:4 G8:4 \
      --model edge_compat \
      --hidden 128 \
      --epochs 100 \
      --batch-size 4096 \
      --amp \
      --output-dir build/omtf_gmt/checkpoints/edge_compat_h128_B3d_100ep_cachev2
    ```
  * Accept h128 only if it improves at least one hard metric without damaging the others:

    ```text
    better G5/G6 recovery, or better G7/G8 FP, or better G1/G2 efficiency,
    while keeping B4 FP = 0 and slot overcounting near zero.
    ```

* **Run DETR with stronger no-object penalty using the B3d mix.**

  * Purpose:

    ```text
    DETR had excellent multiplicity counting but bad noise rejection.
    Test whether stronger no-object loss fixes B4/G7/G8 FP.
    ```
  * First DETR run:

    ```bash
    python src/omtf_gmt/train.py \
      --cache-dir build/omtf_gmt/cache_v2 \
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
      --repeat B4:6 G7:4 G8:4 \
      --model detr_model \
      --hidden 64 \
      --epochs 100 \
      --batch-size 4096 \
      --amp \
      --w-no-obj 0.5 \
      --output-dir build/omtf_gmt/checkpoints/detr_h64_B3d_wnoobj0p5_100ep_cachev2
    ```
  * Second DETR run:

    ```bash
    python src/omtf_gmt/train.py \
      --cache-dir build/omtf_gmt/cache_v2 \
      --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
      --repeat B4:6 G7:4 G8:4 \
      --model detr_model \
      --hidden 64 \
      --epochs 100 \
      --batch-size 4096 \
      --amp \
      --w-no-obj 1.0 \
      --output-dir build/omtf_gmt/checkpoints/detr_h64_B3d_wnoobj1p0_100ep_cachev2
    ```
  * Only run `w_no_obj=2.0` if both `0.5` and `1.0` still give unacceptable B4/G7/G8 FP.

* **Add permutation-invariant DETR evaluation before trusting DETR per-slot metrics.**

  * For DETR, do not use fixed slot-0/slot-1/slot-2 interpretation.
  * Report:

    ```text
    true multiplicity → predicted multiplicity confusion
    zero-candidate FP
    event-level trigger efficiency
    Hungarian-matched pT sigma68
    Hungarian-matched candidate recovery
    ```
  * Fixed per-slot metrics are valid for EdgeCompat/DeepSets, but not for DETR.

* **Run threshold scans for the selected checkpoints.**

  * Do not decide only at threshold `0.0`.
  * Evaluate at least:

    ```text
    threshold = 0.0
    threshold = 0.2
    threshold = 0.5
    threshold = 1.0
    ```
  * Make this table for each candidate model:

    ```text
    threshold | G1 eff | G2 eff | G5 rec | G6 rec | G7 FP | G8 FP | B4 FP
    ```
  * Reason:

    ```text
    A small positive threshold may reduce G7/G8 FP with limited signal-efficiency loss.
    ```

* **Optional technique after these runs: hard-negative weighted loss.**

  * Do not do this before the 100-epoch/h128/DETR checks.
  * Motivation:

    ```text
    Dataset repetition is coarse. A hard-negative loss weight could penalise G7/G8 false candidates
    without making G8 dominate the whole epoch.
    ```
  * First test idea:

    ```text
    Mix: B4×6, G7×2, G8×2
    Loss: hard-negative candidate BCE weight = 2.0
    ```
  * Use `meta_is_hard_neg` to identify G7/G8 windows.

* **Final comparison table to produce after the next runs.**

  * Rows:

    ```text
    EdgeCompat h64 B3d 50ep
    EdgeCompat h64 B3d 100ep
    EdgeCompat h128 B3d 100ep
    DETR h64 B3d w_no_obj=0.5
    DETR h64 B3d w_no_obj=1.0
    ```
  * Columns:

    ```text
    val_loss
    G1 eff
    G2 eff
    G3 eff
    G4 eff
    G5 2-target recovery
    G6 3-target recovery
    G7 FP
    G8 FP
    B4 FP
    slot-1 fake / 1-target overcounting
    pT sigma68 G1
    pT sigma68 G2
    ```

* **Decision rule after the next runs.**

  * Keep EdgeCompat if:

    ```text
    DETR still has bad B4/G7/G8 FP or requires too much threshold tuning.
    ```
  * Switch to DETR only if:

    ```text
    B4 FP ≈ 0
    G7/G8 FP comparable to EdgeCompat
    G5/G6 recovery clearly better
    G1/G2 efficiency not worse
    ```
  * Accept h128 only if:

    ```text
    it improves the physics metrics enough to justify the parameter increase.
    ```
