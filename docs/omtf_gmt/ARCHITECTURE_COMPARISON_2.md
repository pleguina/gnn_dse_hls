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

## Implementation status (2026-05-04)

### Completed

| Item | Files |
| --- | --- |
| Dataset aliases `G1`→`[G1_pos, G1_neg]` etc. | `src/omtf_gmt/dataset.py`: `DATASET_ALIASES`, `expand_datasets()`, `expand_repeats()` |
| Alias expansion in training | `src/omtf_gmt/train.py` uses `expand_datasets` / `expand_repeats` |
| Alias expansion in evaluation | `scripts/omtf_gmt/eval_gmt.py` uses `expand_datasets` |
| Train scripts — B3a | `run_train_edge_compat_B3a.sh`, `run_train_deepsets_B3a.sh`, `run_train_detr_B3a.sh` |
| HTCondor submit | `train_B3a_htcondor.sub` (3 jobs queued in one cluster) |

### Confirmed training mix (dry-run, edge_compat_B3a)

```text
G1_pos:  40,838  (2.9%)       G1_neg:  40,762  (2.8%)
G2_pos:  73,427  (5.1%)       G2_neg:  73,288  (5.1%)
G3_pos:  25,301  (1.8%)       G3_neg:  25,051  (1.8%)
G4_pos:  52,178  (3.6%)       G4_neg:  52,698  (3.7%)
G5_pos:  78,044  (5.5%)       G5_neg:  77,630  (5.4%)
G6_pos: 117,016  (8.2%)       G6_neg: 117,798  (8.2%)
G7 ×2:  21,652  (1.5%)
G8 ×4: 420,404  (29.4%)       ← dominant: real-muon + PU hard negatives
B4 ×8: 214,336  (15.0%)       ← pure-noise rejection
Total train: 1,430,423  |  Val: 161,789
```

Model params: 38,987 (vs 38,795 Phase B1 — +192 = 3 new input features × h64).

Eval cluster 1074118 (3 jobs).
Reports: `build/omtf_gmt/eval/{edge_compat,deepsets,detr}_B3a_eval.md`

---

## Phase B3 results (2026-05-04)

### Training summary

| Model | Params | val_loss (best) | Best epoch | Wall time |
| --- | --- | --- | --- | --- |
| EdgeCompat h64 | 38,987 | 0.7024 | 50 | ~20 min |
| DeepSets h64   | 26,506 | 0.7585 | 50 | ~20 min |
| DETR h64       | 51,205 | 0.5065 | 35 | ~20 min |

Param increase vs Phase B1: +192 = 3 new input features × h64 (N_FEATURES 11→14). ✓

### Section 6 comparison table

All metrics at threshold = 0.0.  Overcounting = fraction of true-1-target windows predicted as 2-candidate.
DETR per-slot metrics (eff, slot-1 fake) are invalid — slot ordering is permutation-invariant during
training.  Confusion-matrix based figures (marked *) are used for DETR instead.

| Metric | Phase B1 EdgeCompat (S/B) | EdgeCompat B3a (G) | DeepSets B3a (G) | DETR B3a (G) |
| --- | --- | --- | --- | --- |
| Clean 1-target overcounting (G1/G3 ↔ S2) | 57.4% | **0.14%** | 0.25% | 2.1%* |
| PU 1-target overcounting (G2/G4 ↔ B2) | 71.7% | **1.2%** | 1.3% | 5.0%* |
| 2-target recovery (G5 true=2→pred=2) | — | 51.4% | 22.8% | 70.2%* |
| 3-target recovery (G6 true=3→pred=3) | S4: 69.3% slot-2 | 43.4% | 35.0% | 97.3%* |
| G7 clean hard-neg FP | n/a | 7.3% | 7.3% | 31.7% |
| G8 PU hard-neg FP | n/a | 10.0% | 9.4% | 30.3% |
| B4 pure-noise FP | 0.0% | **0.0%** | **0.0%** | **24.7%** |
| G1 signal eff (slot 0) | S2: 99.3% | 82.6% | 76.2% | — |
| pT sigma68 (G1 clean) | 0.559 | 0.184 | — | — |

G1/G3 signal efficiency is lower than S2 (82–83% vs 99%) because the G-datasets span pT 2–200 GeV
including many low-pT muons (eff at pT>5 GeV: ~87%).  The S2/B2 comparison datasets used DY-like
pT spectra skewed to higher pT.

pT sigma68 improvement (0.559 → 0.184) is partly from fixing the double-softplus bug in eval
(`positive_pt` was re-applying F.softplus to already-positive model output).

### Finding 23 — G-dataset training resolves slot overcounting completely

Clean 1-target duplicate-slot overcounting (Phase B1's dominant failure):

| Dataset | Phase B1 | B3a EdgeCompat | B3a DeepSets |
| --- | --- | --- | --- |
| Clean 1-target (S2/G1) | 57.4% | **0.14%** | 0.25% |
| PU 1-target (B2/G2) | 71.7% | **1.2%** | 1.3% |

The overcounting was driven by dataset ambiguity (out-of-overlap barrel stubs labelled as noise
but looking like a second muon), not by architectural limitations.  Adding `stub_in_overlap`,
`abs_eta`, `eta_dist_to_overlap` features and training on G7/G8 hard negatives with explicit
domain supervision resolved it without any architecture change.

**Decision (Section 7 Case B):** hard-negative/domain issue is solved.  The remaining open
problem is multi-candidate recovery (G5/G6) and moderate hard-negative FP (G7/G8).

### Finding 24 — EdgeCompat outperforms DeepSets on multi-candidate recovery

| Metric | EdgeCompat B3a | DeepSets B3a |
| --- | --- | --- |
| G5 2-target (pred=2 / true=2) | 51.4% | 22.8% |
| G6 3-target (pred=3 / true=3) | 43.4% | 35.0% |
| G1 signal eff | 82.6% | 76.2% |

Edge-based pairwise stub compatibility is critical for multi-muon disambiguation.  DeepSets global
pooling loses the inter-stub relational structure needed to distinguish two-muon from one-muon
windows.

### Finding 25 — DETR fails noise rejection; confusion-matrix multiplicity counting is excellent

DETR with Hungarian matching loss learned permutation-invariant multiplicity prediction but at
the cost of catastrophic noise firing:

| Dataset | DETR FP | EdgeCompat FP |
| --- | --- | --- |
| B4 (pure noise) | **24.7%** | 0.0% |
| G7 (clean hard neg) | 31.7% | 7.3% |
| G8 (PU hard neg) | 30.3% | 10.0% |

Root cause: the Hungarian matching cost assigns w_no_obj=0.1 to empty slot predictions.  This
weak no-object penalty allows spurious slot 1 and 2 firing on noise windows.  The model never
learned to suppress empty predictions because the training loss never strongly penalised firing
on zero-candidate windows.

DETR multiplicity counting (from confusion matrix, ignoring slot order) is excellent:
G5 2-target: 70.2%, G6 3-target: 97.3% — substantially better than EdgeCompat.
These results are inaccessible under the current fixed-slot eval convention.

DETR requires either (a) permutation-invariant evaluation, or (b) a stronger no-object penalty
(w_no_obj ≥ 0.5) and retraining before it can be used operationally.

### Finding 26 — G7/G8 hard-negative FP rates are moderate and improvable

| Dataset | EdgeCompat | DeepSets |
| --- | --- | --- |
| G7 (clean, no PU) | 7.3% | 7.3% |
| G8 (PU200) | 10.0% | 9.4% |

Both architectures show identical G7 performance (7.3%) — the clean hard-negative case is
architecture-independent at this level.  G8 is slightly better for DeepSets (9.4% vs 10.0%).
Both are new and could be reduced by increasing G7/G8 repeat factors (Section 8 sweep).

### Next steps

1. **B3b repeat-factor sweep** (submitted, cluster 1074126): EdgeCompat h64 with B4×6, G7×4, G8×1.
   Motivation: G8×4 dominated B3a at 29.4% effective fraction, suppressing signal efficiency.
   B3b signal fraction rises from 57% → 69%; G7 supervision quadruples; G8 drops to 9.7%.
   Target: G1/G3 eff ≥ 86%, G7 FP ≤ 5%, G8 FP ≤ 12%, B4 FP ≈ 0.

   | Component | B3a | B3b |
   | --- | --- | --- |
   | G1–G6 signal | 57.1% | 68.8% |
   | G7 ×4 | 1.5% | 4.0% |
   | G8 ×1 | 29.4% | 9.7% |
   | B4 ×6 | 15.0% | 14.8% |
   | Total eff | 1,430,423 | 1,083,188 |

   If G8 FP worsens significantly, follow with B4×6, G7×4, G8×2 as intermediate.

2. **Rerun DETR with stronger no-object penalty** (w_no_obj = 0.5–1.0) and evaluate with
   permutation-invariant metrics.  DETR's multiplicity counting is too good to abandon.
3. **G5/G6 multi-target recovery**: EdgeCompat 43–51% on G5/G6 is the remaining gap.
   Consider larger hidden size (h128) or attention-based slot decoder on top of EdgeCompat encoder.

### Finding 27 — The dominant Phase B1 overcounting was a dataset-domain problem, not a model-capacity problem

The old S/B training mixture suggested a structural failure: all tested architectures overcounted
1-target windows as 2-target windows at the 57–72% level.  After the G-dataset production,
explicit overlap-domain features, and hard-negative training samples, the same EdgeCompat h64
architecture reduces clean 1-target overcounting to 0.14% and PU 1-target overcounting to 1.2%.

This shows that the dominant Phase B1 failure was caused by ambiguity between full-barrel KMTF
stubs and the OMTF-overlap target definition.  Once the task domain is made explicit, standard
EdgeCompat and even DeepSets models learn the correct 1-target multiplicity.

The remaining model problem is no longer 1-target overcounting, but multi-target recovery and
hard-negative rejection.

### Finding 28 — DETR has the best multiplicity counting but needs stronger no-object calibration

DETR achieves the best multiplicity recovery on G5/G6, reaching 70.2% for 2-target windows and
97.3% for 3-target windows.  However, with `w_no_obj=0.1`, it produces unacceptable false
positives on B4/G7/G8.  This indicates that the permutation-invariant matching objective is
useful, but the no-object class is underweighted.  A dedicated `w_no_obj` sweep is required
before accepting or rejecting DETR.

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
