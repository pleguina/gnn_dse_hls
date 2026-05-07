# Safe next steps to improve the GMT/OMTF model before FPGA implementation

## Current starting point

We are starting from the current best physics baseline:

* **Primary input view:** TPS / EMTF-hybrid stubs.
* **Primary model:** EdgeCompatNet.
* **Best stable checkpoint so far:** TPS-h64 EdgeCompat.
* **Alternative checkpoint:** TPS-h128 best checkpoint, useful for background rejection but less stable during training.
* **Main conclusion so far:** TPS clearly outperforms KMTF for this task, especially on G8 hard-negative rejection and displaced-muon efficiency.

The goal now is **not** to start FPGA implementation yet.

The goal is to make the floating-point ML baseline as clean, stable, and defensible as possible before entering:

1. quantization-aware training,
2. fixed-point studies,
3. HLS/RTL implementation,
4. board integration.

---

## 1. Document the current findings first

Before running more experiments, freeze the current state in a short report.

Document:

* why TPS became the primary input view,
* why KMTF is now the secondary comparison,
* why EdgeCompatNet is preferred over DeepSets and DETR,
* the current best TPS-h64 and TPS-h128 results,
* the remaining weaknesses:

  * G5 second-muon recovery,
  * G6 third-muon recovery,
  * G7/G8 hard-negative false positives,
  * pT regression quality,
  * h128 training instability.

The report should include one compact table:

| Model     | G1 eff | G2 eff | G3 eff | G4 eff | G5 eff | G6 eff | G7 FP | G8 FP | B4 FP |
| --------- | -----: | -----: | -----: | -----: | -----: | -----: | ----: | ----: | ----: |
| KMTF-h128 |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |
| TPS-h64   |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |
| TPS-h128  |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |

Decision to write explicitly:

> TPS-h64 EdgeCompat is the current primary baseline because it gives the best balance between signal efficiency, background rejection, stability, and implementation simplicity.

---

## 2. Freeze the evaluation protocol

Before improving the model, make sure every new run is evaluated in exactly the same way.

Use the same:

* cache version,
* dataset list,
* train/validation split seed,
* threshold scan,
* event-level evaluation,
* pos/neg split evaluation,
* G7/G8/B4 background metrics,
* pT and d0 binning.

Recommended fixed evaluation datasets:

```bash
G1 G2 G3 G4 G5 G6 G7 G8 B4
```

Because `expand_datasets()` expands logical names, this should evaluate:

```bash
G1_pos G1_neg
G2_pos G2_neg
G3_pos G3_neg
G4_pos G4_neg
G5_pos G5_neg
G6_pos G6_neg
G7
G8
B4
```

Do not compare new models using only validation loss.

The model selection should be based on physics metrics:

* G1/G2 prompt efficiency,
* G3/G4 displaced efficiency,
* G5/G6 multi-candidate recovery,
* G7 hard-negative false positive rate,
* G8 PU200 hard-negative false positive rate,
* B4 pure-noise false positive rate,
* pT regression quality,
* training stability.

---

## 3. Run threshold optimization on the current TPS-h64 baseline

Before changing the model, scan the operating threshold.

The current reports mostly use:

```text
candidate logit > 0.0
```

But the trigger does not care that the threshold is exactly zero. The trigger cares about the operating point.

Run evaluation at:

```text
threshold = -0.5, 0.0, 0.25, 0.5, 0.75, 1.0
```

Selection rule:

* Keep B4 FP at 0%.
* Keep G8 FP low.
* Keep G7 FP low.
* Maximize G2/G4/G5/G6 efficiency.

Expected outcome:

* A slightly positive threshold may reduce G7/G8 FP.
* It may cost some G1/G2/G3/G4 efficiency.
* The best trigger operating point may not be threshold 0.0.

Deliverable:

```text
build/omtf_gmt/eval/tps_h64_threshold_scan_summary.md
```

---

## 4. Add a hard-negative candidate loss

The current model already learns G7/G8 as zero-candidate samples, but we can make this more explicit.

Use `meta_is_hard_neg`.

For samples where:

```python
meta_is_hard_neg == 1
```

add an auxiliary loss that pushes all candidate logits negative:

```python
hard_neg_target = torch.zeros_like(candidate_logits[hard_neg_mask])
hard_neg_loss = F.binary_cross_entropy_with_logits(
    candidate_logits[hard_neg_mask],
    hard_neg_target,
)
```

Then add it to the total loss:

```python
total_loss = total_loss + w_hard_neg * hard_neg_loss
```

Recommended sweep:

```text
w_hard_neg = 0.25, 0.5, 1.0
```

Use TPS-h64 first.

Do not oversweep. This is a small targeted test.

Expected effect:

* lower G7/G8 FP,
* little or no effect on B4,
* possible small loss in signal efficiency if the weight is too high.

Decision rule:

* Accept only if G7/G8 FP improves without hurting G2/G4/G5/G6 efficiency by more than about 1–2 percentage points.

---

## 5. Add truth-source-weighted node loss

The cache now has `truth_source`.

Current node loss treats all valid stubs as either signal or noise. But not all noise-like stubs are equally reliable.

Use:

| truth_source | Meaning                        | Suggested node-loss weight |
| ------------ | ------------------------------ | -------------------------: |
| 2            | confirmed OMTF signal transfer |                        1.0 |
| 1            | confirmed OMTF noise           |                        1.0 |
| 0            | unmatched / ambiguous          |                0.25 or 0.5 |

The idea is simple:

* confirmed signal should matter strongly,
* confirmed noise should matter strongly,
* unmatched stubs should still be seen by the model, but should not dominate the loss.

Implementation sketch:

```python
truth_source = batch["truth_source"]

node_weight = torch.ones_like(batch["node_label"])
node_weight = torch.where(truth_source == 0, 0.25 * node_weight, node_weight)
node_weight = node_weight * batch["valid_mask"].float()

node_loss_raw = F.binary_cross_entropy_with_logits(
    node_logit,
    node_label,
    reduction="none",
)

node_loss = (node_loss_raw * node_weight).sum() / node_weight.sum().clamp(min=1.0)
```

Recommended sweep:

```text
unmatched weight = 0.25, 0.5, 1.0
```

Where `1.0` is the current baseline.

Expected effect:

* better robustness in PU samples,
* lower G8 FP,
* possibly improved TPS stability,
* possibly less overfitting to unmatched PU stubs.

Decision rule:

* Keep the setting that improves G8/G7 without reducing G3/G4 displaced efficiency.

---

## 6. Combine the two safe loss improvements

After testing both independently, run the best combination:

```text
TPS-h64 EdgeCompat
+ best hard-negative candidate loss
+ best truth-source node weighting
```

Example candidate:

```text
w_hard_neg = 0.5
unmatched_weight = 0.25
```

This should be treated as the main improved baseline candidate.

Compare it against the current TPS-h64 baseline.

Required comparison table:

| Model                  | G1 eff | G2 eff | G3 eff | G4 eff | G5 eff | G6 eff | G7 FP | G8 FP | B4 FP |
| ---------------------- | -----: | -----: | -----: | -----: | -----: | -----: | ----: | ----: | ----: |
| TPS-h64 baseline       |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |
| + hard-neg loss        |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |
| + truth-source weights |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |
| combined               |    ... |    ... |    ... |    ... |    ... |    ... |   ... |   ... |   ... |

---

## 7. Rerun TPS-h128 with a safer schedule

TPS-h128 gave better background rejection, but training became unstable after the best checkpoint.

Do not discard h128 yet.

Rerun h128 with a safer setup:

* fewer epochs,
* lower learning rate,
* no aggressive cosine tail,
* frequent checkpointing,
* select by physics metrics, not only validation loss.

Recommended run:

```bash
python src/omtf_gmt/train.py \
  --cache-dir build/omtf_gmt/cache_tps_v2 \
  --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
  --repeat G7:4 G8:4 B4:6 \
  --model edge_compat \
  --hidden 128 \
  --epochs 50 \
  --batch-size 4096 \
  --lr 5e-4 \
  --amp \
  --output-dir build/omtf_gmt/checkpoints/edge_compat_h128_tps_safe50
```

Expected outcome:

* if stable, h128 may become the best background-rejection checkpoint,
* if still unstable or not clearly better, keep h64.

Decision rule:

* h128 must improve G7/G8 or G5/G6 without losing too much G1/G2/G3/G4 efficiency.
* If the improvement is small, prefer h64 because it is cheaper and more stable.

---

## 8. Try one intermediate hidden size: h96

h64 is stable.
h128 has more capacity but showed instability.

Try:

```text
hidden = 96
```

This is a safe compromise before choosing the final pre-QAT model.

Recommended run:

```bash
python src/omtf_gmt/train.py \
  --cache-dir build/omtf_gmt/cache_tps_v2 \
  --datasets G1 G2 G3 G4 G5 G6 G7 G8 B4 \
  --repeat G7:4 G8:4 B4:6 \
  --model edge_compat \
  --hidden 96 \
  --epochs 50 \
  --batch-size 4096 \
  --lr 1e-3 \
  --amp \
  --output-dir build/omtf_gmt/checkpoints/edge_compat_h96_tps_B5a
```

If h96 improves G5/G6 or G7/G8 without instability, it may be a better final candidate than h64.

Decision rule:

* h96 must clearly improve physics metrics.
* If h96 only gives a tiny gain, keep h64 for implementation simplicity.

---

## 9. Do not start transformers or DETR now

Do not open a new large architecture search yet.

Current conclusion:

* DeepSets is weaker than EdgeCompat for multi-candidate recovery.
* DETR is interesting but more complex and less operationally clean.
* Transformers are likely too expensive for the current hardware-oriented path.

Keep the model family fixed:

```text
EdgeCompatNet + TPS input view
```

The safe improvement space is now:

* better loss weighting,
* better thresholding,
* h64/h96/h128 capacity check,
* stable training schedule.

---

## 10. Final pre-QAT model selection rule

After the safe runs, choose one final floating-point model for QAT.

The selected model must satisfy:

* B4 FP = 0%,
* G8 FP clearly below KMTF baseline,
* G7 FP acceptable,
* G1/G2 prompt efficiency high,
* G3/G4 displaced efficiency high,
* G5/G6 multi-candidate recovery not worse than current TPS-h64,
* stable training,
* reasonable model size.

Recommended ranking criteria:

1. Physics behavior.
2. Stability.
3. Simplicity.
4. Parameter count.
5. Validation loss.

Validation loss is useful, but it should not decide the model alone.

---

## Immediate run list

Run these in order.

### Run 1 — threshold scan on current TPS-h64

Purpose:

* find best operating threshold without retraining.

Output:

```text
build/omtf_gmt/eval/tps_h64_threshold_scan_summary.md
```

**Status: complete.**

### Run 2 — TPS-h64 + hard-negative loss

Sweep:

```text
w_hard_neg = 0.25, 0.5, 1.0
```

Purpose:

* reduce G7/G8 false positives.

**Status: complete.** HTCondor job 1075014.0–2.

### Run 3 — TPS-h64 + truth-source node weighting

Sweep:

```text
unmatched_weight = 0.25, 0.5
```

Purpose:

* reduce PU confusion from unmatched TPS stubs.

**Status: complete.** HTCondor job 1075014.3–4.

### Run 4 — TPS-h64 combined loss

Use best settings from runs 2 and 3.

Purpose:

* create the improved h64 candidate.

**Status: not needed.** Threshold scan showed hn025 @ 0.0 already dominates baseline at all equal-FP operating points. See [operating point comparison](../../build/omtf_gmt/eval/tps_h64_operating_point_comparison.md).

### Run 5 — TPS-h96 baseline

Purpose:

* test whether modest capacity increase improves G5/G6 or G7/G8.

**Status: complete, rejected.** HTCondor job 1075014.5. Best epoch at ep=11 (premature convergence), G7 FP higher than baseline (8.7% vs 6.2%), last epoch overfit (B4 FP = 3.1%).

### Run 6 — TPS-h128 safe schedule

Purpose:

* check whether h128 can be made stable enough to beat h64/h96.

**Status: complete, rejected.** HTCondor job 1075014.6. Best epoch at ep=29, physics metrics equivalent to h64 baseline, last epoch overfit (B4 FP = 2.3%). Instability persists.

---

## Phase B5 results (2026-05-07)

### B5 sweep summary

HTCondor cluster 1075014, 7 jobs, all completed.

| Job | Run | Best epoch | G7 FP% | G8 FP% | B4 FP% | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1075014.0 | h64_hn025 (w=0.25) | 80 | 3.6 | 2.2 | 0.0 | **selected** |
| 1075014.1 | h64_hn050 (w=0.50) | 33 | 3.3 | 2.0 | 0.0 | rejected (−5 pp G2/G4) |
| 1075014.2 | h64_hn100 (w=1.00) | 38 | 1.6 | 1.5 | 0.0 | rejected (−10 pp G2/G4) |
| 1075014.3 | h64_usw025 (uw=0.25) | 29 | 7.2 | 3.6 | 0.0 | rejected (worsens G7/G8) |
| 1075014.4 | h64_usw050 (uw=0.50) | 91 | 5.5 | 2.8 | 0.0 | rejected (marginal) |
| 1075014.5 | h96_B5a | 11 | 8.7 | 2.9 | 0.0 | rejected (see Run 5) |
| 1075014.6 | h128_safe50 | 29 | 6.4 | 2.7 | 0.0 | rejected (see Run 6) |

Baseline reference (h64_B4_tps_100ep, ep=92): G7=6.2%, G8=2.9%, B4=0.0%.

### Threshold scan — operating point comparison

The threshold scan was the deciding analysis. At equal FP budget:

| Equal-FP target | Baseline config | hn025 config | G2 gain | G4 gain | G5 gain |
| --- | --- | --- | ---: | ---: | ---: |
| G7 = 3.6% | baseline @ +0.50 | hn025 @ +0.00 | +4.7 pp | +4.0 pp | +4.1 pp |
| G8 = 2.4% | baseline @ +0.25 | hn025 @ +0.00 | +0.3 pp | −0.4 pp | +0.1 pp |

hn025 @ 0.0 strictly dominates the baseline at any threshold when evaluated at equal FP budget.
The baseline at threshold +0.50 achieves the same G7 FP as hn025 @ 0.0 but loses 4–5 pp on all signal groups.

### Phase B5 decision

**Selected floating-point model:** `h64_hn025` — EdgeCompat h64, w_hard_neg=0.25, threshold 0.0.

**Reference model (frozen, not entering QAT):** `h64_baseline` — EdgeCompat h64, no hard-neg loss, threshold 0.0.

### Output documents

| Document | Path |
| --- | --- |
| Phase B5 final evaluation report | [`build/omtf_gmt/eval/phase_b5_final_report.md`](../../build/omtf_gmt/eval/phase_b5_final_report.md) |
| Operating point comparison (threshold scan) | [`build/omtf_gmt/eval/tps_h64_operating_point_comparison.md`](../../build/omtf_gmt/eval/tps_h64_operating_point_comparison.md) |
| Baseline threshold scan detail | [`build/omtf_gmt/eval/tps_h64_threshold_scan_summary.md`](../../build/omtf_gmt/eval/tps_h64_threshold_scan_summary.md) |
| hn025 threshold scan detail | [`build/omtf_gmt/eval/tps_h64_hn025_threshold_scan_summary.md`](../../build/omtf_gmt/eval/tps_h64_hn025_threshold_scan_summary.md) |
| hn025 eval JSON | [`build/omtf_gmt/eval/edge_compat_h64_tps_hn025_best_eval.json`](../../build/omtf_gmt/eval/edge_compat_h64_tps_hn025_best_eval.json) |
| baseline eval JSON | [`build/omtf_gmt/eval/edge_compat_h64_B4_tps_100ep_best_eval.json`](../../build/omtf_gmt/eval/edge_compat_h64_B4_tps_100ep_best_eval.json) |
| hn025 freeze manifest | [`build/omtf_gmt/checkpoints/edge_compat_h64_tps_hn025/freeze_manifest.json`](../../build/omtf_gmt/checkpoints/edge_compat_h64_tps_hn025/freeze_manifest.json) |
| baseline freeze manifest | [`build/omtf_gmt/checkpoints/edge_compat_h64_B4_tps_100ep/freeze_manifest.json`](../../build/omtf_gmt/checkpoints/edge_compat_h64_B4_tps_100ep/freeze_manifest.json) |

---

## Expected final outcome

At the end of this phase, we should have:

1. a frozen TPS-h64 reference baseline,
2. an improved TPS-h64 loss-weighted model,
3. one h96 comparison,
4. one stable h128 comparison,
5. a final selected floating-point checkpoint,
6. a clean report justifying why that checkpoint enters QAT.

**Status: all six items complete.**

Only after that should we move to:

```text
QAT → fixed-point evaluation → HLS/RTL feasibility → board implementation
```

**Next step: QAT on `h64_hn025`.**
