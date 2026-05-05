# Phase B1 Training Mix Decision

Analysis date: 2026-04-25
Source: `scripts/omtf_gmt/analyze_training_mix.py` on all 9 datasets
Full output: `build/omtf_gmt/TRAINING_MIX_ANALYSIS.md`

---

## Key finding

The dataset builder creates windows with an exact, homogeneous candidate count per dataset:

| Dataset | Candidate count | Character |
| --- | --- | --- |
| S2 | 1-cand only | Clean single-muon (no PU) |
| B2 | 1-cand only | Single-muon under heavy PU |
| S1 | 2-cand only | DY dimuon, moderate noise |
| S3 | 2-cand only | Close dimuon, clean |
| S5 | 2-cand only | 2-muon, moderate noise |
| B1 | 2-cand only | 2-muon under heavy PU |
| B3 | 2-cand only | 2-muon under moderate PU |
| S4 | 3-cand only | **Sole source of slot-2 positive targets** |
| B4 | 0-cand only | **Sole source of all-zero cand_target** |

This means dataset selection directly controls which candidate-head slots receive positive
training signal. Missing a dataset kills the corresponding regime entirely.

---

## What not to do

**Do not use S1+S3+B1+B4 or any mix that excludes S2, B2, or S4.**

| Problem | Root cause | Consequence |
| --- | --- | --- |
| S2 and B2 excluded | Only 1-cand sources missing | Model never trains on single-muon windows; slot 0 fires only when slot 1 also fires |
| S4 excluded | Only 3-cand source missing | Slot 2 receives zero positive targets; learns to always output "no third candidate" silently |
| B4 at 3.5% of mix | Only 0-cand source severely underweighted | False-positive suppression untrained; head overfires on noise |

**Do not rely only on loss weighting to fix a structural supervision gap.**

Loss weights shift the gradient magnitude. They cannot restore training signal that is never
seen. If a slot never has a positive target in a batch, `pos_weight` has no effect on that
slot's behaviour.

---

## Phase B1a — structural candidate-count learning

**Mix: all 9 datasets, B4 oversampled 8×.**

| Dataset | Repeat | Effective train samples | Fraction |
| --- | --- | --- | --- |
| S1 | ×1 | ~291,745 | 20.2% |
| S2 | ×1 | ~67,958 | 4.7% |
| S3 | ×1 | ~254,449 | 17.6% |
| S4 | ×1 | ~133,359 | 9.2% |
| S5 | ×1 | ~51,678 | 3.6% |
| B1 | ×1 | ~201,885 | 14.0% |
| B2 | ×1 | ~80,493 | 5.6% |
| B3 | ×1 | ~150,708 | 10.4% |
| B4 | ×8 | ~214,344 | **14.8%** |
| **Total** | | **~1,446,619** | |

Effective candidate-multiplicity distribution in the training pool:

| Candidate count | Fraction | Head slot(s) trained |
| --- | --- | --- |
| 0-cand | ~14.8% | All slots negative |
| 1-cand | ~10.3% | Slot 0 positive, slots 1–2 negative |
| 2-cand | ~65.7% | Slots 0–1 positive, slot 2 negative |
| 3-cand | ~9.2% | All slots positive |

All four regimes are alive. All three candidate-head slots receive both positive and negative
training signal.

**Goals of Phase B1a:**

- all slots trained (no dead slot)
- model learns 0/1/2/3-candidate regimes properly
- no-candidate behaviour established with adequate frequency

---

## Phase B1b — realism refinement

After the first stable model from Phase B1a:

- keep all 9 datasets
- reduce B4 oversampling to ×4 (puts 0-cand at ~7–8%)
- verify false-positive rate stays low without the aggressive oversampling
- tune only if val metrics show regression on signal recovery or false-positive increase

This gives a more realistic sample distribution while testing whether the Phase B1a
structural learning is durable.

---

## Labeling policy — ambiguous stubs

Ambiguous-of-signal fraction across all datasets: 1.7–3.1%.

Decision for Phase B1: **no masking of ambiguous stubs from node_loss**.

Rationale: the ambiguous fraction is below 5% in every dataset. Adding a mask would
complicate the pipeline without a measurable benefit at this noise level. Revisit if
the edge-compatibility model shows degraded performance on ambiguous-stub windows.

---

## Implementation

Per-dataset repeat factors are specified via `--repeat DS:N` in `src/omtf_gmt/train.py`.

The implementation uses `torch.utils.data.WeightedRandomSampler` with `replacement=True`,
not literal dataset duplication.  Each sample in dataset D receives weight = repeat_factor(D).
The sampler draws `n_train_eff` samples independently from this weighted distribution, so:

- High-weight datasets appear more often on average
- Draws are i.i.d. — no block patterns, no repeated chunks in fixed order
- Batches are well mixed across all datasets throughout every epoch

`num_samples` is set to the same total as the equivalent repeated dataset so epoch length
and wall-clock training time are unchanged.

Standard Phase B1a invocation:

```bash
python src/omtf_gmt/train.py \
    --cache-dir build/omtf_gmt/cache \
    --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \
    --repeat B4:8 \
    --model deepsets \
    --hidden 64 \
    --epochs 50 \
    --output-dir build/omtf_gmt/checkpoints/deepsets_B1a
```
