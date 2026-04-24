# Quick Start Guide (OMTF Migration Branch)

This branch is for the OMTF ML-to-firmware migration workflow described in:

- `docs/omtf/MIGRATION_PLAN.md`
- `docs/omtf/OMTF_MODEL_STUDY.md`
- `docs/omtf/MIGRATION_STATUS.md`

The legacy Cora/GraphSAGE pipeline is still available, but it is secondary in this branch.

## Branch intent (read first)

- OMTF development is additive: legacy files remain in place until migration gates are complete.
- OMTF code lives under `src/omtf/` and is the default path for new work.
- Current active stage is Stage 3 baselines (DeepSets and EdgeMLP), with full training runs pending.

## Prerequisites

- Python 3.9+
- Access to OMTF ROOT datasets (local or Lustre)
- `nvidia-smi` available if you plan to run on GPU

## Installation

```bash
cd gnn_dse_hls
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip setuptools wheel
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install pyyaml
```

## Hardware and GPU readiness

This repository now includes a GPU readiness check:

```bash
./venv/bin/python scripts/verify_gpu_readiness.py
```

Current cluster findings on `mlui01`:

- GPUs detected: 2x Tesla P100 12 GB
- PyTorch in this env supports `sm_60` (P100 compatible)
- Remaining blocker for GPU training: GPU compute mode is `Prohibited`

Until compute mode is changed by admins, training will automatically fall back to CPU.

## Configure OMTF data root

OMTF training/evaluation entrypoints read data from `data/prod` by default.

Expected layout:

```text
data/prod/
    S1/omtf_hits_S1_*.root
    S2/...
    S3/...
    S4/...
    S5/...
    B1/...
    B2/...
    B3/...
    B4/...
```

If your datasets are in Lustre:

```bash
mkdir -p data
rm -rf data/prod
ln -s /lustre/ific.uv.es/ml/uovi156/data/prod data/prod
```

## OMTF baseline training (Stage 3)

Activate environment:

```bash
source venv/bin/activate
```

Quick smoke tests:

```bash
# Baseline A (DeepSets)
python src/omtf/train.py --model deepsets --datasets S1 B1 B4 --epochs 20 --max-files 3 --max-entries 200

# Baseline B (EdgeMLP)
python src/omtf/train.py --model edge_mlp --datasets S1 --epochs 20 --max-files 3 --max-entries 200
```

Recommended intermediate run from migration status:

```bash
python src/omtf/train.py --model deepsets --datasets S1 B1 B4 --epochs 50 --max-files 50
```

## HTCondor GPU training template (recommended for real runs)

For Artemisa, production GPU training should be submitted through HTCondor.

Template files added in this repo:

- `scripts/omtf/train_omtf_htcondor.sub`
- `scripts/omtf/run_omtf_train.sh`

Submit runs:

```bash
mkdir -p build/condor
condor_submit scripts/omtf/train_omtf_htcondor.sub
```

Monitor and manage jobs:

```bash
condor_q
condor_q -better-analyze <cluster_id>.<proc_id>
condor_rm <cluster_id>
```

How to customize runs:

- Edit the `queue ... from (...)` block in `scripts/omtf/train_omtf_htcondor.sub`
- Use comma-separated datasets (example: `S1,B1,B4`)
- Adjust `request_cpus`, `request_memory`, `request_gpus` to match experiment needs

Use `gpurun` only for short interactive validation on UI GPU.
Use HTCondor template for any training you want to keep/reproduce.

Outputs:

- Checkpoints: `build/omtf/checkpoints/*_best.pt`
- History: `build/omtf/checkpoints/*_history.json`

## OMTF evaluation

```bash
python src/omtf/eval.py \
    --checkpoint build/omtf/checkpoints/deepsets_best.pt \
    --model deepsets \
    --datasets S1 B4
```

For EdgeMLP:

```bash
python src/omtf/eval.py \
    --checkpoint build/omtf/checkpoints/edge_mlp_best.pt \
    --model edge_mlp \
    --datasets S1 B4
```

## What not to use as default in this branch

- `run_pipeline.py`
- `src/train.py`
- `src/quantization_ptq.py`
- `src/train_qat.py`

These remain valid for the legacy benchmark, but they are not the primary OMTF migration workflow.

## Legacy GraphSAGE path (kept for reference)

If you need the original Cora benchmark path:

```bash
python run_pipeline.py
```

Legacy model and quantization scripts remain functional, but new migration development should be done under `src/omtf/`.

## Troubleshooting

**`ModuleNotFoundError: yaml`**

```bash
./venv/bin/python -m pip install pyyaml
```

**No ROOT files found under `data/prod`**

Check dataset layout and symlink target. Expected file pattern is `omtf_hits_<DATASET>_*.root` inside each dataset folder.

**ROOT import/runtime errors**

OMTF train/eval paths use `uproot` fallback and do not require PyROOT for normal runs.
If you run audit scripts under `src/audit/`, PyROOT may still be required.

**CUDA appears available but training runs on CPU**

Run:

```bash
./venv/bin/python scripts/verify_gpu_readiness.py
```

If it reports compute mode `Prohibited`, request admin change to `Default` or `Exclusive Process`.
