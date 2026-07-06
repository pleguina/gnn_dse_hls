# OMTF Edge GNN — Displaced Muon Branch

GNN-based muon reconstruction for the OMTF (Overlap Muon Track Finder) trigger.

## Branch Purpose

This branch (`parallel_displaced_lane`) develops a **parallel, non-competing** line alongside the main OMTF prompt muon algorithm. The focus is on **displaced muon signatures** — muons with large impact parameters that are poorly reconstructed by the standard prompt trigger. The goal is to run in parallel with the prompt algorithm and recover displaced tracks without degrading prompt performance.

## Structure

```
src/
├── omtf/          # OMTF edge-GNN models, dataset, training, evaluation
└── omtf_gmt/      # OMTF-GMT multi-muon slot models, dataset, evaluation

scripts/
├── omtf/          # HTCondor submission, build/eval scripts for OMTF
└── omtf_gmt/      # HTCondor submission, build/eval scripts for OMTF-GMT

src/audit/         # Dataset audit and variable checks
scripts/audit/     # Audit runner

docs/
├── omtf/          # OMTF-specific design decisions and studies
├── omtf_gmt/      # OMTF-GMT architecture comparisons and results
├── displaced/     # Displaced muon variable analysis
├── OMTF_GNN_Topology_Analysis.md
└── OMTF_MODEL_STUDY.md

build/
├── omtf/          # OMTF training outputs and caches
├── omtf_gmt/      # OMTF-GMT training outputs
└── condor/        # HTCondor job outputs

data/              # Input ROOT files (not tracked)
```

## Requirements

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Dependencies: PyTorch, PyTorch Geometric, uproot, numpy, scipy, matplotlib, seaborn, pandas, scikit-learn.
