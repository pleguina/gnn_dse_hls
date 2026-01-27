# IEEE Real-Time 2026 — Supporting Document (max 2 pages) — Draft Outline

## Proposed title
Design-Space Exploration and Integer Quantization of Graph Neural Networks for Real-Time FPGA Track Finding

## 1. Motivation and context (3–5 sentences)
- Fixed-latency, high-throughput trigger/DAQ constraints.
- Displaced-muon efficiency motivates learning-based approaches.
- Why GNNs (irregular geometry, sparsity, variable occupancy).

## 2. What this work contributes (bullet list)
- End-to-end pipeline: training → reduction → subgraph extraction → quantization → integer test vectors → HLS projects → report parsing/visualization.
- DSE over algorithm + quantization + HLS pragmas to expose accuracy/latency/resource trade-offs.
- Integer-only INT8 HLS design with fixed-point scaling + bit-exact validation vs Python.

Optional phrasing note (avoid over-claiming novelty):
- Message-passing GNN layers are less directly supported by many automated FPGA ML flows (which often focus on dense/CNN layers), so aggregation/data-movement must be handled explicitly for HLS.

## 3. Method summary (one figure + short text)
**Figure 1 (recommended):** Pipeline block diagram
- PyTorch Geometric GraphSAGE model
- PTQ / QAT branches
- Test-vector generation
- Vitis HLS C-sim and synthesis
- Automated report parsing + Pareto analysis

## 4. Quantization and numeric design (short technical section)
- PTQ-Float vs PTQ-INT8 distinction.
- Fixed-point scaling rule used in integer path (M fractional bits) and rounding.
- Data-driven bit-width optimization and how it reduces DSP/LUT pressure (include a small table).

## 5. Hardware implementation approach (short technical section)
- Modular C++ layers mirroring Python (aggregation, linear projections, activations).
- Notes on loop unrolling vs pipelining and resource/latency implications.

## 6. Representative results (one plot or table)
Choose one of:
- **Table:** Accuracy (proxy dataset) + max numerical error (LSB) for Float / PTQ-Float / PTQ-INT8.
- **Plot:** Pareto scatter (latency vs DSP or accuracy drop vs DSP) from DSE runs.

## 7. Relevance to displaced-muon track finding (closing paragraph)
- State explicitly: Cora used as a public proxy to validate toolchain.
- Describe planned transfer: define track-finder graph (nodes=stubs, edges=geometric relations), retrain on displaced-muon dataset, re-run DSE to meet timing/resource budgets.

## Appendix (optional, 2–3 lines)
- Link to repo structure and reproducibility notes: `run_pipeline.py`, `src/explore_design_space.py`, `hls/` projects.

---
## Quick PDF build options
- Easiest: paste into a word processor and export to PDF.
- Reproducible: convert this Markdown to PDF with `pandoc`.

Suggested `pandoc` command:
```bash
pandoc docs/notes/ieee_rt2026_supporting_doc.md -o build/ieee_rt2026_supporting_doc.pdf
```
