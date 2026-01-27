# Figures and Citations Guide for IEEE RT 2026 Submission

## Figures Included in Supporting Document

### Figure 1: Synthesis Results and DSE Outcomes (Side-by-side)

**Left panel**: `build/plots/hls_comparison.png`
- Shows Float vs PTQ-INT8 HLS synthesis comparison
- Resource utilization (DSP, FF, LUT) and latency metrics
- Demonstrates actual hardware implementation results on Xilinx VU13P
- **Why this figure**: Provides concrete evidence of successful FPGA synthesis

**Right panel**: `build/plots/pareto/pareto_accuracy_vs_dsp_used.png`
- Pareto frontier showing accuracy vs DSP resource trade-off
- Shows design space exploration results
- Currently has 1 point, will expand with ongoing sweeps
- **Why this figure**: Demonstrates automated DSE methodology

## Alternative Figure Options

If you want different figures, here are good alternatives:

### Option A: Single Wide Figure
- **`build/plots/hls_comparison.png`** (full width)
  - Most comprehensive synthesis results
  - Shows both implementations side-by-side

### Option B: 3D Pareto Front
- **`build/plots/pareto/pareto_3d_accuracy_latency_dsp.png`**
  - Shows all three objectives simultaneously
  - More visually striking but harder to read precise values

### Option C: Model Comparison
- **`build/plots/model_comparison.png`**
  - Shows accuracy, parameters, and memory across all model variants
  - Good for showing the full pipeline from base to PTQ-INT8

### Option D: Training Curves
- **`build/plots/base_model_training.png`** or **`reduced_model_no_root_training.png`**
  - Shows training/validation curves
  - Less relevant for hardware focus

## Citations Included

The document now includes 4 key references from your previous proceedings:

1. **Hamilton et al. 2017** (GraphSAGE paper)
   - Original GraphSAGE algorithm
   - Cited when mentioning Pareto analysis and GNN architecture

2. **CMS Phase-2 TDR (2020)**
   - Phase-2 Level-1 Trigger upgrade specifications
   - Cited for 12.5 μs latency budget

3. **OMTF JINST (2016)**
   - Overlap Muon Track Finder description
   - Cited for overlap region context

4. **CMS Trigger JINST (2017)**
   - CMS trigger system overview
   - Available if needed for additional trigger context

## How the Citations Are Used

**In Motivation section**:
- CMS_PHASE2_TDR → 12.5 μs latency budget
- OMTF_JINST → overlap muon track finder region

**In DSE section**:
- Hamilton2017 → GraphSAGE algorithm reference

**Available but not yet used**:
- CMS_TRIGGER → General trigger system (can add if needed)

## Figure Placement Strategy

The figures are placed after the DSE section because:
1. They provide visual summary of key results
2. Reader has context from tables and text
3. Shows both synthesis results AND DSE methodology
4. Doesn't interrupt flow of technical narrative

## Recommended Figure Caption

Current caption:
> "Synthesis results and design-space exploration outcomes demonstrating the complete workflow from training to hardware validation."

This ties both figures together and emphasizes the end-to-end nature of your work.

## If You Want to Change Figures

To change the figures in the LaTeX document:

1. **Edit lines 118 and 124** in `ieee_rt2026_supporting_doc.tex`
2. Replace the file paths with your preferred plots
3. Update the subcaptions to match the new figures

Example:
```latex
\includegraphics[width=\textwidth]{../../build/plots/YOUR_CHOSEN_PLOT.png}
```

## Figure Quality Notes

All plots are PNG format with good resolution:
- hls_comparison.png: 268K (high quality)
- pareto plots: ~55K each (clean, publication-ready)
- All generated from matplotlib with professional styling

## Two-Page Fit Strategy

The current document is designed to fit 2 pages by:
1. Tight margins (0.75in)
2. Negative vertical spacing between sections
3. Compact tables with small font
4. Side-by-side subfigures (saves vertical space)
5. Concise but complete text

**Estimated page count**: ~1.8-2.0 pages (within limit)

## Next Steps

1. Review the figures in the compiled PDF
2. If you want different figures, let me know which plots to use
3. Can adjust figure sizes or layout if needed for space
4. Can add/remove references as needed
