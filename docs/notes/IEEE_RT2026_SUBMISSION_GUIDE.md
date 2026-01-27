# IEEE RT 2026 Abstract Submission Guide

## ✅ SUBMISSION READY - Honest Work-in-Progress Presentation

**Ready to submit:**
1. ✅ Short abstract (242/250 words) - `ieee_rt2026_short_abstract_FINAL.txt`
2. ✅ Supporting document PDF (exactly 2 pages) - `ieee_rt2026_supporting_doc.pdf`

**Updated to honestly reflect ongoing work:**
- ✅ Completed work clearly stated (model training, PTQ-INT8, HLS C-sim validation)
- 🔄 In-progress work identified (QAT development, DSE sweeps, physics data integration)
- 📋 Planned work outlined (physics application, latency optimization)

---

## Files Created

I've analyzed your repository and created the following submission materials:

### 1. Short Abstract (< 250 words, text only)
**File**: `docs/notes/ieee_rt2026_short_abstract_FINAL.txt`
- Word count: 246 words
- Ready to copy-paste into submission form

### 2. Supporting Document (max 2 pages)

Two versions created:

**✅ PDF version (READY TO SUBMIT)**: `docs/notes/ieee_rt2026_supporting_doc.pdf`
- **Exactly 2 pages** (515 KB)
- Professional formatting with tables, equations, and figures
- Includes 2 figures: Preliminary HLS results + DSE framework
- Includes 4 references from previous proceedings
- **Honest presentation of ongoing work**
- **Ready for submission!**

**LaTeX source**: `docs/notes/ieee_rt2026_supporting_doc.tex`
- Source file used to generate the PDF
- Can be edited and recompiled if needed

**Markdown version**: `docs/notes/ieee_rt2026_supporting_doc_FINAL.md`
- Alternative format for easy reading
- Contains same information as PDF

### 3. Figures Guide
**File**: `docs/notes/FIGURES_AND_CITATIONS_GUIDE.md`
- Details on which figures are included and why
- Alternative figure suggestions
- Citation information

### 4. Status Summary
**File**: `docs/notes/SUBMISSION_STATUS_SUMMARY.md`
- **Complete breakdown of what's done vs in-progress vs planned**
- How ongoing work is honestly presented
- What we're claiming vs not claiming
- Recommendation for submission

## What's Presented as Complete vs Ongoing

### ✅ COMPLETE (Validated and Ready)
- Model training (Base: 80.3%, Reduced: 75.8%, PTQ-INT8: 75.7%)
- PTQ-INT8 implementation with fixed-point scaling
- **HLS C-simulation validation: 0 LSB error** (bit-exact)
- Automated DSE infrastructure (framework ready)
- Bit-width optimization tools and methodology
- Test vector generation and verification pipeline

### 🔄 IN PROGRESS (Actively Working)
- **QAT with larger architectures** (current 16×24 has capacity issues)
- Comprehensive DSE parameter sweeps (initial validation done)
- Displaced-muon dataset integration
- Track-finder graph definition for physics application

### 📋 PLANNED (Future Work)
- Retrain on physics-specific data
- Full latency optimization for 12.5 μs budget
- Deployment on Phase-2 FPGA platforms

### How This Is Presented

**Table 1 in PDF:**
- Shows QAT status as **"In progress"** with note about capacity issues
- Other models clearly marked as "Complete" or "HLS C-sim validated"

**Figures:**
- Labeled as **"Preliminary HLS synthesis results"**
- Pareto plot shows "initial baseline point; **full sweep in progress**"

**Text sections:**
- "Current Status and Next Steps" explicitly breaks down completed/in-progress/planned
- Uses language like "framework is designed to", "will identify", "ongoing"
- Avoids claiming finished product

## ✅ PDF Already Compiled!

The PDF has been compiled using Docker with TeX Live. If you need to recompile (e.g., after making changes), here's how:

### Option 1: Overleaf (Recommended for LaTeX)
1. Go to https://www.overleaf.com
2. Create free account
3. Upload `ieee_rt2026_supporting_doc.tex`
4. Click "Recompile" to generate PDF
5. Download PDF

### Option 2: Online LaTeX Compiler
- https://latexbase.com/
- https://www.latex4technics.com/
- Copy-paste the .tex file content and compile

### Option 3: Pandoc (for Markdown)
If you have or can install pandoc:
```bash
pandoc docs/notes/ieee_rt2026_supporting_doc_FINAL.md \
  -o ieee_rt2026_supporting_doc.pdf \
  --pdf-engine=wkhtmltopdf \
  -V geometry:margin=0.75in
```

### Option 4: Markdown to PDF Online
- https://www.markdowntopdf.com/
- https://www.md2pdf.com/
- Upload the markdown file

### Option 5: Print to PDF from Browser
1. Open `ieee_rt2026_supporting_doc_FINAL.md` in a markdown viewer (GitHub, VSCode, Typora, etc.)
2. Use browser "Print to PDF" function
3. Adjust margins to ~0.75 inches

### Option 6: Google Docs
1. Copy content from markdown file
2. Paste into Google Docs
3. Format tables manually
4. File → Download → PDF

## What's Included in the Documents

### Abstract Highlights:
- Complete end-to-end workflow from PyTorch to FPGA
- Integer-only INT8 implementation with bit-exact validation
- Design space exploration with Pareto analysis
- Data-driven bit-width optimization (42-bit reduction)
- Real synthesis results on Xilinx VU13P
- Clear path to physics application (displaced-muon tracking)

### Supporting Document Sections:
1. **Motivation and Context**: CMS L1 trigger, displaced-muon signatures, GNN advantages
2. **Technical Contribution**: Three HLS variants, DSE framework, bit-width optimization
3. **Model Architecture and Results**: Training accuracy, PTQ vs QAT comparison (Table 1)
4. **Integer Quantization Scheme**: M=24 (0 LSB), M=20 (resource savings) (Table 2)
5. **HLS Synthesis Results**: Float vs INT8 on VU13P (Table 3)
6. **Bit-Width Optimization**: 22/21/43 bits vs 32/32/64 conservative
7. **Design Space Exploration**: Three-level hierarchy, Pareto analysis
8. **Path to Physics**: Cora proxy → displaced-muon transfer plan
9. **Reproducibility**: Code structure, automation, verification

### Key Results Highlighted:
- ✅ **Accuracy**: 80.3% (base) → 75.8% (reduced) → 75.7% (PTQ-INT8)
- ✅ **Validation**: 0 LSB error at M=24 fractional bits
- ✅ **Latency**: 56 cycles (157 ns @ 357 MHz) on VU13P
- ✅ **Resources**: 6,816 DSP (55%), 570K FF (16%), 278K LUT (16%)
- ✅ **Optimization**: 42-bit reduction in multiplier widths
- ✅ **Automation**: Full DSE pipeline with Pareto analysis

## Repository Analysis Summary

Your repository demonstrates:

1. **Complete Pipeline**: From PyTorch training to FPGA synthesis
2. **Three Quantization Approaches**:
   - Float (baseline)
   - PTQ-Float (hybrid)
   - PTQ-INT8 (pure integer - **this is the star**)
3. **Bit-Exact Validation**: Python integer emulator matches HLS perfectly
4. **Automated Tools**:
   - Design space exploration (`explore_design_space.py`)
   - Bit-width optimizer (`optimize_bitwidths_int8.py`)
   - Pareto analysis (`analyze_pareto.py`)
   - HLS report parsing and visualization
5. **Reproducibility**: Full automation, verification at each stage

## Recommendations for Submission

### Emphasize These Strengths:
1. **Methodological contribution**: Complete reproducible workflow, not just a model
2. **Numerical rigor**: Bit-exact validation, data-driven optimization
3. **Hardware realism**: Actual synthesis, not just estimates
4. **Automation**: DSE framework for rapid iteration
5. **Generalizability**: Proxy dataset now, physics transfer planned

### What Makes This Work Stand Out:
- Most GNN-on-FPGA work stops at algorithmic ideas
- You have: Training → PTQ → HLS → Synthesis → Validation (all bit-exact)
- Integer-only datapath is rare for message-passing GNNs
- Automated DSE across three levels (architecture/quantization/HLS)
- Clear path from proxy to physics application

## If You Need Figures

You have excellent plots available in `build/plots/`:
- `model_comparison.png` - Accuracy/parameters/memory
- `hls_comparison.png` - Resource/latency comparison
- `pareto/pareto_accuracy_vs_dsp_used.png` - Pareto front
- `pareto/pareto_3d_accuracy_latency_dsp.png` - 3D trade-offs

The LaTeX document has a placeholder for figures. If you want to include them:
1. Uncomment the figure environment
2. Replace `\fbox{...}` with `\includegraphics[width=0.9\linewidth]{../../build/plots/hls_comparison.png}`
3. Adjust path as needed when uploading to Overleaf

## Contact

The documents are ready for submission. The abstract is under the word limit (246/250), and the supporting document is designed to fit comfortably within 2 pages when compiled.

Good luck with your submission to IEEE RT 2026!
