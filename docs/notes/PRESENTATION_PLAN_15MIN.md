# IEEE RT 2026 Presentation Plan (15 minutes: 12 + 3 Q&A)

**Target Audience**: Experimental physicists, FPGA/trigger engineers, ML for HEP community  
**Conference**: IEEE Real-Time 2026 (25-29 May 2026)  
**Location**: La Biodola - Isola d'Elba, Italy

---

## Executive Summary

### What We Have (COMPLETE) ✅
- **Pipeline**: Full workflow from PyTorch training to HLS synthesis
- **Models**: Base (80.3%), Reduced (75.8%), PTQ-INT8 (75.7%)
- **Validation**: HLS C-simulation with **0 LSB error** (bit-exact)
- **Tools**: Automated DSE infrastructure, bit-width optimizer
- **Plots**: Training curves, Pareto fronts, resource utilization

### What's Missing (IN PROGRESS) 🔄
- **QAT**: Current 16×24 architecture shows capacity issues (28.2% acc) → Need to train 32×48
- **DSE Results**: Framework ready, but only 1 Pareto point (need full sweep)
- **Physics Data**: Cora used as proxy; displaced-muon dataset not yet integrated
- **Synthesis Sweep**: Only 2-3 configurations synthesized (need 10-20 for full Pareto)

### What Would Be Good to Have (NICE TO HAVE) 📋
- More Pareto points (different architectures: 16×24, 24×32, 32×48)
- Better QAT results with larger model
- Physics-motivated graph structure example
- Latency breakdown analysis
- Resource projections for CMS Phase-2 FPGAs

---

## Presentation Structure

### Timing Breakdown (12 minutes)
1. **Introduction & Motivation** (2 min)
2. **Technical Approach & Pipeline** (3 min)
3. **Results: Models & Quantization** (3 min)
4. **Results: HLS Implementation** (2.5 min)
5. **Design Space Exploration** (1.5 min)
6. **Conclusions & Next Steps** (30 sec)

### Q&A (3 minutes)
- Be prepared for questions on physics application, latency, scalability

---

## Detailed Slide Content

### SLIDE 1: Title Slide (15 sec)
**Content:**
- Title: "Design-Space Exploration and Integer Quantization of Graph Neural Networks for Real-Time FPGA Track Finding"
- Author: Pelayo Leguina
- Conference: IEEE Real-Time 2026
- Date: 25-29 May 2026

**Notes:**
- Keep simple, no need to read it
- Jump directly to motivation

---

### SLIDE 2: Motivation - Why GNNs for Track Finding? (1 min)

**Visual:**
- Left: Diagram of CMS detector (barrel + endcap overlap region)
- Right: Graph representation (nodes = stubs, edges = geometric compatibility)

**Key Points:**
- CMS Level-1 trigger upgrade: **12.5 μs fixed latency budget**
- Displaced-muon signatures from long-lived particles (LLP physics)
- Traditional pattern-based finders assume prompt tracks
- **GNNs naturally represent sparse, irregular detector geometry**
- Overlap region: barrel (DT) + endcap (CSC) + RPC → complex topology

**What to Say:**
> "The CMS Level-1 upgrade faces a critical challenge: finding displaced muon tracks in the overlap region under a strict 12.5 microsecond latency budget. Graph neural networks offer a natural representation where detector stubs become nodes and geometric compatibility defines edges."

**Existing Material:**
- ✅ Can create CMS detector diagram (simple schematic)
- ✅ Can create graph topology illustration
- 🔄 Need to add this slide (not in supporting doc)

---

### SLIDE 3: Technical Challenge - From ML to FPGA (45 sec)

**Visual:**
- Flow diagram: PyTorch → Quantization → HLS → FPGA
- Highlight challenges at each step

**Key Points:**
- **ML frameworks** (PyTorch Geometric) → floating-point, dynamic graphs
- **FPGA constraints** → fixed-latency, limited precision, DSP/LUT budgets
- **Message-passing GNNs** → irregular aggregation, not like CNN
- **Gap**: Automated ML tools (hls4ml, QKeras) target CNNs, not GNNs

**What to Say:**
> "The challenge is bridging the gap between dynamic floating-point GNN libraries and fixed-latency FPGA implementations. Unlike CNNs, message-passing GNNs have irregular aggregation patterns that automated tools don't handle well."

**Existing Material:**
- 🔄 Need to create this flow diagram
- ✅ Can use parts from supporting doc introduction

---

### SLIDE 4: Our Contribution - Complete Pipeline (30 sec)

**Visual:**
- High-level pipeline diagram with 5 main blocks:
  1. Training (PyTorch Geometric)
  2. Quantization (PTQ & QAT)
  3. HLS Implementation (3 variants)
  4. Validation (bit-exact)
  5. Design Space Exploration

**Key Points:**
- **End-to-end workflow**: Training → HLS validation
- **Three HLS variants**: Float, PTQ-Float, PTQ-INT8
- **Automated DSE**: Architecture × Quantization × Microarchitecture
- **Data-driven optimization**: Bit-width analysis

**What to Say:**
> "We present a complete, reproducible pipeline from PyTorch training to FPGA-ready HLS code, with automated design-space exploration across architectural and quantization parameters."

**Existing Material:**
- ✅ Pipeline diagram partially in supporting doc (can enhance)
- ✅ Repository structure in README

---

### SLIDE 5: Model Architecture & Dataset (30 sec)

**Visual:**
- Left: GraphSAGE layer diagram (feature proj + 2 message-passing layers)
- Right: Cora dataset stats + sample graph

**Key Points:**
- **Architecture**: Feature projection (1433→16) + 2 SAGEConv layers (16→24→7)
- **Dataset**: Cora citation network (proxy for method development)
  - 2,708 nodes, 7 classes
  - Publicly available, reproducible
- **Target application**: Displaced-muon track finder (in preparation)

**What to Say:**
> "We use a GraphSAGE architecture with two message-passing layers. For method development, we use the Cora citation network as a public proxy to stress-test the toolchain before applying it to physics data."

**Existing Material:**
- ✅ Architecture diagram can be created from model_base.py
- 🔄 Cora visualization exists but may need cleanup
- ✅ Stats in supporting doc Table 1

---

### SLIDE 6: Training Results (1 min)

**Visual:**
- **Table** (from supporting doc):
  | Model | Accuracy | Parameters | Memory |
  |-------|----------|------------|--------|
  | Base (float) | 80.3% | 184,391 | 0.70 MB |
  | Reduced (float) | 75.8% | 24,079 | 0.09 MB |
  | PTQ-INT8 | **75.7%** | 24,079 | 0.02 MB |
  | QAT-INT8 (16×24) | 28.2% | 23,527 | 0.02 MB |

- **Training curve plots** (2 subplots):
  - Base model convergence
  - Reduced model convergence

**Key Points:**
- Base model: 80.3% accuracy (184K parameters)
- Reduced model: 75.8% accuracy (24K parameters, **8× smaller**)
- **PTQ-INT8: only 0.1% degradation** from float reduced
- QAT shows capacity issues → **larger architecture needed** (32×48 in progress)

**What to Say:**
> "The reduced model maintains 75.8% accuracy with 8× fewer parameters. Post-training quantization to INT8 preserves accuracy with only 0.1% degradation. Current QAT results show the 16×24 architecture is too small—we're actively training a 32×48 variant."

**Existing Material:**
- ✅ Table in supporting doc (ready)
- ✅ Training curves: `build/plots/base_model_training.png`, `reduced_model_training.png`
- ✅ QAT notes in `docs/notes/QAT_TRAINING_NOTES.md`

---

### SLIDE 7: Integer Quantization Scheme (1.5 min)

**Visual:**
- **Diagram**: Show quantization flow for one linear layer
  - Input: x_q (INT8)
  - Weights: W_q (INT8)
  - Accumulator: acc (INT32)
  - Bias: b_q (INT32, pre-scaled)
  - Scaling: fixed-point multiply (M fractional bits)
  - Output: y_q (INT8)

- **Equation box**:
  ```
  acc = Σ(x_q[i] × W_q[i][o]) + b_q[o]
  y_q = clamp(⌊(acc × s_fp + 2^(M-1)) / 2^M⌋, -128, 127)
  ```

**Key Points:**
- **Symmetric INT8** quantization (weights + activations)
- **INT32 accumulators** (no overflow)
- **Biases pre-scaled** to accumulator domain (INT32)
- **Fixed-point rescaling**: M fractional bits (24 or 20)
- **Adjacency matrices**: Scaled by K=2^12, stored as INT16

**What to Say:**
> "We eliminate all floating-point operations from the datapath. Weights and activations use symmetric INT8 quantization. Biases are pre-scaled to the INT32 accumulator domain. Fixed-point rescaling with M fractional bits converts between layers. This scheme is purely integer—no floats in the critical path."

**Existing Material:**
- ✅ Equations in supporting doc
- 🔄 Need to create clear visual diagram
- ✅ Details in `next_steps_ptq_hls.txt`

---

### SLIDE 8: HLS C-Simulation Validation (1 min)

**Visual:**
- **Table** (from supporting doc):
  | Implementation | Arithmetic | Validation Target | Max Error |
  |----------------|-----------|-------------------|-----------|
  | Float | FP32 | Python float ref | 0 (reference) |
  | PTQ-Float | FP32 + INT8 | Python PTQ-Float | 0 LSB |
  | PTQ-INT8 (M=24) | INT8/INT32/INT16 | Python int emulator | **0 LSB** ✅ |
  | PTQ-INT8 (M=20) | INT8/INT32/INT16 | Python int emulator | ≤13 LSB |

- **Visual**: Screenshot or diagram showing testbench flow
  - Python generates test vectors
  - HLS C-simulation reads vectors
  - Outputs compared node-by-node

**Key Points:**
- **Bit-exact validation** at M=24 (0 LSB error)
- M=20 reduces multiplier width by 21 bits (≤13 LSB error)
- Test vectors: 8-node subgraphs from Cora
- **Gold standard**: Python integer-only emulator

**What to Say:**
> "HLS C-simulation achieves bit-exact validation with zero least-significant-bit error when using M=24 fractional bits. Reducing to M=20 narrows multipliers by 21 bits with acceptable error. This validates our integer-only scheme matches the Python reference exactly."

**Existing Material:**
- ✅ Table in supporting doc (ready)
- ✅ Test vectors in `build/test_vectors_ptq_int8/`
- ✅ Validation scripts in `tests/compare_hls_vs_python.py`

---

### SLIDE 9: HLS Synthesis Results (1.5 min)

**Visual:**
- **Table** (from supporting doc):
  | Implementation | DSP (%) | FF (%) | LUT (%) | Latency (cycles) | Clock (MHz) | Latency (ns) |
  |----------------|---------|--------|---------|------------------|-------------|--------------|
  | Float | 5,896 (48%) | 454K (13%) | 284K (16%) | 42 | 362 | 116 |
  | PTQ-INT8 | 6,816 (55%) | 570K (16%) | 278K (16%) | 56 | 357 | **157** |

- **Bar chart**: Resource comparison (DSP, FF, LUT side-by-side)

**Key Points:**
- Target: **Xilinx VU13P** (xcvu9p-flga2577)
- Float baseline: 116 ns latency (42 cycles @ 362 MHz)
- PTQ-INT8: **157 ns latency** (56 cycles @ 357 MHz)
- **Counter-intuitive**: INT8 uses *more* DSPs due to 64-bit scaling multiplications
- Both implementations well within FPGA capacity (~50-55% DSP)

**What to Say:**
> "Synthesis targets the Xilinx VU13P. Surprisingly, the integer implementation uses more DSPs than float due to 64-bit fixed-point scaling operations combined with aggressive loop unrolling. However, both variants fit comfortably at around 50% DSP utilization, with sub-200-nanosecond latency."

**Existing Material:**
- ✅ Table in supporting doc (ready)
- ✅ HLS reports in `build/hls/`
- ✅ Comparison plot: `build/plots/hls_comparison.png`

---

### SLIDE 10: Bit-Width Optimization (1 min)

**Visual:**
- **Table**: Conservative vs Optimized widths
  | Type | Conservative | Optimized | Savings |
  |------|--------------|-----------|---------|
  | Adjacency | 16 bits | 16 bits | 0 bits |
  | Accumulators | 32 bits | 22 bits | **10 bits** |
  | Scaling factors | 32 bits | 21 bits | **11 bits** |
  | Scaling products | 64 bits | 43 bits | **21 bits** |
  | **Total savings** | | | **42 bits** |

- **Diagram**: Show how bit-width analysis works
  - Trace min/max during simulation
  - Compute required width + margin
  - Generate optimized HLS header

**Key Points:**
- **Data-driven optimization**: Analyze actual ranges during simulation
- Focus on critical paths: accumulators, scaling multiplications
- **42-bit total reduction** in multiplier operands
- Tool: `src/optimize_bitwidths_int8.py`
- Auto-generates `hls/auto_generated_bitwidths.h`

**What to Say:**
> "By analyzing actual numerical ranges during integer simulation, we identify opportunities to narrow bit-widths. The bit-width optimizer reduces accumulator and scaling product widths by up to 42 bits total, directly reducing DSP and LUT consumption without sacrificing accuracy."

**Existing Material:**
- ✅ Table data in supporting doc
- ✅ Tool: `src/optimize_bitwidths_int8.py`
- 🔄 Need visualization of analysis flow

---

### SLIDE 11: Design Space Exploration Framework (1.5 min)

**Visual:**
- **Left**: DSE hierarchy diagram (3 levels)
  1. Algorithm: Hidden dims (16, 24, 32, 48), M (20, 24, 28)
  2. Quantization: Bit-widths, scales
  3. HLS: Unroll factors, pipeline II

- **Right**: Pareto front plot (preliminary)
  - X-axis: Latency (cycles)
  - Y-axis: Accuracy (%)
  - Color: DSP usage
  - **Show single point** (current baseline: 16×24, M=24, unroll=1)
  - Annotate: "Full sweep in progress"

**Key Points:**
- **Automated exploration**: Python script generates configs, runs HLS, parses reports
- Multi-objective optimization: Accuracy × Latency × Resources
- **Current status**: Framework validated, 1 Pareto point
- **In progress**: Sweep over 10-20 configurations

**What to Say:**
> "The DSE framework systematically explores three hierarchical levels: algorithm, quantization, and microarchitecture. Each design point runs the full flow from PTQ parameter generation through HLS synthesis. Pareto analysis identifies non-dominated solutions. The infrastructure is complete—we're currently executing parameter sweeps to populate the Pareto frontier."

**Existing Material:**
- ✅ DSE script: `src/explore_design_space.py`
- ✅ Pareto plots: `build/plots/pareto/*.png`
- ✅ Framework description in supporting doc
- 🔄 Pareto plots currently show limited points (need more synthesis runs)

---

### SLIDE 12: Path to Physics Application (1 min)

**Visual:**
- **Roadmap diagram** (3 phases):
  1. **Phase 1: Method Development** ✅
     - Proxy dataset (Cora)
     - Pipeline validated
     - Tools mature
  
  2. **Phase 2: Dataset Integration** 🔄
     - Displaced-muon events (CMS Phase-2 simulation)
     - Graph topology: DT/CSC/RPC stubs
     - Edge definition: Δη-Δφ compatibility
  
  3. **Phase 3: Optimization & Deployment** 📋
     - Retrain on physics data
     - DSE for 12.5 μs budget
     - FPGA prototype on VU13P

**Key Points:**
- **Cora = proof-of-concept**, not final application
- Physics dataset in preparation (displaced-muon signatures)
- Graph structure: Detector stubs (nodes) + geometric compatibility (edges)
- Target: CMS overlap region (barrel + endcap)

**What to Say:**
> "While we've used Cora as a public proxy to develop the methodology, the target application is displaced-muon track finding in the CMS overlap region. We're preparing the physics dataset where nodes represent detector stubs and edges capture geometric compatibility. Once integrated, the established pipeline ensures rapid assessment of hardware feasibility."

**Existing Material:**
- ✅ Description in supporting doc (Section: Path to Physics Application)
- 🔄 Need to create roadmap visual
- ✅ Physics motivation clear in submission materials

---

### SLIDE 13: Summary & Key Takeaways (30 sec)

**Visual:**
- **Bullet points** with checkmarks:
  ✅ Complete pipeline: PyTorch → Validated HLS  
  ✅ PTQ-INT8: 0.1% accuracy drop, 0 LSB error  
  ✅ Automated DSE infrastructure ready  
  ✅ Bit-width optimization: 42-bit reduction  
  🔄 QAT under development (larger architectures)  
  🔄 Full design space sweep ongoing  
  📋 Physics data integration planned  

**What to Say:**
> "To summarize: we've developed a complete, reproducible workflow from GNN training to FPGA-ready HLS implementations. PTQ achieves minimal accuracy degradation with bit-exact validation. The automated DSE framework is operational and we're actively expanding results with larger models and comprehensive parameter sweeps. Code and documentation are publicly available."

**Existing Material:**
- ✅ Summary from supporting doc
- ✅ Status summary in attached MD file

---

### SLIDE 14: Backup - Detailed Architecture (if asked)

**Visual:**
- Detailed GraphSAGE layer diagram
- Aggregation → Linear → Activation flow
- Show root_weight=False configuration

**Content:**
- Layer-by-layer breakdown
- Why root_weight=False (HLS simplification)
- Aggregation scheme (mean pooling)

---

### SLIDE 15: Backup - Error Analysis (if asked)

**Visual:**
- Error distribution plots (M=24 vs M=20)
- Per-node error breakdown
- LSB error histogram

**Content:**
- How we measure error (LSB comparison)
- M=20 error profile (max 13 LSB, where?)
- Accuracy vs resources trade-off

**Existing Material:**
- ✅ Error analysis in test comparison scripts
- 🔄 May need to generate new plots

---

### SLIDE 16: Backup - QAT Status (if asked)

**Visual:**
- QAT training curves (current 16×24)
- Comparison table: Float vs PTQ vs QAT
- Plan for 32×48 architecture

**Content:**
- Why QAT underperforms (capacity issues)
- PTQ vs QAT trade-offs
- Next steps: larger model training

**Existing Material:**
- ✅ Training plots: `build/plots/qat_model_qat_no_root_training.png`
- ✅ Analysis in `docs/notes/QAT_TRAINING_NOTES.md`

---

## Material Status & Action Items

### ✅ READY (Can Use Immediately)

**Plots:**
- Training curves: `build/plots/base_model_training.png`
- Training curves: `build/plots/reduced_model_training.png`
- QAT training: `build/plots/qat_model_qat_no_root_training.png`
- HLS comparison: `build/plots/hls_comparison.png`
- Pareto fronts: `build/plots/pareto/*.png`
- Resource utilization: `build/plots/resource_utilization.png`
- Accuracy degradation: `build/plots/accuracy_degradation.png`

**Tables/Data:**
- Model comparison table (supporting doc)
- HLS synthesis table (supporting doc)
- Validation table (supporting doc)
- Bit-width optimization table (supporting doc)

**Documentation:**
- Supporting document (2 pages)
- README with full pipeline description
- QAT training notes
- Current status summaries

### 🔄 NEEDS IMPROVEMENT

**Missing Visualizations:**
1. CMS detector diagram (overlap region)
2. Graph topology illustration (stubs + edges)
3. Technical challenge flow diagram
4. Enhanced pipeline diagram
5. Quantization scheme diagram (layer-level)
6. Bit-width optimization flow
7. Physics roadmap visual

**Analysis Gaps:**
1. More Pareto points (need 10-20 synthesis runs)
2. Latency breakdown (per-layer timing)
3. Resource projections for full graph sizes
4. Comparison with other approaches (baseline table)

**Code/Experiments:**
1. Train QAT with 32×48 architecture
2. Run comprehensive DSE sweep
3. Generate more HLS configurations
4. Synthesize 10-20 design points

### 📋 NICE TO HAVE (But Not Critical)

**Advanced Analysis:**
- Activation distribution analysis
- Weight sparsity analysis
- Per-layer resource breakdown
- Energy/power estimates
- Comparison with hls4ml/QKeras

**Physics Context:**
- Displaced-muon event display
- Stub-level feature descriptions
- Edge definition criteria
- CMS Phase-2 FPGA specifications

**Interactive Elements:**
- Live demo of DSE tool
- Jupyter notebook walkthrough
- Animation of message-passing

---

## Recommendations for Next 2 Weeks

### Critical (Must Have Before Presentation)

1. **Train QAT 32×48 Model** (~1-2 days)
   ```bash
   cd src
   python train_qat.py --in-channels 32 --hidden-channels 48 --epochs 200
   ```
   - **Expected result**: 60-70% accuracy (vs current 28%)
   - Updates Table 1 with better QAT numbers

2. **Run DSE Sweep** (~3-4 days, mostly compute time)
   ```bash
   cd src
   python explore_design_space.py --config configs/design_space_minimal.yaml
   ```
   - Generate 10-15 design points
   - Architectures: 16×24, 24×32, 32×48
   - M values: 20, 24
   - Unroll factors: 1, 8
   - **Expected result**: Populated Pareto fronts

3. **Create Missing Diagrams** (~1-2 days)
   - CMS detector schematic (can use existing references)
   - Graph topology illustration
   - Quantization flow diagram
   - Enhanced pipeline visual
   - Physics roadmap
   - **Tool**: PowerPoint, Inkscape, or matplotlib

4. **Synthesize More Configurations** (~2-3 days)
   - At least 5-8 distinct HLS configurations
   - Vary: hidden_dim, M, unroll_factor
   - Parse reports and update comparison tables
   - **Expected result**: Better resource vs latency curves

### Important (Should Have)

5. **Latency Breakdown Analysis** (~1 day)
   - Parse HLS reports for per-layer timing
   - Create latency waterfall chart
   - Identify bottlenecks

6. **Error Distribution Plots** (~half day)
   - Visualize M=24 vs M=20 error profiles
   - Per-node error histograms
   - Max/mean/std error statistics

7. **Resource Projection Study** (~1 day)
   - Extrapolate from 8-node to 100-node graphs
   - Estimate DSP/LUT for full-size problems
   - Compare with VU13P capacity

8. **Update All Plots** (~half day)
   - Ensure consistent style (fonts, colors)
   - Add labels, legends, annotations
   - Export high-res versions (PNG + PDF)

### Nice to Have (Time Permitting)

9. **Baseline Comparison Table**
   - Literature review: other GNN-on-FPGA works
   - Compare metrics (latency, resources, accuracy)

10. **Physics Dataset Prep**
    - Start preparing displaced-muon samples
    - Define graph structure (even if toy version)
    - Show example event

11. **Interactive Demo**
    - Jupyter notebook showing DSE workflow
    - Live parameter modification
    - Real-time Pareto front updates

---

## Presentation Delivery Tips

### Opening (First 30 seconds)
- Start with strong motivation: "Why this matters for CMS"
- Use the 12.5 μs latency budget as a hook
- Mention displaced muons and LLP physics (hot topic)

### Middle (Technical Content)
- **Show, don't just tell**: Use visualizations heavily
- **Emphasize validation**: 0 LSB error is impressive
- **Be honest**: "QAT in progress, PTQ works well"
- **Connect to physics**: Remind audience this is for real detector

### Conclusion
- **Focus on methodology**: "Complete, reproducible pipeline"
- **Acknowledge ongoing work**: "Full DSE sweep underway"
- **Clear path forward**: "Ready for physics data integration"

### Q&A Prep
**Likely questions:**
1. "Why does INT8 use more DSPs than float?"
   - Answer: 64-bit fixed-point scaling with full unrolling
   
2. "What about QAT performance?"
   - Answer: Current architecture too small (28%), training 32×48 now
   
3. "How does this scale to real detector?"
   - Answer: 8-node is subgraph, full graph is ~100 nodes, resource projection in progress
   
4. "When will you have physics data?"
   - Answer: Dataset preparation ongoing, framework ready for rapid iteration
   
5. "Comparison with hls4ml?"
   - Answer: hls4ml targets CNNs well, GNN message-passing needs custom handling

---

## Files to Bring/Have Ready

### For Presentation
- PDF slides (with backups)
- USB backup
- Presenter notes (printed)

### For Discussion
- Supporting document PDF (ieee_rt2026_supporting_doc.pdf)
- README.md (pipeline overview)
- Example configuration files
- Sample HLS reports

### Code Access
- Have laptop with repository ready
- Demo of DSE tool (if asked)
- Jupyter notebook for interactive exploration

---

## Key Messages to Emphasize

1. **Completeness**: "End-to-end workflow, not just training or just HLS"
2. **Validation**: "Bit-exact, 0 LSB error—we trust the hardware"
3. **Automation**: "DSE explores 100s of configs without manual tuning"
4. **Data-driven**: "Bit-width optimization based on actual simulation ranges"
5. **Reproducibility**: "Public dataset, open tooling, documented pipeline"
6. **Honest presentation**: "PTQ works great, QAT needs larger model, physics data coming"

---

## Timeline to Presentation (2-Week Plan)

### Week 1: Complete Critical Items
- **Day 1-2**: Train QAT 32×48
- **Day 3-5**: Run DSE sweep (10-15 configs)
- **Day 6-7**: Create missing diagrams

### Week 2: Synthesis & Refinement
- **Day 8-10**: Synthesize more HLS configs
- **Day 11-12**: Analysis & plots
- **Day 13**: Assemble slides
- **Day 14**: Practice & refinement

### Last 3 Days Before Conference
- Print materials
- Prepare backup slides
- Practice Q&A
- Test equipment

---

## Success Metrics

**Minimum viable presentation:**
- ✅ Clear motivation (CMS + LLP)
- ✅ Complete pipeline description
- ✅ Validated results (0 LSB error)
- ✅ At least 5 Pareto points
- ✅ Honest framing (ongoing work)

**Good presentation:**
- ✅ All above +
- ✅ 10+ Pareto points
- ✅ Better QAT results (>60%)
- ✅ Professional visuals
- ✅ Latency breakdown

**Excellent presentation:**
- ✅ All above +
- ✅ Physics data example
- ✅ Interactive demo
- ✅ Baseline comparisons
- ✅ Resource projections

---

## Contact for Questions

**Before presentation:**
- Review this plan with advisor/collaborators
- Get feedback on priorities
- Adjust timeline based on available compute resources

**During preparation:**
- Test all commands in clean environment
- Verify reproducibility
- Document any issues/workarounds

**At conference:**
- Arrive early to test AV setup
- Have backup plan for technical issues
- Bring business cards for follow-up discussions

---

## Conclusion

This presentation plan provides a complete roadmap from current state to conference-ready talk. The most critical items are training the larger QAT model and running the DSE sweep to populate Pareto fronts. All other materials can be assembled from existing data and documentation.

**Priority ranking:**
1. 🔥 Train QAT 32×48 (changes key result)
2. 🔥 DSE sweep (more Pareto points)
3. 🔥 Create missing diagrams (visual impact)
4. 📊 More HLS synthesis (better analysis)
5. 📊 Latency breakdown (deeper insight)
6. ✨ Nice-to-haves (time permitting)

The honest framing of ongoing work is appropriate and valuable—focus on the validated methodology and complete pipeline as the main contribution.
