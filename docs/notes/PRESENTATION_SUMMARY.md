# IEEE RT 2026 Presentation - Executive Summary

**Date**: May 25-29, 2026  
**Duration**: 15 minutes (12 min talk + 3 min Q&A)  
**Status**: Framework complete, need 2 critical items before presentation

---

## Current State Assessment

### ✅ What's COMPLETE and Ready
1. **Full pipeline working**: Training → Quantization → HLS → Validation
2. **Strong results**: PTQ-INT8 with only 0.1% accuracy drop
3. **Bit-exact validation**: 0 LSB error in HLS C-simulation
4. **Automated tools**: DSE framework, bit-width optimizer
5. **Good visualizations**: 10+ plots ready in `build/plots/`
6. **Documentation**: Supporting doc, README, status summaries

### 🔄 What's IN PROGRESS (Critical Gaps)
1. **QAT performance**: Current 16×24 shows 28.2% accuracy (too low)
   - **Action needed**: Train 32×48 architecture → expect 60-70%
   - **Time**: 1-2 days of training
   
2. **Limited Pareto points**: Only 1-2 design points synthesized
   - **Action needed**: Run DSE sweep for 10-15 configurations
   - **Time**: 3-4 days (mostly compute time)

### 📋 What's MISSING (Nice to Have)
1. **Visual diagrams**: Need 5-7 new diagrams for clarity
   - CMS detector schematic
   - Graph topology illustration
   - Quantization flow diagram
   - Physics roadmap visual
   - **Time**: 1-2 days

2. **Deeper analysis**: Latency breakdown, resource projections
   - **Time**: 1-2 days

---

## Presentation Structure (14 Slides)

### Core Slides (12 minutes)
1. **Title** (15s)
2. **Motivation** - CMS + displaced muons (1m)
3. **Technical Challenge** - ML to FPGA gap (45s)
4. **Our Contribution** - Complete pipeline (30s)
5. **Model Architecture** - GraphSAGE on Cora (30s)
6. **Training Results** - Table with accuracies (1m)
7. **Quantization Scheme** - Integer-only design (1.5m)
8. **HLS Validation** - 0 LSB error proof (1m)
9. **HLS Synthesis** - Resource comparison (1.5m)
10. **Bit-width Optimization** - 42-bit savings (1m)
11. **Design Space Exploration** - Pareto fronts (1.5m)
12. **Path to Physics** - Roadmap (1m)
13. **Summary** - Key takeaways (30s)

### Backup Slides (for Q&A)
14. Detailed architecture
15. Error analysis (M=24 vs M=20)
16. QAT status and next steps

---

## Critical Actions for Next 2 Weeks

### Week 1: Generate Missing Results

#### Priority 1: Train QAT 32×48 (CRITICAL)
```bash
cd src
python train_qat.py --in-channels 32 --hidden-channels 48 --epochs 200
```
- **Why critical**: Changes Table 1 from 28% → 60-70% accuracy
- **Time**: 1-2 days
- **Impact**: Shows QAT feasibility, not a failure

#### Priority 2: Run DSE Sweep (CRITICAL)
```bash
cd src
python explore_design_space.py --config configs/design_space_minimal.yaml
```
- **Target**: 10-15 design points
- **Parameters to vary**:
  - Architectures: 16×24, 24×32, 32×48
  - M values: 20, 24
  - Unroll factors: 1, 8
- **Why critical**: Populates Pareto fronts (currently only 1 point)
- **Time**: 3-4 days
- **Impact**: Main contribution is DSE methodology

#### Priority 3: Create Visual Diagrams (IMPORTANT)
- CMS detector overlap region
- Graph representation (nodes/edges)
- Quantization flow (layer-level)
- Enhanced pipeline diagram
- Physics roadmap
- **Time**: 1-2 days
- **Tools**: PowerPoint, Inkscape, or matplotlib

### Week 2: Analysis & Assembly

#### Priority 4: Synthesize More Configs (IMPORTANT)
- Run HLS synthesis on 5-8 configurations
- Parse reports and update tables
- **Time**: 2-3 days
- **Impact**: Better resource vs latency curves

#### Priority 5: Create Presentation Slides
- Assemble 14 slides from materials
- Ensure consistent formatting
- Add annotations and notes
- **Time**: 1-2 days

#### Priority 6: Practice & Refine
- Rehearse timing (12 min target)
- Prepare for Q&A
- Test backup slides
- **Time**: 1 day

---

## What Material Exists vs What's Needed

### Existing Plots (Ready to Use)
- ✅ `build/plots/base_model_training.png`
- ✅ `build/plots/reduced_model_training.png`
- ✅ `build/plots/qat_model_qat_no_root_training.png`
- ✅ `build/plots/hls_comparison.png`
- ✅ `build/plots/pareto/*.png` (6 Pareto plots)
- ✅ `build/plots/resource_utilization.png`
- ✅ `build/plots/accuracy_degradation.png`

### Existing Tables (Ready to Use)
- ✅ Model comparison (supporting doc Table 1)
- ✅ HLS validation (supporting doc Table 2)
- ✅ HLS synthesis (supporting doc Table 3)
- ✅ Bit-width optimization table

### Missing Visualizations (Need to Create)
- 🔄 CMS detector schematic
- 🔄 Graph topology diagram
- 🔄 Technical challenge flow
- 🔄 Quantization scheme diagram
- 🔄 Bit-width optimization flow
- 🔄 Physics roadmap
- 🔄 Enhanced pipeline diagram

### Missing Results (Need to Generate)
- 🔄 QAT 32×48 training results
- 🔄 10-15 more Pareto points (DSE)
- 🔄 5-8 more HLS synthesis configs
- 🔄 Latency breakdown per layer
- 🔄 Resource projection to larger graphs

---

## Key Messages to Emphasize

### Main Contributions (What to Highlight)
1. **Complete end-to-end workflow** (not just training, not just HLS)
2. **Bit-exact validation** (0 LSB error - trustworthy)
3. **Automated DSE** (explores 100s of configs systematically)
4. **Data-driven optimization** (42-bit width reduction)
5. **Reproducible methodology** (public dataset, documented)

### Honest Framing (How to Present Ongoing Work)
- ✅ "PTQ achieves excellent results" (0.1% drop)
- ✅ "QAT under development with larger architectures"
- ✅ "DSE framework validated, full parameter sweep ongoing"
- ✅ "Cora used as proxy to develop methodology"
- ✅ "Physics data integration planned, pipeline ready"

### What NOT to Claim
- ❌ "Final FPGA implementation ready for deployment"
- ❌ "Complete design space explored"
- ❌ "Optimized for 12.5 μs latency" (stated as future work)
- ❌ "Integrated with displaced-muon data"

---

## Anticipated Q&A

### Question 1: "Why does INT8 use more DSPs than float?"
**Answer**: The integer implementation uses 64-bit fixed-point scaling multiplications (INT32 × scale_fp) which require more DSP blocks than float multiplies. Combined with aggressive loop unrolling, this increases DSP count despite using narrower INT8 data types.

### Question 2: "What about your QAT results? 28% seems low."
**Answer**: You're absolutely right—the current 16×24 architecture is too small for quantization-aware training. We're actively training a 32×48 variant which we expect to achieve 60-70% accuracy, closer to our PTQ results. Meanwhile, PTQ works excellently with only 0.1% degradation.

### Question 3: "How does this scale to real detector graphs?"
**Answer**: The 8-node subgraph is our validation case. For the full CMS overlap region, we're projecting graphs of 50-100 nodes. Our DSE framework will help us identify architectures that fit within the 12.5 microsecond budget and available FPGA resources. Resource projections are underway.

### Question 4: "When will you have physics data?"
**Answer**: The displaced-muon dataset is in preparation. We're defining the graph topology where nodes represent detector stubs from DT, CSC, and RPC chambers, and edges capture geometric compatibility. The established pipeline ensures that once physics data is ready, we can rapidly iterate on hardware designs.

### Question 5: "How does this compare to hls4ml or other tools?"
**Answer**: Tools like hls4ml are excellent for CNNs and dense layers, but message-passing GNNs have irregular aggregation patterns that need custom handling. Our contribution is establishing this end-to-end workflow specifically for GNN message-passing operations, with explicit control over quantization and fixed-point arithmetic.

### Question 6: "What about power/energy consumption?"
**Answer**: We haven't measured power yet—that's planned for future work once we have FPGA prototypes. Current focus is on latency and resource utilization, which correlate with power but don't directly measure it.

---

## Resource Requirements

### Compute Resources Needed
- **GPU training**: QAT 32×48 (~12 hours on modern GPU)
- **HLS synthesis**: 10-15 runs (~2-4 hours each) = 20-60 hours
- **Can parallelize**: Run multiple HLS syntheses simultaneously

### Software Requirements
- PyTorch Geometric (installed ✅)
- Vitis HLS 2024.1 (installed ✅)
- Matplotlib for plots (installed ✅)
- LaTeX/PowerPoint for slides

### Time Budget (2 Weeks Total)
- **Week 1**: Generate results (QAT, DSE, diagrams)
- **Week 2**: Synthesis, analysis, slides, practice
- **Buffer**: 2-3 days for unexpected issues

---

## Success Criteria

### Minimum Viable Presentation
- ✅ Clear motivation (CMS + displaced muons)
- ✅ Complete pipeline explained
- ✅ 0 LSB validation shown
- ✅ At least 5 Pareto points
- ✅ Honest framing of status
- ⚠️ **Status**: Close, need QAT + DSE

### Good Presentation (Target)
- ✅ All minimum items +
- ✅ 10+ Pareto points
- ✅ Better QAT (>60% accuracy)
- ✅ Professional visuals
- ✅ Latency breakdown
- ⚠️ **Status**: Achievable in 2 weeks

### Excellent Presentation (Stretch)
- ✅ All good items +
- ✅ Physics data example
- ✅ Interactive demo capability
- ✅ Baseline comparisons
- ✅ Resource projections
- ⚠️ **Status**: Would need 3-4 weeks

**Realistic target**: "Good Presentation" tier

---

## Presentation Flow (Detailed Timing)

### Opening (2:15 total)
- **0:00-0:15**: Title slide, hello
- **0:15-1:15**: Motivation (CMS, overlap region, LLP physics, 12.5 μs)
- **1:15-2:00**: Technical challenge (ML→FPGA gap, GNN specifics)
- **2:00-2:15**: Quick outline of contribution

### Technical Content (7:30 total)
- **2:15-2:45**: Our pipeline overview
- **2:45-3:15**: Model architecture
- **3:15-4:15**: Training results table + curves
- **4:15-5:45**: Quantization scheme (integer-only design)
- **5:45-6:45**: HLS validation (0 LSB error)
- **6:45-8:15**: HLS synthesis results
- **8:15-9:15**: Bit-width optimization (42-bit savings)
- **9:15-9:45**: Transition to DSE

### Results & Future (2:00 total)
- **9:45-11:15**: DSE framework + Pareto fronts
- **11:15-12:15**: Path to physics application
- **12:15-12:45**: Summary + key messages

### Q&A (3:00)
- **12:45-15:00**: Questions and discussion

**Buffer**: Aim for 11:30 to allow flexibility

---

## Next Steps (Action Plan)

### Immediate (This Week)
1. **Monday-Tuesday**: Train QAT 32×48
2. **Wednesday-Friday**: Launch DSE sweep
3. **Weekend**: Create missing diagrams

### Second Week
4. **Monday-Tuesday**: Additional HLS synthesis runs
5. **Wednesday**: Analyze results, update tables/plots
6. **Thursday**: Assemble presentation slides
7. **Friday**: Practice run, timing check

### Final Days Before Conference
8. Prepare backup materials (USB, printed notes)
9. Final practice session
10. Equipment check (laptop, adapters, etc.)

---

## Files Referenced

### Documentation
- [SUBMISSION_STATUS_SUMMARY.md](/home/pelayo/work/simple-gnn/docs/notes/SUBMISSION_STATUS_SUMMARY.md) - Current status
- [IEEE_RT2026_SUBMISSION_GUIDE.md](/home/pelayo/work/simple-gnn/docs/notes/IEEE_RT2026_SUBMISSION_GUIDE.md) - Submission materials
- [ieee_rt2026_supporting_doc.pdf](/home/pelayo/work/simple-gnn/docs/notes/ieee_rt2026_supporting_doc.pdf) - 2-page supporting doc
- [PRESENTATION_PLAN_15MIN.md](/home/pelayo/work/simple-gnn/docs/notes/PRESENTATION_PLAN_15MIN.md) - Detailed slide-by-slide plan

### Key Scripts
- `src/train_qat.py` - Train QAT models
- `src/explore_design_space.py` - Run DSE
- `run_pipeline.py` - Full pipeline execution

### Existing Plots
- `build/plots/` - All visualization outputs
- `build/plots/pareto/` - Pareto front plots

---

## Conclusion

**Bottom Line**: The presentation is ~80% ready. The two critical items are:
1. Training QAT with a larger architecture (1-2 days)
2. Running DSE to populate Pareto fronts (3-4 days)

Everything else (diagrams, slides, practice) can be done in parallel or after these complete.

**Timeline is tight but achievable** with focused effort over next 2 weeks.

**Confidence level**: High for "Good Presentation", Medium for "Excellent"

**Main risk**: HLS synthesis time (can be mitigated by running jobs in parallel)

**Fallback**: If DSE sweep doesn't complete in time, present framework with current results and emphasize methodology over quantity of Pareto points.
