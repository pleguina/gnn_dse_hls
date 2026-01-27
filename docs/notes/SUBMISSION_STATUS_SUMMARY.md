# IEEE RT 2026 Submission - Status Summary

## ✅ Files Ready for Submission

### 1. Short Abstract (242 words, <250 limit)
**File**: `ieee_rt2026_short_abstract_FINAL.txt`

**Status**: Ready to submit

**Key points emphasized:**
- Presents an end-to-end **workflow** (not claiming finished product)
- Focuses on **methodology and framework**
- **Preliminary results** on Cora dataset
- QAT explicitly stated as **"under active development"**
- HLS C-simulation validation (bit-exact)
- Clear that displaced-muon dataset is **"in preparation"**

### 2. Supporting Document (exactly 2 pages, 515 KB)
**File**: `ieee_rt2026_supporting_doc.pdf`

**Status**: Ready to submit

**What's presented as COMPLETE:**
- ✅ Model training (Base: 80.3%, Reduced: 75.8%)
- ✅ PTQ-INT8 implementation (75.7% accuracy, 0.1% degradation)
- ✅ HLS C-simulation validation (0 LSB error at M=24)
- ✅ Automated DSE infrastructure
- ✅ Bit-width optimization tools (42-bit reduction capability)

**What's presented as IN PROGRESS:**
- 🔄 QAT with larger architectures (current 16×24 shows capacity issues → exploring 32×48)
- 🔄 Comprehensive DSE parameter sweeps (framework ready, full sweeps ongoing)
- 🔄 Displaced-muon dataset integration
- 🔄 Track-finder graph definition

**What's presented as PLANNED:**
- 📋 Retrain on physics data
- 📋 Optimize for 12.5 μs latency budget
- 📋 Deploy on Phase-2 FPGAs

## How Ongoing Work is Presented

### Table 1: Model Results
| Model Variant | Accuracy | Status |
|---------------|----------|--------|
| Base (float) | 80.3% | **Complete** |
| Reduced (float) | 75.8% | **Complete** |
| PTQ-INT8 | 75.7% | **HLS C-sim validated** |
| QAT-INT8 (16×24) | 28.2% | **In progress** |

Caption explicitly states: "QAT under development with larger architectures"

### Figure 1: Preliminary Results
- Caption clearly states: "**Preliminary** HLS synthesis results"
- Pareto plot labeled: "initial baseline point shown; **full sweep in progress**"
- Main caption: "**Preliminary results** and DSE framework. Full design space exploration is **ongoing**."

### Section: Current Status and Next Steps
Explicitly breaks down:
- **Completed**: Training, PTQ-INT8, validation, DSE infrastructure, tools
- **In progress**: QAT, DSE sweeps, dataset integration, graph definition
- **Planned**: Physics data, latency optimization, FPGA deployment

## Honest Framing of Contributions

The documents emphasize:

1. **Methodology over results**: "develops a reproducible pipeline", "framework includes", "designed to explore"

2. **Validation focus**: 0 LSB error in C-simulation (this IS complete and verifiable)

3. **Infrastructure contribution**: Automated DSE, bit-width optimization, test vector generation

4. **Proof-of-concept**: Using Cora as "public proxy to develop and stress-test the toolchain"

5. **Clear path forward**: Specific next steps for physics application

## What We're NOT claiming

- ❌ Final FPGA implementation ready for deployment
- ❌ Complete design space exploration results
- ❌ Optimized QAT model
- ❌ Integration with displaced-muon physics data
- ❌ Meeting the 12.5 μs latency budget (stated as future work)

## What We ARE claiming

- ✅ Complete, reproducible workflow from training to HLS
- ✅ Bit-exact validation of integer implementation
- ✅ PTQ preserves accuracy with minimal degradation
- ✅ Automated DSE framework (infrastructure ready)
- ✅ Data-driven bit-width optimization methodology
- ✅ Proof-of-concept on proxy dataset

## Tone and Language

**Changed from original:**
- "complete" → "reproducible pipeline", "workflow"
- "achieves" → "preliminary results show"
- "synthesis results" → "preliminary HLS synthesis results"
- Removed specific final resource numbers from abstract
- Added "under development", "in progress", "ongoing"

**Maintained:**
- Technical rigor (0 LSB error, bit-exact validation)
- Methodological contributions
- Concrete numbers where validated (model accuracy, PTQ performance)

## Suitability for Abstract Submission

This framing is appropriate because:

1. **IEEE RT conferences expect work-in-progress**: Abstract submissions often present ongoing research with preliminary results

2. **Methodology is valuable**: The complete automated workflow is a contribution even if not all parameter sweeps are complete

3. **Validation is solid**: HLS C-simulation validation with 0 LSB error is a concrete achievement

4. **Honest about status**: Clear about what's done vs planned

5. **Concrete enough**: Real numbers (75.7% accuracy, 0 LSB error, 42-bit optimization) not just promises

## Recommendation

✅ **Ready to submit** - The documents honestly present:
- A validated methodology
- Preliminary but concrete results
- Clear ongoing work
- Specific next steps

This is appropriate for an abstract submission to a conference where full results are expected by presentation time (May 2026).

## Files Location

All files in: `/home/pelayo/work/simple-gnn/docs/notes/`

- `ieee_rt2026_short_abstract_FINAL.txt` - Copy-paste ready
- `ieee_rt2026_supporting_doc.pdf` - Upload ready (2 pages, 515 KB)
- `IEEE_RT2026_SUBMISSION_GUIDE.md` - Full submission guide
- `FIGURES_AND_CITATIONS_GUIDE.md` - Figure details
- This file - Status summary
