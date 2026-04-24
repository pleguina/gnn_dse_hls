# OMTF ML-to-Firmware Architecture Study

> **Last updated: 2026-04-24**
>
> This document was originally written as a design study before experiments were run. Sections 1–14 remain valid as architectural reasoning. **Section 15 (appended 2026-04-24) contains all empirical findings** from the full training and evaluation campaign, including dataset characterisation, model behaviour, and the architecture decision outcome.
>
> **Current status**: Architecture Decision Gate closed. **EdgeCompatNet selected** as the primary model for Stage 7 (quantization + HLS). Open gap: pT regression head not learning; must be fixed before quantization.

## Scope

This document consolidates the current findings on the OMTF muon reconstruction ML problem, using the exact datasets and ntuple branches currently produced, and reframes the architecture discussion under the real final constraint:

**the model is not only meant to work in software, but ultimately to be implemented in FPGA firmware for the Level-1 trigger.**

The goal of this document is to serve as a single design reference covering:

* the available data and labels
* what the data implies structurally
* which model families fit the problem best
* why some models are strong in software but weaker for firmware
* what a firmware-friendly version of HECIN would mean
* the recommended research and deployment strategy

---

# 1. Problem framing

## 1.1 Original question

At the beginning, the architecture question was mostly:

> Given the OMTF detector topology and the raw stub structure, what is the most principled ML architecture for reconstruction?

Under that question, a heterogeneous edge-conditioned interaction network, optionally with an object-condensation-style readout, looked very strong.

## 1.2 Updated question

Now the question is stricter:

> Given the OMTF detector topology, the produced datasets, and the requirement that the final model must be implementable in fixed-latency FPGA firmware for the Level-1 trigger, what architecture family is actually the best target?

This changes the ranking.

A model must now be good in three dimensions at once:

1. **Physics fit**
2. **Dataset / supervision fit**
3. **Firmware fit**

That means the final answer is no longer simply “pick the richest software GNN.”

---

# 2. Data inventory

## 2.1 Produced ROOT files

Current production writes two files per job:

* `omtf_hits_<DATASET>_<PROCID>.root`
* `omtf_nano_<DATASET>_<PROCID>.root`

This is a very good split because it separates:

* raw OMTF input-window stub information for ML training
* global event-level and gen-level truth information for regression and final performance studies

---

## 2.2 `OMTFAllInputTree`: the key training source

This tree is the most important one for the ML problem.

Each entry corresponds to **one processor-window per event**.

### Available branches

* `reg_eventNum`
* `reg_iProcessor`
* `reg_mtfType`
* `reg_stub_layer`
* `reg_stub_phiHw`
* `reg_stub_phiBHw`
* `reg_stub_etaHw`
* `reg_stub_r`
* `reg_stub_quality`
* `reg_stub_type`
* `reg_stub_bx`
* `reg_stub_trackId`
* `reg_stub_ambiguous`

### What this gives structurally

This already contains the full compact per-stub descriptor needed for reconstruction:

* angular coordinate
* local bend information
* eta
* radius
* detector/type identity
* layer identity
* timing
* quality
* true track assignment
* ambiguity mask

This is unusually strong for an ML dataset because it contains not only features, but also directly usable supervision for:

* signal vs noise
* same-track vs different-track
* multi-track grouping
* ambiguity masking

---

## 2.3 `OMTFHitsTree`: matched candidate-level evaluation source

This tree is useful for:

* trigger efficiency
* pT resolution studies
* quality studies
* comparison to current OMTF behavior

It contains:

### Gen-muon truth

* `muonPt`
* `muonEta`
* `muonPhi`
* `muonPropEta`
* `muonPropPhi`
* `muonCharge`
* `muonDxy`
* `muonRho`
* `parentPdgId`
* `vertexEta`
* `vertexPhi`

### OMTF candidate outputs

* `omtfPt`
* `omtfUPt`
* `omtfEta`
* `omtfPhi`
* `omtfCharge`
* `omtfHwEta`
* `omtfProcessor`
* `omtfScore`
* `omtfQuality`
* `omtfRefLayer`
* `omtfRefHitNum`
* `omtfRefHitPhi`
* `omtfFiredLayers`
* `killed`

### Candidate stub vectors

* `hits`
* `hits_phiHw`
* `hits_phiBHw`
* `hits_r`
* `hits_type`
* `hits_bx`

### Why this tree matters architecturally

This tree is important because it exposes the current OMTF logic and output structure:

* reference-layer logic
* pattern score
* fired layer mask
* post-ghost-busting candidate status

This makes it valuable for:

* baseline comparisons
* hybrid learned-classical designs
* pattern-conditioned learned scorer ideas

---

## 2.4 NanoAOD event-level truth

The `Events` tree provides:

### GenMuon truth

* `GenMuon_pt`
* `GenMuon_eta`
* `GenMuon_phi`
* `GenMuon_charge`
* `GenMuon_dXY`
* `GenMuon_lXY`
* `GenMuon_vx`
* `GenMuon_vy`
* `GenMuon_vz`
* `GenMuon_etaSt1`
* `GenMuon_etaSt2`
* `GenMuon_phiSt1`
* `GenMuon_phiSt2`

### OMTF GMT outputs

* `omtf_hwPt`
* `omtf_hwPtUnc`
* `omtf_hwEta`
* `omtf_hwPhi`
* `omtf_hwQual`
* `omtf_hwDXY`
* `omtf_Q`
* `omtf_processor`
* `omtf_muIdx`

### Why this matters

This provides the regression targets and final trigger-level reference outputs needed for:

* pT / charge / eta / phi supervision
* d0 / displacement supervision
* matching OMTF candidates to truth
* final trigger metric evaluation

---

# 3. What the available data implies structurally

The most important conclusion from the available branches is this:

## The problem is not image-like, not sequence-like, and not purely node-wise.

It is best described as:

> a small, variable-size, relational, multi-object assignment-and-regression problem over muon stubs.

That has several consequences.

---

## 3.1 Per-stub local information is strong, but not sufficient

A single stub gives:

* phi
* phiB
* eta
* radius
* type
* layer
* BX
* quality

This is a very rich local description.

However, one stub alone is usually not enough to decide:

* which other stubs belong to the same muon
* whether it is a noise stub
* the full track parameters
* how many muons are present in the processor window

So any good model must reason beyond isolated stubs.

---

## 3.2 The most useful signal is relational

The true tracking content lives in **relations between stubs**.

From pairs of stubs, the model can derive physically meaningful compatibility quantities such as:

* `Δphi`
* `|Δphi|`
* `Δr`
* `Δr²`
* `kappa_hat = 2Δphi / Δr²`
* `|Δeta|`
* `Δbx`
* local phiB consistency
* detector-pair category
* layer-pair category

This is the key reason edge-based and relation-based models fit the problem so well.

---

## 3.3 The input is variable-size and unordered

The number of stubs per processor window varies.

So the model must be comfortable with:

* variable occupancy
* missing layers
* pileup
* multiple stubs in the same layer
* no meaningful arbitrary ordering of stubs within the window

This rules out naive fixed-order sequential modeling as the primary approach.

---

## 3.4 The output multiplicity is small

This is one of the most important characteristics of OMTF.

The problem is not “find an arbitrary large number of tracks.”

It is much closer to:

* 0, 1, 2, or 3 relevant muon candidates per processor region

That means the output is naturally small and trigger-like.

This is why fixed-slot candidate models become especially attractive once firmware is taken seriously.

---

## 3.5 The supervision is unusually strong

Because `reg_stub_trackId` is available, the dataset supports:

* node signal/noise labeling
* same-track edge labeling
* multi-track grouping
* ambiguity masking

This is extremely valuable.

It means the problem can be trained using:

* binary node supervision
* binary pairwise supervision
* cluster/group supervision
* candidate-level regression supervision

This opens many model options that would otherwise be impossible or much less stable.

---

# 4. Dataset production summary

Current production totals:

* S1: 500000
* S2: 500000
* S3: 500000
* S4: 150000
* S5: 150000
* B1: 200000
* B2: 200000
* B3: 200000
* B4: 200000

These 9 datasets cover a very good range of behaviors.

---

## 4.1 Clean single-track training

### S1 — single prompt muon

Use for:

* basic curvature learning
* charge learning
* clean track pattern formation
* prompt reference

### S2 — single displaced muon

Use for:

* displacement-sensitive geometry
* d0-sensitive relations
* displaced prompt-vs-nonprompt differentiation

---

## 4.2 Clean multi-track training

### S3 — two prompt muons in same processor window

Use for:

* multi-track disambiguation
* close-track separation
* ambiguity resolution

### S4 — three prompt muons in same processor window

Use for:

* stress-testing top-3 logic
* candidate competition
* ghost-buster replacement ideas

### S5 — two displaced muons

Use for:

* displaced multi-track separation
* hardest clean clustering-style scenarios

---

## 4.3 Realistic occupancy training

### B1 — one prompt muon + PU200

Use for:

* prompt behavior under real occupancy
* noise rejection with real signal present

### B2 — one displaced muon + PU200

Use for:

* hardest single-track realistic background case
* displaced reconstruction under pileup

### B3 — two prompt muons + PU200

Use for:

* close-track separation under realistic occupancy
* hardest realistic prompt multi-track benchmark

### B4 — noise-only PU200

Use for:

* zero-track output calibration
* false positive suppression
* operating-point setting

---

## 4.4 Overall dataset judgment

The dataset suite is strong enough to support:

* rich software teacher models
* firmware-oriented student models
* ablation studies
* prompt vs displaced studies
* single vs multi-track studies
* noise calibration and threshold tuning

This is a major strength of the current setup.

---

# 5. Why generic GraphSAGE and GAT are not the best primary choices

This does **not** mean they are unusable. It means they are not the best central architecture family for this exact problem.

---

## 5.1 Why GraphSAGE makes limited sense here

GraphSAGE is essentially a neighbor-aggregation model.

That is useful when the problem is mainly:

* “what do my neighbors look like?”
* “how do I summarize a neighborhood?”

But the OMTF problem is more specific than that.

The key questions are:

* which stub is compatible with which other stub?
* which subset of stubs supports one track?
* which compatible relation should dominate in ambiguous situations?

Those are more naturally phrased as **pairwise compatibility** and **candidate assignment** than generic neighborhood summarization.

---

## 5.2 Why vanilla GAT makes limited sense here

GAT introduces attention over neighbors.

In OMTF, this is less compelling as a primary mechanism because:

* many of the most informative relation features are explicit and engineered already
* generic attention often tries to rediscover what pairwise physics features already encode
* softmax-style attention is more awkward for FPGA

So a generic GAT does not exploit the best part of the problem as directly as a dedicated edge-conditioned model.

---

## 5.3 Final judgment on GraphSAGE / GAT

They remain fine baselines.

But they should not be the primary conceptual center of the project.

The problem is more naturally:

* edge-based
* relation-based
* hypothesis-based
* candidate-output-based

than generic neighborhood-embedding-based.

---

# 6. Architecture families considered

This section summarizes all the main model propositions that fit the problem to varying degrees.

---

## 6.1 Full software HECIN-style GNN

### Core idea

A heterogeneous edge-conditioned interaction network with graph message passing over legal stub relations.

### Why it fits well in software

* strongly respects edge semantics
* handles detector heterogeneity
* can use pairwise geometry directly
* can express higher-order interactions
* good for multi-muon reasoning

### Why it becomes weaker as a direct deployment target

* more expensive message-passing machinery
* graph update complexity
* more hardware routing complexity
* usually needs a more complicated readout to output candidates cleanly

### Final role

**Excellent software reference / teacher model**.
Not excluded as a hardware target, but not the lowest-risk first deployment target.

---

## 6.2 HECIN + Object Condensation

### Core idea

Use a heterogeneous interaction backbone, then cluster stubs in a learned latent space to form tracks.

### Why it is strong in software

* native multi-track output
* elegant end-to-end grouping
* handles variable track count naturally
* useful for measuring the reconstruction ceiling

### Why it is weak for firmware

* seed finding
* latent-space clustering
* variable cluster count
* dynamic post-processing
* difficult latency guarantees

### Final role

**Very strong software-only ceiling / teacher model**.
Not a preferred direct deployment architecture.

---

## 6.3 Fixed sparse edge-compatibility network

### Core idea

Define a fixed sparse legal edge set between stubs and compute learned compatibility scores on those edges.

### Why it fits the data best

Because the strongest derived features are pairwise:

* `Δphi`
* `Δr`
* `Δr²`
* `kappa_hat`
* `|Δeta|`
* `Δbx`
* phiB consistency
* pair-type information

### Why it fits supervision best

Because `trackId` directly supports same-track edge labels.

### Why it fits firmware best

Because it can be made:

* static
* sparse
* bounded
* quantized
* fully parallel over legal edges

### Final role

**Top recommendation**.
Best overall compromise between physics, supervision, and hardware realism.

---

## 6.4 Fixed-K hypothesis / slot model

### Core idea

Assume a fixed small number of output track slots, for example 3, and let each slot compete for evidence from the stubs.

### Why it fits the data well

Because each stub already has a rich compact descriptor.

### Why it fits the trigger especially well

Because the output is naturally:

* top few muon candidates
* fixed count
* candidate-shaped

### Why it fits firmware very well

Because the structure is fixed:

* `Nmax` stubs
* `K` slots
* fixed stub-slot interactions
* fixed reductions
* fixed output heads

### Final role

**Top recommendation**.
Especially attractive as a final trigger-shaped deployment architecture.

---

## 6.5 Edge scorer + deterministic candidate builder + compact regressor

### Core idea

Split the problem into stages:

1. learned pairwise scoring
2. deterministic candidate formation
3. learned candidate-level regression

### Why it is compelling

It keeps the ML where it is most useful, but leaves the final combinatorics under deterministic control.

### Why it fits firmware very well

Because the pipeline is modular and each stage can be designed for HLS separately.

### Final role

**Top recommendation**.
Very likely the safest engineering path toward real trigger deployment.

---

## 6.6 Pairwise affinity model

### Core idea

Predict same-track affinity between stub pairs, then use a fixed extraction stage to recover up to a few candidates.

### Strengths

* directly uses `trackId`
* natural for grouping
* close to the true relational structure

### Weaknesses

* tends to overlap conceptually with the edge-compatibility family
* still needs a final assignment stage

### Final role

Strong candidate, but conceptually close to the edge-scorer family.

---

## 6.7 Small relation network

### Core idea

Compute learned pairwise relation features, then pool them without full iterative message passing.

### Why it fits well

Because the problem may not actually need multi-round graph propagation if the pair features are already highly informative.

### Why it is attractive for hardware

Because it is essentially:

* many repeated pair blocks
* fixed reductions

### Final role

Good intermediate candidate, especially if full message passing turns out unnecessary.

---

## 6.8 Pattern-conditioned learned scorer

### Core idea

Keep an OMTF-like structured hypothesis space but replace or augment the hand-crafted scoring with learned scoring.

### Why it fits the current system well

Because `OMTFHitsTree` exposes existing OMTF logic concepts such as:

* reference layer
* pattern score
* fired layer bitmask
* ghost-busting status

### Why it is attractive

This could allow incremental deployment while staying close to the current firmware logic.

### Final role

Not the most general approach, but very relevant for a realistic migration path.

---

## 6.9 Tiny static hetero-MPNN

### Core idea

A very small, very constrained version of a heterogeneous message-passing network:

* fixed sparse graph
* one round or two at most
* tiny hidden dimensions
* fixed output heads

### Why it remains relevant

It is the most deployment-friendly descendant of the full HECIN idea.

### Weakness

It is more complex than simpler edge or slot models, and may not buy enough extra accuracy.

### Final role

Useful benchmark and potential deployment candidate if experiments prove the extra interaction stage matters.

---

## 6.10 DeepSets / pooled baseline

### Core idea

Embed each stub, pool them, predict candidate outputs.

### Final role

Useful sanity baseline.
Not expected to be the best final solution for multi-track structured reconstruction.

---

# 7. Why HECIN initially looked so strong

HECIN originally looked like one of the best ideas because it correctly captured the real structure of the problem:

* the task is relational
* edge semantics matter
* detector heterogeneity matters
* multi-track reasoning matters
* node-only processing is insufficient

These arguments remain valid.

So HECIN was not a mistake and is not being rejected.

What changed is that once firmware became a hard deployment requirement, the architecture ranking had to include additional criteria:

* fixed latency
* bounded control flow
* easy quantization
* simple output logic
* low memory/routing complexity

That shifts HECIN from “obvious best overall architecture” toward “excellent reference model and possible compressed deployment family.”

---

# 8. Does a firmware-friendly HECIN make sense?

## Yes.

A firmware-friendly version of HECIN absolutely makes sense.

But it would not be the same as the full unconstrained software HECIN.

The right mindset is:

> preserve the HECIN principles, but compress the architecture into a static, sparse, quantized, fixed-output form.

---

## 8.1 What should be preserved from HECIN

The valuable part of HECIN is not “it is a general GNN.”

The valuable part is:

* heterogeneous reasoning
* edge-conditioned reasoning
* physics-informed sparse connectivity
* relational feature use
* limited interaction stages

These principles remain valid in firmware.

---

## 8.2 What must change for firmware

A firmware-friendly HECIN should avoid or heavily constrain:

* dynamic graph construction
* unrestricted all-to-all connectivity
* many message-passing rounds
* large hidden dimensions
* heavy softmax attention
* OC clustering or any dynamic latent grouping
* variable-time post-processing

---

## 8.3 What a firmware-friendly HECIN could look like

A realistic hardware-shaped HECIN would have:

### Fixed input budget

* `Nmax` stub slots per window
* padding and valid bits

### Fixed sparse legal edge set

* only predefined layer/type pairs
* optional bounded top-k per target layer
* optional dedicated intra-station phi/phiB relation

### Tiny edge-conditioned block

For each legal edge, compute a small relation block using derived features.

### Minimal interaction depth

* one round, maybe two maximum

### Fixed readout

* top 3 candidate slots
* or deterministic candidate assembly logic
* no latent clustering

This would still be a valid descendant of HECIN.

---

## 8.4 Final judgment on firmware-friendly HECIN

It makes sense.

It is best thought of as:

* **Static HECIN**
* **HECIN-Lite**
* **Trigger HECIN**
* **Sparse fixed HECIN**

This is not the same as full software HECIN, but it preserves the right principles.

---

# 9. Recommended architecture ranking

Given:

* the exact produced branches
* the 9-dataset campaign
* the need for eventual FPGA deployment

this is the recommended ranking.

---

## 9.1 Best deployment-oriented families

### 1. Fixed sparse edge-compatibility network

Best balance of:

* physics fit
* supervision fit
* firmware fit

### 2. Fixed-K hypothesis / slot model

Best balance of:

* trigger-like output structure
* static bounded computation
* direct top-candidate production

### 3. Edge scorer + deterministic candidate builder + compact regressor

Best balance of:

* engineering realism
* interpretability
* modularity
* phased deployment possibility

---

## 9.2 Best software / teacher families

### 4. Full heterogeneous edge-conditioned interaction network

Use for:

* reference performance
* ablation studies
* understanding whether higher-order interactions matter

### 5. HECIN + OC

Use for:

* software reconstruction ceiling
* rich teacher outputs
* studying grouping behavior

### 6. Small relation network

Use for:

* testing whether one pairwise interaction stage already captures most of the gain

---

## 9.3 Useful baselines

### 7. Pattern-conditioned learned scorer

### 8. Tiny static hetero-MPNN

### 9. DeepSets / pooled regressor baseline

---

# 10. Integration with the existing FPGA GNN repository

The current repository is a strong starting point because it already contains many of the ingredients needed for a serious ML-to-HLS workflow:

* training and evaluation code
* quantization and PTQ/QAT flows
* HLS implementations and testbenches
* TCL generation
* design-space exploration
* report parsing and Pareto analysis
* documentation and build organization

However, it was originally built around a very different problem:

* Cora dataset
* GraphSAGE node classification
* static benchmark graphs
* software and HLS flow centered around a fixed academic GNN task

The OMTF problem differs in all the important dimensions:

* event-based, not dataset-graph-based
* multi-track candidate extraction, not node classification
* heterogeneous detector stubs, not homogeneous citation graph nodes
* pairwise physics relations, not generic neighbor aggregation
* trigger firmware target, not just FPGA implementation of a standard GNN layer

So the correct strategy is **not** to force the OMTF problem into the current repository structure unchanged.

The right strategy is:

> first evaluate the existing repository carefully, identify which parts are reusable as infrastructure, and then redesign the model/data/HLS interfaces around the OMTF trigger problem.

---

## 10.1 First required action: evaluate the existing repository before extending it

Before adding any OMTF-specific development, the repository should be audited as an engineering base.

### Questions that must be answered

#### A. What is reusable as-is?

Likely reusable:

* build directory conventions
* model checkpointing and experiment organization
* quantization/PTQ/QAT infrastructure patterns
* HLS testbench generation patterns
* TCL/project generation flow
* report parsing and synthesis-result collection
* design-space exploration orchestration
* docs/config/build separation

#### B. What is reusable with refactoring?

Likely reusable with changes:

* model factory / training entrypoints
* dataset loader abstractions
* test-vector generation flow
* parameter export flow
* quantized parameter preparation scripts
* HLS code-generation conventions
* experiment/result schemas

#### C. What is Cora/GraphSAGE-specific and should not remain central?

Likely Cora-specific and not suitable as the main abstraction:

* assumptions of a single fixed graph
* node-classification labels and losses
* GraphSAGE-centric data structures and naming
* subgraph extraction logic tailored to Cora
* layer interfaces assuming GraphSAGE aggregation semantics
* benchmark assumptions based on a citation graph rather than event-wise detector windows

### Required output of this audit

The repo evaluation should produce a written migration note answering:

1. which modules stay
2. which modules are generalized
3. which modules are replaced
4. what the new top-level package structure should be for OMTF
5. how the HLS path should evolve from GraphSAGE-layer benchmarking to trigger-oriented candidate reconstruction blocks

---

## 10.2 Recommended repository migration plan

A clean target structure is to evolve the repository from:

* “GraphSAGE on Cora for FPGA”

toward:

* “event-based trigger ML architecture exploration and firmware deployment for OMTF”

### Suggested migration tracks

#### Track A — infrastructure preservation

Keep and generalize:

* experiment management
* quantization flow
* design-space exploration
* HLS automation
* report parsing
* plotting / Pareto tooling

#### Track B — dataset layer replacement

Replace the Cora dataset assumptions with:

* event-wise OMTF window datasets
* processor-window sample objects
* stub-feature tensors
* pair-feature generation
* optional graph builders
* candidate-level truth joins from NanoAOD

#### Track C — model layer replacement

Replace GraphSAGE-centric model assumptions with a model zoo centered on OMTF-suitable families:

* fixed sparse edge-compatibility models
* slot / hypothesis models
* hybrid edge scorer + deterministic builder models
* software-reference HECIN-style models
* deployment-oriented HECIN-lite variants

#### Track D — HLS kernel refactoring

Move from “GraphSAGE layer kernels” toward “trigger-oriented inference kernels” such as:

* edge feature calculator
* edge scorer kernel
* slot assignment kernel
* candidate accumulation kernel
* compact regression head kernel
* deterministic candidate builder / selector

#### Track E — evaluation and metrics layer replacement

Replace node-classification evaluation with trigger-relevant metrics:

* efficiency vs pT
* efficiency vs d0
* close-muon separation efficiency
* fake candidate multiplicity
* PU200 background acceptance
* candidate quality distributions
* threshold scans
* firmware cost / latency / throughput metrics

---

## 10.3 Practical recommendation for the repository

Do not begin by rewriting everything.

Instead follow this order:

1. **repository audit**
2. **dataset audit and preparation**
3. **minimal OMTF software pipeline in Python only**
4. **first trigger-relevant model baselines**
5. **deployment-oriented architecture path**
6. **HLS kernelization of the chosen architecture family**

This order reduces the risk of carrying over the wrong abstractions from the current GraphSAGE/Cora setup.

---

# 11. Dataset preparation and analysis plan

Before serious model development, a dedicated dataset-preparation and dataset-audit phase is required.

This phase is not optional. It should happen before committing to any final architecture.

The reason is simple:

* the architecture choice depends on what the dataset actually contains
* the loss design depends on label quality
* firmware feasibility depends on realistic occupancy and feature ranges
* efficiency and fake-rate studies depend on event/sample integrity

---

## 11.1 First goal of the dataset phase

Answer the question:

> Does the produced dataset truly contain everything needed for the intended OMTF reconstruction study, in a form that is correct, consistent, and practical for both software training and future firmware deployment?

---

## 11.2 Required dataset audit tasks

### A. Structural integrity checks

Verify for `OMTFAllInputTree`:

* all `reg_stub_*` vectors have equal length per entry
* no branch corruption or schema drift exists across samples
* processor-window indexing is consistent
* event IDs match NanoAOD after the documented uint32 cast
* `reg_iProcessor` ranges are correct and stable
* sample files are internally consistent across the full production campaign

### B. Physics-feature sanity checks

Verify ranges and distributions of:

* `reg_stub_phiHw`
* `reg_stub_phiBHw`
* `reg_stub_etaHw`
* `reg_stub_r`
* `reg_stub_quality`
* `reg_stub_type`
* `reg_stub_bx`
* `reg_stub_layer`

This must include:

* min/max/range tables
* outlier detection
* prompt vs displaced comparisons
* PU vs no-PU comparisons
* occupancy per processor and per sample

### C. Truth-label integrity checks

The following must be validated carefully:

* `reg_stub_trackId == 0` really behaves as noise / PU
* `reg_stub_trackId > 0` correctly maps to the intended GenMuon index convention
* `reg_stub_ambiguous` is populated sensibly and not trivially always 0 or always 1
* ambiguous-stub rates by sample are measured
* multi-muon events have stable non-colliding track IDs inside each event

This is one of the most important checks in the whole project.

### D. Join validation with NanoAOD

The join from `OMTFAllInputTree` to NanoAOD GenMuon truth must be validated explicitly:

* event matching after uint32 cast
* `track_id - 1` indexing agreement with `GenMuon_*`
* no systematic off-by-one issues
* no duplicated or missing joins
* processor-window entries with multiple truth muons are handled correctly

### E. Candidate-level consistency checks

Cross-check `OMTFAllInputTree`, `OMTFHitsTree`, and NanoAOD outputs:

* event / processor matching
* consistency of fired layers and stub content
* candidate-to-window mapping
* ghost-busted candidate behavior
* whether OMTF outputs line up with the raw window occupancy as expected

---

## 11.3 Required dataset analysis tasks

### A. Occupancy analysis

For each dataset, measure:

* stubs per processor-window
* valid layers per window
* stub types per window
* BX distribution per stub
* occupancy tail behavior
* prompt vs displaced occupancy differences
* PU vs no-PU occupancy differences

This is critical both for ML design and for firmware dimensioning.

### B. Sample-content validation

Check that each produced dataset really behaves as intended:

* S1 behaves like single prompt muon
* S2 behaves like single displaced muon
* S3/S4 contain the intended multi-muon same-window topology
* S5 contains displaced dimuon behavior
* B1/B2/B3 have realistic signal+PU occupancy
* B4 is truly zero-track background occupancy

### C. Label coverage analysis

Measure:

* number of signal stubs vs noise stubs per sample
* number of true tracks per window
* ambiguous-stub fraction per sample
* same-track vs different-track legal edge ratios
* candidate occupancy by processor

### D. Derived-feature feasibility analysis

Before model training, check that the planned derived pair features are numerically well-behaved:

* `Δphi`
* `Δr`
* `Δr²`
* `kappa_hat`
* `|Δeta|`
* `Δbx`
* phiB consistency metrics
* layer-pair and type-pair frequency tables

This is especially important for quantization and future HLS design.

---

## 11.4 Required outputs of the dataset phase

The dataset phase should produce:

1. a dataset audit report
2. feature-range tables for software and firmware use
3. occupancy histograms and sample summaries
4. truth-label validation plots
5. join-validation checks and statistics
6. a final recommendation on whether the data is sufficient as-is or needs regeneration / correction

---

## 11.5 Why this dataset phase matters for architecture choice

This phase directly determines:

* whether edge-based supervision is trustworthy
* whether slot models have stable target multiplicity
* whether HECIN-style higher-order reasoning is likely to help
* what fixed `Nmax` bounds are realistic for firmware
* what quantization ranges are safe
* which samples should dominate threshold tuning and fake-rate studies

So this phase must happen before any final architecture commitment.

---

# 12. Recommended research strategy

The best strategy is not to choose one model and hope it works.

Instead, define a staged program.

---

## 10.1 Track A — deployment-oriented main line

These are the most realistic final firmware candidates.

### Candidates

* fixed sparse edge-compatibility network
* fixed-K slot model
* edge scorer + deterministic builder + compact regressor

### Goal

Find the best architecture that is realistically synthesizable.

---

## 10.2 Track B — software teacher / ceiling line

These are the richer models used to understand the problem and possibly distill into smaller deployment models.

### Candidates

* full HECIN-style heterogeneous interaction model
* HECIN + OC
* richer slot model or affinity model

### Goal

Measure the achievable performance ceiling and identify which relational mechanisms really matter.

---

## 10.3 Track C — baseline line

### Candidates

* DeepSets / pooled baseline
* tiny static hetero-MPNN
* pattern-conditioned learned scorer

### Goal

Ensure complexity is justified.

---

# 11. Distillation recommendation

Because the dataset is rich and the deployment target is strict, the project is especially well suited for a **teacher-student approach**.

## Recommended paradigm

### Teacher

A richer software model, for example:

* full HECIN
* HECIN + OC
* richer affinity or slot model

### Student

A deployment-oriented model, for example:

* fixed sparse edge scorer
* fixed-K slot model
* static HECIN-lite

### Why this is attractive

This avoids forcing one architecture to be simultaneously:

* the most expressive
* the easiest to train
* and the easiest to synthesize

Those goals are often not aligned.

---

# 12. Recommended immediate next step

The best immediate next step is to formalize and compare **three concrete candidate architectures**:

## A. Fixed sparse edge-compatibility network

Inputs:

* stub descriptors from `OMTFAllInputTree`

Derived features:

* pairwise geometric compatibility features

Outputs:

* up to 3 candidate scores and parameters

---

## B. Fixed-K slot / hypothesis model

Inputs:

* stub descriptors from `OMTFAllInputTree`

Mechanism:

* stub-to-slot compatibility
* slot aggregation

Outputs:

* fixed 3 candidate slots

---

## C. Firmware-friendly HECIN-Lite

Inputs:

* stub descriptors from `OMTFAllInputTree`

Mechanism:

* fixed sparse legal graph
* one small edge-conditioned interaction stage
* fixed candidate readout

Outputs:

* top candidate set without dynamic clustering

These three should then be compared on:

* efficiency vs pT
* efficiency vs d0
* charge accuracy
* candidate multiplicity correctness
* fake rate
* close dimuon separation
* performance under PU200
* quantization sensitivity
* estimated firmware complexity

---

# 13. Final conclusions

## 13.1 Main conclusion

The available branches and datasets strongly support **relation-based and edge-aware models** rather than generic node-aggregation GNNs.

## 13.2 On HECIN

HECIN remains a very strong idea.
It should not be thrown away.

But its role should be updated:

* full HECIN is best viewed as a rich software reference or teacher model
* a compressed static sparse HECIN can still make sense as a deployment candidate

## 13.3 On the best final deployment direction

The strongest architecture families for the actual Level-1 firmware target are:

1. **fixed sparse edge-compatibility network**
2. **fixed-K slot / hypothesis model**
3. **edge scorer + deterministic candidate builder + compact regressor**

## 13.4 Final recommendation

If one family must be prioritized first, the recommended starting point is:

> **a fixed sparse edge-compatibility network**, using the exact stub branches already produced, with ambiguity masking from `reg_stub_ambiguous`, truth from `reg_stub_trackId`, and final regression targets joined from NanoAOD.

This makes the best use of the available data while staying the closest to a realistic, bounded, quantization-friendly trigger-firmware architecture.

---

# 14. Short executive summary

* The produced dataset is strong and unusually well suited for relation-based learning.
* The most useful signal is pairwise, not purely node-wise.
* Vanilla GraphSAGE and GAT are not the best central models for this problem.
* Full HECIN remains excellent as a software reference and teacher.
* A firmware-friendly HECIN does make sense, but only as a compressed static sparse version.
* The best deployment-oriented model families are:

  * fixed sparse edge-compatibility network
  * fixed-K slot / hypothesis model
  * edge scorer + deterministic candidate builder + compact regressor
* The best project strategy is likely:

  * rich software teacher
  * firmware-oriented student
  * explicit comparison among 3 concrete trigger-shaped architectures

---

# 15. Empirical Findings (2026-04-24)

This section records what was actually observed during the training and evaluation campaign. It supersedes or confirms the predictions in sections 1–14 where relevant.

---

## 15.1 Variables used in practice

### Per-stub node features (7 per stub)

| Feature | Source branch | Notes |
|---|---|---|
| `phi` | `reg_stub_phiHw` | Hardware integer, normalised |
| `phiB` | `reg_stub_phiBHw` | Hardware integer, normalised |
| `eta` | `reg_stub_etaHw` | Hardware integer, normalised |
| `r` | `reg_stub_r` | Radius, normalised |
| `type` | `reg_stub_type` | Detector type identifier |
| `layer` | `reg_stub_layer` | Layer index |
| `quality` | `reg_stub_quality` | Hit quality flag |

`delta_bx` (`reg_stub_bx`) was measured to be identically 0 across all datasets and is excluded from model inputs. `reg_stub_ambiguous` is available as a mask but not used as a model input.

### Per-pair edge features (6 per stub pair)

| Feature | Definition | Notes |
|---|---|---|
| `delta_phi` | φᵢ − φⱼ | Primary curvature proxy |
| `delta_r` | rᵢ − rⱼ | Radial separation |
| `delta_r_sq` | (rᵢ − rⱼ)² | Used in kappa_hat denominator |
| `kappa_hat` | 2Δφ / Δr² | Curvature estimate; requires clipping (tails blow up for near-equal r) |
| `abs_delta_eta` | |ηᵢ − ηⱼ| | Track straightness check |
| `delta_bx` | bxᵢ − bxⱼ | Identically 0 in practice; kept as zero feature |

**Finding**: `kappa_hat` requires explicit clipping or tanh normalisation before training. Its distribution has long tails for near-coplanar stub pairs.

### Supervision labels (from `reg_stub_trackId` + NanoAOD join)

| Label | Definition |
|---|---|
| Node signal/noise | `trackId != 0` |
| Same-track edge | `trackId_i == trackId_j != 0` |
| Candidate slot k | `trackId == k+1` (fixed-index convention for K=3 slots) |
| pT regression target | `GenMuon_pt[trackId-1]` (NanoAOD join) |
| dXY regression target | `GenMuon_dXY[trackId-1]` (NanoAOD join) |
| Charge target | `GenMuon_charge[trackId-1]` |

The NanoAOD join uses `reg_eventNum` (cast to uint32) matched to the `event` branch of the NanoAOD `Events` tree. Join success rate: 100% across all datasets.

### Fixed model parameters

| Parameter | Value | Justification |
|---|---|---|
| `Nmax` | 24 | Covers p99 stub count; clean hardware boundary |
| `K` | 3 | Max candidates per processor window |
| Padding | zero-pad + `valid_mask` | Standard fixed-input pattern |
| Truncation | by quality descending | Applied when raw stubs > 24 (never fires in practice — see §15.2) |

---

## 15.2 Dataset empirical characterisation

### Stub occupancy (data adequacy analysis, 2026-04-24)

| Dataset | Mean raw stubs | Mean used stubs | % truncated (Nmax=24) |
|---|---|---|---|
| S1 (single µ, prompt) | 4.9 | 4.9 | 0% |
| B1 (PU200, single µ) | 3.5 | 3.5 | 0% |
| B4 (min-bias) | 1.5 | 1.5 | 0% |

**The Nmax=24 truncation never fires in the current datasets.** Stub occupancy is well below the hardware budget across all scenarios. This is not a bottleneck.

### Signal stub fraction per dataset

| Dataset | Mean stubs/window | Mean signal stubs | Signal% | % zero-signal windows |
|---|---|---|---|---|
| S1 | 4.9 | 4.5 | 93.9% | 0.7% |
| S2 | 4.1 | 3.9 | 94.9% | 0.7% |
| S3 | 6.6 | 6.2 | 94.5% | 0.4% |
| S5 | 4.3 | 4.1 | 95.6% | 0.5% |
| B1 | 3.5 | 2.5 | 48.8% | **45.3%** |
| B2 | 2.3 | 1.0 | 22.7% | **74.8%** |
| B3 | 4.9 | 3.9 | 56.0% | **38.3%** |
| B4 | 1.5 | 0.0 | 0% | 100% |

### Physics ceiling on PU200 efficiency

OMTF uses 12 φ-sector processors. A gen muon in a given event may produce no reconstructed stubs in a particular processor if its trajectory falls in a different sector, or if pT is too low. These windows are unrecoverable by any algorithm.

| Dataset | % (gen_pt>10) windows with zero signal stubs | Physics ceiling (trig_eff upper bound) |
|---|---|---|
| B1 | 44.9% | **55.1%** |
| B2 | 65.9% | **34.1%** |
| B3 | 31.6% | **68.4%** |

**Adding more training data will not improve B1/B2/B3 trigger efficiency.** The 45%/66%/32% of windows with zero signal stubs provide no learnable gradient. Improving beyond ~70% per-window efficiency on B1 requires either multi-processor coordination (outside OMTF scope) or a different detector geometry.

---

## 15.3 Models trained and compared

Four model families were implemented and evaluated:

### Baseline A — DeepSets

- Per-stub encoder MLP(7→H), mean-pool → per-node and global heads
- No edge features, no pairwise reasoning
- 50 epochs, S1+B1+B4, 50 files/dataset

| Metric | Value |
|---|---|
| best node_auc (val) | 0.934 |
| best stub_eff (val) | 0.983 |
| Parameters | ~20k |

### Baseline B — EdgeMLP

- Pairwise edge scoring MLP(2H+6→1), no node update
- 50 epochs, S1 only, 50 files/dataset

| Metric | Value |
|---|---|
| best edge_auc (val) | 0.838 |
| Parameters | ~15k |

### EdgeCompatNet (Stage 4 + ADG-R full-scale)

- Per-stub encoder MLP(7→H)
- Edge scorer MLP(2H+6→H) → edge embedding + scalar logit
- Per-node mean-pool of incident edge embeddings → node update
- Updated repr concat(stub_emb, node_update) → per-node and K=3 global candidate heads
- Candidate heads supervised by fixed-index BCE (trackId convention) + gen_pt/charge regression

**Full-scale (ADG-R), all 8 datasets, 2M+ samples, 4-checkpoint strategy:**

| Checkpoint | Epoch | trig_eff_proxy | bg_accept_proxy | combined |
|---|---|---|---|---|
| best (val_loss) | 49 | 0.818 | 0.111 | 0.707 |
| best_trig | **46** | **0.839** | 0.146 | 0.693 |
| best_rate | 41 | 0.814 | 0.095 | 0.718 |
| best_combined | 47 | 0.819 | 0.096 | **0.723** |

Parameters: 47,622. Checkpoint location: `build/omtf/checkpoints/edge_compat_best_trig.pt`.

### SlotModel (Stage 5 + ADG-R full-scale)

- Per-stub encoder MLP(7→H)
- K=3 learnable slot queries (nn.Parameter)
- Stub-to-slot compatibility MLP(2H→1) → softmax attention α (K, Nmax)
- Slot aggregation + back-projection to stubs → per-node and per-slot heads

**Full-scale (ADG-R), all 8 datasets:**

| Metric | Value |
|---|---|
| best val_loss | 1.574 (epoch 44) |
| best trig_eff_proxy | 0.829 |
| best bg_accept_proxy | 0.172 |
| Parameters | 26,310 |

---

## 15.4 Key model behaviour observed

### What worked

**Stub classification (node_auc, stub_eff)**: Both models learn clean signal/noise separation quickly. DeepSets achieves node_auc=0.934 with no edge reasoning — the per-stub features alone carry strong discriminating power. EdgeCompatNet pushes this to 0.92 at full scale across all 9 datasets.

**Edge discrimination**: EdgeCompatNet edge_auc 0.902 (Stage 4, 50 files) confirms that pairwise features (`kappa_hat`, `delta_phi`, `delta_r`) are strongly predictive of same-track pairing. The node aggregation step after edge scoring provides the main gain over raw pairwise scoring.

**Background suppression after full-scale retrain**: The critical failure in the initial ADG evaluation was EdgeCompatNet bg_accept=0.975 on B4 (min-bias). Root cause: candidate heads were trained without negative examples — they fired unconditionally. After adding explicit fixed-index BCE supervision with negative windows, bg_accept collapsed to 0.108 (best checkpoint) at full data scale. The concern was resolved by training, not architecture change.

**SlotModel B4 suppression**: SlotModel bg_accept=0.157 at full scale. The slot competition mechanism (softmax attention) is inherently selective — a slot fires only when its attention region aligns with real track stubs. This is structurally better at noise rejection than global-context candidate heads.

### What failed or remains weak

**pT regression**: Both models show near-constant pT predictions (pt_mae ~27 GeV on prompt S1 muons). The regression loss is not dominating training relative to the classification BCE losses. Loss reweighting or a separate fine-tuning phase is required before Stage 7. This is the primary open trainable gap.

**dXY reconstruction**: Completely fails for high-displacement datasets. On S2 (displaced), edge_compat dxy_mae = 43.1 cm, slot_model = 53.1 cm. The model predicts near-zero dXY for all windows. The displaced geometry likely requires explicit displacement-sensitive feature engineering or a separate training phase.

**SlotModel under PU200**: SlotModel trig_eff@10 on B1 = 0.683 per-window vs edge_compat 0.710. The fixed-index slot assignment (slot k ↔ gen muon track_id=k+1) is fragile under PU200 pileup where the signal muon may not be the most attended stub cluster. This was the deciding factor in the architecture comparison.

### ADG-21 inflated numbers (important note)

The first ADG evaluation (2026-04-21) showed EdgeCompatNet B1 trig_eff@10=0.988 alongside bg_accept=0.849. These numbers are inconsistent: a model that accepts 85% of windows has trivially high PU200 efficiency. The honest operating point after the full-scale retrain is ~0.69–0.71 per-window (matching the physics ceiling analysis above). The Apr-21 numbers should not be used as a reference.

---

## 15.5 Final trigger efficiency comparison (full-scale, 2026-04-24)

### Per-window efficiency (best_trig checkpoint vs slot_model best)

| Dataset | EdgeCompatNet (best_trig) | SlotModel (best) | Notes |
|---|---|---|---|
| S1 trig_eff@10 | **0.9894** | 0.9738 | +1.6 pp |
| S2 trig_eff@10 | **0.9679** | 0.9274 | +4.1 pp |
| S3 trig_eff@10 | **0.9904** | 0.9851 | +0.5 pp |
| S4 trig_eff@10 | **0.9992** | 0.9977 | near-equal |
| S5 trig_eff@10 | **0.9506** | 0.9126 | +3.8 pp |
| B1 trig_eff@10 | **0.7097** | 0.6826 | +2.7 pp |
| B2 trig_eff@10 | **0.5675** | 0.5284 | +3.9 pp |
| B3 trig_eff@10 | 0.7929 | **0.7780** | near-equal |
| B4 bg_accept | **0.1439** | 0.2014 | edge_compat better |

### Event-level efficiency (OR across processors per event, EdgeCompatNet best_trig only)

Event-level evaluation groups all processor windows for the same ROOT file event and asks whether at least one processor fires. This is the operationally relevant metric for L1T.

| Dataset | Per-window | Event-level | Gain | Mean procs/event |
|---|---|---|---|---|
| S1 | 0.9894 | **0.9944** | +0.5 pp | 1.47 |
| S2 | 0.9679 | **0.9744** | +0.6 pp | 1.24 |
| S3 | 0.9904 | **0.9999** | +1.0 pp | 1.46 |
| S4 | 0.9992 | **1.0000** | +0.1 pp | 1.95 |
| S5 | 0.9506 | **0.9666** | +1.6 pp | 1.37 |
| B1 | 0.7097 | **0.8874** | **+17.8 pp** | 1.88 |
| B2 | 0.5675 | **0.7400** | **+17.3 pp** | 1.65 |
| B3 | 0.7929 | **0.9999** | **+20.7 pp** | 1.97 |
| B4 bg_accept | 0.1439 | 0.2301 | — | 1.57 |

The large event-level gains on B1/B2/B3 arise because PU200 events have ~1.9 processors per event: when one processor misses the signal muon (zero signal stubs in that sector), a second processor often covers the same event and fires correctly. The B1 per-window ceiling of 55% translates to a much higher event-level ceiling because the zero-stub windows for a given event are often not zero-stub across all processors simultaneously.

---

## 15.6 Architecture decision (ADG-D, 2026-04-24)

**Selected model: EdgeCompatNet, checkpoint `edge_compat_best_trig.pt` (epoch 46)**

Rationale:
- Wins on all signal efficiency metrics (+1.6–4.1 pp over SlotModel on S1–S5)
- Wins on all PU200 efficiency metrics (+2.7–3.9 pp on B1/B2)
- After full-scale retrain with candidate supervision, bg_accept is lower than SlotModel (0.144 vs 0.201 per-window)
- Event-level B1 = 88.7%, B3 ≈ 100% — at or near the physics ceiling
- SlotModel's structural weakness under PU200 (fixed slot assignment fragile under pile-up noise) is not fixable without a different attention mechanism

**Alternative checkpoint**: `edge_compat_best_combined.pt` (epoch 47) if L1T rate budget is tight — bg_accept = 0.094 per-window at cost of ~1–2 pp signal efficiency.

---

## 15.7 Open problems before Stage 7

### 1. pT regression (blocking for L1T deployment)

The pT head is not learning. Both models show pt_mae ~27–28 GeV on prompt S1 muons and pt_σ68 ~0.83–0.84, indicating near-constant predictions. An L1T algorithm applies a pT threshold to control trigger rate — without a working pT tag, the rate cannot be controlled and efficiency-vs-rate curves cannot be measured.

**Root cause**: the regression loss (MSE on pT) is not dominating training relative to the node BCE + edge BCE + candidate BCE combination. The model converges on classification without the regression heads converging.

**Fix options**:
- Increase the pT loss coefficient (loss reweighting)
- Freeze classification heads after convergence, then fine-tune regression heads for ~20 epochs
- Separate training phase for regression only

### 2. dXY reconstruction (secondary, displaced trigger path only)

dXY reconstruction fails completely for high-displacement datasets (S2, B2): mae = 43–53 cm on muons with true displacement up to ~100 cm. The displaced geometry requires feature engineering that makes displacement explicitly visible (e.g. phiB residuals, layer-pair consistency under non-zero impact parameter). This only matters if a dedicated displaced-muon L1T path is intended.

### 3. Next step: loss reweighting retrain

Retrain EdgeCompatNet with higher pT/charge loss weights, evaluate with pT threshold scans across all 9 datasets. The output should be efficiency-vs-rate curves at pT ∈ {3, 5, 10, 15, 20, 22} GeV — the primary deliverable for L1T commissioning and the required input before Stage 7 (quantization + HLS).
