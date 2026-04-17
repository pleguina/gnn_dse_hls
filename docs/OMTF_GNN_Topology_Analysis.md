# Graph Neural Network Analysis for OMTF Muon Reconstruction

## A Topology-First Study: What Graph Structure Is Optimal for the Overlap Muon Problem?

---

## 0. Scope & Philosophy

This document performs a *first-principles* analysis of the OMTF muon reconstruction problem through the lens of graph theory and geometric deep learning. The question is: **given the physical topology of the OMTF detector, what is the theoretically optimal GNN architecture—ignoring all hardware constraints—to perform muon track-finding and parameter regression in software?**

We work backward from the data structure, not from a pre-chosen architecture. We analyze the symmetries, the natural graph structure, the information flow requirements, and the multi-task nature (track-finding + regression), and only then derive what kind of network fits.

---

## 1. The OMTF Detection Topology as a Graph

### 1.1 What are the physical objects?

In one 120° Phase-2 OMTF processor, after input formation, we have:

- Up to **18 layers**, each potentially holding **up to 8 stubs** (though typically 0–2 meaningful ones)
- Each stub $s_i$ has features:  $\mathbf{f}_i = (\phi_i, \phi_{B,i}, \eta_i, q_i, r_i, \text{type}_i, \ell_i)$
- Typical event occupancy at PU200: ~5–12 valid stubs across 3–8 layers per muon; plus ~2–6 noise/PU stubs
- **Multi-muon events** within one processor region are common (up to 3 muons within 120°)

### 1.2 What is the natural graph?

There are several legitimate ways to build a graph from OMTF stubs. Let us enumerate them all and analyze their properties.

#### Option G1: Fully Connected Graph (Complete Graph)

$$G_\text{full} = (V, E) \quad \text{where } V = \{s_1, \ldots, s_N\}, \quad E = \{(s_i, s_j) \mid i \neq j\}$$

- **Nodes:** All valid stubs as nodes ($N \leq 36$ in theory; $\sim$8–15 typical)
- **Edges:** All pairs $\binom{N}{2}$
- **Edge count:** $\sim$28–105 for typical events

**Pros:** No information loss from graph construction; every pair of stubs can communicate directly.
**Cons:** Quadratic edge count; no inductive bias from detector geometry; the network must learn that most edges are irrelevant.

#### Option G2: Layer-Bipartite Graph (Cross-Layer Only)

$$E = \{(s_i, s_j) \mid \ell_i \neq \ell_j\}$$

- **Edges:** Only between stubs in *different* layers
- **Edge count:** Reduced vs complete; avoids within-layer connections that are never part of a single track

**Pros:** Enforces the physical constraint that a muon deposits exactly one stub per layer (so within-layer stubs are from different muons or noise). No track visits the same layer twice.
**Cons:** Still dense for many layers. Doesn't distinguish nearby vs distant layers.

#### Option G3: Sequential / Chain Graph (Layer-Ordered)

$$E = \{(s_i, s_j) \mid |\ell_i - \ell_j| = 1 \text{ or } \ell_i, \ell_j \text{ are a DT φ/φB pair}\}$$

- **Edges:** Only between adjacent layers in the inside-out ordering
- **Edge count:** Very sparse (~$N$ edges)

**Pros:** Mimics the physical propagation of a particle from one detector station to the next. Most similar to a Kalman filter.
**Cons:** Misses long-range correlations (MB1 ↔ MB3, or DT ↔ CSC). Information must propagate through many hops. Very fragile to missing intermediate layers.

#### Option G4: Hierarchical "Station + Detector" Graph

Group stubs first by physical station (MB1, MB2, MB3, ME1/2, etc.), then connect stations:

- **Level 1 nodes:** Station groups (up to 11 physical stations)
- **Level 1 edges:** Between adjacent stations in radius
- **Level 2 nodes:** Within each station, individual stubs
- **Level 2 edges:** Within-station connections (DT φ ↔ φB pairs)

**Pros:** Reflects the hierarchical nature of the detector (DT station = chamber with both phi and phiB subsystems).
**Cons:** Requires hierarchical message passing, which is architecturally more complex.

#### Option G5: Physics-Informed Radius-Distance Graph

$$E = \{(s_i, s_j) \mid \lVert r_i - r_j \rVert > r_\min \text{ AND } |\phi_i - \phi_j| < \phi_\text{cut}(\kappa_\text{max})\}$$

- **Edges:** Connect stubs that are (a) in different radial regions and (b) within the maximum possible $\Delta\phi$ for the lowest-pT muon of interest (~2 GeV)
- **Edge construction:** Pre-computed from the acceptance window of a helical trajectory

**Pros:** Highly informative inductive bias — every edge is a *plausible* track segment. False edges are suppressed by geometric cut. This is the approach used by the CMS track-finding GNNs (e.g., Exa.TrkX).
**Cons:** Requires careful definition of $\phi_\text{cut}$; too tight loses hard-scattering tails; too loose approaches the complete graph.

#### Option G6: Bipartite "Stub ↔ Track Hypothesis" Graph (Object Condensation style)

$$V = V_\text{stubs} \cup V_\text{hypotheses}, \quad E = V_\text{stubs} \times V_\text{hypotheses}$$

- **Stub nodes:** Physical stubs with measured features
- **Hypothesis nodes:** $K$ learnable track hypothesis slots (e.g., $K = 3$ matching the ghost buster max)
- **Edges:** Every stub connects to every hypothesis; assignment is learned

**Pros:** Directly models the multi-track problem. Each hypothesis slot learns to "attract" the stubs belonging to one muon. Naturally handles track-finding + assignment jointly.
**Cons:** Needs careful initialization of hypothesis node features; the value of $K$ must be chosen or made dynamic.

### 1.3 Topology Analysis: What Properties Does the Optimal Graph Need?

Let us examine the mathematical structure of the OMTF problem to determine which graph properties are actually required.

**Property 1: The graph must support "all-to-all angular comparison."**

The curvature $\kappa$ of a charged particle is encoded in the *pairwise* $\Delta\phi / \Delta r^2$ between *any two* layers. Any pair of position-measuring stubs at different radii can contribute a curvature estimate:

$$\hat{\kappa}_{ij} \approx \frac{2 (\phi_i - \phi_j)}{r_i^2 - r_j^2}$$

This means long-range edges (MB1 ↔ MB3, DT ↔ CSC) are as informative as short-range ones. A purely sequential graph (G3) would require multiple hops to compute this, diluting the gradient signal.

**Conclusion:** The graph should have *at least* cross-layer connectivity strong enough that any two non-adjacent layers can communicate within 1–2 message-passing rounds.

**Property 2: The graph must distinguish position-measuring vs bending-measuring stubs.**

DT phi and DT phiB at the *same* station are fundamentally different measurements. They are trivially "connected" (paired within a station), but the phiB provides curvature information that is independent of bending between layers:

$$\phi_B(r_i) \approx \kappa \cdot r_i \quad \text{(d₀-independent)}$$

This means the graph should make the **intra-station DT φ ↔ DT φB pairing** a first-class structural element, not just another edge to be learned.

**Conclusion:** Use *heterogeneous* node types or at minimum a dedicated edge type for intra-station phi-phiB pairs.

**Property 3: The graph should be permutation-equivariant within layers.**

If two stubs appear in the same layer, they come from different muons (or one is noise). The network should process them symmetrically. This is naturally enforced by GNNs (message-passing is node-permutation equivariant), but rules out architectures that depend on a fixed ordering of stubs within a layer.

**Property 4: The graph must handle variable-size input elegantly.**

The number of valid stubs varies from 3 (a marginal muon grazing just a few layers) to ~30 (multi-muon in PU200). The architecture must not have a fixed input dimension. This eliminates fixed MLPs and favors set-like or graph-like architectures.

**Property 5: The output must support multiple tracks.**

The current OMTF outputs up to 3 candidates per processor. The GNN must perform *track labeling* (which stubs belong to which muon) alongside *parameter regression* (pT, charge, d₀ per track). This is fundamentally a node/edge classification + regression problem.

### 1.4 Verdict: Optimal Base Graph

**The optimal base graph for OMTF is a physics-informed cross-layer graph (G5) enriched with intra-station phiB pairing, operating within a heterogeneous edge framework.**

Specifically:

$$G^* = (V, E_\text{cross} \cup E_\text{intra})$$

Where:
- **$V$:** All valid stubs as nodes (variable $N$, typically 5–20)
- **$E_\text{cross}$:** Edges between stubs in different layers where $|\Delta\phi| < \phi_\text{window}(\text{det\_type}_i, \text{det\_type}_j, \ell_i, \ell_j, q_\text{class})$. The window is precomputed per (detector-type pair, layer pair) to reflect the intrinsic φ-resolution of each detector type (RPC: coarse, DT/CSC: fine) and the lever-arm geometry of the specific radial pair. An optional quality-class factor widens the window for low-quality stubs and tightens it for high-quality ones. A single global $\phi_\text{cut}(\kappa_\min = 2\text{ GeV})$ is *not* used because it is simultaneously too loose for noisy pairs (e.g. RPC–RPC) and too tight for valid but geometrically awkward cases (missing-layer events, boundary stubs, displaced patterns).
- **$E_\text{intra}$:** Edges between DT φ and DT φB within the same station (L0↔L1, L2↔L3, L4↔L5). These carry curvature information.

**Degree cap (safety valve):** Even with the adaptive window, RPC-involving pairs in high-PU events can produce pathologically dense local connections. After window filtering, each node retains at most $k = 4$ cross-layer edges *per layer it connects to*, keeping the $k$ candidates with smallest $|\Delta\phi|$ (equivalently, those most consistent with the softest plausible track). This prevents fanout from blowing up while preserving the edges most likely to belong to a real track. Intra-station edges are exempt (at most 1 per DT station by construction).
- Cross-layer edges: ~30–40 (much less than $\binom{10}{2} = 45$)
- Intra-station edges: ~3 (one per DT station present)
- Total: ~35–45 edges

This is sparse enough for efficient processing and rich enough for full track reconstruction.

---

## 2. Architecture Analysis: Which GNN Paradigm?

### 2.1 The Taxonomy of GNN Approaches

There are three major paradigms for applying GNNs to track reconstruction:

| Paradigm | What the network classifies | Track formation | Examples |
|----------|---------------------------|-----------------|----------|
| **Edge Classification** | Each edge: "does this edge belong to a real track?" | Post-process: walk-through or connected-component on true edges | Exa.TrkX, TrackML GNN |
| **Node Classification / Clustering** | Each node: "which track does this stub belong to?" | Cluster nodes by predicted label | Object Condensation, GravNet |
| **Graph-Level Regression** | Whole graph → regression outputs | Pool all nodes; output (pT, charge, d₀) per track slot | Attention-based pooling models |

Let us analyze each for the OMTF topology.

### 2.2 Edge Classification Network (ECN)

**How it works:** Build graph; run message passing; for each edge, predict $p(\text{true track segment}) \in [0,1]$. Filter edges by threshold. Extract tracks by following connected high-score edges.

**For OMTF:** Each edge $(s_i, s_j)$ represents a candidate 2-hit segment. The ECN learns whether pairs of stubs are consistent with a single muon track.

**Edge features (physics-motivated):**

| Feature | Formula | Physical meaning |
|---------|---------|-----------------|
| $\Delta\phi$ | $\phi_i - \phi_j$ | Position difference → curvature |
| $\Delta r$ | $r_i - r_j$ | Radial lever arm |
| $\Delta\phi / \Delta r^2$ | $\frac{\phi_i - \phi_j}{r_i^2 - r_j^2}$ | Approximate curvature $\hat{\kappa}$ |
| $\phi_{B,i} / r_i$ | (DT only) | Local curvature at station $i$ |
| $\phi_{B,j} / r_j$ | (DT only) | Local curvature at station $j$ |
| $\Delta(\phi_B/r)$ | $\frac{\phi_{B,i}}{r_i} - \frac{\phi_{B,j}}{r_j}$ | Curvature consistency between stations |
| Layer pair type | Categorical | DT-DT, DT-CSC, DT-RPC, CSC-RPC, etc. |
| $|\Delta\eta|$ | $|\eta_i - \eta_j|$ | Compatibility in pseudorapidity |

**Strengths for OMTF:**

1. **Multi-track by design.** Each track is a connected subgraph of true edges. No need for a fixed max-track count.
2. **Physics-interpretable edges.** Each edge is a concrete 2-hit segment with computable features. The network learns to recognize valid segments.
3. **Proven in CMS tracking.** The Exa.TrkX project demonstrated >99% edge efficiency for tracker hits.

**Weaknesses for OMTF:**

1. **Post-processing needed** to extract tracks from edge scores. Connected-component algorithms or walk-through are required.
2. **Track parameter regression** is not built-in. After edge classification, a separate step is needed to fit (κ, φ₀, d₀) to each track cluster.
3. **Sparse events (3–4 stubs, 1 muon)** mean the graph is trivially connected — edge classification adds no value over just fitting all stubs.

**Verdict:** Strong for the multi-muon disambiguation subtask, but needs a separate regression stage. Not a single end-to-end solution.

### 2.3 Node Classification / Object Condensation Network

**How it works:** Each node (stub) is assigned to a track instance via a learned clustering in a latent space. The Object Condensation (OC) approach (Kieseler 2020) maps each node to:
- A *condensation point* coordinate $\mathbf{c}_i \in \mathbb{R}^d$
- A *beta* score $\beta_i \in [0,1]$ indicating whether the node is a "condensation seed" (track center)

Stubs from the same track cluster around the same condensation point. Seeds (high $\beta$) represent the track; their latent features are decoded to track parameters.

**For OMTF:**

The graph is the physics-informed $G^*$. Message passing updates node embeddings:

$$\mathbf{h}_i^{(t+1)} = \phi_\text{node}\left(\mathbf{h}_i^{(t)},\, \bigoplus_{j \in \mathcal{N}(i)} \psi_\text{edge}(\mathbf{h}_i^{(t)}, \mathbf{h}_j^{(t)}, \mathbf{e}_{ij})\right)$$

After $T$ rounds, each node outputs:
- **Clustering coordinates** $\mathbf{c}_i \in \mathbb{R}^2$ (project stubs from the same track to the same point)
- **Beta** $\beta_i$ (seed indicator)
- **Track parameters** $(\hat{\kappa}_i, \hat{\phi}_{0,i}, \hat{d}_{0,i}, \hat{q}_i)$ (regression from each node; averaged over cluster for final output)

**Object Condensation Loss:**

$$\mathcal{L}_\text{OC} = \lambda_\text{att} \sum_{k} \sum_{i \in T_k} \lVert \mathbf{c}_i - \mathbf{c}_{\alpha_k} \rVert^2 + \lambda_\text{rep} \sum_{k \neq l} \max(0, M - \lVert \mathbf{c}_{\alpha_k} - \mathbf{c}_{\alpha_l} \rVert)$$

where $T_k$ is the set of stubs belonging to track $k$, $\alpha_k$ is the seed node for track $k$, and $M$ is a margin.

**Strengths for OMTF:**

1. **Inherently multi-track.** Each track is a cluster; the number of tracks is determined by the number of high-β seeds.
2. **End-to-end.** Track-finding (clustering) and track parameter regression happen simultaneously.
3. **Handles variable track count.** No fixed $K$ for maximum tracks.
4. **Naturally produces a quality metric.** $\beta$ and cluster tightness serve as quality scores.

**Weaknesses for OMTF:**

1. **Complex loss function.** Requires careful tuning of attraction/repulsion balance.
2. **Inference-time clustering.** Need a post-process to extract cluster assignments (e.g., DBSCAN on condensation space).
3. **Overkill for simple events.** Single-muon events (70%+ of OMTF inputs) don't truly benefit from the OC machinery.

**Verdict:** Theoretically optimal for the multi-track problem. The strongest end-to-end approach for handling 0–3 muons per processor.

### 2.4 Interaction Network / Edge Network (Hybrid)

**How it works:** An Interaction Network (Battaglia et al. 2016) has two coupled networks:
- **Edge network** $\phi_e$: computes a message on each edge from the two node features + edge features
- **Node network** $\phi_v$: updates each node from its current state + aggregated incoming messages

$$\mathbf{m}_{ij}^{(t)} = \phi_e(\mathbf{h}_i^{(t)}, \mathbf{h}_j^{(t)}, \mathbf{e}_{ij})$$
$$\mathbf{h}_i^{(t+1)} = \phi_v\left(\mathbf{h}_i^{(t)},\, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(t)}\right)$$

This is the most general MPNN formulation. By suitable choices of $\phi_e$ and $\phi_v$, it subsumes GAT, GCN, GraphSAGE, etc.

**For OMTF, the key questions are:**

1. **What features should edge messages carry?**
2. **How many message-passing rounds?**
3. **What aggregation (sum, mean, max, attention-weighted)?**
4. **What readout for track parameters?**

### 2.5 GravNet / Dynamic Graph Architecture

**How it works:** GravNet (Qasim et al. 2019, originally for calorimetry) builds a *dynamic* graph from learned coordinates:

1. Project each node to a "spatial" coordinate $\mathbf{s}_i \in \mathbb{R}^S$ via a learned linear map
2. Build a $k$-NN graph in this learned coordinate space
3. Aggregate neighbor features with distance-weighted kernel: $w_{ij} = \exp(-\lVert \mathbf{s}_i - \mathbf{s}_j \rVert^2)$
4. Update node features

The graph topology is **not fixed** — it changes per layer of the network based on learned coordinates.

**For OMTF:**

**Pros:** Extremely powerful for discovering non-trivial clustering patterns. The network can learn to group stubs by track in a physics-informed latent space. Combined with Object Condensation, this gives the state-of-the-art for calorimeter clustering at CMS (used in HGCAL).

**Cons:** The dynamic graph construction requires $k$-NN search at each layer, which is expensive. For the small OMTF graph (N ~10–20), a fixed graph with full cross-layer connectivity is cheap, making dynamic construction unnecessary overhead.

**Verdict:** Elegant but over-engineered for the small OMTF graph. Better suited to problems with $N > 100$ (calorimeter, tracker).

---

## 3. The Optimal Architecture: Heterogeneous Interaction Network with Object Condensation

### 3.1 Architecture Summary

Given the analysis above, the optimal software-only architecture for OMTF is:

$$\boxed{\text{Heterogeneous Edge-Conditioned Interaction Network (HECIN)} + \text{Object Condensation Readout}}$$

This combines:
1. **Heterogeneous edges** (cross-layer vs intra-station PhiB) with edge-conditioned message passing
2. **Interaction Network** backbone (explicit edge MLP + node MLP)
3. **Object Condensation** output for simultaneous track-finding and parameter regression
4. **Physics-informed edge features** computed from detector geometry

### 3.2 Full Architecture Specification

```
═══════════════════════════════════════════════════════════════
                    HECIN + OC for OMTF
═══════════════════════════════════════════════════════════════

INPUT:
  Nodes: N valid stubs × F_node features
  Edges: ~40 edges × F_edge features  (constructed from G*)
  Edge types: {CROSS_LAYER, INTRA_STATION_PHIB}

GRAPH CONSTRUCTION (pre-network, deterministic):
  1. Cross-layer edges: connect stubs in different layers where
     |Δφ| < φ_window(det_type_i, det_type_j, layer_i, layer_j, quality_class)
     Window is looked up from a precomputed table indexed by
     (detector-type pair, layer pair), accounting for per-type φ-resolution
     (RPC ≈ 4× coarser than DT/CSC) and radial lever-arm geometry.
     An optional quality-class factor scales the window to handle
     low-quality stubs and displaced/boundary patterns without a single
     global φ_cut(κ_min=2 GeV) that fails for noisy or bad-geometry pairs.
     Degree cap: after window filtering, retain at most k=4 edges per
     (node, target-layer) pair, keeping the k smallest by |Δφ|.
     Prevents pathological fanout for RPC-involving pairs at high PU.
  2. Intra-station edges: connect DT φ stub ↔ DT φB stub
     in same station (L0↔L1, L2↔L3, L4↔L5)
  3. Compute edge features (see §3.4)

NODE ENCODER (parallel over all nodes):
  x_i^(0) = MLP_encode(f_i)
  // f_i: [phi_rel, phiB, eta, quality, r, type_onehot(11), layer_emb(6), bx]
  //   23 dims total; see §3.3 for encoding rationale
  // MLP_encode: F_node → 64 → 64 (ReLU activations)
  // Output: d = 64 per node

EDGE ENCODER (parallel over all edges):
  e_ij^(0) = MLP_edge_encode(a_ij)
  // a_ij: edge features (see §3.4)
  // MLP_edge_encode: F_edge → 32 → 32
  // Output: d_e = 32 per edge

MESSAGE PASSING (T = 3 rounds):
  For t = 1, ..., T:
    // Edge update (edge network)
    For each edge (i, j):
      m_ij^(t) = MLP_edge^(t)(concat(x_i^(t-1), x_j^(t-1), e_ij^(t-1)))
      // MLP_edge: (64 + 64 + 32) → 64 → 32
      e_ij^(t) = e_ij^(t-1) + m_ij^(t)  // residual connection on edges

    // Node update (node network)
    For each node i:
      // Aggregation with edge-type-aware attention
      α_ij = softmax_over_j(score(x_i^(t-1), m_ij^(t), type_ij))
      a_i = Σ_j α_ij · m_ij^(t)
      x_i^(t) = x_i^(t-1) + MLP_node^(t)(concat(x_i^(t-1), a_i))
      // MLP_node: (64 + 32) → 64 → 64
      // Residual connection

OUTPUT HEADS (per node):
  // Object Condensation
  c_i = Linear_cond(x_i^(T))     // ℝ^2 condensation coordinates
  β_i = sigmoid(Linear_beta(x_i^(T)))  // seed score ∈ [0,1]

  // Track parameter regression
  κ_i = Linear_kappa(x_i^(T))    // signed curvature
  φ0_i = Linear_phi(x_i^(T))     // vertex phi
  d0_i = Linear_d0(x_i^(T))      // impact parameter
  q_i = sigmoid(Linear_charge(x_i^(T)))  // charge probability

  // Per-stub quality / noise score
  noise_i = sigmoid(Linear_noise(x_i^(T)))  // 1 = noise, 0 = real

EDGE CLASSIFICATION (bonus: optional):
  For each edge (i, j):
    p_edge_ij = sigmoid(MLP_edge_class(e_ij^(T)))
  // 1 = same track, 0 = different tracks

POST-PROCESSING (track extraction):
  1. Select seeds: nodes where β_i > β_threshold (e.g., 0.5)
  2. For each seed α_k, cluster all nodes with ||c_i - c_αk|| < r_cluster
  3. Average track parameters over cluster members:
     κ_track = Σ(β_i · κ_i) / Σ(β_i)   for i in cluster
     (similarly for φ0, d0, charge)
  4. Quality = function of cluster size, max(β_i), and residual variance
  5. Sort tracks by quality; output top 3

═══════════════════════════════════════════════════════════════
```

### 3.3 Node Features (Detailed)

| Feature | Symbol | Dim | Encoding | Physical meaning |
|---------|--------|-----|----------|-----------------|
| Relative phi | $\Delta\phi_i$ | 1 | Float, normalized to [-1,1] | Position relative to processor center |
| Bending angle | $\phi_{B,i}$ | 1 | Float, normalized; 0 for non-DT | Local curvature measurement |
| Pseudorapidity | $\eta_i$ | 1 | Float, normalized | Longitudinal direction |
| Quality | $q_i$ | 1 | Float, normalized 0–1 | Measurement confidence |
| Radius | $r_i$ | 1 | Float, normalized (by $r_\text{max} = 700$ cm) | Radial position of detector layer |
| Stub type | $\text{type}_i$ | 11 | One-hot encoding of `MuonStub::Type` enum | DT_PHI, DT_PHI_ETA, RPC, CSC_PHI, etc. Fully implies bending identity; `is_bending` flag dropped. |
| Layer index | $\ell_i$ | 6 | Learned embedding (table: 18 logic layers × 6 dims) | Which layer this stub belongs to. Full one-hot (18 dims) avoided: OC can shortcut-cluster same-layer stubs along the layer-index axis, which is counter-productive since same-layer stubs belong to *different* tracks. |
| BX | $\text{bx}_i$ | 1 | Integer, centered at 0 | Timing info |
| **Total** | | **23** | | |

> **Note on dropped/compressed features:** `is_bending` is omitted because it is fully determined by `type` — only DT phiB pseudo-layer types are bending stubs. Including it adds a redundant axis. The 18-dim one-hot layer index is replaced by a 6-dim learned embedding to prevent OC from trivially clustering stubs by layer identity: a degenerate shortcut that would group same-layer stubs (which come from *different* tracks) into the same condensation cluster early in training, when the attractive loss is strongest.

### 3.4 Edge Features (Detailed)

#### Cross-Layer Edges

| Feature | Symbol | Formula | Physical meaning |
|---------|--------|---------|-----------------|
| Δφ | $\Delta\phi_{ij}$ | $\phi_i - \phi_j$ | Angular separation → curvature indicator |
| |Δr| | $|r_i - r_j|$ | Radial lever arm |
| Δr² | $r_i^2 - r_j^2$ | For curvature formula |
| Pairwise κ estimate | $\hat{\kappa}_{ij}$ | $\frac{2(\phi_i - \phi_j)}{r_i^2 - r_j^2}$ | Direct curvature from this stub pair |
| PhiB consistency (if DT-DT) | $\delta_{\kappa B}$ | $\phi_{B,i}/r_i - \phi_{B,j}/r_j$ | Do both DT stations see the same curvature? Set to **0 for non-DT–DT pairs**; a companion binary flag `has_phiB_consistency` (1 for DT–DT, 0 otherwise) is appended to avoid the network mis-interpreting the zero as "perfectly consistent". Alternatively use an edge-type-specific MLP branch that simply omits this feature for non-DT–DT edges. |
| |Δη| | $|\eta_i - \eta_j|$ | Should be ~0 for same track |
| ΔBX | $|bx_i - bx_j|$ | Should be 0 for in-time track |
| Layer pair category | — | 6-dim **learned embedding** (table: 15 pair types × 6 dims) | Edge type conditioning. Embedding preferred over one-hot for the same reason as the layer-id embedding: one-hot categorical axes give the network a shortcut to separate edge types in attention before reading the physics features. |
| **Total** | | **8 continuous + 6-dim embedding** | |

#### Intra-Station PhiB Edges (DT φ ↔ DT φB at same station)

| Feature | Symbol | Formula | Physical meaning |
|---------|--------|---------|-----------------|
| φB value | $\phi_B$ | From the phiB stub | Direct curvature |
| κ from φB | $\hat{\kappa} = \phi_B / r$ | Local curvature | d₀-independent curvature |
| Quality of φ stub | $q_\phi$ | — | Position precision |
| Quality of φB stub | $q_B$ | — | Bending precision |
| Station index | 0/1/2 | MB1/MB2/MB3 | Which DT station |
| **Total** | | **5** | |

### 3.5 Why 3 Message Passing Rounds?

In mathematics, the *receptive field* of a node after $T$ rounds of message passing is all nodes within $T$ hops in the graph. For our physics-informed graph $G^*$:

- **$T = 1$:** Each node sees its direct neighbors. A DT stub can "see" the CSC stubs it's connected to, and vice versa. This is enough for pairwise curvature estimation but not for consistency across >2 layers.

- **$T = 2$:** Each node sees neighbors-of-neighbors. MB1 can "see" MB3 via MB2, even if there's no direct MB1↔MB3 edge. This allows 3-station fit reasoning.

- **$T = 3$:** Full receptive field covers the entire graph (diameter of the typical OMTF graph is ≤3 given the cross-layer connectivity). Every node has information from every other node. This is the minimum for:
  - Full-track consistency checks
  - d₀ inference (requires comparing curvature from phiB with position shifts across 3+ layers)
  - Multi-muon disambiguation (stubs from different tracks must exchange information to realize they belong to different clusters)

- **$T = 4+$:** Diminishing returns. Over-smoothing risk (node embeddings converge). The OMTF graph is small enough that 3 rounds approaches global coverage.

**Starting point:** $T = 3$ covers the worst-case graph diameter and maps to a well-understood 3-stage pipeline. However, with the physics-informed $G^*$, many real events already have graph diameter 2 (e.g. a DT MB1 stub is directly connected to MB3 via its own cross-layer edge). In that regime, the extra round can begin homogenising embeddings — the oversmoothing failure mode that reduces inter-cluster separability in condensation space and degrades OC clustering.

**Treat $T$ as a hyperparameter.** Measure $T = 2$ vs $T = 3$ with everything else held equal. If $T = 2$ matches performance, prefer it: it reduces oversmoothing risk and meaningfully simplifies any future firmware adaptation.

### 3.6 Why Attention-Weighted Aggregation?

The node update aggregates messages from neighbors. The three common choices are:

| Aggregation | Formula | Property |
|-------------|---------|----------|
| Sum | $\mathbf{a}_i = \sum_j \mathbf{m}_{ij}$ | Preserves cardinality info; high-degree nodes get larger vectors |
| Mean | $\mathbf{a}_i = \frac{1}{|\mathcal{N}(i)|} \sum_j \mathbf{m}_{ij}$ | Robust to variable degree; loses count info |
| Attention | $\mathbf{a}_i = \sum_j \alpha_{ij} \mathbf{m}_{ij}$ | Learns which neighbors are most informative |

For OMTF, **attention-weighted aggregation is optimal** because:

1. **Not all neighbors are equally informative.** A high-quality DT MB1 stub paired with a low-quality RPC RB3 stub should weight the MB1 contribution much higher. Attention learns this automatically.

2. **Edge types matter.** An intra-station phiB edge carries curvature information of a fundamentally different nature than a cross-layer Δφ edge. Edge-type-conditioned attention allows the network to learn different weighting schemes per edge type.

3. **Noise rejection.** In PU200, some stubs are from pileup. Attention weights naturally suppress irrelevant stubs (those whose messages don't "fit" with the rest of the track hypothesis).

4. **The graph is small.** Attention over 5–15 neighbors per node is cheap. The overhead of computing attention scores is negligible vs the information gain.

**Attention mechanism:**

$$\alpha_{ij} = \frac{\exp(a_{ij})}{\sum_{j' \in \mathcal{N}(i)} \exp(a_{ij'})}$$

where the unnormalized attention score is:

$$a_{ij} = \text{LeakyReLU}\left(\mathbf{w}_\text{att}^\top \cdot [\mathbf{h}_i \| \mathbf{m}_{ij} \| \text{emb}_\text{type}(\text{type}_{ij})]\right)$$

where $\text{emb}_\text{type}$ is the same learned 6-dim pair-type embedding table used in edge encoding (see §3.4). Reusing the embedding rather than a separate one-hot avoids adding a new shortcut axis in the attention scorer.

### 3.7 Why Object Condensation Over Direct Regression?

**Alternative 1: Direct graph-level regression** (pool all node embeddings → MLP → track parameters)

This fails because:
- It requires a fixed number of output tracks (e.g., 3 slots)
- "Empty" slots are hard to handle without explicit occupancy prediction
- A single pooled vector loses per-stub identity, making it hard to compute quality per track

**Alternative 2: Edge classification** → track extraction → separate regression

This works but:
- Requires a two-stage pipeline (edge scoring then fitting)
- The track extraction post-processing adds complexity
- The edge classifier doesn't "know" about the downstream regression, so can't optimize for it

**Object Condensation combines both in one loss:**
- The clustering components of OC perform the role of edge classification (stubs from the same track cluster together)
- The regression heads operate on every node, but the loss weights them by cluster membership
- The $\beta$ score provides a natural quality metric

**Conclusion:** OC is the most elegant end-to-end formulation for a multi-track GNN on the OMTF graph.

---

## 4. Detailed Loss Function Design

### 4.1 Multi-Component Loss

$$\mathcal{L} = \lambda_\text{att} \mathcal{L}_\text{attract} + \lambda_\text{rep} \mathcal{L}_\text{repulse} + \lambda_\text{\beta} \mathcal{L}_\text{\beta} + \lambda_\text{noise} \mathcal{L}_\text{noise} + \lambda_\text{pt} \mathcal{L}_\text{pt} + \lambda_\text{charge} \mathcal{L}_\text{charge} + \lambda_\text{d0} \mathcal{L}_\text{d0} + \lambda_\text{phi} \mathcal{L}_\text{phi} + [\lambda_\text{edge} \mathcal{L}_\text{edge}]$$

### 4.2 Attraction Loss

$$\mathcal{L}_\text{attract} = \frac{1}{N_\text{signal}} \sum_{k=1}^{K} \sum_{i \in T_k} \lVert \mathbf{c}_i - \mathbf{c}_{\alpha_k} \rVert^2 \cdot \beta_{\alpha_k}$$

where $T_k$ is the set of truth-matched stubs for track $k$, and $\alpha_k = \arg\max_{i \in T_k} \beta_i$ is the seed (highest-β node in the truth cluster).

**Interpretation:** Pulls all stubs from the same track toward their seed's condensation point.

### 4.3 Repulsion Loss

$$\mathcal{L}_\text{repulse} = \frac{1}{K(K-1)} \sum_{k=1}^{K} \sum_{l \neq k} \max\left(0,\, M - \lVert \mathbf{c}_{\alpha_k} - \mathbf{c}_{\alpha_l} \rVert \right)$$

where $M$ is a margin (e.g., $M = 1.0$).

**Interpretation:** Pushes condensation seed points from different tracks apart by at least margin $M$.

### 4.3.1 Beta Push Loss

Without explicit supervision, $\beta$ tends to converge toward “many medium-$\beta$ nodes” rather than a few strong seeds, making seed selection at inference unreliable. The beta push term directly encourages each true track cluster to have at least one high-$\beta$ seed, and penalises large $\beta$ on noise stubs:

$$\mathcal{L}_\beta = -\frac{1}{K} \sum_{k=1}^{K} \log\left(\max_{i \in T_k} \beta_i\right) + \frac{\lambda_\text{noise-\beta}}{N_\text{noise}} \sum_{i \in \text{noise}} \beta_i$$

- **First term:** Encourages the highest-$\beta$ stub in each truth cluster $T_k$ to approach 1. Using $\log(\max \beta_k)$ rather than penalising all non-seed stubs avoids confusion with the attraction loss (any stub in the cluster can be the seed).
- **Second term:** Suppresses $\beta$ on stubs truth-labelled as noise. Complementary to the existing noise classification head: the noise head learns *which* stubs are noise; the $\beta$-push ensures those stubs also have low seed probability.

The noise head already provides substantial beta-on-noise suppression implicitly, so $\lambda_\text{noise-\beta} \approx 0.2$ suffices — enough to stabilise seed selection without over-constraining the network.

**Interpretation:** Creates a soft, two-ended pressure: pull seed $\beta$ high for true tracks; push $\beta$ low for noise stubs. Combined with the noise head, this eliminates the degenerate "many medium-$\beta$" failure mode.

$$\mathcal{L}_\text{noise} = - \frac{1}{N} \sum_i \left[ y_i^\text{noise} \log(\hat{n}_i) + (1 - y_i^\text{noise}) \log(1 - \hat{n}_i) \right]$$

Binary cross-entropy on the per-stub noise probability $\hat{n}_i$.

### 4.5 Track Parameter Regression Losses

Applied only to non-noise stubs, weighted by $\beta_i$ (so the seed's prediction matters most):

$$\mathcal{L}_\text{pt} = \frac{1}{N_\text{signal}} \sum_{k} \sum_{i \in T_k} \beta_i \cdot \text{Huber}(\log |\hat{\kappa}_i| - \log |\kappa_i^\text{true}|)$$

Log-space pT (equivalently $|\kappa|$) regression ensures uniform relative resolution across the full pT range.

$$\mathcal{L}_\text{charge} = \frac{1}{N_\text{signal}} \sum_{i \in \text{signal}} \beta_i \cdot \text{BCE}(\hat{q}_i, q_i^\text{true})$$

$$\mathcal{L}_\text{d0} = \frac{1}{N_\text{signal}} \sum_{i \in \text{signal}} \beta_i \cdot \text{Huber}(\hat{d}_0 - d_0^\text{true})$$

$$\mathcal{L}_\text{phi} = \frac{1}{N_\text{signal}} \sum_{i \in \text{signal}} \beta_i \cdot L_1(\hat{\phi}_0 - \phi_0^\text{true})$$

> **β-weighting guard.** Weighting regression by $\beta_i$ is correct at convergence, but early in training $\beta$ can collapse toward zero, causing regression heads to receive vanishing gradients and stop learning entirely. Two mitigations:
> - **Preferred:** Train regression heads *unweighted* (i.e. $w_i = 1$) during Phase 1 (§8.2); switch to $\beta$-weighted from Phase 2 onward once $\beta$ is producing meaningful seed scores.
> - **Alternative:** Clamp the weight: $w_i = \max(\beta_i,\, \beta_\min)$ with $\beta_\min = 0.1$ throughout all phases. This guarantees a minimum gradient floor without fully decoupling regression from the condensation objective.

### 4.6 Optional Edge Classification Loss

$$\mathcal{L}_\text{edge} = - \frac{1}{|E|} \sum_{(i,j) \in E} \left[ y_{ij} \log \hat{p}_{ij} + (1 - y_{ij}) \log(1 - \hat{p}_{ij}) \right]$$

where $y_{ij} = 1$ if stubs $i$ and $j$ belong to the same track, 0 otherwise. This provides auxiliary supervision that improves message passing even if the edge scores are not used at inference.

### 4.7 Recommended Loss Weights

| Component | $\lambda$ | Rationale |
|-----------|-----------|-----------|
| Attraction | 1.0 | Primary clustering driver |
| Repulsion | 1.0 | Must balance attraction |
| Beta push | 0.5 | Seed-selection stabiliser; log-max term for true tracks + linear penalty for noise stubs |
| Noise | 0.5 | Less critical; noise stubs are minority |
| pT | 2.0 | Primary physics output |
| Charge | 1.0 | Important but simpler |
| d₀ | 1.5 | Critical for displaced physics program |
| φ₀ | 0.5 | Less critical than pT |
| Edge (aux) | 0.3 | Auxiliary signal only |

---

## 5. Handling Key Physics Requirements

### 5.1 Displaced Muon Robustness

The GNN handles displacement through three mechanisms:

**Mechanism 1: PhiB edge features are d₀-independent.**  
Intra-station edges carry $\hat{\kappa} = \phi_B / r$, which directly measures curvature regardless of production vertex. The network learns that these edges provide a "clean" curvature anchor.

**Mechanism 2: Position-shift pattern is a learned cluster feature.**  
For a displaced muon, the cross-layer Δφ values are systematically shifted by $d_0 \cdot \kappa / r$. The OC clustering must separate this coherent shift from random noise. With sufficient training on displaced samples, the condensation learns to assign displaced stubs correctly.

**Mechanism 3: Explicit d₀ regression head.**  
The per-node d₀ prediction allows the loss function to directly optimize for displacement reconstruction.

**Training strategy for displaced robustness:**
- 40% prompt muons (pT-flat 2–200 GeV)
- 40% displaced muons (Lxy-flat 0–50 cm, pT-flat 5–200 GeV)
- 20% noise/PU events (for noise rejection head)
- **Important:** Include the full correlation between pT and d₀ — do not generate them independently, since in physics they are related (long-lived particles have specific pT spectra).

### 5.2 Multi-Muon Disambiguation

This is the area where the GNN **most clearly outperforms** simpler approaches:

**Scenario:** Two 30 GeV muons separated by Δφ = 0.05 rad within one 120° processor. Both generate stubs in L0, L2, L6 (6 stubs total). A non-graph approach sees 6 stubs and must guess which belong to which muon.

**How the GNN handles it:**
1. **Graph construction:** All plausible cross-layer pairs within the precomputed φ-windows are connected (subject to the per-layer degree cap). The two L0 stubs connect to both L2 stubs, creating up to 4 cross-layer edges.
2. **Edge features:** The "correct" edges (L0-stub-A ↔ L2-stub-A, L0-stub-B ↔ L2-stub-B) will have Δφ consistent with ~30 GeV. The "cross" edges (L0-stub-A ↔ L2-stub-B) will have anomalous Δφ.
3. **Message passing:** After 1 round, the edge network scores correct edges high and cross edges low. After 2 rounds, the node embeddings diverge: stub-A's embedding approaches other stubs belonging to the same track.
4. **Object Condensation:** Two clusters form in the condensation space, each corresponding to one muon.

**This is the scenario where GNNs are categorically superior** to any single-track approach (Kalman, circle fit, phiB-dual-hypothesis). Those approaches process one muon hypothesis at a time and rely on ghost busting for deduplication, which fails for close muons with overlapping stubs.

### 5.3 Fake Rejection

The GNN provides multiple fake rejection mechanisms:

1. **Noise classification head** ($\hat{n}_i$): Tags pileup and noise stubs pre-clustering.
2. **Object Condensation β**: A cluster with low max-β contains only weakly associated stubs → fake.
3. **Cluster size**: Requiring ≥ 3 non-noise stubs per cluster rejects combinatorial fakes.
4. **Regression consistency**: If the pT prediction from different stubs in the same cluster varies wildly (high variance of $\hat{\kappa}_i$ within cluster), the quality is degraded.
5. **Edge classification scores** (auxiliary): The mean edge score within a cluster indicates track coherence.

### 5.4 Handling Variable Input

The GNN naturally handles:
- **Missing layers:** Fewer nodes, fewer edges. The message passing adapts.
- **Extra noise stubs:** More nodes, but the noise head classifies them.  No fixed assumption on occupancy.
- **CSC-only tracks** (no DT): No intra-station phiB edges, but cross-layer edges still provide curvature via Δφ between CSC stations.
- **RPC-only tracks:** Weakest scenario (coarse phi, no bending). The network still extracts what it can, but quality will be low. Same as current OMTF.

---

## 6. Comparison of All GNN Paradigms for OMTF

| Criterion | Edge Classification | Object Condensation | GravNet + OC | Direct Regression | Bipartite (Stub↔Hyp) |
|-----------|-------------------|--------------------|--------------|--------------------|----------------------|
| Multi-track capability | ✅ (post-process) | ✅ (native) | ✅ (native) | ❌ (fixed K) | ✅ (native) |
| End-to-end training | ❌ (separate fit) | ✅ | ✅ | ✅ | ✅ |
| Quality estimation | ⚠️ (from edge scores) | ✅ (β + cluster) | ✅ | ⚠️ (single output) | ⚠️ (assignment prob) |
| Physics interpretability | ✅ (edges are segments) | ⚠️ (latent space) | ⚠️ | ❌ | ⚠️ |
| Displaced robustness | ✅ (edge features) | ✅ (d₀ head + phiB) | ✅ | ✅ | ✅ |
| Training complexity | Low | Medium | High | Low | High |
| Inference simplicity | Medium (walk-through) | Medium (DBSCAN) | Medium | High (forward pass) | Low (bipartite matching) |
| **Suitability for OMTF** | **Good (as building block)** | **Excellent** | **Overkill** | **Inadequate (multi-track)** | **Good but complex** |

### Recommendation Ranking

1. **HECIN + Object Condensation** — Best balance of capability, elegance, and tractability
2. **Interaction Network + Edge Classification** → Track extraction → Regression — Proven; more engineering but less training complexity
3. **Bipartite GNN** — Good for multi-track but harder to train; assignment ambiguity
4. **GravNet + OC** — Would work but dynamic graph is unnecessary; adds compute without benefit

---

## 7. Network Sizing & Hyperparameters

### 7.1 Recommended Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Node embedding dim $d$ | 64 | Enough to encode 36 input features + learned representations |
| Edge embedding dim $d_e$ | 32 | Edges carry fewer features; half of node dim |
| Message-passing rounds $T$ | 3 (starting point) | Hypothesised worst-case diameter; **measure T=2 vs T=3** — prefer lower if equivalent (§3.5) |
| MLP hidden layers (all) | 1 hidden layer | Network is small; 2+ layers risk overfitting |
| MLP activation | ReLU | Standard; LeakyReLU for attention scores |
| Condensation dim | 2 | Sufficient for up to 3 tracks |
| Node aggregation | Attention (edge-type conditioned) | Best for heterogeneous features (§3.6) |
| Edge classification | Auxiliary (loss weight 0.3) | Extra supervision signal |
| Dropout | 0.1 | Regularization on MLP layers |
| Batch normalization | Yes (on node/edge features between MP rounds) | Stabilizes training |
| Learning rate | 3e-4 (AdamW) | Standard for GNNs |
| Weight decay | 1e-4 | Prevents overfitting on small graphs |
| Training batch size | 512 graphs per batch | Small graphs → large batch feasible |
| Number of epochs | 200 with early stopping (patience 20) | |

### 7.2 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Layer embedding table (18 layers × 6 dims) | 18×6 = **108** |
| Node encoder (23 → 64 → 64) | 23×64 + 64 + 64×64 + 64 = **5,696** |
| Edge encoder (8 → 32 → 32) | 8×32 + 32 + 32×32 + 32 = **1,344** |
| Edge MLP × 3 rounds (160 → 64 → 32) | 3 × (160×64 + 64 + 64×32 + 32) = 3 × 12,448 = **37,344** |
| Node MLP × 3 rounds (96 → 64 → 64) | 3 × (96×64 + 64 + 64×64 + 64) = 3 × 10,368 = **31,104** |
| Attention weights × 3 rounds | 3 × (97 → 1) = 3 × 97 = **291** |
| Condensation head (64 → 2) | 64×2 + 2 = **130** |
| Beta head (64 → 1) | 64 + 1 = **65** |
| κ head (64 → 1) | 64 + 1 = **65** |
| φ₀ head (64 → 1) | 65 |
| d₀ head (64 → 1) | 65 |
| Charge head (64 → 1) | 65 |
| Noise head (64 → 1) | 65 |
| Edge classifier (32 → 1) | 32 + 1 = **33** |
| **Total** | **~76,500 parameters** |

This is **~33× larger** than the Cross-Layer Attention network proposed earlier (~2,350 params), but still *tiny* by modern ML standards (for reference, a ResNet-18 has 11M parameters). The reduction from the prior ~77,200 estimate reflects dropping the redundant `is_bending` feature and replacing the 18-dim layer one-hot with a 6-dim learned embedding (+ 108-param table).

### 7.3 Computational Cost (Software)

| Operation | Count per event | FLOPs |
|-----------|----------------|-------|
| Node encoding | 20 nodes × 6.5K params | ~130K |
| Edge encoding | 40 edges × 1.3K params | ~52K |
| Edge MLP (3 rounds) | 3 × 40 edges × 12.4K | ~1.5M |
| Node MLP (3 rounds) | 3 × 20 nodes × 10.4K | ~624K |
| Attention (3 rounds) | 3 × 20 × 10 × 97 | ~58K |
| Output heads | 20 × 7 × 65 | ~9K |
| **Total** | | **~2.4M FLOPs per event** |

On a modern GPU (e.g., A100 with 19.5 TFLOPS FP32): **~0.1 μs per event** (throughput-bound: ~10M events/sec in large batches). On CPU (single core, ~50 GFLOPS): **~50 μs per event** — fast enough for emulation.

---

## 8. Training Strategy

### 8.1 Dataset Construction

**Truth matching protocol:**

Each gen-level muon that crosses the OMTF η region is matched to stubs by:
1. Run full GEANT4 simulation → SimTrack → DetId matching
2. For each stub, check if the SimTrack that produced it corresponds to a gen-level muon
3. Label stubs: track-ID $\in \{1, ..., K\}$ for real tracks, or `noise` for PU/fake stubs

> **Risk: stub truth-labeling ambiguity.** OC is robust to noisy labels only up to a point; systematically inconsistent training labels are a primary failure mode. In practice, stub truth-matching can be ambiguous in several ways: merged digis from two close SimTracks, stubs compatible with multiple gen-level muons in the same angular region, and partial-hit clusters that CMS reconstruction assigns to one SimTrack but GEANT4 shows secondary contributions. Mitigation: tag ambiguous stubs (those whose SimTrack association has $>1$ contributing SimTrack or whose hit-fraction for the leading SimTrack is $< 0.5$) and apply the following masking policy:
> - **Keep them as nodes** in the graph — they still participate in message passing and influence neighbor embeddings. Removing them would distort the graph topology.
> - **Mask from $\mathcal{L}_\text{attract}$** — do not pull them toward any seed's condensation point.
> - **Mask from regression losses** ($\mathcal{L}_\text{pt}$, $\mathcal{L}_\text{charge}$, $\mathcal{L}_\text{d0}$, $\mathcal{L}_\text{phi}$) — their track-parameter ground truth is unreliable.
> - **Mask from edge BCE labels** — any edge $(i, j)$ where $i$ or $j$ is ambiguous is excluded from $\mathcal{L}_\text{edge}$, otherwise the edge classifier receives contradictory supervision (the ambiguous stub's ground-truth assignment may disagree with the label inferred from its neighbors).
> - **Do not relabel as noise** — that adds a systematic bias by teaching the noise head to flag legitimate but ambiguous hits.

**Sample composition:**

| Sample | Count | pT range | Lxy range | PU |
|--------|-------|----------|-----------|-----|
| Single prompt muon | 1M | 2–200 GeV (flat in 1/pT) | 0 | 0 |
| Single displaced muon | 500K | 5–200 GeV (flat in 1/pT) | 0–50 cm | 0 |
| Two-muon events | 200K | 5–100 GeV each | 0 | 0 |
| Three-muon events | 50K | 5–100 GeV each | 0 | 0 |
| Single muon + PU200 | 500K | 2–200 GeV | 0 | 200 |
| Displaced muon + PU200 | 200K | 5–200 GeV | 0–50 cm | 200 |
| PU200 only (no muon) | 200K | — | — | 200 |
| **Total** | **2.65M** | | | |

### 8.2 Training Schedule

| Phase | Epochs | Focus | Learning rate |
|-------|--------|-------|---------------|
| Phase 1: Warm-up | 1–20 | OC clustering + noise rejection only; regression heads active but **unweighted** ($\beta$ replaced by 1 to prevent gradient starvation when $\beta$ has not yet converged) | 1e-3 → 3e-4 |
| Phase 2: Full loss | 21–100 | All losses activated; focus on pT regression | 3e-4 |
| Phase 3: Fine-tune displaced | 101–150 | Upweight d₀ loss by 2×; oversample displaced events | 1e-4 |
| Phase 4: Hardening | 151–200 | Add adversarial noise; PU200-heavy; test robustness | 3e-5 |

### 8.3 Data Augmentation

| Augmentation | How | Why |
|--------------|-----|-----|
| φ rotation | Shift all φ values by random offset | OMTF is φ-symmetric within one processor |
| Charge flip | Negate all κ, flip charge labels | C-symmetry of the detector |
| Stub dropout | Randomly mask 1–2 stubs with p=0.1 | Robustness to inefficiency |
| Quality smearing | Randomly degrade quality by 1 unit with p=0.05 | Robustness to data/MC differences |
| PU injection | Add random noise stubs from PU library | Fake rejection training |

---

## 9. Inference-Time Track Extraction Algorithm

### 9.1 Step-by-Step Post-Processing

```python
def extract_tracks(condensation_coords, betas, kappas, phis, d0s, charges, noise_scores,
                   beta_threshold=0.5, cluster_radius=0.3, min_stubs=3, max_tracks=3,
                   refine=True):
    """
    Extract muon tracks from GNN output using mutual nearest-seed assignment.

    Unlike greedy expansion, all seeds are identified first and each stub is
    assigned to its closest seed within cluster_radius. This prevents early
    seeds from "locking in" contested stubs before competing seeds can claim
    them — the dominant failure mode for close multi-muon events early in
    training when condensation seeds are not yet well separated.

    If refine=True, one centroid-recompute-and-reassign step is performed
    after the initial assignment (equivalent to 1 step of weighted k-means),
    which further stabilises multi-muon separation.

    Args:
        condensation_coords: [N, 2] condensation space coordinates
        betas: [N] seed scores
        kappas, phis, d0s, charges: [N] per-stub track parameter predictions
        noise_scores: [N] per-stub noise probability
        beta_threshold: minimum β to be a seed
        cluster_radius: max distance in condensation space for cluster assignment
        min_stubs: minimum stubs per cluster to form a valid track
        max_tracks: maximum output tracks (ghost buster limit)
        refine: if True, perform one weighted-centroid recompute and reassign

    Returns:
        List of TrackCandidate objects
    """
    tracks = []

    # Step 1: Filter noise
    signal_mask = noise_scores < 0.5
    signal_indices = np.where(signal_mask)[0]
    if len(signal_indices) == 0:
        return tracks

    # Step 2: Find all seeds in one pass (non-greedy); cap at max_tracks
    seed_candidates = signal_indices[betas[signal_indices] > beta_threshold]
    seed_candidates = seed_candidates[np.argsort(-betas[seed_candidates])]
    seed_indices = seed_candidates[:max_tracks]
    if len(seed_indices) == 0:
        return tracks

    seed_coords = condensation_coords[seed_indices]   # [K, 2]
    stub_coords = condensation_coords[signal_indices]  # [M, 2]

    # Step 3: Mutual nearest-seed assignment
    # Pairwise distances stub → seed: [M, K]
    dists_all = np.linalg.norm(
        stub_coords[:, None, :] - seed_coords[None, :, :], axis=2
    )
    nearest_seed = np.argmin(dists_all, axis=1)   # [M] index into seed_indices
    nearest_dist = dists_all[np.arange(len(signal_indices)), nearest_seed]
    in_radius    = nearest_dist < cluster_radius   # [M] bool

    # Optional 1-step centroid refinement (weighted k-means style)
    if refine and len(seed_indices) > 1:
        refined_centroids = np.zeros_like(seed_coords)
        for k in range(len(seed_indices)):
            members = signal_indices[(nearest_seed == k) & in_radius]
            if len(members) > 0:
                w = betas[members]
                refined_centroids[k] = np.average(
                    condensation_coords[members], axis=0, weights=w
                )
            else:
                refined_centroids[k] = seed_coords[k]
        # Reassign with refined centroids
        dists_all    = np.linalg.norm(
            stub_coords[:, None, :] - refined_centroids[None, :, :], axis=2
        )
        nearest_seed = np.argmin(dists_all, axis=1)
        nearest_dist = dists_all[np.arange(len(signal_indices)), nearest_seed]
        in_radius    = nearest_dist < cluster_radius

    # Step 4: Build clusters and compute weighted track parameters
    for k, seed_idx in enumerate(seed_indices):
        cluster_mask   = (nearest_seed == k) & in_radius
        cluster_indices = signal_indices[cluster_mask]

        if len(cluster_indices) < min_stubs:
            continue

        weights = betas[cluster_indices]
        w_sum   = weights.sum()

        track = TrackCandidate(
            kappa    = np.dot(weights, kappas[cluster_indices]) / w_sum,
            phi0     = np.dot(weights, phis[cluster_indices])   / w_sum,
            d0       = np.dot(weights, d0s[cluster_indices])    / w_sum,
            charge   = np.sign(np.dot(weights, 2*charges[cluster_indices]-1)),
            quality  = compute_quality(betas[cluster_indices], len(cluster_indices)),
            stubs    = cluster_indices,
            beta_max = betas[seed_idx],
        )
        tracks.append(track)

    # Step 5: Sort by quality and return top max_tracks
    tracks.sort(key=lambda t: -t.quality)
    return tracks[:max_tracks]


def compute_quality(betas, n_stubs):
    """Quality from cluster coherence and size."""
    return min(15, int(
        np.clip(np.mean(betas) * 8, 0, 8) +
        np.clip(n_stubs * 1.0, 0, 7)
    ))
```

### 9.2 Ghost Busting Integration

The GNN's clustering already handles most of the ghost buster's job (two close muons are assigned to separate clusters rather than being duplicated). However, to maintain compatibility with the GMT pipeline:

1. After track extraction, apply the same Δφ-based ghost busting as current OMTF (`|Δφ_GMT| < 8 bins`)
2. If two tracks are within the ghost radius, keep the one with higher quality (higher max-β)
3. Output at most 3 `RegionalMuonCand` objects per processor

---

## 10. Why This Architecture Is Theoretically Optimal for OMTF

Let us return to the 5 properties identified in §1.3 and verify:

| Property | Required | HECIN + OC Provides |
|----------|----------|---------------------|
| All-to-all angular comparison | ✅ | Cross-layer edges; attention aggregation ensures weighted access to all neighbors; $T=3$ gives full coverage |
| Distinguish φ vs φB measurements | ✅ | Heterogeneous edge types; edge-conditioned attention; separate feature channels for phiB |
| Permutation equivariance within layers | ✅ | GNN message passing is permutation equivariant by construction |
| Variable-size input | ✅ | Graph-based: N nodes is variable; no fixed tensor shape |
| Multiple tracks output | ✅ | Object Condensation: number of tracks = number of high-β seeds |

**No simpler architecture satisfies all 5** (a DeepSets model fails Property 5; a fixed MLP fails Property 4; a pure edge classifier fails Property 2 without additional engineering).

### 10.1 Comparison to the Cross-Layer Attention Network (Approach C)

The Attention Network from the prior document is a *special case* of this GNN where:
- The graph is fully connected $G_\text{full}$ (no physics-informed filtering)
- There are no explicit edge features
- There is only 1 attention round ($T=1$)
- The output is graph-level (pooling) rather than per-node (no OC)
- Multi-track is not handled (assumes single-muon)

The HECIN+OC is strictly more powerful:
- Physics-informed graph reduces attention noise
- Edge features inject geometric inductive bias
- 3 rounds give global coverage
- OC handles multi-track natively
- Per-node outputs give per-stub quality metrics

**Cost:** ~33× more parameters, ~10× more FLOPs. In software, this is negligible. In firmware, it would be significant but is explicitly *out of scope* for this document.

---

## 11. Extensions & Variants Worth Exploring

### 11.1 Temporal Extension: Multi-BX Graph

For out-of-time pileup rejection, extend the graph across BX:

$$V = \bigcup_{bx=-1}^{+1} V_\text{bx}, \quad E_\text{cross-BX} = \{(s_i, s_j) \mid |bx_i - bx_j| \leq 1\}$$

The network can learn that real muons have all stubs in BX=0, while OOT pileup produces stubs spread across multiple BX.

### 11.2 Equivariant Architecture (EGNN)

Since the OMTF has an approximate φ-rotation symmetry (within one processor), an equivariant GNN (EGNN, Satorras et al. 2021) could enforce this symmetry architecturally:

- Node features split into scalar invariants ($|φ|, \phi_B, \eta, q, r$) and a vector part ($\phi$ itself)
- Message passing updates both scalar and vector channels
- The vector channel transforms equivariantly under φ-rotation

**Benefit:** Improved sample efficiency (the network doesn't need to learn φ-rotation invariance from data). **Cost:** More complex implementation.

### 11.3 Pre-training with Physics Loss

Before OC training, pre-train the encoder with a self-supervised physics-motivated objective:

$$\mathcal{L}_\text{physics} = \sum_{\text{DT stations}} \left( \hat{\kappa}_\text{emb} - \frac{\phi_B}{r} \right)^2$$

This teaches the encoder that the embedding should correlate with curvature, providing a warm start for downstream clustering.

### 11.4 Uncertainty Quantification

Add a **variance head** alongside each regression head:

$$\hat{\sigma}^2_{\kappa,i} = \text{softplus}(\text{Linear}_\sigma(\mathbf{x}_i^{(T)}))$$

Train with negative log-likelihood loss:

$$\mathcal{L}_\text{NLL} = \frac{(\hat{\kappa_i} - \kappa_i^\text{true})^2}{2\hat{\sigma}^2_{\kappa,i}} + \frac{1}{2}\log\hat{\sigma}^2_{\kappa,i}$$

This gives per-muon pT uncertainty estimates, directly usable as quality scores. It teaches the network to be less confident when input stubs are sparse or low-quality.

---

## 12. Summary: The Optimal OMTF GNN

| Aspect | Recommendation |
|--------|---------------|
| **Graph structure** | Physics-informed cross-layer + intra-station PhiB ($G^*$) |
| **Node features** | 23D: phi_rel, phiB, eta, quality, r, type(11), layer\_emb(6), bx — `is_bending` dropped (implied by type); layer one-hot replaced by 6-dim learned embedding to prevent OC identity shortcutting |
| **Edge features** | 8 continuous + 6-dim pair-type embedding: Δφ, |Δr|, Δr², κ_estimate, PhiB_consistency (0 + flag for non-DT–DT), |Δη|, ΔBX, pair_type_emb(6) |
| **Backbone** | Heterogeneous Edge-Conditioned Interaction Network (HECIN) |
| **Message passing** | T=3 starting point (treat as hyperparameter; measure T=2 vs T=3), attention-weighted aggregation, residual connections |
| **Node dim / Edge dim** | 64 / 32 |
| **Readout** | Object Condensation (2D condensation space + β) + Per-node regression heads |
| **Track extraction** | Mutual nearest-seed assignment in condensation space (all seeds selected first, each stub assigned to closest seed within radius) + optional 1-step centroid refinement; replaces greedy expansion to prevent stub lock-in for close multi-muon events |
| **Multi-track** | Native (OC produces variable number of tracks) |
| **Displaced capability** | Inherent: d₀ regression head + PhiB-based curvature anchoring in edge features |
| **Parameter count** | ~76.5K (reduced from ~77K by dropping `is_bending` and compressing layer one-hot to 6-dim embedding) |
| **FLOPs per event** | ~2.4M (< 50 μs/event on CPU; < 1 μs on GPU) |
| **Training data** | ~2.65M events (prompt + displaced + multi-muon + PU200 + noise-only) |
| **Loss** | OC (attract + repulse + beta-push) + noise BCE + pT/charge/d₀/φ₀ regression + optional edge BCE; ambiguous stubs masked from attract/regression/edge-BCE |
| **Optimal for** | Software emulation; physics R&D studies; performance ceiling measurement |
| **Not designed for** | FPGA deployment (that requires the lighter Attention Network or non-ML methods) |

---

*This document is intended as the theoretical upper bound: the best possible GNN for the OMTF problem with no resource constraints. Hardware-constrained variants (reduced rounds, quantized, pruned) should be derived from this reference architecture.*


---

## 13. Existing Datasets in `emeleTrigger-L1Nano_151X_EdgeClassification_v1`: Compatibility Analysis

The `emeleTrigger-L1Nano_151X_EdgeClassification_v1` repository was developed in parallel by the L1T muon team and contains two independent dataset pipelines and several signal samples. This section audits both pipelines at code level and evaluates how much of their output can be directly reused for training the HECIN+OC network described above.

### 13.1 Two Dataset Pipelines in the Repository

| Pipeline | Class | Source tree | Graph data object |
|----------|-------|-------------|-------------------|
| **OMTF Dumper** | `OMTFDataset` | `simOmtfPhase2Digis/OMTFHitsTree` | Edge features: `[Δφ, Δη, Δr]`; node labels: `stubIsMatched` |
| **L1Nano** | `L1NanoDataset` | `Events` (NanoAOD) | Edge features: `[Δη, Δφ]`; node labels: ΔR matching to GenPart@MB2 |

They target the same physical problem from two different data tiers. The dumper pipeline is richer; the L1Nano pipeline is derived from standard CMS NanoAOD production and is therefore more widely available.

---

### 13.2 OMTF Dumper Pipeline (`OMTFDataset`)

#### Input ROOT files used in practice

```
/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/
  HTo2LongLivedTo2mu2jets/        <- displaced 2-muon exotic decay
  MuGun_Displaced/                <- single displaced muon gun
  MuGun_FullEta_OneOverPt_1to100/ <- prompt single muon, pT-flat [1,100] GeV
```

Graph dataset saved to: `/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250514_250617/`

#### Node features extracted

**Classification config** (`configs/dataset_classification.yml`):

| Variable | Source branch | Maps to GNN feature |
|----------|--------------|---------------------|
| `inputStubEtaG` | η in global units (×`HW_ETA_TO_ETA_FACTOR`) | `eta_i` ✅ |
| `inputStubCosPhi` | cos(φ_global) | supplementary φ encoding ✅ |
| `inputStubSinPhi` | sin(φ_global) | supplementary φ encoding ✅ |
| `inputStubR` | radius from `get_stub_r()` per detector type | `r_i` ✅ |
| `inputStubPhiG` | φ in global radians | `phi_rel` ✅ |
| `inputStubLayer` | Logic layer index 0–17 | `ℓ_i` ✅ |
| `inputStubType` | Detector type (DT=3, CSC=9, RPC=5) | `type_i` ✅ |
| `inputStubIsMatched` | Binary: stub ↔ gen-muon within OMTF window | node label `y` ✅ |

**Missing from current extraction** (would need to add to `add_extra_vars_to_tree`):

| Missing feature | Needed for GNN | Priority |
|----------------|---------------|----------|
| `inputStubPhiB` — bending angle | **Critical** — intra-station PhiB edges (§1.4) and `φB/r` edge feature (§3.4) | 🔴 High |
| `inputStubQuality` | Node feature `q_i` (§3.3) | 🟡 Medium |
| `inputStubBx` | Node feature `bx` (§3.3); OOT PU rejection | 🟡 Medium |
| Per-muon track ID (which-track index) | **Critical** for OC attractive loss — needs integer cluster label, not just binary `isMatched` | 🔴 High |

#### Edge construction

The pipeline calls `getEdgesFromLogicLayer(logicLayer, withRPC=True)` from `converter.py`, which implements the physics-informed connection map (`LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC`). This is **exactly** the rule used to define E_cross in G* (§1.4). The rectangular φ/η/r window provides edge features; it does not yet apply per-pair adaptive windows or a degree cap, but the underlying layer-adjacency table is identical to the GNN design.

**Edge features extracted:** `[Δφ, Δη, Δr]` — partial overlap with the 8-dimensional edge feature set in §3.4. Missing: `Δr²`, `κ̂_{ij} = 2Δφ/Δr²`, PhiB consistency, ΔBX, and the pair-type learned embedding.

**Edge labels:** `edge_label = 1` iff both endpoints have `inputStubIsMatched == True`. This is the binary same-track label from §4.6. Directly usable for the auxiliary edge BCE loss. ✅

#### Task configs and what they support

| Config | `task` | Target | GNN component |
|--------|--------|--------|---------------|
| `dataset_classification.yml` | `classification` | `inputStubIsMatched` (node) + edge label | Node classification head + auxiliary edge BCE ✅ |
| `dataset_regression.yml` | `regression` | `muonQOverPt` (graph-level) | κ regression as graph-level scalar ⚠️ |

For the HECIN+OC design the regression is per-node (`κ_i`). The existing `.pt` graphs can still be loaded; the graph-level `y` from the regression config is ignored and the per-node regression head is supervised by the per-stub truth-matched muon `q/pT`.

---

### 13.3 L1Nano Pipeline (`L1NanoDataset`)

#### Input ROOT files used in practice

```
HTo2LongLivedTo4mu_*.root    <- H->2LLP->4mu exotic signal (displaced, 4-muon event)
```

#### Node features

| Variable | Description | Maps to GNN |
|----------|-------------|-------------|
| `stub_tfLayer` | TF layer index | `ℓ_i` (coarser than logic-layer 0–17) ✅ |
| `stub_offeta1` | Offline-propagated η at MB2 | `eta_i` (propagated, not raw) ⚠️ |
| `stub_offphi1` | Offline-propagated φ at MB2 | `phi_rel` (propagated) ⚠️ |

Only 3 node features. No `phiB`, no `r`, no `stubType`, no quality. **Not sufficient for the HECIN+OC design** as-is. This pipeline was designed for a simpler proof-of-concept edge-classification task, not for full track reconstruction.

#### Truth matching

Uses propagated GenPart muon position at MB2 station (`GenPart_etaSt2`, `GenPart_phiSt2`) with ΔR < 0.15 (configurable). Filters: `|pdgId| == 13`, `statusFlags` bit 13 (`isLastCopy`), `pt > 1 GeV`, `etaSt2 > −999`.

This is a geometry-only matching, less precise than SimTrack-based matching. For the OC truth labeling in §8.1 (SimTrack → DetId), the L1Nano approach is a lower-quality approximation.

#### Edge construction

Connects stubs in consecutive `tfLayer` values with `|Δη| < 0.5`, `|Δφ| < 1.0`, with fallback to the next-next layer. Simpler than `getEdgesFromLogicLayer`; the `tfLayer` index is coarser than the 18-layer logic-layer system used in OMTF hardware.

---

### 13.4 Summary: What Is Directly Usable vs What Needs Extension

#### Directly usable for HECIN+OC training (from OMTF Dumper pipeline)

| Element | Status | Notes |
|---------|--------|-------|
| Graph topology (layer connection map) | ✅ Ready | `getEdgesFromLogicLayer` = E_cross in G* |
| Node features: η, φ, r, layer, type | ✅ Ready | All present in classification config |
| Node label: `isMatched` | ✅ Ready | Maps to noise-classification head |
| Edge label: same-track binary | ✅ Ready | `edge_label` → auxiliary edge BCE loss (§4.6) |
| Dataset: displaced single muon | ✅ Ready | `MuGun_Displaced` ↔ Phase 3 of training schedule (§8.2) |
| Dataset: displaced 2-muon exotic | ✅ Ready | `HTo2LongLivedTo2mu2jets` ↔ multi-muon disambiguation (§5.2) |
| Dataset: prompt muon gun | ✅ Ready | `MuGun_FullEta_OneOverPt_1to100` ↔ Phases 1/2 |
| H→4μ exotic (L1Nano) | ⚠️ Partial | Node features too sparse; useful for edge-clf pre-training only |

#### Needs extension / not yet present

| Element | Gap | Action required |
|---------|-----|----------------|
| `phiB` stub feature | Not extracted in current pipeline | Add `inputStubPhiB` to `stub_vars` and `add_extra_vars_to_tree` |
| Stub quality | Not extracted | Add `inputStubQuality` to `stub_vars` |
| BX | Not extracted | Add `inputStubBx` to `stub_vars` |
| Per-muon track ID for OC | Only binary `isMatched`, not which-track integer index | Parse SimTrack IDs from dumper tree |
| PU200 backgrounds | No sample in current set | Generate `SingleMuon + PU200` and noise-only samples |
| Three-muon events | No dedicated sample | Generate dedicated 3-muon gun or H→6μ |
| Per-pair adaptive φ-window | Fixed rectangular cuts | Replace with precomputed window table keyed by (det-type pair, layer pair) |
| Degree cap on edges | Not applied | Add top-k filtering by `|Δφ|` per (node, target-layer) after window cut |
| Intra-station PhiB edges (E_intra) | Not constructed anywhere | Add DT φ ↔ DT φB pairing to `create_edges` |
| Extended edge features | Only `[Δφ, Δη, Δr]` | Add `Δr²`, `κ̂_{ij}`, PhiB consistency (DT-DT), ΔBX, pair-type embedding |

---

### 13.5 Recommended Integration Path

Build on the **OMTF Dumper pipeline** (`OMTFDataset`) as the primary data source and extend it incrementally:

**Step 1 — Add missing node features (low effort)**

In `OMTFDataset.add_extra_vars_to_tree()`, expose already-available branches and update the classification config:

```yaml
stub_vars:
  - inputStubEtaG
  - inputStubCosPhi
  - inputStubSinPhi
  - inputStubR
  - inputStubPhiG
  - inputStubLayer
  - inputStubType
  - inputStubPhiB    # new
  - inputStubQuality # new
  - inputStubBx      # new
  - inputStubIsMatched
```

**Step 2 — Add intra-station PhiB edges (medium effort)**

In `OMTFDataset.create_edges()`, after the existing cross-layer loop, append DT φ ↔ DT φB pairs within each station (logic-layer pairs (0,1), (2,3), (4,5)).

**Step 3 — Add per-muon track ID for OC (high effort)**

Parse `muonSimTrackId` (or the equivalent per-stub SimTrack association) from the dumper tree to produce an integer *which-track* label per stub. This replaces the binary `isMatched` with a proper cluster label `T_k` needed by the Object Condensation attractive loss (§4.2).

**Step 4 — Generate missing physics samples**

Extend the batch submission script to include `SingleMuonPU200` and `MinBias_PU200` samples.

**Step 5 — Keep L1Nano pipeline as a lightweight cross-check**

The L1Nano pipeline (`L1NanoDataset`) is not the right primary training source for HECIN+OC, but it remains valuable as:
- A cross-check between offline-propagated and OMTF-internal stub coordinates
- A fast edge-classifier benchmark (the `HTo2LongLivedTo4mu` sample provides highly displaced, 4-muon stress-test events)
- A way to run on standard NanoAOD without needing dedicated dumper ntuples

---

### 13.6 Existing Models vs HECIN+OC

The models in `tools/training/models.py` and `Classification/models.py` are lightweight baselines:

| Model | Architecture | Task |
|-------|-------------|------|
| `GATRegressor` | 2-layer GAT + global pool + linear | Graph-level pT regression |
| `GCNRegressor` | 4-layer GCN + global pool + MLP | Graph-level pT regression |
| `SAGEClassifier` | GraphSAGE + edge MLP | Edge binary classification |

Key differences from HECIN+OC:

| Aspect | Existing models | HECIN+OC |
|--------|----------------|----------|
| Output granularity | Graph-level pool (single scalar) | Per-node (OC condensation + regression) |
| Multi-track capability | ❌ single-track only | ✅ native via OC |
| PhiB intra-station edges | ❌ | ✅ heterogeneous edge types |
| Track parameter regression | Graph-level only | Per-node, β-weighted cluster average |
| d₀ output | ❌ | ✅ |
| Quality metric | ❌ | ✅ via β + cluster coherence |

The `TrainEdgeClassificationFromGraph.py` pipeline (GraphSAGE + edge MLP, binary edge labels) corresponds precisely to the **auxiliary edge BCE loss** in HECIN+OC (§4.6). It can serve as a fast sanity check of graph construction quality — if the edge classifier cannot achieve high AUC on the existing `.pt` datasets, the graph topology needs revision before the full OC training loop is attempted.


---

## 14. Optimal Training Datasets for HECIN+OC: Full Specification

This section specifies the *ideal* training dataset independent of what currently exists in any repository. It is derived from first principles: what physical processes exercise every component of the HECIN+OC loss, what kinematic corners stress the graph topology, and what level of statistics is needed to cover the relevant phase space without overfitting.

---

### 14.1 Design Principles

The dataset must simultaneously satisfy seven constraints:

| Constraint | Implication |
|------------|-------------|
| **1. Full pT coverage** | 1/pT-flat sampling from 2 GeV to 200 GeV so that the log-κ regression loss (§4.5) receives uniform gradient across the spectrum; linear-pT sampling over-trains on high-pT muons which are kinematically easier |
| **2. Full η coverage of the OMTF acceptance** | 0.82 < \|η\| < 1.24 uniformly; boundary regions (η ≈ 0.82, η ≈ 1.24) are the hardest for track reconstruction and must not be under-sampled |
| **3. Multi-muon events** | The OC attractive/repulsive loss requires events with ≥ 2 true tracks in the same 120° processor window; single-muon training alone produces a model that never learns inter-cluster repulsion |
| **4. Displaced muons** | d₀ coverage from 0 to 50 cm; displacement is parameterized by sampling the production vertex position (x, y), from which d₀ is derived geometrically — sampling d₀ and Lxy as independent dimensions produces nonphysical (d₀, Lxy) corners; the d₀ head (§3.2) and the PhiB-based displaced-curvature anchoring (§5.1) must receive gradient from explicitly displaced events |
| **5. PU200 backgrounds** | The noise classification head and the β suppression on noise stubs require training on events where real muon stubs coexist with ~200 pileup interactions; noise-only (no muon) events are needed to calibrate the zero-track output |
| **6. Detector inefficiency** | Real OMTF data contains layer gaps, stuck channels, and RPC dead zones; the dataset must include events where specific layers are missing to prevent the GNN from relying on any one layer |
| **7. Charge symmetry** | Equal numbers of μ⁺ and μ⁻; the charge head learns a trivial solution if the training set is imbalanced |

---

### 14.2 Signal Samples (No PU)

These provide clean truth labels for all loss components. All samples are generated with the OMTF Phase-2 configuration and processed through the OMTF custom dumper (tree: `simOmtfPhase2Digis/OMTFHitsTree`) to obtain SimTrack-level stub associations.

#### Sample S1 — Single Prompt Muon, pT-Flat in 1/pT

| Parameter | Value |
|-----------|-------|
| Generator process | `MuGun` (`SingleMuPt` gun with flat-in-1/pT parameterization) |
| pT range | 2 – 200 GeV (flat in 1/pT) |
| η range | ±[0.82, 1.24] (OMTF acceptance only) |
| φ range | 0 – 2π (uniform) |
| d₀ | 0 (prompt) |
| Lxy | 0 (prompt) |
| PU | 0 |
| Charge | Equal μ⁺ / μ⁻ |
| Target event count | **1,000,000** |
| Purpose | Primary training signal for κ, φ₀, charge regression; OC single-cluster convergence; noise head on clean events |

**Why 1/pT-flat?** The curvature κ = q/pT is what the network regresses. 1/pT-flat = flat-in-κ, which gives uniform regression loss gradient across the full kinematic range. pT-flat would produce 100× more 100 GeV events than 2 GeV events, causing the network to optimize heavily for the easy high-pT regime.

#### Sample S2 — Single Displaced Muon, Flat in (1/pT, vertex position)

| Parameter | Value |
|-----------|-------|
| Generator process | `MuGun` with displaced vertex |
| pT range | 2 – 200 GeV (flat in 1/pT) |
| η range | ±[0.82, 1.24] |
| Vertex position | (x, y) sampled uniformly in a disk of radius 50 cm; Lxy = √(x²+y²) is a **derived** quantity |
| d₀ | Computed from gen-level helix given the sampled vertex and momentum; not sampled independently |
| PU | 0 |
| Target event count | **500,000** |
| Purpose | d₀ regression head; PhiB-curvature anchoring under displacement; displaced cluster morphology in OC space |

**Displacement parameterization:** The correct approach is to sample the production vertex (x, y) uniformly within a disk and let the CMSSW generator or a post-hoc vertex smearing tool compute both d₀ and Lxy consistently from the helix geometry. **Do not** sample d₀ and Lxy as two independent flat dimensions — the resulting (d₀, Lxy) pairs can be geometrically impossible (e.g. very large Lxy with near-zero d₀ for a high-pT muon) or so rare in nature that the dataset becomes unrepresentative. Do not use exotic signal samples (e.g. `H→2μ`) as the sole source of displaced muons: those have a strongly correlated (pT, Lxy) distribution set by the LLP lifetime and boost, which will bias the d₀ head toward a narrow region of phase space.

#### Sample S3 — Two Prompt Muons in the Same 120° Processor Window

| Parameter | Value |
|-----------|-------|
| Generator process | Two independent `MuGun` instances anchored to a shared processor center φₚ |
| pT (each muon) | 2 – 100 GeV (flat in 1/pT, independently drawn) |
| η (each muon) | ±[0.82, 1.24] |
| Processor center φₚ | Drawn uniformly from the 3 OMTF processor centers (0°, 120°, 240°) |
| Each muon's φ | Defined as φₚ + δφᵢ where δφᵢ ∈ (−60°, +60°) |
| |Δφ| between the two muons | 60% of events: 0.02 – 0.15 rad (close regime); 40%: 0.15 – 0.4 rad |
| Boundary events | 30% of events have at least one muon within ±5° (≈ 0.09 rad) of a processor boundary |
| |Δη| between the two muons | 0 – 0.4 rad |
| d₀ | 0 (prompt) |
| PU | 0 |
| Target event count | **250,000** |
| Purpose | OC repulsive loss (requires ≥ 2 seeds in condensation space); multi-muon disambiguation (§5.2); the repulsion term receives zero gradient on single-muon events |

**Why anchor to a processor center?** Defining each muon's φ relative to φₚ makes the boundary distance explicit and controllable. Without this, uniform φ sampling across 2π produces the right average boundary distance but cannot guarantee that the hard boundary-crossing topology (where stubs from one muon partially fall in the adjacent processor) is well-represented. The 30% boundary-event fraction is chosen to match the fraction of the processor area that lies within 5° of a boundary.

**Why oversample close Δφ?** The hardest multi-muon case is when two muons are close enough that their stubs can be mixed (Δφ < 0.1 rad ≈ the typical OMTF φ-resolution). Uniform φ sampling would produce too few such events; the 60/40 split ensures the hard cases are well-represented without eliminating the easier wide-separation cases from the loss.

#### Sample S4 — Three Prompt Muons in the Same Window

| Parameter | Value |
|-----------|-------|
| pT (each) | 5 – 80 GeV (flat in 1/pT) |
| η | ±[0.82, 1.24] |
| Processor center φₚ | Anchored as in S3 |
| Each muon's φ | φₚ + δφᵢ, δφᵢ ∈ (−60°, +60°) |
| Minimum |Δφ| between any pair | 0.03 rad |
| Boundary events | 30% have at least one muon within ±5° of a processor boundary |
| PU | 0 |
| Target event count | **100,000** |
| Purpose | Train the OC repulsion to separate three clusters; verify that the ghost buster (§9.2) correctly keeps 3 candidates; stress-test of the max_tracks=3 limit |

#### Sample S5 — Two Displaced Muons (Exotic Signal Template)

| Parameter | Value |
|-----------|-------|
| Generator process | Pythia8 `H→2LLP→2μ+X` or equivalent `2MuGun_Displaced` |
| pT (each) | 5 – 100 GeV (flat in 1/pT) |
| Vertex position (per muon) | (x, y) sampled uniformly in a disk of radius 30 cm; d₀ derived from helix |
| Lxy | Derived from the sampled vertex, not an independent sampling dimension |
| PU | 0 |
| Target event count | **150,000** |
| Purpose | OC must simultaneously handle two displaced tracks; the two tracks have correlated curvature (from the LLP boost) but different spatial origins — the hardest multi-track OC disambiguation scenario |

**Note on Lxy:** As with S2, Lxy is a derived label (√(x²+y²)) and not independently sampled. When using a Pythia8 `H→2LLP` process, the LLP lifetime sets Lxy implicitly; run several lifetime points (cτ = 1, 10, 100 mm) and merge to cover the d₀ distribution needed.

#### Sample S6 — Muon + Hard-Negative Fake Stub Injection (Noise Stress-Test)

| Parameter | Value |
|-----------|-------|
| Base | Single prompt muon (S1 kinematics, 100K events) |
| Fake stub multiplicity | Poisson(λ=3) per event |
| Fake stub placement | **Geometry-aware hard negatives** — see construction rule below |
| PU | 0 (stubs injected manually, not from full PU simulation) |
| Target event count | **100,000** |
| Purpose | Train the noise classification head on plausible-looking fakes that create real edges in the graph; simpler and faster than full PU200 for noise-head warm-up |

**Hard-negative construction rule:** Uniform-random fakes (φ flat across the full processor) are too easy — the noise head learns a trivial `|Δφ| too large` shortcut instead of using the full feature set. Instead, for each real muon stub $s$ in the event:
1. Draw a target layer $\ell'$ from the set of layers connected to $s$ by the `getEdgesFromLogicLayer` map (so the fake stub will create a real edge in $G^*$)
2. Sample the fake stub's φ inside the φ-window for the pair (type$_s$, type$_{\ell'}$, layer$_s$, $\ell'$) — the fake therefore passes the edge-construction filter
3. Deliberately make the stub **inconsistent** on secondary features: draw η outside the ±0.1 band around $s$'s η, or set phiB to a value inconsistent with the φ of the fake (phiB/r ≠ any plausible κ for that (φ, r) pair)

This forces the network to learn that fake stubs have inconsistent multi-feature patterns rather than simply large Δφ.

---

### 14.3 Background and Mixed Samples (PU200)

These samples require full CMS PU200 mixing. They are computationally expensive but irreplaceable: the noise head and β suppression on PU stubs receive no gradient from PU=0 events.

#### Sample B1 — Single Prompt Muon + PU200

| Parameter | Value |
|-----------|-------|
| Hard scatter | Single muon: **50% from pT ∈ [2, 10] GeV** (flat in 1/pT); 50% from pT ∈ [2, 200] GeV (flat in 1/pT) |
| Pileup | 200 minimum-bias interactions (flat PU200 scenario) |
| Target event count | **500,000** |
| Purpose | Noise head calibration in realistic occupancy; OC must correctly suppress ~5–10 PU stubs per event; quality score (§5.3) calibration |

**Why enrich the low-pT corner?** The fake rate is hardest to control for low-pT muons: their stubs are sparser, their curvature is large (easily mimicked by PU patterns), and β calibration is noisiest. Without explicit enrichment, the β threshold is tuned primarily by easy 50–200 GeV muons, and the operating point looks good on average but fails in rate at pT ≈ 2–10 GeV. The 50% low-pT sub-sample corrects this without discarding the high-pT supervision.

#### Sample B2 — Single Displaced Muon + PU200

| Parameter | Value |
|-----------|-------|
| Hard scatter | Displaced muon (S2 kinematics) |
| Pileup | PU200 |
| Target event count | **200,000** |
| Purpose | Most challenging scenario: displaced stub pattern + PU noise; the d₀ head must avoid fitting to PU stubs that accidentally align with the displaced track |

#### Sample B3 — Two Muons + PU200

| Parameter | Value |
|-----------|-------|
| Hard scatter | Two-muon event (S3 kinematics) |
| Pileup | PU200 |
| Target event count | **100,000** |
| Purpose | Multi-muon disambiguation under realistic background; OC repulsion must work when some inter-cluster region is occupied by PU stubs |

#### Sample B4 — Noise-Only (No Hard-Scatter Muon) + PU200

| Parameter | Value |
|-----------|-------|
| Hard scatter | No muon (minimum-bias only) |
| Pileup | PU200 |
| Target event count | **200,000** |
| Purpose | Train the zero-track case; β must be universally suppressed; the GNN must output no valid clusters; critical for fake-rate calibration |

**Important:** The `B4` sample sets the operating point for the β threshold at inference. Without it, the threshold is calibrated only on events where at least one muon exists, systematically biasing the fake rate.

---

### 14.4 Validation and Test Samples (Not Used in Training)

These samples must be **fully independent** of training (different random seeds, different generator runs, or real data).

#### Sample V1 — Physics-Process Validation

| Sample | Purpose |
|--------|---------|
| `DY→μμ` (pT > 20 GeV, full simulation) | Validates realistic pT spectrum, back-to-back topology, implicit 2-muon disambiguation |
| `W→μν` (pT > 20 GeV, full simulation) | Single-muon reference process for trigger efficiency measurement |
| `H→2LLP→4μ` (Lxy = 10, 50, 100 cm) | Displaced exotic benchmark; quantifies efficiency vs Lxy curve; direct comparison to current OMTF performance |
| `TTbar` + PU200 | Multijet + multiple muons; stress-test of noise rejection in complex events |
| `SingleMuon` data (Run-3 or Phase-2 pilot) | Data/MC comparison; checks that the GNN trained on simulation generalizes to data |

#### Sample V2 — Edge Cases

| Sample | Purpose |
|--------|---------|
| Single muon at η boundary (|η| ≈ 0.82 ± 0.02) | Verify performance at the DT/CSC transition; most likely to have missing layers |
| Single muon at η boundary (|η| ≈ 1.24 ± 0.02) | Upper OMTF boundary; CSC-heavy topology |
| Very low pT muon (2–4 GeV) | Tests whether the graph edge cuts are not too tight for high-curvature tracks |
| Very high pT muon (> 100 GeV) | Nearly straight track; tests whether OC can cluster a nearly-aligned stub set |
| Muon with 2 layers missing | Tests GNN robustness to detector gaps (§5.4) |

---

### 14.5 Truth Labeling Requirements

The quality of OC training is directly limited by the quality of the truth labels. The following conventions must be applied uniformly across all samples.

#### 14.5.1 Stub-to-Track Assignment (Per-Muon Track ID)

Each stub must carry an integer **track ID** $k \in \{1, \ldots, K, \text{noise}\}$ where $K$ is the number of gen-level muons that crossed the OMTF acceptance in this event:

```
For each valid stub s:
  1. Retrieve the list of SimTracks that deposited hits contributing to s
     (from SimHit → DetId matching in CMSSW)
  2. Find the dominant SimTrack (the one contributing the largest hit fraction)
  3. If the dominant SimTrack is associated to a gen-level muon (|pdgId| == 13,
     isLastCopy, pT > 1 GeV, |eta| in OMTF acceptance):
       → assign stub s to track k = (index of that gen-level muon in event)
  4. Otherwise (secondary particles, PU hits, noise):
       → assign track ID = 0 (noise label)
  5. Flag as "ambiguous" if:
       - Multiple SimTracks contribute to the stub with leading fraction < 0.5, OR
       - The dominant SimTrack belongs to a muon but its hit fraction < 0.5
     Ambiguous stubs are kept as nodes but masked from OC attract/regression/edge-BCE losses (§8.1)
```

**This per-muon integer track ID is the fundamental label** for:
- OC attractive loss: $\mathcal{L}_\text{attract}$ pulls stubs with the same non-zero track ID together
- OC repulsive loss: seeds from different non-zero track IDs must be separated by ≥ margin M
- Edge BCE: $y_{ij} = 1$ iff $\text{trackID}[i] == \text{trackID}[j] \neq 0$
- Regression: $\kappa^\text{true}$ for stub $i$ = curvature of its assigned gen-level muon

A binary `isMatched` flag (as in the current emeleTrigger repo) is **not sufficient** for OC training in multi-muon events: two stubs from different muons can both have `isMatched == True`, which makes their edge label `y_{ij} = 1` (same track), which is incorrect and will actively confuse the edge BCE loss.

#### 14.5.2 Gen-Level Muon Ground Truth Per Stub

After track-ID assignment, each signal stub (trackID > 0) must also carry the gen-level kinematic variables of its assigned muon:

| Variable | Branch name | Units | Used in |
|----------|-------------|-------|---------|
| Curvature | `muon_kappa` = charge/pT | 1/GeV | κ regression loss |
| φ at vertex | `muon_phi0` | rad | φ₀ regression loss |
| Transverse impact parameter | `muon_d0` | cm | d₀ regression loss |
| Charge | `muon_charge` | ±1 | Charge BCE loss |
| η at vertex | `muon_eta` | — | Quality checks; not regressed |
| pT | `muon_pt` | GeV | Cross-check; κ = charge/pT |

These are attached as per-stub tensor columns in the graph `Data` object, not as graph-level attributes, so that the per-node regression loss can use them directly without indexing back through a separate list.

#### 14.5.3 Handling the PU Truth

For PU200 samples, PU stubs are assigned trackID = 0 (noise). Their `muon_*` columns are filled with sentinel values (−999) and masked in all regression losses. The noise classification head receives supervision from the binary signal/noise distinction implied by trackID.

---

### 14.6 Dataset Statistics Summary

| Sample | Events | Stubs/event (avg) | Tracks/event | PU |
|--------|--------|-------------------|--------------|-----|
| S1 — Single prompt muon | 1,000,000 | 7 | 1 | 0 |
| S2 — Single displaced muon | 500,000 | 6 | 1 | 0 |
| S3 — Two prompt muons | 250,000 | 12 | 2 | 0 |
| S4 — Three prompt muons | 100,000 | 17 | 3 | 0 |
| S5 — Two displaced muons | 150,000 | 11 | 2 | 0 |
| S6 — Muon + manual noise | 100,000 | 10 | 1 | 0 |
| B1 — Single prompt + PU200 | 500,000 | 20 | 1 | 200 |
| B2 — Displaced + PU200 | 200,000 | 19 | 1 | 200 |
| B3 — Two muons + PU200 | 100,000 | 25 | 2 | 200 |
| B4 — Noise-only PU200 | 200,000 | 13 | 0 | 200 |
| **Total (training + val + test)** | **3,100,000** | — | — | — |

Recommended split: 70% training / 15% validation / 15% test, applied **per-file** (not per-event) to prevent same-event leakage through detector-level correlations. Additionally, split by production campaign or by (dataset, run-seed block): CMSSW-style production batches sharing the same random seed or the same pileup library chunk can produce correlated occupancy patterns that inflate apparent validation performance. Assign all files from one seed block exclusively to train, validation, or test — never mix seed blocks across the split boundary.

---

### 14.7 Graph-Level Data Object Specification

Each event must be converted into a `torch_geometric.data.Data` object containing the following tensors. This is the **target schema** that the dataset creation pipeline should produce.

```
Data(
  # --- Node features (split into continuous + categorical) ---
  #
  # Storing type and layer as one-hot or mixed inside a FloatTensor is
  # error-prone and forces the dataset to bake in what should be model
  # parameters (embedding tables).  Keep them separate:
  #
  x_cont         : FloatTensor [N, 6]   # continuous node features only
                   # [phi_rel, phiB, eta, quality, r, bx]
  node_type      : LongTensor  [N]      # stub type index 0..10
                                        # (DT_PHI=0, DT_PHIB=1, RPC=2, CSC=3, ...)
                                        # type_emb(node_type) applied inside model
  layer_id       : LongTensor  [N]      # logic layer index 0..17
                                        # layer_emb(layer_id) applied inside model
  # Model concatenates: x = cat([x_cont, type_emb(node_type), layer_emb(layer_id)])
  # giving [N, 6+type_emb_dim+layer_emb_dim] = [N, 6+11+6] = [N, 23] at forward time

  # --- Full edge connectivity ---
  edge_index     : LongTensor  [2, E]   # COO format (source, target)
  edge_attr_cont : FloatTensor [E, 7]   # continuous edge features only
                                        # [delta_phi, abs_delta_r, delta_r_sq,
                                        #  kappa_hat, phiB_consistency,
                                        #  abs_delta_eta, delta_bx]
  pair_type      : LongTensor  [E]      # layer-pair category index 0..14
                                        # pair_type_emb(pair_type) applied inside model
  # Model concatenates: e = cat([edge_attr_cont, pair_type_emb(pair_type)])

  # --- Edge type mask (to distinguish E_cross from E_intra) ---
  edge_type      : LongTensor  [E]      # 0 = cross-layer, 1 = intra-station phiB

  # --- Truth labels: nodes ---
  track_id       : LongTensor  [N]      # 0=noise, 1..K=track index
  is_ambiguous   : BoolTensor  [N]      # True if stub truth is unreliable (§14.5.1)

  # --- Truth labels: edges ---
  edge_y         : FloatTensor [E]      # 1.0 if same non-noise track, else 0.0
                                        # -1.0 for edges involving ambiguous stubs
                                        # (masked in edge BCE loss)

  # --- Per-stub regression targets (for signal stubs only) ---
  muon_kappa     : FloatTensor [N]      # charge/pT of assigned muon; 0.0 if noise
  muon_phi0      : FloatTensor [N]      # φ at vertex; 0.0 if noise
  muon_d0        : FloatTensor [N]      # d₀ in cm; 0.0 if noise
  muon_charge    : FloatTensor [N]      # ±1.0; 0.0 if noise

  # --- Gen-level muon truth (for cross-checks and β-push loss) ---
  gen_muon_pt    : FloatTensor [K]      # pT of the K gen-level muons in this event
  gen_muon_eta   : FloatTensor [K]      # η
  gen_muon_phi   : FloatTensor [K]      # φ
  gen_muon_d0    : FloatTensor [K]      # d₀
  gen_muon_charge: FloatTensor [K]      # ±1

  # --- Event-level metadata (not used in training, stored for analysis) ---
  num_nodes      : int                  # N
  n_true_tracks  : int                  # K (number of gen-level muons in event)
  has_pu         : bool                 # True if PU200 sample
  sample_id      : int                  # enum: S1=1, S2=2, ..., B4=10
)
```

The split between continuous tensors (`x_cont`, `edge_attr_cont`) and categorical index tensors (`node_type`, `layer_id`, `pair_type`) is deliberate. Embedding tables are trainable model parameters — baking 6-dim or 11-dim embeddings into the `.pt` files would make the dataset depend on a specific embedding dimensionality chosen at generation time and prevent experimenting with different embedding sizes without regenerating data. Storing raw integer indices in `LongTensor` fields is both compact and embedding-dimension-agnostic.

---

### 14.8 Kinematic Coverage Verification

Before training, verify the following distributions are well-covered in the combined training dataset. If any bin is under-populated by more than 5× relative to the mean bin count, the corresponding sample must be augmented or reweighted:

| Variable | Range | Binning | Check |
|----------|-------|---------|-------|
| 1/pT of the harder muon | [0.005, 0.5] 1/GeV | 20 uniform bins | Flat after 1/pT sampling |
| η of each muon | [0.82, 1.24] | 10 bins | Flat |
| φ of each muon | [0, 2π] | 12 bins | Flat |
| d₀ | [0, 50] cm | 10 bins | Flat for S2/S5/B2 |
| Number of true tracks per event | 0, 1, 2, 3 | — | S/B composition as above |
| Number of valid stubs per event (N) | [3, 30] | — | Should not have hard cutoff at any value |
| Stub type composition | DT / CSC / RPC fraction | — | Matches expected detector occupancy per η region |
| Logic layer pattern | Which layers fire | 18 layers | No layer systematically missing from training |

The layer pattern check is critical: if the training dataset is dominated by events where layer L is always present, the network learns to depend on L and fails on real data where L is occasionally inefficient.

---

### 14.9 Data Augmentation Strategy (Applied Online During Training)

The following augmentations are applied **per-batch at training time**, not pre-computed in the `.pt` files, to maximize the effective diversity of each epoch:

| Augmentation | Implementation | Frequency | Physics motivation |
|--------------|---------------|-----------|-------------------|
| **φ-rotation** | Shift all absolute-φ node features (`phi_rel`, `phiSt2` etc.) by a random `Δφ ~ Uniform[-π/3, +π/3]` (±60°, matching the processor half-width); edge features that are already φ-differences (`delta_phi`, `kappa_hat`) are rotation-invariant and need no update | 100% of events | The OMTF processor covers 120° (±60° from its center); φ-translation is an exact symmetry within the window. Rotating beyond ±60° would push stubs outside the window boundary without a matching re-centering of the processor definition |
| **Charge flip** | Implement as a **φ-reflection**: negate all absolute-φ coordinates (`phi_rel → -phi_rel`, `phiSt2 → -phiSt2`, `gen_muon_phi → -gen_muon_phi`, `muon_phi0 → -muon_phi0`) and flip all signed-curvature quantities (`phiB → -phiB`, `delta_phi → -delta_phi`, `kappa_hat → -kappa_hat`, `muon_kappa → -muon_kappa`, `muon_charge → -muon_charge`). `abs_delta_r`, `abs_delta_eta`, `r`, `eta`, `quality`, `bx` are φ-reflection-invariant and unchanged | 50% of events | The φ-reflection φ → −φ is an exact discrete symmetry of the OMTF barrel geometry in a uniform magnetic field. It maps μ⁺ trajectories to μ⁻ trajectories and vice versa, doubling effective statistics at zero computational cost. **Implementation note:** negate φ-coordinates and signed-curvature quantities in a single coherent transform — do not mix C-conjugation (only flip charge/κ) with φ-reflection (also flip φ), as those are different symmetries and mixing them produces an inconsistent data object |
| **Stub dropout** | Randomly remove one stub (node + all its edges) with probability p=0.15 per stub | Applied per stub | Simulates single-layer inefficiency; trains robustness to missing layers (§5.4) |
| **Quality degradation** | Degrade stub quality by 1 step with probability 0.08 | Per stub | Data/MC difference in stub quality is a known systematic; network should not overfit on quality values |
| **BX smearing** | Flip stub BX from 0 to ±1 with probability 0.05 | Per stub | Simulates marginal out-of-time hits (common at the DT/OMTF boundary) |
| **Hard-negative PU injection** | For each real muon stub $s$, inject 1–2 fake stubs in layers connected to $s$ by `getEdgesFromLogicLayer`, with φ inside the corresponding φ-window (so they form real graph edges), but with inconsistent secondary features: η drawn outside ±0.1 of $s$'s η, or phiB inconsistent with the fake's (φ, r) under any plausible κ | 30% of PU=0 events | Forces the noise head to learn full multi-feature discrimination rather than a trivial `|Δφ| too large` shortcut; complementary to B1–B3 which use realistic full-PU mixing |

The **charge flip** augmentation is the most important and the most commonly forgotten. It is implemented as a **φ-reflection** (φ → −φ), which simultaneously reverses the sign of all absolute-φ coordinates and all signed-curvature quantities. This is an exact discrete symmetry of the OMTF barrel detector geometry and strictly doubles training statistics. One implementation pitfall: do not apply only the curvature sign flip (C-conjugation) without also mirroring the φ coordinates — that combination is not a symmetry and produces inconsistent graph states.

The **φ-rotation** augmentation only needs to update absolute-φ node features (`phi_rel`, `phiSt2`, etc.). All edge features that are already φ-differences (`delta_phi`, `kappa_hat`, `abs_delta_eta`) are rotation-invariant by construction and require no update. The rotation is capped at ±π/3 (±60°) to stay within the processor window; rotating further would place stubs outside the acceptance boundary without a matching re-centering of the processor frame.

---

### 14.10 Generation in CMSSW: Recommended Configuration

The following CMSSW configurations are recommended for sample generation:

```
Release        : CMSSW_15_1_X (Phase-2 L1T simulation)
Geometry       : D95 or later (Phase-2 tracker/muon geometry)
Conditions     : 140X_mcRun4_realistic_v*  (Phase-2 scenario)
L1T emulator   : Phase-2 OMTF (simOmtfPhase2Digis)
Custom dumper  : L1Trigger/L1TMuonOverlapPhase2/plugins/OmtfHitsToNtuple
Dumper tree    : simOmtfPhase2Digis/OMTFHitsTree
```

For each signal sample type, the generator fragment controls the kinematic parameterization:

**S1 (single prompt, 1/pT flat):**
```python
generator = cms.EDFilter("MuonGun",
    src = cms.InputTag("generator"),
    particleID = cms.int32(13),          # μ−; separate run for 13+ (μ+)
    pTMin = cms.double(2.0),
    pTMax = cms.double(200.0),
    flat1overPt = cms.bool(True),        # flat in 1/pT
    etaMin = cms.double(0.82),
    etaMax = cms.double(1.24),
    phiMin = cms.double(-3.14159),
    phiMax = cms.double(3.14159),
)
```

**S2 (displaced, flat in 1/pT × d₀):**
Use `Pythia8` with a `MuGun` + vertex smearing:
```python
process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    ...  # standard Pythia8 setup
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring(
            'ParticleDecays:tau0Max = 0',
            'LesHouches:setLifetime = 2',
        )
    )
)
process.VtxSmeared = cms.EDProducer("GaussianVtxSmearingProducer",
    MeanX = cms.double(0.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaX = cms.double(20.0),   # cm; covers d0 ~ 0-50 cm range
    SigmaY = cms.double(20.0),
    SigmaZ = cms.double(0.0),
)
```

For multi-muon samples (S3, S4), generate two/three independent `MuGun` instances in the same event, constraining the φ difference as described in §14.2.

---

### 14.11 Storage and Format Estimate

| Sample group | Events | Avg graph size | Storage (uncompressed `.pt`) |
|-------------|--------|---------------|------------------------------|
| Signal (S1–S6) | 2,100,000 | ~4 KB/graph | ~8.4 GB |
| Background (B1–B4) | 1,000,000 | ~7 KB/graph | ~7.0 GB |
| Validation (V1–V2) | ~200,000 | ~5 KB/graph | ~1.0 GB |
| **Total** | **3,300,000** | — | **~16.4 GB** |

Stored as one `.pt` file per original ROOT file (matching the existing batch submission pattern in `submitJobs_DatasetCreation.py`), with ~2,000 events per file. This allows partial loading during training without reading the entire dataset into memory and simplifies distributed training across multiple GPUs.

Recommended compression: `torch.save` with `_use_new_zipfile_serialization=True` (default since PyTorch 1.10) achieves ~40% size reduction, bringing total to **~10 GB** — easily stored on EOS or any NFS-mounted volume.


---

## 15. Primer: What "Signal" and "Background" Mean for GNN Training

This section clarifies terminology that is often a source of confusion when moving from classical HEP analysis (where "signal" = a physics process) to GNN training (where "signal" = a *label on individual detector hits*). These are fundamentally different uses of the same words.

---

### 15.1 The Classical HEP Meaning (Physics Processes)

In a traditional trigger or analysis:

- **Signal sample** = a Monte Carlo dataset produced from a *specific physics process* you want to select. Example: `MuGun_Displaced` — events where the generator produced a displaced muon.
- **Background sample** = a dataset from processes you want to reject. Example: `MinBias_PU200` — events dominated by soft QCD without a hard muon.

The sample classification (`MuGun`, `TTbar`, `MinBias`) is a property of the **whole event**. Every event in `MuGun` contains a real muon; every event in `MinBias` does not.

**This is NOT how signal/background is used in the GNN training context.**

---

### 15.2 The GNN Training Meaning (Labels on Individual Stubs)

For GNN training, "signal" and "background/noise" are **per-stub labels within a single event**, not per-event classifications. A single event from a `MuGun_Displaced` sample contains *both* signal stubs and noise stubs simultaneously:

```
Event from MuGun_Displaced + PU200:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stub 0  (MB1, η=0.9, φ=0.12)   → from displaced muon → label: signal (trackID=1)  │
  │  Stub 1  (MB2, η=0.9, φ=0.14)   → from displaced muon → label: signal (trackID=1)  │
  │  Stub 2  (ME1, η=0.9, φ=0.11)   → from displaced muon → label: signal (trackID=1)  │
  │  Stub 3  (MB1, η=1.1, φ=1.80)   → from PU interaction → label: noise   (trackID=0) │
  │  Stub 4  (RPC, η=0.8, φ=3.50)   → from PU interaction → label: noise   (trackID=0) │
  │  Stub 5  (MB3, η=0.9, φ=2.10)   → from PU interaction → label: noise   (trackID=0) │
  └─────────────────────────────────────────────────────────────────┘
```

The GNN sees all 6 stubs as input nodes. It must learn to:
1. **Identify** which stubs belong to the muon (stubs 0, 1, 2) versus noise (stubs 3, 4, 5)
2. **Group** the muon stubs into one cluster (Object Condensation)
3. **Regress** the muon parameters (pT, charge, d₀) from those stubs

**Conclusion: Even in a "signal sample", most stubs in most PU200 events can be noise-labeled. A "signal sample" simply guarantees that at least one gen-level muon exists in the event — it does not mean every stub in the event is labeled 1.**

---

### 15.3 The Three Label Types in the GNN Schema

There are three distinct labeling concepts used simultaneously in the training schema, each serving a different loss component:

#### Label Type 1: Node binary label — "Is this stub from any real muon?"

```
y_node[i] = 1   if stub i was produced by a gen-level muon in OMTF acceptance
y_node[i] = 0   if stub i is from pileup, secondary particles, or noise
```

- Used by: **noise classification head** (binary cross-entropy, §4.4)
- Used by: **β push loss** (suppress β on stubs where y_node=0, §4.3.1)
- This is what the current emeleTrigger repo stores as `inputStubIsMatched` or `edge_label`

**This is a binary 0/1 label per stub, not per event.**

#### Label Type 2: Node integer label — "Which specific muon does this stub belong to?"

```
track_id[i] = 0      if stub i is noise
track_id[i] = 1      if stub i belongs to gen-level muon #1 in this event
track_id[i] = 2      if stub i belongs to gen-level muon #2 in this event
track_id[i] = 3      ...
```

- Used by: **OC attractive loss** (pull stubs with the same non-zero track_id toward the same condensation point, §4.2)
- Used by: **OC repulsive loss** (push seeds from different non-zero track_ids apart, §4.3)
- Used by: **per-stub regression targets** (the κ, φ₀, d₀ ground truth for stub i = the kinematics of gen-muon track_id[i])

**Why is this needed when Label Type 1 exists?**

Because in multi-muon events, two stubs can both have `y_node=1` (both belong to real muons) but belong to *different* muons. The OC loss needs to know which stubs to *attract together* and which to *keep apart*. Binary `isMatched=True` on both stubs of an edge would set `edge_y=1` (same track), which is wrong if the stubs come from different muons.

Example:
```
Event with 2 muons (μ₁ at φ=0.1, μ₂ at φ=0.3):
  Stub A: from μ₁  →  track_id=1,  y_node=1
  Stub B: from μ₂  →  track_id=2,  y_node=1
  Stub C: noise    →  track_id=0,  y_node=0

  Edge (A,B): y_node[A]=1 AND y_node[B]=1
              ┗━ binary isMatched would give edge_y=1  ← WRONG
              track_id[A]=1 ≠ track_id[B]=2
              ┗━ integer track_id gives edge_y=0  ← CORRECT

  Edge (A,A'): both from μ₁, track_id=1=1
              ┗━ edge_y=1 (both labels agree this time)
```

#### Label Type 3: Edge binary label — "Are these two stubs from the same muon?"

```
edge_y[i,j] = 1   if track_id[i] == track_id[j]  AND  track_id[i] != 0
edge_y[i,j] = 0   otherwise (different muons, or one/both are noise)
```

- Used by: **auxiliary edge BCE loss** (§4.6) — optional but improves message passing
- Derived entirely from Label Type 2; not an independent label

---

### 15.4 Mapping Physics Samples to Label Populations

The table below shows what stub labels an event from each physics sample will contain. This answers the question: *"do all stubs in a signal sample get label=1?"*

| Sample | y_node=1 stubs (signal) | y_node=0 stubs (noise) | max track_id |
|--------|------------------------|------------------------|--------------|
| **S1** Single prompt muon, PU=0 | All stubs from the 1 muon (~5–9 stubs) | 0 (no PU; occasional secondaries only) | 1 |
| **S2** Single displaced muon, PU=0 | All stubs from the displaced muon (~4–7 stubs) | 0 (rarely a secondary from the displaced vertex) | 1 |
| **S3** Two prompt muons, PU=0 | All stubs from both muons (~8–16 stubs) | 0 | 2 |
| **S4** Three prompt muons, PU=0 | All stubs from all 3 muons | 0 | 3 |
| **S5** Two displaced muons, PU=0 | All stubs from both displaced muons | 0 (possibly secondaries from the displaced decay) | 2 |
| **S6** Muon + injected fake stubs | Stubs from the muon | The injected fake stubs | 1 |
| **B1** Single muon + PU200 | Stubs from the 1 real muon (~5–9 out of ~20 total) | ~10–15 PU stubs per event | 1 |
| **B2** Displaced + PU200 | Stubs from the displaced muon (~4–7) | ~10–15 PU stubs | 1 |
| **B3** Two muons + PU200 | Stubs from both muons (~10–16) | ~10–15 PU stubs | 2 |
| **B4** Noise-only PU200 | **0** (no gen-level muon in OMTF acceptance) | All stubs (~10–15) | 0 |

**Key observations:**

1. **For PU=0 signal samples (S1–S5), nearly all stubs *are* signal-labeled** (y_node=1). This is why they are called "signal samples" — the event was generated from a muon gun, so almost every stub in the OMTF window comes from the real muon. However, this is specific to PU=0 generation, not a general property of signal samples.

2. **For PU200 samples (B1–B3), only a minority of stubs per event are signal-labeled.** In B1, the real muon produces ~7 stubs in a typical event, but there are ~20 stubs total. The GNN must discriminate the 7 real stubs from ~13 PU stubs — a ratio of about 1:2.

3. **Sample B4 (noise-only) has zero signal stubs,** zero true tracks, and the GNN must output no valid clusters. This is needed to set the operating point for the β threshold.

4. **The word "signal sample" refers to the physics process used to generate the event, not to the labels on the stubs inside it.** A stub is "signal" if it was produced by a gen-level muon; it is "background/noise" if it came from pileup or secondaries. These two usages must not be conflated.

---

### 15.5 What the GNN Is Actually Learning

To make the training objective concrete, here is what each output head is learning, expressed in terms of what input the head sees and what output it must produce:

```
For each event with K gen-level muons and N total stubs (signal + noise):

  INPUT: Graph with N nodes, each node = one stub with 23 features
         Edges connecting stubs in adjacent/nearby layers

  PER-NODE OUTPUTS the GNN must produce:

    noise_score[i]  ∈ [0,1]   → "How likely is stub i to be noise?"
                                  supervised by y_node[i] ∈ {0,1}  (Label Type 1)

    beta[i]         ∈ [0,1]   → "Is stub i the best representative of its track?"
                                  supervised by beta-push loss: β→1 for the
                                  highest-beta stub in each true cluster T_k,
                                  β→0 for noise stubs (Label Type 2 + Type 1)

    c[i]            ∈ ℝ²      → "Where in condensation space does this stub land?"
                                  supervised by OC loss: stubs with same track_id
                                  must cluster together; seeds from different
                                  track_ids must be far apart (Label Type 2)

    kappa[i]        ∈ ℝ       → "What curvature is consistent with this stub?"
                                  supervised by muon_kappa[i] (true κ of the muon
                                  this stub belongs to); only for signal stubs (Label Type 2)

    phi0[i], d0[i], charge[i] → similarly supervised by muon-level true values

  PER-EDGE OUTPUTS (auxiliary):

    edge_score[i,j] ∈ [0,1]  → "Do stubs i and j belong to the same muon?"
                                  supervised by edge_y[i,j]  (Label Type 3)

  POST-PROCESSING (not learned, rule-based):
    1. Filter stubs where noise_score > 0.5
    2. Find seeds: stubs where beta > beta_threshold
    3. Cluster remaining stubs around seeds in condensation space (c[i])
    4. For each cluster: average kappa[i], phi0[i], d0[i], charge[i] weighted by beta[i]
    5. Output up to 3 TrackCandidate objects
```

The GNN does **not** output a single "signal/background" score for the whole event. It operates entirely at the stub level and the track level, and produces a variable number of reconstructed tracks (0, 1, 2, or 3) from each event.

---

### 15.6 Why Clean Signal Samples (PU=0) Are Still Needed

Given that PU200 samples are more realistic, one might ask: why include PU=0 samples at all? The answer is that PU=0 events provide **clean supervision** for the regression heads that would be diluted in PU200:

| Training component | PU=0 samples | PU200 samples |
|-------------------|-------------|---------------|
| OC clustering (attract/repulse) | ✅ All stubs are signal → clean attractive gradients | ✅ Also works but fewer signal stubs per event |
| Noise classification head | ⚠️ Almost no noise stubs → head receives no gradient | ✅ Essential |
| β push (seed identification) | ✅ Clean; every non-noise stub should have positive β | ✅ Also good |
| κ regression | ✅ Best-quality gradient: every stub contributes directly | ⚠️ Diluted by the need to first classify signal stubs |
| d₀ regression | ✅ Especially for S2: clean displaced stub pattern | ⚠️ PU stubs can mimic displaced patterns |
| Multi-track OC repulsion | ✅ Essential for S3, S4 (clear separation between two clusters) | ⚠️ Harder to attribute: PU stubs could be mistaken for a third cluster |

In short: **PU=0 samples train the regression heads and OC clustering cleanly; PU200 samples train the noise head and verify that clustering survives realistic occupancy.** Both are required. Neither alone is sufficient.

---

## 16. CMSSW Data Tiers, Inputs, and the Full Data Flow for GNN Training

This section answers the question: **starting from a physics process, what CMSSW data tier do I need, what does it contain, and how does it become a graph `.pt` file for the GNN?**

---

### 16.1 The CMSSW Processing Chain

CMS simulation and reconstruction runs in a fixed sequence of steps. Each step writes a CMSSW data tier — a ROOT file with a specific set of collections. The chain from generator to final training graph is:

```
Generator (Pythia8 / Particle Gun)
    │
    ▼  GEN tier
    │  Contains: GenParticle, HepMCProduct
    │  File size: ~1–5 kB/event
    │
    ▼  SIM step (Geant4)
    │  GEN-SIM tier
    │  Contains: GenParticle, SimTrack, SimVertex, SimHit (all subdetectors)
    │  File size: ~50–200 kB/event
    │
    ▼  DIGI step (detector electronics simulation)
    │  GEN-SIM-DIGI (also called RAW-SIM)
    │  Contains: all SIM collections + digitized hits
    │  For muons: CSCDigi, DTDigi, RPCDigi, GEMDigi
    │  File size: ~100–500 kB/event
    │
    ▼  L1T emulation (run inside DIGI or separately)
    │  Adds: L1TMuon stubs, OMTF candidates, GMT output
    │  Key: simOmtfPhase2Digis — the OMTF emulator output
    │        InputMakerPhase2 builds MuonStub objects here
    │  File size: adds ~5–20 kB/event
    │
    ▼  Custom OMTF Dumper (EDAnalyzer)          ◄── THIS IS WHAT YOU NEED
    │  Reads: MuonStub collections + SimTrack + GenParticle
    │  Writes: flat ROOT TTree ("simOmtfPhase2Digis/OMTFHitsTree")
    │  File size: ~1–5 kB/event (only OMTF branches)
    │
    ▼  Python OMTFDataset.py
    │  Reads: OMTFHitsTree via uproot
    │  Builds: torch_geometric.data.Data objects (graph per event)
    │  Writes: .pt files (one graph list per ROOT file)
    │
    ▼  HECIN+OC GNN training
```

The tiers that come **after** (RECO, AOD, MiniAOD, NanoAOD, L1Nano) progressively discard lower-level information:

```
GEN-SIM-DIGI (full)
    │ RECO step
    ▼
RECO / AOD
    │ PAT step
    ▼
MiniAOD           ← muon candidates, no stubs
    │ NanoAOD step
    ▼
NanoAOD           ← high-level objects only
    │ L1Nano specialization
    ▼
L1Nano            ← L1T trigger objects (GMT muons, jets, MET…), no internal stubs
```

---

### 16.2 What Each Tier Contains (and What It Lacks for the GNN)

| Tier | Has SimTrack/SimHit | Has OMTF stubs (MuonStub) | Has stub phiB/quality/layer | Has gen pT/η/d₀ | Has reco muons | Has L1 GMT muons | Stub-level GNN usable? |
|---|---|---|---|---|---|---|---|
| GEN | ✗ | ✗ | ✗ | ✓ (GenParticle only) | ✗ | ✗ | ✗ No stubs at all |
| GEN-SIM | ✓ SimTrack/SimVertex | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ No digitized stubs |
| **GEN-SIM-DIGI + L1T emulation** | ✓ | ✓ (after InputMakerPhase2) | ✓ | ✓ | ✗ | ✓ | **✓ This is the required tier** |
| RECO | ✓ (kept in RECO) | ✓ (if L1T re-run or saved) | Partially | ✓ | ✓ | ✓ | Possible but messy; prefer DIGI |
| AOD | ✗ SimHit dropped | ✗ Stubs usually dropped | ✗ | ✓ GenParticle | ✓ reco::Muon | ✓ | ✗ Stubs gone |
| MiniAOD | ✗ | ✗ | ✗ | ✓ (pruned) | ✓ pat::Muon | ✓ | ✗ |
| NanoAOD | ✗ | ✗ | ✗ | ✓ (flat branches) | ✓ (flat) | ✓ (flat) | ✗ |
| L1Nano | ✗ | ✗ (GMT output only) | ✗ | ✗ (no gen matching) | ✗ | ✓ (GMT muons, pT, η, φ, quality) | ✗ |

**The short answer:** the GNN needs `GEN-SIM-DIGI` with L1T emulation. Everything coarser than that has dropped the internal OMTF stub information.

---

### 16.3 What the GNN Actually Needs from CMSSW

The GNN graph is built per OMTF processor × BX window. Each node is one OMTF stub. The required information falls into three groups:

#### 16.3.1 Stub features (node features) — from `simOmtfPhase2Digis`

These come from the OMTF emulator's `MuonStub` objects, built by `InputMakerPhase2`:

| Feature | CMSSW source | Collection | Branch name in OMTFHitsTree |
|---|---|---|---|
| `phiHw` — stub φ in HW units | `MuonStub::phiHw()` | `MuonStubPtrs2D` | `inputStubPhi` |
| `phiBHw` — DT bending angle | `MuonStub::phiBHw()` | same | `inputStubPhiB` |
| `etaHw` — stub η in HW units | `MuonStub::etaHw()` | same | `inputStubEta` |
| `qualityHw` — stub quality | `MuonStub::qualityHw()` | same | `inputStubQuality` |
| `logicLayer` — layer index 0–17 | `MuonStub::logicLayer()` | same | `inputStubLayer` |
| `type` — MuonStub::Type enum | `MuonStub::type()` | same | `inputStubType` |
| `r` — radius [cm] | Derived from geometry | OMTFConfiguration | `inputStubR` (computed) |
| Bending layer flag | `logicLayer ∈ {1,3,5}` | Derived | `inputStubPhiB == 0` |

All of these exist in `GEN-SIM-DIGI` after the OMTF emulation step runs (`simOmtfPhase2Digis`).

#### 16.3.2 Truth labels — from `SimTrack` + `GenParticle`

| Label | CMSSW source | Collection | Used for |
|---|---|---|---|
| Integer track ID per stub | `CrossingFrame<SimTrack>` + SimHit→DetId matching | `mixedSimHits` or `g4SimHits` | OC clustering (Label Type 2) — associates each stub to a gen muon |
| Binary `isMatched` | Derived from track ID ≠ -1 | — | Node noise label (Label Type 1) |
| `gen_pt`, `gen_charge` | `GenParticle` | `genParticles` | Per-track regression targets |
| `gen_phi`, `gen_eta` | `GenParticle` | `genParticles` | Per-track regression targets |
| `gen_d0`, `gen_Lxy` | `SimVertex` + `SimTrack` | `g4SimHits:SimVertexesiTruth` | d₀ head, displaced targets |
| Binary edge label `y_ij` | Derived: same track ID → 1 | — | Edge classification loss |

`SimTrack` and `SimVertex` are present in `GEN-SIM` and survive into `GEN-SIM-DIGI`. They are **dropped at AOD**.

#### 16.3.3 Processor geometry — from OMTF configuration

| Info | Source |
|---|---|
| `getEdgesFromLogicLayer` — which layers are connected | `LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC` in `converter.py` (hardcoded from OMTFConfiguration) |
| HW→physical η | `HW_ETA_TO_ETA_FACTOR = 0.010875` |
| Layer radii [cm] | `get_stub_r()` from OMTFConfiguration geometry |
| Processor φ boundaries | `OMTFConfiguration::processorPhiRange()` |

These are read from the OMTF XML configuration, not from the event data. They are baked into `converter.py` and do not require access to any ROOT file.

---

### 16.4 Why L1Nano Is Not Sufficient

L1Nano is a NanoAOD extension that saves **GMT output** — the final L1T regional muon candidates after ghost-busting and GMT merging. It contains:

- `L1Mu_pt`, `L1Mu_eta`, `L1Mu_phi`, `L1Mu_charge`, `L1Mu_quality` — one entry per GMT muon candidate
- Some L1T jet, MET, HT objects

**What L1Nano does NOT have:**
- Individual OMTF stubs (which layers fired, phiB, quality per stub)
- Internal OMTF processor state (which processor, which BX, how many hits)
- SimTrack associations (all sim-level info dropped at NanoAOD step)
- Gen-level d₀, Lxy (these are dropped at AOD; NanoAOD only keeps high-level gen objects)

L1Nano is useful for **trigger efficiency studies** (comparing L1T output to offline reco), but you cannot build a stub-level GNN graph from it. The emeleTrigger `L1NanoDataset.py` uses L1Nano only as a cross-check dataset with 3 features (`tfLayer`, `offeta1`, `offphi1`) — it is not suitable for HECIN+OC training.

---

### 16.5 Why Standard NanoAOD/AOD/MiniAOD Is Not Sufficient

| Tier | Problem |
|---|---|
| **AOD** | `SimTrack`/`SimHit` collections are dropped. Cannot build integer per-stub track ID. `MuonStub` objects also not saved by default (they are transient in the L1T emulation). |
| **MiniAOD** | AOD is further compressed. Only `pat::Muon` objects survive. No L1T internal state. |
| **NanoAOD** | Flat ntuple of high-level objects. No access to subdetector hits of any kind. |

**Exception:** If you explicitly save `MuonStub` collections and `SimTrack` to AOD (via a custom `outputCommands` in the step2 config), you could run the OMTF dumper on AOD. In practice this is never done — it is always simpler to run the dumper at the DIGI step where all collections are naturally present.

---

### 16.6 The Correct CMSSW Workflow for GNN Training Data

```
Step 1 — GEN-SIM  (cmsDriver.py, step1)
──────────────────────────────────────
Input:  None (particle gun or LHE file from MC production)
Output: GEN-SIM ROOT file (~50–200 MB per 10K events)

Collections produced that you will use later:
  • GenParticle           → gen pT, η, φ, charge, status
  • SimTrack             → simTrackId (integer), pT, pdgId
  • SimVertex            → decay position (x, y, z) → d₀, Lxy
  • g4SimHits:*          → hits in each subdetector (CSC, DT, RPC)


Step 2 — DIGI + L1T emulation  (cmsDriver.py, step2)
──────────────────────────────────────────────────────
Input:  GEN-SIM ROOT file
Output: GEN-SIM-DIGI ROOT file  (adds ~100–300 MB per 10K events)

Key additions:
  • CSCDigi, DTDigi, RPCDigi  → digitized subdetector hits
  • simOmtfPhase2Digis        → full OMTF emulation
       ├── InputMakerPhase2 runs: builds MuonStub objects
       ├── GoldenPatternAlgorithm runs: matches stubs to patterns
       └── PtAssignment runs: assigns pT to candidates
  • SimHit in CrossingFrame   → PU200 overlay (if requested)

PU200 overlay:
  Add  --pileup=Run3_Flat55To75_PoissonOOTPU  (or Phase2 equivalent)
  and  --pileup_input=<MinBias_GEN-SIM files>
  to the cmsDriver.py command


Step 3 — OMTF Dumper  (custom EDAnalyzer, run on GEN-SIM-DIGI)
────────────────────────────────────────────────────────────────
Input:  GEN-SIM-DIGI ROOT file
Output: Small flat ROOT ntuple  (~1–5 MB per 10K events)

This is the step that produces the OMTFHitsTree used by OMTFDataset.py.
The dumper reads:
  • simOmtfPhase2Digis stubs  → all node features
  • GenParticle               → gen kinematic targets
  • SimTrack + SimHit→DetId   → per-stub integer track ID (stub_muon_id)
  • SimVertex                 → gen_d0, gen_Lxy

The existing emeleTrigger dumper (INTREPID / Dumper_Ntuples_v250514/) already
does most of this. Missing branches needed for HECIN+OC:
  • inputStubPhiB  (phiB per stub — needed for bending feature + intra-station edges)
  • stub_muon_id   (integer SimTrack ID per stub — needed for OC labels)
  • gen_d0, gen_Lxy (from SimVertex, not GenParticle)

Existing branches that are sufficient:
  • inputStubEtaG, inputStubCosPhi, inputStubSinPhi, inputStubR
  • inputStubLayer, inputStubType, inputStubIsMatched (= stub_muon_id > -1, binary)
  • muon_pt, muon_eta, muon_phi, muon_charge


Step 4 — Python preprocessing  (OMTFDataset.py / custom)
──────────────────────────────────────────────────────────
Input:  OMTFHitsTree ROOT ntuple  (EOS path)
Output: List of torch_geometric.data.Data objects → .pt file

Per event → per processor × BX window:
  • Nodes: one per valid stub → x_cont [N, 6], node_type [N], layer_id [N]
  • Edges: built by getEdgesFromLogicLayer() from converter.py
           → edge_index [2, E], edge_attr_cont [E, 7], pair_type [E]
  • Node labels: y_node [N] (0/1 = noise/signal), track_id [N] (integer)
  • Edge labels: y_edge [E] (0/1 = different/same muon)
  • Regression targets: kappa [N], phi0 [N], d0 [N], charge [N], beta [N]
  • Condensation: pos [N, 2] (2D OC space), target pos from SimTrack


Step 5 — GNN training
──────────────────────
Input:  .pt graph files from Step 4  (EOS path)
Output: trained HECIN+OC model checkpoint (.pt)
```

---

### 16.7 The Existing EOS Datasets and Their Tier

The datasets referenced in `submitJobs_DatasetCreation.py`:

```
/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/
├── HTo2LongLivedTo2mu2jets/       ← GEN process: H → 2 LLPs → 4μ + 2 jets
├── MuGun_Displaced/               ← Single displaced muon gun (Lxy > 0)
└── MuGun_FullEta_OneOverPt_1to100/ ← Single prompt muon gun, 1/pT flat
```

**Tier:** These are already the custom OMTF dumper output — flat ROOT ntuples at **Step 3** in the workflow above. They are NOT raw GEN-SIM-DIGI; the heavy CMSSW processing (Steps 1–2) has already been run by `folguera` and the output condensed into the OMTFHitsTree format.

**What this means practically:**
- You do **not** need to re-run GEN-SIM or DIGI — the OMTF stubs are already extracted
- You **do** need to check which branches are present (run the Phase 1 structural audit from OMTF_CLASN_Build_Guide.md §14.2) before building graphs
- Any missing branches (e.g. `inputStubPhiB`, `stub_muon_id`) cannot be recovered from these ntuples — you would need to go back to the GEN-SIM-DIGI ROOT files and re-run the dumper with extended branches

---

### 16.8 Dataset-to-GNN-Sample Mapping

| EOS dataset | Physics process | Maps to GNN samples | PU | Missing for HECIN+OC |
|---|---|---|---|---|
| `MuGun_FullEta_OneOverPt_1to100` | Single prompt μ, 1/pT flat, pT ∈ [1,100] GeV | **S1** (prompt, PU0) | 0 | `stub_muon_id` (binary `isMatched` present but not integer ID), `gen_d0`, `gen_Lxy` |
| `MuGun_Displaced` | Single displaced μ | **S2** (displaced, PU0) | 0 | Same as above; also need to verify d₀ range covered |
| `HTo2LongLivedTo2mu2jets` | H → 2 LLP → 4μ + jets | **S5** (LLP, PU0) — multi-muon; could provide **S3** if two muons land in same sector | 0 | `stub_muon_id` integer required (multi-muon sectors need OC labels), `gen_d0`, `gen_Lxy` from SimVertex |

**What is entirely absent from the EOS datasets:**
- **S3** (explicit 2-muon same-sector sample) — partially covered by H→2LLP events where two muons land in one processor, but not a dedicated sample
- **S4** (boundary-crossing μ) — not a separate sample; would need φ-filtered subsets of S1
- **S6** (hard-negative fakes) — not present; must be generated or injected as augmentation
- **B1–B4** (any PU200 sample) — no PU200 dataset in the EOS directory
- **B3** (noise-only PU200) — absent

**Coverage verdict using §14 requirements:**

| Requirement | Covered? | Comment |
|---|---|---|
| Prompt μ, pT ∈ [2, 200] GeV | ⚠️ Partial | Range is [1, 100] GeV; missing pT > 100 GeV |
| η range [0.8, 1.24] | ✓ Likely (FullEta gun) | Verify with audit |
| Displaced, d₀ ∈ [0, 50 cm] | ⚠️ Partial | Verify Lxy range; LLP sample may cover but needs check |
| LLP, Lxy ∈ [0, 500 cm] | ⚠️ Partial | H→2LLP ctau depends on mass/ctau settings |
| Multi-muon (S3) | ⚠️ Accidental | Some H→2LLP events have 2μ in sector — not controlled |
| Any PU200 | ✗ Absent | Must generate B1–B4 from scratch |
| Integer SimTrack ID per stub | ✗ Absent | Only binary `isMatched`; fatal for OC training |
| Per-stub d₀/Lxy targets | ✗ Absent | `gen_d0`, `gen_Lxy` not in current dumper branches |
| `inputStubPhiB` | ✗ Check | Not confirmed present; needed for bending feature |

---

### 16.9 What You Actually Need to Do

Given the existing EOS datasets, the practical steps are:

#### Option A — Extend the existing ntuples (preferred if GEN-SIM-DIGI files are accessible)

1. Find the GEN-SIM-DIGI ROOT files that were the input to the existing dumper jobs. Ask `folguera` for the CMSSW job configs (`crab_*.py` or `cmsDriver.py` commands).
2. Extend `OmtfTrainingNtuplizer.cc` (or the equivalent emeleTrigger dumper) to add:
   - `stub_muon_id[N]`: SimTrack ID per stub (SimHit→DetId→MuonStub matching)
   - `gen_d0[3]`, `gen_Lxy[3]`: from SimVertex, not GenParticle
   - `inputStubPhiB` if not already present
3. Re-run the dumper jobs (Step 3 only — no need to redo GEN-SIM or DIGI) on the existing GEN-SIM-DIGI files.
4. Add PU200 samples: run Step 2 with PU200 overlay on the same generator configs, then dump.

#### Option B — Use existing ntuples with degraded training

1. Use binary `isMatched` as a proxy for `stub_muon_id`. For single-muon PU0 events, all matched stubs get ID=0 and unmatched get ID=-1. This is exactly correct for S1/S2/S5 (one muon per event).
2. Skip OC multi-track training (no S3 with correct per-stub IDs). Train OC only on single-muon events.
3. Skip `gen_d0`, `gen_Lxy` targets; disable the d₀ head or initialize it with zero targets. 
4. Accept that the quality/noise head cannot be trained (no noise-only events B3/B4).
5. Performance penalty: OC will not handle multi-muon events correctly; d₀ head untrained; fake rejection untrained.

#### Option C — Full dataset generation from scratch

Follow §14.6 (CMSSW generator configs) to produce all S1–S6, B1–B4 from scratch with the correct ntuplizer. This is the right long-term approach but takes 2–4 weeks of CMSSW work.

---

### 16.10 Quick Decision Table

| You have | You can train | Missing capability |
|---|---|---|
| Existing EOS ntuples as-is (binary `isMatched` only) | HECIN node noise head only; κ/φ₀/d₀ regression on single-muon events | OC multi-track; d₀ head; fake rejection head; PU robustness |
| Existing + add `stub_muon_id` (re-run dumper) | Full HECIN+OC on single-muon PU0 | PU200 robustness; hard fakes |
| Existing + `stub_muon_id` + PU200 B1–B3 | Full HECIN+OC | Hard fakes (S6, B4) |
| Full S1–S6 + B1–B4 (§14 spec) | Complete HECIN+OC as designed | — |
