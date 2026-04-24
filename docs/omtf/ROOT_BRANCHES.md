# ROOT Ntuple Branch Reference

Complete branch listing from the CMSSW_14_2_0_pre2 production jobs.
Inspected from: `omtf_hits_S1_0.root` and `omtf_nano_S1_0.root`
EOS path: `/eos/user/p/pleguina/omtf_hecin_datasets/prod/<SAMPLE>/`

The notes below stay brief for obvious bookkeeping fields and add extra physics context only where the branch name alone is easy to misread.

---

## Coordinate systems and scale compatibility (verified in CMSSW)

This section is code-checked against CMSSW sources, not inferred from names.

### A) OMTF internal phi scale (stubs / pattern matching)

- Used by: `reg_stub_phiHw`, `hits_phiHw`, `omtfRefHitPhi`, and OMTF internal candidate phi code (`omtfPhi` in `OMTFHitsTree`).
- Full range: 5400 bins over $2\pi$ (from OMTF config XML `nPhiBins=5400`).
- **Frame: PROCESSOR-LOCAL (not global CMS phi)**  
  Code proof: `OMTFinputMaker.cc` calls `angleConverter.getProcessorPhi(getProcessorPhiZero(config, iProcessor), ...)`,
  which subtracts the processor's `phiZero` from the global chamber offset before storing. `DataROOTDumper2AllInput.cc` line 122 writes the result directly to the tree without re-adding `phiZero`.  
  `getProcessorPhiZero(proc) = nPhiBins/nProcessors × proc + nPhiBins/24 = 1800 × proc + 225`  
  Processor zero-phi offsets (5400-bin scale, nProcessors=3):  
  - proc 0: phiZero = 225 bins ≈ +0.262 rad (≈ +15°)  
  - proc 1: phiZero = 2025 bins ≈ +2.356 rad (≈ +135°)  
  - proc 2: phiZero = 3825 bins ≈ +4.451 rad (≈ +255°)
- To get **global** phi in radians from `phiHw`:

$$
\phi_{\text{global, rad}} = \bigl(\phi_{\text{hw}} + \text{phiZero}(\text{proc})\bigr) \cdot \frac{2\pi}{5400}
$$

- A naive `phiHw × 2π/5400` (ignoring phiZero) gives a wrong global phi.

### B) GMT phi scale (regional muon candidate hwPhi)

- Used by: Nano `omtf_hwPhi` (`RegionalMuonCand::hwPhi()` scale).
- Full range: 576 bins over $2\pi$ (`phiGmtUnit = 2\pi/576`).
- Convert to radians:

$$
\phi_{\text{rad}} = \phi_{\text{hw,gmt}} \cdot \frac{2\pi}{576}
$$

- Local vs global note: `hwPhi` is processor-local. Global phi = `(hwPhi + proc × 96) mod 576` bins.

### C) OMTF eta hw scale

- Used by: `reg_stub_etaHw`, `hits` packed eta byte, `omtfHwEta`, Nano `omtf_hwEta`.
- LSB is identical across those OMTF quantities:

$$
\eta \approx \eta_{\text{hw}} \cdot 0.010875
$$

### D) DT bending (`phiB`) scale

- Used by: `reg_stub_phiBHw`, `hits_phiBHw`.
- Not an absolute phi; this is a local bending-angle proxy.
- In Phase-2 input conversion:

$$
\phi B_{\text{hw}} = \phi B_{\text{digi}} \cdot \frac{\texttt{dtPhiBUnitsRad}}{2048}
$$

- In your Phase-2 config, `dtPhiBUnitsRad = 1024`, so the effective scale is about $1024$ units/rad.
- Therefore `phiBHw` is a bending proxy, not an absolute position angle.

### E) GMT hybrid stubs (KMTF/TPS) scale

Both Nano tables share identical branch names (`_stubVars` schema, `omtfNanoTables_cff.py`) and the same `l1t::MuonStub` C++ type, but the **integer quantization differs** because the two stub producers (`L1TPhase2GMTBarrelStubProcessor` vs `L1TPhase2GMTEndcapStubProcessor`) use different LSBs. Source: `L1Trigger/Phase2L1GMT/python/gmtStubs_cfi.py` and both `.cc` files.

**`coord1` integer scales (code-verified):**

| Collection | Producer file | `coord1` formula | Internal precision | LSB [rad/count] | NanoAOD storage | Note |
|---|---|---|---|---|---|---|
| KMTF (barrel) | `L1TPhase2GMTBarrelStubProcessor.cc` | `ap_int<18>` from sector+phiBend, `globalPhi = phi × π/2¹⁷` | 18-bit signed (±131071) | `phiLSB = 2.3968450e-05` (≈ π/131072) | `int16` (Short_t) | **Truncation: values for sector ≥1 exceed ±32767 — high 2 bits lost. `coord1` integer is NOT reliably decodable to global phi without `phiRegion`. Use `offlineCoord1` (float) for phi.** |
| TPS (endcap) | `L1TPhase2GMTEndcapStubProcessor.cc` | `int(gp.phi() / coord1LSB_)` | full `int` | `coord1LSB = 0.00076660156 × 8 ≈ 6.133e-3` | `int16` (Short_t) | Range ±512 for [−π, +π]; fits Short_t cleanly |

Ratio KMTF/TPS LSB: ~256. **The integer `coord1` values are not interchangeable.**

**`coord2` integer scales:**
- KMTF: `coord2 = phiS.phiBend()` — raw DT phiBend hardware integer; `offline_coord2 = phiBend × 0.49e-3 rad`.
- TPS: `coord2 = int(rpc_phi / coord1LSB_)` (same 6.133e-3 LSB as `coord1`) for RPC-matched stubs; 0 for CSC-only stubs.

**`eta1`/`eta2` integer scales — same for both:**
- Both producers: `etaLSB = 7.68334e-04 × 32 ≈ 0.02459` (from cfi `eta1LSB = cms.double(7.68334e-04*32)` for both Barrel and Endcap PSet).
- Note: the `DataFormats/L1TMuonPhase2/interface/MuonStub.h` comment says `3.0/512` — that is a stale comment; the actual Phase2L1GMT implementation uses `0.02459`.

**`offlineCoord1/2`, `offlineEta1/2`:**
- `offlineCoord1` = global phi in radians (KMTF: `phi × π/131072`; TPS: `gp.phi().value()` directly).
- `offlineEta1/2` = physical eta (integer × 0.02459).

### Quick same-scale checklist

- Same eta scale: `reg_stub_etaHw`, `omtfHwEta`, Nano `omtf_hwEta` (LSB 0.010875).
- Same OMTF phi scale (processor-local): `reg_stub_phiHw`, `hits_phiHw`, `omtfRefHitPhi`, `omtfPhi` (5400 bins/2π, local frame; add phiZero(proc) to get global bins).
- Same DT bending scale: `reg_stub_phiBHw`, `hits_phiBHw`.
- Same eta integer scale: `MuonStubKmtf_eta1/eta2` and `MuonStubTps_eta1/eta2` (LSB ≈ 0.02459).
- Different `coord1` integer scales: `MuonStubKmtf_coord1` (18-bit LSB 2.40e-5 rad, stored int16 with truncation for sector≥2) vs `MuonStubTps_coord1` (LSB 6.13e-3 rad, fits int16 cleanly).
- `MuonStub{Kmtf,Tps}_offlineCoord1`: both in radians.

---

## omtf_hits_<SAMPLE>_<N>.root

Contains two trees inside directory `simOmtfPhase2Digis/`.

---

### OMTFAllInputTree  (one entry per processor-window per event)

**Use for: GNN training**

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `reg_eventNum` | `UInt_t` | Event number (matches NanoAOD `event` field) |
| `reg_iProcessor` | `UChar_t` | OMTF processor index **0–2** (three 120° sectors per eta side; `omtf_pos` and `omtf_neg` each loop 0–2 independently — `OmtfEmulation.cc` lines 295, 312). This value is passed directly to `getProcessorPhiZero()` |
| `reg_mtfType` | `Char_t` | Processor type (OMTF=1, BMTF=2, EMTF=3) |
| `reg_stub_layer` | `vector<signed char>` | Logic layer index 0–17 per stub |
| `reg_stub_phiHw` | `vector<short>` | Trigger phi of the stub in OMTF hardware bins (5400 bins / 2π), **in the processor's local frame** (phiZero subtracted; see Section A). Same local scale as `hits_phiHw`/`omtfRefHitPhi`. To get global phi add `phiZero(proc) = 1800×proc+225` before converting to radians. Not the same as Nano `omtf_hwPhi` (576-bin global GMT scale) |
| `reg_stub_phiBHw` | `vector<short>` | DT bending-angle proxy in hardware units; local curvature input (not an absolute phi), 0 for CSC/RPC stubs |
| `reg_stub_etaHw` | `vector<signed char>` | Coarse trigger eta code for the stub; physical eta = etaHw × 0.010875 |
| `reg_stub_r` | `vector<short>` | Radial distance in cm |
| `reg_stub_quality` | `vector<signed char>` | Stub quality word from the muon trigger primitive, on the 0–15 hardware scale |
| `reg_stub_type` | `vector<signed char>` | `MuonStub::Type` enum identifying the detector technology/subtype that produced the stub |
| `reg_stub_bx` | `vector<signed char>` | BX offset relative to the in-time bunch crossing (typically −3..+3; 0=in-time) |
| `reg_stub_trackId` | `vector<signed char>` | **0 = noise/PU or no usable truth link; 1..K = 1-indexed gen-muon label** from SimTrack matching |
| `reg_stub_ambiguous` | `vector<unsigned char>` | 1 when the SimTrack digi votes do not give a clear majority for this stub, so the truth label is low-confidence |

All `reg_stub_*` vectors have the same length for a given entry (one element per stub in that window).

---

### OMTFHitsTree  (one entry per OMTF candidate matched to a gen-muon)

**Use for: trigger efficiency, pT resolution, fake rate studies**

#### Gen-muon truth (scalars)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `eventNum` | `UInt_t` | Event number |
| `muonEvent` | `Short_t` | Which muon in the event (index) |
| `muonPt` | `Float_t` | Gen-muon pT [GeV] |
| `muonEta` | `Float_t` | Gen-muon η at vertex |
| `muonPhi` | `Float_t` | Gen-muon φ at vertex |
| `muonPropEta` | `Float_t` | Gen-muon η extrapolated from the production vertex to muon station 2, i.e. the angle actually used when comparing to trigger primitives |
| `muonPropPhi` | `Float_t` | Gen-muon φ extrapolated from the production vertex to muon station 2, i.e. the trigger-facing azimuth at that station |
| `muonCharge` | `Char_t` | Gen-muon charge (±1) |
| `muonDxy` | `Float_t` | Gen-muon transverse impact parameter d₀ [cm] with sign convention from the generator / tracking-particle definition |
| `muonRho` | `Float_t` | Radial distance [cm] of the muon production or decay vertex from the beamline in the transverse plane |
| `parentPdgId` | `Short_t` | PDG ID of the generator-level mother particle that produced the muon |
| `vertexEta` | `Float_t` | Pseudorapidity of the production / decay vertex direction as seen from the origin |
| `vertexPhi` | `Float_t` | Azimuth of the production / decay vertex direction as seen from the origin |

#### OMTF candidate output (scalars)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `omtfPt` | `Float_t` | OMTF assigned pT [GeV] (after LUT pT assignment) |
| `omtfUPt` | `Float_t` | OMTF pT estimate before the final curvature / quality constraint is applied [GeV] |
| `omtfEta` | `Float_t` | OMTF candidate η converted from hw (`hwEta × 0.010875`) |
| `omtfPhi` | `Float_t` | OMTF candidate phi code in OMTF bins (stored as float for ROOT I/O), **processor-local frame** (`getPhi()` returns the uncorrected local code; same frame as `reg_stub_phiHw`). To get global phi in radians: `(omtfPhi + phiZero(omtfProcessor)) × 2π/5400`. Not radians, not global |
| `omtfCharge` | `Char_t` | OMTF charge (±1) |
| `omtfHwEta` | `Short_t` | OMTF candidate η hardware code (LSB 0.010875; same eta scale as `reg_stub_etaHw`) |
| `omtfProcessor` | `Char_t` | OMTF processor index |
| `omtfScore` | `Short_t` | Pattern matcher score (higher = better match) |
| `omtfQuality` | `Char_t` | OMTF quality word |
| `omtfRefLayer` | `Char_t` | Reference layer used for pT assignment |
| `omtfRefHitNum` | `Char_t` | Reference hit index |
| `omtfRefHitPhi` | `Short_t` | Reference hit phi HW |
| `omtfFiredLayers` | `UInt_t` | Bitmask of fired logic layers (bit k = layer k fired) |
| `killed` | `Bool_t` | True if this candidate was ghost-busted (suppressed by a better candidate in same region) |

#### Per-candidate stubs (vectors — one element per stub on the candidate track)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `hits` | `vector<unsigned long>` | Packed uint64: bits 0–7=layer, 8–15=quality, 16–23=etaHw, 24–31=valid, 32–47=deltaR, 48–63=phiDist |
| `hits_phiHw` | `vector<short>` | Stub phi HW in OMTF 5400-bin scale, **processor-local frame** (same local scale as `reg_stub_phiHw`; add `phiZero(omtfProcessor)` to globalize) |
| `hits_phiBHw` | `vector<short>` | DT bending angle HW (0 for non-DT) |
| `hits_r` | `vector<short>` | Radial distance cm |
| `hits_type` | `vector<signed char>` | MuonStub::Type |
| `hits_bx` | `vector<signed char>` | BX offset |

#### Matching quality (scalars)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `deltaEta` | `Float_t` | Δη between OMTF candidate and gen-muon |
| `deltaPhi` | `Float_t` | Δφ between OMTF candidate and gen-muon |

---

## omtf_nano_<SAMPLE>_<N>.root

Standard CMS NanoAOD format; main tree: `Events` (one entry per event).

### Event bookkeeping

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `run` | `UInt_t` | Run number |
| `luminosityBlock` | `UInt_t` | Lumi block |
| `event` | `ULong64_t` | Event number (matches OMTFAllInputTree `reg_eventNum` after uint32 cast/modulo) |
| `bunchCrossing` | `UInt_t` | BX number |
| `orbitNumber` | `UInt_t` | Orbit number |

### Gen-level muons (`nGenMuon` entries per event)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `nGenMuon` | `Int_t` | Number of gen muons in event |
| `GenMuon_pt` | `Float_t` | pT [GeV] |
| `GenMuon_eta` | `Float_t` | η |
| `GenMuon_phi` | `Float_t` | φ |
| `GenMuon_charge` | `Int_t` | ±1 |
| `GenMuon_pdgId` | `Int_t` | PDG ID (±13 for muons) |
| `GenMuon_status` | `Int_t` | Pythia8 status code |
| `GenMuon_dXY` | `Float_t` | Transverse impact parameter d₀ [cm] |
| `GenMuon_lXY` | `Float_t` | Transverse decay length [cm] |
| `GenMuon_vx` | `Float_t` | Production vertex x [cm] |
| `GenMuon_vy` | `Float_t` | Production vertex y [cm] |
| `GenMuon_vz` | `Float_t` | Production vertex z [cm] |
| `GenMuon_etaSt1` | `Float_t` | η propagated to 1st muon station |
| `GenMuon_etaSt2` | `Float_t` | η propagated to 2nd muon station |
| `GenMuon_phiSt1` | `Float_t` | φ propagated to 1st muon station |
| `GenMuon_phiSt2` | `Float_t` | φ propagated to 2nd muon station |

### OMTF GMT output (`nomtf` entries per event)

These are the OMTF candidates **after ghost busting**, as sent to the GMT.

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `nomtf` | `Int_t` | Number of OMTF candidates |
| `omtf_hwPt` | `Short_t` | Final GMT-facing pT code in hardware units |
| `omtf_hwPtUnc` | `Short_t` | Unconstrained pT code before the final trigger constraint is imposed |
| `omtf_hwEta` | `Short_t` | Quantized OMTF η code used by the trigger (LSB 0.010875) |
| `omtf_hwPhi` | `Short_t` | Quantized OMTF regional phi code in GMT scale (576 bins / 2π); use processor index to globalize |
| `omtf_hwQual` | `Short_t` | Quality word |
| `omtf_hwDXY` | `Short_t` | Hardware displacement class, i.e. the trigger's coarse encoding of transverse impact parameter / $d_{xy}$ |
| `omtf_Q` | `Short_t` | Charge |
| `omtf_processor` | `Short_t` | Processor index |
| `omtf_muIdx` | `Short_t` | Index into GenMuon collection (−1 if unmatched) |

### Muon stubs — KMTF format (`nMuonStubKmtf` entries per event)

| Branch | ROOT type | Description |
|--------|-----------|-------------|
| `nMuonStubKmtf` | `Int_t` | Number of stubs |
| `MuonStubKmtf_isBarrel` | `Bool_t` | True if DT/RPC barrel |
| `MuonStubKmtf_isEndcap` | `Bool_t` | True if CSC/RPC endcap |
| `MuonStubKmtf_bxNum` | `Short_t` | BX offset |
| `MuonStubKmtf_coord1` | `Short_t` | Global phi integer, **KMTF barrel scale**: LSB = `phiLSB = 2.3968450e-05 rad` (≈ π/131072), 18-bit internal precision. **Warning: stored as int16 in NanoAOD — the upper 2 bits are lost for all sectors ≥1 (sector=1: max normalization0=21845 fits; sector=2: normalization0=43690 overflows int16). The int16 value alone cannot be converted back to global phi. Use `offlineCoord1` (float branch) for phi.** |
| `MuonStubKmtf_coord2` | `Short_t` | DT phiBend integer (raw hardware units); `offline_coord2 = coord2 × 0.49e-3 rad`; not an absolute phi |
| `MuonStubKmtf_eta1` | `Short_t` | First eta measurement HW; LSB ≈ 0.02459 (= 7.68334e-04 × 32); NOT 3.0/512 (stale header comment) |
| `MuonStubKmtf_eta2` | `Short_t` | Second eta measurement HW; same LSB as `eta1` ≈ 0.02459 |
| `MuonStubKmtf_etaQuality` | `Short_t` | Eta quality |
| `MuonStubKmtf_etaRegion` | `Short_t` | Eta region index |
| `MuonStubKmtf_phiRegion` | `Short_t` | Phi region index |
| `MuonStubKmtf_quality` | `Short_t` | Stub quality |
| `MuonStubKmtf_stubType` | `Short_t` | Stub type |
| `MuonStubKmtf_tfLayer` | `Short_t` | TF logic layer |
| `MuonStubKmtf_depthRegion` | `Short_t` | Depth region |
| `MuonStubKmtf_id` | `Short_t` | Stub ID |
| `MuonStubKmtf_addr` | `Int_t` | Address |
| `MuonStubKmtf_kmtf_addr` | `Int_t` | KMTF-specific address |
| `MuonStubKmtf_offlineCoord1` | `Float_t` | Global phi [rad]: computed from full 18-bit `phi` as `phi × π/131072` before int16 truncation; this is the reliable phi quantity for KMTF barrel stubs |
| `MuonStubKmtf_offlineCoord2` | `Float_t` | DT phiBend in rad (`coord2 × 0.49e-3`) |
| `MuonStubKmtf_offlineEta1` | `Float_t` | Physical η (`eta1 × 0.02459`) |
| `MuonStubKmtf_offlineEta2` | `Float_t` | Physical η (`eta2 × 0.02459`) |

### Muon stubs — TPS format (`nMuonStubTps` entries per event)

Same branch names as KMTF (`MuonStubTps_` prefix) but **`coord1`/`coord2` are on a different integer scale** (see Section E). Key differences:

| Branch | Scale / notes |
|---|---|
| `MuonStubTps_coord1` | Global phi, **TPS endcap scale**: LSB = `coord1LSB = 6.133e-3 rad` (= 0.00076660156 × 8); `phi_rad = coord1 × 6.133e-3`. NOT same as KMTF |
| `MuonStubTps_coord2` | Second phi coordinate (RPC-matched or RPC-only stubs, same 6.133e-3 LSB); 0 for CSC-only stubs |
| `MuonStubTps_eta1/eta2` | Same LSB ≈ 0.02459 as KMTF — directly comparable |
| `MuonStubTps_offlineCoord1` | Global phi [rad] — directly comparable to `MuonStubKmtf_offlineCoord1` |
| `MuonStubTps_offlineCoord2` | Second phi [rad] |
| `MuonStubTps_offlineEta1/2` | Physical η — directly comparable to KMTF |

TPS stubs are endcap-oriented (CSC + endcap RPC), produced by the TPS (Track-based Pair Selector) algorithm before matching with tracks.

---

## Joining the two files

To attach gen-muon kinematics from NanoAOD to `OMTFAllInputTree` stubs:

```python
# Both files have matching event numbers
# OMTFAllInputTree: reg_eventNum  (UInt_t)
# NanoAOD Events:  event          (ULong64_t)

# For a gen-muon regression target on a stub:
#   reg_stub_trackId > 0  →  muon index = reg_stub_trackId - 1  (0-indexed into GenMuon_*)
#   reg_stub_trackId == 0 →  noise; fill targets with 0.0 and mask in loss

# Note: reg_eventNum comes from the CMSSW event ID modulo 2^32.
# NanoAOD 'event' is the full 64-bit ID. Cast to uint32 before matching.
```

To find which OMTF candidate in `OMTFHitsTree` corresponds to a given
`OMTFAllInputTree` window: match on `(eventNum, omtfProcessor)` vs `(reg_eventNum, reg_iProcessor)`.
