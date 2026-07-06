# CMSSW Muon Trigger Variable Audit for Displaced-Muon ML

**Date:** 2026-05-25 (re-verified 2026-07-05 against CMSSW_17_0_0_pre3)  
**CMSSW release audited:** CMSSW_14_2_0_pre2 (local checkout, 2024)  
**Release base:** `/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_2_0_pre2`  
**Cross-checked against:** CMSSW_17_0_0_pre3 (`/cvmfs/cms.cern.ch/el9_amd64_gcc13/cms/cmssw/CMSSW_17_0_0_pre3`, latest available 2026)  
**Scope:** DT, CSC, RPC, GMT/BMTF/OMTF/TPS trigger primitives for displaced-muon ML study

> **Version-currency note:** The local checkout is CMSSW_14_2_0_pre2 (2024). All data-format headers below were diffed against the newest release on cvmfs, CMSSW_17_0_0_pre3 (2026). Of the 12 core muon data-format headers, **9 are byte-identical** and **3 changed** — only two of which affect this audit. See Section 1.5 for the full delta. Every method name, member name, and line number in this report was confirmed present in **both** releases unless explicitly annotated as "2026-only".

---

## 1. Executive Summary

This audit inventories all publicly available muon trigger primitive data formats in CMSSW 14.2.0_pre2 that could be used as inputs to a machine-learning displaced-muon discriminator operating in the Level-1 overlap trigger region.

**Key findings:**

- **DT** provides both phi (position + bending angle `phiB`) and theta (position `z` + slope `k`) primitives. The Phase-2 extended format (`L1Phase2MuDTExtPhDigi`) adds `xLocal`, `tanPsi`, wire-by-wire TDC data, and `t0`/`chi2` — extremely valuable for displaced identification. Run-3 primitives already carry `phiB` (bending angle).
- **CSC** provides the LCT format with `keywire`, `strip`, `pattern`/`slope`, and `bend`. Run-3 adds fractional strip resolution and a `getSlope()` method giving a direct local angle proxy. These are highly useful for pointing-residual studies.
- **RPC** only provides strip number and BX; no local direction or slope. Useful only for timing confirmation and spatial support hits.
- **GMT/TPS stubs** (`l1t::MuonStub`, Phase-2) encode `coord1` (global phi in units of 30°/2048), `coord2` (bending angle for barrel), `eta1`/`eta2` (in units of 3.0/512), plus identity fields. They represent a compact but lossy abstraction — local direction and many per-wire fields are lost.
- **`RegionalMuonCand`** (final BMTF/OMTF/EMTF output) carries `hwPt`, `hwPhi`, `hwEta`, `hwSign`, `hwQuality`, `hwDXY` (Phase-2), `hwPtUnconstrained`. This is the furthest-downstream object before GMT and retains almost no primitive-level local information.
- **OMTF internal `MuonStub`** (Phase-1, `L1Trigger/L1TMuonOverlapPhase1`) stores `phiHw`, `phiBHw`, `etaHw`, `r`, `qualityHw`, `bx`, `logicLayer`, `detId`, and CSC conversion parameters. This is a richer intermediate representation.

**Recommended strategy: Strategy B** (enriched common-coordinate primitive cache using `l1t::MuonStub` Phase-2 as the unified node, supplemented by raw DT `phiB`/`k` and CSC `slope`/`pattern` before aggregation). See Section 13.

---

## 1.5 Cross-Version Verification (2024 CMSSW_14_2_0_pre2 → 2026 CMSSW_17_0_0_pre3)

The local checkout dates from 2024. All 12 core muon data-format headers were `diff`-ed against CMSSW_17_0_0_pre3 (newest release on cvmfs as of 2026-07-05).

| Header | Status 2024→2026 | Impact on audit |
|---|---|---|
| `L1MuDTChambPhDigi.h` | identical | none — audit valid |
| `L1MuDTChambThDigi.h` | identical | none |
| `L1Phase2MuDTPhDigi.h` | identical | none |
| `L1Phase2MuDTThDigi.h` | identical | none |
| `L1Phase2MuDTExtPhDigi.h` | **changed** (added copy/move `operator=` only) | **none** — no new fields |
| `CSCCLCTDigi.h` | identical | none |
| `CSCALCTDigi.h` | identical | none |
| `CSCCorrelatedLCTDigi.h` | **changed** | **RELEVANT** — new `Run3HR` version, `getSlopeEx()`, `getRawSlopeResolution()`, GEM-in-LCT field. See Section 5.3. |
| `RPCDigi.h` | identical | none |
| `MuonStub.h` (Phase-2 TPS) | **changed** | minor — added `hybridStubWord()` firmware serialization + print helpers, no new physics variable. See Section 7.1. |
| `RegionalMuonCand.h` | identical | none — `hwDXY`/`hwPtUnconstrained` unchanged |
| `L1MuKBMTCombinedStub.h` | identical | none |

**Conclusion:** The 2024 checkout is still an accurate basis for this audit. The only substantive 2026 upgrade is the **CSC LCT high-resolution slope (`Run3HR`, 6-bit)** — which *increases* the value of CSC local-direction features for displaced discrimination. If moving to a 2026 release, use `getSlopeEx()` instead of `getSlope()` to access the finer resolution (see Sections 5.3 and 14.1).

---

## 2. Packages and Files Inspected

| Package | Path | Relevance |
|---|---|---|
| `DataFormats/L1DTTrackFinder` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/` | DT run-3 + Phase-2 primitives |
| `DataFormats/CSCDigi` | `$RBASE/src/DataFormats/CSCDigi/interface/` | CSC ALCT, CLCT, LCT digis |
| `DataFormats/RPCDigi` | `$RBASE/src/DataFormats/RPCDigi/interface/` | RPC strip digis |
| `DataFormats/L1TMuon` | `$RBASE/src/DataFormats/L1TMuon/interface/` | RegionalMuonCand, KBMT stub, BMTF track segments |
| `DataFormats/L1TMuonPhase2` | `$RBASE/src/DataFormats/L1TMuonPhase2/interface/` | Phase-2 MuonStub, SAMuon, EMTFHit, KMTFTrack |
| `L1Trigger/L1TMuonOverlapPhase1` | local checkout | OMTF Phase-1 MuonStub, StubMaker, AngleConverter |
| `L1Trigger/L1TMuonOverlapPhase2` | local checkout | OMTF Phase-2 InputMaker, PtAssignment, AngleConverter |
| `L1Trigger/L1TMuon` | local checkout | RegionalMuonRawDigiTranslator |

---

## 3. Data Product Map

| Subdetector | Class | Package | Producer | Stage | Run-3/Phase-2 | EDM branch (typical) | File |
|---|---|---|---|---|---|---|---|
| DT | `L1MuDTChambPhDigi` | `DataFormats/L1DTTrackFinder` | `dttfDigis` / TwinMux | DT phi TP | Run-3 | `l1tTwinMuxDigis:PhIn` | `L1MuDTChambPhDigi.h` |
| DT | `L1MuDTChambThDigi` | `DataFormats/L1DTTrackFinder` | `dttfDigis` / TwinMux | DT theta TP | Run-3 | `l1tTwinMuxDigis:ThIn` | `L1MuDTChambThDigi.h` |
| DT | `L1Phase2MuDTPhDigi` | `DataFormats/L1DTTrackFinder` | Phase-2 DT trigger | Phase-2 DT phi TP | Phase-2 | `dtTriggerPhase2Primitives` | `L1Phase2MuDTPhDigi.h` |
| DT | `L1Phase2MuDTThDigi` | `DataFormats/L1DTTrackFinder` | Phase-2 DT trigger | Phase-2 DT theta TP | Phase-2 | `dtTriggerPhase2Primitives` | `L1Phase2MuDTThDigi.h` |
| DT | `L1Phase2MuDTExtPhDigi` | `DataFormats/L1DTTrackFinder` | Phase-2 DT extended trigger | Phase-2 DT extended phi TP | Phase-2 | `dtTriggerPhase2Primitives:MPHI` | `L1Phase2MuDTExtPhDigi.h` |
| CSC | `CSCALCTDigi` | `DataFormats/CSCDigi` | `muonCSCDigis` / TMB | ALCT wire trigger | Run-3/Phase-2 | `muonCSCDigis:MuonCSCALCTDigi` | `CSCALCTDigi.h` |
| CSC | `CSCCLCTDigi` | `DataFormats/CSCDigi` | `muonCSCDigis` / TMB | CLCT strip trigger | Run-3/Phase-2 | `muonCSCDigis:MuonCSCCLCTDigi` | `CSCCLCTDigi.h` |
| CSC | `CSCCorrelatedLCTDigi` | `DataFormats/CSCDigi` | `muonCSCDigis` / TMB | Correlated LCT | Run-3/Phase-2 | `muonCSCDigis:MuonCSCCorrelatedLCTDigi` | `CSCCorrelatedLCTDigi.h` |
| RPC | `RPCDigi` | `DataFormats/RPCDigi` | `muonRPCDigis` | RPC strip hit | Run-3/Phase-2 | `muonRPCDigis` | `RPCDigi.h` |
| BMTF | `L1MuKBMTCombinedStub` | `DataFormats/L1TMuon` | BMTF KMTF | BMTF Kalman stub (combined phi+eta) | Run-3 | `bmtfDigis` | `L1MuKBMTCombinedStub.h` |
| BMTF | `L1MuBMTrackSegPhi` | `DataFormats/L1TMuon` | BMTF BX | BMTF phi track segment | Run-3 | various | `L1MuBMTrackSegPhi.h` |
| BMTF | `L1MuBMTrackSegEta` | `DataFormats/L1TMuon` | BMTF | BMTF theta track segment | Run-3 | various | `L1MuBMTrackSegEta.h` |
| GMT/TPS | `l1t::MuonStub` (Phase-2) | `DataFormats/L1TMuonPhase2` | BMTF/OMTF/EMTF | GMT input stub (Phase-2) | Phase-2 | `l1tSAMuonsGmt:` | `MuonStub.h` |
| GMT/TPS | `l1t::SAMuon` | `DataFormats/L1TMuonPhase2` | GMT | Standalone muon (Phase-2 GMT output) | Phase-2 | `l1tSAMuonsGmt` | `SAMuon.h` |
| OMTF | `MuonStub` (Phase-1) | `L1Trigger/L1TMuonOverlapPhase1` | OMTF emulator | OMTF internal input stub | Phase-1 | internal only | `MuonStub.h` |
| All TFs | `RegionalMuonCand` | `DataFormats/L1TMuon` | BMTF/OMTF/EMTF | Regional trigger output | Run-3/Phase-2 | `gmtStage2Digis:BMTF`, `OMTF`, `EMTF` | `RegionalMuonCand.h` |

---

## 4. DT Variable Inventory

### 4.1 `L1MuDTChambPhDigi` — Run-3 DT Phi Primitive
**File:** `DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h`  
**Description:** Run-3 DT trigger primitive encoding radial angle (phi position) and bending angle (phiB), plus identity and quality.

| Variable | Method | C++ type | Bit width | Signed | Units | Coordinate | Range | Physical meaning | Firmware | Displaced usefulness |
|---|---|---|---|---|---|---|---|---|---|---|
| Bunch crossing | `bxNum()` | `int` | ~4 | signed | BX units | time | −3..+3 | Collision BX tag | ✓ firmware | MEDIUM — timing consistency |
| Wheel number | `whNum()` | `int` | 3 | signed | detector | barrel | −2..+2 | DT wheel identity | ✓ | MEDIUM — radial geometry |
| Sector number | `scNum()` | `int` | 4 | unsigned | detector | barrel phi | 0..11 | DT sector (30° phi wedge) | ✓ | MEDIUM — phi geometry |
| Station number | `stNum()` | `int` | 2 | unsigned | detector | barrel | 1..4 | DT station = radial depth | ✓ | MEDIUM — radial ordering |
| Radial angle (phi) | `phi()` / `radialAngle` | `int` | 12 | signed | LSB ~1/2048 of π/6 rad | chamber-local phi | −2048..+2047 | Phi position in chamber | ✓ | HIGH — global trajectory phi |
| Bending angle (phiB) | `phiB()` / `bendingAngle` | `int` | 10 | signed | LSB rad/unit | local | −512..+511 | Local segment direction / bend proxy | ✓ | **HIGH** — pointing residual |
| Quality code | `code()` / `qualityCode` | `int` | 3 | unsigned | enum | — | 0..7 | PHTF quality: Lo/Li/Hi/Ho/LL/HL/HH | ✓ | MEDIUM — reject noise |
| TwinMux segment tag | `Ts2Tag()` | `int` | 1 | unsigned | — | — | 0..1 | Second TS tag | ✓ | LOW |
| BX counter | `BxCnt()` | `int` | 2 | unsigned | — | — | 0..3 | BX counter | ✓ | LOW |
| RPC bit | `RpcBit()` | `int` | 1 | unsigned | — | — | −10, 0, 1 | Whether RPC was used to confirm | ✓ | LOW |
| UpDown tag | `UpDownTag()` | `int` | 1 | unsigned | — | — | 0..1 | Superlayer up/down tag | ✓ | LOW |

**DT phiB answer:** `phiB()` / `bendingAngle` is a signed 10-bit integer representing the local segment bending angle. It encodes the **local slope** of the DT segment within the phi superlayer. A muon coming from the beamline at low-pT will show a large but beamline-consistent bending angle; a displaced muon may show an inconsistent combination of `phi` position and `phiB` slope.

**Answer to DT audit questions:**
1. Local direction is encoded in `phiB` (bending angle). No explicit theta direction in Run-3 phi primitive.
2. Theta/z info is in the separate `L1MuDTChambThDigi`.
3. K-like variable: `phiB` is this — it encodes the local phi slope of the track segment.
4. Z variable: see Section 4.3.
5. Phi and theta are **separate** in Run-3 (`L1MuDTChambPhDigi` + `L1MuDTChambThDigi`); merged in Phase-2.
6. Only `phi`, `phiB`, quality, BX, and identity survive into OMTF/BMTF stubs.
7. Per-wire TDC data, t0, chi2 are all lost in Run-3 format.

---

### 4.2 `L1MuDTChambThDigi` — Run-3 DT Theta Primitive
**File:** `DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h`

| Variable | Method | C++ type | Bit width | Signed | Units | Physical meaning | Firmware | Displaced usefulness |
|---|---|---|---|---|---|---|---|---|
| BX | `bxNum()` | `int` | ~4 | signed | BX | Timing | ✓ | MEDIUM |
| Wheel | `whNum()` | `int` | 3 | signed | — | Detector identity | ✓ | MEDIUM |
| Sector | `scNum()` | `int` | 4 | unsigned | — | Detector identity | ✓ | MEDIUM |
| Station | `stNum()` | `int` | 2 | unsigned | — | Radial depth | ✓ | MEDIUM |
| Position [0..6] | `position(i)` / `m_outPos[7]` | `uint8` | 7 bits (7 wires) | unsigned | wire group hit pattern | theta wire pattern | ✓ | HIGH — z/eta consistency |
| Quality [0..6] | `quality(i)` / `m_outQual[7]` | `uint8` | 7 bits | unsigned | — | quality of each theta hit | ✓ | MEDIUM |

**Note:** The `position` array encodes 7 theta wire groups as a hit pattern. This gives coarse z/eta information per chamber. The combination with `L1MuDTChambPhDigi` gives 2D track segment information.

---

### 4.3 `L1Phase2MuDTPhDigi` — Phase-2 DT Phi Primitive
**File:** `DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h`

| Variable | Method | C++ type | Bit width | Physical meaning | Displaced usefulness |
|---|---|---|---|---|---|
| BX | `bxNum()` | `int` | — | Bunch crossing | MEDIUM |
| Wheel | `whNum()` | `int` | — | Barrel wheel | MEDIUM |
| Sector | `scNum()` | `int` | — | Barrel sector | MEDIUM |
| Station | `stNum()` | `int` | — | Station 1–4 | MEDIUM |
| Superlayer | `slNum()` | `int` | — | SL1 (phi) or SL3 (phi) | MEDIUM |
| Phi angle | `phi()` / `m_phiAngle` | `int` | ~12 | Phi position in chamber | HIGH |
| Phi bending | `phiBend()` / `m_phiBending` | `int` | ~10 | Local segment direction | **HIGH** |
| Quality | `quality()` | `int` | — | Segment quality | MEDIUM |
| Index | `index()` | `int` | — | Segment ID in chamber | LOW |
| t0 | `t0()` / `m_t0` | `int` | — | Segment timing (ns scale) | **HIGH** — displaced timing |
| chi2 | `chi2()` / `m_chi2` | `int` | — | Segment fit chi2 | MEDIUM — noise rejection |
| RPC flag | `rpcFlag()` | `int` | — | RPC confirmation | LOW |

**The `t0` field is critical for displaced studies.** A displaced muon will generally arrive later than a prompt one; out-of-time arrival shifts `t0`. Combined with `chi2`, this is a powerful displaced discriminator.

---

### 4.4 `L1Phase2MuDTThDigi` — Phase-2 DT Theta Primitive
**File:** `DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h`

| Variable | Method | C++ type | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| BX | `bxNum()` | `int` | Timing | MEDIUM |
| Wheel/Sector/Station | `whNum()`, `scNum()`, `stNum()` | `int` | Detector identity | MEDIUM |
| z position | `z()` / `m_zGlobal` | `int` | Global z position of the segment | **HIGH** — z consistency |
| k slope | `k()` / `m_kSlope` | `int` | Local z direction / slope (theta direction) | **HIGH** — theta pointing residual |
| Quality | `quality()` | `int` | Segment quality | MEDIUM |
| t0 | `t0()` | `int` | Segment timing | HIGH |
| chi2 | `chi2()` | `int` | Fit quality | MEDIUM |

**Answer to DT K-variable question:** `k()` / `m_kSlope` is the **theta slope** (dz/dl or equivalent) of the DT track segment in the theta superlayer. Combined with `z()`, it gives local theta direction — directly useful for computing whether the segment points back to the beamline longitudinally.

---

### 4.5 `L1Phase2MuDTExtPhDigi` — Phase-2 Extended DT Phi Primitive (inherits from `L1Phase2MuDTPhDigi`)
**File:** `DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h`

| Variable | Method | Physical meaning | Displaced usefulness |
|---|---|---|---|
| All from Phase2MuDTPhDigi | inherited | — | see above |
| Local x position | `xLocal()` / `m_xLocal` | x in cm in chamber-local coordinates | **HIGH** — local impact parameter proxy |
| tan(psi) | `tanPsi()` / `m_tanPsi` | tan of local segment angle = dx/dz | **HIGH** — local direction; directly gives pointing residual |
| CMSSW phi | `phiCMSSW()` / `m_phiCMSSW` | phi in standard CMSSW global frame | HIGH |
| CMSSW phiBend | `phiBendCMSSW()` / `m_phiBendCMSSW` | phiBend in CMSSW global frame | HIGH |
| Wire IDs [8] | `pathWireId(i)` / `m_pathWireId[8]` | Wire IDs used in segment fit | MEDIUM — occupancy info |
| TDC values [8] | `pathTDC(i)` / `m_pathTDC[8]` | TDC times per wire | **HIGH** — per-wire timing for displaced |
| Laterality [8] | `pathLat(i)` / `m_pathLat[8]` | Left/right laterality per wire | MEDIUM |

**Note:** `tanPsi` = tan(local segment angle in phi plane) is directly the "K-like" quantity. This is the most displaced-useful variable in the DT suite. It gives the local direction of the track in the chamber and can be compared with the expected direction from the global inter-station trajectory.

---

## 5. CSC Variable Inventory

### 5.1 `CSCALCTDigi` — Anode LCT
**File:** `DataFormats/CSCDigi/interface/CSCALCTDigi.h`

| Variable | Method | C++ type | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| Valid | `isValid()` | `bool` | Primitive validity | MEDIUM |
| Quality | `getQuality()` | `uint16_t` | ALCT quality (0–3) | MEDIUM |
| Accelerator | `getAccelerator()` | `uint16_t` | Accelerator muon flag | LOW |
| Collision bit | `getCollisionB()` | `uint16_t` | Collision vs. accelerator | MEDIUM |
| Key wire group | `getKeyWG()` | `uint16_t` | Wire group = eta / z position | **HIGH** — z/eta encoding |
| BX | `getBX()` | `uint16_t` | Bunch crossing | MEDIUM |
| Track number | `getTrknmb()` | `uint16_t` | Sorting rank | LOW |
| Full BX | `getFullBX()` | `uint16_t` | Full BX from 12-bit counter | MEDIUM — detailed timing |
| HMT | `getHMT()` | `uint16_t` | High-multiplicity trigger bits | LOW |
| Hit container | `getHits()` | `WireContainer` | Per-BX/wire hit pattern | MEDIUM |

**Wire group** encodes the z position of the muon in CSC. Combined with strip from CLCT, it fully determines the 2D LCT position.

---

### 5.2 `CSCCLCTDigi` — Cathode LCT
**File:** `DataFormats/CSCDigi/interface/CSCCLCTDigi.h`

| Variable | Method | C++ type | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| Valid | `isValid()` | `bool` | Primitive validity | MEDIUM |
| Quality | `getQuality()` | `uint16_t` | CLCT quality (0–15 Run-2; 0–3 Run-3) | MEDIUM |
| Pattern (Run-2) | `getPattern()` | `uint16_t` | LCT pattern ID (1–10) encodes local angle class | **HIGH** |
| Pattern (Run-3) | `getRun3Pattern()` | `uint16_t` | Run-3 CSC pattern (0–4) | **HIGH** |
| Slope (Run-3) | `getSlope()` / `getFractionalSlope()` | `uint16_t` / `float` | Local slope in half-strips/layer; negative=left, positive=right | **HIGH** — local direction |
| Strip type | `getStripType()` | `uint16_t` | Half-strip (1) vs full-strip (0) | LOW |
| Bend | `getBend()` | `uint16_t` | Left (0) or right (1) bending direction | **HIGH** — bend direction |
| Strip | `getStrip()` | `uint16_t` | Key half-strip (0–159) | HIGH — phi position |
| Quarter-strip bit | `getQuartStripBit()` | `bool` | Run-3: sub-half-strip resolution | MEDIUM |
| Eighth-strip bit | `getEighthStripBit()` | `bool` | Run-3: 1/8-strip resolution | MEDIUM |
| CFEB | `getCFEB()` | `uint16_t` | CFEB board ID | LOW |
| BX | `getBX()` | `uint16_t` | Bunch crossing | MEDIUM |
| Hit container | `getHits()` | `ComparatorContainer` | Per-BX/strip comparator hits | MEDIUM |

**The Run-3 `getSlope()` / `getFractionalSlope()` is the key CSC local direction proxy.** It gives slope in units of half-strips/layer, which encodes the local phi slope of the muon track through the CSC cathode planes. This can be used to construct a pointing residual.

---

### 5.3 `CSCCorrelatedLCTDigi` — Correlated LCT (ALCT + CLCT matched)
**File:** `DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h`

| Variable | Method | Physical meaning | Displaced usefulness |
|---|---|---|---|
| Valid | `isValid()` | Hit validity | MEDIUM |
| Quality | `getQuality()` | Combined quality | MEDIUM |
| Key wire group | `getKeyWG()` | z/eta wire group position | **HIGH** |
| Strip | `getStrip(n)` / `getFractionalStrip(n)` | Phi strip position (n=2 for half-strip) | **HIGH** |
| Pattern (Run-2) | `getPattern()` | Pattern ID (local angle class) | **HIGH** |
| Pattern (Run-3) | `getRun3Pattern()` | New CSC pattern | **HIGH** |
| Slope (Run-3) | `getSlope()` / `getFractionalSlope()` | Local slope in half-strips/layer | **HIGH** — pointing residual |
| Bend | `getBend()` | Left/right local bend direction | **HIGH** |
| BX | `getBX()` | Timing | MEDIUM |
| MPC link | `getMPCLink()` | MPC sorting rank (0=unsorted, 1–3) | LOW |
| CSC ID | `getCSCID()` | Chamber ID within sector | MEDIUM |
| HMT | `getHMT()` | High-multiplicity bits | LOW |
| Quarter/eighth strip bits | `getQuartStripBit()`, `getEighthStripBit()` | Sub-half-strip precision | MEDIUM |
| Type (simulation) | (enum) | ALCTCLCT, ALCT2GEM, CLCT2GEM… | DO_NOT_USE (sim label) |

**CSC local direction answer:** The `getSlope()` in Run-3 (units: half-strips/layer) is the direct local direction proxy. Combined with `getKeyWG()` (z position) and `getFractionalStrip()` (phi position), the full local track direction is available.

**2026 update (CMSSW_17_0_0_pre3, `CSCCorrelatedLCTDigi.h`):** The slope encoding was extended. New members/methods not present in the 2024 checkout:

| Variable | Method | Line (17_0_0_pre3) | Meaning | Displaced usefulness |
|---|---|---|---|---|
| Extended slope | `getSlopeEx(resolution=0)` | 119 | Local slope with **6-bit** resolution in `Run3HR` (4-bit in Legacy/Run3); `resolution=0` returns full raw resolution | **HIGH** — finer local-direction proxy → sharper pointing residual |
| Raw slope resolution | `getRawSlopeResolution()` | 120 | Returns 6 for `Run3HR`, else 4 | MEDIUM — needed to interpret slope units |
| Version | `getVersion()` / `enum Version{Legacy,Run3,Run3HR}` | 213 / 21 | LCT format version | LOW — context |
| GEM layer used for slope | `getGemLayerUsedForSlopeComputation()` | 220 | Which GEM layer (GE1/1, GE2/1) was used to compute the CSC slope; GEM info now embedded in the LCT word | MEDIUM — GEM-CSC bending adds a second local-direction lever arm |

`getSlope()` (line 117) is retained but is now a non-inline method returning the **4-bit backward-compatible** value. **For displaced studies on a 2026 release, prefer `getSlopeEx()`** to retain the extra resolution bits. `getGemLayerUsedForSlopeComputation()` reflects that GEM-CSC combined slope measurement is now part of the CSC LCT — a genuinely new local-direction handle for the endcap-overlap transition.

---

## 6. RPC Variable Inventory

### 6.1 `RPCDigi`
**File:** `DataFormats/RPCDigi/interface/RPCDigi.h`

| Variable | Method | C++ type | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| Strip | `strip()` | `uint16_t` | Strip number = coarse phi position in roll | MEDIUM — spatial confirmation |
| BX | `bx()` | `int32_t` | Bunch crossing | **HIGH** — timing confirmation |
| Time | `time()` | `double` | Hit time (ns), only if `hasTime()` | HIGH — detailed timing |
| coordinateX | `coordinateX()` | `double` | Local x coordinate (pseudo-digi) | LOW — offline only |
| coordinateY | `coordinateY()` | `double` | Local y coordinate (pseudo-digi) | LOW — offline only |
| deltaTime | `deltaTime()` | `double` | Timing uncertainty | LOW — offline only |
| isPseudoDigi | `isPseudoDigi()` | `bool` | True if geometry-derived | DO_NOT_USE (offline) |

**RPC answers:**
1. **BX** is the primary useful variable — can confirm in-time vs out-of-time hits. Displaced muons may produce late RPC hits.
2. No local direction or slope from RPC. Only strip (phi position) and BX.
3. RPC enters OMTF as a position-only hit on a specific logic layer.
4. `time()` and `coordinateX/Y` are simulation/offline quantities — **do not use as firmware inputs**.

**RPC for displaced:** MEDIUM usefulness primarily through BX timing. RPC strip alone adds spatial position confirmation but no directionality.

---

## 7. GMT/TPS/KMTF/OMTF Stub Inventory

### 7.1 `l1t::MuonStub` (Phase-2 TPS stub)
**File:** `DataFormats/L1TMuonPhase2/interface/MuonStub.h`

| Field | Type | Units / Scale | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| `etaRegion_` | `int` | barrel: wheel (−2..+2); endcap: 6−ring | Eta region / wheel / ring | MEDIUM |
| `phiRegion_` | `int` | barrel: sector (0..11); endcap: chamber | Phi region / sector / chamber | MEDIUM |
| `depthRegion_` | `int` | station (1..4) | Radial depth / station | MEDIUM |
| `tfLayer_` | `uint` | TF layer index | Logic layer in track finder | MEDIUM |
| `coord1_` | `int` | 30°/2048 per unit (~0.0146°/unit) | Global phi position | **HIGH** |
| `coord2_` | `int` | bending angle units (barrel only) | Local bending angle proxy | **HIGH** — displaced! |
| `id_` | `int` | — | Stub ID per chamber | LOW |
| `quality_` | `int` | 0..7 | Stub quality | MEDIUM |
| `bxNum_` | `int` | BX units | Timing | MEDIUM |
| `eta1_` | `int` | 3.0/512 per unit (~0.00586/unit) | Eta coordinate (primary) | HIGH |
| `eta2_` | `int` | 3.0/512 per unit | Eta coordinate (secondary or SL2) | HIGH |
| `etaQuality_` | `int` | — | Quality of eta measurement | MEDIUM |
| `type_` | `int` | 0=DT/TwinMux, 1=RPC barrel, 2=CSC, 3=RPC endcap | Subdetector type | MEDIUM |
| `offline_coord1_` | `double` | offline | Offline phi for analysis only | DO_NOT_USE (offline) |
| `offline_coord2_` | `double` | offline | Offline coord2 for analysis | DO_NOT_USE (offline) |
| `offline_eta1_` | `double` | offline | Offline eta | DO_NOT_USE (offline) |
| `offline_eta2_` | `double` | offline | Offline eta | DO_NOT_USE (offline) |

**coord1 unit:** 1 unit = 30°/2048 ≈ 0.01465° ≈ 2.558×10⁻⁴ rad. Full 360° = 24576 units.  
**eta1/eta2 unit:** 1 unit = 3.0/512 ≈ 0.00586. Range ≈ ±1.5 → ±256 counts.

**coord2 interpretation:** In the barrel (DT), `coord2` encodes the bending angle (equivalent to `phiB`). In the endcap (CSC), it may encode slope/bend information. For RPC, `coord2` is typically zero or undefined. **This is the key displaced-discrimination field in the Phase-2 TPS stub format.** (Confirmed by the member comment at `MuonStub.h:149` — `coord2_ // bending angle only in barrel for now` — identical in 2024 and 2026.)

**2026 update (CMSSW_17_0_0_pre3):** `MuonStub.h` gained a firmware serialization method `hybridStubWord()` (line 144, returns a packed `Phase2L1GMT::wordtype`) plus `printHybridStub()`/`printHybridStubWord()` helpers. No new physics variable is added, but this confirms the exact bit-packing of the stub is now exposed in the data format — useful if the ML cache is built directly from the hardware word.

**Type-field caveat (unchanged 2024→2026):** `type_` is documented as `0=TwinMux/DT, 1=RPC barrel, 2=CSC, 3=RPC endcap` (`MuonStub.h:163`), but the helper `isBarrel()` returns `type_==1` and `isEndcap()` returns `type_==0` (lines 116–117), which is inconsistent with that comment. **Do not rely on `isBarrel()`/`isEndcap()`; use the raw `type()` value.** Flagged as expert question 12b.

---

### 7.2 `MuonStub` (Phase-1 OMTF internal)
**File:** `L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h`

| Field | Type | Physical meaning | Displaced usefulness |
|---|---|---|---|
| `phiHw` | `int` | Hardware phi angle (processor-local) | HIGH |
| `phiBHw` | `int` | Hardware phi bending angle | **HIGH** |
| `etaHw` | `int` | Hardware eta coordinate | HIGH |
| `r` | `int` | Distance from beam pipe [cm] | **HIGH** — radial position |
| `qualityHw` | `int` | Hardware quality | MEDIUM |
| `bx` | `int` | BX timing | MEDIUM |
| `timing` | `int` | Detailed timing | HIGH |
| `logicLayer` | `uint` | Logic layer index | MEDIUM |
| `input` | `uint` | Input number | LOW |
| `detId` | `int` | Detector ID | MEDIUM |
| `cscOffset` | `int` | CSC angle conversion offset | LOW |
| `cscScale` | `double` | CSC angle conversion scale | LOW |
| `cscOrder` | `int` | CSC angle conversion order | LOW |
| `type` | `enum` | DT_PHI, DT_THETA, DT_PHI_ETA, RPC, CSC_PHI, CSC_ETA, CSC_PHI_ETA, BARREL_SUPER_SEG | Stub type | MEDIUM |

**Note:** The `r` field (distance from beam pipe in cm) is particularly valuable — it makes this format more useful than the Phase-2 TPS stub for computing inter-station curvature and beamline-origin compatibility.

---

### 7.3 `L1MuKBMTCombinedStub` (Kalman BMTF stub)
**File:** `DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h`

| Field | Method | Physical meaning | Displaced usefulness |
|---|---|---|---|
| Wheel | `whNum()` | DT wheel | MEDIUM |
| Sector | `scNum()` | DT sector | MEDIUM |
| Station | `stNum()` | DT station | MEDIUM |
| phi | `phi()` | Phi position | HIGH |
| phiB | `phiB()` | Phi bending angle | **HIGH** |
| Quality | `quality()` | Stub quality | MEDIUM |
| Tag | `tag()` | Second TS tag | LOW |
| BX | `bxNum()` | Timing | MEDIUM |
| eta1, eta2 | `eta1()`, `eta2()` | Eta from theta stubs | HIGH |
| qeta1, qeta2 | `qeta1()`, `qeta2()` | Eta quality | MEDIUM |

**Note:** This is the richest DT-only stub format combining phi and theta into one object. Both `phi` (position) and `phiB` (bending) are preserved.

---

### 7.4 `RegionalMuonCand` (Final trigger output)
**File:** `DataFormats/L1TMuon/interface/RegionalMuonCand.h`

| Field | Method | Units | Physical meaning | Displaced usefulness |
|---|---|---|---|---|
| `m_hwPt` | `hwPt()` | LSB = 0.5 GeV/c | Transverse momentum | MEDIUM — already reconstructed |
| `m_hwPtUnconstrained` | `hwPtUnconstrained()` | varies | pT without vertex constraint | **HIGH** — displaced proxy |
| `m_hwDXY` | `hwDXY()` | 2-bit dXY impact parameter | Impact parameter (Phase-2) | **HIGH** — direct displaced variable |
| `m_hwPhi` | `hwPhi()` | integer units | Global phi | HIGH |
| `m_hwEta` | `hwEta()` | integer units | Global eta | HIGH |
| `m_hwSign` | `hwSign()` | 0/1 | Charge sign | MEDIUM |
| `m_hwSignValid` | `hwSignValid()` | 0/1 | Sign validity | MEDIUM |
| `m_hwQuality` | `hwQuality()` | 0..15 | Combined quality | MEDIUM |
| `m_hwHF` | `hwHF()` | bool | HF bit | LOW |
| Track address | `trackAddress()` | map | Per-TF internal addressing (BMTF: wheel/sector/station segments; OMTF: layers/weight; EMTF: ME chamber segment) | MEDIUM |

**OMTF-specific track address fields:** `kLayers` (which logic layers contributed), `kZero`, `kWeight` (pattern matching weight). These provide a coarse summary of which subdetectors contributed to the track.

**Note:** `hwDXY` (2-bit impact parameter) is Phase-2 OMTF/BMTF/EMTF output — directly encodes transverse displacement. This is the most downstream displaced-relevant variable.  
`hwPtUnconstrained` is the pT estimated without requiring the track to point to the beamline — high values indicate potentially displaced muons.

---

## 8. Coordinate Systems and Conversions

| Product | coord1 / phi | coord1 frame | eta variable | r/z availability | Conversion needed |
|---|---|---|---|---|---|
| `L1MuDTChambPhDigi` | `radialAngle` | Chamber-local phi (LSB = ~1/2048 of 30° sector) | None (use `L1MuDTChambThDigi`) | Station → approximate r | Add phiZero (sector offset) to get global phi |
| `L1MuDTChambThDigi` | None | — | `position[i]` wire pattern | theta wire group → z | Geometry LUT: wire group → eta/z |
| `L1Phase2MuDTPhDigi` | `phi()` | Chamber-local phi | None directly | Station → approximate r | Add phiZero |
| `L1Phase2MuDTThDigi` | None | — | `z()` global z, `k()` slope | `z()` directly available | Minimal — z already converted |
| `L1Phase2MuDTExtPhDigi` | `phiCMSSW()` also available | Chamber-local + CMSSW global | None | `xLocal()` in cm | `phiCMSSW()` is already global |
| `CSCCorrelatedLCTDigi` | `getStrip()` (halfstrip) | Chamber-local strip | `getKeyWG()` → wire group → eta | Station/ring → approximate r,z | CSC geometry lookup (endcap non-trivial) |
| `RPCDigi` | `strip()` | Roll-local strip | Roll/ring → eta | Station → approximate r | RPC geometry lookup |
| `l1t::MuonStub` (Ph2) | `coord1()` | Global phi, 30°/2048 | `eta1()`, unit 3.0/512 | `depthRegion()` + geometry | Minimal — already partially global |
| `MuonStub` (OMTF Ph1) | `phiHw` | Processor-local phi | `etaHw` microGMT scale | `r` field in cm — directly available | Processor phi → global requires phiZero offset |
| `RegionalMuonCand` | `hwPhi()` | Global phi (standard muon scale) | `hwEta()` | None | None needed — already global |

### Key geometry helpers in CMSSW:
- `OmtfAngleConverter` / `OmtfPhase2AngleConverter`: converts DT/CSC/RPC primitives to OMTF processor-local phi/eta
- `getProcessorPhi(phiZero, procType, dtScNum, dtPhi)` — adds the 30° sector offset (`phiZero`) to convert from chamber-local to processor-local phi
- `getGlobalEta(DTChamberId, dtThDigis, bxNum)` — converts theta wire groups to global eta using a LUT

---

## 9. Firmware Availability

| Variable | Source | Firmware status |
|---|---|---|
| DT `phi` (radialAngle) | `L1MuDTChambPhDigi` | ✓ firmware_available_now |
| DT `phiB` (bendingAngle) | `L1MuDTChambPhDigi` | ✓ firmware_available_now |
| DT quality, BX, identity | `L1MuDTChambPhDigi` | ✓ firmware_available_now |
| DT theta wire pattern | `L1MuDTChambThDigi` | ✓ firmware_available_now |
| Phase-2 DT `phi`, `phiBend` | `L1Phase2MuDTPhDigi` | ✓ firmware_available_now (Phase-2) |
| Phase-2 DT `t0`, `chi2` | `L1Phase2MuDTPhDigi` | ✓ firmware_derivable_with_small_logic |
| Phase-2 DT `z`, `k` (slope) | `L1Phase2MuDTThDigi` | ✓ firmware_available_now (Phase-2) |
| Phase-2 Extended `xLocal`, `tanPsi` | `L1Phase2MuDTExtPhDigi` | firmware_derivable_with_small_logic |
| Phase-2 Extended per-wire TDC | `L1Phase2MuDTExtPhDigi.pathTDC` | requires_geometry_lookup / not in standard firmware path |
| CSC `keywire` | `CSCALCTDigi` | ✓ firmware_available_now |
| CSC `strip`, `bend`, `pattern` | `CSCCLCTDigi` | ✓ firmware_available_now |
| CSC Run-3 `slope` | `CSCCLCTDigi` | ✓ firmware_available_now (Run-3+) |
| CSC quarter/eighth-strip bits | `CSCCLCTDigi` | ✓ firmware_available_now (Run-3+) |
| CSC `getBX()` | `CSCCorrelatedLCTDigi` | ✓ firmware_available_now |
| CSC `Type` enum | `CSCCorrelatedLCTDigi` | simulation_only |
| RPC `strip`, `bx` | `RPCDigi` | ✓ firmware_available_now |
| RPC `time()`, `coordinateX/Y` | `RPCDigi` | simulation_only / offline_only |
| OMTF `MuonStub` phi, phiB, eta, quality, bx | Phase-1 OMTF internal | firmware_available_now (in emulator) |
| OMTF `MuonStub.r` | Phase-1 OMTF internal | firmware_derivable_with_small_logic (LUT from layer) |
| Phase-2 `l1t::MuonStub` coord1, coord2, eta1, eta2 | TPS output | ✓ firmware_available_now (Phase-2) |
| `RegionalMuonCand` hwPt, hwEta, hwPhi | Final TF output | ✓ firmware_available_now |
| `RegionalMuonCand` hwDXY | Phase-2 OMTF/BMTF | ✓ firmware_available_now (Phase-2) |
| `RegionalMuonCand` hwPtUnconstrained | Phase-2 | firmware_available_now (Phase-2) |
| Gen-truth pT, dxy, phi | Generator | gen_truth_only — DO NOT USE as input |
| Offline reconstructed muon variables | RecoMuon | offline_only — DO NOT USE as input |

---

## 9.5 Empirical Validation on Production Phase-2 Samples (2026-07-05)

The class-level availability above was cross-checked against **real produced Phase-2 samples** in
`/eos/user/p/pleguina/omtf_hecin_datasets/` (production pipeline
`omtf_hecin_dataset_production`, CMSSW_14_2_0_pre2, era `Phase2C17I13M9`, geometry `Extended2026D110`).

**File types produced per dataset:**
- `omtf_hits_*.root` — `simOmtfPhase2Digis` `DataROOTDumper2` trees (`OMTFHitsTree`, `OMTFAllInputTree`); contain only the OMTF-internal post-`AngleConverter` stub (`phiHw`, `phiBHw`, `etaHw`, `r`, `type`, `bx`, `quality`).
- `omtf_nano_*.root` — NanoAOD flat tables from `L1Trigger/L1MuNano` (`omtfNanoTables_cff.py`), which **do** carry the raw per-detector primitives.

**Products confirmed PRESENT and FILLED** (verified in `prod/C21_disp_pt10to20_overlap_PU200`, `prod/G4_pos`, `prod/C17_prompt…`):

| Nano table | src InputTag | Real branches | Filled | Status |
|---|---|---|---|---|
| `DTPhiDigi` | `simDtTriggerPrimitiveDigis` | phi, phiB, quality, bx, wheel/sector/station, ts2Tag, rpcBit | ✅ | firmware_available_now |
| `Ph2DTPhiDigi` | `dtTriggerPhase2PrimitiveDigis` | phi, **phiBend**, **t0**, **chi2**, sl, quality, bx | ✅ | firmware_available_now (Phase-2) |
| `Ph2DTThDigi` | `dtTriggerPhase2PrimitiveDigis` | **z**, **k**, t0, chi2, quality, bx | ✅ | firmware_available_now (Phase-2) |
| `CSCLctDigi` | `simCscTriggerPrimitiveDigis:MPCSORTED` | **slope**, pattern, **run3Pattern**, bend, keywire, strip, endcap/ring/chamber, valid | ✅ (5957 LCTs/file) | firmware_available_now |
| `RPCDigi` | `simMuonRPCDigis` | strip, bx, roll, region/ring/station/sector/subsector/layer | ✅ | firmware_available_now |
| `MuonStubTps` / `MuonStubKmtf` | `l1tStubsGmt:tps` / `:kmtf` | coord1, coord2, eta1/2, type, quality, bx, offlineCoord1/2 | ✅ | firmware_available_now (Phase-2) |
| `omtf` | `simOmtfPhase2Digis:OMTF` | hwPt, **hwPtUnc**, hwDXY, hwEta, hwPhi, hwQual | ✅ (see caveats) | firmware_available_now |
| `GenMuon` | `genParticles` | pt, eta, phi, vx/vy/vz, dXY, lXY, etaSt1/2, phiSt1/2 | ✅ | gen_truth_only (labels) |

> **Note:** `omtf_nano` files produced **before ~2026-05-27** (e.g. `das_validation/displaced_lowpt`, May 8) predate the raw-primitive tables and contain only `GenMuon` + stub + `omtf`. All C-campaign and G-campaign production nano files (late May–June 2026) contain the full digi tables.

**Resolved uncertainties (measured on displaced C21 PU200):**

| Question | Answer | Evidence |
|---|---|---|
| Is DT `t0` present? | **YES, filled** | `Ph2DTPhiDigi_t0` ∈ [455,803], `Ph2DTThDigi_t0` ∈ [362,1058]. Large ~600 offset → TDC/BX-relative time; subtract a reference before use, do not use raw. |
| Is DT `tanPsi` available? | **NO** | Only base `Ph2DTPhiDigi` is dumped; the extended `L1Phase2MuDTExtPhDigi` (xLocal/tanPsi/per-wire TDC) is **not produced/dumped**. Requires enabling DT extended output + new table. |
| Is CSC `slope` available? | **YES** | `CSCLctDigi_slope` ∈ [0,16] + `bend` + `pattern`/`run3Pattern`, all filled. This is the **integer** slope (`getSlope`), not the signed fractional float — combine `slope`+`bend` for signed local direction. |
| Is TPS `coord2` filled for CSC? | **NO** | `coord2` nonzero only for DT (~42%) and RPC-barrel (~35%). **Type-2 (CSC) and Type-3 (RPC-endcap) GMT stubs = 0** in overlap samples despite 5957 CSC LCTs present. CSC information exists **only** at the raw `CSCLctDigi` level. |
| Can z/r be obtained firmware-realistically? | **YES** | `Ph2DTThDigi_z`/`k` filled; OMTF `hits_r` in `OMTFHitsTree`. No offline geometry needed. |

**Caveats found in real data (must handle before schema freeze):**

1. **`omtf_hwDXY` is uniformly 0** in every production file checked — displaced (C21, G4) *and* prompt (C17). The Phase-2 OMTF emulator in this production does **not** populate the 2-bit `hwDXY`. → **DO_NOT_USE `hwDXY`** as a displaced feature/label in these samples. Use gen `dXY`/`lXY` for labels and `hwPtUnc` as the trigger-level displaced proxy (filled: disp mean ≈ 11.5 vs prompt ≈ 12.8 LSB).
2. **`Ph2DTPhiDigi_chi2` contains ~2–4 % negative values** (range ±34340, symmetric) — a fixed-point **overflow wrap** in the DT phi-segment χ² packing. `Ph2DTThDigi_chi2` is clean ([0,9996]). → Clamp/guard `Ph2DTPhiDigi_chi2` negatives (treat as overflow → high-χ²) before use.

**Implication for strategy:** Because CSC never reaches the `l1t::MuonStub` GMT collection (Section 7.1) in these samples, a **TPS-only (Strategy A) model would be blind to CSC** in the overlap region. This empirically **reinforces Strategy B/C**: the enriched cache must read `CSCLctDigi` directly for endcap-side local direction. See Sections 12–13.

---

## 10. Displaced-Muon Usefulness Assessment

### 10.1 Variables ranked by expected displaced discrimination power

| Rank | Variable | Source | Use case | Usefulness | Reason |
|---:|---|---|---|---|---|
| 1 | `phiB` / `bendingAngle` | `L1MuDTChambPhDigi`, `L1Phase2MuDTPhDigi` | Local phi bend; pointing residual | **HIGH** | Bending angle directly indicates whether track points to beamline |
| 2 | `k()` / `m_kSlope` | `L1Phase2MuDTThDigi` | Local theta slope; z pointing | **HIGH** | Theta direction residual = key displaced discriminator longitudinally |
| 3 | CSC `getSlope()` / `getFractionalSlope()` | `CSCCLCTDigi`, `CSCCorrelatedLCTDigi` | Local phi slope in CSC | **HIGH** | Directly encodes local direction; compare with inter-station line |
| 4 | `t0()` in Phase-2 DT | `L1Phase2MuDTPhDigi`, `L1Phase2MuDTThDigi` | Displaced muon timing | **HIGH** | Displaced muons arrive late; t0 offset is a direct indicator |
| 5 | `coord2` in Phase-2 TPS stub | `l1t::MuonStub` | Bending proxy at stub level | **HIGH** | Preserved bending angle in stub format |
| 6 | `RegionalMuonCand.hwDXY` | Phase-2 regional output | Impact parameter | **HIGH** | 2-bit dXY directly encodes transverse displacement |
| 7 | `RegionalMuonCand.hwPtUnconstrained` | Phase-2 regional output | Unconstrained pT | **HIGH** | Large hwPtUnconstrained with small hwPt → displaced |
| 8 | `L1Phase2MuDTExtPhDigi.tanPsi()` | Extended Phase-2 DT | Local tan(angle) | **HIGH** | Direct local direction → beamline pointing residual |
| 9 | `L1Phase2MuDTExtPhDigi.xLocal()` | Extended Phase-2 DT | Local x position [cm] | **HIGH** | Combined with direction gives impact parameter estimate |
| 10 | CSC `pattern` / `getRun3Pattern()` | `CSCCLCTDigi` | Pattern class → local angle | **HIGH** | Pattern encodes angle class; changes for displaced muons |
| 11 | DT theta `z()` | `L1Phase2MuDTThDigi` | Global z position | HIGH | z consistency across stations |
| 12 | `bx` (all subdetectors) | DT/CSC/RPC | Timing consistency | MEDIUM/HIGH | Out-of-time displaced muons |
| 13 | CSC `bend` direction | `CSCCLCTDigi` | Left/right bend | HIGH | Sign of local bend |
| 14 | `keywire` / `getKeyWG()` | `CSCALCTDigi`, `CSCCorrelatedLCTDigi` | z/eta in CSC | HIGH | z position for pointing |
| 15 | Station / layer identity | all | Radial ordering | MEDIUM | Required for graph construction |
| 16 | Quality codes | all | Noise rejection | MEDIUM | Reject fakes |
| 17 | RPC strip | `RPCDigi` | Spatial confirmation | MEDIUM | Extra hit |
| 18 | RPC BX | `RPCDigi` | Timing confirmation | MEDIUM/HIGH | Out-of-time |

---

## 11. Derived Feature Proposal

### 11.1 Pairwise geometry features (primitive pair i, j)

```
delta_phi_ij = coord1_i - coord1_j          [units: 30°/2048]
delta_eta_ij = eta1_i - eta1_j              [units: 3.0/512]
delta_r_ij   = r_i - r_j                    [cm, from MuonStub.r or station LUT]
delta_station_ij = station_i - station_j    [integer]
detector_pair_type = (type_i, type_j)       [enum pair]
BX_difference_ij = bx_i - bx_j             [BX units]
```

### 11.2 Curvature proxy

```
curv_ij = delta_phi_ij / max(|delta_r_ij|, epsilon)
```

Units: (30°/2048 per unit) / cm ≈ angular curvature proxy. Compare with expected bending for a prompt muon at the measured momentum.

### 11.3 Beamline phi-intercept proxy

```
phi0_proxy_ij = coord1_i - curv_ij * r_i
```

Consistency of `phi0_proxy_ij` across all pairs is a direct displaced discriminator: prompt muons all extrapolate to the same beamline phi, displaced muons do not.

### 11.4 Phi0 consistency across pairs

```
phi0_proxy_mean, phi0_proxy_std, phi0_proxy_range
```

Large `phi0_proxy_std` → inconsistent extrapolation to beamline → displaced candidate.

### 11.5 Local direction residual (where available)

**DT barrel:**
```
expected_phiB_from_global = f(phi, station, r)   [geometry + curvature from pairs]
measured_phiB = phiB()
pointing_residual_phi = measured_phiB - expected_phiB_from_global
```

**DT theta (Phase-2):**
```
expected_k_from_global = f(z, station)
measured_k = k()
pointing_residual_theta = measured_k - expected_k_from_global
```

**CSC:**
```
expected_slope_from_global = f(strip_pair, station, ring)
measured_slope = getFractionalSlope()
pointing_residual_csc = measured_slope - expected_slope_from_global
```

### 11.6 Local-vs-global bend consistency

```
sign_product_i = sign(phiB_i) * sign(delta_phi_ij)  for all pairs j
```

For prompt muons: `sign_product` should be consistent across all pairs (bend direction matches global bending). For displaced: inconsistencies appear.

### 11.7 timing features

```
bx_mean = mean(bx_i for all stubs)
bx_std = std(bx_i)
t0_mean = mean(t0_i)   [Phase-2 DT only]
t0_std = std(t0_i)
```

---

## 12. Common Node Schema Proposal

Based on the audit, the following unified per-primitive node schema is proposed for the displaced-muon ML model. It is realistic given available firmware fields.

```json
{
  "global_phi":            {"type": "int",   "units": "30°/2048",    "source": "coord1 in l1t::MuonStub; phiHw+phiZero in OMTF MuonStub"},
  "global_eta":            {"type": "int",   "units": "3.0/512",     "source": "eta1 in l1t::MuonStub; etaHw in OMTF MuonStub"},
  "r":                     {"type": "int",   "units": "cm",          "source": "MuonStub.r (OMTF Phase-1); station LUT (Phase-2)"},
  "z":                     {"type": "int",   "units": "cm or units", "source": "L1Phase2MuDTThDigi.z(); CSC keywire→z via LUT"},
  "detector_type":         {"type": "int",   "units": "enum",        "source": "l1t::MuonStub.type_ (0=DT,1=RPC barrel,2=CSC,3=RPC endcap)"},
  "station":               {"type": "int",   "units": "1..4",        "source": "depthRegion_ in l1t::MuonStub"},
  "eta_region":            {"type": "int",   "units": "wheel or ring", "source": "etaRegion_ in l1t::MuonStub"},
  "phi_region":            {"type": "int",   "units": "sector or chamber", "source": "phiRegion_ in l1t::MuonStub"},
  "tf_layer":              {"type": "uint",  "units": "logic layer", "source": "tfLayer_ in l1t::MuonStub"},
  "quality":               {"type": "int",   "units": "0..7",        "source": "quality_ in l1t::MuonStub"},
  "bx":                    {"type": "int",   "units": "BX",          "source": "bxNum_ in l1t::MuonStub"},
  "local_bend":            {"type": "int",   "units": "phiB units",  "source": "coord2_ (DT barrel); phiB() in L1MuDTChambPhDigi; 0 for RPC"},
  "local_phi_slope":       {"type": "float", "units": "hs/layer",    "source": "getFractionalSlope() from CSCCLCTDigi/CSCCorrelatedLCTDigi; tanPsi from L1Phase2MuDTExtPhDigi", "optional": true},
  "local_theta_slope":     {"type": "int",   "units": "k units",     "source": "k() from L1Phase2MuDTThDigi", "optional": true},
  "z_position":            {"type": "int",   "units": "cm or units", "source": "z() from L1Phase2MuDTThDigi; keywire→z for CSC", "optional": true},
  "x_local":               {"type": "int",   "units": "cm",          "source": "xLocal() from L1Phase2MuDTExtPhDigi", "optional": true},
  "t0":                    {"type": "int",   "units": "ns-like",     "source": "t0() from L1Phase2MuDTPhDigi/ThDigi", "optional": true},
  "chi2":                  {"type": "int",   "units": "chi2 units",  "source": "chi2() from Phase-2 DT", "optional": true},
  "pattern_code":          {"type": "int",   "units": "0..10",       "source": "getRun3Pattern() / getPattern() from CSCCLCTDigi", "optional": true},
  "wire_or_roll":          {"type": "int",   "units": "wire group or roll", "source": "getKeyWG() for CSC; RPCDetId roll", "optional": true},
  "strip":                 {"type": "int",   "units": "halfstrip",   "source": "getStrip() from CSCCLCTDigi", "optional": true},
  "has_local_bend":        {"type": "bool",  "notes": "true for DT; true for CSC via slope; false for RPC"},
  "has_local_phi_slope":   {"type": "bool",  "notes": "true for Run-3+ CSC, Phase-2 DT extended"},
  "has_local_theta_slope": {"type": "bool",  "notes": "true for Phase-2 DT theta; false for Run-3 DT theta"},
  "has_timing":            {"type": "bool",  "notes": "true if t0 available (Phase-2 DT); RPC BX always available"},
  "valid":                 {"type": "bool",  "source": "isValid() on all primitives"}
}
```

### Schema filling table by subdetector

| Schema field | DT Run-3 source | DT Phase-2 source | CSC source | RPC source | Notes |
|---|---|---|---|---|---|
| `global_phi` | `phi()` + phiZero | `phi()` + phiZero | `getStrip()` → geometry | `strip()` → geometry | All need sector offset |
| `global_eta` | From theta `position[]` → LUT | `z()` → LUT | `getKeyWG()` → LUT | Roll → LUT | Geometry LUT needed |
| `r` | Station LUT (≈430,490,595,700 cm) | Station LUT | Station/ring LUT | Station LUT | Approximate |
| `z` | From theta wire pattern | `z()` directly (Phase-2 Th) | `getKeyWG()` → LUT | Roll → LUT | |
| `local_bend` | `phiB()` | `phiBend()` | `getSlope()` or `getBend()` | N/A (zero) | |
| `local_phi_slope` | N/A (Run-3) | `tanPsi()` (extended) | `getFractionalSlope()` | N/A | |
| `local_theta_slope` | N/A | `k()` (Phase-2 theta) | N/A | N/A | |
| `t0` | N/A (Run-3) | `t0()` (Phase-2) | N/A | timing if available | |
| `pattern_code` | N/A | N/A | `getRun3Pattern()` | N/A | |
| `bx` | `bxNum()` | `bxNum()` | `getBX()` | `bx()` | |
| `quality` | `code()` | `quality()` | `getQuality()` | N/A | |
| `detector_type` | 0 | 0 | 2 | 1 (barrel), 3 (endcap) | From `l1t::MuonStub.type_` |

---

## 13. Model-Design Implications

### Assessment

Given the audit findings:

1. **Phase-2 TPS `l1t::MuonStub`** preserves `coord1` (global phi), `coord2` (bending angle for DT), `eta1/eta2`, `quality`, `bx`, and identity. This is a good baseline but loses CSC-specific `slope`, DT `t0`/`chi2`, and per-wire information.

2. **Raw DT primitives** (especially Phase-2) contain `phiB`, `t0`, `chi2`, `k`, `z`, `tanPsi`, `xLocal` — information highly relevant to displaced-muon identification that is **not preserved** in the TPS stub or `RegionalMuonCand`.

3. **CSC LCT** contains `slope` / `pattern` and `bend` which partially survive through EMTF but are largely lost before `RegionalMuonCand`.

4. **Information loss is significant** when going from raw primitives to TPS stubs: roughly 30–60% of the most displaced-useful variables are lost.

### Recommended strategy: **Strategy B** — enriched common-coordinate primitive cache

**Justification:**
- TPS stubs preserve `coord2` (bending) but lose `t0`, `chi2`, `tanPsi`, CSC slope
- The Phase-2 `l1t::MuonStub` with `coord2` is already a good starting node representation
- Augmenting it with `t0` (DT Phase-2), `chi2` (DT Phase-2), CSC `slope` and `pattern` gives a near-complete set of displaced discriminants
- This avoids the full complexity of Strategy D (raw sparse graph) while recovering the most critical information
- The OMTF Phase-1 `MuonStub` (with `r` field) is particularly useful for the overlap region

**Practical path:**
1. Use `l1t::MuonStub` (Phase-2) as base node format
2. Store extra fields per detector type (t0/chi2 for DT, slope/pattern for CSC)
3. Compute pairwise derived features at inference time (curvature, pointing residuals)
4. For Phase-1/Run-3 compatibility: use OMTF internal `MuonStub` with `phiHw`, `phiBHw`, `r`

---

## 14. Top Recommended Variables

### 14.1 Top 10 variables to extract first

| Rank | Variable | Source class | Method | Usefulness | Reason |
|---:|---|---|---|---|---|
| 1 | DT phi bending angle | `L1MuDTChambPhDigi` | `phiB()` | **HIGH** | Local direction proxy; pointing residual; preserved in BMTF/OMTF stubs |
| 2 | CSC Run-3 slope | `CSCCLCTDigi` / `CSCCorrelatedLCTDigi` | `getFractionalSlope()` | **HIGH** | Local phi direction in CSC; directly usable for pointing residual |
| 3 | Phase-2 DT theta slope | `L1Phase2MuDTThDigi` | `k()` | **HIGH** | Longitudinal direction; theta pointing residual |
| 4 | Phase-2 DT t0 timing | `L1Phase2MuDTPhDigi` | `t0()` | **HIGH** | Out-of-time displaced muon signature |
| 5 | TPS stub coord2 | `l1t::MuonStub` (Phase-2) | `coord2()` | **HIGH** | Bending angle preserved to TPS level |
| 6 | RegionalMuonCand hwDXY | `RegionalMuonCand` | `hwDXY()` | **HIGH** | 2-bit impact parameter — direct displaced flag |
| 7 | RegionalMuonCand hwPtUnconstrained | `RegionalMuonCand` | `hwPtUnconstrained()` | **HIGH** | pT without beamline constraint; large → displaced |
| 8 | Phase-2 DT z position | `L1Phase2MuDTThDigi` | `z()` | HIGH | Longitudinal position for pointing |
| 9 | CSC pattern code | `CSCCLCTDigi` | `getRun3Pattern()` | HIGH | Pattern class encodes local angle |
| 10 | BX (all stubs) | all | `bxNum()` / `getBX()` | MEDIUM/HIGH | Out-of-time hits → displaced timing |

### 14.2 Variables to avoid

| Variable | Source | Reason to avoid |
|---|---|---|
| `gen_pt`, `gen_dxy`, `gen_phi` | Generator | gen_truth_only — labels, not inputs |
| `offline_coord1_`, `offline_coord2_` | `l1t::MuonStub` Phase-2 private | offline_only — explicitly labeled as such in header |
| `offline_eta1_`, `offline_eta2_` | `l1t::MuonStub` Phase-2 private | offline_only |
| `RPCDigi.coordinateX/Y` | `RPCDigi` | simulation_only — `isPseudoDigi()` flags |
| `RPCDigi.time()` | `RPCDigi` | offline / simulation-only in most contexts |
| `CSCCorrelatedLCTDigi.Type` enum | `CSCCorrelatedLCTDigi` | simulation_only — ALCTCLCT, CLCT2GEM types only in sim |
| Offline RecoMuon variables | RecoMuon | offline_only — not in L1 path |
| `CSCCLCTDigi.getHits()` | `CSCCLCTDigi` | Per-comparator hit pattern — simulation detail, not in readout path |
| `CSCALCTDigi.getHits()` | `CSCALCTDigi` | Per-wire hit pattern — simulation detail |

---

## 15. Variables Not Recommended

In addition to the list above:

- **`RegionalMuonCand.trackAddress` for OMTF**: `kLayers`, `kZero`, `kWeight` are coarse summaries that are only 3 integers. Insufficient for reconstructing original information. Use the primitive-level stubs instead.
- **`L1MuDTChambThDigi.position[]` wire patterns** in Run-3: The 7-element wire-group pattern encodes eta but requires a geometry LUT to interpret. Recommend using Phase-2 `z()` instead when available.
- **CSC `getStripType()`**: obsolete since mid-2008.
- **CSC `getMPCLink()`**: MPC sorting rank — not discriminating for displaced vs. prompt.
- **`L1MuBMTrackSegPhi` / `L1MuBMTrackSegEta`**: BMTF internal track segments — same information as `L1MuKBMTCombinedStub` but separated. Prefer the combined stub.

---

## 16. Open Questions for Experts

### 16.1 DT questions
1. **What are the exact fixed-point scales for `phiB`?** The header shows 10-bit signed integer but does not document the angular unit. Is it consistent between BMTF and OMTF? What is the value in rad?
2. **Is `tanPsi` in `L1Phase2MuDTExtPhDigi` available in the firmware data path for OMTF Phase-2?** Or is it emulator-only?
3. **Is `t0` from Phase-2 DT primitives available at OMTF input?** Or is it dropped before feeding OMTF?
4. **What is the unit of `z()` in `L1Phase2MuDTThDigi`?** The comment says global z but the unit (cm? mm? integer scale?) is not in the header.
5. **Is the `OmtfPhase2AngleConverter::dtFixedPointEtaForFirmware_` flag used in the standard Phase-2 MC production?** (affects whether eta is from a LUT or fixed mid-chamber value)

### 16.2 CSC questions
6. **Does `getSlope()` from `CSCCorrelatedLCTDigi` survive into the EMTF/OMTF input stub format?** Or is it only in the raw CLCT digi?
7. **What is the exact angular mapping of Run-3 CSC `slope` (half-strips/layer) to a local phi angle in mrad?** Needs chamber geometry.
8. **Is there a Run-3 CSC analog of the DT pointing residual? Can `pattern` + `bend` + `strip` + `keywire` uniquely determine a local direction in firmware?**
9. **For ME1/1 (overlap region), what is the strip pitch and how does it relate to `getSlope()` units?**
9b. **(2026)** In CMSSW_17_0_0_pre3 the CSC LCT gained `getSlopeEx()` (6-bit `Run3HR`) and `getGemLayerUsedForSlopeComputation()`. Is the `Run3HR` high-resolution slope actually produced in the Phase-2 firmware/MC we will use, or only in a subset of chambers with GEM? Which `getSlopeEx()` resolution should the ML cache store?

### 16.3 RPC questions
10. **Is RPC BX reliable for identifying displaced muons with dxy > 5 cm?** What is the BX resolution?
11. **Are there plans to add RPC cluster width or timing in Phase-2 OMTF input?**

### 16.4 OMTF/TPS questions
12. **What fields from `l1t::MuonStub.coord2_` are actually filled by OMTF Phase-2 for CSC stubs?** The header says "bending angle only in barrel for now" — is this an emulator limitation or firmware limitation?
12b. **(verified inconsistency, 2024 and 2026)** In `l1t::MuonStub`, `type_` is documented as `0=DT, 1=RPC barrel, 2=CSC, 3=RPC endcap` (`MuonStub.h:163`), yet `isBarrel()` returns `type_==1` and `isEndcap()` returns `type_==0` (lines 116–117). Which convention is authoritative, and should downstream code ignore the `isBarrel()`/`isEndcap()` helpers entirely?
13. **What is the exact phiZero value per processor sector for OMTF, and is it stable across configurations?**
14. **What primitive fields are preserved between OMTF Phase-1 internal `MuonStub` and the final `RegionalMuonCand` track address?** Specifically, does `kLayers` encode which stubs contributed?

---

## 17. Next Steps

### Recommended next implementation step: **B. Build enriched common-coordinate primitive cache**

**Justification:** The audit shows that TPS stubs already provide a partially-unified representation with `coord1`, `coord2`, `eta1`, `eta2`, `quality`, `bx`, and type. However, the most powerful displaced discriminants (`t0`, `chi2`, CSC `slope`, DT `tanPsi`, DT `z/k`) are only available at the raw primitive level. The recommended path is:

1. **Implement a CMSSW analyzer/producer** that reads:
   - `L1Phase2MuDTPhDigi` (or `L1MuDTChambPhDigi` for Run-3) 
   - `L1Phase2MuDTThDigi` (or `L1MuDTChambThDigi`)
   - `CSCCorrelatedLCTDigi`
   - `RPCDigi`
   - `l1t::MuonStub` (Phase-2 TPS output)
   
2. **Build a common node representation** per stub using the schema in Section 12, filling optional fields where available per detector type.

3. **Store to flat ROOT tree or ROOT::RNTuple** with one entry per event, storing variable-length arrays of per-stub node vectors.

4. **Compute pairwise features** (curvature, phi0 proxy, local direction residual) offline during training, not in the CMSSW producer.

5. **Confirm with detector experts** questions 1–5 (DT scales, Phase-2 firmware availability) before final schema freeze.

6. **Use `RegionalMuonCand.hwDXY` and `hwPtUnconstrained`** as near-term displaced labels / loose pre-selection for training data (they are already computed by the OMTF firmware).

---

## Appendix: Key File Paths

| File | Full path |
|---|---|
| `L1MuDTChambPhDigi.h` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h` |
| `L1MuDTChambThDigi.h` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h` |
| `L1Phase2MuDTPhDigi.h` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h` |
| `L1Phase2MuDTThDigi.h` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h` |
| `L1Phase2MuDTExtPhDigi.h` | `$RBASE/src/DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h` |
| `CSCALCTDigi.h` | `$RBASE/src/DataFormats/CSCDigi/interface/CSCALCTDigi.h` |
| `CSCCLCTDigi.h` | `$RBASE/src/DataFormats/CSCDigi/interface/CSCCLCTDigi.h` |
| `CSCCorrelatedLCTDigi.h` | `$RBASE/src/DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h` |
| `RPCDigi.h` | `$RBASE/src/DataFormats/RPCDigi/interface/RPCDigi.h` |
| `RegionalMuonCand.h` | `$RBASE/src/DataFormats/L1TMuon/interface/RegionalMuonCand.h` |
| `L1MuKBMTCombinedStub.h` | `$RBASE/src/DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h` |
| `MuonStub.h` (Phase-2) | `$RBASE/src/DataFormats/L1TMuonPhase2/interface/MuonStub.h` |
| `SAMuon.h` | `$RBASE/src/DataFormats/L1TMuonPhase2/interface/SAMuon.h` |
| `Constants.h` (Phase-2) | `$RBASE/src/DataFormats/L1TMuonPhase2/interface/Constants.h` |
| `MuonStub.h` (OMTF Phase-1) | `local: L1Trigger/L1TMuonOverlapPhase1/interface/MuonStub.h` |
| `OmtfPhase2AngleConverter.h` | `local: L1Trigger/L1TMuonOverlapPhase2/interface/OmtfPhase2AngleConverter.h` |

`$RBASE` = `/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_2_0_pre2/src`
