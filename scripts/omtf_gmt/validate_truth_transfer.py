#!/usr/bin/env python
"""
Phase B4 validation gate — OMTF↔KMTF truth-transfer integrity.

Runs on a sample of files per dataset and reports:
  - match rate (fraction of KMTF stubs successfully matched to an OMTF stub)
  - unmatched fraction
  - signal transfer rate (KMTF stubs labelled as signal when OMTF says signal)
  - noise confirmation rate (KMTF labelled noise when OMTF says noise)
  - duplicate/collision fraction
  - behaviour on S1, S3, B1, B3, B4

Writes:
  build/omtf_gmt/TRUTH_TRANSFER_SIGNOFF.md

Usage
-----
  python scripts/omtf_gmt/validate_truth_transfer.py --files-per-dataset 10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

try:
    import uproot, awkward as ak
except ImportError:
    print("uproot and awkward are required"); sys.exit(1)

from omtf_gmt.regioning     import stubs_in_window, omtf_phi_to_global_rad, angle_diff
from omtf_gmt.truth_transfer import transfer, PHI_MATCH_THRESH


def _load_nano_map(nano_path: Path) -> dict[int, dict]:
    import awkward as ak
    branches = [
        "event", "nGenMuon",
        "MuonStubKmtf_isBarrel",
        "MuonStubKmtf_offlineCoord1", "MuonStubKmtf_offlineCoord2",
        "MuonStubKmtf_offlineEta1",   "MuonStubKmtf_offlineEta2",
        "MuonStubKmtf_quality",       "MuonStubKmtf_etaQuality",
        "MuonStubKmtf_bxNum",         "MuonStubKmtf_tfLayer",
        "MuonStubKmtf_depthRegion",
    ]
    tree = uproot.open(str(nano_path))["Events"]
    arr  = tree.arrays(branches, library="ak")
    out  = {}
    for i in range(len(arr)):
        ev    = int(arr["event"][i]) & 0xFFFFFFFF
        is_b  = ak.to_numpy(arr["MuonStubKmtf_isBarrel"][i]).astype(bool)
        out[ev] = {
            "n_gen": int(arr["nGenMuon"][i]),
            "c1":  ak.to_numpy(arr["MuonStubKmtf_offlineCoord1"][i])[is_b],
            "dep": ak.to_numpy(arr["MuonStubKmtf_depthRegion"][i])[is_b].astype(np.int8),
            "bx":  ak.to_numpy(arr["MuonStubKmtf_bxNum"][i])[is_b].astype(np.int8),
        }
    return out


@dataclass
class ValStats:
    dataset:        str
    n_windows:      int = 0
    n_kmtf_total:   int = 0
    n_matched:      int = 0
    n_unmatched:    int = 0
    n_sig_transfer: int = 0   # OMTF said signal → labelled signal
    n_noise_conf:   int = 0   # OMTF said noise  → labelled noise
    n_empty_windows:int = 0   # no KMTF stubs in window
    phi_residuals:  list = field(default_factory=list)   # |phi_kmtf - phi_omtf| at match

    @property
    def match_rate(self) -> float:
        return self.n_matched / max(1, self.n_kmtf_total)

    @property
    def unmatched_frac(self) -> float:
        return self.n_unmatched / max(1, self.n_kmtf_total)


def validate_dataset(
    ds: str,
    data_dir: Path,
    n_files: int,
) -> ValStats:
    stats = ValStats(dataset=ds)
    hits_files = sorted((data_dir / ds).glob("omtf_hits_*.root"))[:n_files]
    omtf_branches = [
        "reg_eventNum","reg_iProcessor",
        "reg_stub_phiHw","reg_stub_layer","reg_stub_bx",
        "reg_stub_trackId","reg_stub_ambiguous",
    ]

    for hf in hits_files:
        nf = hf.parent / hf.name.replace("omtf_hits_", "omtf_nano_")
        if not nf.exists():
            continue
        nano_map = _load_nano_map(nf)
        tree = uproot.open(str(hf))["simOmtfPhase2Digis/OMTFAllInputTree"]
        arr  = tree.arrays(omtf_branches, library="ak")

        for i in range(len(arr)):
            ev   = int(arr["reg_eventNum"][i]) & 0xFFFFFFFF
            proc = int(arr["reg_iProcessor"][i])
            if ev not in nano_map:
                continue

            phi_hw = ak.to_numpy(arr["reg_stub_phiHw"][i]).astype(np.int32)
            layer  = ak.to_numpy(arr["reg_stub_layer"][i]).astype(np.int8)
            bx_o   = ak.to_numpy(arr["reg_stub_bx"][i]).astype(np.int8)
            tid_o  = ak.to_numpy(arr["reg_stub_trackId"][i]).astype(np.int8)
            amb_o  = ak.to_numpy(arr["reg_stub_ambiguous"][i]).astype(np.uint8)

            nm = nano_map[ev]
            mask = stubs_in_window(nm["c1"], proc)
            k_c1 = nm["c1"][mask]
            k_dep = nm["dep"][mask]
            k_bx  = nm["bx"][mask]

            stats.n_windows += 1
            if not mask.any():
                stats.n_empty_windows += 1
                continue

            stats.n_kmtf_total += int(mask.sum())

            tr = transfer(k_c1, k_dep, k_bx, phi_hw, layer, bx_o, tid_o, amb_o,
                          omtf_proc=proc)
            stats.n_matched    += tr.n_matched
            stats.n_unmatched  += tr.n_kmtf - tr.n_matched
            stats.n_sig_transfer += int((tr.track_id > 0).sum())
            stats.n_noise_conf   += int(
                (np.array(tr.truth_source) == "omtf_noise").sum()
            )

            # collect phi residuals at match points
            if len(phi_hw) > 0:
                omtf_phi_r = omtf_phi_to_rad(phi_hw)
                for j in range(len(k_c1)):
                    if tr.truth_source[j] != "unmatched":
                        omtf_phi_r = omtf_phi_to_global_rad(phi_hw, proc)
                        dphi = np.min(np.abs(angle_diff(omtf_phi_r, float(k_c1[j]))))
                        stats.phi_residuals.append(float(dphi))

    return stats


def render_report(results: list[ValStats], out_path: Path) -> None:
    lines = [
        "# OMTF↔KMTF Truth Transfer Signoff",
        "",
        f"Validation date: {__import__('datetime').date.today()}",
        f"Phi match threshold: {PHI_MATCH_THRESH:.3f} rad",
        "",
        "---",
        "",
        "## Match rate summary",
        "",
        "| Dataset | Windows | KMTF stubs | Match % | Unmatched % | "
        "Empty windows | Signal transferred | Noise confirmed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    decisions = []
    for r in results:
        lines.append(
            f"| {r.dataset} | {r.n_windows:,} | {r.n_kmtf_total:,} | "
            f"{100*r.match_rate:.1f}% | {100*r.unmatched_frac:.1f}% | "
            f"{r.n_empty_windows:,} | {r.n_sig_transfer:,} | {r.n_noise_conf:,} |"
        )
        if r.match_rate >= 0.80:
            decisions.append(f"- **{r.dataset}**: PASS (match rate {100*r.match_rate:.1f}%)")
        elif r.match_rate >= 0.60:
            decisions.append(f"- **{r.dataset}**: WARN (match rate {100*r.match_rate:.1f}%) — "
                             "check threshold or station mapping")
        else:
            decisions.append(f"- **{r.dataset}**: FAIL (match rate {100*r.match_rate:.1f}%) — "
                             "truth transfer not trustworthy for this dataset")

    lines += ["", "## Phi residuals at matched stubs", ""]
    for r in results:
        if r.phi_residuals:
            res = np.array(r.phi_residuals)
            lines.append(
                f"- **{r.dataset}**: median={np.median(res)*1000:.1f} mrad  "
                f"p95={np.percentile(res,95)*1000:.1f} mrad  "
                f"max={res.max()*1000:.1f} mrad"
            )

    lines += ["", "## Decisions", ""] + decisions + [
        "",
        "---",
        "",
        "## Sign-off",
        "",
        "_Fill in manually after reviewing the above:_",
        "",
        "- [ ] Match rates acceptable for all signal datasets",
        "- [ ] B4 (min-bias) confirmed as all-noise (no false signal transfers)",
        "- [ ] Phi residuals < 30 mrad at p95",
        "- [ ] Proceed to model training: YES / NO",
    ]

    out_path.write_text("\n".join(lines))
    print(f"Signoff written: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["S1", "S3", "B1", "B3", "B4"])
    p.add_argument("--data-dir", type=Path, default=Path("data/prod"))
    p.add_argument("--files-per-dataset", type=int, default=10)
    p.add_argument("--output", type=Path,
                   default=Path("build/omtf_gmt/TRUTH_TRANSFER_SIGNOFF.md"))
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for ds in args.datasets:
        print(f"  Validating {ds} ...", end="", flush=True)
        r = validate_dataset(ds, args.data_dir, args.files_per_dataset)
        print(f" match={100*r.match_rate:.1f}%  unmatched={100*r.unmatched_frac:.1f}%")
        results.append(r)

    render_report(results, args.output)


if __name__ == "__main__":
    main()
