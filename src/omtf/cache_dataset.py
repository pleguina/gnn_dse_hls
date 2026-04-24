"""
Cache-backed OMTF dataset.

Reads pre-converted .pt shards written by scripts/omtf/build_cache.py.
Same __getitem__ contract as OMTFDataset — drop-in replacement at eval time.

Schema validation happens at construction.  If the manifest constants differ
from the live feature definitions, construction raises ValueError.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from omtf.features import RAW_FEATURE_NAMES, PAIR_FEATURE_NAMES

SCHEMA_VERSION = 1
_KAPPA_CLIP = 500.0
_NMAX = 24
_TRUNC_POLICY = "descending_quality"
_EDGE_POLICY = "cross_layer_i_lt_j"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedOMTFDataset(Dataset):
    """
    Load OMTF samples from pre-built .pt shard cache.

    Parameters
    ----------
    cache_dir     : directory containing manifest.json and dataset sub-dirs
    dataset       : dataset name (e.g. 'S1')
    include_graph : if True, load and expose edge tensors (requires graph cache)
                    if False, graph tensors are not loaded even if present
    max_entries   : cap total samples loaded (None = all)
    """

    def __init__(
        self,
        cache_dir: str | Path,
        dataset: str,
        include_graph: bool = False,
        max_entries: int | None = None,
    ):
        cache_dir = Path(cache_dir)
        manifest = self._load_and_validate_manifest(cache_dir, dataset, include_graph)
        cache_has_graph = manifest.get("include_graph", False)

        self._include_graph = include_graph and cache_has_graph

        ds_dir = cache_dir / dataset
        shard_paths = sorted(ds_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {ds_dir}")

        # Load and concatenate shards
        stubs_list, vm_list, tid_list, amb_list, nl_list = [], [], [], [], []
        gpt_list, gc_list, gd_list, gphi_list = [], [], [], []
        meta_en, meta_ip, meta_ns, meta_ng = [], [], [], []
        ei_list, ea_list, el_list, eamb_list = [], [], [], []

        # Track whether each shard has graph data to catch mixed-shard corruption
        seen_with_graph: int = 0
        seen_without_graph: int = 0

        total = 0
        for path in shard_paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            n = shard["stubs"].shape[0]

            if max_entries is not None and total + n > max_entries:
                n = max_entries - total

            stubs_list.append(shard["stubs"][:n])
            vm_list.append(shard["valid_mask"][:n])
            tid_list.append(shard["track_id"][:n])
            amb_list.append(shard["ambiguous"][:n])
            nl_list.append(shard["node_label"][:n])
            gpt_list.append(shard["gen_pt"][:n])
            gc_list.append(shard["gen_charge"][:n])
            gd_list.append(shard["gen_dxy"][:n])
            gphi_list.append(shard["gen_phi"][:n])

            # Metadata stored as tensors in shards
            meta_en.append(shard["meta_event_num"][:n])
            meta_ip.append(shard["meta_i_proc"][:n])
            meta_ns.append(shard["meta_n_stubs"][:n])
            meta_ng.append(shard["meta_n_gen"][:n])

            shard_has_graph = "edge_index" in shard
            if shard_has_graph:
                seen_with_graph += 1
            else:
                seen_without_graph += 1

            if self._include_graph and shard_has_graph:
                ei_list.extend(shard["edge_index"][:n])
                ea_list.extend(shard["edge_attr"][:n])
                el_list.extend(shard["edge_label"][:n])
                eamb_list.extend(shard["edge_ambig"][:n])

            total += n
            if max_entries is not None and total >= max_entries:
                break

        # Enforce all-or-nothing graph consistency across shards
        if seen_with_graph > 0 and seen_without_graph > 0:
            raise RuntimeError(
                f"Mixed graph/non-graph shards in {ds_dir}: "
                f"{seen_with_graph} with graph, {seen_without_graph} without. "
                "Cache is corrupt — rebuild."
            )

        self._stubs      = torch.cat(stubs_list, dim=0)
        self._valid_mask = torch.cat(vm_list, dim=0)
        self._track_id   = torch.cat(tid_list, dim=0)
        self._ambiguous  = torch.cat(amb_list, dim=0)
        self._node_label = torch.cat(nl_list, dim=0)
        self._gen_pt     = torch.cat(gpt_list, dim=0)
        self._gen_charge = torch.cat(gc_list, dim=0)
        self._gen_dxy    = torch.cat(gd_list, dim=0)
        self._gen_phi    = torch.cat(gphi_list, dim=0)

        # Metadata as tensors — int32 is wide enough for all fields
        self._meta_event_num = torch.cat(meta_en, dim=0)
        self._meta_i_proc    = torch.cat(meta_ip, dim=0)
        self._meta_n_stubs   = torch.cat(meta_ns, dim=0)
        self._meta_n_gen     = torch.cat(meta_ng, dim=0)

        if self._include_graph:
            if len(ei_list) != total:
                raise RuntimeError(
                    f"Graph list length {len(ei_list)} != sample count {total}. "
                    "Cache is corrupt — rebuild."
                )
            self._edge_index = ei_list
            self._edge_attr  = ea_list
            self._edge_label = el_list
            self._edge_ambig = eamb_list

    # ------------------------------------------------------------------
    # Manifest loading and validation
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_validate_manifest(
        cache_dir: Path, dataset: str, include_graph: bool
    ) -> dict:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {cache_dir}")

        with open(manifest_path) as f:
            m = json.load(f)

        errors = []

        if m.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema_version={m.get('schema_version')} != expected {SCHEMA_VERSION}"
            )
        if m.get("nmax") != _NMAX:
            errors.append(
                f"nmax={m.get('nmax')} != expected {_NMAX}"
            )
        if m.get("raw_feature_names") != RAW_FEATURE_NAMES:
            errors.append(
                f"raw_feature_names mismatch\n"
                f"  cache: {m.get('raw_feature_names')}\n"
                f"  live:  {RAW_FEATURE_NAMES}"
            )
        if m.get("pair_feature_names") != PAIR_FEATURE_NAMES:
            errors.append(
                f"pair_feature_names mismatch\n"
                f"  cache: {m.get('pair_feature_names')}\n"
                f"  live:  {PAIR_FEATURE_NAMES}"
            )
        if abs(m.get("kappa_clip", 0.0) - _KAPPA_CLIP) > 1e-6:
            errors.append(
                f"kappa_clip={m.get('kappa_clip')} != expected {_KAPPA_CLIP}"
            )
        if m.get("truncation_policy") != _TRUNC_POLICY:
            errors.append(
                f"truncation_policy={m.get('truncation_policy')!r} != expected {_TRUNC_POLICY!r}"
            )
        if include_graph and m.get("graph_edge_policy") != _EDGE_POLICY:
            errors.append(
                f"graph_edge_policy={m.get('graph_edge_policy')!r} != expected {_EDGE_POLICY!r}"
            )
        if include_graph and not m.get("include_graph", False):
            errors.append(
                "include_graph=True requested but cache was built without graph data. "
                "Rebuild with --include-graph."
            )
        if dataset not in m.get("datasets", {}):
            errors.append(
                f"dataset '{dataset}' not in manifest. "
                f"Available: {list(m.get('datasets', {}).keys())}"
            )

        if errors:
            raise ValueError(
                "Cache schema mismatch — rebuild cache:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

        return m

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._stubs.shape[0]

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "stubs":      self._stubs[idx],
            "valid_mask": self._valid_mask[idx],
            "track_id":   self._track_id[idx],
            "ambiguous":  self._ambiguous[idx],
            "node_label": self._node_label[idx],
            "gen_pt":     self._gen_pt[idx],
            "gen_charge": self._gen_charge[idx],
            "gen_dxy":    self._gen_dxy[idx],
            "gen_phi":    self._gen_phi[idx],
            "meta": {
                "event_num":    self._meta_event_num[idx].item(),
                "i_processor":  self._meta_i_proc[idx].item(),
                "n_stubs":      self._meta_n_stubs[idx].item(),
                "n_gen_tracks": self._meta_n_gen[idx].item(),
            },
        }
        if self._include_graph:
            sample["edge_index"] = self._edge_index[idx]
            sample["edge_attr"]  = self._edge_attr[idx]
            sample["edge_label"] = self._edge_label[idx]
            sample["edge_ambig"] = self._edge_ambig[idx]
        return sample
