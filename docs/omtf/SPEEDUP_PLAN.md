# OMTF Data Pipeline Speedup Plan

**Status**: Implementation in progress  
**Author**: Pablo  
**Created**: 2026-04-21

---

## Objective

Make the OMTF dataset and evaluation pipeline much faster without losing
information or changing sample semantics.

---

## Core principle

The main slowdown is repeated ROOT I/O + per-entry Python loops on every run.
Fix: **convert once, cache once, reuse many times** using `.pt` shards.

---

## Implementation order (follow exactly)

### Step 1 — Timing instrumentation  `[DONE]`
Add `--profile` to eval.py; time dataset construction, loader iteration,
model forward, and metric accumulation.

### Step 2 — ROOT-to-cache converter  `[DONE]`
`scripts/omtf/build_cache.py`  
Uses uproot batch reading (all entries per file in one shot — not one GetEntry
per entry). Applies the same `_process_entry_fn` semantics. Writes sharded
`.pt` files + `manifest.json`.

### Step 3 — Cache validator  `[DONE]`
`scripts/omtf/validate_cache.py`  
Compares N random samples between OMTFDataset (ROOT) and CachedOMTFDataset.
Checks exact tensor equality and aggregate statistics.

### Step 4 — Cache-backed dataset  `[DONE]`
`src/omtf/cache_dataset.py`  
Thin wrapper over `.pt` shards. Same `__getitem__` contract as OMTFDataset.
Validates manifest schema on load.

### Step 5 — Switch evaluation to cache  `[DONE]`
Add `--cache-dir` to eval.py. When cache is available, use CachedOMTFDataset.

### Step 6 — Runtime improvements  `[DONE]`
- `torch.inference_mode()` instead of `torch.no_grad()`
- `pin_memory=True` + `.to(device, non_blocking=True)` on CUDA
- `--batch-size` flag (default 512 for eval)
- One-pass dual-model evaluation (single DataLoader iteration for two models)

---

## Sample contract (must not change)

All of the following must be semantically identical between ROOT and cache paths:

| Field | Shape | Dtype |
|---|---|---|
| `stubs` | `(Nmax, 7)` | float32 |
| `valid_mask` | `(Nmax,)` | bool |
| `track_id` | `(Nmax,)` | int8 |
| `ambiguous` | `(Nmax,)` | uint8 |
| `node_label` | `(Nmax,)` | float32 |
| `gen_pt` | `(3,)` | float32 |
| `gen_charge` | `(3,)` | float32 |
| `gen_dxy` | `(3,)` | float32 |
| `gen_phi` | `(3,)` | float32 |
| `meta` | dict | — |
| `edge_index` | `(2, E)` | int64 (graph only) |
| `edge_attr` | `(E, 6)` | float32 (graph only) |
| `edge_label` | `(E,)` | float32 (graph only) |
| `edge_ambig` | `(E,)` | bool (graph only) |

### Policies that must not change

- Nmax = 24  
- Truncation: keep top Nmax stubs by descending quality  
- node_label = (track_id != 0).float()  
- Slot k → gen target index k corresponds to track_id = k+1  
- RAW_FEATURE_NAMES order: phi, phiB, eta, r, quality, type, layer  
- PAIR_FEATURE_NAMES order: delta_phi, delta_r, delta_r2, kappa_hat, abs_delta_eta, phiB_diff  
- kappa_clip = 500.0  
- same-layer edge exclusion: enabled  
- event join: reg_eventNum (uint32 cast) ↔ nano event (uint64 → uint32)

---

## Cache format

```
build/omtf/cache/
  schema_v1/
    manifest.json          ← schema version, constants, file list
    S1/
      shard_0000.pt
      shard_0001.pt
      ...
    B4/
      shard_0000.pt
```

Each shard is a dict saved with `torch.save`:
```
{
  "stubs":      (N, 24, 7) float32
  "valid_mask": (N, 24)    bool
  "track_id":   (N, 24)    int8
  "ambiguous":  (N, 24)    uint8
  "node_label": (N, 24)    float32
  "gen_pt":     (N, 3)     float32
  "gen_charge": (N, 3)     float32
  "gen_dxy":    (N, 3)     float32
  "gen_phi":    (N, 3)     float32
  "meta_event_num":  list[int]
  "meta_i_proc":     list[int]
  "meta_n_stubs":    list[int]
  "meta_n_gen":      list[int]
  # if include_graph:
  "edge_index": list[Tensor(2, E)]
  "edge_attr":  list[Tensor(E, 6)]
  "edge_label": list[Tensor(E,)]
  "edge_ambig": list[Tensor(E,)]
}
```

### Manifest schema

```json
{
  "schema_version": 1,
  "nmax": 24,
  "raw_feature_names": [...],
  "pair_feature_names": [...],
  "kappa_clip": 500.0,
  "truncation_policy": "descending_quality",
  "include_graph": false,
  "graph_edge_policy": "cross_layer_i_lt_j",
  "converter_version": "1",
  "created_at": "...",
  "datasets": {
    "S1": {"n_samples": ..., "n_shards": ..., "source_files": [...]}
  }
}
```

---

## Cache invalidation

Rebuild the cache whenever any of these change:
- Nmax, RAW_FEATURE_NAMES order, PAIR_FEATURE_NAMES order, kappa_clip
- Truncation policy, graph edge inclusion rule
- Label semantics, gen target slot mapping

The runtime loader refuses to run if manifest schema_version or constants mismatch.

---

## Quick-start

```bash
# Build cache for all datasets (no graph), 8 workers
python scripts/omtf/build_cache.py \
  --datasets S1 S2 S3 S4 S5 B1 B2 B3 B4 \
  --output-dir build/omtf/cache/schema_v1 \
  --num-workers 8

# Validate S1
python scripts/omtf/validate_cache.py \
  --dataset S1 \
  --cache-dir build/omtf/cache/schema_v1

# Evaluate using cache
python src/omtf/eval.py \
  --checkpoint build/omtf/checkpoints/slot_model_best.pt \
  --cache-dir build/omtf/cache/schema_v1 \
  --all-datasets
```
