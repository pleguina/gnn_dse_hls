"""
Event-level trigger efficiency.

Per-window trig_eff asks "did this processor window fire?" — but OMTF runs 12
processors in parallel and the GMT fires if ANY processor does.  This script
groups windows by CMS event number within each ROOT file (detected via
event_num resets in the cache), ORs the firing decision across processor windows
for that event, then computes event-level trigger efficiency.

Key design: event numbers restart at 1 for every ROOT file, so grouping must
happen within individual file segments, not across the full cache.  File
boundaries are detected by event_num decreasing between consecutive samples.

Usage
-----
    python src/omtf/eval_event_level.py \
        --checkpoint build/omtf/checkpoints/edge_compat_best_trig.pt \
        --cache-dir  build/omtf/cache/schema_v1_graph \
        --datasets   B1 B2 B3 S1 S2

    python src/omtf/eval_event_level.py \
        --checkpoint build/omtf/checkpoints/edge_compat_best_trig.pt \
        --cache-dir  build/omtf/cache/schema_v1_graph \
        --all-datasets --save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE_DATASETS = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
PT_THRESHOLDS  = [0, 5, 10, 20]
FIRE_THRESHOLD = 0.0
NOISE_DATASETS = {"B4"}


# ---------------------------------------------------------------------------
# Model loading — mirrors eval.py exactly
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path, device: torch.device):
    from omtf.models.baselines   import build_baseline
    from omtf.models.edge_compat import build_edge_compat
    from omtf.models.slot_model  import build_slot_model

    ckpt       = torch.load(path, map_location="cpu")
    args       = ckpt.get("args", {})
    model_name = args.get("model", "deepsets")
    hidden     = args.get("hidden", 64)
    dropout    = args.get("dropout", 0.0)
    epoch      = ckpt.get("epoch", "?")
    val_loss   = ckpt.get("best_val_loss", float("nan"))

    if model_name == "deepsets":
        model = build_baseline("deepsets", hidden=hidden, dropout=dropout)
    elif model_name == "edge_mlp":
        model = build_baseline("edge_mlp", hidden=hidden, dropout=dropout)
    elif model_name == "edge_compat":
        model = build_edge_compat(hidden=hidden, K=3, dropout=dropout)
    elif model_name == "slot_model":
        model = build_slot_model(hidden=hidden, K=3, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_name!r}")

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "unexpected key" in str(e).lower():
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            raise

    model.to(device).eval()
    print(f"  Loaded {model_name}  epoch={epoch}  val_loss={val_loss:.4f}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model, model_name


# ---------------------------------------------------------------------------
# Shard inference — returns per-sample arrays
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_shard(model, shard: dict, batch_size: int, device: torch.device):
    N         = shard["stubs"].shape[0]
    has_graph = "edge_index" in shard
    fires_out = []

    for i in range(0, N, batch_size):
        j     = min(i + batch_size, N)
        stubs = shard["stubs"][i:j].to(device)
        vm    = shard["valid_mask"][i:j].to(device)

        if has_graph:
            ei = shard["edge_index"][i:j]
            ea = shard["edge_attr"][i:j]
        else:
            ei = [torch.zeros(2, 0, dtype=torch.long)] * (j - i)
            ea = [torch.zeros(0, 6)]                   * (j - i)

        out = model(stubs, vm, ei, ea)

        if "candidate_logits" in out:
            fires = (out["candidate_logits"] > FIRE_THRESHOLD).any(dim=-1)
        else:
            nl    = out.get("node_logits", out.get("node_pred"))
            fires = (nl > FIRE_THRESHOLD).any(dim=-1)

        fires_out.append(fires.cpu())

    return (
        torch.cat(fires_out),                         # (N,) bool
        shard["gen_pt"].max(dim=-1).values,           # (N,) float
        shard["meta_event_num"],                      # (N,) int32
        shard["meta_i_proc"],                         # (N,) int32
    )


# ---------------------------------------------------------------------------
# File-boundary detection
# ---------------------------------------------------------------------------

def find_file_boundaries(event_nums: torch.Tensor) -> list[int]:
    """
    Return start indices of each ROOT-file segment within the array.
    A new file starts wherever event_num decreases (restarts from 1).
    Always includes 0 and len(event_nums) as sentinels.
    """
    en   = event_nums.tolist()
    cuts = [0]
    for i in range(1, len(en)):
        if en[i] < en[i - 1]:
            cuts.append(i)
    cuts.append(len(en))
    return cuts


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def eval_dataset(
    model,
    dataset_name: str,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict:
    ds_dir = cache_dir / dataset_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    if not shards:
        print(f"  [SKIP] no cache for {dataset_name}")
        return {}

    t0 = time.time()

    # ---- 1. Collect per-sample results across all shards ----
    all_fires, all_gpt, all_en, all_ip = [], [], [], []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu")
        fires, gpt, en, ip = infer_shard(model, shard, batch_size, device)
        all_fires.append(fires)
        all_gpt.append(gpt)
        all_en.append(en)
        all_ip.append(ip)

    fires_cat = torch.cat(all_fires)   # (N,)
    gpt_cat   = torch.cat(all_gpt)    # (N,)
    en_cat    = torch.cat(all_en)     # (N,) — event_num 1-500, resets per file
    ip_cat    = torch.cat(all_ip)     # (N,) — processor index 0-11

    N = len(fires_cat)

    # ---- 2. Per-window metrics ----
    win_result: dict[str, float] = {"n_windows": N}
    if dataset_name in NOISE_DATASETS:
        win_result["window_bg_accept"] = float(fires_cat.float().mean())
    else:
        for pt in PT_THRESHOLDS:
            mask = gpt_cat > pt
            if mask.sum() > 0:
                win_result[f"window_trig_eff@{pt}"] = float(
                    fires_cat[mask].float().mean()
                )

    # ---- 3. Find file boundaries and group within each file ----
    cuts = find_file_boundaries(en_cat)   # start indices of each file segment
    n_files = len(cuts) - 1

    # Per-event accumulators
    ev_fires_list:   list[bool]  = []
    ev_gpt_list:     list[float] = []
    ev_procs_list:   list[int]   = []   # distinct proc count per event

    for seg_idx in range(n_files):
        s, e = cuts[seg_idx], cuts[seg_idx + 1]

        seg_en    = en_cat[s:e].tolist()
        seg_fires = fires_cat[s:e].tolist()
        seg_gpt   = gpt_cat[s:e].tolist()
        seg_ip    = ip_cat[s:e].tolist()

        ev_f: dict[int, bool]  = defaultdict(lambda: False)
        ev_g: dict[int, float] = defaultdict(float)
        ev_p: dict[int, set]   = defaultdict(set)

        for en, f, g, ip in zip(seg_en, seg_fires, seg_gpt, seg_ip):
            ev_f[en] = ev_f[en] or f
            ev_g[en] = max(ev_g[en], g)
            ev_p[en].add(ip)

        for k in ev_f:
            ev_fires_list.append(ev_f[k])
            ev_gpt_list.append(ev_g[k])
            ev_procs_list.append(len(ev_p[k]))

    n_events   = len(ev_fires_list)
    ev_fires_t = torch.tensor(ev_fires_list, dtype=torch.bool)
    ev_gpt_t   = torch.tensor(ev_gpt_list,   dtype=torch.float)
    mean_procs = sum(ev_procs_list) / max(n_events, 1)

    # ---- 4. Event-level metrics ----
    ev_result: dict[str, float] = {
        "n_events":            n_events,
        "n_files":             n_files,
        "mean_procs_per_event": mean_procs,
    }

    if dataset_name in NOISE_DATASETS:
        ev_result["event_bg_accept"] = float(ev_fires_t.float().mean())
    else:
        for pt in PT_THRESHOLDS:
            mask = ev_gpt_t > pt
            if mask.sum() > 0:
                ev_result[f"event_trig_eff@{pt}"] = float(
                    ev_fires_t[mask].float().mean()
                )

    elapsed = time.time() - t0
    print(f"  {dataset_name}: {N:,} windows / {n_files} files → {n_events:,} events "
          f"(mean {mean_procs:.2f} proc/event)  [{elapsed:.0f}s]")

    return {**win_result, **ev_result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Event-level trigger efficiency")
    parser.add_argument("--checkpoint",   required=True, type=Path)
    parser.add_argument("--cache-dir",    required=True, type=Path)
    parser.add_argument("--datasets",     nargs="+", default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--batch-size",   type=int, default=256)
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--cpu",          action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    print("Loading checkpoint...")
    model, model_name = load_checkpoint(args.checkpoint, device)

    datasets = CACHE_DATASETS if args.all_datasets else (args.datasets or ["B1", "B2", "B3"])
    print(f"Datasets: {datasets}\n")

    all_results: dict[str, dict] = {}
    for ds in datasets:
        print(f"--- {ds} ---")
        all_results[ds] = eval_dataset(model, ds, args.cache_dir, args.batch_size, device)

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("  Event-level vs per-window trig_eff@10  (grouping: within each ROOT file)")
    print("=" * 80)
    print(f"{'DS':<5}  {'win@10':>8}  {'evt@10':>8}  {'gain':>7}  {'procs/evt':>10}  {'n_events':>10}")
    print("-" * 80)
    for ds, r in all_results.items():
        if ds in NOISE_DATASETS:
            print(f"{ds:<5}  {'—':>8}  {'—':>8}  {'—':>7}  "
                  f"{r.get('mean_procs_per_event', 0):>10.2f}  "
                  f"{r.get('n_events', 0):>10,}  "
                  f"  bg: win={r.get('window_bg_accept','?'):.4f}  "
                  f"evt={r.get('event_bg_accept','?'):.4f}")
        else:
            w  = r.get("window_trig_eff@10", float("nan"))
            e  = r.get("event_trig_eff@10",  float("nan"))
            mp = r.get("mean_procs_per_event", 0)
            ne = r.get("n_events", 0)
            print(f"{ds:<5}  {w:>8.4f}  {e:>8.4f}  {e-w:>+7.4f}  {mp:>10.2f}  {ne:>10,}")

    if args.save:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "build" / "omtf" / "eval" / f"{model_name}_eventlevel_{ts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model": model_name, "checkpoint": str(args.checkpoint),
                       "timestamp": ts, "results": all_results}, f, indent=2)
        print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
