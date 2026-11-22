#!/usr/bin/env python3
"""S7 snapshot-swap of the neural best checkpoints (one run per call), driven by the frozen matrix.

Velocity 5%, mask replica 0, seen220 and VAL123. Pairing and transformation are the S6 ones
(``build_within_condition_swap_pairs``; recipient sites, donor velocity at those sites, recipient
coordinates, p and target; reference RMSE(T_donor, T_recipient)). The S6 functions are reused
through ``17_s6_sparse_eval_and_swap.py``; every identity tag (family, p, seed, epoch, weights hash)
is propagated from the matrix instead of the hard-coded S6 values.

The aligned reference is REUSED from the verified final evaluation (same checkpoint hash, same
frames, same mask_hash) after the checks below, and is also replayed here as a path check.
The swap measures sensitivity to the snapshot, not physical causality.

Modes: ``full`` (COMPLETE.json only after readback and checks) and ``smoke`` (few frames, never COMPLETE).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.snapshot_swap import (  # noqa: E402
    build_within_condition_swap_pairs,
    pairs_by_dest_cache_index,
)
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402

ROLES = ("seen_train_eval", "val_eval")
N_FRAMES = {"seen_train_eval": 220, "val_eval": 123}
U5_PAIRS = ROOT / "results/month2_dense_unet/u5_pon_seed0_snapshot_control_v1/swap_pairs.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] s7swap {msg}", flush=True)


def _macro(rows: list[dict[str, Any]]) -> float:
    df = pd.DataFrame(rows)
    return float(df.groupby("sim_id")["rmse_K"].mean().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--eval-root", type=Path, default=Path("/home/bontempi/official_thesis_s7_c149_eval"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mode", choices=("full", "smoke"), default="full")
    ap.add_argument("--smoke-frames", type=int, default=12)
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    if smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("smoke mode needs an out-root named *smoke*; full must not")
    ev = _load("s7eval_for_swap", "scripts/38_s7_eval.py")
    s6 = ev._s6()
    import torch

    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    matrix = ev.load_matrix(args.matrix)
    entry = ev._run_entry(matrix, args.run_id)
    family = str(entry["family"])
    p_on = bool(entry["p_conditioning"])
    spec = entry["checkpoints"]["best"]
    _log(f"start run={args.run_id} family={family} p_on={p_on} best_epoch={spec['epoch']} mode={args.mode}")
    inspected = ev._inspect_checkpoints(entry, torch)
    consumed = ev._verify_inputs(entry, matrix)
    cache_root = consumed["cache_root"]
    normalizer = Month2FitOnlyNormalizer(matrix["shared_inputs"]["stats"])
    kw = cache_validation_kwargs(str(entry["pool_id"]))
    sh = matrix["shared_inputs"]
    u5 = json.loads(U5_PAIRS.read_text())

    datasets: dict[str, SparseUNetCacheDataset] = {}
    pairs: dict[str, list[dict[str, Any]]] = {}
    for role in ROLES:
        frames_meta = json.loads((cache_root / role / "frames.json").read_text())
        rows = [{"role": role, "sim_id": str(m["sim_id"]), "frame_index": int(m["frame_index"]), "cache_index": i}
                for i, m in enumerate(frames_meta)]
        pairs[role] = build_within_condition_swap_pairs(rows, role=role)
        key = lambda ps: sorted((p["sim_id"], p["dest_frame_index"], p["src_frame_index"]) for p in ps)  # noqa: E731
        if key(pairs[role]) != key(u5[role]):
            raise SystemExit(f"{role}: rebuilt pairing differs from the S6 U5 pairing")
        if len(pairs[role]) != N_FRAMES[role]:
            raise SystemExit(f"{role}: {len(pairs[role])} pairs, expected {N_FRAMES[role]}")
        idx = None
        if smoke:
            dest = [p["dest_cache_index"] for p in pairs[role]]
            idx = sorted(set(dest[:: max(1, len(dest) // int(args.smoke_frames))]))[: int(args.smoke_frames)]
        datasets[role] = SparseUNetCacheDataset(
            role, cache_root=cache_root, indices=idx, mask_mode="evaluation", site_fraction=0.05,
            eval_mask_seed=int(sh["eval_mask_seed"]), layout_id=str(sh["layout_id"]), replica_id=0,
            p_conditioning=p_on, normalizer=normalizer, eval_mask_payload_root=sh["mask_root"],
            require_eval_mask_payload=True, **kw)
        if not smoke and len(datasets[role]) != N_FRAMES[role]:
            raise SystemExit(f"{role}: dataset has {len(datasets[role])} frames")

    model, _ = s6._load_model(family, cache_root, device)
    query = None
    if family == "adapted_glu":
        query = make_synthetic_query_grid(datasets["val_eval"].x, datasets["val_eval"].y).to(device)
    payload = s6.load_sparse_scientific_checkpoint(inspected["best"]["path"], model=model, require_resumable=True, restore_rng=False)
    if int((payload.get("epoch_state") or {}).get("epoch", -1)) != int(spec["epoch"]):
        raise SystemExit("loaded best epoch differs from the matrix")
    model.to(device)
    model.eval()

    out_run = args.out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    tags = {
        "run_id": args.run_id, "family": family, "method": family, "variant": "p_on" if p_on else "p_off",
        "p_conditioning": p_on, "model_seed": int(entry["model_seed"]), "pool_id": entry["pool_id"],
        "checkpoint_kind": "best", "epoch": int(spec["epoch"]), "criterion": spec["criterion"],
        "file_sha256": spec["file_sha256"], "weights_sha256": spec["weights_sha256"],
        "training_git_sha": entry["training_git_sha"], "eval_git_sha": args.eval_git_sha,
        "site_fraction": 0.05, "mask_replica_id": 0,
    }
    summary: dict[str, Any] = {"run_id": args.run_id, "tags": tags, "roles": {}}
    checks: list[str] = []
    for role in ROLES:
        ds = datasets[role]
        keep = {int(i) for i in ds.indices}
        lookup = pairs_by_dest_cache_index([p for p in pairs[role] if p["dest_cache_index"] in keep])
        _log(f"{role}: swap n={len(ds)}")
        sw = s6.evaluate_swap(model=model, family=family, p_conditioning=p_on, ds=ds, role=role, pair_lookup=lookup,
                              method=family, variant="best_swapped", run_id=args.run_id, normalizer=normalizer,
                              query_coords_phys=query, device=device, batch_size=int(args.batch_size))
        al = s6.evaluate_setting(model=model, family=family, p_conditioning=p_on, ds=ds, role=role, replica_id=0,
                                 site_fraction=0.05, method=family, variant="p_on" if p_on else "p_off",
                                 checkpoint_kind="best", run_id=args.run_id, normalizer=normalizer,
                                 query_coords_phys=query, device=device, batch_size=int(args.batch_size))
        swap_rows = ev._tag_rows(sw["rows"], tags)
        for r in swap_rows:
            r["observation_mode"] = "snapshot_swapped_sparse"
        donor_rows = ev._tag_rows(sw["donor_reference_rows"], {"run_id": args.run_id, "role": role})
        replay = pd.DataFrame(al["rows"])
        # reuse of the verified aligned rows: same keys, same mask_hash, same numbers
        ref = pd.read_csv(args.eval_root / args.run_id / "per_frame" / "best" / "cov0.05" / role / "rep0.csv")
        for col, want in (("weights_sha256", spec["weights_sha256"]), ("file_sha256", spec["file_sha256"]), ("checkpoint_kind", "best")):
            if set(ref[col]) != {want}:
                checks.append(f"{role}: aligned reference {col} differs from the matrix")
        sw_df = pd.DataFrame(swap_rows)
        key = ["sim_id", "frame_index"]
        m = sw_df[key + ["mask_hash", "rmse_K"]].merge(ref[key + ["mask_hash", "rmse_K"]], on=key, suffixes=("_swap", "_aligned"))
        m = m.merge(replay[key + ["rmse_K"]].rename(columns={"rmse_K": "rmse_K_replay"}), on=key)
        if len(m) != len(sw_df):
            checks.append(f"{role}: {len(sw_df) - len(m)} swapped frames not found in the aligned reference")
        if not (m["mask_hash_swap"] == m["mask_hash_aligned"]).all():
            checks.append(f"{role}: swap mask_hash differs from the aligned reference (recipient sites must be identical)")
        replay_delta = float((m["rmse_K_replay"] - m["rmse_K_aligned"]).abs().max())
        if not np.isfinite(sw_df["rmse_K"].to_numpy()).all():
            checks.append(f"{role}: non-finite swapped rmse")
        aligned_rows = ref[ref["frame_index"].isin(sw_df["frame_index"]) & ref["sim_id"].isin(sw_df["sim_id"])]
        summary["roles"][role] = {
            "n_frames": int(len(sw_df)),
            "aligned_macro_K": _macro(ref.to_dict("records")) if not smoke else _macro(aligned_rows.to_dict("records")),
            "aligned_replay_macro_K": _macro(al["rows"]),
            "aligned_replay_max_abs_frame_delta_K": replay_delta,
            "swapped_macro_K": _macro(swap_rows),
            "donor_reference_macro_K": _macro(donor_rows),
            "wall_s_swap": sw["wall_s"], "wall_s_replay": al["wall_s"],
        }
        _write = lambda name, rows: ev._write_rows(out_run / name, rows)  # noqa: E731
        _write(f"per_frame_swap_{role}.csv", swap_rows)
        _write(f"per_frame_donor_reference_{role}.csv", donor_rows)
        (out_run / f"swap_pairs_{role}.json").write_text(json.dumps(pairs[role], indent=1) + "\n")
    summary["checks_failed"] = checks
    summary["u5_pairs_file"] = str(U5_PAIRS.relative_to(ROOT))
    summary["u5_pairs_sha256"] = ev.sha256_file(U5_PAIRS)
    summary["consumed"] = {k: str(v) for k, v in consumed.items()}
    summary["max_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    (out_run / "swap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # readback before any success marker
    for role in ROLES:
        for name, n in ((f"per_frame_swap_{role}.csv", summary["roles"][role]["n_frames"]),
                        (f"per_frame_donor_reference_{role}.csv", summary["roles"][role]["n_frames"])):
            frame = pd.read_csv(out_run / name)
            if len(frame) != n or not np.isfinite(frame["rmse_K"].to_numpy()).all():
                checks.append(f"readback failed for {name}")
    if checks:
        (out_run / "SWAP_FAILED.json").write_text(json.dumps({"ok": False, "checks_failed": checks}, indent=2) + "\n")
        _log(f"FAILED: {checks}")
        return 3
    marker = "SMOKE_OK.json" if smoke else "COMPLETE.json"
    (out_run / marker).write_text(json.dumps({"ok": True, "run_id": args.run_id, "mode": args.mode,
                                              "roles": summary["roles"]}, indent=2) + "\n")
    _log(f"done {marker}: " + json.dumps({r: round(v['swapped_macro_K'], 3) for r, v in summary['roles'].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
