#!/usr/bin/env python3
"""Build and verify the Opus C149 densified dense cache.

Uses historical fit-only 760-frame normalizer. Streaming ShardStore prefetch
(one condition at a time). Refuses silent frame drops. Compares the shared
A38 (760) frames against a historical cache when provided.

Exit codes:
  0 ok
  2 build/verify failure
  3 missing raws / inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_protocol import (  # noqa: E402
    load_frame_protocol,
    scientific_protocol_fingerprint,
)
from combustion_recon.data.shard_access import DEFAULT_RAW, ShardStore  # noqa: E402
from combustion_recon.training.dense_cache import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    OPUS_C149_CACHE_ID,
    OPUS_C149_EXPECTED_COUNTS,
    OPUS_C149_FRAME_PROTOCOL,
    build_dense_cache,
    validate_dense_cache_manifest,
)
from combustion_recon.training.scientific_epoch_sampler import (  # noqa: E402
    POOL_FRAME_INDICES,
    assert_pool_nesting,
    cache_indices_for_pool,
)
from combustion_recon.training.trainer import write_json  # noqa: E402


def _compare_shared_a38(
    *,
    c149_root: Path,
    historical_root: Path,
    atol: float = 0.0,
) -> dict:
    hist_frames = json.loads((historical_root / "train_fit" / "frames.json").read_text())
    c149_frames = json.loads((c149_root / "train_fit" / "frames.json").read_text())
    if len(hist_frames) != 760:
        raise RuntimeError(f"historical train_fit has {len(hist_frames)} != 760")
    hist_key = {(f["sim_id"], int(f["frame_index"])): int(f["cache_index"]) for f in hist_frames}
    c149_key = {(f["sim_id"], int(f["frame_index"])): int(f["cache_index"]) for f in c149_frames}
    missing = sorted(set(hist_key) - set(c149_key))
    if missing:
        raise RuntimeError(f"C149 missing {len(missing)} historical A38 keys e.g. {missing[:3]}")

    arrays = ("vi_norm", "vj_norm", "target_T_norm", "target_T_K", "p_norm")
    hist_arr = {n: np.load(historical_root / "train_fit" / f"{n}.npy", mmap_mode="r") for n in arrays}
    c149_arr = {n: np.load(c149_root / "train_fit" / f"{n}.npy", mmap_mode="r") for n in arrays}

    n_checked = 0
    max_abs = {n: 0.0 for n in arrays}
    for key, h_i in hist_key.items():
        c_i = c149_key[key]
        for n in arrays:
            diff = float(np.max(np.abs(np.asarray(hist_arr[n][h_i]) - np.asarray(c149_arr[n][c_i]))))
            max_abs[n] = max(max_abs[n], diff)
            if diff > atol:
                raise RuntimeError(
                    f"shared frame mismatch {key} field={n} hist_i={h_i} c149_i={c_i} max_abs={diff}"
                )
        n_checked += 1

    # Geometry
    hg = np.load(historical_root / "geometry.npz")
    cg = np.load(c149_root / "geometry.npz")
    for k in ("x", "y", "x_scaled", "y_scaled"):
        if not np.allclose(hg[k], cg[k], rtol=0, atol=0):
            raise RuntimeError(f"geometry mismatch on {k}")

    return {
        "ok": True,
        "n_shared_frames_checked": n_checked,
        "atol": atol,
        "max_abs_diff": max_abs,
        "geometry_identical": True,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data/processed/month2_dense_unet_cache_c149",
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument(
        "--historical-cache",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "data/manifests/train_norm_stats_month2_fit.json",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out = Path(args.out_root)
    report_path = Path(args.report) if args.report else out / "c149_build_verify_report.json"

    if not args.verify_only:
        if not raw_dir.exists():
            print(f"MISSING_RAW_DIR {raw_dir}", flush=True)
            return 3
        try:
            store = ShardStore(raw_dir=raw_dir, max_cached_frames=8)
            from combustion_recon.data.dataset import load_splits

            sid0 = load_splits()["train"][0]
            path0 = store.shard_path(sid0)
            if not path0.exists():
                print(f"MISSING_SHARD {path0}", flush=True)
                return 3
        except Exception as exc:  # noqa: BLE001
            print(f"RAW_PROBE_FAILED {exc}", flush=True)
            return 3
        print(f"[c149] building cache at {out} from raw_dir={raw_dir}", flush=True)
        manifest = build_dense_cache(
            cache_root=out,
            raw_dir=raw_dir,
            stats_path=args.stats,
            frame_protocol_path=OPUS_C149_FRAME_PROTOCOL,
            expected_role_counts=OPUS_C149_EXPECTED_COUNTS,
            cache_id=OPUS_C149_CACHE_ID,
            skip_dense_unet_contract=True,
            overwrite=bool(args.overwrite),
            max_cached_frames=160,
            verify_n=3,
        )
    else:
        if not (out / "manifest.json").exists():
            print(f"MISSING_CACHE_MANIFEST {out}", flush=True)
            return 3
        manifest = json.loads((out / "manifest.json").read_text())

    validate_dense_cache_manifest(
        manifest,
        cache_root=out,
        require_roles=("train_fit", "seen_train_eval", "val_eval"),
        expected_role_counts=OPUS_C149_EXPECTED_COUNTS,
        allowed_cache_ids=[OPUS_C149_CACHE_ID],
        frame_protocol_path=OPUS_C149_FRAME_PROTOCOL,
    )

    frames = json.loads((out / "train_fit" / "frames.json").read_text())
    nesting = assert_pool_nesting(frames)
    if not nesting["ok_b19_a38_on_cache"]:
        raise SystemExit("nesting B19/A38 failed on C149 cache")
    if not all(v.get("c149_fully_in_cache") for v in nesting["conditions"].values()):
        raise SystemExit("C149 not fully present after build")
    n_c149 = len(cache_indices_for_pool(frames, "C149"))
    n_a38 = len(cache_indices_for_pool(frames, "A38"))
    n_b19 = len(cache_indices_for_pool(frames, "B19"))
    if (n_c149, n_a38, n_b19) != (2980, 760, 380):
        raise SystemExit(f"pool index counts unexpected: C149={n_c149} A38={n_a38} B19={n_b19}")

    # Disjointness vs seen
    seen = json.loads((out / "seen_train_eval" / "frames.json").read_text())
    train_fi = {(f["sim_id"], int(f["frame_index"])) for f in frames}
    seen_fi = {(f["sim_id"], int(f["frame_index"])) for f in seen}
    overlap = train_fi & seen_fi
    if overlap:
        raise SystemExit(f"train/seen overlap {len(overlap)}")

    # Definition nesting
    b19, a38, c149 = (
        set(POOL_FRAME_INDICES["B19"]),
        set(POOL_FRAME_INDICES["A38"]),
        set(POOL_FRAME_INDICES["C149"]),
    )
    if not (b19 <= a38 <= c149):
        raise SystemExit("definition nesting failed")

    hist = Path(args.historical_cache)
    shared = None
    if hist.exists() and (hist / "manifest.json").exists():
        shared = _compare_shared_a38(c149_root=out, historical_root=hist, atol=0.0)
    else:
        raise SystemExit(f"historical cache required for shared-frame gate: {hist}")

    proto = load_frame_protocol(OPUS_C149_FRAME_PROTOCOL)
    report = {
        "status": "ok",
        "cache_id": OPUS_C149_CACHE_ID,
        "cache_root": str(out),
        "raw_dir": str(raw_dir),
        "counts": manifest.get("counts"),
        "pool_index_counts": {"C149": n_c149, "A38": n_a38, "B19": n_b19},
        "nesting": nesting,
        "train_seen_overlap": 0,
        "shared_a38_vs_historical": shared,
        "c149_protocol_scientific_sha256": scientific_protocol_fingerprint(proto),
        "stats_path": str(args.stats),
        "note": (
            "Normalizer is historical fit-only 760-frame stats; C149 changes only "
            "the supervised optimization pool."
        ),
    }
    write_json(report_path, report)
    print(json.dumps({"status": "ok", "report": str(report_path), "counts": report["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"C149_BUILD_FAILED {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(2) from exc
