#!/usr/bin/env python3
"""Materialize nested sparse eval masks (1% ⊂ 5% ⊂ 20%) and verify 5% join."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_observations import mask_hash  # noqa: E402
from combustion_recon.data.sparse_eval_masks import (  # noqa: E402
    DEFAULT_MANIFEST_OUT,
    DEFAULT_PAYLOAD_ROOT,
    HISTORICAL_LAYOUT_ID,
    HISTORICAL_MASK_SEED,
    assert_nesting,
    historical_5pct_sites,
    load_nested_sites,
    materialize_sparse_eval_masks,
    sites_to_mask,
    verify_payload_against_manifest,
)


def _check_verified_examples(payload_root: Path) -> dict:
    """Join against verified_examples in month2_observations.json at 5% only."""
    obs = json.loads((ROOT / "data/manifests/month2_observations.json").read_text())
    checked = matched = 0
    failures = []
    for ex in obs.get("verified_examples", []):
        if abs(float(ex["site_fraction"]) - 0.05) > 1e-12:
            continue  # 1%/20% nested are new; do not require match to old independent draws
        if ex.get("layout_id") != HISTORICAL_LAYOUT_ID:
            continue
        if int(ex.get("mask_seed", -1)) != HISTORICAL_MASK_SEED:
            continue
        # Role unknown in examples — try all roles via seed regenerate only.
        vel_seed, sites, h5 = historical_5pct_sites(
            sim_id=ex["sim_id"],
            frame_index=int(ex["frame_index"]),
            replica_id=int(ex["replica_id"]),
        )
        checked += 1
        expected = ex.get("velocity_mask_hash") or ex.get("observation_hash")
        if expected is None:
            failures.append({"ex": ex, "reason": "no_hash_in_example"})
            continue
        if h5 != expected:
            failures.append(
                {
                    "sim_id": ex["sim_id"],
                    "frame_index": ex["frame_index"],
                    "replica_id": ex["replica_id"],
                    "expected": expected,
                    "got": h5,
                    "velocity_mask_seed": vel_seed,
                }
            )
        else:
            matched += 1
            # Also check payload reload if role can be found
            for role in ("train_fit", "seen_train_eval", "val_eval"):
                try:
                    loaded = load_nested_sites(
                        role=role,
                        sim_id=ex["sim_id"],
                        frame_index=int(ex["frame_index"]),
                        replica_id=int(ex["replica_id"]),
                        payload_root=payload_root,
                    )
                except KeyError:
                    continue
                assert loaded["hash_5pct"] == h5
                assert_nesting(loaded["sites_1pct"], loaded["sites_5pct"], loaded["sites_20pct"])
                assert mask_hash(sites_to_mask(loaded["sites_5pct"])) == h5
                break
    return {"checked": checked, "matched": matched, "failures": failures, "ok": len(failures) == 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--payload-root", type=Path, default=DEFAULT_PAYLOAD_ROOT)
    ap.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    args = ap.parse_args()

    man = materialize_sparse_eval_masks(
        payload_root=args.payload_root,
        manifest_out=args.manifest_out,
        overwrite=args.overwrite,
    )
    v = verify_payload_against_manifest(payload_root=args.payload_root, manifest_path=args.manifest_out)
    ex = _check_verified_examples(args.payload_root)

    summary = {
        "n_instances": man["n_instances"],
        "historical_5pct_join": man["historical_5pct_join"],
        "payload_verify": v,
        "verified_examples_5pct": ex,
        "aggregate_payload_sha256": man["aggregate_payload_sha256"],
        "manifest_path": str(args.manifest_out),
        "payload_root": str(args.payload_root),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    ok = (
        man["historical_5pct_join"]["ok"]
        and v["ok"]
        and ex["ok"]
        and man["n_instances"] == 3309
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
