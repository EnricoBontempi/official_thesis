#!/usr/bin/env python3
"""Join persisted 5% mask hashes vs consolidated baseline observation_hash rows.

Semantics: at K=0, baseline ``observation_hash`` for sparse methods is the same
digest as ``mask_hash`` / ``velocity_mask_hash`` (SHA256 of HxW uint8 mask).
Non-hex sentinels such as ``dense_full_velocity`` are filtered, not hidden:
they are reported separately and excluded from the equality join.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.sparse_eval_masks import DEFAULT_PAYLOAD_ROOT  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")
KEYS = ("role", "sim_id", "frame_index", "mask_replica_id")
BASELINE_CSV = (
    ROOT / "results/baseline_consolidation/consolidation_v1/month2_baseline_per_frame.csv.gz"
)
OUT_DEFAULT = ROOT / "data/manifests/month2_sparse_mask_baseline_join_5pct.json"


def load_persisted_5pct(payload_root: Path) -> dict[tuple, str]:
    out: dict[tuple, str] = {}
    for role in ("train_fit", "seen_train_eval", "val_eval"):
        path = payload_root / f"{role}_nested_sites.npz"
        data = np.load(path, allow_pickle=True)
        for i in range(len(data["sim_id"])):
            key = (
                role,
                str(data["sim_id"][i]),
                int(data["frame_index"][i]),
                int(data["replica_id"][i]),
            )
            out[key] = str(data["hash_5pct"][i])
    return out


def join_baseline(
    *,
    baseline_csv: Path = BASELINE_CSV,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
) -> dict:
    if not baseline_csv.exists():
        return {
            "ok": False,
            "status": "missing_baseline_artifact",
            "missing": str(baseline_csv),
            "note": "Cannot close historical baseline join before S4 freeze without this CSV.",
        }
    if not payload_root.exists():
        return {
            "ok": False,
            "status": "missing_mask_payload",
            "missing": str(payload_root),
        }

    df = pd.read_csv(baseline_csv, low_memory=False)
    sub = df[
        (df["site_fraction"] == 0.05)
        & (df["K"] == 0)
        & (df["layout_id"] == "random_sparse_primary")
    ].copy()
    sub["observation_hash"] = sub["observation_hash"].astype(str)
    sub["is_hex_mask_hash"] = sub["observation_hash"].map(lambda x: bool(HEX64.match(x)))

    nonhex = (
        sub.loc[~sub["is_hex_mask_hash"], "observation_hash"].value_counts().to_dict()
    )
    real = sub[sub["is_hex_mask_hash"]]
    nuniq = real.groupby(list(KEYS))["observation_hash"].nunique()
    inconsistent = int((nuniq > 1).sum())

    persisted = load_persisted_5pct(payload_root)
    matched = mismatched = missing = 0
    mismatch_samples: list[dict] = []
    for key, g in real.groupby(list(KEYS)):
        hashes = sorted(set(g["observation_hash"].tolist()))
        pk = persisted.get(key)
        if pk is None:
            missing += 1
            continue
        if pk in hashes:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_samples) < 5:
                mismatch_samples.append(
                    {
                        "key": dict(zip(KEYS, key)),
                        "baseline_hashes": hashes,
                        "persisted_hash_5pct": pk,
                    }
                )

    # Method-filtered view (GappyPOD_free): one sparse layout consumer
    gappy = real[real["method"] == "GappyPOD_free"]
    gappy_match = sum(
        1
        for key, g in gappy.groupby(list(KEYS))
        if persisted.get(key) in set(g["observation_hash"].tolist())
    )

    return {
        "ok": mismatched == 0 and missing == 0 and inconsistent == 0 and matched == 3309,
        "status": "joined",
        "semantics": {
            "baseline_field": "observation_hash",
            "persisted_field": "mask_hash_5pct / hash_5pct",
            "equivalence_at_K0": (
                "SHA256 of uint8 HxW boolean velocity mask bytes "
                "(month2_observations.mask_hash)"
            ),
            "filtered_non_hex": nonhex,
            "filter_note": (
                "Sentinel values such as dense_full_velocity are excluded from the "
                "equality join but counted explicitly; they are not silently dropped "
                "from the audit trail."
            ),
        },
        "scope": {
            "site_fraction": 0.05,
            "K": 0,
            "layout_id": "random_sparse_primary",
            "baseline_csv": str(baseline_csv),
            "payload_root": str(payload_root),
        },
        "n_baseline_rows_5pct_K0": int(len(sub)),
        "n_hex_rows": int(len(real)),
        "n_nonhex_rows": int((~sub["is_hex_mask_hash"]).sum()),
        "n_unique_keys_hex": int(len(nuniq)),
        "n_keys_inconsistent_hex": inconsistent,
        "n_matched": matched,
        "n_mismatched": mismatched,
        "n_missing_persisted": missing,
        "gappy_pod_free_matched_keys": int(gappy_match),
        "gappy_pod_free_keys": int(gappy.groupby(list(KEYS)).ngroups),
        "mismatch_samples": mismatch_samples,
        "seed_regeneration_only_is_not_historical_join": True,
        "note": (
            "This join is the historical baseline equivalence check. "
            "The earlier 3309/3309 seed↔payload self-check remains necessary but "
            "is not by itself full historical baseline equivalence."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    report = join_baseline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("ok", "status", "n_matched", "n_mismatched", "n_nonhex_rows", "gappy_pod_free_matched_keys")}, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
