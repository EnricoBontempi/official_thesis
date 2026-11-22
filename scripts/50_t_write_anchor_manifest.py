#!/usr/bin/env python3
"""Write the frozen nested temperature-anchor manifest of the S7 temperature study.

Three uniform random layouts, FIXED across frames, conditions and roles, one per velocity-mask replica (0/1/2).
Deterministic generator: ``nested_temperature_anchor_permutation`` of the project, namespace
``month2_temperature_anchors_nested_v1``, base seed 20261121; nested up to 32 (K=1 < 4 < 9 < 16 < 32).
Nothing here reads temperature values, errors or results: the layout cannot have been selected with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.masks import nested_temperature_anchor_permutation, stable_seed  # noqa: E402

NAMESPACE = "month2_temperature_anchors_nested_v1"
BASE_SEED = 20261121
K_VALUES = [0, 1, 4, 9, 16, 32]
H = W = 128


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    g = np.load(args.geometry)
    x, y = np.asarray(g["x"], dtype=np.float64), np.asarray(g["y"], dtype=np.float64)
    if x.shape != (W,) or y.shape != (H,):
        raise SystemExit("geometry axes must be (128,) each")
    geom_sha = hashlib.sha256(args.geometry.read_bytes()).hexdigest()
    layouts = []
    for rep in (0, 1, 2):
        seed = stable_seed(NAMESPACE, BASE_SEED, rep)
        order = nested_temperature_anchor_permutation(seed=seed, height=H, width=W, max_k=32).astype(np.int64)
        assert len(set(order.tolist())) == 32
        iy, ix = order // W, order % W
        prefix = {str(k): _sha(order[:k]) for k in K_VALUES}
        layouts.append({
            "replica_id": rep,
            "seed": int(seed),
            "site_order_flat": order.tolist(),
            "iy": iy.tolist(),
            "ix": ix.tolist(),
            "x_phys": x[ix].tolist(),
            "y_phys": y[iy].tolist(),
            "prefix_sha256_by_K": prefix,
            "order_sha256": _sha(order),
            "common_domain_excluded_sites": order.tolist(),
            "common_domain_n_pixels": int(H * W - 32),
        })
    # nesting + distinctness checks
    for a in layouts:
        for k0, k1 in zip(K_VALUES[1:-1], K_VALUES[2:]):
            assert a["site_order_flat"][:k0] == a["site_order_flat"][:k1][:k0]
    assert len({tuple(l["site_order_flat"]) for l in layouts}) == 3
    manifest = {
        "schema": "month2_temperature_anchors_nested_v1",
        "algorithm": "numpy.random.default_rng(seed).permutation(H*W)[:32]; K sites = first K of the order (nested); "
                     "seed = zlib.crc32('|'.join([namespace, base_seed, replica_id])) (combustion_recon.data.masks.stable_seed)",
        "namespace": NAMESPACE,
        "base_seed": BASE_SEED,
        "K_values": K_VALUES,
        "max_K": 32,
        "grid": {"H": H, "W": W, "flat_index": "iy * W + ix (row-major)", "geometry_sha256": geom_sha,
                 "x_first_last": [float(x[0]), float(x[-1])], "y_first_last": [float(y[0]), float(y[-1])]},
        "layout_rule": "uniform random, FIXED across frames, conditions and roles; layout r pairs with velocity mask replica r; "
                       "never selected using T values, errors or results",
        "pairing_note": "layout-T / velocity-mask pairs vary jointly: their variability is not attributable to the T sensors alone",
        "common_domain": "grid minus the 32 T sites of the layout (identical for every K); off_anchor(K) = grid minus the first K sites; "
                         "anchors = the K observed sites; full-field = whole grid",
        "observation": "ideal (noise-free) temperature values at the sites, normalized with the frozen fit-only 760 normalizer",
        "historical_T9_grid": "bridge only for the separate correction sweep; NOT part of this manifest",
        "layouts": layouts,
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in manifest.items()}, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(args.out, manifest["manifest_content_sha256"], [l["order_sha256"][:12] for l in layouts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
