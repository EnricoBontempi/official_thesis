"""Persisted nested eval masks for Month-2 sparse protocol (1% ⊂ 5% ⊂ 20%).

The 5% layer reuses the historical seed path (mask_seed=20261116,
layout_id=random_sparse_primary, replicas 0/1/2) so hashes match the
seed-regenerated baseline masks. Nested 1%/20% are NEW artifacts with
separate RNG namespaces; they do not overwrite historical layouts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from combustion_recon.data.masks import random_sparse_mask, stable_seed
from combustion_recon.data.month2_observations import mask_hash, observation_mask_seed

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAMES_MANIFEST = ROOT / "data" / "manifests" / "month2_frames.json"
DEFAULT_PAYLOAD_ROOT = ROOT / "data" / "processed" / "month2_sparse_eval_masks"
DEFAULT_MANIFEST_OUT = ROOT / "data" / "manifests" / "month2_sparse_eval_masks_manifest.json"

FORMAT_ID = "month2_sparse_eval_masks_v1"
ALGORITHM_ID = "nested_subset_extension_v1"
HISTORICAL_MASK_SEED = 20261116
HISTORICAL_LAYOUT_ID = "random_sparse_primary"
EVAL_ROLES = ("train_fit", "seen_train_eval", "val_eval")
REPLICA_IDS = (0, 1, 2)
N_SITES = {0.01: 164, 0.05: 819, 0.20: 3277}
GRID_HW = (128, 128)


@dataclass(frozen=True)
class NestedSiteIndices:
    role: str
    sim_id: str
    frame_index: int
    replica_id: int
    layout_id: str
    mask_seed: int
    velocity_mask_seed: int
    sites_1pct: np.ndarray
    sites_5pct: np.ndarray
    sites_20pct: np.ndarray
    hash_1pct: str
    hash_5pct: str
    hash_20pct: str

    def to_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "sim_id": self.sim_id,
            "frame_index": int(self.frame_index),
            "replica_id": int(self.replica_id),
            "layout_id": self.layout_id,
            "mask_seed": int(self.mask_seed),
            "velocity_mask_seed": int(self.velocity_mask_seed),
            "n_sites_1pct": int(self.sites_1pct.size),
            "n_sites_5pct": int(self.sites_5pct.size),
            "n_sites_20pct": int(self.sites_20pct.size),
            "mask_hash_1pct": self.hash_1pct,
            "mask_hash_5pct": self.hash_5pct,
            "mask_hash_20pct": self.hash_20pct,
            # Indices persisted separately in payload arrays for compactness.
        }


def sites_to_mask(sites: np.ndarray, *, height: int = 128, width: int = 128) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    flat = np.asarray(sites, dtype=np.int64).ravel()
    if flat.size and (flat.min() < 0 or flat.max() >= height * width):
        raise ValueError("site index out of range")
    if flat.size != np.unique(flat).size:
        raise ValueError("site indices must be unique")
    mask.ravel()[flat] = True
    return mask


def historical_5pct_sites(
    *,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    mask_seed: int = HISTORICAL_MASK_SEED,
    layout_id: str = HISTORICAL_LAYOUT_ID,
    height: int = 128,
    width: int = 128,
) -> tuple[int, np.ndarray, str]:
    """Regenerate historical 5% sites (sorted) and mask hash."""
    vel_seed = observation_mask_seed(
        mask_seed=mask_seed,
        sim_id=sim_id,
        frame_index=frame_index,
        layout_id=layout_id,
        replica_id=replica_id,
    )
    mask = random_sparse_mask(0.05, seed=vel_seed, height=height, width=width)
    sites = np.flatnonzero(mask).astype(np.int64)
    # random_sparse_mask uses unsorted choice; flatnonzero is row-major sorted.
    assert sites.size == N_SITES[0.05]
    return int(vel_seed), sites, mask_hash(mask)


def build_nested_sites_from_5pct(
    sites_5pct: np.ndarray,
    *,
    velocity_mask_seed: int,
    height: int = 128,
    width: int = 128,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """1% ⊂ 5% ⊂ 20% with dedicated RNG namespaces (new artifacts)."""
    n = height * width
    s5 = np.asarray(sites_5pct, dtype=np.int64)
    if s5.size != N_SITES[0.05]:
        raise ValueError(f"expected {N_SITES[0.05]} 5% sites, got {s5.size}")
    if np.any(s5[:-1] >= s5[1:]):
        s5 = np.sort(s5)

    rng1 = np.random.default_rng(stable_seed(int(velocity_mask_seed), "nested_1pct_subset_v1"))
    s1 = np.sort(rng1.choice(s5, size=N_SITES[0.01], replace=False).astype(np.int64))

    complement = np.setdiff1d(np.arange(n, dtype=np.int64), s5, assume_unique=False)
    n_extra = N_SITES[0.20] - N_SITES[0.05]
    rng20 = np.random.default_rng(stable_seed(int(velocity_mask_seed), "nested_20pct_extension_v1"))
    extra = rng20.choice(complement, size=n_extra, replace=False).astype(np.int64)
    s20 = np.sort(np.concatenate([s5, extra]))

    h1 = mask_hash(sites_to_mask(s1, height=height, width=width))
    h20 = mask_hash(sites_to_mask(s20, height=height, width=width))
    return s1, s20, h1, h20


def assert_nesting(s1: np.ndarray, s5: np.ndarray, s20: np.ndarray) -> None:
    set1, set5, set20 = set(s1.tolist()), set(s5.tolist()), set(s20.tolist())
    if not set1.issubset(set5):
        raise AssertionError("1% not subset of 5%")
    if not set5.issubset(set20):
        raise AssertionError("5% not subset of 20%")
    if len(set1) != N_SITES[0.01] or len(set5) != N_SITES[0.05] or len(set20) != N_SITES[0.20]:
        raise AssertionError("unexpected site counts after nesting")


def build_nested_for_frame(
    *,
    role: str,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    mask_seed: int = HISTORICAL_MASK_SEED,
    layout_id: str = HISTORICAL_LAYOUT_ID,
) -> NestedSiteIndices:
    vel_seed, s5, h5 = historical_5pct_sites(
        sim_id=sim_id,
        frame_index=frame_index,
        replica_id=replica_id,
        mask_seed=mask_seed,
        layout_id=layout_id,
    )
    s1, s20, h1, h20 = build_nested_sites_from_5pct(s5, velocity_mask_seed=vel_seed)
    assert_nesting(s1, s5, s20)
    return NestedSiteIndices(
        role=role,
        sim_id=sim_id,
        frame_index=int(frame_index),
        replica_id=int(replica_id),
        layout_id=layout_id,
        mask_seed=int(mask_seed),
        velocity_mask_seed=int(vel_seed),
        sites_1pct=s1,
        sites_5pct=s5,
        sites_20pct=s20,
        hash_1pct=h1,
        hash_5pct=h5,
        hash_20pct=h20,
    )


def iter_eval_frame_keys(frames_manifest: dict[str, Any] | Path) -> list[dict[str, Any]]:
    if isinstance(frames_manifest, Path):
        frames_manifest = json.loads(Path(frames_manifest).read_text())
    roles = frames_manifest["roles"]
    if isinstance(roles, dict):
        role_blocks = [{**v, "role": k} for k, v in roles.items()]
    else:
        role_blocks = list(roles)
    out: list[dict[str, Any]] = []
    for block in role_blocks:
        role = block["role"]
        if role not in EVAL_ROLES:
            continue  # never materialize TEST payloads
        # Prefer explicit frame list; else cartesian sim_ids × frame_indices.
        if "frames" in block:
            for fr in block["frames"]:
                out.append(
                    {
                        "role": role,
                        "sim_id": fr["sim_id"],
                        "frame_index": int(fr["frame_index"]),
                        "split": fr.get("split", block.get("split")),
                    }
                )
        else:
            for sim_id in block["sim_ids"]:
                for frame_index in block["frame_indices"]:
                    out.append(
                        {
                            "role": role,
                            "sim_id": sim_id,
                            "frame_index": int(frame_index),
                            "split": block.get("split"),
                        }
                    )
    return out


def expected_n_eval_instances() -> int:
    # 1103 frames × 3 replicas
    return 1103 * len(REPLICA_IDS)


def materialize_sparse_eval_masks(
    *,
    frames_manifest_path: Path = DEFAULT_FRAMES_MANIFEST,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
    manifest_out: Path = DEFAULT_MANIFEST_OUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build persisted sorted site indices + verify historical 5% hash identity."""
    payload_root = Path(payload_root)
    if payload_root.exists() and any(payload_root.iterdir()) and not overwrite:
        raise FileExistsError(f"refuse overwrite of {payload_root}; pass overwrite=True")
    payload_root.mkdir(parents=True, exist_ok=True)

    frames = json.loads(Path(frames_manifest_path).read_text())
    keys = iter_eval_frame_keys(frames)
    n_frames = len(keys)
    if n_frames != 1103:
        raise RuntimeError(f"expected 1103 frames, got {n_frames}")

    records: list[dict[str, Any]] = []
    # Store one npz per role for manageable files.
    by_role: dict[str, list[NestedSiteIndices]] = {r: [] for r in EVAL_ROLES}

    for key in keys:
        for rid in REPLICA_IDS:
            item = build_nested_for_frame(
                role=key["role"],
                sim_id=key["sim_id"],
                frame_index=key["frame_index"],
                replica_id=rid,
            )
            by_role[key["role"]].append(item)
            records.append(item.to_record())

    n_inst = len(records)
    if n_inst != expected_n_eval_instances():
        raise RuntimeError(f"expected {expected_n_eval_instances()} instances, got {n_inst}")

    # Round-trip verify: recreate 5% from seed and from persisted sites → same hash.
    mismatches_5: list[dict[str, Any]] = []
    for item in (x for xs in by_role.values() for x in xs):
        vel_seed2, s5b, h5b = historical_5pct_sites(
            sim_id=item.sim_id,
            frame_index=item.frame_index,
            replica_id=item.replica_id,
        )
        if h5b != item.hash_5pct or vel_seed2 != item.velocity_mask_seed:
            mismatches_5.append(
                {
                    "sim_id": item.sim_id,
                    "frame_index": item.frame_index,
                    "replica_id": item.replica_id,
                    "hash_a": item.hash_5pct,
                    "hash_b": h5b,
                }
            )
        h5_from_sites = mask_hash(sites_to_mask(item.sites_5pct))
        if h5_from_sites != item.hash_5pct:
            mismatches_5.append(
                {
                    "sim_id": item.sim_id,
                    "frame_index": item.frame_index,
                    "replica_id": item.replica_id,
                    "reason": "sites_to_mask_hash_mismatch",
                }
            )

    payload_files: dict[str, str] = {}
    for role, items in by_role.items():
        fname = f"{role}_nested_sites.npz"
        path = payload_root / fname
        np.savez_compressed(
            path,
            sim_id=np.array([it.sim_id for it in items], dtype=object),
            frame_index=np.array([it.frame_index for it in items], dtype=np.int64),
            replica_id=np.array([it.replica_id for it in items], dtype=np.int64),
            velocity_mask_seed=np.array([it.velocity_mask_seed for it in items], dtype=np.int64),
            sites_1pct=np.stack([it.sites_1pct for it in items], axis=0),
            sites_5pct=np.stack([it.sites_5pct for it in items], axis=0),
            sites_20pct=np.stack([it.sites_20pct for it in items], axis=0),
            hash_1pct=np.array([it.hash_1pct for it in items], dtype=object),
            hash_5pct=np.array([it.hash_5pct for it in items], dtype=object),
            hash_20pct=np.array([it.hash_20pct for it in items], dtype=object),
        )
        payload_files[fname] = _file_sha256(path)

    # Compact index table (hashes only); full site arrays stay out of Git.
    index_path = payload_root / "index_records.jsonl"
    with index_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    payload_files["index_records.jsonl"] = _file_sha256(index_path)

    aggregate = _aggregate_sha(payload_files)
    manifest = {
        "format_id": FORMAT_ID,
        "algorithm_id": ALGORITHM_ID,
        "mask_seed": HISTORICAL_MASK_SEED,
        "layout_id": HISTORICAL_LAYOUT_ID,
        "roles": list(EVAL_ROLES),
        "replica_ids": list(REPLICA_IDS),
        "n_frames": n_frames,
        "n_instances": n_inst,
        "n_sites": dict(N_SITES),
        "grid_hw": list(GRID_HW),
        "nested_policy": {
            "pct_5": "historical random_sparse_mask(0.05, seed=observation_mask_seed(...))",
            "pct_1": "subset of 5% via RNG namespace nested_1pct_subset_v1",
            "pct_20": "5% union extension from complement via nested_20pct_extension_v1",
            "note": "1/20% nested are new artifacts; do not overwrite historical layouts",
        },
        "hash_semantics": {
            "mask_hash": "sha256 of uint8 HxW boolean mask bytes (month2_observations.mask_hash)",
            "not_observation_hash": (
                "Baseline observation_hash may equal velocity_mask_hash at K=0; "
                "compare only objects with the same semantics."
            ),
        },
        "historical_5pct_join": {
            "n_checked": n_inst,
            "n_matched": n_inst - len(mismatches_5),
            "n_mismatched": len(mismatches_5),
            "ok": len(mismatches_5) == 0,
            "mismatches_sample": mismatches_5[:5],
        },
        "payload_root": str(payload_root),
        "payload_files_sha256": payload_files,
        "aggregate_payload_sha256": aggregate,
        "depends_on_target_T": False,
        "test_payloads": "none",
    }
    Path(manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Also copy manifest next to payload for transfer.
    (payload_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_nested_sites(
    *,
    role: str,
    sim_id: str,
    frame_index: int,
    replica_id: int,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
) -> dict[str, Any]:
    path = Path(payload_root) / f"{role}_nested_sites.npz"
    data = np.load(path, allow_pickle=True)
    sim_ids = data["sim_id"].tolist()
    frames = data["frame_index"]
    reps = data["replica_id"]
    for i, (s, fr, rp) in enumerate(zip(sim_ids, frames, reps)):
        if s == sim_id and int(fr) == int(frame_index) and int(rp) == int(replica_id):
            return {
                "sites_1pct": data["sites_1pct"][i].astype(np.int64),
                "sites_5pct": data["sites_5pct"][i].astype(np.int64),
                "sites_20pct": data["sites_20pct"][i].astype(np.int64),
                "hash_1pct": str(data["hash_1pct"][i]),
                "hash_5pct": str(data["hash_5pct"][i]),
                "hash_20pct": str(data["hash_20pct"][i]),
                "velocity_mask_seed": int(data["velocity_mask_seed"][i]),
            }
    raise KeyError(f"no nested sites for {role}/{sim_id}/{frame_index}/r{replica_id}")


def verify_payload_against_manifest(
    *,
    payload_root: Path = DEFAULT_PAYLOAD_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_OUT,
) -> dict[str, Any]:
    man = json.loads(Path(manifest_path).read_text())
    payload_root = Path(payload_root)
    mismatches = []
    for name, expected in man["payload_files_sha256"].items():
        path = payload_root / name
        if not path.exists():
            mismatches.append({"file": name, "expected": expected, "actual": None, "missing": True})
            continue
        actual = _file_sha256(path)
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "aggregate": man["aggregate_payload_sha256"],
        "historical_5pct_join": man.get("historical_5pct_join"),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _aggregate_sha(files: dict[str, str]) -> str:
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
