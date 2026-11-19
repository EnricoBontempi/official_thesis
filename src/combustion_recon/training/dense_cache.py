"""Dense single-frame U-Net cache (velocity + coords + p → T; targets separate)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from combustion_recon.data.month2_dataset import Month2ReconDataset
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer
from combustion_recon.data.month2_protocol import (
    FrameRole,
    build_index,
    load_frame_protocol,
    load_yaml,
    scientific_protocol_fingerprint,
    sha256_file,
)
from combustion_recon.data.shard_access import ShardStore

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DENSE_CFG = ROOT / "configs" / "experiments" / "month2_dense_unet.yaml"
DEFAULT_STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"
DEFAULT_CACHE_ROOT = ROOT / "data" / "processed" / "month2_dense_unet_cache"
DEFAULT_SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"

CHANNEL_ORDER = (
    "vi_norm",
    "vj_norm",
    "x_scaled",
    "y_scaled",
    "nh3_norm",
    "phi_norm",
)
CACHEABLE_ROLES: tuple[FrameRole, ...] = ("train_fit", "seen_train_eval", "val_eval")
FORBIDDEN_ROLES = frozenset({"test_interp_eval", "test_extrap_eval"})
REQUIRED_ROLE_FILES = (
    "vi_norm.npy",
    "vj_norm.npy",
    "target_T_norm.npy",
    "target_T_K.npy",
    "p_norm.npy",
    "frames.json",
)


def expected_counts() -> dict[str, int]:
    return {"train_fit": 760, "seen_train_eval": 220, "val_eval": 123}


# Opus C149 densified TRAIN pool (same seen/VAL as historical).
OPUS_C149_EXPECTED_COUNTS: dict[str, int] = {
    "train_fit": 2980,
    "seen_train_eval": 220,
    "val_eval": 123,
}
OPUS_C149_CACHE_ID = "month2_dense_unet_cache_opus_c149_v1"
OPUS_C149_FRAME_PROTOCOL = (
    ROOT / "configs" / "splits" / "month2_frame_protocol_opus_c149.yaml"
)
HISTORICAL_CACHE_ID = "month2_dense_unet_cache_v1"


class DenseCacheError(ValueError):
    """Raised when dense cache construction or reuse fails a contract check."""


class DenseContractError(ValueError):
    """Raised when the dense experiment YAML violates the scientific contract."""


def assert_dense_scientific_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    """Consume / refuse dense experiment YAML fields before I/O.

    Returns a normalized contract dict used by builders and runners.
    """
    if not isinstance(cfg, dict):
        raise DenseContractError("dense config must be a mapping")
    eid = cfg.get("experiment_id")
    if eid != "month2_dense_unet_v1":
        raise DenseContractError(f"unexpected experiment_id={eid!r}")

    inputs = cfg.get("inputs") or {}
    if inputs.get("temporal_mode") != "single_frame":
        raise DenseContractError("temporal_mode must be single_frame")
    if inputs.get("task_mode") != "temperature_free":
        raise DenseContractError("task_mode must be temperature_free")
    if int(inputs.get("K", -1)) != 0:
        raise DenseContractError("K must be 0")
    if float(inputs.get("site_fraction", -1)) != 1.0:
        raise DenseContractError("site_fraction must be 1.0")
    if inputs.get("velocity_components") != "planar_2C":
        raise DenseContractError("velocity_components must be planar_2C")

    ch = inputs.get("model_input_channels") or {}
    order = list(ch.get("order") or [])
    if order != list(CHANNEL_ORDER):
        raise DenseContractError(f"channel order must be {list(CHANNEL_ORDER)}, got {order}")
    if int(ch.get("count", -1)) != 6:
        raise DenseContractError("model_input_channels.count must be 6")

    refs = cfg.get("references") or {}
    stats_rel = (refs.get("fit_only_stats") or {}).get(
        "path", "data/manifests/train_norm_stats_month2_fit.json"
    )
    proto_rel = (refs.get("frame_protocol") or {}).get(
        "path", "configs/splits/month2_frame_protocol.yaml"
    )
    splits_rel = (refs.get("operating_condition_split") or {}).get(
        "path", "configs/splits/splits_interp_extrap.yaml"
    )
    stats_path = ROOT / stats_rel
    proto_path = ROOT / proto_rel
    splits_path = ROOT / splits_rel
    for p, label in (
        (stats_path, "fit_only_stats"),
        (proto_path, "frame_protocol"),
        (splits_path, "operating_condition_split"),
    ):
        if not p.exists():
            raise DenseContractError(f"missing {label} at {p}")

    protocol = load_frame_protocol(proto_path)
    if protocol.get("protocol_id") != "month2_frame_protocol_v1":
        raise DenseContractError("protocol_id must be month2_frame_protocol_v1")
    if (refs.get("frame_protocol") or {}).get("protocol_id") not in (
        None,
        "month2_frame_protocol_v1",
    ):
        raise DenseContractError("references.frame_protocol.protocol_id mismatch")

    roles = cfg.get("split_roles") or {}
    for role, n_exp in expected_counts().items():
        got = (roles.get(role) or {}).get("n_frames_expected")
        if got is not None and int(got) != n_exp:
            raise DenseContractError(f"{role} n_frames_expected={got} != {n_exp}")
    if roles.get("test_roles") not in (None, "forbidden_in_this_mission"):
        raise DenseContractError("test_roles must remain forbidden in this mission")

    targets = cfg.get("targets") or {}
    if targets.get("absent_from_inputs") is not True:
        raise DenseContractError("targets.absent_from_inputs must be true")

    return {
        "experiment_id": eid,
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "temporal_mode": "single_frame",
        "velocity_components": "planar_2C",
        "channel_order": list(CHANNEL_ORDER),
        "stats_path": stats_path,
        "protocol_path": proto_path,
        "splits_path": splits_path,
        "protocol_scientific_sha256": scientific_protocol_fingerprint(protocol),
        "stats_sha256": sha256_file(stats_path),
        "splits_sha256": sha256_file(splits_path),
    }


def dense_fields_to_input_tensor(
    *,
    vi_norm: np.ndarray,
    vj_norm: np.ndarray,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    nh3_norm: float,
    phi_norm: float,
    p_conditioning: bool = True,
    # Explicitly ignored forbidden fields (isolation / mutation tests).
    temperature: np.ndarray | None = None,
    temperature_normalized: np.ndarray | None = None,
    velocity_k: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build the 6-channel model tensor from allowed fields only.

    Forbidden arguments may be supplied for mutation tests; they must not affect
    the returned tensor. There is no vk cache channel in the dense U-Net path.
    """
    _ = temperature, temperature_normalized, velocity_k, metadata
    return build_dense_input_tensor(
        vi_norm=vi_norm,
        vj_norm=vj_norm,
        x_scaled=x_scaled,
        y_scaled=y_scaled,
        nh3_norm=nh3_norm,
        phi_norm=phi_norm,
        p_conditioning=p_conditioning,
    )


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def scale_xy_to_minus1_1(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed affine map of physical grid axes into [-1, 1] (shared extrema)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        raise DenseCacheError("x/y grids must have length >= 2")
    dx = float(x.max() - x.min())
    dy = float(y.max() - y.min())
    if dx <= 0 or dy <= 0:
        raise DenseCacheError("degenerate x/y span for geometric scaling")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    x_scaled = (2.0 * (xx - float(x.min())) / dx - 1.0).astype(np.float32)
    y_scaled = (2.0 * (yy - float(y.min())) / dy - 1.0).astype(np.float32)
    return x_scaled, y_scaled


def build_dense_input_tensor(
    *,
    vi_norm: np.ndarray,
    vj_norm: np.ndarray,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    nh3_norm: float,
    phi_norm: float,
    p_conditioning: bool = True,
) -> np.ndarray:
    """Stack six channels; p_off zeros the parameter maps without reading other priors."""
    H, W = vi_norm.shape
    if vj_norm.shape != (H, W) or x_scaled.shape != (H, W) or y_scaled.shape != (H, W):
        raise DenseCacheError("channel spatial shapes must match")
    if p_conditioning:
        nh3_map = np.full((H, W), float(nh3_norm), dtype=np.float32)
        phi_map = np.full((H, W), float(phi_norm), dtype=np.float32)
    else:
        nh3_map = np.zeros((H, W), dtype=np.float32)
        phi_map = np.zeros((H, W), dtype=np.float32)
    return np.stack(
        [
            np.asarray(vi_norm, dtype=np.float32),
            np.asarray(vj_norm, dtype=np.float32),
            np.asarray(x_scaled, dtype=np.float32),
            np.asarray(y_scaled, dtype=np.float32),
            nh3_map,
            phi_map,
        ],
        axis=0,
    )


def _role_dir(cache_root: Path, role: str) -> Path:
    return Path(cache_root) / role


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _load_inventory_lfs_hashes(
    inventory_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Map arrow basename → {sha256, size_bytes} from the versioned Hub inventory."""
    path = inventory_path or (ROOT / "data/manifests/remote_file_inventory.json")
    if not path.exists():
        return {}
    inv = json.loads(path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for entry in inv.get("entries", []):
        p = str(entry.get("path", ""))
        if not p.endswith(".arrow"):
            continue
        lfs = entry.get("lfs") or {}
        sha = lfs.get("sha256")
        if not sha:
            continue
        out[Path(p).name] = {
            "sha256": sha,
            "size_bytes": int(lfs.get("size") or entry.get("size_bytes") or 0),
            "inventory_path": p,
        }
    return out


def _bytes_on_disk(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


@dataclass
class DenseCachePaths:
    root: Path
    manifest: Path
    geometry: Path

    @classmethod
    def from_root(cls, root: Path) -> "DenseCachePaths":
        root = Path(root)
        return cls(root=root, manifest=root / "manifest.json", geometry=root / "geometry.npz")


def build_dense_cache(
    *,
    cache_root: str | Path | None = None,
    roles: Sequence[FrameRole] = CACHEABLE_ROLES,
    dense_cfg_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    store: ShardStore | None = None,
    max_cached_frames: int = 64,
    overwrite: bool = False,
    verify_n: int = 3,
    frame_protocol_path: str | Path | None = None,
    expected_role_counts: dict[str, int] | None = None,
    cache_id: str | None = None,
    skip_dense_unet_contract: bool = False,
) -> dict[str, Any]:
    """Materialize dense U-Net tensors from Month2ReconDataset (no TEST payloads).

    Opus C149: pass ``frame_protocol_path`` to the densified protocol,
    ``expected_role_counts=OPUS_C149_EXPECTED_COUNTS``, ``cache_id=OPUS_C149_CACHE_ID``,
    and ``skip_dense_unet_contract=True`` while still using historical fit-only stats.
    """
    cfg_path = Path(dense_cfg_path) if dense_cfg_path else DEFAULT_DENSE_CFG
    dense_cfg = load_yaml(cfg_path)
    counts_exp = (
        dict(expected_role_counts)
        if expected_role_counts is not None
        else expected_counts()
    )
    out_cache_id = cache_id or HISTORICAL_CACHE_ID

    if skip_dense_unet_contract:
        stats_p = (
            Path(stats_path)
            if stats_path is not None
            else ROOT / "data/manifests/train_norm_stats_month2_fit.json"
        )
        proto_path = (
            Path(frame_protocol_path)
            if frame_protocol_path is not None
            else OPUS_C149_FRAME_PROTOCOL
        )
        protocol = load_frame_protocol(proto_path)
        contract = {
            "stats_path": stats_p,
            "protocol_path": proto_path,
            "stats_sha256": _file_sha256(stats_p),
            "protocol_scientific_sha256": scientific_protocol_fingerprint(protocol),
            "splits_sha256": sha256_file(DEFAULT_SPLITS),
        }
    else:
        contract = assert_dense_scientific_contract(dense_cfg)
        stats_p = Path(stats_path) if stats_path is not None else Path(contract["stats_path"])
        proto_path = (
            Path(frame_protocol_path)
            if frame_protocol_path is not None
            else Path(contract["protocol_path"])
        )
        protocol = load_frame_protocol(proto_path)

    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    paths = DenseCachePaths.from_root(root)
    if paths.manifest.exists() and not overwrite:
        manifest = json.loads(paths.manifest.read_text())
        validate_dense_cache_manifest(
            manifest,
            cache_root=root,
            require_roles=tuple(roles),
            expected_role_counts=counts_exp,
            allowed_cache_ids=[out_cache_id],
            frame_protocol_path=proto_path,
        )
        return manifest

    if root.exists() and overwrite:
        # Remove only known role payloads under this cache root.
        for role in CACHEABLE_ROLES:
            rd = _role_dir(root, role)
            if rd.exists():
                for p in rd.glob("*"):
                    if p.is_file():
                        p.unlink()
        if paths.geometry.exists():
            paths.geometry.unlink()
        if paths.manifest.exists():
            paths.manifest.unlink()

    for role in roles:
        if role in FORBIDDEN_ROLES:
            raise DenseCacheError(f"refusing to cache TEST role {role!r}")
        if role not in CACHEABLE_ROLES:
            raise DenseCacheError(f"unsupported role {role!r}")

    normalizer = Month2FitOnlyNormalizer(stats_p)
    # Prefer keeping one condition resident to avoid Arrow thrashing.
    store = store or ShardStore(
        raw_dir=raw_dir, max_cached_frames=max(max_cached_frames, 160)
    )

    t0 = time.perf_counter()
    geometry: dict[str, np.ndarray] | None = None
    role_summaries: dict[str, Any] = {}
    shard_provenance: dict[str, Any] = {}
    inventory_hashes = _load_inventory_lfs_hashes()

    for role in roles:
        print(f"[dense_cache] building role={role}", flush=True)
        ds = Month2ReconDataset(
            role,  # type: ignore[arg-type]
            task_mode="temperature_free",
            K=0,
            site_fraction=1.0,
            p_conditioning=True,
            normalizer=normalizer,
            store=store,
            frame_protocol_path=proto_path,
            max_cached_frames=max(max_cached_frames, 160),
            allow_test_payloads=False,
        )
        n = len(ds)
        expected = counts_exp.get(role)
        if expected is not None and n != expected:
            raise DenseCacheError(f"{role}: expected {expected} frames, got {n}")

        # Group indices by condition; prefetch one shard at a time (max_cached_frames
        # holds one condition's frames without thrashing across all 20 shards).
        by_sim: dict[str, list[int]] = {}
        order: list[str] = []
        for i in range(n):
            ident = ds.get_identity(i)
            sid = str(ident["sim_id"])
            if sid not in by_sim:
                by_sim[sid] = []
                order.append(sid)
            by_sim[sid].append(i)

        for sid in order:
            if sid not in shard_provenance:
                shard_path = store.shard_path(sid)
                fname = shard_path.name
                inv = inventory_hashes.get(fname)
                size = int(shard_path.stat().st_size) if shard_path.exists() else None
                if inv is not None and size is not None and int(inv["size_bytes"]) != size:
                    raise DenseCacheError(
                        f"{sid}: local size {size} != inventory {inv['size_bytes']}"
                    )
                shard_provenance[sid] = {
                    "path": str(shard_path.relative_to(ROOT))
                    if shard_path.is_relative_to(ROOT)
                    else str(shard_path),
                    "filename": fname,
                    "size_bytes": size,
                    "sha256": inv["sha256"] if inv is not None else None,
                    "sha256_source": "remote_file_inventory.lfs"
                    if inv is not None
                    else None,
                    "size_verified_local": bool(inv is not None and size is not None),
                }

        H = W = 128
        vi = np.zeros((n, H, W), dtype=np.float32)
        vj = np.zeros((n, H, W), dtype=np.float32)
        t_norm = np.zeros((n, H, W), dtype=np.float32)
        t_phys = np.zeros((n, H, W), dtype=np.float32)
        p_norm = np.zeros((n, 2), dtype=np.float32)
        frames_meta: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]

        for sid in order:
            idxs = by_sim[sid]
            frame_indices = sorted({int(ds.get_identity(i)["frame_index"]) for i in idxs})
            print(
                f"[dense_cache] {role} prefetch {sid} ({len(frame_indices)} frames)",
                flush=True,
            )
            store.prefetch(sid, frame_indices)
            for i in idxs:
                if i % 50 == 0:
                    print(f"[dense_cache] {role} {i}/{n}", flush=True)
                sample = ds[i]
                mi = sample["model_inputs"]
                tg = sample["targets"]
                md = sample["metadata"]

                # Dense velocity from observation pack (site_fraction=1 → full grid).
                dense = np.asarray(mi["dense_values"], dtype=np.float32)
                names = list(mi["dense_channel_names"])
                if names[:2] != ["velocity_i_filled0", "velocity_j_filled0"]:
                    raise DenseCacheError(f"unexpected dense channels: {names}")
                mask = np.asarray(mi["mask_velocity"], dtype=bool)
                if mask.shape != (H, W) or not bool(mask.all()):
                    raise DenseCacheError(f"{role}[{i}]: dense mask must be all-True")
                if int(mi["K"]) != 0 or float(mi["site_fraction"]) != 1.0:
                    raise DenseCacheError("dense cache requires K=0 and site_fraction=1.0")
                if "temperature" in "".join(names).lower():
                    raise DenseCacheError("temperature channel leaked into dense_values")

                x = np.asarray(md["x"], dtype=np.float64)
                y = np.asarray(md["y"], dtype=np.float64)
                if geometry is None:
                    x_s, y_s = scale_xy_to_minus1_1(x, y)
                    geometry = {
                        "x": x.astype(np.float64),
                        "y": y.astype(np.float64),
                        "x_scaled": x_s,
                        "y_scaled": y_s,
                    }
                else:
                    if not np.allclose(geometry["x"], x) or not np.allclose(geometry["y"], y):
                        raise DenseCacheError(
                            f"grid identity failed for {md['sim_id']} frame {md['frame_index']}"
                        )

                op = mi["op_params"]["normalized"]
                vi[i] = dense[0]
                vj[i] = dense[1]
                t_norm[i] = np.asarray(tg["temperature_normalized"], dtype=np.float32)
                t_phys[i] = np.asarray(tg["temperature"], dtype=np.float32)
                p_norm[i, 0] = float(op[0])
                p_norm[i, 1] = float(op[1])

                frames_meta[i] = {
                    "cache_index": i,
                    "sim_id": sid,
                    "frame_index": int(md["frame_index"]),
                    "split": md["split"],
                    "role": role,
                    "nh3_fraction": float(md["nh3_fraction"]),
                    "phi": float(md["phi"]),
                    "equivalence_ratio": float(md["equivalence_ratio"]),
                    "nh3_norm": float(op[0]),
                    "phi_norm": float(op[1]),
                    "has_original_time_coordinate": bool(md["has_original_time_coordinate"]),
                    "timeline_regular": bool(md["timeline_regular"]),
                    "field_time_association_verified": bool(
                        md["field_time_association_verified"]
                    ),
                    "time_validated": bool(md["time_validated"]),
                    "duplicate_of": md["duplicate_of"],
                    "time_coordinate_original": float(md["time_coordinate_original"]),
                    "grid": [H, W],
                    "mask_all_true": True,
                }

        rd = _role_dir(root, role)
        rd.mkdir(parents=True, exist_ok=True)
        np.save(rd / "vi_norm.npy", vi)
        np.save(rd / "vj_norm.npy", vj)
        np.save(rd / "target_T_norm.npy", t_norm)
        np.save(rd / "target_T_K.npy", t_phys)
        np.save(rd / "p_norm.npy", p_norm)
        _write_json(rd / "frames.json", frames_meta)

        # Independent spot-check against a fresh dataset read.
        verify_idxs = sorted({0, n // 2, n - 1})[: max(1, min(verify_n, n))]
        checks = []
        for vi_idx in verify_idxs:
            s = ds[vi_idx]
            packed = build_dense_input_tensor(
                vi_norm=vi[vi_idx],
                vj_norm=vj[vi_idx],
                x_scaled=geometry["x_scaled"],
                y_scaled=geometry["y_scaled"],
                nh3_norm=float(p_norm[vi_idx, 0]),
                phi_norm=float(p_norm[vi_idx, 1]),
                p_conditioning=True,
            )
            dens = np.asarray(s["model_inputs"]["dense_values"], dtype=np.float32)
            assert np.allclose(packed[0], dens[0], rtol=0, atol=1e-5)
            assert np.allclose(packed[1], dens[1], rtol=0, atol=1e-5)
            assert np.allclose(
                t_norm[vi_idx],
                np.asarray(s["targets"]["temperature_normalized"], dtype=np.float32),
                rtol=0,
                atol=1e-5,
            )
            checks.append({"cache_index": vi_idx, "ok": True})

        role_summaries[role] = {
            "n_frames": n,
            "verify_checks": checks,
            "bytes": {
                "vi_norm": (rd / "vi_norm.npy").stat().st_size,
                "vj_norm": (rd / "vj_norm.npy").stat().st_size,
                "target_T_norm": (rd / "target_T_norm.npy").stat().st_size,
                "target_T_K": (rd / "target_T_K.npy").stat().st_size,
                "p_norm": (rd / "p_norm.npy").stat().st_size,
                "frames_json": (rd / "frames.json").stat().st_size,
            },
        }

    assert geometry is not None
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths.geometry,
        x=geometry["x"],
        y=geometry["y"],
        x_scaled=geometry["x_scaled"],
        y_scaled=geometry["y_scaled"],
    )

    # TRAIN normalized moments (diagnostic compatibility with fit-only stats).
    train_T = np.load(_role_dir(root, "train_fit") / "target_T_norm.npy")
    train_vi = np.load(_role_dir(root, "train_fit") / "vi_norm.npy")
    moments = {
        "train_fit_T_norm_mean": float(train_T.mean()),
        "train_fit_T_norm_std": float(train_T.std(ddof=1)),
        "train_fit_vi_norm_mean": float(train_vi.mean()),
        "train_fit_vi_norm_std": float(train_vi.std(ddof=1)),
        "stats_Temperature_mean": float(normalizer.temperature_mean),
        "stats_Temperature_std": float(normalizer.temperature_std),
        "note": (
            "Normalized TRAIN mean of T should be near 0 under fit-only z-score; "
            "exact float32 equality is not required."
        ),
    }

    data_fps = {
        "stats_sha256": contract["stats_sha256"],
        "frame_protocol_scientific_sha256": contract["protocol_scientific_sha256"],
        "splits_sha256": contract["splits_sha256"],
        "geometry_sha256": _file_sha256(paths.geometry),
        "channel_order": list(CHANNEL_ORDER),
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "temporal_mode": "single_frame",
        "velocity_components": "planar_2C",
    }
    # Training/model knobs are recorded separately so width/budget edits do not
    # invalidate the scientific data cache fingerprint.
    model_fps = {
        "dense_config_sha256": _file_sha256(cfg_path),
        "model_config_sha256": _file_sha256(ROOT / "configs/models/unet_small.yaml"),
        "note": "Informational only; not required to match for data-cache reuse",
    }
    try:
        stats_rel = str(stats_p.relative_to(ROOT))
    except ValueError:
        stats_rel = str(stats_p)
    try:
        proto_rel = str(proto_path.relative_to(ROOT))
    except ValueError:
        proto_rel = str(proto_path)
    try:
        cfg_rel = str(cfg_path.relative_to(ROOT))
    except ValueError:
        cfg_rel = str(cfg_path)
    manifest = {
        "cache_id": out_cache_id,
        "cache_schema_version": 2,
        "experiment_id": dense_cfg.get("experiment_id", "month2_dense_unet_v1"),
        "created_unix": time.time(),
        "build_wall_s": float(time.perf_counter() - t0),
        "git_commit": _git_head(),
        "channel_order": list(CHANNEL_ORDER),
        "dtype": "float32",
        "grid": [128, 128],
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "temporal_mode": "single_frame",
        "velocity_components": "planar_2C",
        "roles": {r: role_summaries[r] for r in roles},
        "counts": {r: role_summaries[r]["n_frames"] for r in roles},
        "bytes_total": _bytes_on_disk(root),
        "geometry_sha256": data_fps["geometry_sha256"],
        "data_fingerprints": data_fps,
        # Backward-compatible alias: data-only (no model width/budget).
        "fingerprints": {
            "stats_sha256": data_fps["stats_sha256"],
            "frame_protocol_scientific_sha256": data_fps[
                "frame_protocol_scientific_sha256"
            ],
            "splits_sha256": data_fps["splits_sha256"],
            "geometry_sha256": data_fps["geometry_sha256"],
        },
        "model_fingerprints": model_fps,
        "stats_path": stats_rel,
        "frame_protocol_path": proto_rel,
        "dense_config_path": cfg_rel,
        "shard_provenance": shard_provenance,
        "shard_sha256_policy": (
            "inventory_lfs_sha256_plus_local_size_verify; "
            "NOT locally recomputed — do not claim local content verification"
        ),
        "moments_train_fit": moments,
        "temporal_policy": (
            "single_frame; preserve temporal quality flags; "
            "field_time_association unverified on anomalous shard — do not invent times"
        ),
        "test_payloads": "none",
    }
    _write_json(paths.manifest, manifest)
    validate_dense_cache_manifest(
        manifest,
        cache_root=root,
        require_roles=tuple(roles),
        expected_role_counts=counts_exp,
        allowed_cache_ids=[out_cache_id],
        frame_protocol_path=proto_path,
    )
    return manifest


def _data_fingerprints(manifest: dict[str, Any]) -> dict[str, Any]:
    if "data_fingerprints" in manifest:
        return dict(manifest["data_fingerprints"])
    fps = dict(manifest.get("fingerprints") or {})
    # Legacy manifests mixed model hashes into fingerprints; ignore those for reuse.
    return {
        "stats_sha256": fps.get("stats_sha256"),
        "frame_protocol_scientific_sha256": fps.get("frame_protocol_scientific_sha256"),
        "splits_sha256": fps.get("splits_sha256"),
        "geometry_sha256": manifest.get("geometry_sha256") or fps.get("geometry_sha256"),
        "channel_order": list(manifest.get("channel_order") or CHANNEL_ORDER),
        "task_mode": manifest.get("task_mode", "temperature_free"),
        "K": int(manifest.get("K", 0)),
        "site_fraction": float(manifest.get("site_fraction", 1.0)),
        "temporal_mode": manifest.get("temporal_mode", "single_frame"),
        "velocity_components": manifest.get("velocity_components", "planar_2C"),
    }


def validate_dense_cache_manifest(
    manifest: dict[str, Any],
    *,
    cache_root: str | Path,
    require_roles: Sequence[str] | None = None,
    check_payload_shapes: bool = True,
    check_frame_identities: bool = True,
    expected_role_counts: dict[str, int] | None = None,
    allowed_cache_ids: Sequence[str] | None = None,
    frame_protocol_path: str | Path | None = None,
) -> None:
    root = Path(cache_root)
    allowed_ids = list(allowed_cache_ids) if allowed_cache_ids is not None else [HISTORICAL_CACHE_ID]
    if manifest.get("cache_id") not in allowed_ids:
        raise DenseCacheError(
            f"unknown cache_id {manifest.get('cache_id')!r}; allowed={allowed_ids}"
        )
    if manifest.get("test_payloads") not in (None, "none"):
        raise DenseCacheError("cache claims TEST payloads")
    if manifest.get("task_mode", "temperature_free") != "temperature_free":
        raise DenseCacheError("cache task_mode must be temperature_free")
    if int(manifest.get("K", 0)) != 0:
        raise DenseCacheError("cache K must be 0")
    if float(manifest.get("site_fraction", 1.0)) != 1.0:
        raise DenseCacheError("cache site_fraction must be 1.0")
    if list(manifest.get("channel_order") or []) != list(CHANNEL_ORDER):
        raise DenseCacheError("cache channel_order mismatch")

    roles_present = set(manifest.get("counts") or {})
    forbidden = roles_present & FORBIDDEN_ROLES
    if forbidden:
        raise DenseCacheError(f"cache contains forbidden roles: {sorted(forbidden)}")
    for role in roles_present:
        if role not in CACHEABLE_ROLES:
            raise DenseCacheError(f"unsupported role in cache: {role}")

    counts_exp = (
        dict(expected_role_counts)
        if expected_role_counts is not None
        else expected_counts()
    )
    proto_path = Path(frame_protocol_path) if frame_protocol_path is not None else None
    if proto_path is None and manifest.get("frame_protocol_path"):
        proto_path = ROOT / str(manifest["frame_protocol_path"])
    protocol = load_frame_protocol(proto_path)

    req = list(require_roles) if require_roles is not None else sorted(roles_present)
    for role in req:
        if role not in roles_present:
            raise DenseCacheError(f"required role {role!r} missing from cache")
        n_exp = counts_exp.get(role)
        n_got = int(manifest["counts"][role])
        if n_exp is not None and n_got != n_exp:
            raise DenseCacheError(f"{role} count mismatch: got {n_got}, expected {n_exp}")
        rd = _role_dir(root, role)
        for name in REQUIRED_ROLE_FILES:
            if not (rd / name).exists():
                raise DenseCacheError(f"missing {rd / name}")

    geom = root / "geometry.npz"
    if not geom.exists():
        raise DenseCacheError("missing geometry.npz")
    geom_sha = _file_sha256(geom)
    data_fps = _data_fingerprints(manifest)
    if data_fps.get("geometry_sha256") and data_fps["geometry_sha256"] != geom_sha:
        raise DenseCacheError("geometry fingerprint mismatch on reuse")

    stats_path = ROOT / manifest.get(
        "stats_path", "data/manifests/train_norm_stats_month2_fit.json"
    )
    if not stats_path.exists():
        raise DenseCacheError(f"missing stats at {stats_path}")
    if data_fps.get("stats_sha256") != _file_sha256(stats_path):
        raise DenseCacheError("stats fingerprint mismatch on reuse")
    if data_fps.get("frame_protocol_scientific_sha256") != scientific_protocol_fingerprint(
        protocol
    ):
        raise DenseCacheError("protocol scientific fingerprint mismatch on reuse")
    splits_path = DEFAULT_SPLITS
    if data_fps.get("splits_sha256") and data_fps["splits_sha256"] != sha256_file(
        splits_path
    ):
        raise DenseCacheError("splits fingerprint mismatch on reuse")

    # Explicit: inventory SHA is not a local content hash.
    if manifest.get("shard_provenance"):
        for sid, meta in manifest["shard_provenance"].items():
            src = meta.get("sha256_source")
            if src and "inventory" in str(src) and meta.get("locally_rehashed"):
                raise DenseCacheError(
                    f"{sid}: contradictory sha256_source vs locally_rehashed"
                )

    if check_payload_shapes or check_frame_identities:
        for role in req:
            rd = _role_dir(root, role)
            frames = json.loads((rd / "frames.json").read_text())
            n = int(manifest["counts"][role])
            if len(frames) != n:
                raise DenseCacheError(f"{role}: frames.json length {len(frames)} != {n}")
            if check_payload_shapes:
                for name, expect_ndim, expect_tail in (
                    ("vi_norm.npy", 3, (128, 128)),
                    ("vj_norm.npy", 3, (128, 128)),
                    ("target_T_norm.npy", 3, (128, 128)),
                    ("target_T_K.npy", 3, (128, 128)),
                    ("p_norm.npy", 2, (2,)),
                ):
                    arr = np.load(rd / name, mmap_mode="r")
                    if arr.dtype != np.float32:
                        raise DenseCacheError(f"{role}/{name}: dtype {arr.dtype} != float32")
                    if arr.ndim != expect_ndim or arr.shape[0] != n:
                        raise DenseCacheError(
                            f"{role}/{name}: shape {arr.shape} incompatible with n={n}"
                        )
                    if tuple(arr.shape[1:]) != expect_tail:
                        raise DenseCacheError(
                            f"{role}/{name}: trailing shape {arr.shape[1:]} != {expect_tail}"
                        )
            if check_frame_identities:
                expected = build_index(role, protocol=protocol)  # type: ignore[arg-type]
                if len(expected) != n:
                    raise DenseCacheError(
                        f"{role}: protocol index length {len(expected)} != cache {n}"
                    )
                for i, (row, exp) in enumerate(zip(frames, expected)):
                    if int(row.get("cache_index", i)) != i:
                        raise DenseCacheError(f"{role}[{i}]: cache_index mismatch")
                    if str(row["sim_id"]) != str(exp["sim_id"]):
                        raise DenseCacheError(
                            f"{role}[{i}]: sim_id {row['sim_id']!r} != {exp['sim_id']!r}"
                        )
                    if int(row["frame_index"]) != int(exp["frame_index"]):
                        raise DenseCacheError(
                            f"{role}[{i}]: frame_index {row['frame_index']} != {exp['frame_index']}"
                        )
                    if str(row.get("role", role)) != role:
                        raise DenseCacheError(f"{role}[{i}]: role field mismatch")


def refresh_dense_cache_manifest_metadata(
    cache_root: str | Path | None = None,
    *,
    dense_cfg_path: str | Path | None = None,
) -> dict[str, Any]:
    """Upgrade manifest metadata without rebuilding tensor payloads."""
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    paths = DenseCachePaths.from_root(root)
    if not paths.manifest.exists():
        raise FileNotFoundError(paths.manifest)
    manifest = json.loads(paths.manifest.read_text())
    cfg_path = Path(dense_cfg_path) if dense_cfg_path else DEFAULT_DENSE_CFG
    contract = assert_dense_scientific_contract(load_yaml(cfg_path))

    # Validate existing payloads first; abort before rewriting metadata on failure.
    validate_dense_cache_manifest(
        manifest,
        cache_root=root,
        require_roles=tuple(manifest.get("counts") or {}),
    )

    data_fps = {
        "stats_sha256": contract["stats_sha256"],
        "frame_protocol_scientific_sha256": contract["protocol_scientific_sha256"],
        "splits_sha256": contract["splits_sha256"],
        "geometry_sha256": _file_sha256(paths.geometry),
        "channel_order": list(CHANNEL_ORDER),
        "task_mode": "temperature_free",
        "K": 0,
        "site_fraction": 1.0,
        "temporal_mode": "single_frame",
        "velocity_components": "planar_2C",
    }
    model_fps = {
        "dense_config_sha256": _file_sha256(cfg_path),
        "model_config_sha256": _file_sha256(ROOT / "configs/models/unet_small.yaml"),
        "note": "Informational only; not required to match for data-cache reuse",
    }
    # Clarify shard hash provenance on legacy entries.
    for meta in (manifest.get("shard_provenance") or {}).values():
        if meta.get("sha256_source") is None and meta.get("sha256"):
            meta["sha256_source"] = "remote_file_inventory.lfs"
        meta["locally_rehashed"] = False

    manifest.update(
        {
            "cache_schema_version": 2,
            "channel_order": list(CHANNEL_ORDER),
            "dtype": "float32",
            "grid": [128, 128],
            "task_mode": "temperature_free",
            "K": 0,
            "site_fraction": 1.0,
            "temporal_mode": "single_frame",
            "velocity_components": "planar_2C",
            "geometry_sha256": data_fps["geometry_sha256"],
            "data_fingerprints": data_fps,
            "fingerprints": {
                "stats_sha256": data_fps["stats_sha256"],
                "frame_protocol_scientific_sha256": data_fps[
                    "frame_protocol_scientific_sha256"
                ],
                "splits_sha256": data_fps["splits_sha256"],
                "geometry_sha256": data_fps["geometry_sha256"],
            },
            "model_fingerprints": model_fps,
            "shard_sha256_policy": (
                "inventory_lfs_sha256_plus_local_size_verify; "
                "NOT locally recomputed — do not claim local content verification"
            ),
            "metadata_refreshed_unix": time.time(),
            "metadata_refreshed_commit": _git_head(),
        }
    )
    _write_json(paths.manifest, manifest)
    validate_dense_cache_manifest(
        manifest,
        cache_root=root,
        require_roles=tuple(manifest.get("counts") or {}),
    )
    return manifest


class DenseUNetCacheDataset(Dataset):
    """Torch dataset over a built dense cache; targets never enter the input tensor."""

    def __init__(
        self,
        role: FrameRole,
        *,
        cache_root: str | Path | None = None,
        p_conditioning: bool = True,
        indices: Sequence[int] | None = None,
    ):
        if role in FORBIDDEN_ROLES:
            raise DenseCacheError(f"refusing TEST role {role!r}")
        self.role = role
        self.p_conditioning = bool(p_conditioning)
        self.root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
        self.paths = DenseCachePaths.from_root(self.root)
        if not self.paths.manifest.exists():
            raise FileNotFoundError(f"Missing dense cache manifest at {self.paths.manifest}")
        self.manifest = json.loads(self.paths.manifest.read_text())
        validate_dense_cache_manifest(
            self.manifest,
            cache_root=self.root,
            require_roles=(role,),
        )

        rd = _role_dir(self.root, role)
        self.vi = np.load(rd / "vi_norm.npy", mmap_mode="r")
        self.vj = np.load(rd / "vj_norm.npy", mmap_mode="r")
        self.t_norm = np.load(rd / "target_T_norm.npy", mmap_mode="r")
        self.t_K = np.load(rd / "target_T_K.npy", mmap_mode="r")
        self.p_norm = np.load(rd / "p_norm.npy", mmap_mode="r")
        self.frames = json.loads((rd / "frames.json").read_text())
        geom = np.load(self.paths.geometry)
        self.x_scaled = np.asarray(geom["x_scaled"], dtype=np.float32)
        self.y_scaled = np.asarray(geom["y_scaled"], dtype=np.float32)
        self.x = np.asarray(geom["x"], dtype=np.float64)
        self.y = np.asarray(geom["y"], dtype=np.float64)

        n = int(self.vi.shape[0])
        if indices is None:
            self.indices = list(range(n))
        else:
            self.indices = [int(i) for i in indices]
            if any(i < 0 or i >= n for i in self.indices):
                raise IndexError("cache index out of range")

    def __len__(self) -> int:
        return len(self.indices)

    def get_frame_meta(self, i: int) -> dict[str, Any]:
        return dict(self.frames[self.indices[i]])

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        x = build_dense_input_tensor(
            vi_norm=np.asarray(self.vi[idx]),
            vj_norm=np.asarray(self.vj[idx]),
            x_scaled=self.x_scaled,
            y_scaled=self.y_scaled,
            nh3_norm=float(self.p_norm[idx, 0]),
            phi_norm=float(self.p_norm[idx, 1]),
            p_conditioning=self.p_conditioning,
        )
        y = np.asarray(self.t_norm[idx], dtype=np.float32)[None, ...]
        meta = dict(self.frames[idx])
        # Explicit allowlist for the neural forward path.
        model_inputs = {
            "input_tensor": torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            "channel_order": list(CHANNEL_ORDER),
            "p_conditioning": self.p_conditioning,
            "K": 0,
            "site_fraction": 1.0,
            "task_mode": "temperature_free",
        }
        if self.p_conditioning:
            model_inputs["op_params"] = {
                "normalized": np.asarray(self.p_norm[idx], dtype=np.float64),
                "nh3_fraction": meta["nh3_fraction"],
                "phi": meta["phi"],
                "equivalence_ratio": meta["equivalence_ratio"],
            }
        return {
            "model_inputs": model_inputs,
            "targets": {
                "temperature_normalized": torch.from_numpy(
                    np.array(y, dtype=np.float32, copy=True)
                ),
                "temperature": torch.from_numpy(
                    np.array(self.t_K[idx], dtype=np.float32, copy=True)[None, ...]
                ),
            },
            "metadata": meta,
        }


def collate_dense_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x = torch.stack([b["model_inputs"]["input_tensor"] for b in batch], dim=0)
    y = torch.stack([b["targets"]["temperature_normalized"] for b in batch], dim=0)
    y_K = torch.stack([b["targets"]["temperature"] for b in batch], dim=0)
    return {
        "input_tensor": x,
        "target_T_norm": y,
        "target_T_K": y_K,
        "metadata": [b["metadata"] for b in batch],
        "p_conditioning": bool(batch[0]["model_inputs"]["p_conditioning"]),
    }


def subset_indices_for_overfit(
    frames: Sequence[dict[str, Any]],
    *,
    n_frames: int = 8,
    min_conditions: int = 2,
    max_conditions: int = 4,
    seed: int = 0,
) -> list[int]:
    """Pick a tiny TRAIN subset spanning 2–4 conditions (deterministic)."""
    rng = np.random.default_rng(int(seed))
    by_sim: dict[str, list[int]] = {}
    for row in frames:
        by_sim.setdefault(str(row["sim_id"]), []).append(int(row["cache_index"]))
    sims = sorted(by_sim)
    if len(sims) < min_conditions:
        raise DenseCacheError("not enough conditions for overfit subset")
    n_cond = int(rng.integers(min_conditions, max_conditions + 1))
    n_cond = min(n_cond, len(sims), n_frames)
    chosen_sims = list(rng.choice(sims, size=n_cond, replace=False))
    # Round-robin frames across chosen conditions.
    pools = {s: list(by_sim[s]) for s in chosen_sims}
    for s in pools:
        rng.shuffle(pools[s])
    out: list[int] = []
    while len(out) < n_frames:
        progress = False
        for s in chosen_sims:
            if pools[s] and len(out) < n_frames:
                out.append(int(pools[s].pop()))
                progress = True
        if not progress:
            break
    if len(out) < n_frames:
        raise DenseCacheError(f"could only select {len(out)} frames")
    return sorted(out)


def manifest_fingerprint_blob(manifest: dict[str, Any]) -> str:
    payload = {
        "cache_id": manifest.get("cache_id"),
        "counts": manifest.get("counts"),
        "fingerprints": manifest.get("fingerprints"),
        "geometry_sha256": manifest.get("geometry_sha256"),
        "channel_order": manifest.get("channel_order"),
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


_CACHE_PAYLOAD_NAMES = (
    "geometry.npz",
    "manifest.json",
    "train_fit/vi_norm.npy",
    "train_fit/vj_norm.npy",
    "train_fit/target_T_norm.npy",
    "train_fit/target_T_K.npy",
    "train_fit/p_norm.npy",
    "train_fit/frames.json",
    "seen_train_eval/vi_norm.npy",
    "seen_train_eval/vj_norm.npy",
    "seen_train_eval/target_T_norm.npy",
    "seen_train_eval/target_T_K.npy",
    "seen_train_eval/p_norm.npy",
    "seen_train_eval/frames.json",
    "val_eval/vi_norm.npy",
    "val_eval/vj_norm.npy",
    "val_eval/target_T_norm.npy",
    "val_eval/target_T_K.npy",
    "val_eval/p_norm.npy",
    "val_eval/frames.json",
)


def compute_cache_payload_hashes(
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """SHA256 of each cache payload file (for post-transfer verification)."""
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    files: dict[str, str] = {}
    missing: list[str] = []
    for rel in _CACHE_PAYLOAD_NAMES:
        path = root / rel
        if not path.is_file():
            missing.append(rel)
            continue
        files[rel] = sha256_file(path)
    if missing:
        raise FileNotFoundError(
            f"cache payload files missing under {root}: {missing}"
        )
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cache_root": str(root),
        "files": files,
        "aggregate_sha256": hashlib.sha256(blob).hexdigest(),
        "n_files": len(files),
    }


def verify_cache_payload_hashes(
    expected: dict[str, Any],
    *,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute payload hashes and compare to an expected manifest."""
    actual = compute_cache_payload_hashes(cache_root)
    exp_files = expected.get("files") or {}
    mismatches = sorted(
        k
        for k in sorted(set(exp_files) | set(actual["files"]))
        if exp_files.get(k) != actual["files"].get(k)
    )
    ok = (
        not mismatches
        and expected.get("aggregate_sha256") == actual["aggregate_sha256"]
    )
    return {
        "ok": ok,
        "mismatches": mismatches,
        "expected_aggregate_sha256": expected.get("aggregate_sha256"),
        "actual_aggregate_sha256": actual["aggregate_sha256"],
        "actual": actual,
    }
