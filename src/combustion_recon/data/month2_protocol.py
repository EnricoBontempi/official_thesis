"""Month-2 frame protocol helpers (configs/splits/month2_frame_protocol.yaml)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from combustion_recon.data.dataset import load_splits, parse_sim_id

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAME_PROTOCOL = ROOT / "configs" / "splits" / "month2_frame_protocol.yaml"
DEFAULT_PILOT = ROOT / "configs" / "experiments" / "month2_pilot.yaml"
DEFAULT_SPLITS = ROOT / "configs" / "splits" / "splits_interp_extrap.yaml"
DEFAULT_OC = ROOT / "data" / "manifests" / "operating_conditions.yaml"

FrameRole = Literal[
    "train_fit",
    "seen_train_eval",
    "val_eval",
    "test_interp_eval",
    "test_extrap_eval",
]

TEST_ROLES: frozenset[FrameRole] = frozenset({"test_interp_eval", "test_extrap_eval"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def expand_frame_slice(start: int, stop_exclusive: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    return list(range(int(start), int(stop_exclusive), int(step)))


def load_frame_protocol(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_FRAME_PROTOCOL
    cfg = load_yaml(p)
    # Verify referenced split hash when present
    ref = cfg.get("operating_condition_split", {})
    split_path = ROOT / ref.get("path", "configs/splits/splits_interp_extrap.yaml")
    expected = ref.get("sha256")
    if expected:
        actual = sha256_file(split_path)
        if actual != expected:
            raise ValueError(
                f"Split file hash mismatch for {split_path}: expected {expected}, got {actual}"
            )
    return cfg


def frames_for_role(protocol: dict[str, Any], role: FrameRole) -> list[int]:
    spec = protocol["frames"][role]
    frames = expand_frame_slice(spec["start"], spec["stop_exclusive"], spec["step"])
    if len(frames) != int(spec["n_per_condition"]):
        raise ValueError(
            f"Role {role}: expected {spec['n_per_condition']} frames, got {len(frames)}"
        )
    if frames and (frames[0] != int(spec["first_index"]) or frames[-1] != int(spec["last_index"])):
        raise ValueError(f"Role {role}: endpoint mismatch with protocol spec")
    return frames


def split_name_for_role(role: FrameRole) -> str:
    return {
        "train_fit": "train",
        "seen_train_eval": "train",
        "val_eval": "val",
        "test_interp_eval": "test_interp",
        "test_extrap_eval": "test_extrap",
    }[role]


def build_index(
    role: FrameRole,
    *,
    protocol: dict[str, Any] | None = None,
    splits_path: str | Path | None = None,
    sim_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a list of sample identity dicts for a frame role.

    Does not read field payloads. Test roles are indexable for manifests but
    payload access is gated elsewhere.
    """
    protocol = protocol if protocol is not None else load_frame_protocol()
    splits = load_splits(splits_path)
    split = split_name_for_role(role)
    ids = list(sim_ids) if sim_ids is not None else list(splits[split])
    frames = frames_for_role(protocol, role)
    out: list[dict[str, Any]] = []
    for sid in ids:
        nh3_pct, phi = parse_sim_id(sid)
        for fi in frames:
            out.append(
                {
                    "sim_id": sid,
                    "frame_index": int(fi),
                    "role": role,
                    "split": split,
                    "nh3_fraction": nh3_pct / 100.0,
                    "phi": float(phi),
                    "equivalence_ratio": float(phi),
                    "nh3_pct": int(nh3_pct),
                }
            )
    return out


def assert_fit_seen_disjoint(protocol: dict[str, Any] | None = None) -> None:
    protocol = protocol if protocol is not None else load_frame_protocol()
    fit = set(frames_for_role(protocol, "train_fit"))
    seen = set(frames_for_role(protocol, "seen_train_eval"))
    if fit & seen:
        raise AssertionError(f"train_fit and seen_train_eval overlap: {sorted(fit & seen)}")


def load_shard_map(oc_path: str | Path | None = None) -> dict[str, str]:
    """Map sim_id -> shard filename using operating_conditions.yaml only (no Arrow I/O)."""
    path = Path(oc_path) if oc_path is not None else DEFAULT_OC
    oc = load_yaml(path)
    return {c["sim_id"]: c["shard_filename"] for c in oc["conditions"]}
