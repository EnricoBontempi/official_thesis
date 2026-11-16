"""Month-2 fit-only normalizer (train_fit frames; fold-capable provenance checks)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from combustion_recon.data.dataset import load_splits, parse_sim_id
from combustion_recon.data.month2_protocol import (
    DEFAULT_FRAME_PROTOCOL,
    DEFAULT_SPLITS,
    frames_for_role,
    load_frame_protocol,
    scientific_protocol_fingerprint,
    sha256_file,
    validate_sim_ids_for_role,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MONTH2_STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"

# Explicit name mapping for operating parameters
OP_PARAM_ALIASES = {
    "phi": "equivalence_ratio",
    "equivalence_ratio": "equivalence_ratio",
    "nh3_fraction": "nh3_fraction",
}

ProvenanceMode = Literal["production", "debug_subset"]


class Month2FitOnlyNormalizer:
    """Z-score normalizer with enforceable fit-condition provenance.

    Primary path (``expected_sim_ids is None``): fit set must be the full locked
    TRAIN split and ``frame_indices`` must equal the current protocol
    ``train_fit`` list. Evaluating VAL/seen still uses this TRAIN fit set.

    Fold path: pass ``expected_sim_ids`` as a unique TRAIN subset; frames must
    still be the canonical ``train_fit`` indices unless
    ``allow_noncanonical_frame_indices`` is explicitly enabled for debug fixtures.
    """

    def __init__(
        self,
        stats_path: str | Path | None = None,
        *,
        expected_sim_ids: Sequence[str] | None = None,
        expected_frame_role: str = "train_fit",
        require_protocol_id: str | None = "month2_frame_protocol_v1",
        frame_protocol_path: str | Path | None = None,
        splits_path: str | Path | None = None,
        allow_noncanonical_frame_indices: bool = False,
    ):
        path = Path(stats_path) if stats_path is not None else DEFAULT_MONTH2_STATS
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Month-2 fit-only stats at {path}. "
                "Run scripts/05_validate_month2_pipeline.py --write-artifacts first."
            )
        with path.open() as f:
            self.stats: dict[str, Any] = json.load(f)
        self.path = path
        self._validate(
            expected_sim_ids=expected_sim_ids,
            expected_frame_role=expected_frame_role,
            require_protocol_id=require_protocol_id,
            frame_protocol_path=frame_protocol_path,
            splits_path=splits_path,
            allow_noncanonical_frame_indices=allow_noncanonical_frame_indices,
        )
        self.fields = self.stats["fields"]
        self.op_params = self.stats["operating_parameters"]

    def _validate(
        self,
        *,
        expected_sim_ids: Sequence[str] | None,
        expected_frame_role: str,
        require_protocol_id: str | None,
        frame_protocol_path: str | Path | None,
        splits_path: str | Path | None,
        allow_noncanonical_frame_indices: bool,
    ) -> None:
        meta = self.stats.get("meta", {})
        if meta.get("split") != "train":
            raise ValueError(f"Month-2 stats meta.split must be 'train', got {meta.get('split')!r}")
        if meta.get("frame_role") != expected_frame_role:
            raise ValueError(
                f"Expected frame_role={expected_frame_role!r}, got {meta.get('frame_role')!r}"
            )
        if require_protocol_id is not None and meta.get("protocol_id") != require_protocol_id:
            raise ValueError(
                f"Expected protocol_id={require_protocol_id!r}, got {meta.get('protocol_id')!r}"
            )
        if meta.get("includes_seen_frames"):
            raise ValueError("Month-2 fit-only stats must not include seen frames")
        if meta.get("source_kind") == "month1_all_train_frames":
            raise ValueError("Refusing Month-1 all-frame stats as Month-2 fit-only normalizer")
        if meta.get("source_kind") == "month2_debug_subset" and not allow_noncanonical_frame_indices:
            raise ValueError(
                "Refusing debug_subset stats in production path; "
                "pass allow_noncanonical_frame_indices=True only for explicit fixtures"
            )

        protocol = load_frame_protocol(frame_protocol_path)
        splits_file = Path(splits_path) if splits_path is not None else DEFAULT_SPLITS
        splits = load_splits(splits_file)
        train_ids = list(splits["train"])
        train_set = set(train_ids)

        sim_ids = list(meta.get("sim_ids", []))
        if len(sim_ids) != len(set(sim_ids)):
            raise ValueError("Stats sim_ids contain duplicates")
        n = meta.get("n_conditions")
        if n is not None and int(n) != len(sim_ids):
            raise ValueError(f"n_conditions={n} inconsistent with len(sim_ids)={len(sim_ids)}")
        if any(sid not in train_set for sid in sim_ids):
            bad = [sid for sid in sim_ids if sid not in train_set]
            raise ValueError(f"Stats sim_ids include non-TRAIN conditions: {bad}")

        if expected_sim_ids is None:
            # Primary path: full TRAIN fit set (also when evaluating VAL/seen).
            if set(sim_ids) != train_set or len(sim_ids) != len(train_ids):
                raise ValueError(
                    "Primary Month-2 stats must cover exactly the locked TRAIN conditions; "
                    f"extra={set(sim_ids)-train_set}, missing={train_set-set(sim_ids)}"
                )
        else:
            exp = list(expected_sim_ids)
            validate_sim_ids_for_role("train_fit", exp, splits=splits)
            if set(sim_ids) != set(exp) or len(sim_ids) != len(exp):
                raise ValueError(
                    "Stats sim_ids do not match expected fold fit set "
                    f"(extra={set(sim_ids)-set(exp)}, missing={set(exp)-set(sim_ids)})"
                )

        # Frame indices: must match protocol train_fit unless explicit debug escape.
        got_frames = [int(f) for f in meta.get("frame_indices", [])]
        if len(got_frames) != len(set(got_frames)):
            raise ValueError("Stats frame_indices contain duplicates")
        if int(meta.get("n_frames_per_condition", -1)) != len(got_frames):
            raise ValueError("n_frames_per_condition inconsistent with frame_indices")
        canonical = frames_for_role(protocol, "train_fit")
        if got_frames != canonical:
            if not allow_noncanonical_frame_indices:
                raise ValueError(
                    "Stats frame_indices must equal protocol train_fit frames; "
                    f"got {got_frames[:5]}... ({len(got_frames)}) vs canonical "
                    f"{canonical[:5]}... ({len(canonical)})"
                )
            # Debug fixtures: still forbid seen/val/test role frames by set membership
            forbidden = set(frames_for_role(protocol, "seen_train_eval"))
            if set(got_frames) & forbidden:
                raise ValueError("Debug stats must not include seen_train_eval frame indices")

        # Splits file hash (binding)
        expected_splits_hash = meta.get("splits_sha256")
        actual_splits_hash = sha256_file(splits_file)
        if expected_splits_hash and expected_splits_hash != actual_splits_hash:
            raise ValueError(
                f"Stats splits_sha256 mismatch: stats={expected_splits_hash} "
                f"current={actual_splits_hash}"
            )

        # Scientific protocol fingerprint (binding; ignores descriptive YAML churn).
        # Full protocol file sha256 is recorded historically but not required to match.
        sci = scientific_protocol_fingerprint(protocol)
        stored_sci = meta.get("protocol_scientific_sha256")
        if stored_sci is None:
            raise ValueError(
                "Stats missing protocol_scientific_sha256; regenerate or patch provenance meta"
            )
        if stored_sci != sci:
            raise ValueError(
                "Stats protocol_scientific_sha256 does not match current scientific protocol "
                f"fields (stats={stored_sci}, current={sci})"
            )

        # Field stats sanity: finite mean/std, positive std for fields we normalize
        for name, fld in self.stats.get("fields", {}).items():
            mean = float(fld["mean"])
            std = float(fld["std"])
            if not math.isfinite(mean) or not math.isfinite(std):
                raise ValueError(f"Non-finite mean/std for field {name}")
            if std <= 0:
                raise ValueError(f"Non-positive std for field {name}")

    def normalize_field(self, name: str, arr: np.ndarray) -> np.ndarray:
        s = self.fields[name]
        std = float(s["std"])
        if std <= 0:
            raise ValueError(f"Non-positive std for field {name}")
        return (arr.astype(np.float64) - float(s["mean"])) / std

    def denormalize_field(self, name: str, arr: np.ndarray) -> np.ndarray:
        s = self.fields[name]
        return arr.astype(np.float64) * float(s["std"]) + float(s["mean"])

    def _op_key(self, name: str) -> str:
        if name not in OP_PARAM_ALIASES:
            raise KeyError(f"Unknown operating parameter {name!r}")
        return OP_PARAM_ALIASES[name]

    def normalize_op_params(
        self,
        nh3_fraction: float,
        phi: float | None = None,
        *,
        equivalence_ratio: float | None = None,
    ) -> np.ndarray:
        if phi is None and equivalence_ratio is None:
            raise ValueError("Provide phi or equivalence_ratio")
        if phi is not None and equivalence_ratio is not None and float(phi) != float(equivalence_ratio):
            raise ValueError("phi and equivalence_ratio disagree")
        eq = float(equivalence_ratio if equivalence_ratio is not None else phi)
        out = np.empty(2, dtype=np.float64)
        for i, (name, val) in enumerate(
            [("nh3_fraction", nh3_fraction), ("equivalence_ratio", eq)]
        ):
            s = self.op_params[name]
            std = float(s["std"])
            if std <= 0:
                raise ValueError(f"Non-positive std for operating parameter {name}")
            out[i] = (float(val) - float(s["mean"])) / std
        return out

    def denormalize_op_params(self, z: np.ndarray) -> tuple[float, float]:
        nh3 = float(z[0]) * float(self.op_params["nh3_fraction"]["std"]) + float(
            self.op_params["nh3_fraction"]["mean"]
        )
        phi = float(z[1]) * float(self.op_params["equivalence_ratio"]["std"]) + float(
            self.op_params["equivalence_ratio"]["mean"]
        )
        return nh3, phi

    @property
    def temperature_std(self) -> float:
        return float(self.fields["Temperature"]["std"])

    @property
    def temperature_mean(self) -> float:
        return float(self.fields["Temperature"]["mean"])

    @property
    def sim_ids(self) -> list[str]:
        return list(self.stats["meta"]["sim_ids"])


def _welford_update(st: dict[str, Any], arr: np.ndarray) -> None:
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    chunk = 1_000_000
    for i in range(0, flat.size, chunk):
        x = flat[i : i + chunk]
        n_b = int(x.size)
        if n_b == 0:
            continue
        mean_b = float(x.mean())
        m2_b = float(np.dot(x - mean_b, x - mean_b))
        n_a = int(st["n"])
        if n_a == 0:
            st["mean"] = mean_b
            st["M2"] = m2_b
            st["n"] = n_b
        else:
            delta = mean_b - float(st["mean"])
            n = n_a + n_b
            st["mean"] = float(st["mean"]) + delta * n_b / n
            st["M2"] = float(st["M2"]) + m2_b + delta * delta * n_a * n_b / n
            st["n"] = n
        st["min"] = float(min(st["min"], float(x.min())))
        st["max"] = float(max(st["max"], float(x.max())))


def _finalize(st: dict[str, Any]) -> dict[str, float]:
    n = int(st["n"])
    if n < 2:
        raise ValueError("Need at least 2 voxels for std")
    var = float(st["M2"]) / (n - 1)
    return {
        "n_voxels_total": n,
        "mean": float(st["mean"]),
        "std": float(np.sqrt(var)),
        "min": float(st["min"]),
        "max": float(st["max"]),
    }


def compute_month2_fit_stats(
    *,
    sim_ids: Sequence[str],
    frame_indices: Sequence[int],
    store: Any,
    protocol_id: str | None = None,
    protocol_path: str | Path | None = None,
    protocol_sha256: str | None = None,
    splits_path: str | Path | None = None,
    splits_sha256: str | None = None,
    channels: dict[str, int] | None = None,
    provenance_mode: ProvenanceMode = "production",
) -> dict[str, Any]:
    """Stream fit-only statistics over a TRAIN subset (fold-capable).

    Arguments are validated against the current protocol **before** any payload
    read. Production mode requires exact ``train_fit`` frame indices and TRAIN
    membership. ``provenance_mode='debug_subset'`` allows non-canonical frame
    subsets for fixtures and labels ``source_kind`` accordingly (rejected by the
    production normalizer unless explicitly allowed).
    """
    protocol_file = Path(protocol_path) if protocol_path is not None else DEFAULT_FRAME_PROTOCOL
    splits_file = Path(splits_path) if splits_path is not None else DEFAULT_SPLITS
    protocol = load_frame_protocol(protocol_file)
    splits = load_splits(splits_file)

    if protocol_id is None:
        protocol_id = str(protocol["protocol_id"])
    elif protocol_id != protocol.get("protocol_id"):
        raise ValueError(
            f"protocol_id argument {protocol_id!r} != protocol file {protocol.get('protocol_id')!r}"
        )

    actual_splits_hash = sha256_file(splits_file)
    if splits_sha256 is None:
        splits_sha256 = actual_splits_hash
    elif splits_sha256 != actual_splits_hash:
        raise ValueError("splits_sha256 argument does not match splits file")

    actual_protocol_file_hash = sha256_file(protocol_file)
    if protocol_sha256 is None:
        protocol_sha256 = actual_protocol_file_hash
    # Do not require caller-supplied stale file hashes; prefer recomputed value.
    protocol_sha256 = actual_protocol_file_hash
    sci_hash = scientific_protocol_fingerprint(protocol)

    # Membership before any store I/O
    sim_list = validate_sim_ids_for_role("train_fit", list(sim_ids), splits=splits)
    frames = [int(f) for f in frame_indices]
    if len(frames) != len(set(frames)):
        raise ValueError("frame_indices contain duplicates")
    canonical = frames_for_role(protocol, "train_fit")
    source_kind = "month2_train_fit_only"
    if frames != canonical:
        if provenance_mode != "debug_subset":
            raise ValueError(
                "Production compute_month2_fit_stats requires exact protocol train_fit "
                f"frames ({len(canonical)} idxs); got {len(frames)}. "
                "Use provenance_mode='debug_subset' only for explicit test fixtures."
            )
        source_kind = "month2_debug_subset"
        seen_frames = set(frames_for_role(protocol, "seen_train_eval"))
        if set(frames) & seen_frames:
            raise ValueError("debug_subset frames must not include seen_train_eval indices")

    channels = channels or {
        "Temperature": 10,
        "Velocity_i": 11,
        "Velocity_j": 12,
    }
    field_states = {
        name: {"n": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}
        for name in channels
    }
    nh3_vals: list[float] = []
    phi_vals: list[float] = []

    for i_sid, sid in enumerate(sim_list):
        print(f"[month2_fit_stats] {i_sid+1}/{len(sim_list)} {sid}", flush=True)
        store.prefetch(sid, frames)
        nh3_pct, phi = parse_sim_id(sid)
        nh3_vals.append(nh3_pct / 100.0)
        phi_vals.append(float(phi))
        for fi in frames:
            fr = store.get_frame(sid, fi)
            _welford_update(field_states["Temperature"], fr.temperature)
            _welford_update(field_states["Velocity_i"], fr.velocity_i)
            _welford_update(field_states["Velocity_j"], fr.velocity_j)
        with store._lock:
            drop = [k for k in store._frame_cache if k[0] == sid]
            for k in drop:
                store._frame_cache.pop(k, None)

    fields = {name: _finalize(st) for name, st in field_states.items()}
    nh3_arr = np.asarray(nh3_vals, dtype=np.float64)
    phi_arr = np.asarray(phi_vals, dtype=np.float64)
    return {
        "meta": {
            "split": "train",
            "frame_role": "train_fit",
            "protocol_id": protocol_id,
            "protocol_path": str(
                protocol_file.relative_to(ROOT)
                if protocol_file.is_absolute() and ROOT in protocol_file.parents
                else protocol_file
            ),
            "protocol_sha256": protocol_sha256,
            "protocol_scientific_sha256": sci_hash,
            "protocol_hash_policy": (
                "protocol_scientific_sha256 binds frames+split ref+protocol_id; "
                "protocol_sha256 is the full YAML file hash at write time and may "
                "change under descriptive edits without invalidating scientific provenance"
            ),
            "splits_path": str(
                splits_file.relative_to(ROOT)
                if splits_file.is_absolute() and ROOT in splits_file.parents
                else splits_file
            ),
            "splits_sha256": splits_sha256,
            "n_conditions": len(sim_list),
            "sim_ids": list(sim_list),
            "n_frames_per_condition": len(frames),
            "frame_indices": frames,
            "n_frames_total": len(frames) * len(sim_list),
            "channels": channels,
            "includes_seen_frames": False,
            "clipping": False,
            "method": "welford_online_mean_var_unclipped",
            "source_kind": source_kind,
            "provenance_mode": provenance_mode,
            "parameter_name_mapping": {
                "nh3_fraction": "nh3_fraction",
                "phi": "equivalence_ratio",
                "equivalence_ratio": "equivalence_ratio",
            },
        },
        "fields": fields,
        "operating_parameters": {
            "nh3_fraction": {
                "n_conditions": int(nh3_arr.size),
                "mean": float(nh3_arr.mean()),
                "std": float(nh3_arr.std(ddof=1)) if nh3_arr.size > 1 else 0.0,
                "min": float(nh3_arr.min()),
                "max": float(nh3_arr.max()),
            },
            "equivalence_ratio": {
                "n_conditions": int(phi_arr.size),
                "mean": float(phi_arr.mean()),
                "std": float(phi_arr.std(ddof=1)) if phi_arr.size > 1 else 0.0,
                "min": float(phi_arr.min()),
                "max": float(phi_arr.max()),
            },
        },
    }


def validate_stats_reject_incompatible(
    stats: dict[str, Any],
    *,
    expected_sim_ids: Iterable[str],
) -> None:
    """Helper used by tests: raise if provenance does not match a fit subset."""
    got = set(stats["meta"]["sim_ids"])
    exp = set(expected_sim_ids)
    if got != exp:
        raise ValueError(f"incompatible stats provenance: extra={got-exp} missing={exp-got}")


def patch_stats_scientific_provenance(
    stats_path: str | Path,
    *,
    frame_protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    """Add/refresh protocol_scientific_sha256 without recomputing field statistics."""
    path = Path(stats_path)
    with path.open() as f:
        stats = json.load(f)
    protocol = load_frame_protocol(frame_protocol_path)
    # Verify frames still match before patching
    got = [int(f) for f in stats["meta"]["frame_indices"]]
    canonical = frames_for_role(protocol, "train_fit")
    if got != canonical:
        raise ValueError("Cannot patch provenance: frame_indices != current train_fit")
    stats["meta"]["protocol_scientific_sha256"] = scientific_protocol_fingerprint(protocol)
    stats["meta"]["protocol_hash_policy"] = (
        "protocol_scientific_sha256 binds frames+split ref+protocol_id; "
        "protocol_sha256 is the full YAML file hash at original compute time"
    )
    stats["meta"]["protocol_sha256_at_compute"] = stats["meta"].get("protocol_sha256")
    # Refresh full-file hash note without claiming recompute of fields
    stats["meta"]["protocol_sha256_current_file"] = sha256_file(
        Path(frame_protocol_path) if frame_protocol_path is not None else DEFAULT_FRAME_PROTOCOL
    )
    path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats
