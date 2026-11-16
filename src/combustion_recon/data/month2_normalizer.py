"""Month-2 fit-only normalizer (train_fit frames; fold-capable provenance checks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MONTH2_STATS = ROOT / "data" / "manifests" / "train_norm_stats_month2_fit.json"

# Explicit name mapping for operating parameters
OP_PARAM_ALIASES = {
    "phi": "equivalence_ratio",
    "equivalence_ratio": "equivalence_ratio",
    "nh3_fraction": "nh3_fraction",
}


class Month2FitOnlyNormalizer:
    """Z-score normalizer with configurable fit-condition provenance.

    Unlike the Month-1 ``TrainOnlyNormalizer``, this class does **not** require
    exactly 20 conditions: a future fold may fit on a reduced TRAIN subset.
    """

    def __init__(
        self,
        stats_path: str | Path | None = None,
        *,
        expected_sim_ids: Sequence[str] | None = None,
        expected_frame_role: str = "train_fit",
        require_protocol_id: str | None = "month2_frame_protocol_v1",
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
        )
        self.fields = self.stats["fields"]
        self.op_params = self.stats["operating_parameters"]

    def _validate(
        self,
        *,
        expected_sim_ids: Sequence[str] | None,
        expected_frame_role: str,
        require_protocol_id: str | None,
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
        sim_ids = list(meta.get("sim_ids", []))
        n = meta.get("n_conditions")
        if n is not None and int(n) != len(sim_ids):
            raise ValueError(f"n_conditions={n} inconsistent with len(sim_ids)={len(sim_ids)}")
        if expected_sim_ids is not None:
            exp = set(expected_sim_ids)
            got = set(sim_ids)
            if exp != got:
                raise ValueError(
                    "Stats sim_ids do not match expected fit set "
                    f"(extra={got - exp}, missing={exp - got})"
                )
        # Must never silently claim Month-1 all-frame provenance
        if meta.get("source_kind") == "month1_all_train_frames":
            raise ValueError("Refusing Month-1 all-frame stats as Month-2 fit-only normalizer")

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
            out[i] = (float(val) - float(s["mean"])) / float(s["std"])
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
        # M2_b = sum((x-mean_b)^2)
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
    protocol_id: str,
    protocol_path: str,
    protocol_sha256: str,
    splits_path: str,
    splits_sha256: str,
    channels: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Stream fit-only statistics over an arbitrary TRAIN subset (fold-capable).

    Temperature is included as a supervised target field (mean/std on fit frames).
    No per-frame target-dependent normalization is computed here.
    Raw physical values are left unclipped.
    """
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

    frames = [int(f) for f in frame_indices]
    sim_list = list(sim_ids)
    for i_sid, sid in enumerate(sim_list):
        print(f"[month2_fit_stats] {i_sid+1}/{len(sim_list)} {sid}", flush=True)
        store.prefetch(sid, frames)
        # op params once per condition
        from combustion_recon.data.dataset import parse_sim_id

        nh3_pct, phi = parse_sim_id(sid)
        nh3_vals.append(nh3_pct / 100.0)
        phi_vals.append(float(phi))
        for fi in frames:
            fr = store.get_frame(sid, fi)
            _welford_update(field_states["Temperature"], fr.temperature)
            _welford_update(field_states["Velocity_i"], fr.velocity_i)
            _welford_update(field_states["Velocity_j"], fr.velocity_j)
        # Drop this condition's frames from cache to bound memory during full fit
        # (optional; leave to caller). We clear only keys for this sim_id:
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
            "protocol_path": protocol_path,
            "protocol_sha256": protocol_sha256,
            "splits_path": splits_path,
            "splits_sha256": splits_sha256,
            "n_conditions": len(list(sim_ids)),
            "sim_ids": list(sim_ids),
            "n_frames_per_condition": len(frames),
            "frame_indices": frames,
            "n_frames_total": len(frames) * len(list(sim_ids)),
            "channels": channels,
            "includes_seen_frames": False,
            "clipping": False,
            "method": "welford_online_mean_var_unclipped",
            "source_kind": "month2_train_fit_only",
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
