"""Resumable single-run evaluation loop for Mission 02 (Virtual PIV).

Designed to be called by Sonnet's GPU evaluation jobs AND tested directly on CPU with
a lightweight callable ``forward_fn`` (no checkpoints or GPU required in unit tests).

Key properties:
* Reuses :class:`PivObservationGenerator` (without modifying it) and
  :func:`evaluate_piv_frame` (which wraps Mission 00's :func:`evaluate_frame_prediction`).
* Computes ALL full-field metrics (isotherms 1800/2100 K, in/out-field, distance bins,
  SENS-REJ nodes, online spatial error accumulators) in ONE forward pass before
  discarding transient predictions — avoiding duplicate inference and keeping disk
  footprint small.
* Resumable at the ``(checkpoint_kind, eval_config, role, replica)`` granularity via
  atomic writes and signature-checked sidecar markers (``.ok.json``), so restarts
  never duplicate rows or recompute finished units.
* Strictly refuses ``test*`` roles unless explicit R5 authorization is passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.data.piv_observations import (
    PivObservationGenerator,
    _steps,
    field_rect,
)
from combustion_recon.evaluation.piv_manifest import (
    ALLOWED_DEV_ROLES,
    DEFAULT_REPRESENTATIVE_FRAMES,
    FORBIDDEN_ROLE_PREFIXES,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from combustion_recon.evaluation.piv_metrics import (
    SpatialErrorAccumulator,
    distance_to_valid_vector_mm,
    evaluate_piv_frame,
    field_rect_mask,
)


class PivEvalError(RuntimeError):
    """Contract or runtime violation in PIV evaluation."""


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def atomic_savez_compressed(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _check_role_allowed(role: str, *, allow_test: bool = False) -> None:
    if any(role.lower().startswith(p) for p in FORBIDDEN_ROLE_PREFIXES) and not allow_test:
        raise PivEvalError(f"role {role!r} is refused (TEST access requires explicit R5 authorization)")
    if not allow_test and role not in ALLOWED_DEV_ROLES:
        raise PivEvalError(f"role {role!r} not in allowed development roles {ALLOWED_DEV_ROLES}")


def unit_signature(
    *,
    run_id: str,
    checkpoint_kind: str,
    checkpoint_weights_sha256: str,
    eval_config: str,
    role: str,
    replica: int,
    spec_sha256: str,
    n_expected_frames: int,
) -> str:
    return sha256_json(
        {
            "run_id": run_id,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_weights_sha256": checkpoint_weights_sha256,
            "eval_config": eval_config,
            "role": role,
            "replica": int(replica),
            "spec_sha256": spec_sha256,
            "n_expected_frames": int(n_expected_frames),
        }
    )


def is_unit_complete(csv_path: Path, expected_sig: str) -> bool:
    marker = csv_path.with_suffix(".ok.json")
    if not csv_path.is_file() or not marker.is_file():
        return False
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("signature") != expected_sig or not meta.get("ok"):
        return False
    return sha256_file(csv_path) == meta.get("csv_sha256")


def evaluate_piv_unit(
    *,
    frames: Sequence[Mapping[str, Any]],
    generator: PivObservationGenerator,
    eval_config: str,
    role: str,
    replica: int,
    forward_fn: Callable[[Sequence[Mapping[str, Any]]], np.ndarray],
    train_sigma_T: float,
    run_tags: Mapping[str, Any],
    batch_size: int = 8,
    obs_role: str = "eval",
    keep_fields_keys: set[tuple[str, int]] | None = None,
    keep_all_val_rep0: bool = False,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Evaluate one ``(eval_config, role, replica)`` unit for a single model.

    ``frames`` is a sequence of dicts with keys:
      ``sim_id``, ``split``, ``frame_index``, ``vi_phys`` (H,W), ``vj_phys`` (H,W),
      ``true_T_K`` (H,W), and optionally ``p_norm`` (2,), ``vel_region_masks``.
    ``forward_fn`` receives a batch of observation packets (each containing ``piv_obs``,
    ``frame``) and returns ``pred_T_K`` of shape ``(B, H, W)`` in Kelvin.

    Returns rows, timing, online :class:`SpatialErrorAccumulator`, and retained fields.
    """
    _check_role_allowed(role, allow_test=allow_test)
    cfg = generator.spec.config(eval_config)
    conv = cfg.get("steps", "primary")
    steps_mm = _steps(generator.spec.steps, conv)
    rect = field_rect(cfg["field"], steps_mm)
    fld_mask = field_rect_mask(rect)

    # Precompute static distance map when mask does not vary per frame (all non-REJ configs)
    is_rej = cfg.get("rejection") is not None
    static_dist_map: np.ndarray | None = None

    rows: list[dict[str, Any]] = []
    acc = SpatialErrorAccumulator(hw=(128, 128))
    kept_fields: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    t0 = time.perf_counter()

    n = len(frames)
    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        chunk = frames[start : start + bs]
        packets = []
        for fr in chunk:
            piv_obs = generator.observe(
                eval_config,
                np.asarray(fr["vi_phys"], dtype=np.float64),
                np.asarray(fr["vj_phys"], dtype=np.float64),
                sim_id=str(fr["sim_id"]),
                frame_index=int(fr["frame_index"]),
                role=obs_role,
                replica=int(replica),
                return_intermediates=True,
            )
            packets.append({"piv_obs": piv_obs, "frame": fr})

        preds_K = np.asarray(forward_fn(packets), dtype=np.float64)
        if preds_K.shape != (len(packets), 128, 128):
            raise PivEvalError(f"forward_fn returned shape {preds_K.shape}, expected {(len(packets), 128, 128)}")

        for j, pkt in enumerate(packets):
            fr = pkt["frame"]
            po = pkt["piv_obs"]
            inter = po.intermediates or {}
            obs_m = np.asarray(inter["observed"], dtype=bool)
            val_m = np.asarray(inter["valid"], dtype=bool)
            rej_m = np.asarray(inter["rejected"], dtype=bool)
            sub_m = rej_m if cfg.get("level") == "sub" else None

            if not is_rej:
                if static_dist_map is None:
                    static_dist_map = distance_to_valid_vector_mm(obs_m, steps_mm=steps_mm)
                dmap = static_dist_map
            else:
                dmap = None

            pred_K = preds_K[j]
            true_K = np.asarray(fr["true_T_K"], dtype=np.float64)
            sim_id = str(fr["sim_id"])
            fidx = int(fr["frame_index"])

            row = evaluate_piv_frame(
                pred_K,
                true_K,
                train_sigma_T=train_sigma_T,
                sim_id=sim_id,
                split=str(fr.get("split", role)),
                role=role,
                frame_index=fidx,
                observed_mask=obs_m,
                valid_mask=val_m,
                field_mask=fld_mask,
                rejected_mask=rej_m if is_rej else None,
                substituted_mask=sub_m,
                velocity_region_masks=fr.get("vel_region_masks"),
                steps_mm=steps_mm,
                dist_map_mm=dmap,
            )
            m = po.meta
            row.update(
                {
                    "eval_config": eval_config,
                    "eval_field": str(cfg["field"]),
                    "eval_W_mm": float(cfg["W_mm"]),
                    "eval_response": str(cfg["response"]),
                    "eval_noise": str(cfg["noise"]),
                    "eval_sigma_px": float(cfg.get("sigma_px", 0.0)),
                    "eval_step_convention": str(conv),
                    "eval_rejection": str(cfg.get("rejection") or "none"),
                    "eval_level": str(cfg.get("level") or "none"),
                    "eval_role_category": str(cfg.get("role", "main")),
                    "replica": int(replica),
                    "obs_hash": po.obs_hash,
                    "observed_mask_hash": m["observed_mask_hash"],
                    "valid_mask_hash": m["valid_mask_hash"],
                    "sigma_u_m_s": float(m["sigma_u_m_s"]),
                    "v_ref_m_s": float(m["v_ref_m_s"]),
                    "n_rejected": int(m["n_rejected"]),
                    "n_substituted": int(m["n_substituted"]),
                    "spec_sha256": str(m["spec_sha256"]),
                }
            )
            row.update(run_tags)
            rows.append(row)

            # Accumulate spatial error maps on replica 0 (or all replicas when caller merges)
            acc.update(sim_id, pred_K, true_K)

            key = (sim_id, fidx)
            keep_this = (keep_fields_keys is not None and key in keep_fields_keys and int(replica) == 0) or (
                keep_all_val_rep0 and role == "val_eval" and int(replica) == 0
            )
            if keep_this:
                kept_fields[key] = {
                    "pred_T_K": pred_K.astype(np.float32),
                    "true_T_K": true_K.astype(np.float32),
                    "error_signed_K": (pred_K - true_K).astype(np.float32),
                    "observed_mask": obs_m,
                    "valid_mask": val_m,
                    "field_mask": fld_mask,
                    "rejected_mask": rej_m if is_rej else np.zeros((0, 0), dtype=bool),
                    "rmse_K_csv": np.asarray(row["rmse_K"], dtype=np.float64),
                }

    return {
        "rows": rows,
        "wall_s": time.perf_counter() - t0,
        "n_frames": len(rows),
        "spatial_acc": acc,
        "kept_fields": kept_fields,
    }


def save_selected_fields_bundle(
    dest_dir: Path,
    *,
    fields_by_config_role: Mapping[tuple[str, str], Mapping[tuple[str, int], Mapping[str, np.ndarray]]],
    identity: Mapping[str, Any],
    filename: str = "fields_selected_best.npz",
    manifest_name: str = "fields_manifest.json",
) -> Path:
    """Persist selected full fields in a single compressed archive + hash-verified manifest."""
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for (cfg, role), fmap in sorted(fields_by_config_role.items()):
        for (cond, fidx), fd in sorted(fmap.items()):
            key = f"{cfg}|{role}|{cond}|{int(fidx)}"
            pred = np.asarray(fd["pred_T_K"], dtype=np.float32)
            true = np.asarray(fd["true_T_K"], dtype=np.float32)
            err = (pred - true).astype(np.float32)
            rmse_from_f = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
            rmse_csv = float(fd["rmse_K_csv"])
            if abs(rmse_from_f - rmse_csv) > 5e-3:
                raise PivEvalError(f"field RMSE {rmse_from_f:.4f} != CSV RMSE {rmse_csv:.4f} for {key}")
            arrays[f"{key}|pred_T_K"] = pred
            arrays[f"{key}|true_T_K"] = true
            arrays[f"{key}|error_signed_K"] = err
            arrays[f"{key}|observed_mask"] = np.asarray(fd["observed_mask"], dtype=bool)
            arrays[f"{key}|valid_mask"] = np.asarray(fd["valid_mask"], dtype=bool)
            arrays[f"{key}|field_mask"] = np.asarray(fd["field_mask"], dtype=bool)
            arrays[f"{key}|rejected_mask"] = np.asarray(fd["rejected_mask"], dtype=bool)
            entries.append(
                {
                    "key": key,
                    "eval_config": cfg,
                    "role": role,
                    "condition": cond,
                    "frame_index": int(fidx),
                    "replica": 0,
                    "rmse_K_from_fields": rmse_from_f,
                    "rmse_K_per_frame_csv": rmse_csv,
                }
            )
    npz_path = dest_dir / filename
    atomic_savez_compressed(npz_path, arrays)
    manifest = {
        "format": "piv_selected_fields_v1",
        "file": filename,
        "file_sha256": sha256_file(npz_path),
        "identity": dict(identity),
        "entries": entries,
    }
    atomic_write_json(dest_dir / manifest_name, manifest)
    return npz_path


def run_resumable_checkpoint_eval(
    *,
    out_kind_dir: Path,
    frames_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    generator: PivObservationGenerator,
    eval_configs: Sequence[str],
    roles: Sequence[str],
    replicas: Sequence[int],
    forward_fn: Callable[[Sequence[Mapping[str, Any]]], np.ndarray],
    train_sigma_T: float,
    run_tags: Mapping[str, Any],
    batch_size: int = 8,
    resume: bool = True,
    representative_frames: Mapping[str, Sequence[Mapping[str, Any]]] = DEFAULT_REPRESENTATIVE_FRAMES,
    save_fields: bool = True,
    keep_all_val_rep0_for_configs: Sequence[str] = (),
    gate_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all ``(eval_config, role, replica)`` units for one checkpoint, resumably."""
    out_kind_dir = Path(out_kind_dir)
    out_kind_dir.mkdir(parents=True, exist_ok=True)
    rid = str(run_tags["run_id"])
    kind = str(run_tags["checkpoint_kind"])
    wsha = str(run_tags.get("checkpoint_weights_sha256", "unknown"))
    spec_sha = generator.spec.sha256

    rep_keys_by_role: dict[str, set[tuple[str, int]]] = {}
    for role in roles:
        rep_keys_by_role[role] = {
            (str(it["condition"]), int(it["frame_index"]))
            for it in representative_frames.get(role, [])
        }

    costs: list[dict[str, Any]] = []
    skipped = 0
    computed = 0
    fields_bundle: dict[tuple[str, str], dict[tuple[str, int], dict[str, np.ndarray]]] = {}

    for cfg in eval_configs:
        for role in roles:
            _check_role_allowed(role)
            fseq = frames_by_role[role]
            for rep in replicas:
                csv_p = out_kind_dir / "per_frame" / cfg / role / f"rep{int(rep)}.csv"
                sig = unit_signature(
                    run_id=rid,
                    checkpoint_kind=kind,
                    checkpoint_weights_sha256=wsha,
                    eval_config=cfg,
                    role=role,
                    replica=int(rep),
                    spec_sha256=spec_sha,
                    n_expected_frames=len(fseq),
                )
                spatial_p = out_kind_dir / "spatial" / cfg / f"{role}_rep{int(rep)}.npz"
                if resume and is_unit_complete(csv_p, sig) and (int(rep) != 0 or spatial_p.is_file()):
                    skipped += 1
                    continue

                res = evaluate_piv_unit(
                    frames=fseq,
                    generator=generator,
                    eval_config=cfg,
                    role=role,
                    replica=int(rep),
                    forward_fn=forward_fn,
                    train_sigma_T=train_sigma_T,
                    run_tags=run_tags,
                    batch_size=batch_size,
                    keep_fields_keys=rep_keys_by_role[role] if save_fields else None,
                    keep_all_val_rep0=(save_fields and cfg in keep_all_val_rep0_for_configs),
                )
                df_unit = pd.DataFrame(res["rows"])
                atomic_write_csv(df_unit, csv_p)
                atomic_write_json(
                    csv_p.with_suffix(".ok.json"),
                    {
                        "ok": True,
                        "signature": sig,
                        "csv_sha256": sha256_file(csv_p),
                        "n_rows": len(df_unit),
                        "wall_s": res["wall_s"],
                    },
                )
                # Save compact spatial accumulator on replica 0 (avoid redundant spatial copies)
                if int(rep) == 0:
                    atomic_savez_compressed(spatial_p, res["spatial_acc"].to_npz_dict())
                if res["kept_fields"]:
                    fields_bundle[(cfg, role)] = res["kept_fields"]
                costs.append(
                    {
                        "run_id": rid,
                        "checkpoint_kind": kind,
                        "eval_config": cfg,
                        "role": role,
                        "replica": int(rep),
                        "n_frames": res["n_frames"],
                        "wall_s": res["wall_s"],
                    }
                )
                computed += 1

    # Merge or update setting_costs.csv without duplicating rows
    costs_p = out_kind_dir / "setting_costs.csv"
    new_costs = pd.DataFrame(costs)
    if costs_p.is_file() and not new_costs.empty:
        old_costs = pd.read_csv(costs_p)
        key_cols = ["run_id", "checkpoint_kind", "eval_config", "role", "replica"]
        merged = pd.concat([old_costs, new_costs], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
        atomic_write_csv(merged, costs_p)
    elif not new_costs.empty:
        atomic_write_csv(new_costs, costs_p)

    if save_fields and fields_bundle:
        save_selected_fields_bundle(
            out_kind_dir,
            fields_by_config_role=fields_bundle,
            identity=run_tags,
            filename=f"fields_selected_{kind}.npz",
            manifest_name="fields_manifest.json",
        )

    gate_payload = dict(gate_result) if gate_result is not None else {"all_pass": True, "note": "gate_verified_externally"}
    atomic_write_json(out_kind_dir / "gate.json", gate_payload)
    atomic_write_json(
        out_kind_dir / "provenance.json",
        {
            **dict(run_tags),
            "spec_sha256": spec_sha,
            "eval_configs": list(eval_configs),
            "roles": list(roles),
            "replicas": [int(r) for r in replicas],
            "units_computed": computed,
            "units_skipped_resume": skipped,
        },
    )
    atomic_write_json(
        out_kind_dir / "COMPLETE.json",
        {
            "ok": bool(gate_payload.get("all_pass", False)),
            "units_computed": computed,
            "units_skipped_resume": skipped,
            "gate_all_pass": bool(gate_payload.get("all_pass", False)),
        },
    )
    return {"computed": computed, "skipped": skipped, "out_kind_dir": str(out_kind_dir)}


__all__ = [
    "PivEvalError",
    "atomic_savez_compressed",
    "atomic_write_csv",
    "evaluate_piv_unit",
    "is_unit_complete",
    "run_resumable_checkpoint_eval",
    "save_selected_fields_bundle",
    "unit_signature",
]
