"""Technical package for the finished Opus O5 evaluation.

Reads the per-frame outputs of the eight runs, checks completeness and
provenance, and builds the tables the Neon review needs. It trains nothing,
selects nothing, and ranks nothing. Deltas are descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.evaluation.opus_o5_plan import (
    CHECKPOINTS,
    COVERAGES,
    REPLICAS,
    REQUIRED_RUN_IDS,
    ROLE_N_FRAMES,
    ROLES,
)

CORE_METRICS = ("rmse_K", "mae_K", "mean_signed_error_K", "abs_bias_K", "centered_rmse_K")
REGIONAL_METRICS = (
    "high_rmse_K",
    "high_mae_K",
    "non_high_rmse_K",
    "peak_rmse_K",
    "jet_rmse_K",
    "high_iou",
    "high_rel_area_error",
    "frac_omega_high_true",
    "frac_omega_high_pred",
)
ALL_METRICS = CORE_METRICS + REGIONAL_METRICS
EXPECTED_CONDITIONS = {"train_fit": 20, "seen_train_eval": 20, "val_eval": 3}
LOOP_TOLERANCE_K = 0.05
ID_COLUMNS = (
    "run_id",
    "checkpoint_kind",
    "role",
    "site_fraction",
    "mask_replica_id",
    "sim_id",
    "frame_index",
    "mask_hash",
    "weights_sha256",
    "file_sha256",
    "epoch",
    "criterion",
    "pool_id",
    "model_seed",
    "training_git_sha",
    "eval_git_sha",
    "shared_forward_from",
)
RUN_ORDER = tuple(REQUIRED_RUN_IDS)


def run_table(matrix: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in matrix["runs"]:
        rows.append(
            {
                "run_id": entry["run_id"],
                "pool_id": entry["pool_id"],
                "p_conditioning": bool(entry["p_conditioning"]),
                "model_seed": int(entry["model_seed"]),
                "training_git_sha": entry["training_git_sha"],
                "training_job_id": int(entry["job_id"]),
            }
        )
    return pd.DataFrame(rows)


def _cov(value: float) -> str:
    return f"{float(value):.2f}"


def load_and_check(
    out_root: Path,
    matrix: Mapping[str, Any],
    *,
    allowed_eval_shas: Sequence[str],
    run_ids: Sequence[str] = RUN_ORDER,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Return (frames, per-run completeness rows, issues). Empty issues means ready."""
    issues: list[str] = []
    entries = {row["run_id"]: row for row in matrix["runs"]}
    shared = matrix["shared_inputs"]
    reference_frames: dict[str, list[tuple[str, int]]] = {}
    frames: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    for run_id in run_ids:
        entry = entries[run_id]
        run_dir = out_root / run_id
        info: dict[str, Any] = {"run_id": run_id, "n_csv": 0, "complete_json": False}
        marker = run_dir / "COMPLETE.json"
        if not marker.exists():
            issues.append(f"{run_id}: COMPLETE.json missing")
            runs.append(info)
            continue
        complete = json.loads(marker.read_text())
        info["complete_json"] = bool(complete.get("ok") is True)
        if complete.get("ok") is not True or int(complete.get("n_settings", -1)) != 81:
            issues.append(f"{run_id}: COMPLETE.json does not report 81 settings")
        prov_path = run_dir / "provenance.json"
        if not prov_path.exists():
            issues.append(f"{run_id}: provenance.json missing")
            runs.append(info)
            continue
        prov = json.loads(prov_path.read_text())
        info["eval_git_sha"] = prov.get("eval_git_sha")
        if prov.get("smoke"):
            issues.append(f"{run_id}: provenance marks a smoke run")
        if prov.get("eval_git_sha") not in set(allowed_eval_shas):
            issues.append(f"{run_id}: eval_git_sha {prov.get('eval_git_sha')} not in the declared set")
        if prov.get("training_git_sha") != entry["training_git_sha"]:
            issues.append(f"{run_id}: training_git_sha differs from the matrix")
        consumed = prov.get("consumed", {})
        pool = str(entry["pool_id"])
        want_agg = shared["c149_cache_aggregate_sha256" if pool == "C149" else "historical_cache_aggregate_sha256"]
        want_id = shared["c149_cache_id" if pool == "C149" else "historical_cache_id"]
        for key, want in (
            ("cache_aggregate_sha256", want_agg),
            ("cache_id", want_id),
            ("stats_sha256", shared["stats_sha256"]),
            ("eval_mask_manifest_file_sha256", shared["eval_mask_manifest_file_sha256"]),
            ("eval_mask_aggregate_payload_sha256", shared["eval_mask_aggregate_payload_sha256"]),
        ):
            if consumed.get(key) != want:
                issues.append(f"{run_id}: consumed {key} differs from the matrix")
        run_frames: list[pd.DataFrame] = []
        for name in CHECKPOINTS:
            spec = entry["checkpoints"][name]
            for coverage in COVERAGES:
                for role in ROLES:
                    for replica in REPLICAS:
                        path = run_dir / "per_frame" / name / f"cov{_cov(coverage)}" / role / f"rep{replica}.csv"
                        if not path.exists():
                            issues.append(f"{run_id}: missing {path.relative_to(out_root)}")
                            continue
                        header = pd.read_csv(path, nrows=0).columns
                        cols = [c for c in (*ID_COLUMNS, *ALL_METRICS) if c in header]
                        frame = pd.read_csv(path, usecols=cols)
                        info["n_csv"] += 1
                        tag = f"{run_id}/{name}/cov{_cov(coverage)}/{role}/rep{replica}"
                        if len(frame) != ROLE_N_FRAMES[role]:
                            issues.append(f"{tag}: {len(frame)} rows, expected {ROLE_N_FRAMES[role]}")
                        missing = [c for c in ALL_METRICS if c not in frame.columns]
                        if missing:
                            issues.append(f"{tag}: metric columns missing {missing}")
                            continue
                        # Core metrics must be finite. Regional metrics may be undefined on a
                        # frame (for example an empty peak region); those are counted, not hidden.
                        if not np.isfinite(frame[list(CORE_METRICS)].to_numpy(dtype=float)).all():
                            issues.append(f"{tag}: non-finite core metric")
                        if np.isinf(frame[list(REGIONAL_METRICS)].to_numpy(dtype=float)).any():
                            issues.append(f"{tag}: infinite regional metric")
                        for column, want in (
                            ("checkpoint_kind", name),
                            ("run_id", run_id),
                            ("weights_sha256", spec["weights_sha256"]),
                            ("file_sha256", spec["file_sha256"]),
                            ("criterion", spec["criterion"]),
                            ("training_git_sha", entry["training_git_sha"]),
                        ):
                            if set(frame[column]) != {want}:
                                issues.append(f"{tag}: {column} differs from the matrix")
                        if set(frame["epoch"]) != {int(spec["epoch"])}:
                            issues.append(f"{tag}: epoch differs from the matrix")
                        if set(frame["site_fraction"].round(2)) != {round(float(coverage), 2)}:
                            issues.append(f"{tag}: site_fraction differs from the path")
                        if set(frame["mask_replica_id"]) != {int(replica)}:
                            issues.append(f"{tag}: mask_replica_id differs from the path")
                        if set(frame["eval_git_sha"]) != {prov.get("eval_git_sha")}:
                            issues.append(f"{tag}: eval_git_sha differs from provenance")
                        key = sorted(zip(frame["sim_id"].astype(str), frame["frame_index"].astype(int)))
                        if role not in reference_frames:
                            reference_frames[role] = key
                            n_cond = frame["sim_id"].nunique()
                            if n_cond != EXPECTED_CONDITIONS[role]:
                                issues.append(f"{tag}: {n_cond} conditions, expected {EXPECTED_CONDITIONS[role]}")
                        elif key != reference_frames[role]:
                            issues.append(f"{tag}: frame set differs from the first {role} file")
                        run_frames.append(frame)
        if run_frames:
            frames.append(pd.concat(run_frames, ignore_index=True))
        field_dir = run_dir / "fields"
        for name in CHECKPOINTS:
            if not (field_dir / f"val_frame1000_{name}_cov05_rep0.npz").exists():
                issues.append(f"{run_id}: field file for {name} missing")
        if not (run_dir / "setting_costs.csv").exists():
            issues.append(f"{run_id}: setting_costs.csv missing")
        if info["n_csv"] != 81:
            issues.append(f"{run_id}: {info['n_csv']} CSV files, expected 81")
        runs.append(info)
    if not frames:
        return pd.DataFrame(), runs, issues
    table = pd.concat(frames, ignore_index=True)
    table["coverage"] = table["site_fraction"].round(2)
    table["replica"] = table["mask_replica_id"].astype(int)
    issues += _cross_checks(table, entries)
    return table, runs, issues


def _cross_checks(table: pd.DataFrame, entries: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    key = ["role", "sim_id", "frame_index", "coverage", "replica"]
    per_key = table.groupby(key)["mask_hash"].nunique()
    bad = int((per_key != 1).sum())
    if bad:
        issues.append(f"{bad} (role, frame, coverage, replica) keys have different mask hashes across runs or checkpoints")
    for run_id, entry in entries.items():
        for name in CHECKPOINTS:
            source = entry["checkpoints"][name].get("reuse_forward_from")
            if not source:
                continue
            a = table[(table.run_id == run_id) & (table.checkpoint_kind == name)]
            b = table[(table.run_id == run_id) & (table.checkpoint_kind == source)]
            if a.empty or b.empty:
                continue
            sort_key = ["role", "coverage", "replica", "sim_id", "frame_index"]
            a = a.sort_values(sort_key)
            b = b.sort_values(sort_key)
            if not np.array_equal(a[list(ALL_METRICS)].to_numpy(), b[list(ALL_METRICS)].to_numpy(), equal_nan=True):
                issues.append(f"{run_id}: {name} metrics differ from its shared source {source}")
            if set(a["shared_forward_from"].fillna("")) != {source}:
                issues.append(f"{run_id}: {name} does not record shared_forward_from={source}")
    return issues


def _meta(table: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    return table.merge(runs[["run_id", "p_conditioning"]], on="run_id", how="left")


def build_tables(table: pd.DataFrame, runs: pd.DataFrame, matrix: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Frame -> condition -> replica -> run. Mean at every step, no ranking."""
    table = _meta(table, runs)
    metrics = list(ALL_METRICS)
    ident = ["run_id", "pool_id", "p_conditioning", "model_seed", "checkpoint_kind", "epoch", "role", "coverage"]
    cond_rep = (
        table.groupby(ident + ["replica", "sim_id"], dropna=False)[metrics].mean().reset_index()
    )
    cond_rep["n_frames"] = (
        table.groupby(ident + ["replica", "sim_id"], dropna=False)["frame_index"].nunique().to_numpy()
    )
    per_condition = cond_rep.groupby(ident + ["sim_id"], dropna=False)[metrics].mean().reset_index()
    spread = (
        cond_rep.groupby(ident + ["sim_id"], dropna=False)["rmse_K"].agg(rmse_rep_min_K="min", rmse_rep_max_K="max").reset_index()
    )
    per_condition = per_condition.merge(spread, on=ident + ["sim_id"])
    glob_rep = cond_rep.groupby(ident + ["replica"], dropna=False)[metrics].mean().reset_index()
    glob = glob_rep.groupby(ident, dropna=False)[metrics].mean().reset_index()
    rmse_rep = glob_rep.groupby(ident, dropna=False)["rmse_K"].agg(rmse_rep_min_K="min", rmse_rep_max_K="max").reset_index()
    glob = glob.merge(rmse_rep, on=ident)
    glob["n_conditions"] = cond_rep.groupby(ident, dropna=False)["sim_id"].nunique().to_numpy()
    order = {name: i for i, name in enumerate(RUN_ORDER)}
    for frame in (glob, per_condition, glob_rep):
        frame.sort_values(
            by=["run_id", "checkpoint_kind", "role", "coverage"],
            key=lambda s: s.map(order) if s.name == "run_id" else s,
            inplace=True,
        )
        frame.reset_index(drop=True, inplace=True)
    regional_cols = ident + list(REGIONAL_METRICS) + ["rmse_K"]
    return {
        "global_by_setting": glob,
        "global_by_replica": glob_rep,
        "per_condition": per_condition,
        "regional_by_setting": glob[regional_cols],
        "regional_by_condition": per_condition[regional_cols + ["sim_id"]],
    }


def loop_comparison(
    glob_rep: pd.DataFrame,
    loop: pd.DataFrame,
    *,
    tolerance_k: float = LOOP_TOLERANCE_K,
) -> pd.DataFrame:
    """O5 5% against the training-loop value at the same epoch, role and replicas.

    Loop train and seen are replica 0; loop VAL is the mean of the three masks.
    """
    rows = []
    sel = glob_rep[np.isclose(glob_rep["coverage"], 0.05)]
    mapping = (
        ("train_fit", "train_rep0_rmse_K", (0,)),
        ("seen_train_eval", "seen_rep0_rmse_K", (0,)),
        ("val_eval", "val_mean3_rmse_K", (0, 1, 2)),
    )
    for (run_id, name, epoch), group in sel.groupby(["run_id", "checkpoint_kind", "epoch"]):
        for role, loop_col, reps in mapping:
            part = group[(group["role"] == role) & (group["replica"].isin(reps))]
            if part["replica"].nunique() != len(reps):
                continue
            o5 = float(part["rmse_K"].mean())
            hit = loop[(loop["run_id"] == run_id) & (loop["epoch"] == int(epoch))]
            loop_value = float(hit[loop_col].iloc[0]) if len(hit) == 1 else float("nan")
            rows.append(
                {
                    "run_id": run_id,
                    "checkpoint_kind": name,
                    "epoch": int(epoch),
                    "role": role,
                    "replicas": "+".join(str(r) for r in reps),
                    "o5_rmse_K": o5,
                    "loop_rmse_K": loop_value,
                    "delta_K": o5 - loop_value,
                    "tolerance_K": tolerance_k,
                    "within_tolerance": bool(abs(o5 - loop_value) <= tolerance_k),
                    "loop_source": "training_loop_not_o5",
                }
            )
    return pd.DataFrame(rows)


def comparison_pairs(runs: pd.DataFrame) -> list[dict[str, Any]]:
    """Fixed contrasts. a minus b. A38 seed 0 is the matched reference for the data effect."""
    index = {
        (r.pool_id, bool(r.p_conditioning), int(r.model_seed)): r.run_id for r in runs.itertuples()
    }
    pairs: list[dict[str, Any]] = []
    for p in (True, False):
        for a, b in (("B19", "A38"), ("C149", "A38"), ("C149", "B19")):
            pairs.append(
                {
                    "family": "data",
                    "label": f"{a}-{b} seed0 p_{'on' if p else 'off'}",
                    "a": index[(a, p, 0)],
                    "b": index[(b, p, 0)],
                    "same_checkpoint": True,
                }
            )
        pairs.append(
            {
                "family": "seed",
                "label": f"A38 seed1-seed0 p_{'on' if p else 'off'}",
                "a": index[("A38", p, 1)],
                "b": index[("A38", p, 0)],
                "same_checkpoint": True,
            }
        )
    for pool, seed in (("B19", 0), ("A38", 0), ("A38", 1), ("C149", 0)):
        pairs.append(
            {
                "family": "p",
                "label": f"{pool} seed{seed} on-off",
                "a": index[(pool, True, seed)],
                "b": index[(pool, False, seed)],
                "same_checkpoint": True,
            }
        )
    return pairs


def _paired(per_condition: pd.DataFrame, a: str, b: str, ckpt_a: str, ckpt_b: str) -> pd.DataFrame:
    keys = ["role", "coverage", "sim_id"]
    left = per_condition[(per_condition.run_id == a) & (per_condition.checkpoint_kind == ckpt_a)]
    right = per_condition[(per_condition.run_id == b) & (per_condition.checkpoint_kind == ckpt_b)]
    long_a = left.melt(id_vars=keys, value_vars=list(ALL_METRICS), var_name="metric", value_name="a_value")
    long_b = right.melt(id_vars=keys, value_vars=list(ALL_METRICS), var_name="metric", value_name="b_value")
    merged = long_a.merge(long_b, on=keys + ["metric"], how="inner")
    merged["delta_a_minus_b"] = merged["a_value"] - merged["b_value"]
    return merged


def comparisons(per_condition: pd.DataFrame, runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Descriptive paired deltas. No test statistic, no winner."""
    detail: list[pd.DataFrame] = []
    for pair in comparison_pairs(runs):
        for ckpt in CHECKPOINTS:
            frame = _paired(per_condition, pair["a"], pair["b"], ckpt, ckpt)
            frame.insert(0, "checkpoint_a", ckpt)
            frame.insert(1, "checkpoint_b", ckpt)
            frame.insert(0, "run_b", pair["b"])
            frame.insert(0, "run_a", pair["a"])
            frame.insert(0, "label", pair["label"])
            frame.insert(0, "family", pair["family"])
            detail.append(frame)
    for run_id in runs["run_id"]:
        for a, b in (("last", "best"), ("best_seen_diagnostic", "best"), ("last", "best_seen_diagnostic")):
            frame = _paired(per_condition, run_id, run_id, a, b)
            frame.insert(0, "checkpoint_b", b)
            frame.insert(0, "checkpoint_a", a)
            frame.insert(0, "run_b", run_id)
            frame.insert(0, "run_a", run_id)
            frame.insert(0, "label", f"{run_id} {a}-{b}")
            frame.insert(0, "family", "selection")
            detail.append(frame)
    by_condition = pd.concat(detail, ignore_index=True)
    group = ["family", "label", "run_a", "run_b", "checkpoint_a", "checkpoint_b", "role", "coverage", "metric"]
    summary = (
        by_condition.groupby(group, dropna=False)["delta_a_minus_b"]
        .agg(
            mean_delta="mean",
            min_delta="min",
            max_delta="max",
            n_conditions="count",
            n_a_lower=lambda s: int((s < 0).sum()),
            n_a_higher=lambda s: int((s > 0).sum()),
        )
        .reset_index()
    )
    return {"compare_summary": summary, "compare_by_condition": by_condition}


def setting_costs(out_root: Path, run_ids: Sequence[str] = RUN_ORDER) -> pd.DataFrame:
    rows = []
    for run_id in run_ids:
        path = out_root / run_id / "setting_costs.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        wall = pd.to_numeric(frame.get("wall_s"), errors="coerce")
        prov = json.loads((out_root / run_id / "provenance.json").read_text())
        rows.append(
            {
                "run_id": run_id,
                "n_settings": len(frame),
                "n_forward_settings": int(wall.notna().sum()),
                "n_shared_settings": int(frame["shared_forward_from"].fillna("").ne("").sum()),
                "forward_wall_s": float(wall.sum()),
                "max_memory_bytes": int(prov.get("max_memory_bytes", -1)),
            }
        )
    return pd.DataFrame(rows)


def nan_report(table: pd.DataFrame) -> pd.DataFrame:
    """Frames where a regional metric is undefined (NaN). Means skip them and say so."""
    rows = []
    keys = ["run_id", "checkpoint_kind", "role", "coverage"]
    for metric in ALL_METRICS:
        bad = table[table[metric].isna()]
        if bad.empty:
            continue
        counts = bad.groupby(keys).size().rename("n_nan_frames").reset_index()
        counts.insert(4, "metric", metric)
        rows.append(counts)
    if not rows:
        return pd.DataFrame(columns=keys + ["metric", "n_nan_frames"])
    return pd.concat(rows, ignore_index=True)
