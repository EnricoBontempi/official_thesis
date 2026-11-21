"""Aggregation of the finished S7 evaluation (12 runs, both families). No ranking, no winner.

Reads the per-frame CSVs of ``scripts/38_s7_eval.py``, refuses an incomplete or inconsistent
matrix, and builds descriptive tables: frame -> condition -> mask replica -> run. Differences are
paired and descriptive; no test statistic and no selection rule is applied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from combustion_recon.evaluation.opus_o5_plan import (
    CHECKPOINTS,
    COVERAGES,
    REPLICAS,
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
N_SETTINGS = len(CHECKPOINTS) * len(COVERAGES) * len(ROLES) * len(REPLICAS)  # 81
ID_COLUMNS = (
    "run_id", "family", "p_conditioning", "checkpoint_kind", "role", "site_fraction", "mask_replica_id",
    "sim_id", "frame_index", "mask_hash", "weights_sha256", "file_sha256", "epoch", "criterion",
    "pool_id", "model_seed", "training_git_sha", "eval_git_sha", "shared_forward_from",
)
IDENT = ["run_id", "family", "p_conditioning", "model_seed", "checkpoint_kind", "epoch", "role", "coverage"]
PAYLOAD_HASH_KEY = {0.01: "hash_1pct", 0.05: "hash_5pct", 0.2: "hash_20pct"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cov(value: float) -> str:
    return f"{float(value):.2f}"


def run_table(matrix: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": e["run_id"],
                "family": e["family"],
                "p_conditioning": bool(e["p_conditioning"]),
                "model_seed": int(e["model_seed"]),
                "reused_not_relaunched": bool(e["reused_not_relaunched"]),
                "training_git_sha": e["training_git_sha"],
                "status": e["status"],
                "epochs_completed": int(e["epochs_completed"]),
                "extended_past_initial_tranche": bool(e["extended_past_initial_tranche"]),
            }
            for e in matrix["runs"]
        ]
    )


def payload_mask_hashes(mask_root: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    """(role, sim_id, frame_index, replica) -> {hash_1pct,...} from the verified payload."""
    out: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for role in ROLES:
        data = np.load(Path(mask_root) / f"{role}_nested_sites.npz", allow_pickle=True)
        for i, (s, f, r) in enumerate(zip(data["sim_id"].tolist(), data["frame_index"], data["replica_id"])):
            out[(role, str(s), int(f), int(r))] = {k: str(data[k][i]) for k in PAYLOAD_HASH_KEY.values()}
    return out


def load_and_check(
    eval_root: Path,
    matrix: Mapping[str, Any],
    *,
    matrix_sha256: str,
    allowed_eval_shas: Sequence[str],
    payload_hashes: Mapping[tuple, Mapping[str, str]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Return (frames, per-run completeness, issues). Empty issues means the matrix is valid."""
    issues: list[str] = []
    shared = matrix["shared_inputs"]
    frames: list[pd.DataFrame] = []
    runs: list[dict[str, Any]] = []
    reference: dict[str, list[tuple[str, int]]] = {}
    for entry in matrix["runs"]:
        rid = entry["run_id"]
        run_dir = Path(eval_root) / rid
        info: dict[str, Any] = {"run_id": rid, "family": entry["family"], "p_conditioning": entry["p_conditioning"],
                                "model_seed": entry["model_seed"], "complete_json": False, "gate_all_pass": False,
                                "n_csv": 0, "gate_max_abs_delta_K": None, "eval_git_sha": None}
        runs.append(info)
        marker = run_dir / "COMPLETE.json"
        if not marker.exists():
            issues.append(f"{rid}: COMPLETE.json missing" + (" (GATE_FAILED.json present)" if (run_dir / "GATE_FAILED.json").exists() else ""))
            continue
        done = json.loads(marker.read_text())
        info["complete_json"] = done.get("ok") is True
        gate = done.get("gate") or {}
        info["gate_all_pass"] = bool(gate.get("all_pass"))
        info["gate_max_abs_delta_K"] = gate.get("max_abs_delta_K")
        if done.get("ok") is not True or int(done.get("n_settings", -1)) != N_SETTINGS:
            issues.append(f"{rid}: COMPLETE.json does not report {N_SETTINGS} settings")
        if not info["gate_all_pass"] or gate.get("mode") != "full" or int(gate.get("n_comparisons", -1)) != 18:
            issues.append(f"{rid}: gate not a full 18/18 pass")
        prov_path = run_dir / "provenance.json"
        if not prov_path.exists():
            issues.append(f"{rid}: provenance.json missing")
            continue
        prov = json.loads(prov_path.read_text())
        info["eval_git_sha"] = prov.get("eval_git_sha")
        if prov.get("mode") != "full" or prov.get("smoke") or not prov.get("scientific"):
            issues.append(f"{rid}: provenance is not a scientific full run")
        if prov.get("matrix_sha256") != matrix_sha256:
            issues.append(f"{rid}: provenance matrix_sha256 differs from the frozen matrix")
        if prov.get("eval_git_sha") not in set(allowed_eval_shas):
            issues.append(f"{rid}: eval_git_sha {prov.get('eval_git_sha')} not in the declared set")
        if prov.get("training_git_sha") != entry["training_git_sha"]:
            issues.append(f"{rid}: training_git_sha differs from the matrix")
        consumed = prov.get("consumed", {})
        for key, want in (
            ("cache_aggregate_sha256", shared["c149_cache_aggregate_sha256"]),
            ("cache_id", shared["c149_cache_id"]),
            ("stats_sha256", shared["stats_sha256"]),
            ("eval_mask_manifest_file_sha256", shared["eval_mask_manifest_file_sha256"]),
            ("eval_mask_aggregate_payload_sha256", shared["eval_mask_aggregate_payload_sha256"]),
        ):
            if consumed.get(key) != want:
                issues.append(f"{rid}: consumed {key} differs from the matrix")
        run_frames: list[pd.DataFrame] = []
        for name in CHECKPOINTS:
            spec = entry["checkpoints"][name]
            source = spec.get("reuse_forward_from") or ""
            for coverage in COVERAGES:
                for role in ROLES:
                    for replica in REPLICAS:
                        path = run_dir / "per_frame" / name / f"cov{_cov(coverage)}" / role / f"rep{replica}.csv"
                        tag = f"{rid}/{name}/cov{_cov(coverage)}/{role}/rep{replica}"
                        if not path.exists():
                            issues.append(f"{tag}: file missing")
                            continue
                        header = pd.read_csv(path, nrows=0).columns
                        frame = pd.read_csv(path, usecols=[c for c in (*ID_COLUMNS, *ALL_METRICS) if c in header])
                        info["n_csv"] += 1
                        if len(frame) != ROLE_N_FRAMES[role]:
                            issues.append(f"{tag}: {len(frame)} rows, expected {ROLE_N_FRAMES[role]}")
                        missing = [c for c in (*ID_COLUMNS, *ALL_METRICS) if c not in frame.columns]
                        if missing:
                            issues.append(f"{tag}: columns missing {missing}")
                            continue
                        if not np.isfinite(frame[list(CORE_METRICS)].to_numpy(dtype=float)).all():
                            issues.append(f"{tag}: non-finite core metric")
                        if np.isinf(frame[list(REGIONAL_METRICS)].to_numpy(dtype=float)).any():
                            issues.append(f"{tag}: infinite regional metric")
                        expect = {
                            "run_id": rid,
                            "family": entry["family"],
                            "checkpoint_kind": name,
                            "weights_sha256": spec["weights_sha256"],
                            "file_sha256": spec["file_sha256"],
                            "criterion": spec["criterion"],
                            "training_git_sha": entry["training_git_sha"],
                            "pool_id": "C149",
                        }
                        for column, want in expect.items():
                            if set(frame[column]) != {want}:
                                issues.append(f"{tag}: {column} differs from the frozen matrix")
                        if set(frame["epoch"]) != {int(spec["epoch"])} or set(frame["model_seed"]) != {int(entry["model_seed"])}:
                            issues.append(f"{tag}: epoch/model_seed differ from the frozen matrix")
                        if set(frame["p_conditioning"].astype(str).str.lower()) != {str(bool(entry["p_conditioning"])).lower()}:
                            issues.append(f"{tag}: p_conditioning differs from the frozen matrix")
                        if set(frame["site_fraction"].round(2)) != {round(float(coverage), 2)}:
                            issues.append(f"{tag}: site_fraction differs from the path")
                        if set(frame["mask_replica_id"]) != {int(replica)}:
                            issues.append(f"{tag}: mask_replica_id differs from the path")
                        if set(frame["eval_git_sha"]) != {prov.get("eval_git_sha")}:
                            issues.append(f"{tag}: eval_git_sha differs from provenance")
                        if set(frame["shared_forward_from"].fillna("")) != {source}:
                            issues.append(f"{tag}: shared_forward_from differs from the matrix ({source!r})")
                        key = sorted(zip(frame["sim_id"].astype(str), frame["frame_index"].astype(int)))
                        if role not in reference:
                            reference[role] = key
                            if frame["sim_id"].nunique() != EXPECTED_CONDITIONS[role]:
                                issues.append(f"{tag}: {frame['sim_id'].nunique()} conditions, expected {EXPECTED_CONDITIONS[role]}")
                        elif key != reference[role]:
                            issues.append(f"{tag}: frame set differs from the first {role} file")
                        run_frames.append(frame)
        for name in CHECKPOINTS:
            if not (run_dir / "fields" / f"val_frame1000_{name}_cov05_rep0.npz").exists():
                issues.append(f"{rid}: field file for {name} missing")
        if not (run_dir / "setting_costs.csv").exists():
            issues.append(f"{rid}: setting_costs.csv missing")
        if info["n_csv"] != N_SETTINGS:
            issues.append(f"{rid}: {info['n_csv']} CSV files, expected {N_SETTINGS}")
        if run_frames:
            frames.append(pd.concat(run_frames, ignore_index=True))
    if not frames:
        return pd.DataFrame(), runs, issues
    table = pd.concat(frames, ignore_index=True)
    table["coverage"] = table["site_fraction"].round(2)
    table["replica"] = table["mask_replica_id"].astype(int)
    issues += cross_checks(table, matrix, payload_hashes)
    return table, runs, issues


def cross_checks(
    table: pd.DataFrame,
    matrix: Mapping[str, Any],
    payload_hashes: Mapping[tuple, Mapping[str, str]] | None,
) -> list[str]:
    issues: list[str] = []
    key = ["run_id", "checkpoint_kind", "role", "coverage", "replica", "sim_id", "frame_index"]
    dup = int(table.duplicated(subset=key).sum())
    if dup:
        issues.append(f"{dup} duplicated (run, checkpoint, role, coverage, replica, frame) keys")
    expected_rows = len(matrix["runs"]) * len(CHECKPOINTS) * len(COVERAGES) * len(REPLICAS) * sum(ROLE_N_FRAMES.values())
    if len(table) != expected_rows and not dup:
        issues.append(f"{len(table)} rows in total, expected {expected_rows}")
    per_key = table.groupby(["role", "sim_id", "frame_index", "coverage", "replica"])["mask_hash"].nunique()
    if int((per_key != 1).sum()):
        issues.append(f"{int((per_key != 1).sum())} (role, frame, coverage, replica) keys have several mask hashes across runs/checkpoints")
    if payload_hashes is not None:
        uniq = table[["role", "sim_id", "frame_index", "coverage", "replica", "mask_hash"]].drop_duplicates()
        bad = 0
        for role, sim, frame, cov, rep, mh in uniq.itertuples(index=False):
            ref = payload_hashes.get((role, str(sim), int(frame), int(rep)))
            col = PAYLOAD_HASH_KEY.get(round(float(cov), 2))
            if ref is None or col is None or ref[col] != str(mh):
                bad += 1
        if bad:
            issues.append(f"{bad} (role, frame, coverage, replica) mask hashes differ from the verified payload")
    metrics = list(ALL_METRICS)
    sort_key = ["role", "coverage", "replica", "sim_id", "frame_index"]
    for entry in matrix["runs"]:
        for name in CHECKPOINTS:
            source = entry["checkpoints"][name].get("reuse_forward_from")
            if not source:
                continue
            a = table[(table.run_id == entry["run_id"]) & (table.checkpoint_kind == name)].sort_values(sort_key)
            b = table[(table.run_id == entry["run_id"]) & (table.checkpoint_kind == source)].sort_values(sort_key)
            if a.empty or b.empty:
                continue
            if not np.array_equal(a[metrics].to_numpy(), b[metrics].to_numpy(), equal_nan=True):
                issues.append(f"{entry['run_id']}: {name} metrics differ from its shared source {source}")
    return issues


def build_tables(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Frame -> condition -> replica -> run, mean at every step. Regional means skip NaN frames."""
    metrics = list(ALL_METRICS)
    cond_rep = table.groupby(IDENT + ["replica", "sim_id"], dropna=False)[metrics].mean().reset_index()
    cond_rep["n_frames"] = table.groupby(IDENT + ["replica", "sim_id"], dropna=False)["frame_index"].nunique().to_numpy()
    per_condition = cond_rep.groupby(IDENT + ["sim_id"], dropna=False)[metrics].mean().reset_index()
    spread = cond_rep.groupby(IDENT + ["sim_id"], dropna=False)["rmse_K"].agg(rmse_rep_min_K="min", rmse_rep_max_K="max").reset_index()
    per_condition = per_condition.merge(spread, on=IDENT + ["sim_id"])
    glob_rep = cond_rep.groupby(IDENT + ["replica"], dropna=False)[metrics].mean().reset_index()
    glob = glob_rep.groupby(IDENT, dropna=False)[metrics].mean().reset_index()
    rep_rmse = glob_rep.groupby(IDENT, dropna=False)["rmse_K"].agg(rmse_rep_min_K="min", rmse_rep_max_K="max").reset_index()
    glob = glob.merge(rep_rmse, on=IDENT)
    glob["n_conditions"] = cond_rep.groupby(IDENT, dropna=False)["sim_id"].nunique().to_numpy()
    order = ["family", "p_conditioning", "model_seed", "checkpoint_kind", "role", "coverage"]
    for frame in (glob, per_condition, glob_rep):
        frame.sort_values(by=order, inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return {
        "global_by_setting": glob,
        "global_by_replica": glob_rep,
        "per_condition": per_condition,
        "regional_by_setting": glob[IDENT + list(REGIONAL_METRICS) + ["rmse_K"]],
    }


def family_p_summary(glob: pd.DataFrame) -> pd.DataFrame:
    """Descriptive dispersion over the three training seeds. No ranking column."""
    cols = ["rmse_K", "mae_K", "mean_signed_error_K", "abs_bias_K", "centered_rmse_K"]
    group = ["family", "p_conditioning", "checkpoint_kind", "role", "coverage"]
    parts = []
    for c in cols:
        agg = glob.groupby(group)[c].agg(["mean", "min", "max", "std", "count"]).reset_index()
        agg.columns = group + [f"{c}_mean", f"{c}_min", f"{c}_max", f"{c}_std_ddof1", f"{c}_n_seeds"]
        parts.append(agg)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part.drop(columns=[c for c in part.columns if c.endswith("_n_seeds")]), on=group)
    return out


def paired_contrasts(per_condition: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """GLU minus U-Net (same p, same seed index) and p_on minus p_off (same family, seed)."""
    pc = per_condition
    key = ["checkpoint_kind", "role", "coverage", "sim_id"]
    rows = []
    for kind, pairs in (
        ("family", [({"family": "adapted_glu", "p_conditioning": p, "model_seed": s}, {"family": "sparse_unet", "p_conditioning": p, "model_seed": s},
                     f"GLU-UNet p_{'on' if p else 'off'} seed{s}") for p in (True, False) for s in (0, 1, 2)]),
        ("p", [({"family": f, "p_conditioning": True, "model_seed": s}, {"family": f, "p_conditioning": False, "model_seed": s},
                f"{'GLU' if f == 'adapted_glu' else 'UNet'} on-off seed{s}") for f in ("sparse_unet", "adapted_glu") for s in (0, 1, 2)]),
    ):
        for a_sel, b_sel, label in pairs:
            a = pc
            for k, v in a_sel.items():
                a = a[a[k] == v]
            b = pc
            for k, v in b_sel.items():
                b = b[b[k] == v]
            for metric in ALL_METRICS:
                m = a[key + [metric, "run_id"]].merge(b[key + [metric, "run_id"]], on=key, suffixes=("_a", "_b"))
                m["delta_a_minus_b"] = m[f"{metric}_a"] - m[f"{metric}_b"]
                m.insert(0, "metric", metric)
                m.insert(0, "label", label)
                m.insert(0, "contrast", kind)
                rows.append(m.rename(columns={"run_id_a": "run_a", "run_id_b": "run_b", f"{metric}_a": "a_value", f"{metric}_b": "b_value"}))
    detail = pd.concat(rows, ignore_index=True)
    group = ["contrast", "label", "run_a", "run_b", "checkpoint_kind", "role", "coverage", "metric"]
    summary = (
        detail.groupby(group, dropna=False)["delta_a_minus_b"]
        .agg(mean_delta="mean", min_delta="min", max_delta="max", n_conditions="count",
             n_a_lower=lambda s: int((s < 0).sum()), n_a_higher=lambda s: int((s > 0).sum()))
        .reset_index()
    )
    return {"paired_summary": summary, "paired_by_condition": detail}


def degradations(glob: pd.DataFrame, per_condition: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Coverage, role and selection shifts per run, plus the worst VAL condition."""
    r = glob.pivot_table(index=["run_id", "family", "p_conditioning", "model_seed", "checkpoint_kind", "role"],
                         columns="coverage", values="rmse_K").reset_index()
    r.columns.name = None
    cov = r.rename(columns={0.01: "rmse_cov01_K", 0.05: "rmse_cov05_K", 0.2: "rmse_cov20_K"})
    cov["delta_cov01_minus_cov05_K"] = cov["rmse_cov01_K"] - cov["rmse_cov05_K"]
    cov["delta_cov20_minus_cov05_K"] = cov["rmse_cov20_K"] - cov["rmse_cov05_K"]
    role = glob.pivot_table(index=["run_id", "family", "p_conditioning", "model_seed", "checkpoint_kind", "coverage"],
                            columns="role", values="rmse_K").reset_index()
    role.columns.name = None
    role["val_minus_seen_K"] = role["val_eval"] - role["seen_train_eval"]
    role["val_minus_ref760_K"] = role["val_eval"] - role["train_fit"]
    sel = glob.pivot_table(index=["run_id", "family", "p_conditioning", "model_seed", "role", "coverage"],
                           columns="checkpoint_kind", values="rmse_K").reset_index()
    sel.columns.name = None
    sel["last_minus_best_K"] = sel["last"] - sel["best"]
    sel["best_seen_minus_best_K"] = sel["best_seen_diagnostic"] - sel["best"]
    val = per_condition[per_condition.role == "val_eval"]
    idx = val.groupby(["run_id", "checkpoint_kind", "coverage"])["rmse_K"].idxmax()
    worst = val.loc[idx, ["run_id", "family", "p_conditioning", "model_seed", "checkpoint_kind", "coverage", "sim_id", "rmse_K"]]
    worst = worst.rename(columns={"sim_id": "worst_val_condition", "rmse_K": "worst_val_condition_rmse_K"}).reset_index(drop=True)
    return {"coverage_degradation": cov, "role_gap": role, "selection_shift": sel, "worst_val_condition": worst}


def nan_report(table: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "checkpoint_kind", "role", "coverage"]
    rows = []
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
