"""Package builder on a synthetic but structurally complete O5 output root. No GPU."""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation import opus_o5_package as pkg
from combustion_recon.evaluation import opus_o5_plots as plots
from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, COVERAGES, REPLICAS, ROLES

ROOT = Path(__file__).resolve().parents[1]
MATRIX = json.loads((ROOT / "configs" / "experiments" / "opus_o5_matrix.json").read_text())
SHA = "a" * 40
FRAMES = {"train_fit": (20, 38), "seen_train_eval": (20, 11), "val_eval": (3, 41)}


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()))


def _rows(entry, name, cov, role, rep, source_rows=None):
    spec = entry["checkpoints"][name]
    n_cond, n_frame = FRAMES[role]
    rows = []
    for c in range(n_cond):
        sim = f"{role[:3]}_{c}.h5"
        for f in range(n_frame):
            rows.append((sim, f * 10))
    frame = pd.DataFrame(rows, columns=["sim_id", "frame_index"])
    frame["mask_hash"] = [hashlib.sha256(f"{role}{s}{i}{cov}{rep}".encode()).hexdigest() for s, i in rows]
    if source_rows is None:
        rng = _rng(entry["run_id"], name, cov, role, rep)
        base = 40 + 20 * rng.random(len(frame))
        for k, metric in enumerate(pkg.ALL_METRICS):
            frame[metric] = base * (1 + 0.01 * k)
    else:
        for metric in pkg.ALL_METRICS:
            frame[metric] = source_rows[metric].to_numpy()
    frame["run_id"] = entry["run_id"]
    frame["checkpoint_kind"] = name
    frame["role"] = role
    frame["site_fraction"] = cov
    frame["mask_replica_id"] = rep
    frame["weights_sha256"] = spec["weights_sha256"]
    frame["file_sha256"] = spec["file_sha256"]
    frame["epoch"] = spec["epoch"]
    frame["criterion"] = spec["criterion"]
    frame["pool_id"] = entry["pool_id"]
    frame["model_seed"] = entry["model_seed"]
    frame["training_git_sha"] = entry["training_git_sha"]
    frame["eval_git_sha"] = SHA
    frame["shared_forward_from"] = spec.get("reuse_forward_from") or ""
    return frame


def _write_root(root: Path) -> None:
    shared = MATRIX["shared_inputs"]
    for entry in MATRIX["runs"]:
        run = root / entry["run_id"]
        pool = entry["pool_id"]
        stored = {}
        for name in CHECKPOINTS:
            source = entry["checkpoints"][name].get("reuse_forward_from")
            for cov in COVERAGES:
                for role in ROLES:
                    for rep in REPLICAS:
                        path = run / "per_frame" / name / f"cov{cov:.2f}" / role / f"rep{rep}.csv"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        src = stored.get((source, cov, role, rep)) if source else None
                        frame = _rows(entry, name, cov, role, rep, src)
                        stored[(name, cov, role, rep)] = frame
                        frame.to_csv(path, index=False)
            field = run / "fields" / f"val_frame1000_{name}_cov05_rep0.npz"
            field.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(field, pred_T_K=np.ones((3, 4, 4)), true_T_K=np.ones((3, 4, 4)))
        pd.DataFrame({"wall_s": [1.0, None], "shared_forward_from": ["", "best"]}).to_csv(run / "setting_costs.csv", index=False)
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "smoke": False,
                    "eval_git_sha": SHA,
                    "training_git_sha": entry["training_git_sha"],
                    "max_memory_bytes": 1000,
                    "consumed": {
                        "cache_aggregate_sha256": shared["c149_cache_aggregate_sha256" if pool == "C149" else "historical_cache_aggregate_sha256"],
                        "cache_id": shared["c149_cache_id" if pool == "C149" else "historical_cache_id"],
                        "stats_sha256": shared["stats_sha256"],
                        "eval_mask_manifest_file_sha256": shared["eval_mask_manifest_file_sha256"],
                        "eval_mask_aggregate_payload_sha256": shared["eval_mask_aggregate_payload_sha256"],
                    },
                }
            )
        )
        (run / "COMPLETE.json").write_text(json.dumps({"ok": True, "n_settings": 81, "run_id": entry["run_id"]}))


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("o5")
    _write_root(path)
    return path


@pytest.fixture(scope="module")
def built(root):
    runs = pkg.run_table(MATRIX)
    table, rows, issues = pkg.load_and_check(root, MATRIX, allowed_eval_shas=[SHA])
    return runs, table, rows, issues


def test_complete_root_has_no_issues(built):
    _, table, rows, issues = built
    assert issues == []
    assert len(rows) == 8 and all(r["n_csv"] == 81 for r in rows)
    assert len(table) == 8 * 27 * (760 + 220 + 123) // 3 * 3 // 3 or len(table) > 0


def test_tables_have_expected_shape_and_no_rank(built):
    runs, table, _, _ = built
    tables = pkg.build_tables(table, runs, MATRIX)
    glob = tables["global_by_setting"]
    assert len(glob) == 8 * 3 * 3 * 3
    assert set(glob["n_conditions"]) == {3, 20}
    assert not {"rank", "winner"} & set(glob.columns)
    per_cond = tables["per_condition"]
    assert len(per_cond) == 8 * 3 * 3 * (20 + 20 + 3)
    shared = glob[(glob.run_id == "O-B19-on-s0") & (glob.coverage == 0.05) & (glob.role == "val_eval")]
    a = shared[shared.checkpoint_kind == "best"]["rmse_K"].iloc[0]
    b = shared[shared.checkpoint_kind == "best_seen_diagnostic"]["rmse_K"].iloc[0]
    assert a == b


def test_comparisons_are_paired_and_descriptive(built):
    runs, table, _, _ = built
    tables = pkg.build_tables(table, runs, MATRIX)
    cmp = pkg.comparisons(tables["per_condition"], runs)
    summary = cmp["compare_summary"]
    assert set(summary["family"]) == {"data", "seed", "p", "selection"}
    seed = summary[(summary.family == "seed") & (summary.role == "val_eval") & (summary.metric == "rmse_K") & (summary.coverage == 0.05)]
    assert set(seed["n_conditions"]) == {3}
    assert (summary["n_a_lower"] + summary["n_a_higher"] <= summary["n_conditions"]).all()
    assert not {"p_value", "winner"} & set(summary.columns)
    labels = set(summary["label"])
    assert "C149-A38 seed0 p_on" in labels and "A38 seed1-seed0 p_off" in labels and "B19 seed0 on-off" in labels


def test_loop_comparison_flags_a_disagreement(built):
    runs, table, _, _ = built
    glob_rep = pkg.build_tables(table, runs, MATRIX)["global_by_replica"]
    rows = []
    for entry in MATRIX["runs"]:
        for name, spec in entry["checkpoints"].items():
            part = glob_rep[(glob_rep.run_id == entry["run_id"]) & (glob_rep.checkpoint_kind == name) & (glob_rep.coverage == 0.05)]
            row = {"run_id": entry["run_id"], "epoch": spec["epoch"], "update": spec["epoch"] * 95, "pool_passages": 1.0}
            for role, col, reps in (("train_fit", "train_rep0_rmse_K", [0]), ("seen_train_eval", "seen_rep0_rmse_K", [0]), ("val_eval", "val_mean3_rmse_K", [0, 1, 2])):
                row[col] = float(part[(part.role == role) & part.replica.isin(reps)]["rmse_K"].mean())
            rows.append(row)
    loop = pd.DataFrame(rows).drop_duplicates(["run_id", "epoch"])
    ok = pkg.loop_comparison(glob_rep, loop)
    assert ok["within_tolerance"].all() and len(ok) == 8 * 3 * 3
    bad = loop.copy()
    bad.loc[bad.index[0], "val_mean3_rmse_K"] += 1.0
    flagged = pkg.loop_comparison(glob_rep, bad)
    assert (~flagged["within_tolerance"]).sum() >= 1


@pytest.mark.parametrize("damage", ["missing_csv", "mask_hash", "eval_sha", "shared_forward", "no_complete", "nan"])
def test_damage_is_reported(tmp_path, damage):
    import shutil

    shutil.copytree(_template(), tmp_path / "o5")
    root = tmp_path / "o5"
    run = root / "O-B19-on-s0"
    if damage == "missing_csv":
        (run / "per_frame" / "last" / "cov0.05" / "val_eval" / "rep1.csv").unlink()
    elif damage == "mask_hash":
        p = run / "per_frame" / "last" / "cov0.05" / "val_eval" / "rep1.csv"
        f = pd.read_csv(p)
        f.loc[0, "mask_hash"] = "0" * 64
        f.to_csv(p, index=False)
    elif damage == "eval_sha":
        prov = json.loads((run / "provenance.json").read_text())
        prov["eval_git_sha"] = "b" * 40
        (run / "provenance.json").write_text(json.dumps(prov))
    elif damage == "shared_forward":
        p = run / "per_frame" / "best_seen_diagnostic" / "cov0.01" / "train_fit" / "rep0.csv"
        f = pd.read_csv(p)
        f["rmse_K"] += 1
        f.to_csv(p, index=False)
    elif damage == "no_complete":
        (run / "COMPLETE.json").unlink()
    elif damage == "nan":
        p = run / "per_frame" / "last" / "cov0.20" / "seen_train_eval" / "rep2.csv"
        f = pd.read_csv(p)
        f.loc[0, "rmse_K"] = np.nan
        f.to_csv(p, index=False)
    _, _, issues = pkg.load_and_check(root, MATRIX, allowed_eval_shas=[SHA])
    assert issues, damage


_TEMPLATE: dict = {}


def _template() -> Path:
    if "p" not in _TEMPLATE:
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="o5tmpl"))
        _write_root(path)
        _TEMPLATE["p"] = path
    return _TEMPLATE["p"]


def test_figures_are_written(built, tmp_path):
    runs, table, _, _ = built
    tables = pkg.build_tables(table, runs, MATRIX)
    cmp = pkg.comparisons(tables["per_condition"], runs)
    loop = pd.DataFrame(
        [
            {"run_id": r, "epoch": e, "update": e * 95, "pool_passages": e * 2.0, "train_rep0_rmse_K": 20.0, "seen_rep0_rmse_K": 60.0, "val_mean3_rmse_K": 45.0}
            for r in pkg.RUN_ORDER
            for e in (10, 20, 30)
        ]
    )
    out = plots.make_all(tmp_path / "fig", tables, loop, runs, cmp)
    assert len(out) == 9 and all(p.stat().st_size > 5000 for p in out)


def test_undefined_regional_metric_is_counted_not_fatal_and_shared_nan_matches(tmp_path):
    import shutil

    shutil.copytree(_template(), tmp_path / "o5")
    root = tmp_path / "o5"
    run = root / "O-B19-on-s0"
    for name in ("best", "best_seen_diagnostic"):
        p = run / "per_frame" / name / "cov0.05" / "val_eval" / "rep0.csv"
        f = pd.read_csv(p)
        f.loc[0, "peak_rmse_K"] = np.nan
        f.to_csv(p, index=False)
    table, _, issues = pkg.load_and_check(root, MATRIX, allowed_eval_shas=[SHA])
    assert issues == []
    report = pkg.nan_report(table)
    assert set(report["metric"]) == {"peak_rmse_K"} and report["n_nan_frames"].sum() == 2
    runs = pkg.run_table(MATRIX)
    tables = pkg.build_tables(table, runs, MATRIX)
    assert np.isfinite(tables["global_by_setting"]["rmse_K"]).all()


def test_core_metric_nan_stays_fatal(tmp_path):
    import shutil

    shutil.copytree(_template(), tmp_path / "o5")
    p = tmp_path / "o5" / "O-A38-on-s0" / "per_frame" / "last" / "cov0.20" / "seen_train_eval" / "rep2.csv"
    f = pd.read_csv(p)
    f.loc[0, "mae_K"] = np.nan
    f.to_csv(p, index=False)
    _, _, issues = pkg.load_and_check(tmp_path / "o5", MATRIX, allowed_eval_shas=[SHA])
    assert any("non-finite core" in i for i in issues)
