"""The O5 evaluator must import and call the real thesis code. No GPU, no forward.

Attempt1 (array 29716019) died on ``combustion_recon.training.normalizer``, a
module that does not exist. These tests resolve every import and bind every
keyword the script passes against the real signatures.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "26_opus_o5_eval.py"
MATRIX = ROOT / "configs" / "experiments" / "opus_o5_matrix.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def o5():
    return _load(SCRIPT, "o5_eval_under_test")


@pytest.fixture(scope="module")
def s6():
    return _load(ROOT / "scripts" / "17_s6_sparse_eval_and_swap.py", "s6_under_test")


def test_every_project_import_resolves():
    tree = ast.parse(SCRIPT.read_text())
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("combustion_recon"):
            module = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
                checked += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("combustion_recon"):
                    importlib.import_module(alias.name)
    assert checked >= 8


def test_the_attempt1_module_does_not_exist_and_is_not_named():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("combustion_recon.training.normalizer")
    assert "combustion_recon.training.normalizer" not in SCRIPT.read_text()
    from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer

    assert Month2FitOnlyNormalizer.__module__ == "combustion_recon.data.month2_normalizer"


def _callable_for(name: str, o5, s6):
    if name.startswith("s6."):
        return getattr(s6, name[3:])
    return getattr(o5, name)


CALLS = (
    "SparseUNetCacheDataset",
    "Month2FitOnlyNormalizer",
    "check_eval_mask_contract",
    "verify_cache_payload_hashes",
    "cache_validation_kwargs",
    "cache_indices_for_pool",
    "reuse_plan",
    "s6.evaluate_setting",
    "s6.load_sparse_scientific_checkpoint",
    "s6._load_model",
    "s6._forward",
    "s6.denormalize_temperature",
    "s6.collate_sparse_batch",
)


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "s6":
        return f"s6.{func.attr}"
    return None


def test_every_effective_call_binds_to_the_real_signature(o5, s6):
    tree = ast.parse(SCRIPT.read_text())
    seen: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name not in CALLS:
            continue
        target = _callable_for(name, o5, s6)
        signature = inspect.signature(target)
        keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
        unknown = [
            kw for kw in keywords if kw not in signature.parameters
            and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        ]
        assert not unknown, f"{name} called with unknown keywords {unknown}"
        has_splat = any(kw.arg is None for kw in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args
        )
        if not has_splat:
            dummy = {kw: object() for kw in keywords}
            signature.bind(*[object()] * len(node.args), **dummy)
        seen[name] = seen.get(name, 0) + 1
    for required in (
        "SparseUNetCacheDataset",
        "Month2FitOnlyNormalizer",
        "check_eval_mask_contract",
        "s6.evaluate_setting",
        "s6.load_sparse_scientific_checkpoint",
    ):
        assert seen.get(required, 0) >= 1, f"{required} is never called"


def test_s6_helpers_used_for_field_dump_exist(s6):
    for name in ("collate_sparse_batch", "denormalize_temperature", "_forward", "_load_model", "evaluate_setting"):
        assert callable(getattr(s6, name))


def test_matrix_stats_build_the_normalizer_when_the_file_is_present():
    stats = Path(json.loads(MATRIX.read_text())["shared_inputs"]["stats"])
    if not stats.is_file():
        pytest.skip("stats file is not on this machine")
    from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer

    normalizer = Month2FitOnlyNormalizer(stats)
    assert float(normalizer.temperature_std) > 0


def test_smoke_flag_and_out_root_must_agree(o5, monkeypatch, tmp_path):
    for flags, out in (
        (["--smoke"], tmp_path / "scientific"),
        ([], tmp_path / "x_smoke_runs"),
    ):
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--matrix", str(MATRIX), "--run-id", "O-B19-on-s0", "--out-root", str(out),
             "--eval-git-sha", "0" * 40, *flags],
        )
        with pytest.raises(SystemExit, match="smoke"):
            o5.main()
    assert not (tmp_path / "scientific").exists()
    assert not (tmp_path / "x_smoke_runs").exists()


def test_smoke_subset_is_spread_and_keeps_frame_1000(o5):
    frames = [{"frame_index": i % 5 * 250} for i in range(60)]
    chosen = o5._smoke_positions(frames, list(range(60)), 6, keep_frame_1000=True)
    assert len(chosen) >= 6
    assert chosen == sorted(set(chosen))
    assert [i for i in range(60) if frames[i]["frame_index"] == 1000] == [
        i for i in chosen if frames[i]["frame_index"] == 1000
    ]
    assert chosen[0] == 0 and chosen[-1] >= 50


def _write_synthetic_run(root: Path, entry: dict, n: dict, shared: bool, run_id: str) -> None:
    from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, COVERAGES, REPLICAS, ROLES

    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        for coverage in COVERAGES:
            for role in ROLES:
                for replica in REPLICAS:
                    path = root / "per_frame" / name / f"cov{coverage:.2f}" / role / f"rep{replica}.csv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        {
                            "rmse_K": np.linspace(10.0, 20.0, n[role]),
                            "checkpoint_kind": name,
                            "weights_sha256": spec["weights_sha256"],
                            "file_sha256": spec["file_sha256"],
                            "run_id": run_id,
                            "shared_forward_from": spec.get("reuse_forward_from") or "",
                        }
                    ).to_csv(path, index=False)
        field = root / "fields" / f"val_frame1000_{name}_cov05_rep0.npz"
        field.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(field, pred_T_K=np.ones((3, 4, 4)), true_T_K=np.ones((3, 4, 4)))
    (root / "provenance.json").write_text(json.dumps({"run_id": run_id}))


def _entry():
    matrix = json.loads(MATRIX.read_text())
    return next(row for row in matrix["runs"] if row["run_id"] == "O-B19-on-s0")


def test_readback_accepts_a_consistent_run_and_rejects_damage(o5, tmp_path):
    entry = _entry()
    counts = {"train_fit": 4, "seen_train_eval": 3, "val_eval": 5}
    run = tmp_path / "run"
    _write_synthetic_run(run, entry, counts, shared=True, run_id="O-B19-on-s0")
    report = o5.verify_outputs(run, entry, counts, "O-B19-on-s0")
    assert report["n_csv"] == 81
    victim = run / "per_frame" / "last" / "cov0.05" / "val_eval" / "rep1.csv"
    frame = pd.read_csv(victim)
    frame.loc[0, "rmse_K"] = np.nan
    frame.to_csv(victim, index=False)
    with pytest.raises(SystemExit, match="non-finite"):
        o5.verify_outputs(run, entry, counts, "O-B19-on-s0")
    frame["rmse_K"] = 1.0
    frame["weights_sha256"] = "0" * 64
    frame.to_csv(victim, index=False)
    with pytest.raises(SystemExit, match="weights_sha256"):
        o5.verify_outputs(run, entry, counts, "O-B19-on-s0")


def test_readback_checks_the_shared_forward_identity(o5, tmp_path):
    entry = _entry()
    assert entry["checkpoints"]["best_seen_diagnostic"]["reuse_forward_from"] == "best"
    counts = {"train_fit": 4, "seen_train_eval": 3, "val_eval": 5}
    run = tmp_path / "run"
    _write_synthetic_run(run, entry, counts, shared=True, run_id="O-B19-on-s0")
    victim = run / "per_frame" / "best_seen_diagnostic" / "cov0.01" / "train_fit" / "rep0.csv"
    frame = pd.read_csv(victim)
    frame["rmse_K"] = frame["rmse_K"] + 1.0
    frame.to_csv(victim, index=False)
    with pytest.raises(SystemExit, match="shared forward differs"):
        o5.verify_outputs(run, entry, counts, "O-B19-on-s0")
