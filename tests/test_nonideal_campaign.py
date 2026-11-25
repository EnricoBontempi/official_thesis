"""CPU tests of the specialist campaign logic: status classification, freeze, aggregation refusal/acceptance (synthetic)."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation import nonideal_eval as ne
from combustion_recon.training import nonideal_campaign as nc

ROOT = Path(__file__).resolve().parents[1]


def _mod(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _summary(root, rid, status, stop, epochs):
    d = root / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"status": status, "stop_reason": stop, "epochs_completed": epochs}))


def test_classify_run_states(tmp_path):
    _summary(tmp_path, "a", "stopped_early_plateau", "early_stop", 410)
    _summary(tmp_path, "b", "completed", "reached_max_epochs", 1000)
    _summary(tmp_path, "c", "completed", "reached_max_epochs", 1200)
    _summary(tmp_path, "d", "stopped_wall_time", "wall_time_cap", 300)
    _summary(tmp_path, "e", "stopped_error", "non_finite_loss", 12)
    _summary(tmp_path, "f", "completed", "reached_max_epochs", 777)
    (tmp_path / "g").mkdir()
    got = {r: nc.classify_run(tmp_path, r)["state"] for r in "abcdefgh"}
    assert got == {"a": nc.PLATEAU, "b": nc.NEEDS_EXTENSION, "c": nc.BUDGET_EXHAUSTED, "d": nc.INCOMPLETE,
                   "e": nc.ERROR, "f": nc.ERROR, "g": nc.INCOMPLETE, "h": nc.INCOMPLETE}


def test_validate_refuses_non_terminal_states_without_loading(tmp_path):
    _summary(tmp_path, "r", "stopped_wall_time", "wall_time_cap", 300)
    out = nc.validate_and_freeze_run(tmp_path, {"run_id": "r", "regime": "G1", "family": "sparse_unet", "seed": 0},
                                     expect_git_sha="x", expect_spec_sha="y", load_checkpoints=False)
    assert not out["valid"] and "state incomplete" in out["problems"][0]


def test_validate_checks_identity_fields(tmp_path):
    _summary(tmp_path, "r", "stopped_early_plateau", "early_stop", 410)
    (tmp_path / "r" / "run_identity.json").write_text(json.dumps({
        "run_id": "r", "family": "sparse_unet", "p_conditioning": True, "git_sha": "WRONG",
        "nonideal_specialist": {"regime": "N1", "degradation_spec_sha256": "bad"}}))
    out = nc.validate_and_freeze_run(tmp_path, {"run_id": "r", "regime": "G1", "family": "sparse_unet", "seed": 0},
                                     expect_git_sha="x", expect_spec_sha="y", load_checkpoints=False)
    assert not out["valid"]
    text = " ".join(out["problems"])
    assert "regime" in text and "git_sha" in text and "spec hash" in text


def test_weights_hash_matches_s7_definition():
    import torch
    s7 = _mod("scripts/38_s7_eval.py")
    sd = {"b": torch.arange(4.0), "a": torch.ones(2, 2)}
    assert nc.weights_sha256(sd) == s7.weights_sha256(sd)


def test_stage_extension_lines_follow_assignment(tmp_path):
    st = _mod("scripts/66_nonideal_spec_stage.py")
    root = tmp_path / "runs"
    _summary(root, "A", "completed", "reached_max_epochs", 1000)
    _summary(root, "B", "stopped_early_plateau", "early_stop", 300)
    _summary(root, "C", "completed", "reached_max_epochs", 1000)
    (root / "_ledger").mkdir()
    (root / "_ledger" / "assignment.json").write_text(json.dumps({"normal": ["A", "B"], "preempt": ["C"]}))
    mat = tmp_path / "m.json"
    mat.write_text(json.dumps({"out_root": str(root), "new_runs": [{"run_id": r} for r in "ABC"]}))
    import io, contextlib, sys
    buf = io.StringIO()
    sys.argv = ["x", "extension", "--matrix", str(mat)]
    with contextlib.redirect_stdout(buf):
        assert st.main() == 0
    assert buf.getvalue().split() == ["normal:A", "preempt:C"]


# ----------------------------------------------------------------------------- aggregation

def _synth_specialist_frames(run, regime, shift_own):
    from tests.test_nonideal_eval import synth_run  # type: ignore
    return None


def test_aggregate_refuses_incomplete_and_accepts_complete(tmp_path, monkeypatch):
    agg = _mod("scripts/69_nonideal_specialist_aggregate.py")
    te = _mod("tests/test_nonideal_eval.py")
    spec_sha = "S" * 64
    runs, freeze_runs = [], {}
    baseline_ids = {("sparse_unet", 0): "BU0", ("sparse_unet", 1): "BU1", ("sparse_unet", 2): "BU2",
                    ("adapted_glu", 0): "BG0", ("adapted_glu", 1): "BG1", ("adapted_glu", 2): "BG2"}
    base_parts = []
    for (fam, seed), bid in baseline_ids.items():
        r = {"run_id": bid, "family": fam, "p_conditioning": True, "model_seed": seed, "best_file_sha256": "f" * 64}
        df = te.synth_run(r, seed=seed, spec_sha="b", with_diag=False)
        base_parts.append(df)
    base = pd.concat(base_parts, ignore_index=True)
    bpath = tmp_path / "base.csv.gz"
    base.to_csv(bpath, index=False)
    bman = {"block2_per_frame_csv": str(bpath), "block2_per_frame_csv_sha256": hashlib.sha256(bpath.read_bytes()).hexdigest(),
            "runs": {k: {} for k in baseline_ids.values()}}
    (tmp_path / "bman.json").write_text(json.dumps(bman))
    monkeypatch.setattr(agg, "BASELINES", tmp_path / "bman.json")
    eval_root = tmp_path / "eval"
    i = 0
    for fam in ("sparse_unet", "adapted_glu"):
        for regime in agg.REGIMES:
            for seed in (0, 1, 2):
                rid = f"NI-{fam[:4]}-{regime}-s{seed}"
                r = {"run_id": rid, "family": fam, "p_conditioning": True, "model_seed": seed, "best_file_sha256": "f" * 64}
                df = te.synth_run(r, seed=100 + i, spec_sha=spec_sha, with_diag=False)
                # specialist is better in its own scenario by 2 K, worse on clean by 0.5 K
                df.loc[df.cell == regime.replace("-eq", "-eq"), "rmse_K"] -= 2.0
                df.loc[df.cell == "R0", "rmse_K"] += 0.5
                df["family"] = fam
                kdir = eval_root / rid / "best"
                for (cell, role, rep), g in df.groupby(["cell", "role", "replica"]):
                    p = kdir / "per_frame" / cell / role / f"rep{rep}.csv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    g.to_csv(p, index=False)
                (kdir / "gate.json").write_text(json.dumps({"all_pass": True}))
                (kdir / "COMPLETE.json").write_text("{}")
                (eval_root / rid / "last").mkdir(parents=True, exist_ok=True)
                (eval_root / rid / "last" / "SAME_AS_BEST.json").write_text("{}")
                freeze_runs[rid] = {"valid": True, "family": fam, "regime": regime, "seed": seed, "problems": [], "state": "plateau",
                                    "epochs_completed": 300, "best_val_epoch": 250, "best_val_metric": 40.0, "specialist_val_at_best": 41.0,
                                    "n_evals": 30, "clip_events": 1, "last_same_as_best": True, "best": {"epoch": 250}, "last": {"epoch": 300},
                                    "ledger": {"attempts": 1, "extension": False, "signals": 0}}
                i += 1
    freeze = {"n_runs": 30, "n_valid": 30, "problems": [], "degradation_spec_sha256": spec_sha, "runs": freeze_runs}
    fpath = tmp_path / "freeze.json"
    fpath.write_text(json.dumps(freeze))
    import sys
    # incomplete: one run without COMPLETE -> refusal with a BLOCK_REPORT, no package
    victim = next(iter(freeze_runs))
    (eval_root / victim / "best" / "COMPLETE.json").unlink()
    dest = tmp_path / "out" / "package"
    dest.parent.mkdir()
    sys.argv = ["x", "--freeze", str(fpath), "--eval-root", str(eval_root), "--dest", str(dest)]
    assert agg.main() == 3
    assert (dest.parent / "BLOCK_REPORT.md").exists() and not dest.exists()
    assert victim in (dest.parent / "BLOCK_REPORT.md").read_text()
    (eval_root / victim / "best" / "COMPLETE.json").write_text("{}")
    (dest.parent / "BLOCK_REPORT.md").unlink()
    assert agg.main() == 0
    t = pd.read_csv(dest / "tables" / "seed_summary.csv")
    q = t[(t.family == "sparse_unet") & (t.checkpoint_kind == "best") & (t.role == "val_eval") & (t.train_regime == "G1") & (t.cell == "G1")].iloc[0]
    assert q["n_seeds"] == 3
    assert q["gain_vs_baseline_K__mean"] == pytest.approx(2.0 - (0.0), abs=0.6)  # own-scenario gain ~ +2 K (noise in the synthetic baseline)
    c = t[(t.family == "sparse_unet") & (t.checkpoint_kind == "best") & (t.role == "val_eval") & (t.train_regime == "G1") & (t.cell == "R0")].iloc[0]
    assert c["clean_cost_K__mean"] == pytest.approx(0.5, abs=0.3)
    assert (dest / "COMPLETE.json").exists() and (dest / "FINAL_SUMMARY.md").exists()
    assert len(list((dest / "figures").glob("*.png"))) == 4
