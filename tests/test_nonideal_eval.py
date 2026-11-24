"""CPU tests of the Block II evaluation helpers, verification and aggregation (synthetic data, no checkpoints)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.data.nonideal_observations import file_sha256, load_degradation_spec
from combustion_recon.evaluation import nonideal_eval as ne

ROOT = Path(__file__).resolve().parents[1]
SPEC12 = ROOT / "configs/nonideal/degradation_spec_v1_12models.yaml"
SPEC6 = ROOT / "configs/nonideal/degradation_spec_v1.yaml"
REF = ROOT / "results/month2/s7_c149_comparison/package/tables/global_by_replica.csv"
MATRIX = ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json"


def _mod(rel):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- synthetic run tables

def synth_run(run: dict, *, seed=0, with_diag=None, spec_sha="s", eval_sha="e") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fam, var = run["family"], "p_on" if run["p_conditioning"] else "p_off"
    rows = []
    conds = {"seen_train_eval": [f"c{i:02d}.h5" for i in range(20)], "val_eval": ["v0.h5", "v1.h5", "v2.h5"]}
    per = {"seen_train_eval": 11, "val_eval": 41}
    effect = {"R0": 0, "N1": 3, "G1": 8, "N1G1": 11, "G1-eq": 5, "N2": 9}
    cells = list(ne.CELL_ORDER) + ([ne.DIAG_CELL] if (fam == "adapted_glu" if with_diag is None else with_diag) else [])
    for cell in cells:
        for role in ne.EVAL_ROLES:
            for rep in ne.REPLICAS:
                for c in conds[role]:
                    for f in range(per[role]):
                        fr = 10 * f
                        base = 40 + 5 * (hash(c) % 3 if False else 0)
                        key = "R0" if cell == ne.DIAG_CELL else cell
                        r = base + effect[key] + (0.01 if cell == ne.DIAG_CELL else 0) + rng.normal(0, 0.1)
                        n_orig = 819
                        rem = {"G1": 205, "N1G1": 205, "G1-eq": 205}.get(key, 0) + (rep if key in ("G1", "N1G1", "G1-eq") else 0)
                        rem += (fr % 7) if key in ("G1", "N1G1", "G1-eq") else 0
                        reg = key in ("G1", "N1G1")
                        rows.append({
                            "run_id": run["run_id"], "family": fam, "variant": var, "model_seed": run["model_seed"],
                            "role": role, "condition": c, "frame_index": fr, "replica": rep, "cell": cell,
                            "rmse_K": r, "mae_K": r * 0.6, "bias_K": 1.0, "mean_signed_error_K": 1.0, "abs_bias_K": 2.0,
                            "centered_rmse_K": r * 0.9, "jet_rmse_K": r * 2, "high_rmse_K": r * .8, "peak_rmse_K": r * 1.2, "high_iou": 0.7,
                            "g1in_rmse_K": r * 1.1 if reg else np.nan, "g1out_rmse_K": r * .9 if reg else np.nan,
                            "g1in_bias_K": 1.0 if reg else np.nan, "g1out_bias_K": 0.5 if reg else np.nan,
                            "g1in_n": 4000 if reg else 0, "g1out_n": 12000 if reg else 0,
                            "n_original": n_orig, "n_surviving": n_orig - rem, "n_removed": rem,
                            "removed_site_fraction": rem / n_orig, "area_fraction_effective": 0.25 if reg else np.nan,
                            "area_mm2": 1800.0 if reg else np.nan,
                            "batching": "adapted_glu:bs1" if (fam == "adapted_glu" and cell != ne.DIAG_CELL) else (
                                "adapted_glu:s7_bs8" if fam == "adapted_glu" else "sparse_unet:bs8"),
                            "original_sites_hash": hashlib.sha256(f"{role}{c}{fr}{rep}".encode()).hexdigest(),
                            "surviving_sites_hash": hashlib.sha256(f"{role}{c}{fr}{rep}{'G' if 'G1' in key and key != 'G1-eq' else key}".encode()).hexdigest(),
                            "spec_sha256": spec_sha, "checkpoint_file_sha256": run["best_file_sha256"], "eval_git_sha": eval_sha,
                        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def spec12():
    return load_degradation_spec(SPEC12)


@pytest.fixture(scope="module")
def runs12(spec12):
    return spec12.raw["models"]["runs"]


# --------------------------------------------------------------------------- spec revision

def test_revision_preserves_v1_operators_and_widens_models(spec12):
    a, b = load_degradation_spec(SPEC6), spec12
    for k in ("operators", "seeds", "observation_model", "cells", "roles"):
        assert a.raw[k] == b.raw[k]
    assert a.cells == b.cells and a.g1 == b.g1 and a.degradation_seed == b.degradation_seed
    assert b.raw["base_spec"]["sha256"] == file_sha256(SPEC6)
    runs = b.raw["models"]["runs"]
    assert len(runs) == 12
    assert {(r["family"], r["p_conditioning"], r["model_seed"]) for r in runs} == {
        (f, p, s) for f in ("sparse_unet", "adapted_glu") for p in (True, False) for s in (0, 1, 2)}
    assert b.raw["evaluation_counts"]["nominal_frame_evaluations"] == ne.NOMINAL_TOTAL == 74088
    assert b.raw["models"]["excluded"] == ["last", "best_seen_diagnostic"]


def test_spec_models_match_s7_matrix(runs12):
    m = {r["run_id"]: r for r in json.loads(MATRIX.read_text())["runs"]}
    for r in runs12:
        e, best = m[r["run_id"]], m[r["run_id"]]["checkpoints"]["best"]
        assert (e["family"], bool(e["p_conditioning"]), e["model_seed"]) == (r["family"], r["p_conditioning"], r["model_seed"])
        assert (best["epoch"], best["file_sha256"], best["weights_sha256"]) == (r["best_epoch"], r["best_file_sha256"], r["best_weights_sha256"])


def test_original_v1_spec_unchanged_hash():
    assert file_sha256(SPEC6) == "18554a5cd55a203f571a41bcc6a0e0cea6c8e70dda5b5ca3aa21b1a8761197b5"


# --------------------------------------------------------------------------- batching / gate

def test_batch_positions_uniform_policy():
    assert ne.batch_positions(5, "adapted_glu") == [[0], [1], [2], [3], [4]]
    assert ne.batch_positions(20, "sparse_unet") == [list(range(8)), list(range(8, 16)), list(range(16, 20))]
    assert ne.batch_positions(10, "adapted_glu", diagnostic_s7_batching=True) == [list(range(8)), [8, 9]]
    with pytest.raises(ne.NonidealEvalError):
        ne.batch_positions(3, "x")


def _r0_from_reference(run_id, shift=0.0):
    ref = pd.read_csv(REF)
    ref = ref[(ref.run_id == run_id) & (ref.checkpoint_kind == "best") & np.isclose(ref.coverage, 0.05)]
    rows = []
    for _, r in ref[ref.role.isin(ne.EVAL_ROLES)].iterrows():
        n_c, n_f = ne.ROLE_N_CONDITIONS[r.role], ne.ROLE_N_FRAMES[r.role] // ne.ROLE_N_CONDITIONS[r.role]
        for c in range(n_c):
            for f in range(n_f):
                rows.append({"role": r.role, "replica": int(r.replica), "condition": f"c{c}", "frame_index": f, "rmse_K": r.rmse_K + shift})
    return ref, pd.DataFrame(rows)


def test_gate_passes_on_reference_values_and_fails_outside_tolerance():
    ref, r0 = _r0_from_reference("S7-C149-glu-on-s0", 0.0)
    g = ne.gate_against_reference(r0, ref, "S7-C149-glu-on-s0")
    assert g["all_pass"] and g["n_within_tolerance"] == 6 and g["max_abs_delta_K"] < 1e-9
    _, r0b = _r0_from_reference("S7-C149-glu-on-s0", 0.06)
    assert not ne.gate_against_reference(r0b, ref, "S7-C149-glu-on-s0")["all_pass"]
    _, r0c = _r0_from_reference("S7-C149-glu-on-s0", 0.04)
    assert ne.gate_against_reference(r0c, ref, "S7-C149-glu-on-s0")["all_pass"]


def test_gate_fails_on_missing_reference_or_missing_frames():
    ref, r0 = _r0_from_reference("S7-C149-glu-on-s0")
    assert not ne.gate_against_reference(r0, ref[ref.run_id != "S7-C149-glu-on-s0"], "S7-C149-glu-on-s0")["all_pass"]
    short = r0.iloc[:-5]
    assert not ne.gate_against_reference(short, ref, "S7-C149-glu-on-s0")["all_pass"]
    assert not ne.gate_against_reference(r0.iloc[0:0], ref, "S7-C149-glu-on-s0")["all_pass"]


def test_macro_is_frame_condition_uniform_mean():
    df = pd.DataFrame({"condition": ["a"] * 3 + ["b"], "rmse_K": [1.0, 2.0, 3.0, 10.0]})
    assert ne.macro_rmse(df) == pytest.approx((2.0 + 10.0) / 2)


# --------------------------------------------------------------------------- completeness checks

def test_check_run_frame_table_ok_and_failure_modes(runs12):
    unet = next(r for r in runs12 if r["family"] == "sparse_unet")
    glu = next(r for r in runs12 if r["family"] == "adapted_glu")
    assert ne.check_run_frame_table(synth_run(unet), family="sparse_unet", expect_diag=False)["n_nominal"] == ne.NOMINAL_PER_MODEL
    d = synth_run(glu)
    assert ne.check_run_frame_table(d, family="adapted_glu", expect_diag=True)["n_diagnostic"] == 1029
    base = synth_run(unet)
    with pytest.raises(ne.NonidealEvalError, match="duplicate"):
        ne.check_run_frame_table(pd.concat([base, base.iloc[:1]]), family="sparse_unet", expect_diag=False)
    with pytest.raises(ne.NonidealEvalError, match="nominal rows"):
        ne.check_run_frame_table(base.iloc[:-1], family="sparse_unet", expect_diag=False)
    bad = base.copy(); bad.loc[bad.index[5], "rmse_K"] = np.nan
    with pytest.raises(ne.NonidealEvalError, match="non-finite"):
        ne.check_run_frame_table(bad, family="sparse_unet", expect_diag=False)
    bad = base.copy(); m = (bad.cell == "N1G1"); bad.loc[m, "n_removed"] += 1
    with pytest.raises(ne.NonidealEvalError, match="removed count"):
        ne.check_run_frame_table(bad, family="sparse_unet", expect_diag=False)
    bad = base.copy(); bad.loc[(bad.cell == "R0").idxmax(), "n_removed"] = 3
    with pytest.raises(ne.NonidealEvalError, match="R0 removed"):
        ne.check_run_frame_table(bad, family="sparse_unet", expect_diag=False)
    bad = base.copy(); bad.loc[bad.cell == "N1G1", "surviving_sites_hash"] = "x"
    with pytest.raises(ne.NonidealEvalError, match="survivors"):
        ne.check_run_frame_table(bad, family="sparse_unet", expect_diag=False)
    bad = base.copy(); i = bad.index[(bad.cell == "G1")][0]; bad.loc[i, "original_sites_hash"] = "y"
    with pytest.raises(ne.NonidealEvalError, match="original sites"):
        ne.check_run_frame_table(bad, family="sparse_unet", expect_diag=False)
    with pytest.raises(ne.NonidealEvalError, match="diagnostic"):
        ne.check_run_frame_table(base, family="adapted_glu", expect_diag=True)


def test_delta_and_contrast_logic():
    rows = []
    for cell, v in {"R0": 40, "N1": 43, "G1": 48, "N1G1": 52, "G1-eq": 45, "N2": 49}.items():
        rows.append({"run_id": "a", "family": "f", "variant": "p_on", "model_seed": 0, "role": "val_eval", "replica": 0, "cell": cell, "rmse_K": v})
    t = pd.DataFrame(rows)
    d = ne.delta_vs_r0(t, ["run_id", "family", "variant", "model_seed", "role", "replica"], ["rmse_K"])
    assert d.set_index("cell").loc["G1", "d_rmse_K"] == 8 and d.set_index("cell").loc["G1", "rel_rmse_K"] == pytest.approx(0.2)
    c = ne.structure_contrasts(t).iloc[0]
    assert c["G1_minus_G1eq"] == 3 and c["interaction_N1G1_N1_G1_R0"] == 52 - 43 - 48 + 40 and c["N2_minus_N1"] == 6
    with pytest.raises(ne.NonidealEvalError):
        ne.delta_vs_r0(t[t.cell != "R0"], ["run_id", "family", "variant", "model_seed", "role", "replica"], ["rmse_K"])


# --------------------------------------------------------------------------- verify + aggregate end to end (synthetic)

def _write_eval_root(root: Path, runs, spec_path, *, drop_complete=None, bad_gate=None, eval_sha="e1"):
    spec_sha = file_sha256(spec_path)
    for i, r in enumerate(runs):
        d = root / r["run_id"]
        (d / "per_frame").mkdir(parents=True)
        df = synth_run(r, seed=i, spec_sha=spec_sha, eval_sha=eval_sha)
        for (cell, role, rep), g in df.groupby(["cell", "role", "replica"]):
            p = d / "per_frame" / cell / role / f"rep{rep}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            g.to_csv(p, index=False)
        (d / "gate.json").write_text(json.dumps({"all_pass": r["run_id"] != bad_gate, "max_abs_delta_K": 0.01}))
        (d / "provenance.json").write_text(json.dumps({
            "spec_sha256": spec_sha, "base_spec_sha256": "b", "eval_git_sha": eval_sha, "matrix_sha256": "m", "s7_reference_sha256": "s",
            "consumed": {"cache": "c"}, "checkpoint_file_sha256": r["best_file_sha256"], "checkpoint_weights_sha256": r["best_weights_sha256"]}))
        if r["run_id"] != drop_complete:
            (d / "COMPLETE.json").write_text("{}")


def test_verify_accepts_complete_and_reports_problems(tmp_path, runs12):
    v = _mod("scripts/61_nonideal_verify.py")
    ok = tmp_path / "ok"; _write_eval_root(ok, runs12, SPEC12)
    assert v.verify(SPEC12, ok)["ok"]
    miss = tmp_path / "miss"; _write_eval_root(miss, runs12, SPEC12, drop_complete=runs12[3]["run_id"])
    res = v.verify(SPEC12, miss)
    assert not res["ok"] and any("no COMPLETE" in p for p in res["problems"])
    gf = tmp_path / "gf"; _write_eval_root(gf, runs12, SPEC12, bad_gate=runs12[0]["run_id"])
    assert any("gate not passed" in p for p in v.verify(SPEC12, gf)["problems"])
    (ok / "CAMPAIGN_HALT.json").write_text("{}")
    assert any("HALT" in p for p in v.verify(SPEC12, ok)["problems"])
    mixed = tmp_path / "mixed"; _write_eval_root(mixed, runs12, SPEC12)
    pj = json.loads((mixed / runs12[1]["run_id"] / "provenance.json").read_text()); pj["eval_git_sha"] = "other"
    (mixed / runs12[1]["run_id"] / "provenance.json").write_text(json.dumps(pj))
    assert any("eval_git_sha" in p for p in v.verify(SPEC12, mixed)["problems"])


def test_aggregate_synthetic_package(tmp_path, runs12, monkeypatch):
    agg = _mod("scripts/62_nonideal_aggregate.py")
    root = tmp_path / "eval"; _write_eval_root(root, runs12, SPEC12)
    vd = tmp_path / "verify"; vd.mkdir()
    dest = tmp_path / "pkg"
    monkeypatch.setattr("sys.argv", ["x", "--spec", str(SPEC12), "--eval-root", str(root), "--dest", str(dest), "--verify-dir", str(vd)])
    with pytest.raises(SystemExit, match="VERIFY_OK"):
        agg.main()
    (vd / "VERIFY_OK.json").write_text("{}")
    assert agg.main() == 0
    assert (dest / "COMPLETE.json").exists() and (dest / "INDEX.md").exists() and (dest / "FINAL_SUMMARY.md").exists()
    assert len(list((dest / "figures").glob("fig[1-4]*.png"))) == 4
    t = pd.read_csv(dest / "tables/delta_seed_summary.csv")
    g = t[(t.family == "sparse_unet") & (t.variant == "p_on") & (t.role == "val_eval") & (t.cell == "G1")].iloc[0]
    assert g["d_rmse_K__mean"] == pytest.approx(8.0, abs=0.1) and g["n_seeds"] == 3
    c = pd.read_csv(dest / "tables/pon_minus_poff.csv")
    assert abs(c["d_rmse_K_p_on_minus_p_off"]).max() < 0.2
    # refuses an existing destination and an incomplete matrix
    with pytest.raises(FileExistsError):
        agg.main()
    (root / runs12[2]["run_id"] / "COMPLETE.json").unlink()
    monkeypatch.setattr("sys.argv", ["x", "--spec", str(SPEC12), "--eval-root", str(root), "--dest", str(tmp_path / "pkg2"), "--verify-dir", str(vd)])
    with pytest.raises(SystemExit, match="not COMPLETE"):
        agg.main()


# --------------------------------------------------------------------------- p_off isolation (random-init models)

def test_p_off_ignores_p_glu_and_unet_inputs():
    import torch
    from combustion_recon.models.glu_reconstructor import build_glu_from_config
    from combustion_recon.training.sparse_view import build_sparse_unet_input
    x = np.linspace(-2, 2, 8)
    m = build_glu_from_config(domain_x=x, domain_y=x, height=8, width=8).eval()
    c, v = torch.randn(1, 20, 2) * .5, torch.randn(1, 20, 2)
    s, Y = torch.arange(20)[None], torch.randn(1, 64, 2) * .5
    with torch.no_grad():
        a = m(c, v, s, torch.zeros(1, 2), Y, p_conditioning=False, decode_chunked=False, return_flat=True)
        b = m(c, v, s, torch.tensor([[3.0, -2.0]]), Y, p_conditioning=False, decode_chunked=False, return_flat=True)
        on = m(c, v, s, torch.tensor([[3.0, -2.0]]), Y, p_conditioning=True, decode_chunked=False, return_flat=True)
    assert torch.equal(a, b) and not torch.equal(a, on)
    xs = np.zeros((128, 128), np.float32)
    g1 = build_sparse_unet_input(sites=np.array([5, 9]), velocity_values_norm=np.ones((2, 2)), x_scaled=xs, y_scaled=xs, nh3_norm=1.0, phi_norm=2.0, p_conditioning=False)
    g2 = build_sparse_unet_input(sites=np.array([5, 9]), velocity_values_norm=np.ones((2, 2)), x_scaled=xs, y_scaled=xs, nh3_norm=-4.0, phi_norm=7.0, p_conditioning=False)
    assert np.array_equal(g1, g2) and not g1[-2:].any()
