"""Targeted CPU tests for Mission 02 (Virtual PIV) post-processing pipeline.

Uses small synthetic test fixtures + the real frozen observation spec
(``configs/observations/piv_observation_spec_v1.yaml``) and frozen normalisation stats.
No training, no neural inference, no GPU jobs, and NO access to TEST.
Synthetic outputs live strictly inside pytest ``tmp_path`` and are never scientific results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/observations/piv_observation_spec_v1.yaml"
SPEC_SHA_PATH = ROOT / "configs/observations/piv_observation_spec_v1.sha256"
STATS_PATH = ROOT / "data/manifests/train_norm_stats_month2_fit.json"
STATS_SHA = "678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e"

from combustion_recon.data.nonideal_observations import load_velocity_scale
from combustion_recon.data.piv_observations import PivObservationGenerator, load_piv_spec
from combustion_recon.evaluation import piv_accounting as pa
from combustion_recon.evaluation import piv_aggregate as pagg
from combustion_recon.evaluation import piv_eval as pe
from combustion_recon.evaluation import piv_manifest as pm
from combustion_recon.evaluation import piv_metrics as pmet


@pytest.fixture(scope="module")
def piv_spec():
    return load_piv_spec(SPEC_PATH, expected_sha256=SPEC_SHA_PATH.read_text().split()[0])


@pytest.fixture(scope="module")
def piv_gen(piv_spec):
    scale = load_velocity_scale(STATS_PATH, expected_sha256=STATS_SHA)
    x = np.linspace(-0.036907, 0.036888, 128)
    y = np.linspace(0.062853, 0.159134, 128)
    return PivObservationGenerator(piv_spec, scale, x_vec_m=x, y_vec_m=y)


# --------------------------------------------------------------------------- 1. Metrics tests

def test_isotherm_boundary_distance_known_shift_and_empty_flags():
    dr, dc = pmet.DEFAULT_STEPS_MM
    H, W = 128, 128
    # Vertical isotherm interface at column 60 (true) vs column 64 (pred) => shift = 4 * dc
    true_T = np.full((H, W), 1500.0)
    pred_T = np.full((H, W), 1500.0)
    true_T[:, 60:] = 2200.0
    pred_T[:, 64:] = 2200.0

    r = pmet.isotherm_boundary_distance_mm(pred_T, true_T, threshold_K=2100.0, steps_mm=(dr, dc))
    assert r["iso2100_valid"] is True
    assert r["iso2100_true_empty"] is False and r["iso2100_pred_empty"] is False
    assert np.isclose(r["iso2100_sym_dist_mm"], 4.0 * dc, atol=1e-6)

    # Identical fields => exact 0.0 mm
    r0 = pmet.isotherm_boundary_distance_mm(true_T, true_T, threshold_K=1800.0, steps_mm=(dr, dc))
    assert r0["iso1800_valid"] is True
    assert r0["iso1800_sym_dist_mm"] == 0.0

    # Empty predicted mask => NaN + explicit flags (never silently 0)
    cold = np.full((H, W), 1200.0)
    re_ = pmet.isotherm_boundary_distance_mm(cold, true_T, threshold_K=2100.0, steps_mm=(dr, dc))
    assert re_["iso2100_valid"] is False
    assert re_["iso2100_pred_empty"] is True
    assert re_["iso2100_true_empty"] is False
    assert np.isnan(re_["iso2100_sym_dist_mm"])


def test_distance_bins_and_rej_nodes_with_real_piv_generator(piv_gen):
    rng = np.random.default_rng(11)
    vi = rng.normal(0.0, 8.0, (128, 128))
    vj = rng.normal(10.0, 8.0, (128, 128))
    true_T = np.full((128, 128), 1900.0)
    true_T[30:95, 50:110] = 2250.0
    true_T[55:75, 95:125] = 1400.0

    # PIV-B-in has large outside-field region covering all distance bins
    po_b = piv_gen.observe("PIV-B-in", vi, vj, sim_id="0NH3_1.05.h5", frame_index=400, role="eval", replica=0, return_intermediates=True)
    inter = po_b.intermediates
    fld_mask = pmet.field_rect_mask(po_b.meta["field_rect_rows_cols"])
    pred_T = true_T + np.where(fld_mask, 15.0, 80.0)
    row_b = pmet.evaluate_piv_frame(
        pred_T,
        true_T,
        train_sigma_T=300.0,
        sim_id="0NH3_1.05.h5",
        split="val",
        role="val_eval",
        frame_index=400,
        observed_mask=inter["observed"],
        valid_mask=inter["valid"],
        field_mask=fld_mask,
    )
    assert np.isclose(row_b["in_field_rmse_K"], 15.0, atol=1e-6)
    assert np.isclose(row_b["out_field_rmse_K"], 80.0, atol=1e-6)
    assert row_b["dist_d0_n"] == int(inter["observed"].sum()) == 1824
    assert row_b["dist_dgt20_n"] > 0
    assert row_b["dist_d0_rmse_K"] < row_b["dist_dgt20_rmse_K"]

    # SENS-REJ-clu-sub-A: rejected/substituted node decomposition
    po_r = piv_gen.observe("SENS-REJ-clu-sub-A", vi, vj, sim_id="0NH3_1.05.h5", frame_index=400, role="eval", replica=0, return_intermediates=True)
    ir = po_r.intermediates
    pred_r = true_T + np.where(ir["rejected"], 120.0, 10.0)
    row_r = pmet.evaluate_piv_frame(
        pred_r,
        true_T,
        train_sigma_T=300.0,
        sim_id="0NH3_1.05.h5",
        split="val",
        role="val_eval",
        frame_index=400,
        observed_mask=ir["observed"],
        valid_mask=ir["valid"],
        field_mask=pmet.field_rect_mask(po_r.meta["field_rect_rows_cols"]),
        rejected_mask=ir["rejected"],
        substituted_mask=ir["rejected"],
    )
    assert row_r["has_rejections"] is True
    assert row_r["rej_nodes_n"] == int(ir["rejected"].sum()) > 0
    assert np.isclose(row_r["rej_nodes_rmse_K"], 120.0, atol=1e-6)
    assert np.isclose(row_r["sub_nodes_rmse_K"], 120.0, atol=1e-6)
    assert np.isclose(row_r["nonrej_valid_rmse_K"], 10.0, atol=1e-6)


def test_spatial_accumulator_and_interseed_stats():
    rng = np.random.default_rng(42)
    acc = pmet.SpatialErrorAccumulator(hw=(128, 128))
    errs = []
    for k in range(4):
        t = rng.normal(2000.0, 50.0, (128, 128))
        p = t + (k - 1.5) * 10.0 + rng.normal(0.0, 5.0, (128, 128))
        acc.update("0NH3_1.05.h5", p, t)
        errs.append(p - t)
    d = acc.to_npz_dict()
    stack = np.stack(errs, axis=0)
    assert np.allclose(d["cond|0NH3_1.05.h5|mean_error_K"], stack.mean(axis=0), atol=1e-4)
    assert np.allclose(d["cond|0NH3_1.05.h5|rmse_K"], np.sqrt((stack**2).mean(axis=0)), atol=1e-4)

    true = np.full((16, 16), 2000.0)
    preds = {0: true + 20.0, 1: true - 20.0}
    ist = pmet.compute_interseed_field_stats(preds, true)
    assert np.isclose(ist["mean_of_seed_rmse_K"], 20.0)
    assert np.isclose(ist["rmse_of_mean_pred_K"], 0.0)
    assert np.isclose(ist["mean_spatial_seed_std_K"], np.std([20.0, -20.0], ddof=1))


# --------------------------------------------------------------------------- 2. Manifest & TEST refusal & Status auditor

def test_manifest_refuses_test_roles(tmp_path):
    m = {
        "schema": pm.MANIFEST_SCHEMA,
        "authorized_roles": ["val_eval", "test_interp_eval"],
        "runs": [
            {"run_id": "R1", "group": "T1", "family": "sparse_unet", "train_cell": "IDEAL-A", "model_seed": 0}
        ],
    }
    mp = tmp_path / "bad_manifest.json"
    mp.write_text(json.dumps(m))
    with pytest.raises(pm.ManifestError, match="TEST role"):
        pm.load_manifest(mp)


def test_status_auditor_separates_all_run_states(tmp_path, piv_spec):
    tr_root = tmp_path / "train"
    ev_root = tmp_path / "eval"

    def _mk_train(rid, status, stop, epochs, best_ep=100, init_sha="init_aaa", make_ckpts=True):
        d = tr_root / rid
        d.mkdir(parents=True)
        (d / "run_identity.json").write_text(json.dumps({
            "run_id": rid, "family": "sparse_unet", "p_conditioning": True, "git_sha": "abc123",
            "piv_observation": {"spec_sha256": piv_spec.sha256},
        }))
        (d / "init_manifest.json").write_text(json.dumps({"init_state_dict_sha256": init_sha}))
        if status is not None:
            (d / "summary.json").write_text(json.dumps({
                "status": status, "stop_reason": stop, "epochs_completed": epochs,
                "best_val_epoch": best_ep, "best_val_metric": 42.0, "elapsed_s": 120.0,
            }))
        if make_ckpts:
            (d / "checkpoints").mkdir()
            (d / "checkpoints/best.pt").write_bytes(b"best_bytes_" + rid.encode())
            (d / "checkpoints/last.pt").write_bytes(b"last_bytes_" + rid.encode())

    # 5 runs covering all 5 training states:
    _mk_train("R_plateau", "stopped_early_plateau", "early_stop", 420)
    _mk_train("R_budget", "completed", "reached_max_epochs", 1200)
    _mk_train("R_incomplete", "stopped_wall_time", "wall_time_cap", 300)
    _mk_train("R_failed", "stopped_error", "exception:RuntimeError:oom", 50)
    # R_missing has no training_dir created

    # Mark R_plateau best eval complete & last SAME_AS_BEST; R_budget eval failed gate
    eb = ev_root / "R_plateau" / "best"
    (eb / "per_frame/PIV-A/val_eval").mkdir(parents=True)
    pd.DataFrame([{"rmse_K": 40.0, "mae_K": 25.0, "bias_K": 0.5}]).to_csv(eb / "per_frame/PIV-A/val_eval/rep0.csv", index=False)
    (eb / "gate.json").write_text(json.dumps({"all_pass": True}))
    (eb / "provenance.json").write_text(json.dumps({"spec_sha256": piv_spec.sha256, "checkpoint_epoch": 100}))
    (eb / "COMPLETE.json").write_text(json.dumps({"ok": True}))
    el = ev_root / "R_plateau" / "last"
    el.mkdir(parents=True)
    (el / "SAME_AS_BEST.json").write_text(json.dumps({"reason": "same"}))

    ef = ev_root / "R_budget" / "best"
    ef.mkdir(parents=True)
    (ef / "gate.json").write_text(json.dumps({"all_pass": False}))
    (ef / "GATE_FAILED.json").write_text(json.dumps({"failed": True}))

    manifest = {
        "schema": pm.MANIFEST_SCHEMA,
        "observation_spec_sha256": piv_spec.sha256,
        "training_git_sha": "abc123",
        "authorized_roles": ["val_eval"],
        "expected_replicas": [0],
        "expected_checkpoints": ["best", "last"],
        "eval_configs": ["PIV-A"],
        "runs": [
            {"run_id": rid, "group": "T2", "family": "sparse_unet", "train_cell": "PIV-A", "train_config_id": "PIV-A",
             "model_seed": i, "training_dir": str(tr_root / rid), "eval_dir": str(ev_root / rid)}
            for i, rid in enumerate(["R_plateau", "R_budget", "R_incomplete", "R_failed", "R_missing"])
        ],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    loaded = pm.load_manifest(mp)
    aud = pm.audit_manifest(loaded)
    counts = aud["summary"]["training_status_counts"]
    assert counts["plateau"] == 1
    assert counts["budget_exhausted"] == 1
    assert counts["incomplete"] == 1
    assert counts["failed"] == 1
    assert counts["missing"] == 1
    assert aud["summary"]["eval_status_counts"]["complete"] == 1
    assert aud["summary"]["eval_status_counts"]["same_as_best"] == 1
    assert aud["summary"]["eval_status_counts"]["failed"] == 1
    assert aud["summary"]["strict_complete"] is False

    md = pm.render_status_report_md(aud)
    for tok in ("plateau", "budget_exhausted", "incomplete", "failed", "missing"):
        assert tok in md


# --------------------------------------------------------------------------- 3. Resumable eval + full end-to-end CPU pipeline

def _synthetic_frames(role: str) -> list[dict]:
    rng = np.random.default_rng(100 if role == "val_eval" else 200)
    conds = ["0NH3_1.05.h5", "20NH3_1.1.h5", "40NH3_0.9.h5"] if role == "val_eval" else ["80NH3_1.2.h5", "40NH3_1.1.h5", "0NH3_0.85.h5"]
    fidxs = [400, 1000] if role == "val_eval" else [1600, 2000]
    frames = []
    yy, xx = np.mgrid[0:128, 0:128]
    for ci, cond in enumerate(conds):
        for fi in fidxs:
            vi = 12.0 * np.sin(xx / 15.0 + ci) + rng.normal(0.0, 2.0, (128, 128))
            vj = 20.0 * np.cos((yy - 64.0) / 20.0) + rng.normal(0.0, 2.0, (128, 128))
            # Realistic thermal structure with hot core (>=2100 K) and cold inlet jet (<1800 K)
            true_T = 1950.0 + 320.0 * np.exp(-((yy - 64.0) ** 2 + (xx - 55.0) ** 2) / 600.0)
            true_T -= 650.0 * np.exp(-((yy - 64.0) ** 2 + (xx - 115.0) ** 2) / 220.0)
            true_T += 15.0 * ci + 0.01 * fi
            frames.append({
                "sim_id": cond,
                "split": "val" if role == "val_eval" else "train",
                "frame_index": fi,
                "vi_phys": vi,
                "vj_phys": vj,
                "true_T_K": true_T,
            })
    return frames


def test_resumable_eval_and_full_postprocessing_end_to_end(tmp_path, piv_spec, piv_gen):
    """Exercises piv_eval (resumable), audit, aggregate, accounting, main figs, atlas, resume & pruner."""
    frames_by_role = {
        "val_eval": _synthetic_frames("val_eval"),
        "seen_train_eval": _synthetic_frames("seen_train_eval"),
    }
    eval_configs = [
        "IDEAL-full",
        "IDEAL-A",
        "FILT-A",
        "PIV-A",
        "IDEAL-B-in",
        "PIV-B-in",
        "PIV-B-down",
        "PIV-B-off",
        "SENS-TH-A",
        "SENS-REJ-clu-flag-A",
        "SENS-REJ-clu-sub-A",
    ]
    roles = ["val_eval", "seen_train_eval"]
    replicas = [0, 1]

    tr_root = tmp_path / "train"
    ev_root = tmp_path / "eval"
    led_root = tr_root / "_ledger"
    led_root.mkdir(parents=True)

    # Mini-campaign covering U-Net (T1, T2, T3, T4 x 2 seeds, one budget_exhausted) and GLU (T5, T6, T7 x 2 seeds)
    run_specs = [
        ("PIV-T1-unet-IDEAL-A-s0", "T1", "sparse_unet", "IDEAL-A", "IDEAL-A", None, 0, False, 30.0),
        ("PIV-T1-unet-IDEAL-A-s1", "T1", "sparse_unet", "IDEAL-A", "IDEAL-A", None, 1, False, 32.0),
        ("PIV-T2-unet-PIV-A-s0", "T2", "sparse_unet", "PIV-A", "PIV-A", None, 0, False, 34.0),
        ("PIV-T2-unet-PIV-A-s1", "T2", "sparse_unet", "PIV-A", "PIV-A", None, 1, True, 36.0),
        ("PIV-T3-unet-IDEAL-B-s0", "T3", "sparse_unet", "IDEAL-B", "IDEAL-B-in", "B-in", 0, False, 48.0),
        ("PIV-T3-unet-IDEAL-B-s1", "T3", "sparse_unet", "IDEAL-B", "IDEAL-B-in", "B-in", 1, False, 50.0),
        ("PIV-T4-unet-PIV-B-s0", "T4", "sparse_unet", "PIV-B", "PIV-B-in", "B-in", 0, False, 53.0),
        ("PIV-T4-unet-PIV-B-s1", "T4", "sparse_unet", "PIV-B", "PIV-B-in", "B-in", 1, False, 55.0),
        ("PIV-T5-glu-PIV-A-s0", "T5", "adapted_glu", "PIV-A", "PIV-A", None, 0, False, 38.0),
        ("PIV-T5-glu-PIV-A-s1", "T5", "adapted_glu", "PIV-A", "PIV-A", None, 1, True, 40.0),
        ("PIV-T6-glu-PIV-B-s0", "T6", "adapted_glu", "PIV-B", "PIV-B-in", "B-in", 0, False, 58.0),
        ("PIV-T7-glu-BRIDGE-s0", "T7", "adapted_glu", "BRIDGE-5pct-IDEAL", "IDEAL-A", None, 0, False, 44.0),
    ]

    manifest_runs = []
    for rid, grp, fam, tcell, tcfg, bpos, sd, is_bud, base_err in run_specs:
        td = tr_root / rid
        (td / "checkpoints").mkdir(parents=True)
        (td / "run_identity.json").write_text(json.dumps({
            "run_id": rid, "family": fam, "p_conditioning": True, "git_sha": "deadbeef",
            "piv_observation": {"spec_sha256": piv_spec.sha256},
        }))
        (td / "init_manifest.json").write_text(json.dumps({"init_state_dict_sha256": f"init_{fam}_s{sd}"}))
        epochs = 1200 if is_bud else 450
        best_ep = 950 if is_bud else 360
        (td / "summary.json").write_text(json.dumps({
            "status": "completed" if is_bud else "stopped_early_plateau",
            "stop_reason": "reached_max_epochs" if is_bud else "early_stop",
            "epochs_completed": epochs,
            "best_val_epoch": best_ep,
            "best_val_metric": base_err,
            "clip_events": 2,
            "elapsed_s": 3600.0 * (2.5 if fam == "adapted_glu" else 1.2),
        }))
        (td / "checkpoints/best.pt").write_bytes(f"best_{rid}".encode())
        (td / "checkpoints/last.pt").write_bytes(f"last_{rid}".encode())
        # Synthetic training & validation curves matching SparseScientificTrainer format
        c_rows = [{"epoch": e, "update": e * 10, "loss": 0.4 * np.exp(-e / 150.0) + 0.02, "lr": 1e-3 if e < 300 else 5e-4, "grad_norm_pre_clip": 0.6} for e in range(1, 21)]
        pd.DataFrame(c_rows).to_csv(td / "curves.csv.gz", index=False, compression="gzip")
        eh = [{"epoch": e * 20, "lr": 1e-3, "val_mean_condition_rmse_K": base_err + 12.0 * np.exp(-e / 3.0) + (0.4 if e > 8 else 0.0), "seen_mean_condition_rmse_K": base_err * 0.85} for e in range(1, 12)]
        (td / "eval_history.json").write_text(json.dumps(eh))
        (led_root / f"{rid}.jsonl").write_text(
            json.dumps({"event": "attempt_start", "partition": "mit_preemptable" if is_bud else "mit_normal_gpu"}) + "\n"
            + (json.dumps({"event": "signal"}) + "\n" + json.dumps({"event": "extension_start"}) + "\n" if is_bud else "")
        )

        # Dummy deterministic forward callable (pure numpy, NO neural inference)
        def _make_fwd(err_scale: float, train_cfg: str):
            def _fwd(packets):
                preds = []
                for pkt in packets:
                    t = pkt["frame"]["true_T_K"]
                    po = pkt["piv_obs"]
                    obs_m = po.intermediates["observed"]
                    mismatch_pen = 1.0 if po.config_id == train_cfg else (1.35 if "B-" in po.config_id else 1.18)
                    # Higher error outside observed region so distance-binned metrics are realistic
                    spatial_err = np.where(obs_m, err_scale * 0.7, err_scale * 1.6) * mismatch_pen
                    # Deterministic spatial ripple for non-trivial bias/cRMSE
                    yy, xx = np.mgrid[0:128, 0:128]
                    ripple = np.sin(xx / 10.0) * (0.4 * err_scale) + 2.5
                    preds.append(t + spatial_err * 0.5 + ripple)
                return np.stack(preds, axis=0)
            return _fwd

        ed = ev_root / rid
        tags_best = {
            "run_id": rid, "family": fam, "variant": "p_on", "model_seed": sd,
            "checkpoint_kind": "best", "checkpoint_epoch": best_ep,
            "checkpoint_file_sha256": hashlib.sha256(f"best_{rid}".encode()).hexdigest(),
            "checkpoint_weights_sha256": hashlib.sha256(f"wbest_{rid}".encode()).hexdigest(),
            "eval_git_sha": "deadbeef",
        }
        res1 = pe.run_resumable_checkpoint_eval(
            out_kind_dir=ed / "best",
            frames_by_role=frames_by_role,
            generator=piv_gen,
            eval_configs=eval_configs,
            roles=roles,
            replicas=replicas,
            forward_fn=_make_fwd(base_err, tcfg),
            train_sigma_T=300.0,
            run_tags=tags_best,
            resume=True,
            save_fields=True,
        )
        assert res1["computed"] == len(eval_configs) * len(roles) * len(replicas)
        assert res1["skipped"] == 0

        # Test eval resumability on first run: second call must skip 100% and delete-one recomputes 1
        if rid == run_specs[0][0]:
            res2 = pe.run_resumable_checkpoint_eval(
                out_kind_dir=ed / "best",
                frames_by_role=frames_by_role,
                generator=piv_gen,
                eval_configs=eval_configs,
                roles=roles,
                replicas=replicas,
                forward_fn=_make_fwd(base_err, tcfg),
                train_sigma_T=300.0,
                run_tags=tags_best,
                resume=True,
                save_fields=True,
            )
            assert res2["computed"] == 0
            assert res2["skipped"] == len(eval_configs) * len(roles) * len(replicas)

            # Remove one unit CSV -> only 1 recomputed
            one_csv = ed / "best/per_frame/PIV-A/val_eval/rep0.csv"
            one_csv.unlink()
            res3 = pe.run_resumable_checkpoint_eval(
                out_kind_dir=ed / "best",
                frames_by_role=frames_by_role,
                generator=piv_gen,
                eval_configs=eval_configs,
                roles=roles,
                replicas=replicas,
                forward_fn=_make_fwd(base_err, tcfg),
                train_sigma_T=300.0,
                run_tags=tags_best,
                resume=True,
                save_fields=True,
            )
            assert res3["computed"] == 1
            assert res3["skipped"] == len(eval_configs) * len(roles) * len(replicas) - 1

        # Add a fake Tier-2 scratch file on best to test prune_scratch_fields.py
        (ed / "best/scratch_val_rep0_diag.npz").write_bytes(b"0" * 4096)

        # For seed 0 mark last == SAME_AS_BEST; for seed 1 run distinct last eval (slightly higher error)
        if sd == 0:
            (ed / "last").mkdir(parents=True, exist_ok=True)
            (ed / "last/SAME_AS_BEST.json").write_text(json.dumps({"weights_sha256": tags_best["checkpoint_weights_sha256"]}))
        else:
            tags_last = {**tags_best, "checkpoint_kind": "last", "checkpoint_epoch": epochs,
                         "checkpoint_weights_sha256": hashlib.sha256(f"wlast_{rid}".encode()).hexdigest()}
            pe.run_resumable_checkpoint_eval(
                out_kind_dir=ed / "last",
                frames_by_role=frames_by_role,
                generator=piv_gen,
                eval_configs=eval_configs,
                roles=roles,
                replicas=replicas,
                forward_fn=_make_fwd(base_err + 1.5, tcfg),
                train_sigma_T=300.0,
                run_tags=tags_last,
                resume=True,
                save_fields=False,
            )

        manifest_runs.append({
            "run_id": rid, "group": grp, "family": fam, "variant": "p_on",
            "train_cell": tcell, "train_config_id": tcfg, "b_position": bpos,
            "model_seed": sd, "conditional": False,
            "training_dir": str(td), "eval_dir": str(ed), "ledger_path": str(led_root / f"{rid}.jsonl"),
        })

    # Sample sacct -D -P file with PREEMPTED + COMPLETED records
    sacct_p = tmp_path / "sacct_test.psv"
    sacct_p.write_text(
        "JobID|JobName|Partition|QOS|Account|State|ExitCode|Submit|Start|End|ElapsedRaw|AllocCPUS|AllocTRES|NodeList|Timelimit|MaxRSS|TotalCPU\n"
        "90001_0|piv_train_normal|mit_normal_gpu|mit_amf_advanced_gpu|mit_amf_advanced_gpu|COMPLETED|0:0|2026-11-28T00:00:00|2026-11-28T00:01:00|2026-11-28T01:01:00|3600|8|cpu=8,gres/gpu=1|node01|12:00:00|4G|01:00:00\n"
        "90001_1|piv_train_preempt|mit_preemptable|mit_preemptable|mit_amf_advanced_gpu|PREEMPTED|0:15|2026-11-28T00:00:00|2026-11-28T00:05:00|2026-11-28T01:05:00|3600|8|cpu=8,gres/gpu=1|node02|48:00:00|4G|01:00:00\n"
        "90001_1|piv_train_preempt|mit_preemptable|mit_preemptable|mit_amf_advanced_gpu|COMPLETED|0:0|2026-11-28T00:00:00|2026-11-28T01:10:00|2026-11-28T03:10:00|7200|8|cpu=8,gres/gpu=1|node02|48:00:00|4G|02:00:00\n"
    )
    # Sample Mission 00 baseline CSV for T7 bridge
    m00_csv = tmp_path / "m00_base.csv"
    pd.DataFrame([
        {"family": "adapted_glu", "variant": "p_on", "cell": "R0", "role": "val_eval", "rmse_K": 43.0},
        {"family": "adapted_glu", "variant": "p_on", "cell": "R0", "role": "seen_train_eval", "rmse_K": 39.0},
    ]).to_csv(m00_csv, index=False)

    manifest = {
        "schema": pm.MANIFEST_SCHEMA,
        "campaign_phase": "phase4_campaign",
        "observation_spec_path": str(SPEC_PATH),
        "observation_spec_sha256": piv_spec.sha256,
        "normalization_stats_path": str(STATS_PATH),
        "normalization_stats_sha256": STATS_SHA,
        "training_git_sha": "deadbeef",
        "authorized_roles": roles,
        "expected_replicas": replicas,
        "expected_checkpoints": ["best", "last"],
        "eval_configs": eval_configs,
        "mission00_baseline_csv": str(m00_csv),
        "slurm_accounting": {
            "sacct_psv": str(sacct_p),
            "array_run_map": {"90001": ["PIV-T1-unet-IDEAL-A-s0", "PIV-T2-unet-PIV-A-s1"]},
        },
        "runs": manifest_runs,
    }
    mp = tmp_path / "manifest_full.json"
    mp.write_text(json.dumps(manifest))

    # Run full post-processing CLI
    dest = tmp_path / "pkg_out"
    cmd = [sys.executable, str(ROOT / "scripts/piv_postproc/run_postprocessing.py"),
           "--manifest", str(mp), "--dest", str(dest), "--n-boot", "40"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "COMPLETE.json").is_file()
    assert (dest / "INDEX.md").is_file()
    assert (dest / "provenance.json").is_file()
    assert (dest / "package_manifest.json").is_file()

    # Check confirmatory & exploratory tables exist
    for rel in (
        "tables/confirmatory/per_condition.csv.gz",
        "tables/confirmatory/macro_by_replica.csv.gz",
        "tables/confirmatory/per_run_scenario.csv",
        "tables/confirmatory/cell_seed_summary.csv",
        "tables/confirmatory/confirmatory_contrasts.csv",
        "tables/confirmatory/r4_decision_indicators.json",
        "tables/confirmatory/transfer_matrix_best.csv",
        "tables/confirmatory/sensitivity_summary.csv",
        "tables/confirmatory/b_positions_and_distance.csv",
        "tables/confirmatory/rej_node_metrics.csv",
        "tables/confirmatory/glu_bridge_t7.csv",
        "tables/confirmatory/hierarchical_bootstrap_val.csv",
        "tables/exploratory/frame_error_quantiles.csv",
        "tables/exploratory/paired_frame_deltas.csv.gz",
        "tables/exploratory/condition_vs_vref.csv",
        "tables/exploratory/training_curves_summary.csv",
        "tables/exploratory/selected_cases.json",
        "tables/exploratory/compute_accounting_jobs.csv",
        "tables/exploratory/compute_accounting_by_run.csv",
    ):
        assert (dest / rel).is_file(), f"missing {rel}"

    # Verify sacct -D summed BOTH preempted + completed attempts for PIV-T2-unet-PIV-A-s1 (1h + 2h = 3h)
    by_run = pd.read_csv(dest / "tables/exploratory/compute_accounting_by_run.csv")
    row_pre = by_run[by_run["run_id"] == "PIV-T2-unet-PIV-A-s1"].iloc[0]
    assert np.isclose(row_pre["sacct_gpu_h"], 3.0)
    assert int(row_pre["sacct_preemptions"]) == 1

    # Verify R4 indicators JSON populated
    r4 = json.loads((dest / "tables/confirmatory/r4_decision_indicators.json").read_text())
    assert r4["D_REJ"]["status"] == "evaluated"
    assert r4["D_kern_train_T10"]["status"] == "evaluated"
    assert r4["D_mix"]["status"] == "evaluated"

    # Check main figures & all 7 atlas sections generated
    main_pngs = sorted((dest / "figures/main").glob("figM*.png"))
    assert len(main_pngs) == 8, [p.name for p in main_pngs]
    for sec in (
        "01_training_curves",
        "02_distributions_and_paired",
        "03_by_condition_and_region",
        "04_spatial_maps_and_seed_var",
        "05_transfer_and_coverage",
        "06_case_studies",
        "07_compute_and_status",
    ):
        pngs = list((dest / "figures/atlas" / sec).glob("*.png"))
        assert len(pngs) >= 1, f"no figures in atlas section {sec}"

    # Resumability test: second run skips every stage and preserves exact package hashes
    pkg_before = json.loads((dest / "package_manifest.json").read_text())
    proc2 = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc2.returncode == 0
    assert "up-to-date (skipped)" in proc2.stdout
    pkg_after = json.loads((dest / "package_manifest.json").read_text())
    assert pkg_after["files"]["tables/confirmatory/per_run_scenario.csv"] == pkg_before["files"]["tables/confirmatory/per_run_scenario.csv"]

    # Pruner test: dry-run preserves scratch, --execute deletes scratch while preserving Tier-1 files
    prune_cmd = [sys.executable, str(ROOT / "scripts/piv_postproc/prune_scratch_fields.py"),
                 "--manifest", str(mp), "--package-dest", str(dest)]
    p_dry = subprocess.run(prune_cmd, capture_output=True, text=True, check=True)
    assert "DRY-RUN: 12 scratch files" in p_dry.stdout
    assert (ev_root / "PIV-T1-unet-IDEAL-A-s0/best/scratch_val_rep0_diag.npz").is_file()

    p_exec = subprocess.run(prune_cmd + ["--execute"], capture_output=True, text=True, check=True)
    assert "DELETED: 12 scratch files" in p_exec.stdout
    assert not (ev_root / "PIV-T1-unet-IDEAL-A-s0/best/scratch_val_rep0_diag.npz").exists()
    assert (ev_root / "PIV-T1-unet-IDEAL-A-s0/best/fields_selected_best.npz").is_file()

    # Partial / missing run handling: strict blocks (rc=3), --allow-partial succeeds with PARTIAL_DIAGNOSTIC.json
    manifest_partial = json.loads(mp.read_text())
    manifest_partial["runs"].append({
        "run_id": "PIV-T2-unet-PIV-A-s4", "group": "T2", "family": "sparse_unet", "variant": "p_on",
        "train_cell": "PIV-A", "train_config_id": "PIV-A", "b_position": None, "model_seed": 4,
        "conditional": False, "training_dir": str(tr_root / "DOES_NOT_EXIST"), "eval_dir": str(ev_root / "DOES_NOT_EXIST"),
    })
    mp_part = tmp_path / "manifest_partial.json"
    mp_part.write_text(json.dumps(manifest_partial))

    dest_blk = tmp_path / "pkg_blocked"
    p_blk = subprocess.run(
        [sys.executable, str(ROOT / "scripts/piv_postproc/run_postprocessing.py"), "--manifest", str(mp_part), "--dest", str(dest_blk)],
        capture_output=True, text=True, check=False,
    )
    assert p_blk.returncode == 3
    assert (dest_blk / "BLOCK_REPORT.md").is_file()
    assert not (dest_blk / "COMPLETE.json").exists()

    dest_part = tmp_path / "pkg_partial"
    p_part = subprocess.run(
        [sys.executable, str(ROOT / "scripts/piv_postproc/run_postprocessing.py"), "--manifest", str(mp_part), "--dest", str(dest_part), "--allow-partial", "--n-boot", "20"],
        capture_output=True, text=True, check=False,
    )
    assert p_part.returncode == 0, p_part.stdout + p_part.stderr
    assert (dest_part / "PARTIAL_DIAGNOSTIC.json").is_file()
    assert not (dest_part / "COMPLETE.json").exists()


def test_slurm_chain_dry_run_and_example_manifest_schema(tmp_path):
    # Example manifest loads cleanly under load_manifest (no TEST roles, valid schema)
    ex_path = ROOT / "configs/piv/piv_postproc_manifest_example.json"
    ex = pm.load_manifest(ex_path)
    assert ex["schema"] == pm.MANIFEST_SCHEMA
    assert ex["observation_spec_sha256"] == SPEC_SHA_PATH.read_text().split()[0]

    # Slurm helper dry-run produces dependency wiring without calling real sbatch
    res = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/engaging/piv_submit_postproc_chain.sh"),
            "--tip", str(ROOT),
            "--sha", "0000000",
            "--manifest", str(ex_path),
            "--dest", str(tmp_path / "slurm_dry"),
            "--train-jobs", "11111:22222",
            "--eval-cmd", "scripts/piv_phase4/eval_piv_run.py",
            "--run-ids", "R1:R2",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "stage1_freeze=DRY_JOB_ID" in res.stdout
    assert "stage2_eval_array=DRY_JOB_ID (n_runs=2)" in res.stdout
    assert "stage3_postproc=DRY_JOB_ID" in res.stdout
    assert "--dependency=afterany:11111:22222" in res.stderr
    assert "--dependency=afterok:DRY_JOB_ID" in res.stderr
