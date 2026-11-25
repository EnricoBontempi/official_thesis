"""Synthetic success and refusal for the specialist graphics package. No cluster and no training."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combustion_recon.evaluation import specialist_graphics as sg

H = W = 8
VAL = ["v0", "v1"]
SEEN = ["seenA"]
CELLS = list(sg.CELLS)


def _runs():
    out = []
    for token, family in (("unet", "sparse_unet"), ("glu", "adapted_glu")):
        for seed in (0, 1, 2):
            out.append({
                "run_id": f"NI-{token}-on-N1-s{seed}",
                "family": family,
                "regime": "N1",
                "seed": seed,
                "weights": f"weights-{token}-{seed}",
            })
    return out


def _field_arrays(cond_i: int, seed: int):
    true = (1800 + 40 * cond_i + np.arange(H * W, dtype=np.float32).reshape(H, W) * 0.1)
    bump = np.zeros((H, W), np.float32)
    bump[0, 0] = np.float32(30 * (seed - 1))
    pred = true + np.float32(4) + bump
    err = (pred - true).astype(np.float32)
    rmse = float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
    return true, pred.astype(np.float32), err, rmse


def _spec():
    return sg.Spec(
        expect_runs=tuple(r["run_id"] for r in _runs()),
        shape=(H, W),
        frame_index=1000,
        replica=0,
        n_val_conditions=len(VAL),
        nominal_rows_best=len(CELLS) * (len(VAL) + len(SEEN)) * 2,
        baseline_sha256=None,
        err_clip_K=400.0,
    )


def _write_campaign(root: Path, *, fault: str | None = None) -> dict:
    runs = _runs()
    if fault == "drop_run":
        runs = runs[:-1]
    spec = _spec()
    eval_root = root / "eval"
    package = root / "package"
    (package / "tables").mkdir(parents=True)
    (package / "COMPLETE.json").write_text('{"ok": true}\n')
    freeze_runs = {}
    index_runs = []
    frames = []
    for run in runs:
        best = eval_root / run["run_id"] / "best"
        (best / "per_frame").mkdir(parents=True)
        (best / "COMPLETE.json").write_text('{"ok": true}\n')
        (best / "gate.json").write_text(json.dumps({"all_pass": True}) + "\n")
        rows = []
        for cell in CELLS:
            for role, conds in (("val_eval", VAL), ("seen_train_eval", SEEN)):
                for cond_i, cond in enumerate(conds):
                    field_rmse = _field_arrays(cond_i if role == "val_eval" else 0, run["seed"])[3]
                    for frame in (1000, 0):
                        rmse = field_rmse if (role == "val_eval" and frame == 1000) else 12.0 + run["seed"]
                        rows.append({
                            "run_id": run["run_id"], "family": run["family"], "variant": "p_on",
                            "model_seed": run["seed"], "cell": cell, "role": role, "replica": 0,
                            "condition": cond, "frame_index": frame,
                            "rmse_K": rmse, "mae_K": rmse * 0.8, "bias_K": -1.0, "abs_bias_K": 1.0,
                            "centered_rmse_K": rmse * 0.9, "jet_rmse_K": rmse + 1, "high_rmse_K": rmse + 2,
                            "peak_rmse_K": rmse + 3,
                            "surviving_sites_hash": f"h|{cell}|{role}|{cond}|{frame}",
                        })
        pd.DataFrame(rows).to_csv(best / "per_frame" / "rep0.csv", index=False)
        frames.extend(rows)
        arrays = {"x_m": np.linspace(0, 1, W), "y_m": np.linspace(0, 1, H)}
        entries = []
        for cond_i, cond in enumerate(VAL):
            true, pred, err, rmse = _field_arrays(cond_i, run["seed"])
            if fault == "bad_error" and run["seed"] == 0 and cond_i == 0:
                err = (-err).astype(np.float32)
            stated = rmse + (1.0 if fault == "field_rmse" and run["seed"] == 0 else 0.0)
            for cell in CELLS:
                key = f"{cell}|{cond}"
                arrays[f"{key}|pred_T_K"] = pred
                arrays[f"{key}|true_T_K"] = true
                arrays[f"{key}|error_signed_K"] = err
                arrays[f"{key}|original_sites"] = np.array([0, 1, 2, 3], np.int64)
                arrays[f"{key}|surviving_sites"] = np.array([0, 1], np.int64)
                arrays[f"{key}|removed_sites"] = np.array([2, 3], np.int64)
                region = np.zeros((H, W), bool) if cell in ("G1", "N1G1") else np.zeros((0, 0), bool)
                if cell in ("G1", "N1G1"):
                    region[1:3, 1:3] = True
                arrays[f"{key}|g1_region_mask"] = region
                entries.append({
                    "key": key, "cell": cell, "condition": cond, "frame_index": 0 if fault == "frame" else 1000,
                    "replica": 0, "role": "val_eval",
                    "rmse_K_from_fields": stated, "rmse_K_per_frame_csv": rmse,
                    "n_original": 4, "n_surviving": 2, "n_removed": 2,
                    "has_g1_region": cell in ("G1", "N1G1"),
                })
        npz = best / "fields_val_frame1000_best.npz"
        np.savez_compressed(npz, **arrays)
        digest = sg.sha256_file(npz)
        if fault == "bad_hash" and run["seed"] == 0:
            digest = "0" * 64
        manifest = {
            "format": "nonideal_specialist_fields_v1", "file": npz.name, "file_sha256": digest,
            "identity": {
                "run_id": run["run_id"], "family": run["family"], "model_seed": run["seed"],
                "checkpoint_weights_sha256": run["weights"], "train_regime": "N1",
            },
            "frame_index": 0 if fault == "frame" else 1000, "replica": 0, "role": "val_eval",
            "conditions": list(VAL), "cells": list(CELLS), "entries": entries,
        }
        (best / "fields_manifest.json").write_text(json.dumps(manifest))
        if fault == "missing_npz" and run["seed"] == 0 and run["family"] == "sparse_unet":
            npz.unlink()
        freeze_runs[run["run_id"]] = {
            "valid": True, "family": run["family"], "regime": "N1", "seed": run["seed"],
            "best": {"weights_sha256": run["weights"]}, "problems": [],
        }
        index_runs.append({
            "run_id": run["run_id"], "family": run["family"], "regime": "N1", "seed": run["seed"],
            "fields_npz": str(npz), "fields_manifest": str(best / "fields_manifest.json"),
            "fields_sha256": sg.sha256_file(npz) if npz.is_file() else digest, "n_entries": len(entries),
        })
    expect = runs if fault != "drop_run" else _runs()
    freeze = {"n_runs": len(expect), "n_valid": len(freeze_runs), "problems": [], "runs": freeze_runs}
    if fault == "drop_run":
        freeze["n_valid"] = len(freeze_runs)
        freeze["problems"] = ["one run missing"]
    (root / "freeze.json").write_text(json.dumps(freeze))
    index = {"format": "nonideal_specialist_fields_index_v1", "n_runs": len(index_runs), "runs": index_runs}
    (package / "fields_index.json").write_text(json.dumps(index))
    allf = pd.DataFrame(frames)
    macro = sg.per_scenario(allf)
    macro["checkpoint_kind"] = "best"
    macro["base_rmse_K"] = macro["rmse_K"] + 1.0
    macro["gain_vs_baseline_K"] = 1.0
    macro["delta_vs_baseline_K"] = -1.0
    macro["clean_cost_K"] = 0.5
    macro["specialist_deterioration_K"] = np.where(macro["cell"] == "R0", 0.0, 2.0)
    if fault == "package_rmse":
        macro.loc[macro.index[0], "rmse_K"] = macro.loc[macro.index[0], "rmse_K"] + 5
    macro.to_csv(package / "tables" / "per_run_scenario.csv", index=False)
    base_rows = []
    for (family, seed), g in allf.groupby(["family", "model_seed"]):
        rid = str(g["run_id"].iloc[0])
        part = allf[allf.run_id == rid].copy()
        part["run_id"] = sg.BASELINE_OF[(family, int(seed))]
        part["rmse_K"] = part["rmse_K"] + 1.0
        base_rows.append(part)
    pd.concat(base_rows, ignore_index=True).to_csv(root / "baseline.csv", index=False)
    return {
        "freeze": root / "freeze.json",
        "eval_root": eval_root,
        "package": package,
        "index": package / "fields_index.json",
        "baseline": root / "baseline.csv",
        "spec": spec,
    }


def _run(tmp: Path, fault: str | None = None, **kw):
    camp = _write_campaign(tmp / "in", fault=fault)
    dest = tmp / "graphics"
    sg.run_graphics(
        freeze_path=camp["freeze"], eval_root=camp["eval_root"], package_dir=camp["package"],
        fields_index_path=camp["index"], baseline_per_frame=camp["baseline"], dest=dest,
        spec=camp["spec"], baseline_eval_root=kw.get("baseline_eval_root"),
    )
    return dest


def test_success_writes_cards_for_every_run_and_no_block(tmp_path: Path):
    dest = _run(tmp_path)
    assert (dest / "COMPLETE.json").is_file()
    assert not (dest / "GRAPHICS_BLOCK_REPORT.md").exists()
    note = (dest / "NOTE.md").read_text()
    assert "Guadagno nello scenario di training" in note
    assert "+1.00" in note
    cards = list((dest / "cards").glob("*.pdf"))
    assert len(cards) == 6
    assert (dest / "cards" / "NI-unet-on-N1-s0" / "v0.png").is_file()
    assert (dest / "figures" / "matrix_absolute_u-net_VAL123.png").is_file()
    assert (dest / "figures" / "matrix_delta_glu_seen220.png").is_file()
    assert (dest / "variability" / "sparse_unet_N1.pdf").is_file()
    man = json.loads((dest / "figure_manifest.json").read_text())
    assert man["sources"]["n_runs"] == 6
    assert man["sources"]["frame_index"] == 1000
    assert "not a calibrated" in man["sources"]["std_across_seeds"]
    text = (dest / "INDEX.md").read_text()
    assert "NI-glu-on-N1-s2.pdf" in text
    assert "NOTE.md" in text


def test_missing_npz_blocks_without_complete(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="npz missing"):
        _run(tmp_path, "missing_npz")
    dest = tmp_path / "graphics"
    assert (dest / "GRAPHICS_BLOCK_REPORT.md").is_file()
    assert not (dest / "COMPLETE.json").exists()
    assert not (dest / "NOTE.md").exists()
    assert not list(dest.rglob("*.pdf"))


def test_bad_hash_blocks(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="hash differs"):
        _run(tmp_path, "bad_hash")
    assert (tmp_path / "graphics" / "GRAPHICS_BLOCK_REPORT.md").is_file()
    assert not (tmp_path / "graphics" / "COMPLETE.json").exists()


def test_signed_error_must_be_pred_minus_true(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="signed error is not pred-true"):
        _run(tmp_path, "bad_error")


def test_field_rmse_must_match_manifest_csv(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="field vs per-frame RMSE"):
        _run(tmp_path, "field_rmse")


def test_package_rmse_mismatch_blocks(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="package RMSE"):
        _run(tmp_path, "package_rmse")
    assert not (tmp_path / "graphics" / "COMPLETE.json").exists()


def test_frame_zero_is_not_accepted_as_frame_1000(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="frame"):
        _run(tmp_path, "frame")


def test_incomplete_freeze_blocks(tmp_path: Path):
    with pytest.raises(sg.GraphicsBlocked, match="freeze"):
        _run(tmp_path, "drop_run")


def _cli():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "70_nonideal_specialist_graphics.py"
    spec = importlib.util.spec_from_file_location("graphics_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_refuses_a_partial_matrix_against_the_production_contract(tmp_path: Path):
    camp = _write_campaign(tmp_path / "in")
    cli = _cli()
    rc = cli.main([
        "--freeze", str(camp["freeze"]), "--eval-root", str(camp["eval_root"]),
        "--package", str(camp["package"]), "--fields-index", str(camp["index"]),
        "--baseline-per-frame", str(camp["baseline"]), "--dest", str(tmp_path / "out"),
    ])
    assert rc == 3
    report = (tmp_path / "out" / "GRAPHICS_BLOCK_REPORT.md").read_text()
    assert "30" in report or "expected" in report
    assert not (tmp_path / "out" / "COMPLETE.json").exists()


def test_legacy_npz_frame_zero_is_not_called_frame_1000(tmp_path: Path):
    path = tmp_path / "fields_figure_frames.npz"
    np.savez_compressed(path, **{"R0|val_eval|v0|0|pred_T_K": np.zeros((4, 4), np.float32)})
    assert sg.legacy_field_frames(path) == [0]
    root = tmp_path / "legacy"
    (root / "O-C149-on-s0").mkdir(parents=True)
    shutil.copy(path, root / "O-C149-on-s0" / "fields_figure_frames.npz")
    text = sg.baseline_fields_needed(root)
    assert "frame presenti nel file vecchio: [0]" in text
    assert "fields_val_frame1000_best.npz" in text
    assert "O-C149-off-s0" in text
    assert "pred_T_K" in text and "Non vanno ricostruiti" in text


def test_contract_lists_the_thirty_run_ids():
    doc = json.loads((Path(__file__).resolve().parents[1] / "configs/nonideal/specialist_graphics_contract_v1.json").read_text())
    assert doc["matrix"]["n_runs"] == 30
    assert len(sg.EXPECTED_RUNS) == 30
    assert sg.EXPECTED_RUNS[0] == "NI-glu-on-N1-s0"
    assert "NI-unet-on-G1eq-s2" in sg.EXPECTED_RUNS
    assert doc["fields"]["frame_index"] == 1000
    assert doc["hook"]["output"] == "$NI_DEST/graphics"
    sbatch = (Path(__file__).resolve().parents[1] / "scripts/engaging/nonideal_graphics.sbatch").read_text()
    assert "NI_FIELDS_INDEX" in sbatch and "EXPECT_SHA:?" not in sbatch
    assert 'dest "$NI_DEST/graphics"' in sbatch


def test_observation_hash_mismatch_blocks(tmp_path: Path):
    camp = _write_campaign(tmp_path / "in")
    df = pd.read_csv(camp["baseline"])
    df.loc[0, "surviving_sites_hash"] = "changed"
    df.to_csv(camp["baseline"], index=False)
    with pytest.raises(sg.GraphicsBlocked, match="hashes"):
        sg.run_graphics(
            freeze_path=camp["freeze"], eval_root=camp["eval_root"], package_dir=camp["package"],
            fields_index_path=camp["index"], baseline_per_frame=camp["baseline"],
            dest=tmp_path / "graphics", spec=camp["spec"],
        )
