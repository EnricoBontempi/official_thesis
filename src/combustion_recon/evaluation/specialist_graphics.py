"""Graphics for the 30 p_on specialists. Reads frozen eval outputs. Does not train or infer.

The reading contract matches the field files the specialist evaluation is written to emit
(format nonideal_specialist_fields_v1): best checkpoint, VAL, frame 1000, replica 0.
Scenarios are categories. A min–max span across three training seeds is dispersion, not a
confidence interval. Mask replicas are already averaged and are not independent trainings.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

CELLS = ["R0", "N1", "N2", "G1-eq", "G1", "N1G1"]  # label order, not a physical axis
REGIMES = ["N1", "N2", "G1", "G1-eq", "N1G1"]
ROLES = ["seen_train_eval", "val_eval"]
ROLE_LABEL = {"seen_train_eval": "seen220", "val_eval": "VAL123"}
FAM_LABEL = {"sparse_unet": "U-Net", "adapted_glu": "GLU"}
FIELD_NAMES = (
    "pred_T_K", "true_T_K", "error_signed_K", "original_sites", "surviving_sites",
    "removed_sites", "g1_region_mask",
)
METRIC_COLS = (
    "rmse_K", "mae_K", "bias_K", "abs_bias_K", "centered_rmse_K",
    "jet_rmse_K", "high_rmse_K", "peak_rmse_K",
)
ERR_CLIP_K = 400.0
BASELINE_PON = (
    "O-C149-on-s0", "S7-C149-unet-on-s1", "S7-C149-unet-on-s2",
    "S7-C149-glu-on-s0", "S7-C149-glu-on-s1", "S7-C149-glu-on-s2",
)
# Same family and seed as the specialist. Observations must match this original run.
BASELINE_OF = {
    ("sparse_unet", 0): "O-C149-on-s0",
    ("sparse_unet", 1): "S7-C149-unet-on-s1",
    ("sparse_unet", 2): "S7-C149-unet-on-s2",
    ("adapted_glu", 0): "S7-C149-glu-on-s0",
    ("adapted_glu", 1): "S7-C149-glu-on-s1",
    ("adapted_glu", 2): "S7-C149-glu-on-s2",
}
BASELINE_ALL12 = BASELINE_PON + (
    "O-C149-off-s0", "S7-C149-unet-off-s1", "S7-C149-unet-off-s2",
    "S7-C149-glu-off-s0", "S7-C149-glu-off-s1", "S7-C149-glu-off-s2",
)
# Present on the cluster as fields_figure_frames.npz (Block II). Not the frame-1000 contract.
BASELINE_EVAL_ROOT_DEFAULT = "/home/bontempi/official_thesis_nonideal_b2_eval"
BASELINE_PER_FRAME_DEFAULT = "/home/bontempi/official_thesis_nonideal_b2_pkg/package/tables/per_frame_all.csv.gz"
BASELINE_PER_FRAME_SHA256 = "cb96687cf3614c5393bc9658ec3469980313413f495f32930fd2e810ffc070b3"
EXPECTED_RUNS = [
    f"NI-{fam}-on-{reg}-s{seed}"
    for fam, reg in (
        ("glu", "N1"), ("glu", "N2"), ("glu", "G1"), ("glu", "G1eq"), ("glu", "N1G1"),
        ("unet", "N1"), ("unet", "N2"), ("unet", "G1"), ("unet", "G1eq"), ("unet", "N1G1"),
    )
    for seed in (0, 1, 2)
]
REGIME_FROM_TOKEN = {"N1": "N1", "N2": "N2", "G1": "G1", "G1eq": "G1-eq", "N1G1": "N1G1"}
FAMILY_FROM_TOKEN = {"glu": "adapted_glu", "unet": "sparse_unet"}


class GraphicsBlocked(Exception):
    """Inputs are missing or inconsistent. No partial package is published."""


@dataclass(frozen=True)
class Spec:
    expect_runs: tuple[str, ...] = tuple(EXPECTED_RUNS)
    shape: tuple[int, int] = (128, 128)
    frame_index: int = 1000
    replica: int = 0
    n_val_conditions: int = 3
    nominal_rows_best: int = 6174
    baseline_sha256: str | None = BASELINE_PER_FRAME_SHA256
    err_clip_K: float = ERR_CLIP_K


PRODUCTION = Spec()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_identity(run_id: str) -> dict[str, Any]:
    # NI-glu-on-G1eq-s0
    parts = run_id.split("-")
    if len(parts) != 5 or parts[0] != "NI" or parts[2] != "on":
        raise GraphicsBlocked(f"run id {run_id} is outside the 30-specialist scheme")
    try:
        return {
            "run_id": run_id,
            "family": FAMILY_FROM_TOKEN[parts[1]],
            "regime": REGIME_FROM_TOKEN[parts[3]],
            "seed": int(parts[4][1:]),
        }
    except (KeyError, ValueError) as exc:
        raise GraphicsBlocked(f"run id {run_id} is outside the 30-specialist scheme") from exc


def _need(cond: bool, msg: str, problems: list[str]) -> None:
    if not cond:
        problems.append(msg)


def validate_freeze(freeze: dict, spec: Spec) -> dict[str, dict]:
    problems: list[str] = []
    runs = freeze.get("runs") or {}
    _need(set(runs) == set(spec.expect_runs), f"freeze runs {sorted(runs)} != expected {len(spec.expect_runs)}", problems)
    _need(freeze.get("n_valid") == len(spec.expect_runs), "freeze n_valid is not the full matrix", problems)
    _need(not freeze.get("problems"), "freeze lists training problems: " + "; ".join(freeze.get("problems") or []), problems)
    out = {}
    for rid in spec.expect_runs:
        fr = runs.get(rid) or {}
        ident = expected_identity(rid)
        _need(fr.get("valid") is True, f"{rid}: not valid", problems)
        _need(fr.get("family") == ident["family"], f"{rid}: family {fr.get('family')}", problems)
        _need(fr.get("regime") == ident["regime"], f"{rid}: regime {fr.get('regime')}", problems)
        _need(int(fr.get("seed", -1)) == ident["seed"], f"{rid}: seed {fr.get('seed')}", problems)
        best = fr.get("best") or {}
        _need(bool(best.get("weights_sha256")), f"{rid}: best weights hash missing", problems)
        out[rid] = fr
    if problems:
        raise GraphicsBlocked("\n".join(problems))
    return out


def _entry_map(manifest: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for e in manifest.get("entries") or []:
        out[(e["cell"], e["condition"])] = e
    return out


def load_specialist_fields(npz_path: Path, manifest: dict, freeze_run: dict, spec: Spec) -> dict[str, Any]:
    """Load one best-checkpoint card and check it against the manifest, the freeze and the arrays."""
    problems: list[str] = []
    rid = manifest.get("identity", {}).get("run_id")
    _need(manifest.get("format") == "nonideal_specialist_fields_v1", f"{rid}: field format", problems)
    _need(manifest.get("frame_index") == spec.frame_index, f"{rid}: frame {manifest.get('frame_index')} != {spec.frame_index}", problems)
    _need(manifest.get("replica") == spec.replica, f"{rid}: replica", problems)
    _need(manifest.get("role") == "val_eval", f"{rid}: role", problems)
    ident = manifest.get("identity") or {}
    _need(ident.get("checkpoint_weights_sha256") == (freeze_run.get("best") or {}).get("weights_sha256"), f"{rid}: weights hash", problems)
    _need(ident.get("train_regime") == freeze_run.get("regime"), f"{rid}: regime in fields", problems)
    _need(ident.get("family") == freeze_run.get("family"), f"{rid}: family in fields", problems)
    _need(int(ident.get("model_seed", -1)) == int(freeze_run.get("seed", -2)), f"{rid}: seed in fields", problems)
    if not npz_path.is_file():
        problems.append(f"{rid}: npz missing {npz_path}")
        raise GraphicsBlocked("\n".join(problems))
    digest = sha256_file(npz_path)
    _need(digest == manifest.get("file_sha256"), f"{rid}: npz hash differs from fields_manifest", problems)
    z = np.load(npz_path, allow_pickle=False)
    entries = _entry_map(manifest)
    conds = sorted({c for _, c in entries})
    _need(len(entries) == 6 * spec.n_val_conditions, f"{rid}: {len(entries)} field entries", problems)
    _need(len(conds) == spec.n_val_conditions, f"{rid}: {len(conds)} VAL conditions", problems)
    _need(set(entries) == {(cell, c) for cell in ("R0", "N1", "G1", "N1G1", "G1-eq", "N2") for c in conds}, f"{rid}: cells x conditions", problems)
    cards = {}
    for (cell, cond), ent in entries.items():
        _need(int(ent.get("frame_index", -1)) == spec.frame_index, f"{rid} {cell} {cond}: frame", problems)
        _need(int(ent.get("replica", -1)) == spec.replica, f"{rid} {cell} {cond}: replica", problems)
        key = f"{cell}|{cond}"
        arrays = {}
        for name in FIELD_NAMES:
            k = f"{key}|{name}"
            if k not in z.files:
                problems.append(f"{rid}: missing array {k}")
                continue
            arrays[name] = z[k]
        if len(arrays) != len(FIELD_NAMES):
            continue
        pred, true, err = arrays["pred_T_K"], arrays["true_T_K"], arrays["error_signed_K"]
        _need(pred.shape == spec.shape and true.shape == spec.shape and err.shape == spec.shape, f"{rid} {key}: shape {pred.shape}", problems)
        if pred.shape == true.shape == err.shape:
            _need(float(np.max(np.abs(err - (pred - true)))) < 1e-3, f"{rid} {key}: signed error is not pred-true", problems)
            rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - true.astype(np.float64)) ** 2)))
            _need(abs(rmse - float(ent["rmse_K_from_fields"])) <= 5e-3, f"{rid} {key}: manifest RMSE", problems)
            _need(abs(float(ent["rmse_K_from_fields"]) - float(ent["rmse_K_per_frame_csv"])) <= 5e-3, f"{rid} {key}: field vs per-frame RMSE", problems)
        region = arrays["g1_region_mask"]
        if cell in ("G1", "N1G1"):
            _need(region.shape == spec.shape, f"{rid} {key}: G1 region shape", problems)
        else:
            _need(region.size == 0, f"{rid} {key}: region mask should be empty outside G1 and N1G1", problems)
        cards[(cell, cond)] = {"arrays": arrays, "entry": ent, "rmse_fields": float(ent["rmse_K_from_fields"])}
    if "x_m" not in z.files or "y_m" not in z.files:
        problems.append(f"{rid}: x_m/y_m missing")
    if problems:
        raise GraphicsBlocked("\n".join(problems))
    return {"run_id": rid, "conditions": conds, "cards": cards, "sha256": digest, "x_m": z["x_m"], "y_m": z["y_m"],
            "family": freeze_run["family"], "regime": freeze_run["regime"], "seed": int(freeze_run["seed"])}


def load_run_fields(rec: dict, freeze_run: dict, spec: Spec) -> dict[str, Any]:
    manifest = json.loads(Path(rec["fields_manifest"]).read_text())
    return load_specialist_fields(Path(rec["fields_npz"]), manifest, freeze_run, spec)


def validate_index(index: dict, freeze_runs: dict, spec: Spec) -> list[dict]:
    problems: list[str] = []
    _need(index.get("format") == "nonideal_specialist_fields_index_v1", "fields_index format", problems)
    recs = index.get("runs") or []
    _need(index.get("n_runs") == len(spec.expect_runs) == len(recs), f"fields_index n={index.get('n_runs')} records={len(recs)}", problems)
    got = {r.get("run_id") for r in recs}
    _need(got == set(spec.expect_runs), "fields_index run ids", problems)
    if problems:
        raise GraphicsBlocked("\n".join(problems))
    loaded = []
    for rec in recs:
        try:
            card = load_run_fields(rec, freeze_runs[rec["run_id"]], spec)
        except GraphicsBlocked as exc:
            problems.append(str(exc))
            continue
        if rec.get("fields_sha256") != card["sha256"]:
            problems.append(f"{rec.get('run_id')}: fields_index hash != npz")
        if int(rec.get("n_entries", -1)) != 6 * spec.n_val_conditions:
            problems.append(f"{rec.get('run_id')}: n_entries {rec.get('n_entries')}")
        loaded.append(card)
    if problems or not loaded:
        raise GraphicsBlocked("\n".join(problems) or "no field cards")
    # Same truth across every specialist, for the same condition and cell.
    ref = loaded[0]
    for other in loaded[1:]:
        for key, card in other["cards"].items():
            if key not in ref["cards"]:
                continue
            a = ref["cards"][key]["arrays"]["true_T_K"]
            b = card["arrays"]["true_T_K"]
            if a.shape == b.shape and float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) > 1e-3:
                problems.append(f"true T differs for {key} between {ref['run_id']} and {other['run_id']}")
    if problems:
        raise GraphicsBlocked("\n".join(problems))
    return loaded


def per_scenario(frames: pd.DataFrame) -> pd.DataFrame:
    """frame -> condition mean -> uniform mean over conditions -> mean over mask replicas.

    Replicas stay inside the mean. They are not a sample of trainings.
    """
    need = ["run_id", "family", "model_seed", "cell", "role", "replica", "condition", *METRIC_COLS]
    missing = [c for c in need if c not in frames.columns]
    if missing:
        raise GraphicsBlocked("per-frame columns missing: " + ", ".join(missing))
    if "checkpoint_kind" in frames.columns:
        frames = frames[frames["checkpoint_kind"] == "best"]
    keys = ["run_id", "family", "model_seed", "cell", "role", "replica"]
    if "train_regime" in frames.columns:
        keys = ["run_id", "family", "model_seed", "train_regime", "cell", "role", "replica"]
    per_cond = frames.groupby(keys + ["condition"], dropna=False)[list(METRIC_COLS)].mean().reset_index()
    macro = per_cond.groupby(keys, dropna=False)[list(METRIC_COLS)].mean().reset_index()
    group = [c for c in keys if c != "replica"]
    return macro.groupby(group, dropna=False)[list(METRIC_COLS)].mean().reset_index()


def crosscheck_package(package_table: pd.DataFrame, recomputed: pd.DataFrame) -> None:
    best = package_table[package_table["checkpoint_kind"] == "best"] if "checkpoint_kind" in package_table.columns else package_table
    keys = ["run_id", "cell", "role"]
    a = best[keys + ["rmse_K"]].rename(columns={"rmse_K": "pkg"})
    b = recomputed[keys + ["rmse_K"]].rename(columns={"rmse_K": "again"})
    m = a.merge(b, on=keys, how="outer", indicator=True)
    if (m["_merge"] != "both").any() or (m["pkg"] - m["again"]).abs().max() > 1e-4:
        raise GraphicsBlocked("package RMSE does not match a fresh mean of the per-frame CSVs (max allowed 1e-4 K)")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white", "savefig.bbox": "tight",
    })


def _heatmap(values: np.ndarray, row_labels: list[str], path: Path, title: str, cbar: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    if "assoluta" in title.lower() or "RMSE assoluta" in title:
        im = ax.imshow(values, cmap="viridis", aspect="auto")
    else:
        im = ax.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Scenario di valutazione (etichette, non una scala)")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8, color="k")
    fig.colorbar(im, ax=ax, fraction=0.046, label=cbar)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_matrices(pm: pd.DataFrame, base: pd.DataFrame, dest: Path) -> list[Path]:
    """Rows: clean baseline and five train regimes. Columns: six scenarios. No connecting line."""
    out = []
    figdir = dest / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for fam in ("sparse_unet", "adapted_glu"):
        for role in ROLES:
            rows_abs, rows_d, labels = [], [], []
            b = base[(base.family == fam) & (base.role == role)]
            labels.append("baseline pulita")
            rows_abs.append([float(b[b.cell == c]["rmse_K"].mean()) if len(b[b.cell == c]) else np.nan for c in CELLS])
            rows_d.append([0.0 for _ in CELLS])
            sub = pm[(pm.family == fam) & (pm.role == role)]
            for reg in REGIMES:
                labels.append(f"TRAIN {reg}")
                s = sub[sub.train_regime == reg]
                rows_abs.append([float(s[s.cell == c]["rmse_K"].mean()) if len(s[s.cell == c]) else np.nan for c in CELLS])
                rows_d.append([float(s[s.cell == c]["delta_vs_baseline_K"].mean()) if len(s[s.cell == c]) else np.nan for c in CELLS])
            tag = f"{FAM_LABEL[fam].lower()}_{ROLE_LABEL[role]}"
            p1 = figdir / f"matrix_absolute_{tag}.png"
            p2 = figdir / f"matrix_delta_{tag}.png"
            _heatmap(np.array(rows_abs), labels, p1, f"{FAM_LABEL[fam]} · {ROLE_LABEL[role]} · RMSE assoluta (media sui seed)", "RMSE (K)")
            _heatmap(np.array(rows_d), labels, p2, f"{FAM_LABEL[fam]} · {ROLE_LABEL[role]} · RMSE specialista − baseline, stesso scenario", "Δ RMSE (K); negativo = specialista più basso")
            out += [p1, p2]
    return out


def write_seed_spans(pm: pd.DataFrame, dest: Path) -> list[Path]:
    """Mean, min–max across the three training seeds, and the seeds themselves. Not a confidence interval."""
    out = []
    figdir = dest / "figures"
    colors = {"N1": "#0072B2", "N2": "#56B4E9", "G1-eq": "#009E73", "G1": "#D55E00", "N1G1": "#CC79A7"}
    for fam in ("sparse_unet", "adapted_glu"):
        for role in ROLES:
            fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
            sub = pm[(pm.family == fam) & (pm.role == role)]
            panels = (
                (axes[0], "gain_vs_baseline_K", "Guadagno sullo scenario\n(baseline − specialista, K)"),
                (axes[1], "clean_cost_K", "Costo sul pulito\n(specialista − baseline su R0, K)"),
                (axes[2], "specialist_deterioration_K", "Rispetto al proprio R0\n(RMSE scenario − RMSE R0, K)"),
            )
            for ax, col, title in panels:
                for i, reg in enumerate(REGIMES):
                    s = sub[sub.train_regime == reg]
                    if col == "clean_cost_K":
                        s = s[s.cell == "R0"]
                        xs = np.array([i])
                        # one x position per regime
                        ys = s[col].to_numpy()
                        if len(ys) == 0:
                            continue
                        ax.vlines(i, ys.min(), ys.max(), color=colors[reg], lw=4, alpha=0.35)
                        ax.scatter([i], [ys.mean()], c=colors[reg], s=36, zorder=3)
                        ax.scatter(np.full(len(ys), i), ys, c=colors[reg], s=16, zorder=4)
                    else:
                        for j, cell in enumerate(CELLS):
                            ss = s[s.cell == cell][col].to_numpy()
                            if len(ss) == 0:
                                continue
                            x = j + (i - 2) * 0.12
                            ax.vlines(x, ss.min(), ss.max(), color=colors[reg], lw=3, alpha=0.35)
                            ax.scatter([x], [ss.mean()], c=colors[reg], s=22, zorder=3)
                            ax.scatter(np.full(len(ss), x), ss, c=colors[reg], s=10, zorder=4)
                ax.axhline(0, color="0.5", lw=0.7)
                ax.set_title(title, fontsize=10)
                if col == "clean_cost_K":
                    ax.set_xticks(range(len(REGIMES)))
                    ax.set_xticklabels(REGIMES, rotation=30, ha="right")
                else:
                    ax.set_xticks(range(len(CELLS)))
                    ax.set_xticklabels(CELLS, rotation=30, ha="right")
            handles = [plt.Line2D([], [], color=colors[r], marker="o", ls="", label=f"TRAIN {r}") for r in REGIMES]
            fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
            fig.suptitle(
                f"{FAM_LABEL[fam]} · {ROLE_LABEL[role]} · media, minimo–massimo e singoli seed di training",
                fontsize=12,
            )
            fig.text(0.5, -0.16, "Il segmento è il minimo e il massimo dei tre seed. Non è un intervallo di confidenza. Le repliche di maschera sono già mediate.", ha="center", fontsize=8)
            path = figdir / f"seeds_{fam}_{ROLE_LABEL[role]}.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            out.append(path)
    return out


def write_level_and_regions(spec_m: pd.DataFrame, base_m: pd.DataFrame, dest: Path) -> list[Path]:
    out = []
    figdir = dest / "figures"
    metrics = (("bias_K", "Bias (K)"), ("centered_rmse_K", "Centered-RMSE (K)"),
               ("jet_rmse_K", "Getti, T vera < 1800 K"), ("high_rmse_K", "Zona calda, T ≥ 2100 K"), ("peak_rmse_K", "Picco, T ≥ 2200 K"))
    for fam in ("sparse_unet", "adapted_glu"):
        for role in ROLES:
            fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.8), sharex=True)
            sub = spec_m[(spec_m.family == fam) & (spec_m.role == role)]
            b = base_m[(base_m.family == fam) & (base_m.role == role)]
            for ax, (col, title) in zip(axes, metrics):
                # baseline as black ticks at each scenario (mean over its seeds)
                for j, cell in enumerate(CELLS):
                    bb = b[b.cell == cell][col].to_numpy()
                    if len(bb):
                        ax.scatter([j], [bb.mean()], marker="x", c="k", s=28, zorder=2)
                    for i, reg in enumerate(REGIMES):
                        ss = sub[(sub.train_regime == reg) & (sub.cell == cell)][col].to_numpy()
                        if len(ss) == 0:
                            continue
                        x = j + (i - 2) * 0.12
                        ax.vlines(x, ss.min(), ss.max(), color="0.6", lw=2)
                        ax.scatter(np.full(len(ss), x), ss, s=9, zorder=3)
                ax.set_title(title, fontsize=9)
                ax.set_xticks(range(len(CELLS)))
                ax.set_xticklabels(CELLS, rotation=40, ha="right", fontsize=8)
            fig.suptitle(f"{FAM_LABEL[fam]} · {ROLE_LABEL[role]} · livello, struttura, regioni. Crocetta nera = baseline pulita, media dei suoi seed.", fontsize=11)
            path = figdir / f"level_regions_{fam}_{ROLE_LABEL[role]}.png"
            fig.savefig(path, dpi=130)
            plt.close(fig)
            out.append(path)
    return out


def _sites(ax, shape, surviving, removed, region) -> None:
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    if region.size and region.shape == shape:
        ax.contour(region.astype(float), levels=[0.5], colors="k", linewidths=0.6)
    if surviving.size:
        ax.scatter(surviving % shape[1], surviving // shape[1], s=2, c="#0072B2", marker=".")
    if removed.size:
        ax.scatter(removed % shape[1], removed // shape[1], s=2, c="#D55E00", marker=".")
    ax.set_xticks([])
    ax.set_yticks([])


def write_cards(loaded: list[dict], dest: Path, spec: Spec) -> list[Path]:
    """One multipage PDF per run (one page per VAL condition) and one PNG per page. All  runs, no selection."""
    out = []
    card_dir = dest / "cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    for run in sorted(loaded, key=lambda r: r["run_id"]):
        pdf_path = card_dir / f"{run['run_id']}.pdf"
        conds = run["conditions"]
        with PdfPages(pdf_path) as pdf:
            for cond in conds:
                true = run["cards"][("R0", cond)]["arrays"]["true_T_K"]
                vmin, vmax = float(np.min(true)), float(np.max(true))
                fig, axes = plt.subplots(4, 6, figsize=(13.5, 8.2))
                real_max = 0.0
                clipped = False
                pred_lo, pred_hi = float("inf"), float("-inf")
                for j, cell in enumerate(CELLS):
                    a = run["cards"][(cell, cond)]["arrays"]
                    pred, err = a["pred_T_K"], a["error_signed_K"]
                    rmax = float(np.max(np.abs(err)))
                    real_max = max(real_max, rmax)
                    if rmax > spec.err_clip_K:
                        clipped = True
                    pred_lo = min(pred_lo, float(np.min(pred)))
                    pred_hi = max(pred_hi, float(np.max(pred)))
                    axes[0, j].imshow(true, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
                    axes[1, j].imshow(pred, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
                    axes[2, j].imshow(err, cmap="coolwarm", vmin=-spec.err_clip_K, vmax=spec.err_clip_K, origin="upper")
                    _sites(axes[3, j], spec.shape, a["surviving_sites"], a["removed_sites"], a["g1_region_mask"])
                    axes[0, j].set_title(cell, fontsize=9)
                    for r in range(3):
                        axes[r, j].set_xticks([])
                        axes[r, j].set_yticks([])
                axes[0, 0].set_ylabel("T vera")
                axes[1, 0].set_ylabel("T predetta")
                axes[2, 0].set_ylabel("errore firmato")
                axes[3, 0].set_ylabel("siti")
                clip_note = (
                    f"errore firmato clip ±{spec.err_clip_K:.0f} K; massimo |errore| reale {real_max:.0f} K"
                    + ("; valori oltre la scala saturati" if clipped else "; nessun valore oltre la scala")
                )
                pred_note = ""
                if pred_lo < vmin - 1e-3 or pred_hi > vmax + 1e-3:
                    pred_note = f" Predizione fuori scala della verità [{pred_lo:.0f}, {pred_hi:.0f}] K, saturata nel disegno."
                fig.suptitle(
                    f"{run['run_id']}  ·  {FAM_LABEL[run['family']]}  ·  TRAIN {run['regime']}  ·  seed {run['seed']}\n"
                    f"{cond}  ·  frame {spec.frame_index}  ·  replica {spec.replica}  ·  T vera e predetta sulla stessa scala [{vmin:.0f}, {vmax:.0f}] K\n"
                    f"{clip_note}.{pred_note} Blu = siti osservati, arancio = siti mancanti, contorno nero = regione G1.",
                    fontsize=9,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                png = card_dir / run["run_id"] / f"{cond.replace('.h5', '')}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png, dpi=120)
                pdf.savefig(fig)
                plt.close(fig)
                out.append(png)
        out.append(pdf_path)
    return out


def write_seed_variability(loaded: list[dict], dest: Path, spec: Spec) -> list[Path]:
    """Mean prediction, std across the three trainings, and the error of that mean field.

    The std is dispersion across trainings, not a calibrated physical uncertainty.
    The error map is (mean of the three fields − truth). That map's RMSE is not the mean of the three RMSEs.
    """
    out = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for run in loaded:
        groups.setdefault((run["family"], run["regime"]), []).append(run)
    var_dir = dest / "variability"
    for (fam, reg), runs in sorted(groups.items()):
        if len(runs) != 3:
            raise GraphicsBlocked(f"{fam} TRAIN {reg}: {len(runs)} seeds, expected 3")
        conds = runs[0]["conditions"]
        pdf_path = var_dir / f"{fam}_{reg}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for cond in conds:
                fig, axes = plt.subplots(3, 6, figsize=(13.5, 7.2))
                for j, cell in enumerate(CELLS):
                    stack = np.stack([r["cards"][(cell, cond)]["arrays"]["pred_T_K"].astype(np.float64) for r in runs], axis=0)
                    true = runs[0]["cards"][(cell, cond)]["arrays"]["true_T_K"].astype(np.float64)
                    mean = stack.mean(axis=0)
                    std = stack.std(axis=0)  # population std of the three trainings
                    err_mean = mean - true
                    mean_of_rmse = float(np.mean([np.sqrt(np.mean((stack[i] - true) ** 2)) for i in range(3)]))
                    rmse_of_mean = float(np.sqrt(np.mean(err_mean ** 2)))
                    vmin, vmax = float(true.min()), float(true.max())
                    std_max = float(std.max())
                    axes[0, j].imshow(mean, cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
                    axes[1, j].imshow(std, cmap="magma", vmin=0, vmax=max(std_max, 1.0), origin="upper")
                    axes[2, j].imshow(err_mean, cmap="coolwarm", vmin=-spec.err_clip_K, vmax=spec.err_clip_K, origin="upper")
                    axes[0, j].set_title(
                        f"{cell}\nRMSE del campo medio {rmse_of_mean:.1f} K\nmedia delle RMSE {mean_of_rmse:.1f} K",
                        fontsize=7,
                    )
                    for r in range(3):
                        axes[r, j].set_xticks([])
                        axes[r, j].set_yticks([])
                axes[0, 0].set_ylabel("media dei 3 campi")
                axes[1, 0].set_ylabel("dev. std fra seed")
                axes[2, 0].set_ylabel("errore della media")
                fig.suptitle(
                    f"{FAM_LABEL[fam]} · TRAIN {reg} · {cond} · frame {spec.frame_index}\n"
                    "La deviazione standard è la dispersione fra i tre training, non un'incertezza fisica calibrata.\n"
                    "L'errore mostrato è (media dei campi − verità). La sua RMSE non è la media delle RMSE dei singoli modelli.",
                    fontsize=9,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.88))
                png = var_dir / f"{fam}_{reg}" / f"{cond.replace('.h5', '')}.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png, dpi=110)
                pdf.savefig(fig)
                plt.close(fig)
                out.append(png)
        out.append(pdf_path)
    return out


def baseline_fields_needed(eval_root: Path | None) -> str:
    """Precise list of arrays still required for the same frame-1000 collection of the 12 baselines.

    The Block II npz files, when present, are a different product (minimum frame index, no original_sites).
    Predicted temperature is not reconstructed from an absolute-error image.
    """
    lines = [
        "# Campi mancanti per la stessa raccolta dei 12 modelli originali",
        "",
        "La raccolta dei 30 specialisti chiede, per il checkpoint best, replica 0, frame 1000, le tre condizioni VAL, i sei scenari:",
        "`pred_T_K`, `true_T_K`, `error_signed_K` (predizione − verità), `original_sites`, `surviving_sites`, `removed_sites`, `g1_region_mask`,",
        "nel file `fields_val_frame1000_best.npz` con `fields_manifest.json` (formato `nonideal_specialist_fields_v1`).",
        "",
        "Quei file non esistono per i 12 modelli del Blocco II. Non vanno ricostruiti dal modulo dell'errore delle figure già pubblicate.",
        "Le figure pubblicate `fig5_fields_*.png` coprono solo quattro configurazioni a seed 0 (U-Net/GLU × con e senza φ/NH3) e mostrano il modulo dell'errore, non la temperatura predetta.",
        "",
        "Sul cluster, per tutti i 12 run, esiste invece `fields_figure_frames.npz` (formato Block II).",
        "Ispezione di `O-C149-on-s0`: 150 chiavi; VAL alle condizioni `0NH3_1.05`, `20NH3_1.1`, `40NH3_0.9` con frame_index 0 (non 1000);",
        "seen a `0NH3_0.85` e `80NH3_1.2` con frame 1600; array `pred_T_K`, `true_T_K`, `kept_sites`, `removed_sites`, `region_mask`.",
        "Mancano, rispetto al contratto dei 30: frame 1000, `error_signed_K` salvato, `original_sites`, il nome `surviving_sites`, il manifest e il controllo di RMSE.",
        "",
        "Elenco dei file presenti (formato vecchio) e di ciò che servirebbe al loro posto:",
        "",
    ]
    for rid in BASELINE_ALL12:
        old = None if eval_root is None else eval_root / rid / "fields_figure_frames.npz"
        state = "presente sul cluster" if (old is not None and old.is_file()) else "non verificato in questo ambiente"
        frames = ""
        if old is not None and old.is_file():
            found = legacy_field_frames(old)
            frames = f"; frame presenti nel file vecchio: {found}. Nessuno di questi è il frame 1000 del contratto, e il file non viene disegnato come scheda dei 30."
        lines.append(f"- `{rid}`")
        lines.append(f"  - vecchio, se c'è: `{BASELINE_EVAL_ROOT_DEFAULT}/{rid}/fields_figure_frames.npz` ({state}{frames})")
        lines.append(f"  - richiesto, assente: `{BASELINE_EVAL_ROOT_DEFAULT}/{rid}/fields_val_frame1000_best.npz` + `fields_manifest.json`")
        lines.append("  - per ciascuno dei 6 scenari × 3 condizioni VAL, frame 1000, replica 0: i sette array del formato v1")
    lines += [
        "",
        "Il recupero di questi campi è un'inferenza in più. Non è stato avviato: va concordato con chi gestisce la valutazione, senza una campagna autonoma.",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_note(pm: pd.DataFrame, dest: Path) -> Path:
    lines = [
        "# Lettura dei 30 specialisti",
        "",
        "Numeri letti dai CSV per frame e dai campi congelati di questo job, dopo il controllo di completezza.",
        "Sono stress test sintetici. VAL123 è già servita alla selezione dei modelli originali. seen220 non è un dominio nuovo.",
        "Tre seed di training; le repliche di maschera sono mediate prima e non sono training indipendenti.",
        "Nessun test di significatività. Il minimo–massimo fra seed non è un intervallo di confidenza.",
        "",
        "Guadagno = RMSE della baseline pulita − RMSE dello specialista, stesso scenario. Positivo: lo specialista ha errore più basso.",
        "Il costo sul pulito è la differenza su R0. Il recupero nello scenario di training è il guadagno sulla diagonale.",
        "Il trasferimento è il guadagno sugli altri scenari. Una diagonale favorevole non dice nulla, da sola, sugli altri scenari.",
        "",
        "## Guadagno nello scenario di training (media sui seed, minimo e massimo fra seed), checkpoint best",
        "",
    ]
    for role in ROLES:
        lines.append(f"### {ROLE_LABEL[role]}")
        lines.append("")
        lines.append("| famiglia | " + " | ".join(REGIMES) + " |")
        lines.append("|---|" + "---|" * len(REGIMES))
        for fam in ("sparse_unet", "adapted_glu"):
            cells = []
            for reg in REGIMES:
                s = pm[(pm.family == fam) & (pm.role == role) & (pm.train_regime == reg) & (pm.cell == reg)]
                if s.empty:
                    cells.append("—")
                else:
                    v = s["gain_vs_baseline_K"] if "gain_vs_baseline_K" in s.columns else (s["base_rmse_K"] - s["rmse_K"])
                    cells.append(f"{float(v.mean()):+.2f} [{float(v.min()):+.2f}, {float(v.max()):+.2f}]")
            lines.append(f"| {FAM_LABEL[fam]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "Il bias è il livello (errore medio con segno). La centered-RMSE è la struttura dopo aver tolto quel livello.",
        "Sulle medie di frame e condizioni, bias e centered-RMSE non si sommano alla RMSE.",
        "La deviazione standard fra seed, nelle schede di variabilità, è dispersione fra training, non incertezza fisica.",
        "La RMSE del campo medio non è la media delle RMSE dei singoli modelli: i due numeri sono scritti separati su quelle pagine.",
        "",
    ]
    path = dest / "NOTE.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_index(dest: Path, files: Iterable[Path]) -> None:
    lines = [
        "# Pacchetto grafico dei 30 specialisti",
        "",
        "Prodotto solo se freeze, campi, CSV e tabelle del pacchetto numerico sono completi e coerenti.",
        "Schede: una PDF per run best, una pagina per condizione VAL, frame 1000, replica 0. Nessuna selezione dei run migliori.",
        "Gli scenari sono categorie: le matrici non sono curve.",
        "",
        "## File",
        "",
    ]
    for p in sorted(files, key=lambda x: str(x)):
        if p.is_file():
            lines.append(f"- `{p.relative_to(dest).as_posix()}`")
    lines.append("")
    (dest / "INDEX.md").write_text("\n".join(lines) + "\n")


def write_manifest(dest: Path, files: Iterable[Path], sources: dict) -> None:
    items = []
    for p in sorted({q for q in files if q.is_file()}, key=lambda x: str(x)):
        items.append({"path": p.relative_to(dest).as_posix(), "sha256": sha256_file(p)})
    payload = {"format": "nonideal_specialist_graphics_manifest_v1", "sources": sources, "files": items}
    (dest / "figure_manifest.json").write_text(json.dumps(payload, indent=1) + "\n")


def legacy_field_frames(path: Path) -> list[int]:
    """Frame indices inside a Block II fields_figure_frames.npz.

    The key is ``cell|role|condition|frame|field``. Those frames stay labelled as stored.
    They are not promoted to the frame-1000 contract, and predicted T is not rebuilt from |error|.
    """
    z = np.load(path, allow_pickle=False)
    frames: set[int] = set()
    for key in z.files:
        parts = str(key).split("|")
        if len(parts) >= 4 and parts[3].lstrip("-").isdigit():
            frames.add(int(parts[3]))
    return sorted(frames)


def check_domains(df: pd.DataFrame, spec: Spec) -> None:
    problems: list[str] = []
    if set(df["cell"].astype(str)) != set(CELLS):
        problems.append("cells present: " + ", ".join(sorted(set(df["cell"].astype(str)))))
    if set(df["role"].astype(str)) != set(ROLES):
        problems.append("roles present: " + ", ".join(sorted(set(df["role"].astype(str)))))
    if "variant" in df.columns and set(df["variant"].astype(str)) != {"p_on"}:
        problems.append("rows that are not p_on")
    if spec.nominal_rows_best == PRODUCTION.nominal_rows_best:
        if set(int(x) for x in df["replica"].unique()) != {0, 1, 2}:
            problems.append("replicas are not 0, 1 and 2")
        for rid, g in df.groupby("run_id"):
            seen_n = int(g.loc[g["role"] == "seen_train_eval", "condition"].nunique())
            val_n = int(g.loc[g["role"] == "val_eval", "condition"].nunique())
            if seen_n != 20 or val_n != 3:
                problems.append(f"{rid}: {seen_n} seen conditions, {val_n} VAL conditions")
            cover = g[(g["role"] == "val_eval") & (g["replica"] == spec.replica) & (g["frame_index"] == spec.frame_index)]
            if cover["condition"].nunique() != 3 or cover.groupby(["cell", "condition"]).ngroups != 18:
                problems.append(f"{rid}: frame {spec.frame_index} replica {spec.replica} does not cover 6 cells x 3 VAL conditions")
    if problems:
        raise GraphicsBlocked("\n".join(problems))


def check_observations(spec_df: pd.DataFrame, base_df: pd.DataFrame, spec: Spec) -> None:
    """Surviving-site hashes must agree with the original model of the same family and seed.

    Mask replicas are the same observations, not extra trainings. A hash mismatch blocks the package.
    """
    col = "surviving_sites_hash"
    full = spec.nominal_rows_best == PRODUCTION.nominal_rows_best
    if col not in spec_df.columns or col not in base_df.columns:
        if full:
            raise GraphicsBlocked("surviving_sites_hash missing: observation compatibility was not checked")
        return
    keys = ["cell", "role", "replica", "condition", "frame_index"]
    problems: list[str] = []
    for (fam, seed), g in spec_df.groupby(["family", "model_seed"]):
        brid = BASELINE_OF.get((str(fam), int(seed)))
        if brid is None:
            problems.append(f"no baseline run for {fam} seed {int(seed)}")
            continue
        b = base_df[base_df["run_id"] == brid]
        rid = str(g["run_id"].iloc[0])
        a = g[g["run_id"] == rid]
        m = a[keys + [col]].merge(b[keys + [col]], on=keys, how="outer", suffixes=("_s", "_b"), indicator=True)
        if m.empty or (m["_merge"] != "both").any() or (m[f"{col}_s"].astype(str) != m[f"{col}_b"].astype(str)).any():
            problems.append(f"{rid} vs {brid}: surviving-site hashes or keys differ")
        for other in sorted(set(g["run_id"].astype(str)) - {rid}):
            o = g[g["run_id"] == other][keys + [col]]
            mm = a[keys + [col]].merge(o, on=keys, suffixes=("_a", "_o"))
            if len(mm) != len(a) or (mm[f"{col}_a"].astype(str) != mm[f"{col}_o"].astype(str)).any():
                problems.append(f"{rid} vs {other}: site hashes differ across train regimes")
    if problems:
        raise GraphicsBlocked("\n".join(problems))


def _block(dest: Path, message: str, *, wipe: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if wipe:
        for child in list(dest.iterdir()):
            if child.name == "GRAPHICS_BLOCK_REPORT.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    report = dest / "GRAPHICS_BLOCK_REPORT.md"
    report.write_text(
        "# GRAPHICS_BLOCK_REPORT\n\nIl pacchetto grafico non è stato pubblicato: input mancanti o incoerenti.\n"
        "Nessuna scheda parziale va letta come risultato.\n\n" + message.strip() + "\n"
    )
    complete = dest / "COMPLETE.json"
    if complete.exists():
        complete.unlink()


def run_graphics(
    *,
    freeze_path: Path,
    eval_root: Path,
    package_dir: Path,
    fields_index_path: Path,
    baseline_per_frame: Path,
    dest: Path,
    spec: Spec = PRODUCTION,
    baseline_eval_root: Path | None = None,
    git_sha: str = "",
) -> None:
    _style()
    created = False
    leave_existing = False
    try:
        if not (package_dir / "COMPLETE.json").is_file():
            raise GraphicsBlocked(f"package COMPLETE.json missing in {package_dir}")
        freeze = json.loads(freeze_path.read_text())
        freeze_runs = validate_freeze(freeze, spec)
        index = json.loads(fields_index_path.read_text())
        loaded = validate_index(index, freeze_runs, spec)
        frames = []
        problems: list[str] = []
        for rid in spec.expect_runs:
            best = eval_root / rid / "best"
            if not (best / "COMPLETE.json").is_file():
                problems.append(f"{rid}: best/COMPLETE.json missing")
                continue
            gate_path = best / "gate.json"
            if not gate_path.is_file() or json.loads(gate_path.read_text()).get("all_pass") is not True:
                problems.append(f"{rid}: R0 gate did not pass")
                continue
            d = best / "per_frame"
            parts = sorted(d.rglob("rep*.csv")) if d.is_dir() else []
            if not parts:
                problems.append(f"{rid}: no per-frame CSV")
                continue
            df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
            if len(df) != spec.nominal_rows_best:
                problems.append(f"{rid}: {len(df)} per-frame rows != {spec.nominal_rows_best}")
                continue
            if str(df["run_id"].iloc[0]) != rid or set(df["run_id"].astype(str)) != {rid}:
                problems.append(f"{rid}: per-frame run_id differs from the directory")
                continue
            df["train_regime"] = freeze_runs[rid]["regime"]
            df["checkpoint_kind"] = "best"
            frames.append(df)
        if problems or len(frames) != len(spec.expect_runs):
            raise GraphicsBlocked("\n".join(problems) or "per-frame tables incomplete")
        allf = pd.concat(frames, ignore_index=True)
        check_domains(allf, spec)
        for run in loaded:
            sub = allf[allf.run_id == run["run_id"]]
            for (cell, cond), card in run["cards"].items():
                hit = sub[(sub.cell == cell) & (sub.condition == cond) & (sub.frame_index == spec.frame_index) & (sub.replica == spec.replica) & (sub.role == "val_eval")]
                if len(hit) != 1:
                    problems.append(f"{run['run_id']} {cell} {cond}: {len(hit)} per-frame rows at frame {spec.frame_index}")
                    continue
                if abs(float(hit["rmse_K"].iloc[0]) - card["rmse_fields"]) > 5e-3:
                    problems.append(f"{run['run_id']} {cell} {cond}: field RMSE != per-frame rmse_K")
        if problems:
            raise GraphicsBlocked("\n".join(problems))
        recomputed = per_scenario(allf)
        pkg = pd.read_csv(package_dir / "tables" / "per_run_scenario.csv")
        pkg = pkg[pkg.checkpoint_kind == "best"]
        crosscheck_package(pkg, recomputed)
        if not baseline_per_frame.is_file():
            raise GraphicsBlocked(f"baseline per-frame CSV missing: {baseline_per_frame}")
        if spec.baseline_sha256 and sha256_file(baseline_per_frame) != spec.baseline_sha256:
            raise GraphicsBlocked("baseline per-frame CSV hash differs from the Block II manifest")
        base_raw = pd.read_csv(baseline_per_frame)
        if "cell" in base_raw.columns:
            base_raw = base_raw[base_raw.cell.isin(CELLS)]
        base_raw = base_raw[base_raw.run_id.isin(BASELINE_PON)]
        check_observations(allf, base_raw, spec)
        base_m = per_scenario(base_raw)
        keys = ["run_id", "cell", "role"]
        keep = ("gain_vs_baseline_K", "delta_vs_baseline_K", "clean_cost_K", "specialist_deterioration_K", "base_rmse_K", "train_regime", "family", "model_seed")
        extra = pkg[keys + [c for c in keep if c in pkg.columns]]
        on = [c for c in keys + ["train_regime", "family", "model_seed"] if c in extra.columns and c in recomputed.columns]
        pm = recomputed.merge(extra, on=on, how="left", suffixes=("", "_pkg"))
        if "delta_vs_baseline_K" not in pm.columns or pm["delta_vs_baseline_K"].isna().any():
            raise GraphicsBlocked("package table has no delta_vs_baseline_K for every run, cell and role")
        if dest.exists() and any(dest.iterdir()):
            leave_existing = True
            raise GraphicsBlocked(f"destination already exists and is not empty: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        created = True
        (dest / "BASELINE_FIELDS_NEEDED.md").write_text(baseline_fields_needed(baseline_eval_root))
        files: list[Path] = []
        files += write_matrices(pm, base_m, dest)
        files += write_seed_spans(pm, dest)
        files += write_level_and_regions(pm, base_m, dest)
        files += write_cards(loaded, dest, spec)
        files += write_seed_variability(loaded, dest, spec)
        note = write_note(pm, dest)
        files.append(note)
        files.append(dest / "BASELINE_FIELDS_NEEDED.md")
        sources = {
            "freeze_sha256": sha256_file(freeze_path),
            "fields_index_sha256": sha256_file(fields_index_path),
            "package_table_sha256": sha256_file(package_dir / "tables" / "per_run_scenario.csv"),
            "baseline_per_frame_sha256": sha256_file(baseline_per_frame),
            "graphics_git_sha": git_sha,
            "n_runs": len(loaded),
            "frame_index": spec.frame_index,
            "err_clip_K": spec.err_clip_K,
            "std_across_seeds": "population standard deviation of the three predicted fields (ddof=0); dispersion across trainings, not a calibrated uncertainty",
        }
        write_index(dest, files)
        write_manifest(dest, list(files) + [dest / "INDEX.md"], sources)
        (dest / "COMPLETE.json").write_text(json.dumps({"ok": True, "n_runs": len(loaded), "frame_index": spec.frame_index}) + "\n")
    except GraphicsBlocked as exc:
        if leave_existing:
            raise
        _block(dest, str(exc), wipe=created)
        raise
    except Exception as exc:
        if created:
            _block(dest, f"{type(exc).__name__}: {exc}", wipe=True)
        raise
