#!/usr/bin/env python3
"""Aggregate the final evaluation of the 12 temperature-assisted runs: checks, tables, figures, reviewer index. CPU only.

Refuses unless every run is COMPLETE with a passed loop gate and all consistency checks pass (frozen matrix hash, keys, mask hashes vs the
verified payload, identities, shared forwards). Descriptive only: no ranking, no winner. ``--smoke`` relaxes completeness for tests.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.sparse_eval_masks import check_eval_mask_contract  # noqa: E402
from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, ROLE_N_FRAMES, ROLES  # noqa: E402
from combustion_recon.evaluation.s7_package import payload_mask_hashes  # noqa: E402

K_VALUES = (0, 1, 4, 9, 16, 32)
COLOR = {"sparse_unet": "#2a6fb0", "adapted_glu": "#d9731a"}
CORE = ["full_rmse_K", "common_rmse_K", "offanchor_rmse_K", "anchors_rmse_K", "common_mae_K", "common_bias_K", "common_centered_rmse_K",
        "full_mae_K", "full_bias_K", "full_centered_rmse_K", "high_rmse_K", "peak_rmse_K", "jet_rmse_K", "non_high_rmse_K"]
IDENT = ["run_id", "family", "p_conditioning", "model_seed", "checkpoint_kind", "epoch", "role", "K"]


def sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_and_check(eval_root: Path, matrix: dict, msha: str, payload, smoke: bool):
    issues, frames, runs = [], [], []
    for e in matrix["runs"]:
        d = eval_root / e["run_id"]
        info = {"run_id": e["run_id"], "family": e["family"], "p_conditioning": e["p_conditioning"], "model_seed": e["model_seed"], "complete": False, "gate_all_pass": False, "n_csv": 0, "gate_max_abs_delta_K": None}
        runs.append(info)
        marker = d / ("SMOKE_OK.json" if smoke else "COMPLETE.json")
        if not marker.exists():
            issues.append(f"{e['run_id']}: {marker.name} missing" + (" (GATE_FAILED present)" if (d / "GATE_FAILED.json").exists() else ""))
            continue
        m = json.loads(marker.read_text())
        info["complete"] = bool(m.get("ok"))
        info["gate_all_pass"] = bool((m.get("gate") or {}).get("all_pass"))
        info["gate_max_abs_delta_K"] = (m.get("gate") or {}).get("max_abs_delta_K")
        if not smoke and not info["gate_all_pass"]:
            issues.append(f"{e['run_id']}: gate not passed")
        prov = json.loads((d / "provenance.json").read_text())
        if prov.get("matrix_sha256") != msha:
            issues.append(f"{e['run_id']}: provenance matrix sha differs from the frozen matrix")
        if prov.get("scientific") is smoke:
            issues.append(f"{e['run_id']}: provenance mode mismatch")
        for name in CHECKPOINTS:
            spec = e["checkpoints"][name]
            for role in ROLES:
                for rep in (0, 1, 2):
                    path = d / "per_frame" / name / role / f"rep{rep}.csv"
                    if not path.exists():
                        issues.append(f"{path.relative_to(eval_root)} missing")
                        continue
                    hdr = pd.read_csv(path, nrows=0).columns
                    cols = [c for c in (*IDENT, "frame_index", "sim_id", "mask_replica_id", "mask_hash", "weights_sha256", "file_sha256", "shared_forward_from", *CORE) if c in hdr]
                    df = pd.read_csv(path, usecols=cols)
                    info["n_csv"] += 1
                    if not smoke and len(df) != ROLE_N_FRAMES[role] * len(K_VALUES):
                        issues.append(f"{path.relative_to(eval_root)}: {len(df)} rows")
                    if set(df["weights_sha256"]) != {spec["weights_sha256"]} or set(df["file_sha256"]) != {spec["file_sha256"]}:
                        issues.append(f"{path.relative_to(eval_root)}: checkpoint hashes differ from the matrix")
                    if set(df["family"]) != {e["family"]} or set(df["run_id"]) != {e["run_id"]} or set(df["K"]) != set(K_VALUES):
                        issues.append(f"{path.relative_to(eval_root)}: identity/K")
                    if set(df["shared_forward_from"].fillna("")) != {spec.get("reuse_forward_from") or ""}:
                        issues.append(f"{path.relative_to(eval_root)}: shared_forward_from")
                    if not np.isfinite(df[["full_rmse_K", "common_rmse_K"]].to_numpy()).all():
                        issues.append(f"{path.relative_to(eval_root)}: non-finite")
                    frames.append(df)
    if not frames:
        return pd.DataFrame(), runs, issues
    t = pd.concat(frames, ignore_index=True)
    t["replica"] = t["mask_replica_id"].astype(int)
    key = ["run_id", "checkpoint_kind", "role", "replica", "K", "sim_id", "frame_index"]
    if int(t.duplicated(subset=key).sum()):
        issues.append("duplicated keys")
    mh = t.groupby(["role", "sim_id", "frame_index", "replica"])["mask_hash"].nunique()
    if int((mh != 1).sum()):
        issues.append("several mask hashes for one (role, frame, replica)")
    bad = 0
    for role, sim, fr, rep, h in t[["role", "sim_id", "frame_index", "replica", "mask_hash"]].drop_duplicates().itertuples(index=False):
        ref = payload.get((role, str(sim), int(fr), int(rep)))
        if ref is None or ref["hash_5pct"] != str(h):
            bad += 1
    if bad:
        issues.append(f"{bad} mask hashes differ from the verified payload")
    for e in matrix["runs"]:
        for name in CHECKPOINTS:
            src = e["checkpoints"][name].get("reuse_forward_from")
            if src:
                a = t[(t.run_id == e["run_id"]) & (t.checkpoint_kind == name)].sort_values(key)[CORE].to_numpy()
                b = t[(t.run_id == e["run_id"]) & (t.checkpoint_kind == src)].sort_values(key)[CORE].to_numpy()
                if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                    issues.append(f"{e['run_id']}: {name} differs from its shared source {src}")
    return t, runs, issues


def tables(t: pd.DataFrame):
    m = [c for c in CORE if c in t.columns]
    cr = t.groupby(IDENT + ["replica", "sim_id"], dropna=False)[m].mean().reset_index()
    pc = cr.groupby(IDENT + ["sim_id"], dropna=False)[m].mean().reset_index()
    gr = cr.groupby(IDENT + ["replica"], dropna=False)[m].mean().reset_index()
    g = gr.groupby(IDENT, dropna=False)[m].mean().reset_index()
    rr = gr.groupby(IDENT, dropna=False)["common_rmse_K"].agg(common_rmse_rep_min_K="min", common_rmse_rep_max_K="max").reset_index()
    g = g.merge(rr, on=IDENT)
    return {"global_by_setting": g, "global_by_replica": gr, "per_condition": pc}


def sacct(ids):
    out = subprocess.run(["sacct", "-D", "-X", "-P", "-j", ",".join(ids), "-o", "JobID,JobIDRaw,State,Submit,Start,End,Partition,Account"], capture_output=True, text=True, check=True).stdout
    rows = [l.split("|") for l in out.strip().splitlines()]
    return pd.DataFrame(rows[1:], columns=rows[0])


def costs(matrix, train_jobs, eval_jobs, eval_root):
    f = lambda s: datetime.fromisoformat(s) if s and s not in ("Unknown", "None") else None  # noqa: E731
    raw2run = {str(j): e["run_id"] for e in matrix["runs"] for j in e["training_job_ids"]}
    d = sacct(train_jobs)
    d = d[d["JobIDRaw"].isin(raw2run)].copy()
    d["run_id"] = d["JobIDRaw"].map(raw2run)
    d["compute_s"] = [(f(b) - f(a)).total_seconds() if f(a) and f(b) else np.nan for a, b in zip(d["Start"], d["End"])]
    d["queue_wait_s"] = [(f(b) - f(a)).total_seconds() if f(a) and f(b) else np.nan for a, b in zip(d["Submit"], d["Start"])]
    d = d.sort_values(["run_id", "Start"])
    d["attempt"] = d.groupby("run_id").cumcount() + 1
    by_run = d.groupby("run_id").agg(n_attempts=("attempt", "max"), compute_h=("compute_s", lambda s: s.sum() / 3600), queue_wait_h=("queue_wait_s", lambda s: s.sum() / 3600),
                                     partition=("Partition", "first")).reset_index()
    ev = pd.DataFrame()
    if eval_jobs:
        e = sacct(eval_jobs)
        ids = [r["run_id"] for r in matrix["runs"]]
        rows = []
        for r in e.itertuples():
            if "_" in r.JobID and r.JobID.split("_")[1].isdigit():
                rows.append({"run_id": ids[int(r.JobID.split("_")[1])], "State": r.State, "compute_s": (f(r.End) - f(r.Start)).total_seconds() if f(r.Start) and f(r.End) else np.nan,
                             "queue_wait_s": (f(r.Start) - f(r.Submit)).total_seconds() if f(r.Start) and f(r.Submit) else np.nan})
        ev = pd.DataFrame(rows)
    return d, by_run, ev


def figures(g, pcond, dest, matrix, by_run, ev, eval_root):
    cfgs = [("sparse_unet", True), ("sparse_unet", False), ("adapted_glu", True), ("adapted_glu", False)]
    lab = lambda f, p: f"{'U-Net' if f == 'sparse_unet' else 'GLU'} p_{'on' if p else 'off'}"  # noqa: E731
    for ck in CHECKPOINTS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, role in zip(axes, ROLES):
            s = g[(g.checkpoint_kind == ck) & (g.role == role)]
            for (run, f, p), h in s.groupby(["run_id", "family", "p_conditioning"]):
                h = h.sort_values("K")
                ax.plot(range(len(K_VALUES)), [h[h.K == k]["common_rmse_K"].iloc[0] for k in K_VALUES], "-" if p else "--", color=COLOR[f], marker="o", ms=3, lw=1, alpha=0.8)
            ax.set_xticks(range(len(K_VALUES)))
            ax.set_xticklabels(K_VALUES)
            ax.set_xlabel("number of observed T sites K")
            ax.set_title({"train_fit": "reference760", "seen_train_eval": "seen220", "val_eval": "VAL123"}[role])
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("macro RMSE on the common domain (K)")
        fig.legend(handles=[plt.Line2D([], [], color=COLOR[f], ls="-" if p else "--", label=lab(f, p)) for f, p in cfgs], loc="outside lower center", ncol=4, frameon=False)
        fig.suptitle(f"Assisted trainings, checkpoint {ck}: one line per run (3 seeds per configuration)")
        fig.savefig(dest / f"rmse_vs_K_common_{ck}.png", dpi=140)
        plt.close(fig)
    s = g[(g.checkpoint_kind == "best") & (g.role == "val_eval")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, col, ttl in zip(axes, ("full_rmse_K", "offanchor_rmse_K", "anchors_rmse_K"), ("full field", "off-anchor(K)", "at the K anchors")):
        for (run, f, p), h in s.groupby(["run_id", "family", "p_conditioning"]):
            h = h.sort_values("K")
            ax.plot(range(len(K_VALUES)), [h[h.K == k][col].iloc[0] for k in K_VALUES], "-" if p else "--", color=COLOR[f], marker="o", ms=3, lw=1, alpha=0.8)
        ax.set_xticks(range(len(K_VALUES)))
        ax.set_xticklabels(K_VALUES)
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
        ax.set_xlabel("K")
    axes[0].set_ylabel("VAL123 macro RMSE, best (K)")
    fig.savefig(dest / "domains_vs_K_val_best.png", dpi=140)
    plt.close(fig)
    base = s[s.K == 0].set_index("run_id")["common_rmse_K"]
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    for (run, f, p), h in s.groupby(["run_id", "family", "p_conditioning"]):
        h = h.sort_values("K")
        ax.plot(range(len(K_VALUES)), [h[h.K == k]["common_rmse_K"].iloc[0] - base[run] for k in K_VALUES], "-" if p else "--", color=COLOR[f], marker="o", ms=3, lw=1, alpha=0.8)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels(K_VALUES)
    ax.set_xlabel("K")
    ax.set_ylabel("common-domain RMSE minus the same run at K=0 (K)")
    ax.set_title("VAL123, best: gain from observed T (negative = better)")
    ax.grid(alpha=0.25)
    fig.savefig(dest / "gain_vs_K0_val_best.png", dpi=140)
    plt.close(fig)
    if len(by_run):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        b = by_run.sort_values("run_id")
        axes[0].bar(range(len(b)), b["compute_h"], color="#4c6a92", label="compute")
        axes[0].bar(range(len(b)), b["queue_wait_h"], bottom=b["compute_h"], color="#c9c9c9", label="queue wait")
        axes[0].set_xticks(range(len(b)))
        axes[0].set_xticklabels(b["run_id"].str.replace("TA-", ""), rotation=60, ha="right", fontsize=8)
        axes[0].set_ylabel("hours")
        axes[0].set_title("Training (all attempts)")
        axes[0].legend(frameon=False)
        if len(ev):
            e = ev.sort_values("run_id")
            axes[1].bar(range(len(e)), e["compute_s"] / 60, color="#4c6a92")
            axes[1].bar(range(len(e)), e["queue_wait_s"] / 60, bottom=e["compute_s"] / 60, color="#c9c9c9")
            axes[1].set_xticks(range(len(e)))
            axes[1].set_xticklabels(e["run_id"].str.replace("TA-", ""), rotation=60, ha="right", fontsize=8)
            axes[1].set_ylabel("minutes")
            axes[1].set_title("Final evaluation")
        fig.savefig(dest / "costs_compute_vs_queue.png", dpi=140)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--train-jobs", default="")
    ap.add_argument("--eval-jobs", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--free-package-tables", type=Path, default=ROOT / "results/month2/s7_c149_comparison/package/tables")
    args = ap.parse_args()
    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")
    matrix = json.loads(args.matrix.read_text())
    msha = sha(args.matrix)
    sh = matrix["shared_inputs"]
    mask_root = Path(sh["mask_root"])
    check_eval_mask_contract(manifest_path=mask_root / "manifest.json", payload_root=mask_root, expected_manifest_file_sha256=sh["eval_mask_manifest_file_sha256"],
                             expected_aggregate_payload_sha256=sh["eval_mask_aggregate_payload_sha256"])
    t, runs, issues = load_and_check(args.eval_root, matrix, msha, payload_mask_hashes(mask_root), args.smoke)
    if issues:
        print("AGGREGATION REFUSED. Issues:")
        [print(" -", i) for i in issues[:100]]
        return 1
    T = tables(t)
    g, pc, gr = T["global_by_setting"], T["per_condition"], T["global_by_replica"]
    tdir, fdir = args.dest / "tables", args.dest / "figures"
    tdir.mkdir(parents=True)
    fdir.mkdir()
    for k, v in T.items():
        v.to_csv(tdir / f"{k}.csv", index=False)
    # gain vs K=0 and seed dispersion
    base = g[g.K == 0].set_index(["run_id", "checkpoint_kind", "role"])
    gain = g.copy()
    for c in ("full_rmse_K", "common_rmse_K", "offanchor_rmse_K", "anchors_rmse_K", "common_centered_rmse_K"):
        gain[f"{c}_minus_K0"] = gain.apply(lambda r: r[c] - base.loc[(r.run_id, r.checkpoint_kind, r.role), c], axis=1)
    gain.to_csv(tdir / "gain_vs_K0_by_run.csv", index=False)
    disp = g.groupby(["family", "p_conditioning", "checkpoint_kind", "role", "K"])[["common_rmse_K", "full_rmse_K", "offanchor_rmse_K", "anchors_rmse_K", "common_centered_rmse_K", "common_bias_K"]].agg(["mean", "min", "max", "std"]).reset_index()
    disp.columns = ["_".join([c for c in col if c]) for col in disp.columns]
    disp.to_csv(tdir / "family_p_seed_dispersion.csv", index=False)
    # paired contrasts (descriptive)
    rows = []
    gi = g.set_index(["family", "p_conditioning", "model_seed", "checkpoint_kind", "role", "K"])["common_rmse_K"]
    have = set(gi.index)
    for ck in CHECKPOINTS:
        for role in ROLES:
            for K in K_VALUES:
                for s_ in (0, 1, 2):
                    for p in (True, False):
                        a, b = ("adapted_glu", p, s_, ck, role, K), ("sparse_unet", p, s_, ck, role, K)
                        if a in have and b in have:
                            rows.append({"contrast": "family_GLU_minus_UNet", "p_conditioning": p, "seed": s_, "checkpoint_kind": ck, "role": role, "K": K, "delta_common_rmse_K": gi[a] - gi[b]})
                    for f in ("sparse_unet", "adapted_glu"):
                        a, b = (f, True, s_, ck, role, K), (f, False, s_, ck, role, K)
                        if a in have and b in have:
                            rows.append({"contrast": f"p_on_minus_off_{f}", "p_conditioning": None, "seed": s_, "checkpoint_kind": ck, "role": role, "K": K, "delta_common_rmse_K": gi[a] - gi[b]})
    pd.DataFrame(rows).to_csv(tdir / "paired_contrasts.csv", index=False)
    # free S7 reference (different models; no identity promised)
    fp = args.free_package_tables / "global_by_setting.csv"
    if fp.exists():
        fr = pd.read_csv(fp)
        fr = fr[(fr.checkpoint_kind == "best") & (np.isclose(fr.coverage, 0.05)) & (fr.role == "val_eval")][["family", "p_conditioning", "model_seed", "rmse_K"]].rename(columns={"rmse_K": "free_S7_full_field_rmse_K"})
        ref = g[(g.checkpoint_kind == "best") & (g.role == "val_eval") & (g.K == 0)][["family", "p_conditioning", "model_seed", "full_rmse_K", "common_rmse_K"]]
        ref["p_conditioning"] = ref["p_conditioning"].astype(bool)
        fr["p_conditioning"] = fr["p_conditioning"].astype(bool)
        ref.merge(fr, on=["family", "p_conditioning", "model_seed"], how="left").rename(columns={"full_rmse_K": "assisted_K0_full_field_rmse_K", "common_rmse_K": "assisted_K0_common_rmse_K"}).to_csv(tdir / "free_S7_reference_vs_assisted_K0.csv", index=False)
    by_run, ev = pd.DataFrame(), pd.DataFrame()
    if args.train_jobs and not args.smoke:
        att, by_run, ev = costs(matrix, args.train_jobs.split(","), args.eval_jobs.split(",") if args.eval_jobs else [], args.eval_root)
        att.to_csv(tdir / "training_costs_by_attempt.csv", index=False)
        by_run.to_csv(tdir / "training_costs_by_run.csv", index=False)
        ev.to_csv(tdir / "eval_costs_by_task.csv", index=False)
    pd.DataFrame(runs).to_csv(tdir / "completeness.csv", index=False)
    figures(g, pc, fdir, matrix, by_run, ev, args.eval_root)
    gmax = max(float(r["gate_max_abs_delta_K"] or 0.0) for r in runs)
    manifest = {"matrix_sha256": msha, "n_runs": len(runs), "n_rows": int(len(t)), "gate_max_abs_delta_K": gmax, "smoke": bool(args.smoke), "ranking": "none", "winner": "none",
                "tables": sorted(p.name for p in tdir.iterdir()), "figures": sorted(p.name for p in fdir.iterdir())}
    (args.dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Studio con temperatura — valutazione finale dei 12 training assisted", "",
             "Pacchetto tecnico, descrittivo: nessuna classifica, nessun vincitore, nessuna regola di scelta applicata. Metriche = valutazione finale, non del loop.",
             "" if not args.smoke else "**SMOKE: sottoinsieme di frame, non scientifico.**", "", "## Completezza", "",
             f"- Run validi: {len(runs)}/12; righe per-frame: {len(t)}; gate di parità con il loop (macro RMSE di dominio comune entro 0.05 K per run/checkpoint/ruolo/replica e per K su VAL): scarto massimo {gmax:.3g} K.",
             f"- Matrice congelata SHA256 `{msha}`; manifest T e protocollo come da matrice.",
             "- Controlli: chiavi uniche e complete, `mask_hash` per riga = payload verificato, identità/hash dei checkpoint = matrice, forward condivisi identici alla sorgente.", "",
             "## Contenuto", "",
             "- `tables/global_by_setting.csv`, `global_by_replica.csv`, `per_condition.csv`: macro per run × checkpoint (best, last, best_seen) × ruolo × K, con full-field, dominio comune (griglia senza i 32 siti del layout), off-anchor(K) e alle ancore; RMSE, MAE, bias, centered, regionali.",
             "- `gain_vs_K0_by_run.csv`, `family_p_seed_dispersion.csv`, `paired_contrasts.csv`, `free_S7_reference_vs_assisted_K0.csv` (riferimento free: modelli diversi, nessuna identità promessa), costi (calcolo e coda separati).", "",
             "## Limiti", "",
             "- Il criterio di selezione (media dei sei K, dominio comune, VAL 5%) è nuovo rispetto a S7; VAL123 resta usata nella selezione.",
             "- Le tre coppie layout-T/maschera-velocità misurano variabilità congiunta: non attribuibile ai soli sensori T.",
             "- Assisted(K=0) è un altro modello: non è il free originale. T ideali, nessun rumore, nessun TEST.", "",
             "## Figure", ""] + [f"- `{f}`" for f in manifest["figures"]]
    (args.dest / "INDEX.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
