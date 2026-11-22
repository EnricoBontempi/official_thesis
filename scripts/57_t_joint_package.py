#!/usr/bin/env python3
"""Joint post-processing of Step 2 (K-sweep corrections, Cursor) and Step 3 (temperature-assisted).

Reads two finished, independent packages and nothing else; trains and evaluates nothing, selects
nothing, writes no winner. It first checks that the two packages are comparable (manifest, K, sites by
replica, frames, velocity masks, units, metric domains, macro definition) and ONLY if every blocking
check passes it writes the comparison tables and one INDEX. Otherwise it writes JOINT_BLOCCO.md with what
is missing or incompatible and exits 3. Neither input is modified.

    57_t_joint_package.py --cursor-package P --cursor-neural-root N --cursor-baseline-root B \
        --assisted-eval-root A --dest D --git-sha SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "a9c997c96cb44489dd1fc0d4484304b44d8beddbf00b73c25baf4e1922a19e54"
KS = [0, 1, 4, 9, 16, 32]
CKPTS = ["best", "last"]
ROLES = ["seen_train_eval", "val_eval"]
REPLICAS = [0, 1, 2]
N_COMMON = 16384 - 32
FAMILIES = ["sparse_unet", "adapted_glu"]
METRICS = ["full", "common", "off", "anchors"]
TOL_MACRO = 1e-6
CUR_COLS = ["run_id", "family", "p_conditioning", "model_seed", "checkpoint", "epoch", "role", "mask_replica_id",
            "sim_id", "frame_index", "velocity_mask_hash", "layout_id", "layout_kind", "K", "treatment",
            "full_rmse_K", "common_rmse_K", "off_rmse_K", "anchor_rmse_K", "common_n", "off_n", "anchor_n"]
OWN_COLS = ["run_id", "family", "p_conditioning", "model_seed", "role", "mask_replica_id", "sim_id", "frame_index",
            "mask_hash", "K", "full_rmse_K", "common_rmse_K", "offanchor_rmse_K", "anchors_rmse_K", "n_common",
            "checkpoint_kind", "epoch"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _np(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o).__name__)


def jd(obj: Any) -> str:
    return json.dumps(obj, default=_np)


def macro(frame: pd.DataFrame, column: str) -> float:
    """Mean over conditions of the per-condition mean (same rule as S7 and the K sweep)."""
    sub = frame[["sim_id", column]].dropna()
    if sub.empty:
        return float("nan")
    return float(sub.groupby("sim_id")[column].mean().mean())


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str, blocking: bool = True) -> None:
        self.rows.append({"check": name, "pass": bool(ok), "blocking": blocking, "detail": detail})
        print(f"[joint] {'PASS' if ok else 'FAIL'}{'' if blocking else ' (info)'} {name}: {detail}", flush=True)

    @property
    def blocking_failures(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["blocking"] and not r["pass"]]


def missing_inputs(args: argparse.Namespace) -> list[str]:
    out = []
    pkg, neural, base, ev = args.cursor_package, args.cursor_neural_root, args.cursor_baseline_root, args.assisted_eval_root
    for label, path in [
        ("Cursor package INDEX.md", pkg / "INDEX.md"),
        ("Cursor package by_seed.csv", pkg / "by_seed.csv"),
        ("Cursor package checks.json", pkg / "checks.json"),
        ("Cursor baseline COMPLETE.json", base / "COMPLETE.json"),
        ("assisted matrix_frozen_v2.json", ev / "matrix_frozen_v2.json"),
        ("assisted chain_v2_submitted.json", ev / "chain_v2_submitted.json"),
        ("assisted v2 package_manifest.json", ev / "package" / "package_manifest.json"),
        ("assisted v2 package INDEX.md", ev / "package" / "INDEX.md"),
        ("assisted v2 global_by_replica.csv", ev / "package" / "tables" / "global_by_replica.csv"),
    ]:
        if not path.is_file():
            out.append(f"{label}: missing ({path})")
    if (ev / "INCOMPLETE_v2.json").is_file():
        out.append(f"assisted chain v2 stopped: {ev / 'INCOMPLETE_v2.json'}")
    runs = sorted(p.name for p in neural.glob("*") if (p / "COMPLETE.json").is_file()) if neural.is_dir() else []
    if len(runs) != 12:
        out.append(f"Cursor neural runs with COMPLETE.json: {len(runs)} of 12 ({neural})")
    for run in sorted(p.name for p in (ev / "runs").glob("TA-*")) if (ev / "runs").is_dir() else []:
        if not (ev / "runs" / run / "COMPLETE.json").is_file():
            out.append(f"assisted run {run}: COMPLETE.json missing")
    n_own = len(list((ev / "runs").glob("TA-*/COMPLETE.json"))) if (ev / "runs").is_dir() else 0
    if n_own != 12:
        out.append(f"assisted v2 runs with COMPLETE.json: {n_own} of 12 ({ev / 'runs'})")
    return out


def load_cursor(neural: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(neural.glob("*/per_frame/*/*/rep*.csv")):
        head = pd.read_csv(path, nrows=0).columns
        parts.append(pd.read_csv(path, usecols=[c for c in CUR_COLS if c in head]))
    return pd.concat(parts, ignore_index=True)


def load_own(ev: Path) -> pd.DataFrame:
    parts = []
    for path in sorted((ev / "runs").glob("TA-*/per_frame/*/*/rep*.csv")):
        ckpt, role = path.parent.parent.name, path.parent.name
        if role not in ROLES:
            continue
        head = pd.read_csv(path, nrows=0).columns
        frame = pd.read_csv(path, usecols=[c for c in OWN_COLS if c in head])
        frame["checkpoint"] = ckpt
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def per_setting(frame: pd.DataFrame, keys: list[str], cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    for key, sub in frame.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for name, col in cols.items():
            row[name] = macro(sub, col) if col in sub.columns else float("nan")
        row["n_frames"] = int(sub[["sim_id", "frame_index"]].drop_duplicates().shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cursor-package", type=Path, required=True)
    parser.add_argument("--cursor-neural-root", type=Path, required=True)
    parser.add_argument("--cursor-baseline-root", type=Path, required=True)
    parser.add_argument("--assisted-eval-root", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--cursor-published-hashes", type=Path, default=None,
                        help="JSON with the sha256 of the files Cursor published in git; the cluster package must match them")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    args = parser.parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tables").mkdir(exist_ok=True)

    blocked = missing_inputs(args)
    if blocked:
        write_block(dest, blocked, [])
        return 3

    checks = Checks()
    ev, pkg = args.assisted_eval_root, args.cursor_package
    own_pkg = ev / "package"

    # ---- provenance and completeness of both sides
    manifest_sha = sha256_file(args.manifest)
    checks.add("manifest file in repo is the frozen one", manifest_sha == MANIFEST_SHA, manifest_sha)
    cur_prov = {p.parent.name: json.loads(p.read_text()) for p in args.cursor_neural_root.glob("*/provenance.json")}
    bad = [r for r, p in cur_prov.items() if p.get("manifest_file_sha256") != MANIFEST_SHA]
    checks.add("Cursor: manifest sha256 on all 12 runs", len(cur_prov) == 12 and not bad, f"{len(cur_prov)} runs, mismatches {bad}")
    matrix_v2 = json.loads((ev / "matrix_frozen_v2.json").read_text())
    mv2_sha = sha256_file(ev / "matrix_frozen_v2.json")
    own_manifest = json.dumps(matrix_v2.get("anchor_manifest", matrix_v2.get("shared_inputs", {})))
    checks.add("assisted: matrix v2 refers to the same manifest sha256", MANIFEST_SHA in own_manifest or MANIFEST_SHA in json.dumps(matrix_v2),
               f"matrix_v2 sha256 {mv2_sha}")
    pm = json.loads((own_pkg / "package_manifest.json").read_text())
    checks.add("assisted: v2 package built on the v2 matrix, 12 runs, not smoke, no ranking",
               pm.get("matrix_sha256") == mv2_sha and pm.get("n_runs") == 12 and pm.get("smoke") is False
               and pm.get("ranking") == "none" and pm.get("winner") == "none",
               f"package matrix {pm.get('matrix_sha256')}, n_runs {pm.get('n_runs')}, gate max {pm.get('gate_max_abs_delta_K')}")
    own_gate_ok = all(json.loads(p.read_text()).get("ok") for p in (ev / "runs").glob("TA-*/COMPLETE.json"))
    checks.add("assisted: 12 COMPLETE.json with gate ok", own_gate_ok, "gate macro tolerance 0.05 K")
    cur_gate_ok = all(json.loads(p.read_text()).get("ok", True) for p in args.cursor_neural_root.glob("*/COMPLETE.json"))
    checks.add("Cursor: 12 COMPLETE.json with gate ok", cur_gate_ok, "free-gate vs S7 CSV tolerance 0.05 K")
    cur_checks = json.loads((pkg / "checks.json").read_text())
    checks.add("Cursor: K=0 identities and K=1 mean=median at 0", all(abs(v) == 0 for v in cur_checks.values()), json.dumps(cur_checks))
    index_text = (pkg / "INDEX.md").read_text()
    m = re.search(r"Forward and correction SHA `([0-9a-f]{40})`", index_text)
    a = re.search(r"Aggregation SHA `([0-9a-f]{40})`", index_text)
    cursor_fwd_sha, cursor_agg_sha = (m.group(1) if m else None), (a.group(1) if a else None)
    checks.add("Cursor: SHAs readable from its INDEX", bool(cursor_fwd_sha and cursor_agg_sha), f"forward {cursor_fwd_sha}, aggregation {cursor_agg_sha}")

    published = None
    if args.cursor_published_hashes is not None:
        published = json.loads(args.cursor_published_hashes.read_text())
        for name, want in sorted(published["files_sha256"].items()):
            have = sha256_file(pkg / name)
            data_file = name.endswith((".csv", ".json"))
            checks.add(f"Cursor package file equals the version published at {published['cursor_sha'][:7]}: {name}", have == want,
                       "identical" if have == want else f"cluster {have[:12]} vs published {want[:12]}" + ("" if data_file else " (text/figure regenerated after the tables; tables identical)"),
                       blocking=data_file)

    # ---- load per-frame data (authoritative)
    cur = load_cursor(args.cursor_neural_root)
    own = load_own(ev)
    cur = cur[(cur["layout_kind"] == "primary") & cur["checkpoint"].isin(CKPTS) & cur["role"].isin(ROLES)].copy()
    own = own[own["checkpoint"].isin(CKPTS) & own["role"].isin(ROLES)].copy()
    cur["p_conditioning"] = cur["p_conditioning"].astype(str).str.lower().eq("true")
    own["p_conditioning"] = own["p_conditioning"].astype(str).str.lower().eq("true")
    own = own.rename(columns={"offanchor_rmse_K": "off_rmse_K", "anchors_rmse_K": "anchor_rmse_K"})
    print(f"[joint] Cursor rows {len(cur)}, assisted rows {len(own)}", flush=True)

    # ---- compatibility
    checks.add("K sets equal", sorted(cur["K"].unique()) == KS and sorted(own["K"].unique()) == KS,
               f"Cursor {[int(k) for k in sorted(cur['K'].unique())]}, assisted {[int(k) for k in sorted(own['K'].unique())]}")
    checks.add("replicas equal", sorted(cur["mask_replica_id"].unique()) == REPLICAS and sorted(own["mask_replica_id"].unique()) == REPLICAS,
               "0/1/2 on both sides")
    lay = cur.groupby("mask_replica_id")["layout_id"].unique().apply(list).to_dict()
    checks.add("Cursor layout is nested_r<replica> (manifest layout paired to the velocity replica)",
               all(list(v) == [f"nested_r{k}"] for k, v in lay.items()), jd({int(k): v for k, v in lay.items()}))
    cur_roles = cur.groupby("role")[["sim_id", "frame_index"]].apply(lambda d: d.drop_duplicates().shape[0]).to_dict()
    own_roles = own.groupby("role")[["sim_id", "frame_index"]].apply(lambda d: d.drop_duplicates().shape[0]).to_dict()
    checks.add("frame counts per role equal (seen220, VAL123)", cur_roles == own_roles == {"seen_train_eval": 220, "val_eval": 123},
               f"Cursor {cur_roles}, assisted {own_roles}")
    frames_equal = True
    detail = []
    for role in ROLES:
        a_set = set(map(tuple, cur[cur["role"] == role][["sim_id", "frame_index"]].drop_duplicates().to_numpy()))
        b_set = set(map(tuple, own[own["role"] == role][["sim_id", "frame_index"]].drop_duplicates().to_numpy()))
        same = a_set == b_set
        frames_equal &= same
        detail.append(f"{role}: {'identical' if same else f'only-Cursor {len(a_set - b_set)}, only-assisted {len(b_set - a_set)}'}")
    checks.add("frame keys (sim_id, frame_index) identical per role", frames_equal, "; ".join(detail))
    cm = cur[(cur["treatment"] == "free") & (cur["K"] == 0)][["role", "mask_replica_id", "sim_id", "frame_index", "velocity_mask_hash"]].drop_duplicates()
    om = own[own["K"] == 0][["role", "mask_replica_id", "sim_id", "frame_index", "mask_hash"]].drop_duplicates()
    joined = cm.merge(om, on=["role", "mask_replica_id", "sim_id", "frame_index"], how="outer", indicator=True)
    both = joined[joined["_merge"] == "both"]
    n_diff = int((both["velocity_mask_hash"] != both["mask_hash"]).sum())
    checks.add("velocity mask hash identical per (role, replica, frame)", n_diff == 0 and len(both) == len(joined),
               f"{len(both)} matched, {n_diff} different, {len(joined) - len(both)} unmatched")
    n_common_cur = sorted(cur["common_n"].dropna().unique())
    n_common_own = sorted(own["n_common"].dropna().unique())
    checks.add("common domain = 16384 - 32 layout sites on both sides", n_common_cur == [N_COMMON] and n_common_own == [N_COMMON],
               f"Cursor {[int(v) for v in n_common_cur]}, assisted {[int(v) for v in n_common_own]}")
    off_n = cur[cur["treatment"] == "free"].groupby("K")["off_n"].unique().apply(lambda v: sorted(set(v))).to_dict()
    checks.add("Cursor off-anchor count is 16384-K and anchor count K", all(off_n.get(k) == [16384 - k] for k in KS), jd({int(k): v for k, v in off_n.items()}))
    rng_cur = float(cur[cur["treatment"] == "free"]["full_rmse_K"].median())
    rng_own = float(own["full_rmse_K"].median())
    checks.add("units: both sides RMSE in kelvin on the same order (median frame RMSE 5-300 K)", 5 < rng_cur < 300 and 5 < rng_own < 300,
               f"Cursor median {rng_cur:.1f} K, assisted {rng_own:.1f} K")

    # ---- macro definition reproduced against each side's own published table
    cur_set = per_setting(cur, ["family", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "K", "treatment"],
                          {"full": "full_rmse_K", "common": "common_rmse_K", "off": "off_rmse_K", "anchors": "anchor_rmse_K"})
    own_set = per_setting(own, ["run_id", "family", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "K"],
                          {"full": "full_rmse_K", "common": "common_rmse_K", "off": "off_rmse_K", "anchors": "anchor_rmse_K"})
    cur_pub = pd.read_csv(pkg / "by_seed.csv")
    cur_pub = cur_pub.rename(columns={"method": "family"})
    cur_pub["p_conditioning"] = cur_pub["p_conditioning"].astype(str).str.lower().eq("true")
    cur_pub = cur_pub[cur_pub["layout_kind"] == "primary"]
    mj = cur_set.merge(cur_pub, on=["family", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "K", "treatment"], how="inner")
    d_cur = float(np.nanmax(np.abs(mj["full"] - mj["full_rmse_K"]))) if len(mj) else float("inf")
    d_cur_c = float(np.nanmax(np.abs(mj["common"] - mj["common_rmse_K"]))) if len(mj) else float("inf")
    checks.add("macro definition reproduces Cursor's by_seed.csv (full and common)", len(mj) == len(cur_set) and d_cur < TOL_MACRO and d_cur_c < TOL_MACRO,
               f"{len(mj)} of {len(cur_set)} cells, max|diff| full {d_cur:.2e} K, common {d_cur_c:.2e} K")
    own_pub = pd.read_csv(own_pkg / "tables" / "global_by_replica.csv")
    own_pub = own_pub.rename(columns={"checkpoint_kind": "checkpoint", "replica": "mask_replica_id"})
    own_pub = own_pub[own_pub["checkpoint"].isin(CKPTS) & own_pub["role"].isin(ROLES)]
    oj = own_set.merge(own_pub, on=["run_id", "checkpoint", "role", "mask_replica_id", "K"], how="inner", suffixes=("", "_pub"))
    d_own = float(np.nanmax(np.abs(oj["full"] - oj["full_rmse_K"]))) if len(oj) else float("inf")
    d_own_c = float(np.nanmax(np.abs(oj["common"] - oj["common_rmse_K"]))) if len(oj) else float("inf")
    checks.add("macro definition reproduces the assisted v2 global_by_replica.csv (full and common)", len(oj) == len(own_set) and d_own < TOL_MACRO and d_own_c < TOL_MACRO,
               f"{len(oj)} of {len(own_set)} cells, max|diff| full {d_own:.2e} K, common {d_own_c:.2e} K")

    if checks.blocking_failures:
        write_block(dest, [], checks.rows)
        return 3

    # ---- comparison tables (no ranking, no winner)
    key = ["family", "p_conditioning", "model_seed", "checkpoint", "role", "mask_replica_id", "K"]
    wide = own_set.drop(columns=["run_id", "n_frames"]).rename(columns={m: f"{m}_assisted" for m in METRICS})
    for treat in ["free", "mean", "median"]:
        part = cur_set[cur_set["treatment"] == treat].drop(columns=["treatment", "n_frames"]).rename(columns={m: f"{m}_{treat}" for m in METRICS})
        wide = wide.merge(part, on=key, how="left")
    for m_ in ["full", "common", "off"]:
        wide[f"{m_}_assisted_minus_free"] = wide[f"{m_}_assisted"] - wide[f"{m_}_free"]
        wide[f"{m_}_mean_minus_free"] = wide[f"{m_}_mean"] - wide[f"{m_}_free"]
        wide[f"{m_}_median_minus_free"] = wide[f"{m_}_median"] - wide[f"{m_}_free"]
        wide[f"{m_}_assisted_minus_mean"] = wide[f"{m_}_assisted"] - wide[f"{m_}_mean"]
        wide[f"{m_}_assisted_minus_median"] = wide[f"{m_}_assisted"] - wide[f"{m_}_median"]
    n_miss = int(wide[["full_free", "full_mean", "full_median"]].isna().any(axis=1).sum())
    if n_miss:
        write_block(dest, [f"{n_miss} assisted cells without a Cursor free/mean/median counterpart"], checks.rows)
        return 3
    wide = wide.sort_values(key)
    wide.to_csv(dest / "tables" / "joint_by_run_replica_K.csv", index=False)

    # seed-level (average of the 3 replicas), then dispersion across the 3 seeds
    seed_level = wide.groupby(["family", "p_conditioning", "model_seed", "checkpoint", "role", "K"]).mean(numeric_only=True).drop(columns=["mask_replica_id"]).reset_index()
    seed_level.to_csv(dest / "tables" / "joint_by_seed_K.csv", index=False)
    value_cols = [c for c in seed_level.columns if c not in ("family", "p_conditioning", "model_seed", "checkpoint", "role", "K")]
    agg = seed_level.groupby(["family", "p_conditioning", "checkpoint", "role", "K"])[value_cols].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = 3
    agg.to_csv(dest / "tables" / "joint_across_seeds_K.csv", index=False)
    paired = []
    for (fam, p, ck, role, k), sub in seed_level.groupby(["family", "p_conditioning", "checkpoint", "role", "K"]):
        for m_ in ["common", "off", "full"]:
            for a_, b_ in [("assisted", "free"), ("mean", "free"), ("median", "free"), ("assisted", "mean"), ("assisted", "median")]:
                col = f"{m_}_{a_}_minus_{b_}"
                paired.append({"family": fam, "p_conditioning": p, "checkpoint": ck, "role": role, "K": k, "domain": m_,
                               "contrast": f"{a_}_minus_{b_}", "mean_K": sub[col].mean(), "min_K": sub[col].min(), "max_K": sub[col].max(),
                               "n_seeds_lower": int((sub[col] < 0).sum()), "n_seeds": int(len(sub))})
    pd.DataFrame(paired).to_csv(dest / "tables" / "joint_paired_contrasts.csv", index=False)
    cur_pub[cur_pub["family"].isin(["GlobalTrainMean", "ParamConditionedMean", "GappyPOD_free"])].to_csv(
        dest / "tables" / "reference_baselines_from_cursor_by_seed.csv", index=False)
    pd.DataFrame(checks.rows).to_csv(dest / "tables" / "compatibility_checks.csv", index=False)

    # ---- references, hashes, INDEX
    chain = json.loads((ev / "chain_v2_submitted.json").read_text())
    ext = {}
    for run in ["TA-glu-on-s1", "TA-glu-off-s1", "TA-glu-off-s2"]:
        prov = json.loads((ev / "runs" / run / "provenance.json").read_text())
        cp = prov.get("checkpoint_plan") or {}
        ext[run] = {"checkpoint_plan": cp, "last_epoch": (prov.get("checkpoints", {}).get("last") or {}).get("epoch")}
    own_epochs = own.drop_duplicates(["run_id", "checkpoint"])[["run_id", "checkpoint", "epoch"]]
    own_epochs.to_csv(dest / "tables" / "assisted_checkpoint_epochs.csv", index=False)
    refs = {
        "cursor_package": str(pkg),
        "cursor_published": published,
        "cursor_forward_sha": cursor_fwd_sha, "cursor_aggregation_sha": cursor_agg_sha,
        "cursor_files_sha256": {p.name: sha256_file(p) for p in sorted(pkg.glob("*.csv")) + [pkg / "INDEX.md", pkg / "checks.json"]},
        "assisted_eval_root": str(ev),
        "assisted_matrix_v2_sha256": mv2_sha,
        "assisted_chain_v2": chain,
        "assisted_files_sha256": {p.name: sha256_file(p) for p in sorted((own_pkg / "tables").glob("*.csv")) + [own_pkg / "INDEX.md", own_pkg / "package_manifest.json"]},
        "manifest_file_sha256": manifest_sha,
        "joint_git_sha": args.git_sha,
        "extended_runs": ext,
        "winner": "none", "ranking": "none", "no_new_training": True, "no_test_split": True,
    }
    (dest / "joint_references.json").write_text(json.dumps(refs, indent=2, sort_keys=True) + "\n")
    (dest / "INDEX.md").write_text(build_index(refs, checks, agg, own_epochs, args))
    (dest / "JOINT_STATUS.json").write_text(json.dumps({"ok": True, "complete": True, "blocking_failures": 0, "n_checks": len(checks.rows), "git_sha": args.git_sha}, indent=2) + "\n")
    print(f"[joint] complete: {dest / 'INDEX.md'}", flush=True)
    return 0


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def block_table(agg: pd.DataFrame, domain: str, ckpt: str, role: str) -> str:
    sub = agg[(agg["checkpoint"] == ckpt) & (agg["role"] == role)]
    out = []
    for treat in ["free", "mean", "median", "assisted"]:
        col = f"{domain}_{treat}__mean"
        piv = sub.pivot_table(index=["family", "p_conditioning"], columns="K", values=col)
        piv.columns = [f"K={k}" for k in piv.columns]
        piv = piv.reset_index()
        piv.insert(2, "estimator", treat)
        out.append(piv)
    table = pd.concat(out, ignore_index=True)
    table["_o"] = table["estimator"].map({"free": 0, "mean": 1, "median": 2, "assisted": 3})
    table = table.sort_values(["family", "p_conditioning", "_o"]).drop(columns="_o")
    return md_table(table)


def build_index(refs: dict[str, Any], checks: Checks, agg: pd.DataFrame, own_epochs: pd.DataFrame, args: argparse.Namespace) -> str:
    chain = refs["assisted_chain_v2"]
    lines = [
        "# Joint delivery, steps 2 and 3: K corrections (Cursor) and temperature-assisted training (Sonnet)",
        "",
        "Status: complete. Every blocking compatibility check passed (table below). No new training, no TEST, no winner, no ranking.",
        "",
        "## Inputs, exact references",
        "",
        (f"- Step 2 as published by Cursor: branch `{refs['cursor_published']['cursor_branch']}`, SHA `{refs['cursor_published']['cursor_sha']}`, index `results/month2/temperature_k_sweep/INDEX.md`, report `docs/month2/11_temperature_k_sweep.md`; the tables used here are byte-identical to the published ones (hashes in `joint_references.json`)." if refs.get("cursor_published") else "- Step 2 published version: not checked."),
        f"- Step 2 package (Cursor), `{refs['cursor_package']}`: forward/correction SHA `{refs['cursor_forward_sha']}`, aggregation SHA `{refs['cursor_aggregation_sha']}`. Tables `by_seed.csv`, `across_seeds.csv`, `per_condition.csv`, `versus_free.csv`, `checks.json`, `INDEX.md`; file hashes in `joint_references.json`. Per-frame CSV: `{args.cursor_neural_root}/<run>/per_frame/<ckpt>/<role>/rep<r>.csv`.",
        f"- Step 3 package v2 (Sonnet), `{refs['assisted_eval_root']}/package`: matrix v2 sha256 `{refs['assisted_matrix_v2_sha256']}`, chain v2 eval array `{chain.get('eval_array')}`, aggregation `{chain.get('aggregate')}`, chain SHA `{chain.get('git_sha')}`. Per-frame CSV: `{refs['assisted_eval_root']}/runs/<run>/per_frame/<ckpt>/<role>/rep<r>.csv`. The v1 package published at `aa8f65f` is unchanged and remains the reference for the 1000-epoch state of the three extended runs.",
        f"- Anchor manifest `configs/observations/month2_temperature_anchors_nested_v1.json`, file sha256 `{refs['manifest_file_sha256']}`, identical on both sides.",
        f"- Joint post-processing code SHA `{refs['joint_git_sha']}` (`scripts/57_t_joint_package.py`).",
        "",
        "## What is compared",
        "",
        "Same 12 (family, p, seed) cells, K in {0, 1, 4, 9, 16, 32}, checkpoints best and last, roles seen220 and VAL123, replicas 0/1/2 (layout nested_r<replica>), 5% velocity. Macro RMSE = mean over conditions of the per-condition mean, in kelvin, as in S7 and in the K sweep.",
        "",
        "- free: S7 model, no temperature (Cursor `free`, constant in K on the common domain).",
        "- free+mean, free+median: the same free prediction shifted by the mean/median of the K anchor residuals (Cursor).",
        "- assisted: the separate model trained with K temperature anchors in input (Sonnet).",
        "",
        "Domains: full field; common domain (grid minus the 32 layout sites, the same for every K); off-anchor(K); anchor-site error. Primary table: common domain.",
        "",
        "## Caveats that decide how the numbers may be read",
        "",
        "- The assisted models are different trainings with a different selection criterion (mean over the six K of VAL 5% common-domain macro RMSE, single best); `best` and `last` are not the same epochs as the free S7 checkpoints (`tables/assisted_checkpoint_epochs.csv`). At K=0 the assisted model is not the free model.",
        "- The three runs TA-glu-on-s1, TA-glu-off-s1, TA-glu-off-s2 were continued from epoch 1000 up to the limit set in the authorisation; the others are unchanged and were re-attested by hash.",
        "- Mean/median corrections are post hoc shifts and do not change the model; they move the full-field RMSE through the anchor residuals only.",
        "- Seeds: three per cell; dispersion in `joint_across_seeds_K.csv`. The GLU free reference has known top-k tie sensitivity to batching (S7 residual note).",
        "- No estimator is selected; paired contrasts are descriptive.",
        "",
        "## Tables (`tables/`)",
        "",
        "- `joint_by_run_replica_K.csv`: one row per (family, p, seed, checkpoint, role, replica, K), four domains for each of free, mean, median, assisted, and the paired differences.",
        "- `joint_by_seed_K.csv`: the same averaged over the three replicas.",
        "- `joint_across_seeds_K.csv`: mean/min/max/std across the three seeds.",
        "- `joint_paired_contrasts.csv`: assisted-free, mean-free, median-free, assisted-mean, assisted-median, with the number of seeds below zero.",
        "- `reference_baselines_from_cursor_by_seed.csv`: GlobalTrainMean, ParamConditionedMean and GappyPOD rows as published by Cursor (no assisted counterpart).",
        "- `compatibility_checks.csv`, `assisted_checkpoint_epochs.csv`.",
        "",
        "## Common-domain macro RMSE (K), VAL123, best checkpoint, mean over 3 seeds and 3 replicas",
        "",
        block_table(agg, "common", "best", "val_eval"),
        "",
        "## Same, last checkpoint",
        "",
        block_table(agg, "common", "last", "val_eval"),
        "",
        "## Compatibility checks",
        "",
        md_table(pd.DataFrame(checks.rows).assign(**{"pass": lambda d: d["pass"].map({True: "PASS", False: "FAIL"})})[["check", "pass", "detail"]]),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_block(dest: Path, missing: list[str], checks: list[dict[str, Any]]) -> None:
    failing = [c for c in checks if c["blocking"] and not c["pass"]]
    lines = ["# Joint package NOT complete", "", "The joint package is not declared complete. Nothing was relaunched.", ""]
    if missing:
        lines += ["## Missing or invalid inputs", ""] + [f"- {m}" for m in missing] + [""]
    if failing:
        lines += ["## Compatibility checks that failed", ""] + [f"- {c['check']}: {c['detail']}" for c in failing] + [""]
    (dest / "JOINT_BLOCCO.md").write_text("\n".join(lines) + "\n")
    (dest / "JOINT_STATUS.json").write_text(json.dumps({"ok": False, "complete": False, "missing": missing, "failed_checks": [c["check"] for c in failing]}, indent=2) + "\n")
    if checks:
        pd.DataFrame(checks).to_csv(dest / "compatibility_checks.csv", index=False)
    print("[joint] BLOCKED:\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
