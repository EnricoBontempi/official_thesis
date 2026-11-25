#!/usr/bin/env python3
"""Checks over the shared specialist smoke outputs of one family (non-scientific). Writes smoke_checks.json."""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import torch

REG = ("N1", "N2", "G1", "G1eq", "N1G1")


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()] if Path(p).is_file() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-dir", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sm, fam, seed = a.smoke_dir, a.family, a.seed
    glu = fam == "glu"
    rid = {r: f"NI-{fam}-on-{r}-s{seed}" for r in REG}
    orig = f"S7-C149-{fam}-on-s{seed}"
    r0s = f"NI-{fam}-on-R0smoke-s{seed}"
    runs = {"ORIG": sm / "ORIG" / orig, "R0S": sm / "R0S" / r0s, **{r: sm / r / rid[r] for r in REG}}
    rep: dict = {"family": fam, "non_scientific": True, "checks": {}}
    ck = rep["checks"]

    def summ(d):
        return json.loads((d / "summary.json").read_text())

    def ckpt(d, name):
        return torch.load(d / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)

    # 1 completion + finiteness + readback of checkpoints
    for k, d in runs.items():
        s = summ(d)
        c = ckpt(d, "last"); b = ckpt(d, "best")
        fin = all(torch.isfinite(v).all() for v in c["model_state_dict"].values() if v.is_floating_point())
        ev = c["eval_history"]
        vals_ok = all(math.isfinite(e["val_mean_condition_rmse_K"]) for e in ev) and all(math.isfinite(x["loss"]) for x in c["curves"])
        ck[f"{k}_completed_finite"] = bool(s["status"] in ("completed", "reached_max_epochs") and fin and vals_ok and len(ev) >= 1
                                           and b["epoch_state"]["epoch"] >= 1)
        ck[f"{k}_status"] = s["status"]
    # 2 init identical across the five regimes and equal to the original clean run
    init = {k: json.loads((d / "init_manifest.json").read_text())["init_state_dict_sha256"] for k, d in runs.items()}
    ck["init_equal_across_regimes"] = len({init[r] for r in REG}) == 1
    ck["init_equal_original_clean"] = init["R0S"] == init["ORIG"] == init["N1"]
    # 3 same presentation stream everywhere
    st = {k: [t["indices_sha256"] for t in jl(d / "stream_trace.jsonl")] for k, d in runs.items()}
    ck["stream_identical_all_runs"] = all(st[k] == st["ORIG"] for k in st)
    # 4 identity specialist reproduces the original path
    co, cr = ckpt(runs["ORIG"], "last"), ckpt(runs["R0S"], "last")
    dl = max(abs(x["loss"] - y["loss"]) for x, y in zip(co["curves"], cr["curves"]))
    dv = max(abs(x["val_mean_condition_rmse_K"] - y["val_mean_condition_rmse_K"]) for x, y in zip(co["eval_history"], cr["eval_history"]))
    dw = max(float((v - cr["model_state_dict"][k]).abs().max()) for k, v in co["model_state_dict"].items() if v.is_floating_point())
    rep["identity_vs_original"] = {"max_abs_loss_diff": dl, "max_abs_val_rmse_diff_K": dv, "max_abs_weight_diff": dw,
                                   "bitwise_weights": bool(all(torch.equal(v, cr["model_state_dict"][k]) for k, v in co["model_state_dict"].items()))}
    ck["identity_reproduces_original_numerically"] = bool(dl < (1e-4 if glu else 1e-6) and dv < (0.05 if glu else 1e-3))
    # 5 specialist VAL recorded, never selection
    for r in REG:
        ev = ckpt(runs[r], "last")["eval_history"]
        ck[f"{r}_specialist_val_recorded_not_selection"] = bool(all(
            "specialist_val_mean_condition_rmse_K" in e and e.get("specialist_val_used_for_selection") is False
            and math.isfinite(e["specialist_val_mean_condition_rmse_K"]) for e in ev))
        c = ckpt(runs[r], "last")
        scores = [e["val_mean_condition_rmse_K"] for e in c["eval_history"]]
        ck[f"{r}_best_from_clean_val"] = bool(abs(c["epoch_state"]["best_val_metric"] - min(scores)) < 1e-9)
    ck["original_has_no_specialist_fields"] = all("specialist_val_mean_condition_rmse_K" not in e for e in ckpt(runs["ORIG"], "last")["eval_history"])
    # 6 resume
    cmp_p = sm / "compare_resume.json"
    if cmp_p.is_file():
        c = json.loads(cmp_p.read_text())
        rep["resume_compare"] = {k: c.get(k) for k in ("state_all_equal", "lr_equal", "scheduler_state_equal", "sampler_state_equal",
                                                      "stream_resumed_matches_continuous", "curve_loss_max_abs_diff", "val_scores_max_abs_diff", "weights_continuous_vs_resumed")}
        ck["resume_ok"] = bool(c.get("state_all_equal") and c.get("lr_equal") and c.get("scheduler_state_equal")
                               and c.get("sampler_state_equal") and c.get("stream_resumed_matches_continuous"))
    else:
        ck["resume_ok"] = False
    # 7 ledger of the killed run: fresh then resume
    led = jl(sm / "KILL" / "_ledger" / f"{rid['N1G1']}.jsonl")
    acts = [e.get("action") for e in led if e["event"] == "attempt_start"]
    ck["kill_ledger_fresh_then_resume"] = acts[:2] == ["fresh", "resume"]
    # 8 timing from the two real-size runs
    t = {}
    for name in ("T1", "T2"):
        d = sm / name / rid["N1G1"]
        s = summ(d)
        t[name] = float(s["elapsed_s"])
    rep["timing_s"] = {"T1_one_epoch_plus_eval": t["T1"], "T2_two_epochs_plus_eval": t["T2"],
                       "epoch_s": t["T2"] - t["T1"], "eval_s": t["T1"] - (t["T2"] - t["T1"])}
    rep["epoch_s"], rep["eval_s"] = rep["timing_s"]["epoch_s"], rep["timing_s"]["eval_s"]
    # 9 memory
    mem = Path(sm / "gpu_mem.csv")
    if mem.is_file():
        vals = [int(x.split(",")[2].strip().split()[0]) for x in mem.read_text().splitlines()[1:] if len(x.split(",")) >= 3]
        rep["vram_peak_MiB_sampled"] = max(vals) if vals else None
    ck["all_pass"] = bool(all(v for k, v in ck.items() if isinstance(v, bool)))
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"all_pass": ck["all_pass"], **{k: v for k, v in ck.items() if v is False}}, indent=1))
    return 0 if ck["all_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
