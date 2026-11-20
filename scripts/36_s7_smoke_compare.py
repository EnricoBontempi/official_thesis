#!/usr/bin/env python3
"""Compare an uninterrupted smoke run with a killed-and-resumed one (non-scientific)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _close(a, b, rel: float = 1e-4) -> bool:
    """Exact for ints/strings/structure; relative tolerance for floats (GPU run-to-run noise)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if a == b:  # also covers inf == inf
            return True
        return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def _trace(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def _load(d: Path):
    return torch.load(d / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuous", type=Path, required=True)
    ap.add_argument("--continuous2", type=Path, default=None, help="second uninterrupted run (nondeterminism control)")
    ap.add_argument("--resumed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    c, r = _load(args.continuous), _load(args.resumed)
    rep: dict = {"non_scientific_smoke": True}

    def wdiff(a, b):
        worst, equal = 0.0, True
        for k, v in a["model_state_dict"].items():
            w = b["model_state_dict"][k]
            equal &= bool(torch.equal(v, w))
            if v.is_floating_point():
                worst = max(worst, float((v - w).abs().max()))
        return {"bitwise_equal": equal, "max_abs_diff": worst}

    rep["weights_continuous_vs_resumed"] = wdiff(c, r)
    if args.continuous2:
        rep["weights_continuous_vs_continuous2"] = wdiff(c, _load(args.continuous2))
    keys = ("epoch", "update", "samples_seen", "mask_epoch", "best_val_epoch",
            "early_stop_bad_evals", "sampler_cycle", "sampler_pos", "sampler_presentations_served")
    rep["state_equal"] = {k: (c["epoch_state"].get(k), r["epoch_state"].get(k)) for k in keys}
    rep["state_all_equal"] = all(a == b for a, b in rep["state_equal"].values())
    rep["lr"] = (c["optimizer_state_dict"]["param_groups"][0]["lr"], r["optimizer_state_dict"]["param_groups"][0]["lr"])
    rep["lr_equal"] = rep["lr"][0] == rep["lr"][1]
    # Integer counters (num_bad_epochs, last_epoch, cooldown) must match exactly; the float
    # `best` follows the VAL score, which carries GPU run-to-run noise (see the control run).
    rep["scheduler_state_equal"] = _close(c["plateau_scheduler_state_dict"], r["plateau_scheduler_state_dict"])
    rep["scheduler_state_exactly_equal"] = c["plateau_scheduler_state_dict"] == r["plateau_scheduler_state_dict"]
    rep["sampler_state_equal"] = c["scientific_sampler_state"] == r["scientific_sampler_state"]
    rep["eval_history_epochs"] = ([e["epoch"] for e in c["eval_history"]], [e["epoch"] for e in r["eval_history"]])
    rep["val_scores_max_abs_diff"] = max(
        abs(a["val_mean_condition_rmse_K"] - b["val_mean_condition_rmse_K"])
        for a, b in zip(c["eval_history"], r["eval_history"]))
    rep["curve_loss_max_abs_diff"] = max(abs(a["loss"] - b["loss"]) for a, b in zip(c["curves"], r["curves"]))
    rep["curves_len"] = (len(c["curves"]), len(r["curves"]))
    ct, rt = _trace(args.continuous / "stream_trace.jsonl"), _trace(args.resumed / "stream_trace.jsonl")
    by = {x["epoch_index_0based"]: x["indices_sha256"] for x in ct}
    rep["stream_epochs_continuous"] = [x["epoch_index_0based"] for x in ct]
    rep["stream_epochs_resumed_attempts"] = [x["epoch_index_0based"] for x in rt]
    rep["stream_resumed_matches_continuous"] = all(by.get(x["epoch_index_0based"]) == x["indices_sha256"] for x in rt)
    rep["resume_identity"] = {k: r["extra"]["run_identity"].get(k) for k in
                              ("resumed_from", "resume_checkpoint_epoch", "resume_checkpoint_update", "resume_mask_epoch")}
    ok = (rep["state_all_equal"] and rep["lr_equal"] and rep["scheduler_state_equal"]
          and rep["sampler_state_equal"] and rep["stream_resumed_matches_continuous"]
          and rep["curves_len"][0] == rep["curves_len"][1])
    rep["PASS_state_and_stream"] = bool(ok)
    args.out.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    print(json.dumps(rep, indent=2, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
