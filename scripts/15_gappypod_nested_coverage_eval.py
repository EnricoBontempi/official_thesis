#!/usr/bin/env python3
"""Evaluate frozen-λ GappyPOD free at nested 1%/20% layouts.

Requires pre-fitted POD artifacts. Does NOT retune lambda or silently refit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/month2_sparse_gappypod_nested.yaml",
    )
    p.add_argument(
        "--pod-artifacts",
        type=Path,
        required=False,
        help="Path to torch/npz archive with pod_p_off and pod_p_on",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_id": cfg.get("experiment_id"),
        "retune_lambda": bool(cfg.get("retune_lambda")),
        "refit_pod_basis_authorized": bool(cfg.get("refit_pod_basis_authorized")),
        "frozen_from_5pct": cfg.get("frozen_from_5pct"),
        "status": "blocked_missing_pod_artifacts",
    }

    if args.pod_artifacts is None or not Path(args.pod_artifacts).exists():
        report["error"] = (
            "Fitted GappyPOD artifacts missing. Nested 1%/20% eval not executed. "
            "Frozen lambdas recorded; no silent refit."
        )
        report["prerequisite"] = cfg.get("prerequisites", {})
        (out / "blocked_missing_pod_artifacts.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2))
        return 2

    report["status"] = "pod_artifacts_present_but_eval_not_implemented_in_this_turn"
    report["note"] = (
        "Artifacts path provided; full nested eval runner is prepared to consume "
        "frozen lambdas only after Enrico confirms artifact provenance."
    )
    report["pod_artifacts"] = str(args.pod_artifacts)
    (out / "pending_artifact_confirmation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
