#!/usr/bin/env python3
"""Check a wall-time-cap continuation chain from the ledger and run directory (smoke)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--real-requeue", default="0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run = a.out_root / a.run_id
    ledger = [json.loads(x) for x in (a.out_root / "_ledger" / f"{a.run_id}.jsonl").read_text().splitlines()]
    events = [(e["event"], e.get("action"), e.get("restart_count"), e.get("rc")) for e in ledger]
    rep: dict = {"non_scientific_smoke": True, "events": events}
    starts = [e for e in ledger if e["event"] == "attempt_start"]
    ends = [e for e in ledger if e["event"] == "attempt_end"]
    rep["first_action"] = starts[0]["action"] if starts else None
    rep["second_action"] = starts[1]["action"] if len(starts) > 1 else None
    first = run / "summary_before_resume_1.json"
    final = run / "summary.json"
    s1 = json.loads(first.read_text()) if first.is_file() else None
    s2 = json.loads(final.read_text()) if final.is_file() else None
    rep["summary_attempt1"] = {k: (s1 or {}).get(k) for k in ("status", "epochs_completed", "updates")}
    rep["summary_final"] = {k: (s2 or {}).get(k) for k in ("status", "epochs_completed", "updates")}
    ident = json.loads((run / "run_identity.json").read_text())
    rep["resume_identity"] = {k: ident.get(k) for k in ("resumed_from", "resume_checkpoint_epoch", "resume_checkpoint_status")}
    rep["requeue_event_present"] = any(e["event"] == "continuation_requeue" for e in ledger)
    trace = [json.loads(x) for x in (run / "stream_trace.jsonl").read_text().splitlines()]
    epochs = [t["epoch_index_0based"] for t in trace]
    rep["stream_epoch_sequence_monotone_after_dedup"] = sorted(set(epochs)) == list(range(max(epochs) + 1))
    ok = (
        rep["first_action"] == "fresh" and rep["second_action"] == "resume"
        and s1 is not None and s1["status"] == "stopped_wall_time"
        and s2 is not None and s2["epochs_completed"] > s1["epochs_completed"]
        and rep["resume_identity"]["resumed_from"].endswith("checkpoints/last.pt")
        and rep["stream_epoch_sequence_monotone_after_dedup"]
        and (rep["requeue_event_present"] or a.real_requeue != "1")
    )
    rep["PASS"] = bool(ok)
    a.out.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
