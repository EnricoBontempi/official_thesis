#!/usr/bin/env python3
"""Small, separate diagnostic of batch dependence of the best checkpoints (GLU p_on, GLU p_off, U-Net control).

NOT a scientific result: it changes no weights, no CSV and no model. Frames: the few TRAIN/VAL frames of the
smoke subset (6 + 9), 5%, mask replica 0, identical inputs.

* T0  repeat: the same batch twice (numerical reference of the run itself).
* A   fixed batch shape B=8, frame x at position 0, only the other 7 examples change:
      self copies / real frames / the same companions reversed / extreme companions (velocity x10, p +3).
      Any difference at fixed shape is a functional dependence on other examples (or shape-fixed numerics).
* B   shape: B in {1,2,3,6,8} with x repeated (no other examples), plus B in {2,3,6,8} with real companions;
      compared with the B=8 self reference. A difference that appears only when B changes, and is zero at fixed B,
      is numerical sensitivity of the batched kernels, not dependence on the other examples.
* C   torch.cdist with direct distances (compute_mode donot_use_mm) vs default, B=1 and B=8, and the top-k neighbour
      sets (distance, site index) recorded for every decode chunk, with the k-th/(k+1)-th distance gap.
Reported: prediction differences in K, frame RMSE differences, top-k set changes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.models import glu_reconstructor as glu_mod  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402

RUNS = ("S7-C149-glu-on-s0", "S7-C149-glu-off-s0", "S7-C149-unet-on-s1")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TopkRecorder:
    """Wraps the lexicographic top-k selection to record the chosen sets and the k/(k+1) distance gap."""

    def __init__(self):
        self.sets: list[np.ndarray] = []
        self.gaps: list[np.ndarray] = []
        self._orig = glu_mod.topk_distance_then_site_index

    def __enter__(self):
        rec = self

        def wrapped(d_scaled, site_index, k):
            out = rec._orig(d_scaled, site_index, k)
            if d_scaled.shape[2] > k:
                v = torch.sort(d_scaled, dim=2).values[..., k - 1 : k + 1]
                rec.gaps.append((v[..., 1] - v[..., 0]).detach().cpu().numpy())
            rec.sets.append(out.detach().cpu().numpy())
            return out

        glu_mod.topk_distance_then_site_index = wrapped
        return self

    def __exit__(self, *exc):
        glu_mod.topk_distance_then_site_index = self._orig

    def collect(self):
        sets = np.concatenate(self.sets, axis=1) if self.sets else None  # (B, P_total, k)
        gaps = np.concatenate(self.gaps, axis=1) if self.gaps else None
        self.sets, self.gaps = [], []
        return sets, gaps


class DirectCdist:
    def __enter__(self):
        self._orig = torch.cdist
        orig = self._orig
        torch.cdist = lambda a, b, p=2.0, compute_mode="donot_use_mm_for_euclid_dist": orig(a, b, p=p, compute_mode="donot_use_mm_for_euclid_dist")
        return self

    def __exit__(self, *exc):
        torch.cdist = self._orig


def main() -> int:
    global torch
    import torch  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--smoke-frames", type=int, default=6)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    ev = _load("s7eval_for_diag", "scripts/38_s7_eval.py")
    s6 = ev._s6()
    matrix = ev.load_matrix(args.matrix)
    normalizer = Month2FitOnlyNormalizer(matrix["shared_inputs"]["stats"])
    out = args.out_root
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"eval_git_sha": args.eval_git_sha, "runs": {}}
    for run_id in args.runs.split(","):
        entry = ev._run_entry(matrix, run_id)
        family, p_on = entry["family"], bool(entry["p_conditioning"])
        spec = entry["checkpoints"]["best"]
        inspected = ev._inspect_checkpoints(entry, torch)
        consumed = ev._verify_inputs(entry, matrix)
        datasets, expected = ev._build_datasets(entry, matrix, consumed, normalizer, smoke_frames=int(args.smoke_frames))
        model, _ = s6._load_model(family, consumed["cache_root"], device)
        query = None
        if family == "adapted_glu":
            query = make_synthetic_query_grid(datasets["train_fit"].x, datasets["train_fit"].y).to(device)
        s6.load_sparse_scientific_checkpoint(inspected["best"]["path"], model=model, require_resumable=True, restore_rng=False)
        model.to(device).eval()
        pool = []
        for role in ("train_fit", "val_eval"):
            ds = datasets[role]
            ds.replica_id, ds.site_fraction, ds.mask_mode = 0, 0.05, "evaluation"
            for j in range(len(ds)):
                pool.append((role, ds, ds[j]))
        meta["runs"][run_id] = {"family": family, "p_on": p_on, "best_epoch": spec["epoch"], "weights_sha256": spec["weights_sha256"],
                                "n_frames": len(pool), "frames_per_role": expected}

        def make_batch(items: list[dict[str, Any]], ds, mutate: set[int] | None = None):
            batch = s6.collate_sparse_batch(items)
            if family == "adapted_glu":
                s6._attach_p(batch, ds)
            if mutate:
                for i in mutate:
                    if family == "adapted_glu":
                        t = batch["glu_tokens"][i]
                        t["values_norm"] = np.asarray(t["values_norm"]) * 10.0
                        batch["p_norm_tensor"][i] = batch["p_norm_tensor"][i] + 3.0
                    else:
                        batch["input_tensor"][i] = batch["input_tensor"][i] * 10.0
            return batch

        def predict(batch, *, direct: bool = False):
            rec = TopkRecorder()
            cm = DirectCdist() if direct else None
            with torch.no_grad():
                if cm:
                    cm.__enter__()
                try:
                    with rec:
                        pred_n = s6._forward(model, family, batch, p_conditioning=p_on, query_coords_phys=query, device=device)
                finally:
                    if cm:
                        cm.__exit__()
            sets, gaps = rec.collect()
            pred = np.stack([s6.denormalize_temperature(pred_n[i, 0], normalizer) for i in range(pred_n.shape[0])])
            return pred, sets, gaps

        def frame_rmse(pred_i, batch, i):
            true = batch["target_T_K"].numpy()[i, 0]
            return float(np.sqrt(np.mean((pred_i - true) ** 2)))

        def diff_row(test, variant, frame_i, pred_i, ref_i, sets_i, ref_sets_i, gaps_ref, batch, bpos, extra=None):
            d = np.abs(pred_i - ref_i)
            row = {"run_id": run_id, "family": family, "p_on": p_on, "test": test, "variant": variant, "frame": frame_i,
                   "role": pool[frame_i][0], "max_abs_pred_diff_K": float(d.max()), "mean_abs_pred_diff_K": float(d.mean()),
                   "frame_rmse_K": frame_rmse(pred_i, batch, bpos), "frame_rmse_ref_K": frame_rmse(ref_i, batch, bpos)}
            row["frame_rmse_delta_K"] = row["frame_rmse_K"] - row["frame_rmse_ref_K"]
            if sets_i is not None and ref_sets_i is not None:
                changed = np.any(np.sort(sets_i, axis=-1) != np.sort(ref_sets_i, axis=-1), axis=-1)
                row["topk_changed_queries"] = int(changed.sum())
                row["n_queries"] = int(changed.size)
                if gaps_ref is not None:
                    row["min_kth_gap_ref"] = float(gaps_ref.min())
                    row["queries_gap_below_1e-6"] = int((gaps_ref < 1e-6).sum())
                    row["queries_gap_below_1e-4"] = int((gaps_ref < 1e-4).sum())
            if extra:
                row.update(extra)
            rows.append(row)

        t0 = time.perf_counter()
        # T0: repeat
        items8 = [p[2] for p in pool[:8]]
        b = make_batch(items8, pool[0][1])
        a1, s1, _ = predict(b)
        a2, s2, _ = predict(b)
        for i in range(8):
            diff_row("T0_repeat", "same_batch_twice", i, a2[i], a1[i], s2[i] if s2 is not None else None, s1[i] if s1 is not None else None, None, b, i)
        # A and B per frame
        n = len(pool)
        for fi in range(n):
            role, ds, x = pool[fi]
            comp = [pool[(fi + k) % n][2] for k in range(1, 8)]
            comp_ds = ds
            self_b = make_batch([x] * 8, ds)
            ref, ref_sets, ref_gaps = predict(self_b)
            variants = {
                "real": make_batch([x] + comp, ds),
                "real_reversed": make_batch([x] + comp[::-1], ds),
                "extreme": make_batch([x] + comp, ds, mutate=set(range(1, 8))),
            }
            for name, vb in variants.items():
                if family == "adapted_glu" and name != "extreme":
                    pass
                pr, ps, _ = predict(vb)
                diff_row("A_fixedB8_companions", name, fi, pr[0], ref[0], ps[0] if ps is not None else None,
                         ref_sets[0] if ref_sets is not None else None, ref_gaps[0] if ref_gaps is not None else None, self_b, 0)
            for B in (1, 2, 3, 6):
                sb = make_batch([x] * B, ds)
                pr, ps, _ = predict(sb)
                diff_row("B_shape_self", f"B={B}", fi, pr[0], ref[0], ps[0] if ps is not None else None,
                         ref_sets[0] if ref_sets is not None else None, ref_gaps[0] if ref_gaps is not None else None, self_b, 0)
            for B in (2, 3, 6):
                rb = make_batch([x] + comp[: B - 1], ds)
                pr, ps, _ = predict(rb)
                diff_row("B_shape_real", f"B={B}", fi, pr[0], ref[0], ps[0] if ps is not None else None,
                         ref_sets[0] if ref_sets is not None else None, ref_gaps[0] if ref_gaps is not None else None, self_b, 0)
            if family == "adapted_glu":
                for B in (1, 8):
                    db = make_batch([x] * B, ds)
                    pr, ps, _ = predict(db, direct=True)
                    prd, psd, _ = predict(db)
                    diff_row("C_cdist_direct_vs_default", f"B={B}", fi, pr[0], prd[0], ps[0] if ps is not None else None,
                             psd[0] if psd is not None else None, None, db, 0)
        meta["runs"][run_id]["wall_s"] = time.perf_counter() - t0
        del model
        torch.cuda.empty_cache()
        print(f"[{time.strftime('%T')}] done {run_id}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "batch_diag_rows.csv", index=False)
    g = df.groupby(["run_id", "test", "variant"]).agg(
        n=("max_abs_pred_diff_K", "size"), max_abs_pred_diff_K=("max_abs_pred_diff_K", "max"), mean_of_mean_abs_K=("mean_abs_pred_diff_K", "mean"),
        max_abs_frame_rmse_delta_K=("frame_rmse_delta_K", lambda s: float(np.abs(s).max())),
        topk_changed_queries_total=("topk_changed_queries", "sum") if "topk_changed_queries" in df else ("n_queries", "size"),
        min_kth_gap=("min_kth_gap_ref", "min") if "min_kth_gap_ref" in df else ("n_queries", "size")).reset_index()
    g.to_csv(out / "batch_diag_summary.csv", index=False)
    meta["note"] = "diagnostic only; no scientific output was modified"
    (out / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "DIAG_OK.json").write_text(json.dumps({"ok": True, "n_rows": len(df)}) + "\n")
    print(g.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
