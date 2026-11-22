#!/usr/bin/env python3
"""Final evaluation of one temperature-assisted run on the whole K sweep (K = 0/1/4/9/16/32), from the frozen eval matrix.

Roles reference760 / seen220 / VAL123, velocity 5%, mask replicas 0/1/2 (layout r paired with replica r), checkpoints best / last /
best_seen_diagnostic (distinct; identical weights share one forward). Per frame: full-field metrics (the project evaluator, incl. regional),
plus full / common-domain / off-anchor(K) / anchors metrics. Then the loop-parity gate (macro common-domain RMSE within 0.05 K of the
training loop per checkpoint/role/replica, and per K on VAL; a missing reference is a failure) and a read-back of every file.
``COMPLETE.json`` only after both; otherwise ``GATE_FAILED.json`` with diagnostics. Modes: full | smoke (frame subset, never COMPLETE).
"""

from __future__ import annotations

import argparse
import hashlib
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

import torch  # noqa: E402

from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.t_anchors import K_VALUES, AnchorManifest, domain_metrics  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import denormalize_temperature, evaluate_frame_prediction  # noqa: E402
from combustion_recon.evaluation.opus_o5_plan import CHECKPOINTS, ROLE_N_FRAMES, ROLES  # noqa: E402
from combustion_recon.models.glu_assisted import build_assisted_glu_from_config  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.models.unet import build_sparse_unet_assisted_from_config  # noqa: E402
from combustion_recon.training.assisted_dataset import AssistedDataset  # noqa: E402
from combustion_recon.training.assisted_trainer import DIAG_K, AssistedTrainer  # noqa: E402
from combustion_recon.training.dense_cache import cache_validation_kwargs  # noqa: E402
from combustion_recon.training.scientific_epoch_sampler import cache_indices_for_pool  # noqa: E402
from combustion_recon.training.sparse_dataset import collate_sparse_batch  # noqa: E402
from combustion_recon.training.sparse_scientific_loop import SparseScientificConfig, load_sparse_scientific_checkpoint  # noqa: E402

TOL_K = 0.05
REPLICAS = (0, 1, 2)


def _log(m: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] t_eval {m}", flush=True)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _weights_sha(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(str(k).encode())
        h.update(np.ascontiguousarray(sd[k].detach().contiguous().cpu().numpy()).tobytes())
    return h.hexdigest()


def macro_per_K(df: pd.DataFrame) -> dict[int, float]:
    return {int(K): float(g.groupby("sim_id")["common_rmse_K"].mean().mean()) for K, g in df.groupby("K")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--mode", choices=("full", "smoke"), default="full")
    ap.add_argument("--smoke-frames", type=int, default=9)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cpu only for non-scientific smoke tests")
    ap.add_argument("--anchor-manifest", type=Path, default=ROOT / "configs/observations/month2_temperature_anchors_nested_v1.json")
    args = ap.parse_args()
    smoke = args.mode == "smoke"
    if smoke != ("smoke" in args.out_root.name.lower()):
        raise SystemExit("smoke mode needs an out-root named *smoke*; full must not")
    matrix = json.loads(args.matrix.read_text())
    if matrix.get("experiment_id") != "month2_t_assisted_eval_v1":
        raise SystemExit("unexpected matrix")
    entry = [r for r in matrix["runs"] if r["run_id"] == args.run_id]
    if len(entry) != 1:
        raise SystemExit("run not in the matrix")
    entry = entry[0]
    fam, p_on, seed = entry["family"], bool(entry["p_conditioning"]), int(entry["model_seed"])
    anchors = AnchorManifest.load(args.anchor_manifest)
    if anchors.sha256 != matrix["anchor_manifest_sha256"]:
        raise SystemExit("anchor manifest differs from the frozen matrix")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cpu" and not smoke:
        raise SystemExit("a scientific evaluation needs CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = torch.device(args.device)
    sh = matrix["shared_inputs"]
    cache_root = Path(sh["c149_cache_root"])
    _log(f"start {args.run_id} family={fam} p_on={p_on} mode={args.mode}")
    # ---- verify checkpoints against the frozen matrix
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        path = Path(entry["training_dir"]) / "checkpoints" / f"{name}.pt"
        if _sha_file(path) != spec["file_sha256"]:
            raise SystemExit(f"{name}: file sha differs from the frozen matrix (checkpoint modified after the freeze)")
    # ---- inputs (cache payload hashes, masks, stats) via the S7 evaluator helpers
    ev = importlib.util.module_from_spec(importlib.util.spec_from_file_location("s7eval_t", ROOT / "scripts/38_s7_eval.py"))
    ev.__spec__.loader.exec_module(ev)
    consumed = ev._verify_inputs({**entry, "pool_id": "C149", "training_git_shas": entry["training_git_shas"]}, {"shared_inputs": sh})
    normalizer = Month2FitOnlyNormalizer(sh["stats"])
    kw = cache_validation_kwargs("C149")
    ref760 = cache_indices_for_pool(json.loads((cache_root / "train_fit" / "frames.json").read_text()), "A38")

    def mk(role):
        idx = ref760 if role == "train_fit" else None
        if smoke:
            n = len(idx) if idx is not None else len(json.loads((cache_root / role / "frames.json").read_text()))
            base = idx if idx is not None else list(range(n))
            idx = base[:: max(1, len(base) // args.smoke_frames)][: args.smoke_frames]
        return AssistedDataset(role, cache_root=cache_root, indices=idx, mask_mode="evaluation", site_fraction=0.05,
                               eval_mask_seed=int(sh["eval_mask_seed"]), layout_id=str(sh["layout_id"]), replica_id=0, p_conditioning=p_on,
                               normalizer=normalizer, eval_mask_payload_root=sh["mask_root"], require_eval_mask_payload=True,
                               anchors=anchors, **kw)

    datasets = {r: mk(r) for r in ROLES}
    expected = {r: (len(datasets[r]) if smoke else ROLE_N_FRAMES[r]) for r in ROLES}
    for r in ROLES:
        if len(datasets[r]) != expected[r]:
            raise SystemExit(f"{r}: {len(datasets[r])} frames, expected {expected[r]}")
    cfg = SparseScientificConfig(family=fam, p_conditioning=p_on, seed=seed, device=args.device, batch_size=8, decode_chunked_eval=True)
    torch.manual_seed(seed)
    if fam == "sparse_unet":
        model, queries = build_sparse_unet_assisted_from_config(init_seed=seed), None
    else:
        model = build_assisted_glu_from_config(domain_geometry_path=cache_root / "geometry.npz")
        queries = make_synthetic_query_grid(datasets["val_eval"].x, datasets["val_eval"].y)
    model.to(dev)
    trainer = AssistedTrainer(model, cfg=cfg, normalizer=normalizer, n_train_samples=2980, query_coords_phys=queries, anchors=anchors)
    out_run = args.out_root / args.run_id
    out_run.mkdir(parents=True, exist_ok=False)
    tags_base = {"run_id": args.run_id, "family": fam, "p_conditioning": p_on, "model_seed": seed, "pool_id": "C149", "site_fraction": 0.05,
                 "task_mode": "temperature_assisted", "training_git_sha": entry["training_git_shas"][-1], "eval_git_sha": args.eval_git_sha,
                 "matrix_sha256": _sha_file(args.matrix)}
    stored: dict[str, pd.DataFrame] = {}
    fields: dict[str, Any] = {}
    n_files = 0
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        src = spec.get("reuse_forward_from")
        tags = {**tags_base, "checkpoint_kind": name, "epoch": spec["epoch"], "criterion": spec["criterion"],
                "file_sha256": spec["file_sha256"], "weights_sha256": spec["weights_sha256"], "shared_forward_from": src or ""}
        if src:
            df = stored[src].copy()
            for k, v in tags.items():
                df[k] = v
            stored[name] = df
        else:
            payload = load_sparse_scientific_checkpoint(Path(entry["training_dir"]) / "checkpoints" / f"{name}.pt", model=model,
                                                        require_resumable=True, restore_rng=False)
            if int(payload["epoch_state"]["epoch"]) != int(spec["epoch"]) or _weights_sha(model.state_dict()) != spec["weights_sha256"]:
                raise SystemExit(f"{name}: loaded weights differ from the frozen matrix")
            model.to(dev)
            trainer.eval_mode()
            rows = []
            t0 = time.perf_counter()
            with torch.no_grad():
                for role in ROLES:
                    ds = datasets[role]
                    for rep in REPLICAS:
                        ds.replica_id = rep
                        layout32 = anchors.orders[rep]
                        for K in K_VALUES:
                            ds.eval_K = K
                            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_sparse_batch)
                            for batch in loader:
                                if fam == "adapted_glu":
                                    batch["p_norm_tensor"] = torch.tensor([[float(ds.p_norm[int(m["cache_index"]), 0]), float(ds.p_norm[int(m["cache_index"]), 1])]
                                                                            for m in batch["metadata"]], dtype=torch.float32)
                                pred_n = trainer._forward_norm(batch, decode_chunked=True).detach().cpu().numpy()
                                true_K = batch["target_T_K"].numpy()
                                for i, meta in enumerate(batch["metadata"]):
                                    pred_K = denormalize_temperature(pred_n[i, 0], normalizer)
                                    row = evaluate_frame_prediction(pred_K, true_K[i, 0], train_sigma_T=float(normalizer.temperature_std), sim_id=str(meta["sim_id"]),
                                                                    split=str(meta.get("split", role)), role=role, frame_index=int(meta["frame_index"]), method=fam,
                                                                    variant="p_on" if p_on else "p_off", p_conditioning=p_on, K=int(K), site_fraction=0.05, model_seed=seed,
                                                                    mask_replica_id=rep, layout_id=str(sh["layout_id"]), task_mode="temperature_assisted")
                                    row.pop("_spatial_bias_map", None)
                                    row.update(domain_metrics(pred_K, true_K[i, 0], meta["t_sites"], layout32))
                                    row["mask_hash"] = str(meta.get("mask_hash", ""))
                                    row.update(tags)
                                    rows.append(row)
                                    if role == "val_eval" and rep == 0 and int(meta["frame_index"]) == 1000:
                                        fields.setdefault(name, {}).setdefault(int(K), []).append((str(meta["sim_id"]), pred_K.astype(np.float32), true_K[i, 0].astype(np.float32)))
                        _log(f"{name} {role} rep{rep} done ({len(rows)} rows, {time.perf_counter() - t0:.0f}s)")
            stored[name] = pd.DataFrame(rows)
        for role in ROLES:
            for rep in REPLICAS:
                d = stored[name][(stored[name]["role"] == role) & (stored[name]["mask_replica_id"] == rep)]
                p = out_run / "per_frame" / name / role / f"rep{rep}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                d.to_csv(p, index=False)
                n_files += 1
    # fields for figures (best ckpt and shared ones): val frame 1000, replica 0, six K
    (out_run / "fields").mkdir()
    for name in CHECKPOINTS:
        src = entry["checkpoints"][name].get("reuse_forward_from") or name
        f = fields.get(src)
        if not f:
            continue
        sims = sorted({s for s, _, _ in f[0]})
        pred = np.stack([np.stack([next(p for s2, p, _ in f[K] if s2 == s) for s in sims]) for K in K_VALUES])
        true = np.stack([next(t for s2, _, t in f[0] if s2 == s) for s in sims])
        np.savez_compressed(out_run / "fields" / f"val_frame1000_{name}_rep0.npz", pred_T_K=pred, true_T_K=true, sim_id=np.array(sims),
                            K=np.array(K_VALUES), checkpoint_kind=np.array(name), run_id=np.array(args.run_id))
    # ---- gate vs the training loop
    gate_rows = []
    for name in CHECKPOINTS:
        spec = entry["checkpoints"][name]
        ref = spec["loop_reference"]
        df = stored[name]
        for role, key in (("train_fit", "train_mean_condition_rmse_K"), ("seen_train_eval", "seen_mean_condition_rmse_K")):
            sub = df[(df.role == role) & (df.mask_replica_id == 0) & (df.K.isin(DIAG_K))]
            mine = float(np.mean(list(macro_per_K(sub).values()))) if len(sub) else None
            lv = ref.get(key)
            gate_rows.append({"checkpoint": name, "role": role, "replica": "0", "K": "diag{0,9,32}", "final_K": mine, "loop_K": lv,
                              "delta_K": None if (mine is None or lv is None) else mine - lv})
        vr = ref.get("val_per_replica") or []
        for item in vr:
            rep = int(item["replica_id"])
            sub = df[(df.role == "val_eval") & (df.mask_replica_id == rep)]
            pk = macro_per_K(sub)
            lv = item.get("mean_condition_rmse_K")
            mine = float(np.mean(list(pk.values()))) if pk else None
            gate_rows.append({"checkpoint": name, "role": "val_eval", "replica": str(rep), "K": "mean6", "final_K": mine, "loop_K": lv,
                              "delta_K": None if (mine is None or lv is None) else mine - lv})
            for K, v in pk.items():
                lk = (item.get("per_K_common_rmse_K") or {}).get(str(K))
                gate_rows.append({"checkpoint": name, "role": "val_eval", "replica": str(rep), "K": str(K), "final_K": v, "loop_K": lk,
                                  "delta_K": None if lk is None else v - lk})
    g = pd.DataFrame(gate_rows)
    g["within_tolerance"] = g["delta_K"].map(lambda x: bool(x is not None and np.isfinite(x) and abs(x) <= TOL_K))
    g["reference_present"] = g["loop_K"].notna()
    g.to_csv(out_run / "gate_macro.csv", index=False)
    expected_gate = len(CHECKPOINTS) * (2 + 3 * 7)
    gate = {"tolerance_K": TOL_K, "n_expected": expected_gate, "n_comparisons": int(len(g)), "n_within_tolerance": int(g["within_tolerance"].sum()),
            "n_missing_reference": int((~g["reference_present"]).sum()), "max_abs_delta_K": float(g["delta_K"].abs().max()) if g["delta_K"].notna().any() else None,
            "all_pass": bool(len(g) == expected_gate and g["within_tolerance"].all() and g["reference_present"].all()), "mode": args.mode}
    # per-frame diagnostic against the loop CSVs (not a threshold)
    diag = []
    for name in CHECKPOINTS:
        ep = int(entry["checkpoints"][name]["epoch"])
        for role in ("val_eval", "seen_train_eval", "train_fit"):
            f = Path(entry["training_dir"]) / "evals" / f"epoch_{ep:04d}" / f"{role}_per_frame.csv.gz"
            if f.is_file():
                lo = pd.read_csv(f, usecols=["sim_id", "frame_index", "mask_replica_id", "K", "common_rmse_K"])
                m = stored[name][["sim_id", "frame_index", "mask_replica_id", "K", "common_rmse_K", "role"]]
                m = m[m.role == role].merge(lo, on=["sim_id", "frame_index", "mask_replica_id", "K"], suffixes=("", "_loop"))
                diag.append({"checkpoint": name, "role": role, "n_matched": len(m), "max_abs_frame_delta_K": float((m.common_rmse_K - m.common_rmse_K_loop).abs().max()) if len(m) else None})
    pd.DataFrame(diag).to_csv(out_run / "gate_per_frame_diagnostic.csv", index=False)
    (out_run / "gate.json").write_text(json.dumps({"summary": gate}, indent=2) + "\n")
    # ---- read back
    n_csv, issues = 0, []
    for name in CHECKPOINTS:
        for role in ROLES:
            for rep in REPLICAS:
                fr = pd.read_csv(out_run / "per_frame" / name / role / f"rep{rep}.csv", usecols=["rmse_K", "common_rmse_K", "K", "sim_id", "weights_sha256", "mask_hash"])
                n_csv += 1
                if len(fr) != expected[role] * len(K_VALUES):
                    issues.append(f"{name}/{role}/rep{rep}: {len(fr)} rows")
                if not np.isfinite(fr[["rmse_K", "common_rmse_K"]].to_numpy()).all():
                    issues.append(f"{name}/{role}/rep{rep}: non-finite")
                if set(fr["weights_sha256"]) != {entry["checkpoints"][name]["weights_sha256"]} or sorted(fr["K"].unique()) != list(K_VALUES):
                    issues.append(f"{name}/{role}/rep{rep}: identity/K")
    prov = {"run_id": args.run_id, "matrix_sha256": tags_base["matrix_sha256"], "mode": args.mode, "eval_git_sha": args.eval_git_sha, "family": fam,
            "p_conditioning": p_on, "consumed": {k: str(v) for k, v in consumed.items()}, "anchor_manifest_sha256": anchors.sha256, "scientific": not smoke,
            "loop_metrics_are_not_this_output": True, "no_test_split": True, "n_csv": n_csv, "issues": issues,
            "numerics": {"precision": "fp32", "allow_tf32": False}, "max_memory_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0}
    (out_run / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    if smoke:
        (out_run / "SMOKE_OK.json").write_text(json.dumps({"ok": not issues, "gate": gate, "issues": issues}, indent=2) + "\n")
        _log(f"smoke done gate={gate}")
        return 0 if not issues else 3
    if issues or not gate["all_pass"]:
        (out_run / "GATE_FAILED.json").write_text(json.dumps({"ok": False, "gate": gate, "issues": issues}, indent=2) + "\n")
        _log(f"GATE_FAILED gate={gate} issues={issues[:3]}")
        return 3
    (out_run / "COMPLETE.json").write_text(json.dumps({"ok": True, "n_csv": n_csv, "gate": gate, "run_id": args.run_id}, indent=2) + "\n")
    _log(f"done COMPLETE gate max|delta|={gate['max_abs_delta_K']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
