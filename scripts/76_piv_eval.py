#!/usr/bin/env python3
"""GPU evaluation wrapper of ONE checkpoint (best|last) of a PIV-campaign run (mission 02).

Loads the run's own checkpoint (identity, epoch and weights hash verified), builds the evaluation observations with the frozen
spec v1.1 in the `eval` namespace (same generator, same normalisation, same historical network coordinates as in training),
replays the training-time selection metric as gate (selection namespace, 3 VAL replicas, <= 0.05 K against eval_history),
and delegates the resumable evaluation loop to ``piv_eval.run_resumable_checkpoint_eval`` (units skipped on restart).
`last` with weights identical to `best` writes SAME_AS_BEST.json and performs NO second inference. TEST roles are refused.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.nonideal_observations import file_sha256, load_velocity_scale  # noqa: E402
from combustion_recon.data.piv_observations import PivObservationGenerator, load_piv_spec  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import denormalize_temperature  # noqa: E402
from combustion_recon.evaluation.piv_eval import evaluate_piv_unit, run_resumable_checkpoint_eval  # noqa: E402
from combustion_recon.evaluation.piv_manifest import atomic_write_json  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402
from combustion_recon.training.sparse_dataset import SparseUNetCacheDataset  # noqa: E402

GATE_TOL_K = 0.05
ROLES = ("seen_train_eval", "val_eval")
TIER1 = ["PIV-A", "PIV-B-in", "PIV-B-down", "PIV-B-off", "IDEAL-full", "IDEAL-A", "IDEAL-B-in", "IDEAL-B-down", "IDEAL-B-off",
         "FILT-A", "FILT-B-in", "FILT-B-down", "FILT-B-off", "SENS-RHO0-A", "SENS-RHO0-B-in"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] piv_eval {msg}", flush=True)


def weights_sha(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(str(k).encode()); h.update(np.ascontiguousarray(sd[k].detach().contiguous().cpu().numpy()).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/piv_campaign_matrix.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--kind", choices=("best", "last"), required=True)
    ap.add_argument("--train-root", type=Path, default=None, help="default: matrix out_root")
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--eval-set", choices=("tier1", "full"), default="full")
    ap.add_argument("--replicas", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0, help="smoke only: first N frames per role (output dir must contain 'smoke')")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None, help="override the evaluation set (tests)")
    ap.add_argument("--gate-tol", type=float, default=GATE_TOL_K)
    args = ap.parse_args()
    import torch

    if args.max_frames and "smoke" not in str(args.eval_root):
        raise SystemExit("--max-frames needs an eval-root containing 'smoke'")
    matrix = json.loads(args.matrix.read_text())
    row = next(r for r in matrix["new_runs"] if r["run_id"] == args.run_id)
    family, shared = row["family"], matrix["shared_inputs"]
    train_dir = (args.train_root or Path(matrix["out_root"])) / args.run_id
    s6 = _load("s6eval", "scripts/17_s6_sparse_eval_and_swap.py")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

    # ---- identity of the checkpoint -------------------------------------------------------
    ck_path = train_dir / "checkpoints" / f"{args.kind}.pt"
    out_kind = args.eval_root / args.run_id / args.kind
    payload = torch.load(ck_path, map_location="cpu", weights_only=False)
    ident = (payload.get("extra") or {}).get("run_identity") or {}
    if ident.get("run_id") != args.run_id or ident.get("family") != family:
        raise SystemExit("checkpoint identity (run/family) does not match the matrix")
    cfgs_raw = payload.get("epoch_config") or {}
    if int(cfgs_raw.get("seed", -1)) != int(row["seed"]):
        raise SystemExit("checkpoint seed does not match the matrix")
    epoch = int((payload.get("epoch_state") or {}).get("epoch", -1))
    wsha = weights_sha(payload["model_state_dict"])
    file_sha = file_sha256(ck_path)
    spec_path = ROOT / matrix["observation_spec"]
    if file_sha256(spec_path) != matrix["observation_spec_sha256"]:
        raise SystemExit("observation spec hash differs from the matrix")
    pb = (ident.get("piv_campaign") or {})
    if pb.get("spec_sha256") != matrix["observation_spec_sha256"]:
        raise SystemExit("run was trained with a different observation spec")
    log(f"run={args.run_id} kind={args.kind} epoch={epoch} weights={wsha[:12]} file={file_sha[:12]}")
    if args.kind == "last":
        best = torch.load(train_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
        if weights_sha(best["model_state_dict"]) == wsha:
            out_kind.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_kind / "SAME_AS_BEST.json", {"run_id": args.run_id, "weights_sha256": wsha, "epoch": epoch, "no_inference_repeated": True})
            log("last == best (same weights): SAME_AS_BEST.json, no inference"); return 0
        del best

    # ---- data, normalisation, generator ----------------------------------------------------
    normalizer = Month2FitOnlyNormalizer(ROOT / shared["stats_relpath"])
    scale = load_velocity_scale(ROOT / shared["stats_relpath"], expected_sha256=shared["stats_sha256"])
    cache_root = Path(shared["cache_root"])
    geo = np.load(cache_root / "geometry.npz")
    spec = load_piv_spec(spec_path, expected_sha256=matrix["observation_spec_sha256"])
    gen = PivObservationGenerator(spec, scale, x_vec_m=geo["x"], y_vec_m=geo["y"])
    p_cond = True
    frames_by_role, ds_by_role = {}, {}
    for role in ROLES:
        ds = SparseUNetCacheDataset(role, cache_root=cache_root, mask_mode="evaluation", site_fraction=0.05, p_conditioning=p_cond, normalizer=normalizer,
                                    require_eval_mask_payload=False, eval_mask_payload_root=shared["mask_root"],
                                    **__import__("combustion_recon.training.dense_cache", fromlist=["x"]).cache_validation_kwargs("C149"))
        ds_by_role[role] = ds
        fr = []
        idxs = ds.indices[: args.max_frames] if args.max_frames else ds.indices
        for idx in idxs:
            meta = ds.frames[idx]; vi, vj = ds._phys_velocities(idx)
            fr.append({"sim_id": meta["sim_id"], "condition": meta["sim_id"], "split": meta.get("split", role), "frame_index": int(meta["frame_index"]),
                       "vi_phys": vi, "vj_phys": vj, "true_T_K": np.asarray(ds.t_K[idx], dtype=np.float64), "p_norm": np.asarray(ds.p_norm[idx], dtype=np.float32)})
        frames_by_role[role] = fr
    ds0 = ds_by_role[ROLES[0]]
    model, _ = s6._load_model(family, cache_root, device)
    s6.load_sparse_scientific_checkpoint(ck_path, model=model, require_resumable=False, restore_rng=False)
    model.to(device).eval()
    query = make_synthetic_query_grid(ds0.x, ds0.y).to(device) if family == "adapted_glu" else None
    bs = 1 if family == "adapted_glu" else 8

    def forward_fn(packets):
        outs = []
        for s in range(0, len(packets), bs):
            chunk = packets[s:s + bs]
            if family == "sparse_unet":
                x = np.stack([p["piv_obs"].obs.to_unet_grid(x_scaled=ds0.x_scaled, y_scaled=ds0.y_scaled, nh3_norm=float(p["frame"]["p_norm"][0]),
                                                           phi_norm=float(p["frame"]["p_norm"][1]), p_conditioning=p_cond) for p in chunk])
                batch = {"input_tensor": torch.from_numpy(x.astype(np.float32))}
            else:
                toks = [p["piv_obs"].obs.to_glu_tokens() for p in chunk]
                if len({t["site_index"].size for t in toks}) != 1:
                    raise SystemExit("ragged GLU batch")
                batch = {"glu_tokens": toks, "p_norm_tensor": torch.tensor(np.stack([p["frame"]["p_norm"] for p in chunk]), dtype=torch.float32)}
            pred_n = s6._forward(model, family, batch, p_conditioning=p_cond, query_coords_phys=query, device=device)
            outs.append(np.stack([denormalize_temperature(pred_n[j, 0], normalizer) for j in range(len(chunk))]))
        return np.concatenate(outs, axis=0)

    sigma = float(normalizer.temperature_std)
    run_tags = {"run_id": args.run_id, "checkpoint_kind": args.kind, "checkpoint_epoch": epoch, "checkpoint_file_sha256": file_sha,
                "checkpoint_weights_sha256": wsha, "family": family, "variant": "p_on", "model_seed": int(row["seed"]), "group": row["group"],
                "train_cell": row["cell"], "train_config_id": row["observation"], "eval_git_sha": args.eval_git_sha, "spec_sha256": spec.sha256}

    # ---- replay gate: training selection metric (selection namespace, 3 VAL replicas) ----------
    hist = json.loads((train_dir / "eval_history.json").read_text())
    ent = [h for h in (hist if isinstance(hist, list) else hist.get("history", [])) if int(h.get("epoch", -1)) == epoch]
    gate = {"tolerance_K": args.gate_tol, "epoch": epoch, "all_pass": True, "note": ""}
    if row["observation"] != "HIST-5PCT" and ent and "val_mean_condition_rmse_K" in ent[0]:
        vals = []
        for rep in (0, 1, 2):
            res = evaluate_piv_unit(frames=frames_by_role["val_eval"], generator=gen, eval_config=row["observation"], role="val_eval", replica=rep,
                                    forward_fn=forward_fn, train_sigma_T=sigma, run_tags=run_tags, batch_size=bs, obs_role="selection")
            df = pd.DataFrame(res["rows"]); vals.append(float(df.groupby("sim_id")["rmse_K"].mean().mean()))
        ref = float(ent[0]["val_mean_condition_rmse_K"]); m = float(np.mean(vals))
        gate.update({"replayed_val_macro_K_per_replica": vals, "replayed_mean_K": m, "training_loop_val_K": ref, "delta_K": m - ref, "all_pass": abs(m - ref) <= args.gate_tol})
    else:
        gate["note"] = "no training-loop metric at this epoch (or bridge cell): replay gate not applicable; identity and weights verified"
        gate["applicable"] = False
    out_kind.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_kind / "gate_replay.json", gate)
    log(f"gate {json.dumps(gate)}")
    if not gate["all_pass"]:
        atomic_write_json(out_kind / "GATE_FAILED.json", gate); return 3

    configs = args.configs or (TIER1 + [c for c in spec.raw["configs"] if c not in TIER1] if args.eval_set == "full" else TIER1)
    out = run_resumable_checkpoint_eval(out_kind_dir=out_kind, frames_by_role=frames_by_role, generator=gen, eval_configs=configs, roles=ROLES,
                                        replicas=args.replicas, forward_fn=forward_fn, train_sigma_T=sigma, run_tags=run_tags, batch_size=bs,
                                        resume=not args.no_resume, save_fields=True, gate_result=gate)
    log(f"done {out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
