#!/usr/bin/env python3
"""Phase 3 (mission 02): ONE frozen original checkpoint (the 12 models of mission 00, best) evaluated on PIV observations.

Reuses the identity/dataset/model path of the Block-II evaluation (scripts/60_nonideal_eval.py) and the resumable PIV loop of
piv_eval. Order: (1) replay gate = E0 (ideal values at the historical sites, through the NEW generator path) must reproduce the
S7/00 reference macro within 0.05 K for seen+VAL x 3 replicas, else GATE_FAILED and nothing else is evaluated; (2) E0, E0v, E1, E2,
E3 (IDEAL-full), E3a (IDEAL-A), E4 (FILT-A), E5 (PIV-A), E6 (IDEAL-B-*), E7 (PIV-B-*), the MANDATORY SENS-RHO0 (A, B-in) and the other
v1.1 sensitivities. Weights are never changed; TEST is refused; temperature never enters an observation.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from combustion_recon.data.month2_normalizer import Month2FitOnlyNormalizer  # noqa: E402
from combustion_recon.data.nonideal_observations import file_sha256, load_degradation_spec, load_velocity_scale  # noqa: E402
from combustion_recon.data.piv_observations import load_piv_spec  # noqa: E402
from combustion_recon.evaluation import nonideal_eval as ne  # noqa: E402
from combustion_recon.evaluation.month2_evaluator import denormalize_temperature  # noqa: E402
from combustion_recon.evaluation.piv_eval import evaluate_piv_unit, run_resumable_checkpoint_eval  # noqa: E402
from combustion_recon.evaluation.piv_manifest import atomic_write_json  # noqa: E402
from combustion_recon.evaluation.piv_phase3 import HistoricalSiteGenerator, PSEUDO, with_pseudo_configs  # noqa: E402
from combustion_recon.models.glu_reconstructor import make_synthetic_query_grid  # noqa: E402

PIV_SPEC = ROOT / "configs/observations/piv_observation_spec_v1_1.yaml"
MANDATORY = ["E0", "E0v", "E1", "E2", "IDEAL-full", "IDEAL-A", "FILT-A", "PIV-A", "IDEAL-B-in", "IDEAL-B-down", "IDEAL-B-off", "FILT-B-in", "FILT-B-down", "FILT-B-off",
             "PIV-B-in", "PIV-B-down", "PIV-B-off", "SENS-RHO0-A", "SENS-RHO0-B-in"]


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, ROOT / rel); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def log(m): print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] piv_phase3 {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=ROOT / "configs/nonideal/degradation_spec_v1_12models.yaml")
    ap.add_argument("--matrix", type=Path, default=ROOT / "configs/experiments/month2_s7_c149_eval_matrix.json")
    ap.add_argument("--s7-reference", type=Path, default=ROOT / "results/month2/s7_c149_comparison/package/tables/global_by_replica.csv")
    ap.add_argument("--run-id", required=True); ap.add_argument("--out-root", type=Path, required=True); ap.add_argument("--eval-git-sha", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--mode", choices=("full", "smoke"), default="full")
    ap.add_argument("--smoke-frames", type=int, default=8); ap.add_argument("--eval-set", choices=("mandatory", "full"), default="full")
    ap.add_argument("--configs", nargs="+", default=None)
    a = ap.parse_args()
    import torch
    if (a.mode != "full") != ("smoke" in a.out_root.name.lower()):
        raise SystemExit("smoke needs an out-root named '*smoke*'; full must not")
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    n60 = _load("n60", "scripts/60_nonideal_eval.py"); s7e = _load("s7e_p3", "scripts/38_s7_eval.py"); s6 = s7e._s6()
    spec12 = load_degradation_spec(a.spec); srun = n60._spec_entry(spec12, a.run_id); family, p_cond = srun["family"], bool(srun["p_conditioning"])
    matrix = s7e.load_matrix(a.matrix); entry = s7e._run_entry(matrix, a.run_id); n60._check_entry_matches_spec(entry, srun)
    sh = matrix["shared_inputs"]
    scale = load_velocity_scale(sh["stats"], expected_sha256=sh["stats_sha256"])
    consumed = s7e._verify_inputs(entry, matrix); inspected = n60._inspect_best(s7e, entry, torch)
    normalizer = Month2FitOnlyNormalizer(sh["stats"])
    smoke_frames = a.smoke_frames if a.mode == "smoke" else None
    datasets, expected = n60._build_datasets(s7e, entry, matrix, consumed, normalizer, smoke_frames, ne.EVAL_ROLES)
    ds0 = datasets["seen_train_eval"]
    model, _ = s6._load_model(family, consumed["cache_root"], dev)
    s6.load_sparse_scientific_checkpoint(inspected["path"], model=model, require_resumable=True, restore_rng=False)
    model.to(dev).eval()
    query = make_synthetic_query_grid(ds0.x, ds0.y).to(dev) if family == "adapted_glu" else None
    pspec = with_pseudo_configs(load_piv_spec(PIV_SPEC))
    role_by_sim = {ds.frames[i]["sim_id"]: r for r, ds in datasets.items() for i in ds.indices}
    geo = np.load(Path(consumed["cache_root"]) / "geometry.npz")
    gen = HistoricalSiteGenerator(pspec, scale, x_vec_m=geo["x"], y_vec_m=geo["y"], mask_root=sh["mask_root"], role_by_sim=role_by_sim)
    frames_by_role = {}
    for r, ds in datasets.items():
        fr = []
        for idx in ds.indices:
            m = ds.frames[idx]; vi, vj = ds._phys_velocities(idx)
            fr.append({"sim_id": m["sim_id"], "condition": m["sim_id"], "split": m.get("split", r), "frame_index": int(m["frame_index"]), "vi_phys": vi, "vj_phys": vj,
                       "true_T_K": np.asarray(ds.t_K[idx], dtype=np.float64), "p_norm": np.asarray(ds.p_norm[idx], dtype=np.float32)})
        frames_by_role[r] = fr
    bs = 1 if family == "adapted_glu" else 8

    def fwd(pk):
        outs = []
        for s in range(0, len(pk), bs):
            ch = pk[s:s + bs]
            if family == "sparse_unet":
                x = np.stack([p["piv_obs"].obs.to_unet_grid(x_scaled=ds0.x_scaled, y_scaled=ds0.y_scaled, nh3_norm=float(p["frame"]["p_norm"][0]),
                                                           phi_norm=float(p["frame"]["p_norm"][1]), p_conditioning=p_cond) for p in ch])
                batch = {"input_tensor": torch.from_numpy(x.astype(np.float32))}
            else:
                tk = [p["piv_obs"].obs.to_glu_tokens() for p in ch]
                batch = {"glu_tokens": tk, "p_norm_tensor": torch.tensor(np.stack([p["frame"]["p_norm"] for p in ch]), dtype=torch.float32)}
            pn = s6._forward(model, family, batch, p_conditioning=p_cond, query_coords_phys=query, device=dev)
            outs.append(np.stack([denormalize_temperature(pn[j, 0], normalizer) for j in range(len(ch))]))
        return np.concatenate(outs, 0)

    sigma = float(normalizer.temperature_std)
    tags = {"run_id": a.run_id, "checkpoint_kind": "best", "checkpoint_epoch": int(srun["best_epoch"]), "checkpoint_file_sha256": srun["best_file_sha256"],
            "checkpoint_weights_sha256": srun["best_weights_sha256"], "family": family, "variant": "p_on" if p_cond else "p_off", "model_seed": int(srun["model_seed"]),
            "group": "P3", "train_cell": "HIST-5PCT-ideal", "train_config_id": "E0", "eval_git_sha": a.eval_git_sha, "spec_sha256": pspec.sha256}
    out_kind = a.out_root / a.run_id / "best"; out_kind.mkdir(parents=True, exist_ok=True)
    # ---- (1) replay gate through the new generator path ----------------------------------------------
    rows = []
    for role in ne.EVAL_ROLES:
        for rep in ne.REPLICAS:
            res = evaluate_piv_unit(frames=frames_by_role[role], generator=gen, eval_config="E0", role=role, replica=rep, forward_fn=fwd, train_sigma_T=sigma, run_tags=tags, batch_size=bs)
            df = pd.DataFrame(res["rows"]); df["condition"] = df["sim_id"]; df["replica"] = rep; df["role"] = role; rows.append(df)
    gate = ne.gate_against_reference(pd.concat(rows), pd.read_csv(a.s7_reference), a.run_id)
    if a.mode == "smoke":
        gate["note"] = "smoke: subset of frames, gate not meaningful"; gate["all_pass"] = True
    atomic_write_json(out_kind / "gate_replay_E0.json", gate)
    log(f"E0 replay gate all_pass={gate['all_pass']} max_abs_delta={gate.get('max_abs_delta_K')}")
    if not gate["all_pass"]:
        atomic_write_json(out_kind / "GATE_FAILED.json", {"run_id": a.run_id, "gate": gate}); return 3
    cfgs = a.configs or (MANDATORY if a.eval_set == "mandatory" else MANDATORY + [c for c in pspec.raw["configs"] if c not in MANDATORY and c not in PSEUDO])
    out = run_resumable_checkpoint_eval(out_kind_dir=out_kind, frames_by_role=frames_by_role, generator=gen, eval_configs=cfgs, roles=ne.EVAL_ROLES, replicas=ne.REPLICAS,
                                        forward_fn=fwd, train_sigma_T=sigma, run_tags=tags, batch_size=bs, resume=True, save_fields=True, gate_result=gate)
    log(f"done {out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
