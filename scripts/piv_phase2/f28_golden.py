"""F2.8/F2.9 helper: golden packages of the observation generator.

--write FILE    : compute, for a list of configs x 3 TRAIN conditions x 2 frames x replicas {0,1}, the obs hash and the
                  values; save hashes (JSON) and, with --packages, the values (NPZ) of the cross-machine subset.
--compare FILE  : recompute and compare with a previous FILE (hash equality; max |value difference| when --packages-ref given).
Works on the historical A38 cache (frames 400 and 1000 of the three figure conditions). Velocity only.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default=str(C.REPO / "configs/observations/piv_observation_spec_v1.yaml"))
ap.add_argument("--data-root", default=str(C.DEFAULT_DATA_ROOT))
ap.add_argument("--cache", default=C.CACHE_A38)
ap.add_argument("--write"); ap.add_argument("--compare"); ap.add_argument("--packages"); ap.add_argument("--packages-ref")
ap.add_argument("--configs", default="all", help="'all', 'main4' (PIV-A, PIV-B-in, IDEAL-A, FILT-A) or comma list")
ap.add_argument("--only-v1-configs-from")
a = ap.parse_args()
dr = Path(a.data_root)
sha = Path(a.spec).with_suffix(".sha256").read_text().split()[0]
spec = P.load_piv_spec(a.spec, expected_sha256=sha)
scale = load_velocity_scale(dr / C.STATS_REL, expected_sha256=C.STATS_SHA256_EXPECTED)
g = np.load(dr / a.cache / "geometry.npz")
gen = P.PivObservationGenerator(spec, scale, x_vec_m=g["x"], y_vec_m=g["y"])
import json as _j
frames = _j.load(open(dr / a.cache / "train_fit/frames.json"))
stats = _j.load(open(dr / C.STATS_REL))["fields"]
vi_all = np.load(dr / a.cache / "train_fit/vi_norm.npy", mmap_mode="r"); vj_all = np.load(dr / a.cache / "train_fit/vj_norm.npy", mmap_mode="r")
conds = ["80NH3_1.2.h5", "40NH3_1.1.h5", "0NH3_0.85.h5"]; FR = (400, 1000)
if a.configs == "all":
    cfgs = list(spec.raw["configs"].keys())
elif a.configs == "main4":
    cfgs = ["PIV-A", "PIV-B-in", "IDEAL-A", "FILT-A"]
else:
    cfgs = a.configs.split(",")
if a.only_v1_configs_from:
    ref = json.load(open(a.only_v1_configs_from)); cfgs = [c for c in cfgs if c in ref["configs"]]
out = {"spec_sha256": sha, "configs": {}}; pk = {}
for cfg in cfgs:
    out["configs"][cfg] = {}
    for c in conds:
        for fr in FR:
            k = next(i for i, r in enumerate(frames) if r["sim_id"] == c and r["frame_index"] == fr)
            vi = np.asarray(vi_all[k], np.float64) * stats["Velocity_i"]["std"] + stats["Velocity_i"]["mean"]
            vj = np.asarray(vj_all[k], np.float64) * stats["Velocity_j"]["std"] + stats["Velocity_j"]["mean"]
            for rep in (0, 1):
                o = gen.observe(cfg, vi, vj, sim_id=c, frame_index=fr, role="eval", replica=rep)
                key = f"{c}|{fr}|{rep}"
                out["configs"][cfg][key] = o.obs_hash
                if a.packages and rep == 0:
                    pk[f"{cfg}|{key}|sites"] = o.obs.sites; pk[f"{cfg}|{key}|values"] = o.obs.velocity_values_phys
if a.write:
    json.dump(out, open(a.write, "w"), indent=1)
    print("written", a.write, len(cfgs), "configs")
if a.packages:
    np.savez_compressed(a.packages, **pk); print("packages", a.packages, len(pk) // 2)
if a.compare:
    ref = json.load(open(a.compare)); bad = []
    for cfg in cfgs:
        for key, h in out["configs"][cfg].items():
            if ref["configs"].get(cfg, {}).get(key) != h: bad.append((cfg, key))
    print("compared", sum(len(v) for v in out["configs"].values()), "packages; hash mismatches:", len(bad), bad[:3])
if a.packages_ref:
    ref = np.load(a.packages_ref); mx = 0.0; nbad = 0
    for k in pk:
        if k not in ref.files: continue
        if k.endswith("|sites"):
            nbad += int(not np.array_equal(pk[k], ref[k]))
        else:
            mx = max(mx, float(np.abs(pk[k] - ref[k]).max()))
    print(f"packages vs ref: site-set mismatches {nbad}; max |value diff| = {mx:.3e} m/s")
