"""F2.9 supplementary qualification (760 train_fit frames, velocity only): SENS-TAIL and the 5x5 substitution."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, t as st_t
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1")); sys.path.insert(0, str(HERE))
import common as C, common2 as C2
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

out = C.REPO / "results/piv/phase2/f29"; (out / "figures").mkdir(parents=True, exist_ok=True)
sp = C.REPO / "configs/observations/piv_observation_spec_v1_1.yaml"
sha = sp.with_suffix(".sha256").read_text().split()[0]; spec = P.load_piv_spec(sp, expected_sha256=sha)
scale = load_velocity_scale(C.DEFAULT_DATA_ROOT / C.STATS_REL, expected_sha256=C.STATS_SHA256_EXPECTED)
g = np.load(C.DEFAULT_DATA_ROOT / C.CACHE_A38 / "geometry.npz")
gen = P.PivObservationGenerator(spec, scale, x_vec_m=g["x"], y_vec_m=g["y"])
d = C.load_role("train_fit", cache=C.CACHE_A38)
VI, VJ, SIMS, FR = d["vi"].astype(np.float64), d["vj"].astype(np.float64), d["sim_ids"], d["frame_idx"]
steps = P._steps(spec.steps, "primary")
conds = list(dict.fromkeys(SIMS)); t0 = time.time()
du_r, du_c = np.gradient(VI, steps[0], steps[1], axis=(1, 2)); dv_r, dv_c = np.gradient(VJ, steps[0], steps[1], axis=(1, 2))
G = np.sqrt(du_r**2 + du_c**2 + dv_r**2 + dv_c**2); UM = np.hypot(VI, VJ)
HG = np.zeros(VI.shape, bool); JET = np.zeros(VI.shape, bool)
for c in conds:
    m = SIMS == c; HG[m] = G[m] >= np.percentile(G[m], 90); JET[m] = UM[m] >= np.percentile(UM[m], 90)
REG = {"all": np.ones(VI.shape, bool), "HG": HG, "JET": JET, "REST": ~(HG | JET)}
CF = ["PIV-A", "PIV-B-in", "SENS-TAIL-A", "SENS-TAIL-B-in", "SENS-REJ-iso-sub-A", "SENS-REJ-iso-sub5-A", "SENS-REJ-clu-sub-A", "SENS-REJ-clu-sub5-A", "SENS-REJ-iso-flag-A", "SENS-REJ-clu-flag-A"]
Z = {c: [] for c in ("PIV-A", "PIV-B-in", "SENS-TAIL-A", "SENS-TAIL-B-in")}
AC = {}; SUB = {}; PASS = {"iso-sub": [], "iso-sub5": [], "clu-sub": [], "clu-sub5": []}; RANKS = []
for k in range(VI.shape[0]):
    c, fr = SIMS[k], int(FR[k]); sig = 0.10 * spec.vref(c) / 6.0
    res = {cf: gen.observe(cf, VI[k], VJ[k], sim_id=c, frame_index=fr, role="train", replica=0, return_intermediates=True) for cf in CF}
    for cf in Z:
        it = res[cf].intermediates; v = it["valid"]
        for comp in ("vi", "vj"):
            z = it[f"err_{comp}"] / sig; Z[cf].append(z[v][::5])
            for axis, ax_ in (("rows", 0), ("cols", 1)):
                for lag in (1, 2, 3, 4):
                    z1, z2, v1, v2 = (z[:-lag], z[lag:], v[:-lag], v[lag:]) if ax_ == 0 else (z[:, :-lag], z[:, lag:], v[:, :-lag], v[:, lag:])
                    mm = v1 & v2; e = AC.setdefault((cf, axis, lag), [0.0, 0.0])
                    e[0] += float((z1[mm] * z2[mm]).sum()); e[1] += float(0.5 * ((z1[mm]**2).sum() + (z2[mm]**2).sum()))
    for rej in ("iso", "clu"):
        un = res["PIV-A"].intermediates
        for rule, cf in (("sub", f"SENS-REJ-{rej}-sub-A"), ("sub5", f"SENS-REJ-{rej}-sub5-A")):
            it = res[cf].intermediates; rm = res[f"SENS-REJ-{rej}-flag-A"].intermediates["rejected"]
            assert np.array_equal(it["rejected"], rm)
            PASS[f"{rej}-{rule}"].append(res[cf].meta["sub_passes"])
            for comp in ("vi", "vj"):
                for rn, rmask in REG.items():
                    m = rm & rmask[k]
                    if m.any():
                        x = it[f"vals_{comp}"][m] - un[f"vals_{comp}"][m]
                        e = SUB.setdefault((rej, rule, comp, rn, c), [0.0, 0]); e[0] += float((x**2).sum()); e[1] += int(x.size)
    if k % 150 == 0: print(k, f"{time.time()-t0:.0f}s", flush=True)
rows = []
for cf, lst in Z.items():
    z = np.concatenate(lst)
    rows.append({"config": cf, "n": z.size, "mean": z.mean(), "std": z.std(), "skew": skew(z), "excess_kurtosis": kurtosis(z), "frac_abs_gt_3": (np.abs(z) > 3).mean(),
                 "frac_abs_gt_4": (np.abs(z) > 4).mean(), "gauss_frac_abs_gt_3": 2 * norm.sf(3), "t7_scaled_frac_abs_gt_3": 2 * st_t.sf(3 * np.sqrt(7 / 5), 7)})
dist = pd.DataFrame(rows); dist.to_csv(out / "tail_distribution.csv", index=False)
acr = pd.DataFrame([{"config": k[0], "axis": k[1], "lag_nodes": k[2], "rho_pearson": v[0] / v[1]} for k, v in AC.items()]); acr.to_csv(out / "tail_autocorrelation.csv", index=False)
sub_rows = []
for (rej, rule, comp, rn, c), (s2, n) in SUB.items():
    sub_rows.append({"rejection": rej, "rule": rule, "component": comp, "region": rn, "condition": c, "sum_sq": s2, "n": n, "sigma_u": 0.10 * spec.vref(c) / 6})
sub = pd.DataFrame(sub_rows); sub.to_csv(out / "sub_rule_rms_by_condition.csv", index=False)
agg = []
for (rej, rule, comp, rn), q in sub.groupby(["rejection", "rule", "component", "region"]):
    agg.append({"rejection": rej, "rule": rule, "component": comp, "region": rn, "rms_m_s": float(np.sqrt(q.sum_sq.sum() / q.n.sum())), "rms_over_sigma_u": float(np.sqrt((q.sum_sq / q.sigma_u**2).sum() / q.n.sum()))})
agg = pd.DataFrame(agg); agg.to_csv(out / "sub_rule_rms_pooled.csv", index=False)
ps = {k: {"mean": float(np.mean(v)), "max": int(np.max(v)), "frac_gt3": float(np.mean(np.array(v) > 3))} for k, v in PASS.items()}
json.dump(ps, open(out / "substitution_passes.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(dist.round(4).to_string()); print(agg.pivot_table(index=["rejection", "region", "component"], columns="rule", values="rms_over_sigma_u").round(3)); print(ps)
BL, OR, GR, VE = "#0072B2", "#E69F00", "#009E73", "#D55E00"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.7), constrained_layout=True)
bins = np.linspace(-6, 6, 121); cen = 0.5 * (bins[1:] + bins[:-1])
for cf, col in (("PIV-A", BL), ("SENS-TAIL-A", VE)):
    h, _ = np.histogram(np.concatenate(Z[cf]), bins=bins, density=True); axs[0].semilogy(cen, h, color=col, label=cf)
axs[0].semilogy(cen, norm.pdf(cen), "k--", lw=0.8, label="N(0,1)"); nu = 7; axs[0].semilogy(cen, st_t.pdf(cen * np.sqrt(nu / (nu - 2)), nu) * np.sqrt(nu / (nu - 2)), color=GR, lw=0.8, ls=":", label="Student t7, unit variance")
axs[0].set_ylim(1e-6, 1); axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_xlabel("error / sigma_u"); axs[0].set_title("marginal distribution (A, 760 frames)", fontsize=8.5, loc="left")
for ax, axis in ((axs[1], "rows"), (axs[2], "cols")):
    for cf, col, mk in (("PIV-A", BL, "o"), ("SENS-TAIL-A", VE, "s"), ("PIV-B-in", GR, "o"), ("SENS-TAIL-B-in", OR, "s")):
        q = acr[(acr.config == cf) & (acr.axis == axis)]; ax.plot(q.lag_nodes, q.rho_pearson, marker=mk, ms=4, color=col, lw=0.8, label=cf)
    ax.axhline(0, color="k", lw=0.4); ax.set_xlabel(f"lag along {axis} (nodes)"); ax.set_title(f"Pearson autocorrelation, {axis}", fontsize=8.5, loc="left")
axs[1].legend(frameon=False, fontsize=7)
fig.suptitle("Fig. T — SENS-TAIL against the principal error: same sigma_u and rank order, heavier tails, slightly lower Pearson correlation", fontsize=9.5, x=0.005, ha="left")
fig.savefig(out / "figures/figT_sens_tail.png"); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
for ax, rej in ((axs[0], "iso"), (axs[1], "clu")):
    labs = []
    for j, rn in enumerate(("all", "HG", "REST")):
        for comp, off in (("vi", -0.3), ("vj", 0.3)):
            for rule, col, o2 in (("sub", BL, -0.1), ("sub5", VE, 0.1)):
                v = agg[(agg.rejection == rej) & (agg.rule == rule) & (agg.component == comp) & (agg.region == rn)].rms_over_sigma_u.iloc[0]
                ax.bar(j * 2 + (off + o2) * 1.0, v, width=0.18, color=col, label=f"{rule}" if (j == 0 and comp == "vi") else None)
    ax.set_xticks([0, 2, 4]); ax.set_xticklabels(["all", "HG", "REST"]); ax.set_ylabel("RMS(substituted - unrejected) / sigma_u"); ax.set_title(f"REJ-{rej}: blue 3x3, red 5x5 (left bar vi, right vj of each pair group)", fontsize=8, loc="left")
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Fig. S5 — Effect of the 3x3 and 5x5 substitution rules at the rejected vectors (760 frames)", fontsize=9.5, x=0.005, ha="left")
fig.savefig(out / "figures/figS5_sub5_vs_sub3.png"); plt.close(fig)
C2.dump_provenance(out / "f29_provenance.json", {"script": "f29_qualify_v11.py", "spec_v11_sha256": sha, "runtime_s": time.time() - t0})
print("done")
