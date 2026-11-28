"""F2.6 figures and derived tables (paired panels, comparable scales, zooms where effects are small). Velocity only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

R = C.REPO / "results/piv/phase2/f26"; FIG = R / "figures"; TAB = R / "tables"
FIG.mkdir(exist_ok=True, parents=True); TAB.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, VERM, GREY, SKY, PURP = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b", "#56B4E9", "#CC79A7"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.grid": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

sha = (C.REPO / "configs/observations/piv_observation_spec_v1.sha256").read_text().split()[0]
spec = P.load_piv_spec(C.REPO / "configs/observations/piv_observation_spec_v1.yaml", expected_sha256=sha)
scale = load_velocity_scale(C.DEFAULT_DATA_ROOT / C.STATS_REL, expected_sha256=C.STATS_SHA256_EXPECTED)
g = np.load(C.DEFAULT_DATA_ROOT / C.CACHE_A38 / "geometry.npz")
gen = P.PivObservationGenerator(spec, scale, x_vec_m=g["x"], y_vec_m=g["y"])
d = C.load_role("train_fit", cache=C.CACHE_A38)
VI, VJ, SIMS, FR = d["vi"].astype(np.float64), d["vj"].astype(np.float64), d["sim_ids"], d["frame_idx"]
steps = P._steps(spec.steps, "primary")
x_mm, y_mm = g["x"] * 1e3, g["y"] * 1e3
EXT = [y_mm[0], y_mm[-1], x_mm[-1], x_mm[0]]  # horizontal = y (cols, axial), vertical = x (rows, lateral)

vref = {c: spec.vref(c) for c in spec.raw["reference_velocity"]["per_condition_m_s"]}
train = [c for c in dict.fromkeys(SIMS)]
order = sorted(train, key=lambda c: vref[c])
CONDS = {"low v_ref": order[0], "median v_ref": order[len(order) // 2 - 1], "high v_ref": order[-1]}
FRAMES = (400, 1000)
json.dump({"rule": "TRAIN conditions with minimum, median-rank and maximum v_ref; frames 400 and 1000 fixed a priori", "conditions": CONDS,
           "v_ref": {k: vref[v] for k, v in CONDS.items()}, "frames": FRAMES}, open(R / "figure_selection.json", "w"), indent=1)

def get(c, fr):
    k = np.where((SIMS == c) & (FR == fr))[0][0]
    return VI[k], VJ[k], k

def obs(cfg, c, fr, vi, vj, **kw):
    return gen.observe(cfg, vi, vj, sim_id=c, frame_index=fr, role="train", replica=0, return_intermediates=True, **kw)

def style(ax, title, ylabel=False, xlabel=False):
    ax.set_title(title, fontsize=8, loc="left")
    ax.tick_params(length=2)
    if not ylabel: ax.set_yticklabels([])
    if not xlabel: ax.set_xticklabels([])

# ------------------------------------------------------------------------- Fig A: paired panels per condition/frame
for label, c in CONDS.items():
    for fr in FRAMES:
        vi, vj, k = get(c, fr)
        sig = 0.10 * vref[c] / 6.0
        pa = obs("PIV-A", c, fr, vi, vj); pi_ = obs("PIV-B-in", c, fr, vi, vj); pd_ = obs("PIV-B-down", c, fr, vi, vj); po = obs("PIV-B-off", c, fr, vi, vj)
        fig, axs = plt.subplots(2, 6, figsize=(15.5, 5.4), constrained_layout=True)
        for row, (comp, ide) in enumerate((("vi", vi), ("vj", vj))):
            lim = np.percentile(np.abs(ide - np.median(ide)), 99.5)
            kw = dict(cmap="RdBu_r", vmin=np.median(ide) - lim, vmax=np.median(ide) + lim, extent=EXT, aspect="auto", interpolation="nearest")
            filt = pa.intermediates[f"filtered_{comp}"]; err = pa.intermediates[f"err_{comp}"]; ob = pa.intermediates[f"vals_{comp}"]
            valid = pa.intermediates["valid"]
            im = axs[row, 0].imshow(ide, **kw); style(axs[row, 0], f"{comp}: LES (ideal)", ylabel=True)
            axs[row, 1].imshow(np.where(valid, filt, np.nan), **kw); style(axs[row, 1], f"{comp}: filtered (MHG, W=1.3 mm)")
            im2 = axs[row, 2].imshow(np.where(valid, filt - ide, np.nan), cmap="PuOr", vmin=-2, vmax=2, extent=EXT, aspect="auto", interpolation="nearest")
            style(axs[row, 2], f"filtered - ideal (zoom, +-2 m/s)")
            im3 = axs[row, 3].imshow(np.where(valid, ob - filt, np.nan), cmap="PuOr", vmin=-3 * sig, vmax=3 * sig, extent=EXT, aspect="auto", interpolation="nearest")
            style(axs[row, 3], f"error (+-3 sigma_u = +-{3*sig:.1f} m/s)")
            axs[row, 4].imshow(np.where(pa.intermediates["observed"], ob, np.nan), **kw); style(axs[row, 4], f"{comp}: observed PIV-A")
            comp_img = np.full((128, 128), np.nan)
            for o_, col in ((pd_, ORANGE), (pi_, BLUE), (po, GREEN)):
                v_ = o_.intermediates["observed"]; comp_img[v_] = o_.intermediates[f"vals_{comp}"][v_]
            axs[row, 5].imshow(comp_img, **kw); style(axs[row, 5], f"{comp}: observed PIV-B (3 windows)")
            for o_, col, nm in ((pi_, BLUE, "B-in"), (pd_, ORANGE, "B-down"), (po, GREEN, "B-off")):
                r0, r1, c0, c1 = o_.meta["field_rect_rows_cols"]
                yy0, yy1 = y_mm[c0] - 0.38, y_mm[c1] + 0.38; xx0, xx1 = x_mm[r0] - 0.29, x_mm[r1] + 0.29
                axs[row, 5].add_patch(Rectangle((yy0, xx0), yy1 - yy0, xx1 - xx0, fill=False, ec=col, lw=1.2))
                if row == 0: axs[row, 5].text(yy0 + 0.5, xx1 + 1.8, nm, color=col, fontsize=7.5)
            plt.colorbar(im, ax=axs[row, 0], fraction=0.05, pad=0.02, label="m/s (scale shared by panels 1, 2, 5, 6)")
            plt.colorbar(im2, ax=axs[row, 2], fraction=0.06, pad=0.01); plt.colorbar(im3, ax=axs[row, 3], fraction=0.06, pad=0.01)
        for ax in axs[1]: ax.set_xlabel("y, axial (mm)", fontsize=7.5)
        for ax in axs[:, 0]: ax.set_ylabel("x, lateral (mm) = rows", fontsize=7.5)
        fig.suptitle(f"Fig. A — {c.replace('.h5','')} ({label}, v_ref = {vref[c]:.1f} m/s, sigma_u = {sig:.2f} m/s), frame {fr}. Same colour scale in the four velocity panels of a row; "
                     f"the other two are zoomed. Grey = no vector. Horizontal axis: y, axial (= columns), inlet at the right; vertical: x, lateral (= rows).", fontsize=9.5, x=0.005, ha="left")
        fig.savefig(FIG / f"figA_{label.split()[0]}_{c.replace('.h5','')}_f{fr}.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig B: zoom on a high-gradient neighbourhood with profiles
c = CONDS["median v_ref"]; fr = 1000
vi, vj, k = get(c, fr)
Gm = np.hypot(*np.gradient(vj, steps[0], steps[1]))
valid_A = P.PivObservationGenerator._valid_mask((0, 127, 0, 127), steps, 1.3)
Gm_ = np.where(valid_A, Gm, 0)
Gm_[:20] = 0; Gm_[-20:] = 0; Gm_[:, :20] = 0; Gm_[:, -20:] = 0   # keep the crop away from the borders: edge behaviour is shown elsewhere
ri, ci = np.unravel_index(np.argmax(Gm_), Gm_.shape)
h = 18
sl = (slice(ri - h, ri + h), slice(ci - h, ci + h))
pa = obs("PIV-A", c, fr, vi, vj); th = obs("SENS-TH-A", c, fr, vi, vj); fl = obs("FILT-A", c, fr, vi, vj)
sig = 0.10 * vref[c] / 6.0
fig = plt.figure(figsize=(14.5, 7.2), constrained_layout=True)
gs = fig.add_gridspec(2, 5)
top_axes, bot_axes = [], []
ext = [y_mm[ci - h], y_mm[ci + h - 1], x_mm[ri + h - 1], x_mm[ri - h]]
rng_ = (vj[sl].min(), vj[sl].max())
for j, (nm, arr) in enumerate((("LES (ideal)", vj), ("filtered MHG (PIV-A response)", pa.intermediates["filtered_vj"]), ("filtered top-hat (SENS-TH)", th.intermediates["filtered_vj"]),
                               ("observed PIV-A (filtered + error)", pa.intermediates["vals_vj"]))):
    ax = fig.add_subplot(gs[0, j]); top_axes.append(ax); im = ax.imshow(arr[sl], cmap="RdBu_r", vmin=rng_[0], vmax=rng_[1], extent=ext, aspect="auto", interpolation="nearest")
    ax.set_title(nm, fontsize=8.5, loc="left"); ax.axvline(y_mm[ci], color="k", lw=0.5, ls=":")
    if j == 0: ax.set_ylabel("x, lateral (mm)")
    ax.set_xlabel("y, axial (mm)")
plt.colorbar(im, ax=top_axes[-1], fraction=0.05, pad=0.02, label="vj (m/s); scale shared by the 4 panels")
for j, (nm, arr, ref) in enumerate((("MHG - ideal", pa.intermediates["filtered_vj"], vj), ("top-hat - MHG", th.intermediates["filtered_vj"], pa.intermediates["filtered_vj"]),
                                    ("observed - MHG (the error)", pa.intermediates["vals_vj"], pa.intermediates["filtered_vj"]))):
    ax = fig.add_subplot(gs[1, j]); bot_axes.append(ax); im = ax.imshow((arr - ref)[sl], cmap="PuOr", vmin=-3 * sig, vmax=3 * sig, extent=ext, aspect="auto", interpolation="nearest")
    ax.set_title(f"{nm} (+-3 sigma_u = +-{3*sig:.1f} m/s)", fontsize=8.5, loc="left"); ax.set_xlabel("y, axial (mm)")
    if j == 0: ax.set_ylabel("x, lateral (mm)")
plt.colorbar(im, ax=bot_axes[-1], fraction=0.05, pad=0.02, label="m/s")
ax = fig.add_subplot(gs[1, 3:5])
rows_ = np.arange(ri - h, ri + h); xx = x_mm[rows_]
ax.plot(xx, vj[rows_, ci], color="k", lw=1.6, label="LES ideal"); ax.plot(xx, pa.intermediates["filtered_vj"][rows_, ci], color=BLUE, lw=1.4, label="MHG")
ax.plot(xx, th.intermediates["filtered_vj"][rows_, ci], color=VERM, lw=1.2, ls="--", label="top-hat"); ax.plot(xx, pa.intermediates["vals_vj"][rows_, ci], color=GREEN, lw=0.9, marker="o", ms=2.5, label="observed")
ax.fill_between(xx, pa.intermediates["filtered_vj"][rows_, ci] - sig, pa.intermediates["filtered_vj"][rows_, ci] + sig, color=BLUE, alpha=0.12, lw=0, label="MHG +- sigma_u")
ax.set_xlabel("x, lateral (mm) along the dotted line"); ax.set_ylabel("vj (m/s)"); ax.legend(fontsize=7, frameon=False, ncol=2)
ax.set_title("profile across the strongest gradient", fontsize=8.5, loc="left")
fig.suptitle(f"Fig. B — Zoom on the steepest-gradient neighbourhood, {c.replace('.h5','')}, frame {fr} (crop of 36 x 36 nodes; the filter effect is small except here)", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figB_zoom_gradient.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig C: masks and coverage
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True)
vi, vj, k = get(CONDS["median v_ref"], 1000)
cfgs = [("PIV-A", "A: W=1.3 mm"), ("PIV-B-in", "B-in"), ("PIV-B-down", "B-down"), ("PIV-B-off", "B-off")]
for ax, (cfg, nm) in zip(axs, cfgs):
    o = obs(cfg, CONDS["median v_ref"], 1000, vi, vj)
    it = o.intermediates; r0, r1, c0, c1 = o.meta["field_rect_rows_cols"]
    img = np.zeros((128, 128)); img[it["valid"]] = 1
    which = "response_mhg"; edge = gen._edge_mask((r0, r1, c0, c1), which, "primary", float(spec.config(cfg)["W_mm"]))
    img[it["valid"] & edge] = 2
    ax.imshow(img, cmap=matplotlib.colors.ListedColormap(["#eeeeee", SKY, ORANGE]), vmin=0, vmax=2, extent=EXT, aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((y_mm[c0] - 0.38, x_mm[r0] - 0.29), y_mm[c1] - y_mm[c0] + 0.76, x_mm[r1] - x_mm[r0] + 0.58, fill=False, ec="k", lw=1))
    ax.set_title(f"{nm}: {o.meta['n_valid_vectors']} vectors ({100*o.meta['n_valid_vectors']/16384:.1f} % of the slice); edge vectors {o.meta['n_edge_vectors']}", fontsize=8, loc="left")
    ax.set_xlabel("y, axial (mm); inlet at the right")
axs[0].set_ylabel("x, lateral (mm)")
fig.suptitle("Fig. C — Valid vectors (blue), edge vectors whose kernel support leaves the field (orange), vector-free area (grey); black = field. "
             "B windows: 52 rows x 40 cols = 30 x 30 mm", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figC_masks_coverage.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig D: error distribution and autocorrelation
zq = json.load(open(R / "error_distribution.json")); ac = pd.read_csv(R / "error_autocorrelation_empirical.csv")
from scipy.stats import norm
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.7), constrained_layout=True)
zs = np.load(R / "z_samples_PIV-A_B-in.npz")
bins = np.linspace(-5, 5, 101); cen = 0.5 * (bins[1:] + bins[:-1])
for key, col, lab in (("PIV_A", BLUE, "PIV-A"), ("PIV_B_in", GREEN, "PIV-B-in")):
    hh, _ = np.histogram(zs[key], bins=bins, density=True); axs[0].semilogy(cen, hh, color=col, lw=1.3, label=lab)
axs[0].semilogy(cen, norm.pdf(cen), color="k", lw=0.9, ls="--", label="N(0,1)"); axs[0].set_ylim(1e-5, 1); axs[0].legend(frameon=False, fontsize=7.5)
axs[0].set_xlabel("error / sigma_u,c"); axs[0].set_ylabel("density")
s = zq["PIV-A"]; axs[0].set_title(f"PIV-A: std {s['std']:.3f}, skew {s['skew']:+.3f}, excess kurtosis {s['excess_kurtosis']:+.3f}", fontsize=8.5, loc="left")
for ax, axis, step in ((axs[1], "rows", steps[0]), (axs[2], "cols", steps[1])):
    for cfg, col, mk, lab in (("PIV-A", BLUE, "o", "PIV-A (W=1.3) measured"), ("PIV-B-in", GREEN, "s", "PIV-B-in (W=1.5) measured"), ("SENS-RHO0-A", GREY, "x", "SENS-RHO0 (independent)")):
        q = ac[(ac.config == cfg) & (ac.axis == axis) & (ac.lag_nodes <= 6)]
        ax.plot(q.lag_mm, q.rho, ls="none", marker=mk, color=col, ms=5, label=lab)
    for W, col in ((1.3, BLUE), (1.5, GREEN)):
        e = spec.kernel("noise", "primary", W, axis)
        kk, w = P.kernel_weights("mhg", e["param_mm"], step)
        a_ = P.autocorr_discrete(w, 6); ax.plot(np.arange(7) * step, a_, color=col, lw=0.9, ls="-", alpha=0.6)
    ax.axhline(0, color="k", lw=0.4); ax.set_xlabel(f"lag along {axis} (mm)"); ax.set_ylabel("error autocorrelation")
    ax.set_title(f"along {axis} (step {step:.3f} mm); lines = analytic discrete autocorrelation", fontsize=8.5, loc="left")
axs[1].legend(frameon=False, fontsize=7.5)
fig.suptitle("Fig. D — The random error: Gaussian by construction (check of the realisation), correlated over ~1.2 W; points = measured on 760 frames", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figD_error_stats.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig E: realised response, interior vs edge (analytic from the operators)
LAMS = [2.0, 4.0, 10.0]
fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.2), constrained_layout=True)
rows_out = []
for col, (axis, step) in enumerate((("rows", steps[0]), ("cols", steps[1]))):
    for fld, n, ls in (("A (128 nodes)", 128, "-"), ("B (40-52 nodes)", 52 if axis == "rows" else 40, "--")):
        W = 1.3 if fld.startswith("A") else 1.5
        e = spec.kernel("response_mhg", "primary", W, axis)
        kk, w = P.kernel_weights("mhg", e["param_mm"], step)
        M = P.axis_matrix(n, kk, w)
        L = step / (M**2).sum(1)
        axs[0, col].plot(np.arange(n), L, ls=ls, label=f"{fld}, W={W}", lw=1.3)
        axs[0, col].axhline(0.93 * W, color=GREY, lw=0.5, ls=":")
        for lam, c_ in zip(LAMS, (BLUE, ORANGE, GREEN)):
            kq = 2 * np.pi / lam
            amp = np.abs((M * np.exp(1j * kq * (np.arange(n)[None, :] - np.arange(n)[:, None]) * step)).sum(1))
            axs[1, col].plot(np.arange(n), amp, ls=ls, color=c_, lw=1.2, label=f"lambda={lam:g} mm" if fld.startswith("A") else None)
            mid = n // 2
            mg = P.window_margin(W, step)
            rows_out.append({"axis": axis, "field": fld, "W_mm": W, "lambda_mm": lam, "H_interior": float(amp[mid]), "H_first_valid_vector": float(amp[mg]), "H_second_valid_vector": float(amp[mg + 1]),
                             "H_excluded_node0": float(amp[0]), "first_valid_node_index": mg})
        mg = P.window_margin(W, step)
        rows_out.append({"axis": axis, "field": fld, "W_mm": W, "lambda_mm": "L_sr", "H_interior": float(L[n // 2]), "H_first_valid_vector": float(L[mg]), "H_second_valid_vector": float(L[mg + 1]),
                         "H_excluded_node0": float(L[0]), "first_valid_node_index": mg})
    axs[0, col].set_title(f"{axis}: realised discrete L_sr per node (dotted = target 0.93 W)", fontsize=8.5, loc="left"); axs[0, col].set_ylabel("L_sr,disc (mm)"); axs[0, col].legend(frameon=False, fontsize=7)
    axs[1, col].set_title(f"{axis}: amplitude of the realised transfer function |H|", fontsize=8.5, loc="left"); axs[1, col].set_xlabel("node index along the field"); axs[1, col].set_ylabel("|H|"); axs[1, col].legend(frameon=False, fontsize=7)
fig.suptitle("Fig. E — Realised response of the discrete kernel: interior vs field edge (truncated and renormalised kernel; solid A, dashed B)", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figE_realised_response.png"); plt.close(fig)
pd.DataFrame(rows_out).to_csv(TAB / "realised_response_interior_vs_edge.csv", index=False)

# ------------------------------------------------------------------------- derived tables (per-condition normalised by sigma_u)
df = pd.read_csv(R / "rms_by_config_condition_region.csv")
sig_c = {c: 0.10 * vref[c] / 6.0 for c in vref}
def pooled_ratio(cfg, kind, comp, region, sigma_scale=1.0):
    q = df[(df.config == cfg) & (df.kind == kind) & (df.component == comp) & (df.region == region)].copy()
    if q.empty: return np.nan, np.nan
    q["su"] = q.condition.map(sig_c) * sigma_scale
    ratio2 = (q.sum_sq / q.su**2).sum() / q.n.sum()
    rms = np.sqrt(q.sum_sq.sum() / q.n.sum())
    return float(rms), float(np.sqrt(ratio2))
rows = []
for cfg, kind in (("FILT-A", "filt_minus_ideal"), ("PIV-A", "error"), ("PIV-A", "obs_minus_ideal"), ("FILT-B-in", "filt_minus_ideal"), ("PIV-B-in", "error"), ("PIV-B-in", "obs_minus_ideal"),
                  ("PIV-B-down", "filt_minus_ideal"), ("PIV-B-down", "error"), ("PIV-B-off", "filt_minus_ideal"), ("PIV-B-off", "error"),
                  ("SENS-TH-A", "filt_minus_ideal"), ("SENS-TH-B-in", "filt_minus_ideal"), ("SENS-SWAP-A", "filt_minus_ideal"), ("SENS-SWAP-B-in", "filt_minus_ideal"),
                  ("SENS-W1.0-A", "filt_minus_ideal"), ("SENS-W1.9-A", "filt_minus_ideal"), ("TH-vs-MHG-A", "filtered_diff"), ("TH-vs-MHG-B-in", "filtered_diff"),
                  ("SWAP-vs-primary-A", "filtered_diff"), ("SWAP-vs-primary-B-in", "filtered_diff")):
    for comp in ("vi", "vj"):
        r = {"config": cfg, "quantity": kind, "component": comp}
        for region in ("all", "HG", "JET", "REST", "interior", "edge"):
            rms, ratio = pooled_ratio(cfg, kind, comp, region)
            r[f"{region} RMS (m/s)"] = rms; r[f"{region} / sigma_u"] = ratio
        rows.append(r)
tab = pd.DataFrame(rows); tab.to_csv(TAB / "rms_pooled_by_region.csv", index=False)
# per-condition sigma check (error)
chk = []
for c in train:
    q = df[(df.config == "PIV-A") & (df.kind == "error") & (df.condition == c) & (df.region == "all")]
    chk.append({"condition": c, "v_ref": vref[c], "sigma_u_target": sig_c[c], "sigma_u_realised_vi": float(np.sqrt(q[q.component == "vi"].sum_sq.sum() / q[q.component == "vi"].n.sum())),
                "sigma_u_realised_vj": float(np.sqrt(q[q.component == "vj"].sum_sq.sum() / q[q.component == "vj"].n.sum()))})
chk = pd.DataFrame(chk); chk["ratio_vi"] = chk.sigma_u_realised_vi / chk.sigma_u_target; chk["ratio_vj"] = chk.sigma_u_realised_vj / chk.sigma_u_target
chk.to_csv(TAB / "sigma_u_target_vs_realised_PIV-A.csv", index=False)

# ------------------------------------------------------------------------- Fig F: distance to the edge
dd = pd.read_csv(R / "distance_to_field_edge.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
for cfg, col in (("PIV-A", BLUE), ("PIV-B-in", GREEN)):
    q = dd[(dd.config == cfg) & (dd.kind == "err2_over_sigma2")]; axs[0].plot(q.dist_nodes, q["mean"], marker="o", ms=3.5, color=col, label=cfg)
    for kind, ls in (("filt2_vi", "-"), ("filt2_vj", "--")):
        q = dd[(dd.config == cfg) & (dd.kind == kind)]; axs[1].plot(q.dist_nodes, np.sqrt(q["mean"]), marker="o", ms=3.5, color=col, ls=ls, label=f"{cfg} {kind[-2:]}")
axs[0].axhline(1, color="k", lw=0.7, ls=":"); axs[0].set_ylim(0.8, 1.2); axs[0].set_xlabel("distance of the vector from the field edge (nodes)"); axs[0].set_ylabel("mean (error / sigma_u)^2")
axs[0].set_title("error variance is stationary up to the edge", fontsize=8.5, loc="left"); axs[0].legend(frameon=False, fontsize=7.5)
axs[1].set_xlabel("distance of the vector from the field edge (nodes)"); axs[1].set_ylabel("RMS(filtered - ideal) (m/s)"); axs[1].set_title("filter effect near the edge (truncated kernel)", fontsize=8.5, loc="left"); axs[1].legend(frameon=False, fontsize=7)
fig.suptitle("Fig. F — Behaviour inside and near the edges of the field (TRAIN, 760 frames, valid vectors only)", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figF_edge_behaviour.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig G: sensitivities
fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8), constrained_layout=True, sharey=True)
items = [("PIV-A (MHG 1.3)", "FILT-A", "filt_minus_ideal"), ("SENS-TH-A", "SENS-TH-A", "filt_minus_ideal"), ("SENS-SWAP-A", "SENS-SWAP-A", "filt_minus_ideal"),
         ("SENS-W1.0-A", "SENS-W1.0-A", "filt_minus_ideal"), ("SENS-W1.9-A", "SENS-W1.9-A", "filt_minus_ideal"), ("PIV-B-in (MHG 1.5)", "FILT-B-in", "filt_minus_ideal"),
         ("SENS-TH-B-in", "SENS-TH-B-in", "filt_minus_ideal"), ("TH - MHG (A)", "TH-vs-MHG-A", "filtered_diff"), ("SWAP - primary (A)", "SWAP-vs-primary-A", "filtered_diff")]
for ax, region, ttl in ((axs[0], "all", "all valid nodes"), (axs[1], "HG", "high-gradient nodes")):
    for j, (lab, cfg, kind) in enumerate(items):
        for comp, off, col in (("vi", -0.18, BLUE), ("vj", 0.18, ORANGE)):
            _, ratio = pooled_ratio(cfg, kind, comp, region)
            ax.barh(j + off, ratio, height=0.34, color=col, label=comp if j == 0 else None)
    ax.axvline(1, color="k", lw=0.8); ax.axvline(0.6, color=GREY, lw=0.6, ls=":"); ax.axvline(1.5, color=GREY, lw=0.6, ls=":")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([i[0] for i in items], fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("RMS effect / sigma_u,c (sigma_px = 0.10 px)"); ax.set_title(ttl, fontsize=8.5, loc="left")
axs[0].legend(frameon=False, fontsize=7.5)
axs[1].text(1.02, 8.7, "1 = main error; dotted: sigma_px 0.06 and 0.15", fontsize=7, color=GREY, ha="left")
fig.suptitle("Fig. G — Size of the deterministic sensitivities (filter and kernel variants) compared with the main random error", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figG_sensitivities.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig H: rejections and substitutions
c = CONDS["median v_ref"]; fr = 400
vi, vj, k = get(c, fr)
fig, axs = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
base = obs("PIV-A", c, fr, vi, vj)
sub_rms = {}
for row, rej in enumerate(("iso", "clu")):
    fl = obs(f"SENS-REJ-{rej}-flag-A", c, fr, vi, vj); sb = obs(f"SENS-REJ-{rej}-sub-A", c, fr, vi, vj)
    rm = fl.intermediates["rejected"]
    ys_, xs_ = np.where(rm); cy, cx = 70, 100
    h = 22; sl = (slice(cy - h, cy + h), slice(cx - h, cx + h)); e_ = [y_mm[cx - h], y_mm[cx + h - 1], x_mm[cy + h - 1], x_mm[cy - h]]
    ide = base.intermediates["vals_vj"][sl]; lim = (np.nanmin(ide), np.nanmax(ide))
    kw = dict(cmap="RdBu_r", vmin=lim[0], vmax=lim[1], extent=e_, aspect="auto", interpolation="nearest")
    axs[row, 0].imshow(ide, **kw); axs[row, 0].set_title("observed PIV-A (no rejection)", fontsize=8.5, loc="left")
    axs[row, 1].imshow(np.where(fl.intermediates["observed"], fl.intermediates["vals_vj"], np.nan)[sl], **kw); axs[row, 1].set_title(f"REJ-{rej} flag: rejected = missing (white)", fontsize=8.5, loc="left")
    axs[row, 2].imshow(sb.intermediates["vals_vj"][sl], **kw)
    rr, cc = np.where(rm[sl]); axs[row, 2].plot(y_mm[cx - h + cc], x_mm[cy - h + rr], ls="none", marker="s", ms=2.2, mfc="none", mec="k", mew=0.4)
    axs[row, 2].set_title(f"REJ-{rej} sub: substituted values (boxed) look measured", fontsize=8.5, loc="left")
    dif = (sb.intermediates["vals_vj"] - base.intermediates["vals_vj"])[sl]
    sig = 0.10 * vref[c] / 6
    im = axs[row, 3].imshow(dif, cmap="PuOr", vmin=-3 * sig, vmax=3 * sig, extent=e_, aspect="auto", interpolation="nearest")
    axs[row, 3].set_title(f"substituted - unrejected (+-3 sigma_u)", fontsize=8.5, loc="left")
    for ax in axs[row]: ax.set_xlabel("y, axial (mm)", fontsize=7.5)
    axs[row, 0].set_ylabel("x, lateral (mm)", fontsize=7.5)
plt.colorbar(im, ax=axs[:, 3], fraction=0.03, pad=0.01)
fig.suptitle(f"Fig. H — SENS-REJ (10 % of the vectors; NOT calibrated, NOT transferred), {c.replace('.h5','')} frame {fr}, crop 44 x 44 nodes. Top: isolated; bottom: clusters", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figH_rejections.png"); plt.close(fig)
rm_ = json.load(open(R / "rejection_morphology.json"))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3), constrained_layout=True)
for rej, col in (("iso", BLUE), ("clu", VERM)):
    hist = np.array(rm_[rej]["size_hist"][1:], dtype=float); axs[0].semilogy(np.arange(1, len(hist) + 1), hist / hist.sum(), color=col, marker="o", ms=3, label=f"REJ-{rej}: mean {rm_[rej]['size_mean']:.2f}, p90 {rm_[rej]['size_p90']:.0f}")
axs[0].set_xlabel("cluster size (8-connected nodes)"); axs[0].set_ylabel("share of clusters"); axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_title("cluster sizes of the rejected vectors (60+ pooled)", fontsize=8.5, loc="left")
subt = df[(df.kind == "diff") & (df.config.str.startswith("REJ-"))]
vals = []
for rej in ("iso", "clu"):
    for region in ("all", "HG", "REST"):
        q = subt[(subt.config == f"REJ-{rej}-sub-vs-unrejected") & (subt.region == region)]
        _, r1 = pooled_ratio(f"REJ-{rej}-sub-vs-unrejected", "diff", "vi", region); _, r2 = pooled_ratio(f"REJ-{rej}-sub-vs-unrejected", "diff", "vj", region)
        vals.append((f"{rej}\n{region}", r1, r2))
x = np.arange(len(vals)); axs[1].bar(x - 0.18, [v[1] for v in vals], 0.34, color=BLUE, label="vi"); axs[1].bar(x + 0.18, [v[2] for v in vals], 0.34, color=ORANGE, label="vj")
axs[1].set_xticks(x); axs[1].set_xticklabels([v[0] for v in vals], fontsize=7.5); axs[1].set_ylabel("RMS(substituted - unrejected) / sigma_u"); axs[1].legend(frameon=False, fontsize=7.5)
axs[1].set_title("what the substitution rule changes at the rejected vectors", fontsize=8.5, loc="left")
fig.savefig(FIG / "figH2_rejection_stats.png"); plt.close(fig)

# ------------------------------------------------------------------------- Fig I: B windows on the speed map
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
c = CONDS["median v_ref"]
mean_sp = np.hypot(VI[SIMS == c], VJ[SIMS == c]).mean(0)
im = axs[0].imshow(mean_sp, cmap="viridis", extent=EXT, aspect="auto"); plt.colorbar(im, ax=axs[0], label="mean |u| over the 38 frames (m/s)")
for f_, col in (("B-in", BLUE), ("B-down", ORANGE), ("B-off", GREEN)):
    r0, r1, c0, c1 = P.field_rect(f_, steps)
    axs[0].add_patch(Rectangle((y_mm[c0] - 0.38, x_mm[r0] - 0.29), y_mm[c1] - y_mm[c0] + 0.76, x_mm[r1] - x_mm[r0] + 0.58, fill=False, ec=col, lw=1.6)); axs[0].text(y_mm[c0] + 0.6, x_mm[r0] - 1.2, f_, color=col, fontsize=8)
axs[0].set_xlabel("y, axial (mm); inlet at the right"); axs[0].set_ylabel("x, lateral (mm)"); axs[0].set_title(f"B windows on the mean speed, {c.replace('.h5','')}", fontsize=8.5, loc="left")
bw = json.load(open(C.REPO / "results/piv/phase2/f20/b_windows_content.json"))
names = ["slice_full", "B-in|primary", "B-down|primary", "B-off|primary"]
vals = {n: bw[n] for n in names}
xx = np.arange(len(names))
axs[1].bar(xx - 0.2, [100 * vals[n]["frac_u_gt_20"] for n in names], 0.38, color=BLUE, label="% of nodes with |u| > 20 m/s")
axs[1].bar(xx + 0.2, [vals[n]["mean_speed"] for n in names], 0.38, color=ORANGE, label="mean |u| (m/s)")
axs[1].set_xticks(xx); axs[1].set_xticklabels(["slice", "B-in", "B-down", "B-off"]); axs[1].legend(frameon=False, fontsize=7.5)
axs[1].set_title("velocity content of the three windows (TRAIN, C149 pool; velocity only)", fontsize=8.5, loc="left")
fig.suptitle("Fig. I — Position and content of the three PIV-B windows", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figI_B_windows.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
