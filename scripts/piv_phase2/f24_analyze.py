"""F2.4 analysis - residuals MS (OpenPIV) - HDR (reference) of the Neal PIV-UQ unsteady core, against the noise model.

The raw residual contains (i) PIV random error, (ii) systematic single-pass bias, (iii) a large-scale component caused by the
reference being a 5-pair sliding average while the MS pair is instantaneous in a pulsating core.  Only (i) is comparable with the
model.  We therefore report the autocorrelation after three treatments: V0 global mean removed; V1 per-pair mean removed;
V2 per-pair plane removed, always after a 5-MAD outlier mask.  Statistics are reported for all three: the conclusion on the
short-range correlation must not depend on a single choice.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
import common2 as C2
out = C.REPO / "results/piv/phase2/f24"
FIG = out / "figures"; FIG.mkdir(exist_ok=True)
st = np.load(os.path.expanduser("~/piv_qualification_data/neal_unsteady_core/residual_stacks.npz"))
stats = json.load(open(out / "neal_residual_statistics.json"))
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})
LAGS = list(range(1, 7))


def clean(R):
    x = R[np.isfinite(R)]; med = np.median(x); mad = 1.4826 * np.median(np.abs(x - med))
    return np.where(np.abs(R - med) <= 5 * mad, R, np.nan)


def remove(R, mode):
    R = R.copy()
    if mode == "V0":
        return R - np.nanmean(R)
    n, ny, nx = R.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    for i in range(n):
        r = R[i]; ok = np.isfinite(r)
        if ok.sum() < 30: R[i] = np.nan; continue
        if mode == "V1":
            R[i] = r - np.nanmean(r)
        else:
            A = np.stack([np.ones(ok.sum()), xx[ok], yy[ok]], 1)
            coef = np.linalg.lstsq(A, r[ok], rcond=None)[0]
            R[i] = r - (coef[0] + coef[1] * xx + coef[2] * yy)
    return R


def rho(R, axis, lags):
    o = []
    for m in lags:
        r1, r2 = (R[:, :, :-m], R[:, :, m:]) if axis == "x" else (R[:, :-m, :], R[:, m:, :])
        ok = np.isfinite(r1) & np.isfinite(r2)
        o.append(float((r1[ok] * r2[ok]).sum() / (0.5 * ((r1[ok] ** 2).sum() + (r2[ok] ** 2).sum()))))
    return np.array(o)


# model: continuous noise kernel (MHG b=0.35, L_sr = 1.2 W)  ->  autocorrelation as function of lag/W
b = 0.35
sig = 1.2 / (np.sqrt(4 * np.pi) * (1 - b) ** 2 / (1 - b + 3 * b**2 / 4))   # in units of W
t = np.linspace(-8, 8, 16001); dt = t[1] - t[0]
gk = (1 - b * t**2 / sig**2) * np.exp(-t**2 / (2 * sig**2)) / ((1 - b) * sig * np.sqrt(2 * np.pi))
ac_full = np.correlate(gk, gk, mode="full") * dt
lag_axis = (np.arange(ac_full.size) - (gk.size - 1)) * dt
ac_full = ac_full / ac_full.max()
def model_rho(s_over_W): return np.interp(s_over_W, lag_axis, ac_full)
model_L = float(ac_full.sum() * dt)

rows = []
curves = {}
for ds in ("F001", "F005"):
    for cfg, e in stats[ds]["configs"].items():
        W, ov, d = e["W_px"], e["overlap"], e["step_px"]
        for comp in ("u", "v"):
            R0 = clean(st[f"{ds}|W{W}|ov{int(ov*100)}|R{comp.upper()}"].astype(np.float64))
            for mode in ("V0", "V1", "V2"):
                R = remove(R0, mode)
                sd = float(np.nanstd(R))
                for axis in ("x", "y"):
                    rh = rho(R, axis, LAGS)
                    mod = model_rho(np.array(LAGS) * d / W)
                    # integral length from the autocorrelation up to the first non-positive lag (>=1)
                    cut = next((i for i, v in enumerate(rh) if v <= 0), len(rh))
                    L_int = d * (1 + 2 * rh[:cut].sum())
                    rows.append({"dataset": ds, "W_px": W, "overlap": ov, "step_px": d, "component": comp, "removal": mode, "axis": axis, "std_px": sd,
                                 **{f"rho_lag{m}": rh[m - 1] for m in LAGS}, **{f"model_rho_lag{m}": mod[m - 1] for m in LAGS},
                                 "rho_at_lag_W_(2d_if_50pct)": float(np.interp(W, [m * d for m in LAGS], rh)), "model_rho_at_lag_W": float(model_rho(1.0)),
                                 "L_int_px": L_int, "L_int_over_W": L_int / W, "model_L_over_W": 1.2})
                    curves[(ds, cfg, comp, mode, axis)] = rh
df = pd.DataFrame(rows); df.to_csv(out / "neal_correlation_analysis.csv", index=False)
json.dump({"model_noise_kernel_L_over_W": model_L, "model_rho_at_W": float(model_rho(1.0)), "model_rho_at_0.5W": float(model_rho(0.5)),
           "reference_values": {"WIE17_rho_2d_50pct_representative": 0.11, "WIE17_rho_2d_75pct_representative": 0.45, "review_rho_2h_le_50pct": 0.0}},
          open(out / "model_reference.json", "w"), indent=1)

# summary printed
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
sm = df[(df.component == "u")][["dataset", "W_px", "overlap", "removal", "axis", "std_px", "rho_lag1", "rho_lag2", "rho_lag3", "model_rho_lag1", "model_rho_lag2", "model_rho_lag3", "L_int_over_W"]]
print(sm.round(3).to_string())

# figure
fig, axs = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True, sharey=True)
for j, (W, ov) in enumerate(((16, 0.5), (16, 0.75), (32, 0.5), (32, 0.75))):
    for i, axis in enumerate(("x", "y")):
        ax = axs[i, j]; d = int(W - round(W * ov)); lagW = np.array(LAGS) * d / W
        for ds, mk in (("F001", "o"), ("F005", "s")):
            for mode, col, al in (("V0", GREY, 0.5), ("V1", BLUE, 1), ("V2", GREEN, 1)):
                ax.plot(lagW, curves[(ds, f"W{W}_ov{int(ov*100)}", "u", mode, axis)], marker=mk, ms=3.5, color=col, alpha=al, lw=0.8, label=f"{ds} {mode}" if j == 0 and i == 0 else None)
        xs = np.linspace(0, lagW.max(), 200); ax.plot(xs, model_rho(xs), color=VERM, lw=1.6, label="model (g, L = 1.2 W)" if j == 0 and i == 0 else None)
        ax.axhline(0, color="k", lw=0.4); ax.set_title(f"u, {axis}-direction, W = {W} px, overlap {int(ov*100)} %", fontsize=8.5, loc="left")
        ax.set_xlabel("lag / W");
    axs[0, j].set_ylim(-0.3, 1)
axs[0, 0].legend(fontsize=6.5, frameon=False, ncol=2); axs[0, 0].set_ylabel("residual autocorrelation"); axs[1, 0].set_ylabel("residual autocorrelation")
fig.suptitle("Fig. N1 — Neal (non-reacting jet): autocorrelation of MS - HDR residuals vs the model noise kernel. V0/V1/V2 = global mean / per-pair mean / per-pair plane removed", fontsize=9.5, x=0.005, ha="left")
fig.savefig(FIG / "figN1_neal_autocorrelation.png"); plt.close(fig)

# residual histogram + std table
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True)
for cfg, col in (("W16_ov50", BLUE), ("W16_ov75", GREEN), ("W32_ov50", ORANGE), ("W32_ov75", VERM)):
    R = remove(clean(st[f"F001|W{cfg.split('_')[0][1:]}|ov{cfg.split('ov')[1]}|RU".replace('W' + cfg.split('_')[0][1:], 'W' + cfg.split('_')[0][1:])].astype(np.float64)), "V1")
    x = R[np.isfinite(R)]
    h, e = np.histogram(x, bins=np.linspace(-0.8, 0.8, 81), density=True); axs[0].semilogy(0.5 * (e[1:] + e[:-1]), h, color=col, label=cfg)
axs[0].set_xlabel("u residual, per-pair mean removed (px)"); axs[0].set_ylabel("density"); axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_ylim(1e-3, 10)
axs[0].set_title("F001 residuals (outliers > 5 MAD masked)", fontsize=8.5, loc="left")
tab = df[(df.component == "u") & (df.axis == "x") & (df.removal == "V1")]
for ds, mk in (("F001", "o"), ("F005", "s")):
    q = tab[tab.dataset == ds]; axs[1].plot(range(4), q.std_px.values, marker=mk, color=BLUE if ds == "F001" else ORANGE, label=ds)
axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["16 px\n50 %", "16 px\n75 %", "32 px\n50 %", "32 px\n75 %"]); axs[1].set_ylabel("std of residual (px)")
axs[1].axhline(0.06, color=GREY, ls=":"); axs[1].text(0.05, 0.063, "WIE17 measured 0.060 px (16 px, 75 %)", fontsize=7, color=GREY)
axs[1].axhline(0.146, color=GREY, ls=":"); axs[1].text(0.05, 0.149, "WIE17 synthetic DaVis 0.146 px", fontsize=7, color=GREY); axs[1].legend(frameon=False, fontsize=7.5)
axs[1].set_title("residual std (per-pair mean removed) vs literature values", fontsize=8.5, loc="left")
fig.savefig(FIG / "figN2_neal_residual_distribution.png"); plt.close(fig)
print("done")
