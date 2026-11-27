"""Figures for the Phase-1 (R1) report. Reads only the CSV/NPZ/JSON produced by the f1x scripts."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import common as C

R = C.REPO / "results/piv/phase1"; FIG = R / "figures"; FIG.mkdir(exist_ok=True, parents=True)
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 130, "savefig.bbox": "tight"})
g = C.load_geometry(); dx, dy = g["dx"], g["dy"]

# ---------------------------------------------------------------- F1 spectra
sp = np.load(R / "f12/spectra_1d.npz")
conds = sorted({k.split("|")[0] for k in sp.files})
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for ax, axis, D in zip(axs, ("x", "y"), (dx, dy)):
    for comp, col in (("vi", BLUE), ("vj", ORANGE)):
        P = np.array([sp[f"{c}|{comp}|{axis}|P"] for c in conds]); k = sp[f"{conds[0]}|{comp}|{axis}|k"][1:]
        P = P[:, 1:]
        ax.fill_between(k, P.min(0), P.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(k, np.median(P, 0), color=col, lw=1.4)
        ax.text(0.03, 0.30 if comp == "vi" else 0.38, comp, color=col, fontsize=9, transform=ax.transAxes)
    ax.axvline(0.5 / D, color=GREY, lw=0.8, ls="--"); ax.text(0.5 / D * 0.93, 0.25, "Nyquist", rotation=90, ha="right", va="bottom", color=GREY, fontsize=8, transform=ax.get_xaxis_transform())
    for W, ls in ((1.3, ":"), (1.9, "-.")):
        ax.axvline(1 / (2 * W), color=GREY, lw=0.6, ls=ls)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(f"wavenumber along {axis} (cycles/mm)")
    ax.set_title(f"cell {D:.3f} mm (array axis: {'columns' if axis=='x' else 'rows'})", fontsize=9, loc="left")
axs[0].set_ylabel("1D power spectral density (m²/s² per cycle/mm)")
fig.text(0.01, -0.04, "vertical dotted / dash-dot: 1/(2W) for W = 1.3 / 1.9 mm.  Line: median of the 20 TRAIN conditions; band: min–max.", fontsize=7.5, color=GREY)
fig.suptitle("Fig. 1 — 1D spectra of vi, vj (TRAIN, C149 pool, 2980 frames): almost no energy at the grid scale", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig1_spectra.png"); plt.close(fig)

# ---------------------------------------------------------------- F2 structure functions
sf = pd.read_csv(R / "f12/structure_functions.csv")
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
for ax, axis in zip(axs, ("x", "y")):
    for comp, col in (("vi", BLUE), ("vj", ORANGE)):
        s = sf[(sf.axis == axis) & (sf.component == comp)]
        pv = s.pivot(index="lag_mm", columns="condition", values="S2_over_2var")
        ax.fill_between(pv.index, pv.min(1), pv.max(1), color=col, alpha=0.15, lw=0)
        ax.plot(pv.index, pv.median(1), color=col, lw=1.4)
    for W in (1.0, 1.3, 1.5, 1.9):
        ax.axvline(W, color=GREY, lw=0.5, ls=":")
    ax.set_xlim(0, 12); ax.set_xlabel(f"lag along {axis} (mm)"); ax.set_title(f"axis {axis}", fontsize=9, loc="left")
axs[0].set_ylabel("S2(r) / (2 var)"); axs[0].text(8.6, 0.12, "vi", color=BLUE); axs[0].text(8.6, 0.30, "vj", color=ORANGE)
fig.suptitle("Fig. 2 — Normalised structure functions (dotted verticals: W = 1.0, 1.3, 1.5, 1.9 mm)", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig2_structure_functions.png"); plt.close(fig)

# ---------------------------------------------------------------- F3 filter effect vs W
fe = pd.read_csv(R / "f12/filter_effect.csv"); ctx = json.load(open(R / "f12/f12_context.json"))
su = np.array(list(ctx["sigma_u_sigma_px_0.1_dmax_6"].values()))
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
for ax, reg, ttl in zip(axs, ("all", "HG_top10pct_gradient"), ("all valid nodes", "high-gradient nodes (top 10 % of |∇u|)")):
    for kern, col, lab in (("mhg", BLUE, "MHG (calibrated)"), ("tophat", VERM, "top-hat")):
        s = fe[(fe.condition == "POOLED") & (fe.region == reg) & (fe.kernel == kern)]
        for comp, ls in (("vi", "-"), ("vj", "--")):
            q = s[s.component == comp].sort_values("W_mm")
            ax.plot(q.W_mm, q.rms_m_s, color=col, ls=ls, marker="o", ms=4, lw=1.2)
    ax.axhspan(su.min(), su.max(), color=GREEN, alpha=0.15, lw=0); ax.axhline(np.median(su), color=GREEN, lw=1)
    ax.set_xlabel("window W (mm)"); ax.set_title(ttl, fontsize=9, loc="left"); ax.set_xticks([1.0, 1.3, 1.5, 1.9])
axs[0].set_ylabel("RMS |u − u_filtered| (m/s)")
axs[0].text(1.02, np.median(su) * 1.06, "σ_u (σ_px = 0.10 px): range over TRAIN conditions", color=GREEN, fontsize=7.5)
fig.text(0.01, -0.04, "blue = MHG, red = top-hat; solid = vi, dashed = vj. Green: σ_u median (line) and range over the 20 TRAIN conditions.", fontsize=7.5, color=GREY)
fig.suptitle("Fig. 3 — Effect of the spatial filter on the LES velocity, against the proposed random-error amplitude", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig3_filter_effect.png"); plt.close(fig)

# ---------------------------------------------------------------- F4 D-kern
dk = pd.read_csv(R / "f12/dkern_mhg_vs_tophat.csv"); wm = pd.read_csv(R / "f12/dkern_width_matched.csv")
fig, ax = plt.subplots(figsize=(6.6, 3.5))
xs = []
for i, (W, comp) in enumerate([(1.3, "vi"), (1.3, "vj"), (1.5, "vi"), (1.5, "vj")]):
    a = dk[(dk.W_mm == W) & (dk.component == comp) & (dk.region == "HG_top10pct_gradient") & (dk.condition != "POOLED")].ratio_rms_over_sigma_u
    b = wm[(wm.W_mm == W) & (wm.component == comp) & (wm.region == "HG_top10pct_gradient") & (wm.condition != "POOLED")].ratio
    for off, v, col in ((-0.17, a, VERM), (0.17, b, BLUE)):
        ax.scatter(np.full(len(v), i + off) + np.random.default_rng(1).uniform(-0.06, 0.06, len(v)), v, s=14, color=col, alpha=0.8, lw=0)
        ax.plot([i + off - 0.12, i + off + 0.12], [np.median(v)] * 2, color=col, lw=2)
ax.axhline(0.25, color="k", lw=0.9, ls="--"); ax.text(3.45, 0.26, "0.25 σ_u (rule D-kern)", ha="right", fontsize=8)
ax.set_xticks(range(4)); ax.set_xticklabels(["W=1.3\nvi", "W=1.3\nvj", "W=1.5\nvi", "W=1.5\nvj"])
ax.set_ylabel("RMS(MHG − top-hat) / σ_u, high-gradient nodes"); ax.set_ylim(0, 0.9)
ax.text(-0.45, 0.82, "red: top-hat as specified (L_sr,disc = 1.3–1.5 W wider)", color=VERM, fontsize=8)
ax.text(-0.45, 0.76, "blue: top-hat with width matched to the MHG (shape only)", color=BLUE, fontsize=8)
ax.set_title("Fig. 4 — D-kern criterion, one dot per TRAIN condition (bar = median)", fontsize=10, loc="left")
fig.savefig(FIG / "fig4_dkern.png"); plt.close(fig)

# ---------------------------------------------------------------- F5 v_ref
t = pd.read_csv(R / "f13/vref_sigma_u_table.csv")
tr = t[t.role == "train_fit_C149"].sort_values(["nh3", "phi"]); va = t[t.role == "val_eval"].sort_values(["nh3", "phi"])
a38 = t[t.role == "train_fit_A38"].set_index("condition"); se = t[t.role == "seen_train_eval"].set_index("condition")
fig, ax = plt.subplots(figsize=(8.6, 3.6))
allc = pd.concat([tr, va]); order = list(allc.sort_values(["nh3", "phi"]).condition)
for i, c in enumerate(order):
    r = allc[allc.condition == c].iloc[0]; isv = r.role == "val_eval"
    col = ORANGE if isv else BLUE
    ax.plot([i, i], [r.boot95_lo, r.boot95_hi], color=col, lw=1.2)
    ax.scatter([i], [r["v_ref_p99.9"]], color=col, s=22, zorder=3, label=None)
    if not isv:
        ax.scatter([i - 0.18], [a38.loc[c, "v_ref_p99.9"]], marker="_", color=GREY, s=30, lw=1)
        ax.scatter([i + 0.18], [se.loc[c, "v_ref_p99.9"]], marker="x", color=GREEN, s=14, lw=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels([c.replace(".h5", "") for c in order], rotation=70, ha="right", fontsize=7)
ax.set_ylabel("v_ref,c = p99.9 of |u_planar| (m/s)"); ax.set_ylim(25, 56)
sec = ax.secondary_yaxis("right", functions=(lambda v: 0.10 * v / 6, lambda s: s * 6 / 0.10)); sec.set_ylabel("σ_u at σ_px = 0.10 px, d_max = 6 px (m/s)")
fig.text(0.01, -0.07, "blue: TRAIN (C149 pool, 149 frames; bar = frame-bootstrap 95 %)   orange: VAL (41 frames)\ngrey dash: historical 38-frame pool   green ×: seen_train_eval (11 frames)", fontsize=7.5, color=GREY)
ax.set_title("Fig. 5 — Reference velocity per condition and the resulting random-error amplitude", fontsize=10, loc="left")
fig.savefig(FIG / "fig5_vref.png"); plt.close(fig)

# ---------------------------------------------------------------- F6 orientation: mean maps + symmetry
M = np.load(R / "f11/mean_maps_train_conditions.npz")
sim = "0NH3_1.h5"
fig, axs = plt.subplots(1, 4, figsize=(12.6, 3.3), constrained_layout=True)
xe, ye = g["x_mm"], g["y_mm"]
for ax, comp, ttl, cm in zip(axs, ("T", "vi", "vj", "vk"), ("mean T (K) [context]", "mean Velocity[i] (m/s)", "mean Velocity[j] (m/s)", "mean Velocity[k] (m/s)"),
                             ("magma", "RdBu_r", "RdBu_r", "RdBu_r")):
    f = M[f"{sim}|{comp}"]
    v = np.abs(f - (f.mean() if comp == "T" else 0)).max()
    im = ax.imshow(f, origin="lower", aspect="auto", cmap=cm, extent=[0, 127, 0, 127], vmin=None if comp == "T" else -v, vmax=None if comp == "T" else v)
    ax.axhline(63.5, color="k", lw=0.8, ls="--"); ax.set_title(ttl, fontsize=8.5, loc="left")
    ax.set_xlabel("column index"); ax.grid(False); plt.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
axs[0].set_ylabel("row index")
fig.suptitle(f"Fig. 6 — Time-mean maps, condition {sim.replace('.h5','')} (TRAIN, frames 0–1480 step 10). Dashed: central row 63.5", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig6_mean_maps.png"); plt.close(fig)

sy = pd.read_csv(R / "f11/mean_field_symmetry_centre_lines.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.1))
for j, (comp, lab) in enumerate((("T", "T [context]"), ("vi", "vi"), ("vj", "vj"), ("vk", "vk"))):
    for off, mir, col in ((-0.18, "about_row_centre(63.5): rows mirrored", BLUE), (0.18, "about_col_centre(63.5): cols mirrored", ORANGE)):
        v = sy[(sy.component == comp) & (sy.mirror == mir)].sym_energy_frac
        ax.scatter(np.full(len(v), j + off), v, s=16, color=col, lw=0)
ax.set_xticks(range(4)); ax.set_xticklabels(["T [context]", "vi", "vj", "vk"]); ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("symmetric fraction of energy\n(1 = symmetric, 0 = antisymmetric)")
ax.text(0.02, 0.5, "blue: mirror of the rows about row 63.5\norange: mirror of the columns about column 63.5\n(spatial mean removed; 4 TRAIN conditions)", fontsize=7.5, color=GREY, transform=ax.transAxes)
ax.set_title("Fig. 7 — Mirror symmetry of the mean fields", fontsize=10, loc="left")
fig.savefig(FIG / "fig7_symmetry.png"); plt.close(fig)

# ---------------------------------------------------------------- F7 displacement relation
d = pd.read_csv(R / "f11/displacement_patches.csv")
q = d[(d.tracer == "T") & (d.edge == 0) & (d.peak >= 0.9) & (d.contrast >= 0.3)]
fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
for ax, (tgt, pred, ttl) in zip(axs, (("dy_mm", "vi_ms", "row-axis shift vs Δt·vi"), ("dx_mm", "vj_ms", "column-axis shift vs Δt·vj"), ("dy_mm", "vk_ms", "row-axis shift vs Δt·vk"))):
    ax.scatter(q[pred], q[tgt], s=5, color=BLUE, alpha=0.35, lw=0)
    r = np.corrcoef(q[pred], q[tgt])[0, 1]; ax.set_title(f"{ttl}\nr = {r:+.2f}, n = {len(q)}", fontsize=8.5, loc="left")
    ax.set_xlabel("Δt · patch-mean velocity (mm)"); ax.set_ylabel("measured shift (mm)")
    ax.plot([-3, 3], [-3, 3], color=GREY, lw=0.6, ls=":")
fig.suptitle("Fig. 8 — Auxiliary check: shift of T structures between consecutive frames (0.25 ms) vs velocity. Dotted: 1:1", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig8_displacement.png"); plt.close(fig)

# ---------------------------------------------------------------- F8 kernels
reg = json.load(open(R / "f12/kernel_registry.json"))
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
for ax, axis, D in zip(axs, ("x", "y"), (dx, dy)):
    for W, col in ((1.3, BLUE), (1.5, ORANGE)):
        r = [q for q in reg["rows"] if q["W_mm"] == W and q["axis"] == axis][0]
        n = (len(r["mhg_weights"]) - 1) // 2
        s = np.arange(-n, n + 1) * D
        ax.step(s, r["mhg_weights"], where="mid", color=col, lw=1.3)
        nt = (len(r["th_weights"]) - 1) // 2; st = np.arange(-nt, nt + 1) * D
        ax.step(st, r["th_weights"], where="mid", color=col, lw=1.0, ls="--")
        ax.text(s[-1] * 0.5, max(r["mhg_weights"]) * (0.95 if W == 1.3 else 0.82), f"W={W}", color=col, fontsize=8)
    ax.set_xlabel(f"offset along {axis} (mm)"); ax.set_title(f"cell {D:.3f} mm", fontsize=9, loc="left")
axs[0].set_ylabel("node weight (sum = 1)")
axs[1].text(0.55, 0.1, "solid: MHG (calibrated)\ndashed: top-hat", transform=axs[1].transAxes, fontsize=7.5, color=GREY)
fig.suptitle("Fig. 9 — Discrete response kernels on the LES grid", fontsize=10, x=0.01, ha="left")
fig.savefig(FIG / "fig9_kernels.png"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
