"""Markdown tables for the R1 report, generated from the CSV/JSON outputs (no hand transcription)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import common as C
R = C.REPO / "results/piv/phase1"; T = R / "tables"; T.mkdir(exist_ok=True)
def md(df, fmt=None):
    df = df.copy()
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)

# T1 kernel registry
reg = json.load(open(R / "f12/kernel_registry.json"))["rows"]
t1 = pd.DataFrame([{"W (mm)": r["W_mm"], "asse": r["axis"], "cella (mm)": round(r["cell_mm"], 3), "W/cella": round(r["W_over_cell"], 2),
                    "margine (nodi)": r["margin_nodes"], "σ_r nom. (mm)": round(r["mhg_sigma_nominal_mm"], 3),
                    "L_disc nom. (err %)": f'{r["mhg_Lsr_disc_nominal_mm"]:.3f} ({r["mhg_Lsr_disc_nominal_err_pct"]:+.1f})',
                    "σ_r tarato (mm)": round(r["mhg_sigma_used_mm"], 3), "L_disc tarato (mm)": round(r["mhg_Lsr_disc_used_mm"], 3),
                    "target 0,93 W": round(0.93 * r["W_mm"], 3), "R_r = 4σ (mm)": round(r["mhg_support_R_mm"], 2),
                    "nodi/lato": r["mhg_half_nodes"], "L_disc top-hat (mm)": round(r["th_Lsr_disc_mm"], 3)} for r in reg])
open(T / "t1_kernel_response.md", "w").write(md(t1))
t1n = pd.DataFrame([{"W (mm)": r["W_mm"], "asse": r["axis"], "σ_n nom. (mm)": round(r["noise_sigma_nominal_mm"], 3),
                     "L_disc nom. (err %)": f'{r["noise_Lsr_disc_nominal_mm"]:.3f} ({r["noise_Lsr_disc_nominal_err_pct"]:+.1f})',
                     "σ_n tarato (mm)": round(r["noise_sigma_used_mm"], 3), "target 1,2 W": round(1.2 * r["W_mm"], 3),
                     "ρ(1 nodo)": round(r["noise_rho_lag0to4_nodes"][1], 3), "ρ(2)": round(r["noise_rho_lag0to4_nodes"][2], 3),
                     "ρ(3)": round(r["noise_rho_lag0to4_nodes"][3], 3), "ρ(4)": round(r["noise_rho_lag0to4_nodes"][4], 3)} for r in reg])
open(T / "t1b_kernel_noise.md", "w").write(md(t1n))
hk = pd.DataFrame([{"W (mm)": r["W_mm"], "asse": r["axis"], **{f"λ={l} mm": round(r["mhg_H"][l], 3) for l in ("2", "3", "4", "6", "10")}} for r in reg])
open(T / "t1c_transfer_mhg.md", "w").write(md(hk))

# T2 filter effect pooled
fe = pd.read_csv(R / "f12/filter_effect.csv"); ctx = json.load(open(R / "f12/f12_context.json"))
su_pool = float(np.mean(list(ctx["sigma_u_sigma_px_0.1_dmax_6"].values())))
rows = []
for W in (1.0, 1.3, 1.5, 1.9):
    for kern, kl in (("mhg", "MHG"), ("tophat", "top-hat")):
        r = {"W (mm)": W, "kernel": kl}
        for comp in ("vi", "vj"):
            for reg_, rl in (("all", "tutti"), ("HG_top10pct_gradient", "alto grad.")):
                s = fe[(fe.condition == "POOLED") & (fe.W_mm == W) & (fe.kernel == kern) & (fe.component == comp) & (fe.region == reg_)].iloc[0]
                r[f"{comp} {rl} RMS / p95"] = f'{s.rms_m_s:.2f} / {s.p95_abs_m_s:.2f}'
        rows.append(r)
open(T / "t2_filter_effect_pooled.md", "w").write(md(pd.DataFrame(rows)))
# per region, W=1.3 and 1.5, MHG, vs sigma_u
rows = []
for W in (1.3, 1.5):
    for reg_ in ("all", "HG_top10pct_gradient", "HG_top5pct_gradient", "JET_top10pct_speed", "REST", "DIAG_T<1800K", "DIAG_T>=1800K"):
        r = {"W (mm)": W, "regione": reg_}
        for comp in ("vi", "vj"):
            s = fe[(fe.condition == "POOLED") & (fe.W_mm == W) & (fe.kernel == "mhg") & (fe.component == comp) & (fe.region == reg_)].iloc[0]
            r[f"{comp} RMS (m/s)"] = round(s.rms_m_s, 3); r[f"{comp} RMS/σ_u"] = round(s.rms_m_s / su_pool, 2)
        rows.append(r)
open(T / "t2b_filter_by_region.md", "w").write(md(pd.DataFrame(rows)))
open(T / "sigma_u_pooled_mean.txt", "w").write(f"{su_pool:.4f}\n")

# T3 D-kern
dk = pd.read_csv(R / "f12/dkern_mhg_vs_tophat.csv"); wm = pd.read_csv(R / "f12/dkern_width_matched.csv")
rows = []
for W in (1.0, 1.3, 1.5, 1.9):
    for comp in ("vi", "vj"):
        a = dk[(dk.W_mm == W) & (dk.component == comp) & (dk.region == "HG_top10pct_gradient")]
        b = wm[(wm.W_mm == W) & (wm.component == comp) & (wm.region == "HG_top10pct_gradient")]
        ap, bp = a[a.condition == "POOLED"].iloc[0], b[b.condition == "POOLED"].iloc[0]
        ac, bc = a[a.condition != "POOLED"].ratio_rms_over_sigma_u, b[b.condition != "POOLED"].ratio
        rows.append({"W (mm)": W, "comp.": comp, "top-hat come da roadmap: mediana [min–max] cond.": f"{ac.median():.2f} [{ac.min():.2f}–{ac.max():.2f}]",
                     "n. cond. > 0,25": f"{(ac > 0.25).sum()}/20",
                     "top-hat a larghezza pareggiata: mediana [min–max]": f"{bc.median():.2f} [{bc.min():.2f}–{bc.max():.2f}]", "n. cond. > 0,25 ": f"{(bc > 0.25).sum()}/20"})
open(T / "t3_dkern.md", "w").write(md(pd.DataFrame(rows)))
tw = json.load(open(R / "f12/tophat_width_matched.json"))
open(T / "t3b_tophat_matched_width.md", "w").write(md(pd.DataFrame([{"W (mm)": r["W_mm"], "asse": r["axis"], "L_disc MHG (mm)": round(r["mhg_Lsr_disc"], 3),
    "larghezza top-hat pareggiata (mm)": round(r["tophat_W_matched_mm"], 3), "rapporto a W": round(r["tophat_W_over_W"], 2)} for r in tw])))

# T4 spectra tail
sp = pd.read_csv(R / "f12/spectra_tail_metrics.csv")
g = sp.groupby(["component", "axis"]).agg(frac_lt4=("frac_var_lambda_lt_4cells", "median"), frac_lt4_max=("frac_var_lambda_lt_4cells", "max"),
                                          slope_mid=("slope_mid_0.15-0.4kN", "median"), slope_hi=("slope_high_0.5-0.95kN", "median"),
                                          psd_ratio=("psd_ratio_0.9kN_over_0.2kN", "median")).reset_index()
g["frac_lt4"] = (100 * g.frac_lt4).round(3); g["frac_lt4_max"] = (100 * g.frac_lt4_max).round(3)
g.columns = ["comp.", "asse", "% varianza a λ<4 celle (mediana)", "(max cond.)", "pendenza 0,15–0,4 k_N", "pendenza 0,5–0,95 k_N", "PSD(0,9k_N)/PSD(0,2k_N)"]
open(T / "t4_spectra_tail.md", "w").write(md(g))
sf = pd.read_csv(R / "f12/structure_functions.csv")
s = sf[sf.lag_cells.isin([1, 2, 4, 8])].groupby(["component", "axis", "lag_cells"]).S2_over_2var.median().unstack().reset_index()
s.columns = ["comp.", "asse", "S2/2var @1 cella", "@2", "@4", "@8"]
open(T / "t4b_structure_functions.md", "w").write(md(s))

# T5 vref table
t = pd.read_csv(R / "f13/vref_sigma_u_table.csv")
a = t[t.role == "train_fit_C149"].sort_values(["nh3", "phi"])
a38 = t[t.role == "train_fit_A38"].set_index("condition")["v_ref_p99.9"]; se = t[t.role == "seen_train_eval"].set_index("condition")["v_ref_p99.9"]
v = pd.DataFrame({"condizione": a.condition.str.replace(".h5", "", regex=False), "NH3": a.nh3, "φ": a.phi, "n frame": a.n_frames,
                  "p50 |u|": a.p50.round(1), "p99": a.p99.round(1), "v_ref (p99,9)": a["v_ref_p99.9"].round(2),
                  "IC95 bootstrap": [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(a.boot95_lo, a.boot95_hi)],
                  "v_ref pool 38 fr.": a.condizione.map(a38).round(1) if False else [round(a38[c], 1) for c in a.condition],
                  "v_ref seen (11 fr.)": [round(se[c], 1) for c in a.condition],
                  "σ_u 0,06 px": a["sigma_u_lo_0.06_m_s"].round(2), "σ_u 0,10 px (princ.)": a["sigma_u_main_0.10_m_s"].round(2), "σ_u 0,15 px": a["sigma_u_hi_0.15_m_s"].round(2)})
open(T / "t5_vref_sigma_train.md", "w").write(md(v))
b = t[t.role == "val_eval"].sort_values(["nh3", "phi"])
vv = pd.DataFrame({"condizione": b.condition.str.replace(".h5", "", regex=False), "NH3": b.nh3, "φ": b.phi, "n frame": b.n_frames, "p50 |u|": b.p50.round(1),
                   "p99": b.p99.round(1), "v_ref (p99,9)": b["v_ref_p99.9"].round(2), "IC95 bootstrap": [f"{lo:.1f}–{hi:.1f}" for lo, hi in zip(b.boot95_lo, b.boot95_hi)],
                   "σ_u 0,06 px": b["sigma_u_lo_0.06_m_s"].round(2), "σ_u 0,10 px (princ.)": b["sigma_u_main_0.10_m_s"].round(2), "σ_u 0,15 px": b["sigma_u_hi_0.15_m_s"].round(2)})
open(T / "t5b_vref_sigma_val.md", "w").write(md(vv))

# T6 orientation summary tables
rg = pd.read_csv(R / "f11/displacement_regressions.csv")
x = rg[(rg.tracer == "T") & (rg.subset.isin(["all_structured", "cold_Tmean<1800K"]))].copy()
x["r [IC95]"] = [f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]" for r, lo, hi in zip(x.r, x.r_lo, x.r_hi)]
x["pendenza [IC95]"] = [f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]" for r, lo, hi in zip(x.slope, x.slope_lo, x.slope_hi)]
open(T / "t6_displacement_regressions_T.md", "w").write(md(x[["subset", "n", "target", "predictor", "r [IC95]", "pendenza [IC95]"]]))
dc = pd.read_csv(R / "f11/decorrelation_drivers.csv")
dd = pd.DataFrame([{"tracciante": r.tracer, "lag (frame)": r.lag, "n": r.n, "|vi|": f"{r['coef_|vi|_per_sd']:+.4f} [{r.lo_vi:+.4f}, {r.hi_vi:+.4f}]",
                    "|vj|": f"{r['coef_|vj|_per_sd']:+.4f} [{r.lo_vj:+.4f}, {r.hi_vj:+.4f}]", "|vk|": f"{r['coef_|vk|_per_sd']:+.4f} [{r.lo_vk:+.4f}, {r.hi_vk:+.4f}]"} for _, r in dc.iterrows()])
open(T / "t6b_decorrelation.md", "w").write(md(dd))
sy = pd.read_csv(R / "f11/mean_field_symmetry_centre_lines.csv")
ss = sy.groupby(["component", "mirror"]).sym_energy_frac.agg(["min", "median", "max"]).round(3).reset_index()
ss.columns = ["campo", "specchio", "frazione simmetrica: min", "mediana", "max"]
open(T / "t6c_symmetry.md", "w").write(md(ss))
print("tables:", sorted(p.name for p in T.glob("*.md")))
