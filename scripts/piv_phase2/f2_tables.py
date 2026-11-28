"""Markdown tables for the R2 report, generated from the CSV/JSON outputs of F2.0-F2.6 (no hand transcription)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P
from combustion_recon.data.nonideal_observations import load_velocity_scale

R2 = C.REPO / "results/piv/phase2"; T = R2 / "tables"; T.mkdir(exist_ok=True)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)

# ---- v_ref
v = pd.read_csv(R2 / "f20/vref_full_record.csv").sort_values(["role", "nh3", "phi"], ascending=[False, True, True])
t = pd.DataFrame({"condizione": v.condition.str.replace(".h5", "", regex=False), "ruolo": v.role, "v_ref (p99,9) m/s": v["v_ref_p99.9"].round(2),
                  "IC95 (blocchi di 10 frame)": [f"{a:.1f}–{b:.1f}" for a, b in zip(v.boot95_lo, v.boot95_hi)], "pool C149 (Fase 1)": v.phase1_pool_v_ref.round(2),
                  "diff. relativa": [f"{100*x:+.1f} %" for x in v.rel_diff_vs_phase1_pool], "σ_u 0,06 px": v.sigma_u_lo_0_06_m_s.round(2) if "sigma_u_lo_0_06_m_s" in v else v["sigma_u_lo_0.06_m_s"].round(2),
                  "σ_u 0,10 px (princ.)": v["sigma_u_main_0.10_m_s"].round(2), "σ_u 0,15 px": v["sigma_u_hi_0.15_m_s"].round(2)})
open(T / "t1_vref_full_record.md", "w").write(md(t))
tr = v[v.role == "TRAIN"]
stat = {"train_min": float(tr["v_ref_p99.9"].min()), "train_max": float(tr["v_ref_p99.9"].max()), "train_mean": float(tr["v_ref_p99.9"].mean()),
        "sigma_train_min": float(tr["sigma_u_main_0.10_m_s"].min()), "sigma_train_max": float(tr["sigma_u_main_0.10_m_s"].max()), "sigma_train_mean": float(tr["sigma_u_main_0.10_m_s"].mean()),
        "val_min": float(v[v.role == "VAL"]["v_ref_p99.9"].min()), "val_max": float(v[v.role == "VAL"]["v_ref_p99.9"].max()),
        "rel_diff_phase1_mean": float(v.rel_diff_vs_phase1_pool.mean()), "rel_diff_phase1_max_abs": float(v.rel_diff_vs_phase1_pool.abs().max()),
        "ci_halfwidth_max": float(((v.boot95_hi - v.boot95_lo) / 2).max()), "ci_halfwidth_median": float(((v.boot95_hi - v.boot95_lo) / 2).median()),
        "sha_matches_inventory_all": bool(v.sha_matches_inventory.all())}
json.dump(stat, open(T / "vref_summary.json", "w"), indent=1)

# ---- kernel registry
reg = json.load(open(R2 / "f20/kernel_registry_v2.json"))["kernels"]
rows = []
for conv in ("primary", "swap"):
    for W in ("1.0", "1.3", "1.5", "1.9"):
        e = reg[conv][W]
        for axis in ("rows", "cols"):
            r_, n_, t_ = e["response_mhg"][axis], e["noise"][axis], e["response_tophat"][axis]
            rows.append({"convenzione": "R1" if conv == "primary" else "SENS-SWAP", "W (mm)": float(W), "asse": axis, "passo (mm)": round(e["window"][axis]["step_mm"], 4), "margine (nodi)": e["window"][axis]["margin_nodes"],
                         "σ_r nom. → tarato (mm)": f"{r_['param_nominal_mm']:.3f} → {r_['param_mm']:.3f}", "L_sr,disc nom. (target 0,93 W)": f"{r_['lsr_disc_nominal_mm']:.3f} ({100*(r_['lsr_disc_nominal_mm']/r_['lsr_target_mm']-1):+.1f} %)",
                         "L_sr,disc tarato": f"{r_['lsr_disc_mm']:.4f}", "nodi/lato": r_["half_nodes"], "R_r (mm)": round(r_["support_mm"], 2),
                         "σ_n tarato (mm)": round(n_["param_mm"], 3), "L_n,disc (target 1,2 W)": f"{n_['lsr_disc_mm']:.4f}", "W_eff top-hat (mm)": round(t_["param_mm"], 3), "W_eff/W": round(t_["param_mm"] / float(W), 3)})
kd = pd.DataFrame(rows)
open(T / "t2_kernel_registry_primary.md", "w").write(md(kd[kd.convenzione == "R1"].drop(columns="convenzione")))
open(T / "t2b_kernel_registry_swap.md", "w").write(md(kd[kd.convenzione == "SENS-SWAP"].drop(columns="convenzione")))
rr = json.load(open(R2 / "f20/kernel_registry_v2.json"))
mono = json.load(open(R2 / "f20/kernel_calibration_monotonicity.json"))

# ---- swapcells index
si = pd.read_csv(R2 / "f20/swapcells_index.csv")
t = pd.DataFrame({"W (mm)": si.W_mm, "asse (R1)": si.axis_R1, "passo coincide con R1": si.step_matches_R1, "σ_r swapcells (mm)": si.sigma_resp_swapcells.round(4), "σ_r R1 (mm)": si.sigma_resp_R1.round(4),
                  "diff. rel. risposta": [f"{100*x:+.2f} %" for x in si.resp_rel_diff], "diff. rel. rumore": [f"{100*x:+.2f} %" for x in si.noise_rel_diff], "tabelle F1.2 riusabili": si.filter_table_reusable})
open(T / "t3_swapcells_index.md", "w").write(md(t))

# ---- windows
bw = json.load(open(R2 / "f20/b_windows_content.json"))
rows = [{"finestra": "intera slice", "righe × colonne": "0–127 × 0–127", "vettori validi (W)": "A: 15 624 (W = 1,3)", "% nodi con |u| > 20 m/s": round(100 * bw["slice_full"]["frac_u_gt_20"], 1),
         "|u| medio (m/s)": round(bw["slice_full"]["mean_speed"], 1), "RMS di vj (m/s)": round(bw["slice_full"]["rms_vj"], 1), "% T < 1800 K (diagnostica)": round(100 * bw["slice_full"]["frac_T_lt_1800_DIAGNOSTIC"], 1)}]
for f in ("B-in", "B-down", "B-off"):
    e = bw[f"{f}|primary"]
    rows.append({"finestra": f, "righe × colonne": f"{e['rows'][0]}–{e['rows'][1]} × {e['cols'][0]}–{e['cols'][1]}", "vettori validi (W)": f"{e['n_valid_vectors_W1.5']} (W = 1,5)",
                 "% nodi con |u| > 20 m/s": round(100 * e["frac_u_gt_20"], 1), "|u| medio (m/s)": round(e["mean_speed"], 1), "RMS di vj (m/s)": round(e["rms_vj"], 1),
                 "% T < 1800 K (diagnostica)": round(100 * e["frac_T_lt_1800_DIAGNOSTIC"], 1)})
for f in ("B-in", "B-down", "B-off"):
    e = bw[f"{f}|swap"]
    rows.append({"finestra": f"{f} (SENS-SWAP)", "righe × colonne": f"{e['rows'][0]}–{e['rows'][1]} × {e['cols'][0]}–{e['cols'][1]}", "vettori validi (W)": f"{e['n_valid_vectors_W1.5']} (W = 1,5)",
                 "% nodi con |u| > 20 m/s": round(100 * e["frac_u_gt_20"], 1), "|u| medio (m/s)": round(e["mean_speed"], 1), "RMS di vj (m/s)": round(e["rms_vj"], 1),
                 "% T < 1800 K (diagnostica)": round(100 * e["frac_T_lt_1800_DIAGNOSTIC"], 1)})
open(T / "t4_B_windows.md", "w").write(md(pd.DataFrame(rows)))

# ---- counts, edge vectors
sha = (C.REPO / "configs/observations/piv_observation_spec_v1.sha256").read_text().split()[0]
spec = P.load_piv_spec(C.REPO / "configs/observations/piv_observation_spec_v1.yaml", expected_sha256=sha)
scale = load_velocity_scale(C.DEFAULT_DATA_ROOT / C.STATS_REL, expected_sha256=C.STATS_SHA256_EXPECTED)
g = np.load(C.DEFAULT_DATA_ROOT / C.CACHE_A38 / "geometry.npz")
gen = P.PivObservationGenerator(spec, scale, x_vec_m=g["x"], y_vec_m=g["y"])
z = np.zeros((128, 128))
rows = []
for cfg in ("PIV-A", "PIV-B-in", "PIV-B-down", "PIV-B-off", "IDEAL-full", "SENS-EDGE-A", "SENS-EDGE-B-in", "SENS-TH-A", "SENS-SWAP-A", "SENS-SWAP-B-in", "SENS-W1.0-A", "SENS-W1.9-A", "SENS-REJ-iso-flag-A", "SENS-REJ-clu-sub-A"):
    o = gen.observe(cfg, z, z, sim_id="0NH3_1.h5", frame_index=0, role="train")
    rows.append({"configurazione": cfg, "campo (righe0, righe1, col0, col1)": str(tuple(o.meta["field_rect_rows_cols"])), "vettori validi": o.meta["n_valid_vectors"], "vettori osservati": o.meta["n_observed"],
                 "% della slice": round(100 * o.meta["n_observed"] / 16384, 1), "vettori di bordo": o.meta["n_edge_vectors"], "rigettati": o.meta["n_rejected"], "sostituiti": o.meta["n_substituted"]})
open(T / "t5_counts.md", "w").write(md(pd.DataFrame(rows)))

# ---- realised response
rp = pd.read_csv(R2 / "f26/tables/realised_response_interior_vs_edge.csv")
rp = rp.rename(columns={"axis": "asse", "field": "campo", "lambda_mm": "λ (mm) / L_sr", "H_interior": "interno", "H_edge_node0": "nodo di bordo (0)", "H_edge_node1": "nodo 1"})
open(T / "t6_realised_response.md", "w").write(md(rp))

# ---- RMS pooled
rm = pd.read_csv(R2 / "f26/tables/rms_pooled_by_region.csv")
def pick(cfg, kind, label):
    q = rm[(rm.config == cfg) & (rm.quantity == kind)]
    out = {"quantità": label}
    for comp in ("vi", "vj"):
        s = q[q.component == comp].iloc[0]
        for reg in ("all", "HG", "JET", "REST", "edge", "interior"):
            out[f"{comp} {reg}"] = f"{s[f'{reg} RMS (m/s)']:.2f} ({s[f'{reg} / sigma_u']:.2f})" if np.isfinite(s[f'{reg} RMS (m/s)']) else "–"
    return out
rows = [pick("FILT-A", "filt_minus_ideal", "A: filtrato − ideale"), pick("PIV-A", "error", "A: errore"), pick("PIV-A", "obs_minus_ideal", "A: osservato − ideale"),
        pick("FILT-B-in", "filt_minus_ideal", "B-in: filtrato − ideale"), pick("PIV-B-in", "error", "B-in: errore"), pick("PIV-B-in", "obs_minus_ideal", "B-in: osservato − ideale"),
        pick("FILT-B-down" if False else "PIV-B-down", "filt_minus_ideal", "B-down: filtrato − ideale"), pick("PIV-B-down", "error", "B-down: errore"),
        pick("PIV-B-off", "filt_minus_ideal", "B-off: filtrato − ideale"), pick("PIV-B-off", "error", "B-off: errore")]
tt = pd.DataFrame(rows)
open(T / "t7_rms_vi.md", "w").write(md(tt[["quantità"] + [c for c in tt.columns if c.startswith("vi ")]]))
open(T / "t7b_rms_vj.md", "w").write(md(tt[["quantità"] + [c for c in tt.columns if c.startswith("vj ")]]))
rows = [pick("SENS-TH-A", "filt_minus_ideal", "SENS-TH A: filtrato − ideale"), pick("TH-vs-MHG-A", "filtered_diff", "A: top-hat − MHG"), pick("SENS-SWAP-A", "filt_minus_ideal", "SENS-SWAP A: filtrato − ideale"),
        pick("SWAP-vs-primary-A", "filtered_diff", "A: SWAP − convenzione R1"), pick("SENS-W1.0-A", "filt_minus_ideal", "SENS-W 1,0 A: filtrato − ideale"), pick("SENS-W1.9-A", "filt_minus_ideal", "SENS-W 1,9 A: filtrato − ideale"),
        pick("SENS-TH-B-in", "filt_minus_ideal", "SENS-TH B-in: filtrato − ideale"), pick("TH-vs-MHG-B-in", "filtered_diff", "B-in: top-hat − MHG"), pick("SENS-SWAP-B-in", "filt_minus_ideal", "SENS-SWAP B-in: filtrato − ideale"),
        pick("SWAP-vs-primary-B-in", "filtered_diff", "B-in: SWAP − R1")]
tt = pd.DataFrame(rows)
open(T / "t8_sensitivities_vi.md", "w").write(md(tt[["quantità"] + [c for c in tt.columns if c.startswith("vi ")]]))
open(T / "t8b_sensitivities_vj.md", "w").write(md(tt[["quantità"] + [c for c in tt.columns if c.startswith("vj ")]]))

# ---- sigma target vs realised
ch = pd.read_csv(R2 / "f26/tables/sigma_u_target_vs_realised_PIV-A.csv")
sg = {"ratio_vi_min": float(ch.ratio_vi.min()), "ratio_vi_max": float(ch.ratio_vi.max()), "ratio_vj_min": float(ch.ratio_vj.min()), "ratio_vj_max": float(ch.ratio_vj.max())}
json.dump(sg, open(T / "sigma_realised_summary.json", "w"), indent=1)

# ---- autocorrelation empirical vs analytic
ac = pd.read_csv(R2 / "f26/error_autocorrelation_empirical.csv")
rows = []
for cfg, W in (("PIV-A", 1.3), ("PIV-B-in", 1.5)):
    for axis, step in (("rows", spec.steps["rows"]), ("cols", spec.steps["cols"])):
        e = spec.kernel("noise", "primary", W, axis); k, w = P.kernel_weights("mhg", e["param_mm"], step); an = P.autocorr_discrete(w, 4)
        for lag in (1, 2, 3, 4):
            q = ac[(ac.config == cfg) & (ac.axis == axis) & (ac.lag_nodes == lag)].iloc[0]
            rows.append({"config": cfg, "asse": axis, "ritardo (nodi)": lag, "ritardo (mm)": round(lag * step, 3), "ρ misurata": round(q.rho, 4), "ρ analitica discreta": round(an[lag], 4), "differenza": round(q.rho - an[lag], 4)})
open(T / "t9_error_autocorrelation.md", "w").write(md(pd.DataFrame(rows)))
ed = json.load(open(R2 / "f26/error_distribution.json"))
open(T / "t9b_error_distribution.md", "w").write(md(pd.DataFrame([{"config": k, "n": v["n"], "media": round(v["mean"], 4), "std": round(v["std"], 4), "asimmetria": round(v["skew"], 4), "curtosi in eccesso": round(v["excess_kurtosis"], 4),
                                                                      "frazione |z|>3": round(v["frac_abs_gt_3"], 5), "gaussiana": round(v["gauss_frac_abs_gt_3"], 5)} for k, v in ed.items()])))

# ---- rejections
rj = json.load(open(R2 / "f26/rejection_morphology.json"))
open(T / "t10_rejections.md", "w").write(md(pd.DataFrame([{"tipo": k, "frazione realizzata": round(v["realised_fraction"], 5), "cluster per frame": round(v["n_clusters_per_frame"], 0), "dimensione media (nodi)": round(v["size_mean"], 2),
                                                                 "mediana": v["size_median"], "p90": v["size_p90"], "max": v["size_max"], "% dei rigettati in cluster ≥ 3": round(100 * v["share_of_rejected_in_clusters_ge3"], 1),
                                                                 "% in cluster ≥ 10": round(100 * v["share_of_rejected_in_clusters_ge10"], 1), "passate di sostituzione medie/max": f"{v['substitution_passes_mean']:.2f} / {v['substitution_passes_max']}"} for k, v in rj.items()])))
sub = pd.read_csv(R2 / "f26/rms_by_config_condition_region.csv"); sub = sub[sub.kind == "diff"]
sig_c = {c: 0.10 * spec.vref(c) / 6.0 for c in spec.raw["reference_velocity"]["per_condition_m_s"]}
rows = []
for rej in ("iso", "clu"):
    for region in ("all", "HG", "REST"):
        r = {"rigetti": rej, "regione": region}
        for comp in ("vi", "vj"):
            q = sub[(sub.config == f"REJ-{rej}-sub-vs-unrejected") & (sub.region == region) & (sub.component == comp)].copy()
            q["su"] = q.condition.map(sig_c)
            r[f"{comp}: RMS(sostituito − non rigettato) (m/s)"] = round(float(np.sqrt(q.sum_sq.sum() / q.n.sum())), 3)
            r[f"{comp} / σ_u"] = round(float(np.sqrt((q.sum_sq / q.su**2).sum() / q.n.sum())), 3)
        rows.append(r)
open(T / "t10b_substitution_error.md", "w").write(md(pd.DataFrame(rows)))

# ---- Neal
nc = pd.read_csv(R2 / "f24/neal_correlation_analysis.csv")
nl = json.load(open(R2 / "f24/neal_residual_statistics.json"))
rows = []
for ds in ("F001", "F005"):
    for cfg, e in nl[ds]["configs"].items():
        q = nc[(nc.dataset == ds) & (nc.W_px == e["W_px"]) & (nc.overlap == e["overlap"]) & (nc.component == "u") & (nc.removal == "V1")]
        x, y = q[q.axis == "x"].iloc[0], q[q.axis == "y"].iloc[0]
        rows.append({"serie": ds, "W (px)": e["W_px"], "overlap": f"{int(100*e['overlap'])} %", "vettori con riferimento": e["n_vectors_with_reference"], "σ residuo u, dopo media per coppia (px)": round(x.std_px, 3),
                     "σ robusta (MAD) (px)": round(e["u"]["robust_sigma_px"], 3), "frazione outlier > 5 MAD": round(e["u"]["frac_outliers_gt_5MAD"], 3), "curtosi in eccesso (pulita)": round(e["u"]["excess_kurtosis_clean"], 2),
                     "ρ x a ritardo 1 / 2 passi": f"{x.rho_lag1:.2f} / {x.rho_lag2:.2f}", "ρ y a ritardo 1 / 2 passi": f"{y.rho_lag1:.2f} / {y.rho_lag2:.2f}", "modello a 1 / 2 passi": f"{x.model_rho_lag1:.2f} / {x.model_rho_lag2:.2f}",
                     "ρ a ritardo = W: x / y": f"{x['rho_at_lag_W_(2d_if_50pct)']:.2f} / {y['rho_at_lag_W_(2d_if_50pct)']:.2f} (modello {x['model_rho_at_lag_W']:.2f})"})
open(T / "t11_neal.md", "w").write(md(pd.DataFrame(rows)))
al = {ds: nl[ds]["alignment"] for ds in nl}
json.dump(al, open(T / "neal_alignment.json", "w"), indent=1)

# ---- Cardiff
cs = json.load(open(R2 / "f25/chain_summary.json"))
sh = pd.read_csv(R2 / "f25/scout_sha256_reproduction.csv")
cv = json.load(open(R2 / "f25/decoder_and_mask_validation.json"))
rows = [{"voce": "frame decodificati", "valore": f"{cs['frames']} / 514; errori di decodifica: {len(cv['decode_errors'])}"},
        {"voce": "sha256 dei 12 file già estratti dallo scout riprodotti", "valore": f"{int(sh.in_scout_manifest.sum())} / {len(sh)}"},
        {"voce": "regione non finita al primo stadio (frame 0, 1, 257, 513)", "valore": f"{cv['mask_check']['stage1_nonfinite_counts_frames_0_1_257_513']} vettori; intersezione {cv['mask_check']['nonfinite_intersection']}, unione {cv['mask_check']['nonfinite_union']}"},
        {"voce": "statistica `StatisticsVec#0` confrontata con la media dei 514 campi", "valore": "NON eseguibile: " + str(cv["decoder_validation"].get("error", "n/d"))},
        {"voce": "quota media di codice 16 allo stadio finale (min–max per frame)", "valore": f"{100*cs['final_code16_share_mean_min_max'][0]:.1f} % ({100*cs['final_code16_share_mean_min_max'][1]:.1f}–{100*cs['final_code16_share_mean_min_max'][2]:.1f} %), di cui {100*cs['mean_share_nonfinite']['stage1']:.1f} % non finiti già al primo stadio"},
        {"voce": "vettori con codice 0 modificati fra primo e ultimo stadio", "valore": str(cs["code0_vectors_changed_raw_to_final_total"])}]
open(T / "t12_cardiff_checks.md", "w").write(md(pd.DataFrame(rows)))
sc = cs["substitution_comparison"]
rows = []
for comp in ("U", "V"):
    s_ = sc[comp]
    rows.append({"componente": comp, "vettori sostituiti confrontabili": s_["n_compared"], "mediana |reale − media 3×3|": round(s_["median_abs_actual_minus_3x3"], 3), "mediana |reale − media 5×5|": round(s_["median_abs_actual_minus_5x5"], 3),
                 "mediana |reale − regola del generatore|": round(s_["median_abs_actual_minus_iterative_3x3"], 3), "RMS reale − 3×3": round(s_["rms_actual_minus_3x3_mean_known"], 2), "RMS reale − 5×5": round(s_["rms_actual_minus_5x5_mean_known"], 2),
                 "scala di variabilità (RMS vettore valido − media dei vicini)": round(s_["rms_valid_vector_minus_3x3_mean_of_neighbours (leave-one-out variability scale)"], 2)})
open(T / "t12b_cardiff_substitution.md", "w").write(md(pd.DataFrame(rows)))
print("tables:", sorted(p.name for p in T.glob("*.md")))
