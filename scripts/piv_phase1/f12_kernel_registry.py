"""F1.2 part 0: discrete kernel registry on the LES grid (formulas, calibration, H(k), noise autocorr)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C
import piv_kernels as K

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(C.REPO / "results/piv/phase1/f12"))
ap.add_argument("--swap-cells", action="store_true", help="sensitivity: exchange the cell sizes of the two array axes")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

g = C.load_geometry()
dx, dy = g["dx"], g["dy"]
if a.swap_cells: dx, dy = dy, dx
x, y = g["x_mm"], g["y_mm"]
grid_check = {
    "dx_mean": dx, "dy_mean": dy,
    "dx_rel_spread": float(np.ptp(np.diff(x)) / dx), "dy_rel_spread": float(np.ptp(np.diff(y)) / dy),
    "x_extent_mm": float(x[-1] - x[0]), "y_extent_mm": float(y[-1] - y[0]),
}
Ws = [1.0, 1.3, 1.5, 1.9]
lams = [2, 3, 4, 5, 6, 8, 10]
rows = []
for W in Ws:
    for axis, D in (("x", dx), ("y", dy)):
        rec = {"W_mm": W, "axis": axis, "cell_mm": D, "W_over_cell": W / D,
               "margin_nodes": K.valid_margin(W, D)}
        # MHG response
        s0, sg, l0, l1, recal, reach = K.calibrate_sigma(K.LSR_RESPONSE * W, D)
        s, w = K.mhg_weights(sg, D)
        rec.update({"mhg_sigma_nominal_mm": s0, "mhg_sigma_used_mm": sg, "mhg_sigma_nominal_over_W": s0 / W,
                    "mhg_Lsr_target_mm": K.LSR_RESPONSE * W, "mhg_Lsr_disc_nominal_mm": l0,
                    "mhg_Lsr_disc_nominal_err_pct": 100 * (l0 / (K.LSR_RESPONSE * W) - 1),
                    "mhg_Lsr_disc_used_mm": l1, "mhg_recalibrated": recal, "mhg_reachable": reach,
                    "mhg_support_R_mm": K.TRUNC_SIGMA * sg, "mhg_half_nodes": int((len(w) - 1) // 2),
                    "mhg_n_weights": int(len(w)), "mhg_neg_mass": float(w[w < 0].sum()),
                    "mhg_weights": w.tolist()})
        H = K.transfer(s, w, lams)
        rec["mhg_H"] = dict(zip(map(str, lams), H.tolist()))
        # continuous H for comparison (analytic from continuous kernel: numerical)
        ss = np.linspace(-K.TRUNC_SIGMA * sg, K.TRUNC_SIGMA * sg, 20001)
        Fc = K.mhg_F(ss, sg); Fc = Fc / np.trapezoid(Fc, ss)
        Hc = [float(np.trapezoid(Fc * np.cos(2 * np.pi / l * ss), ss)) for l in lams]
        rec["mhg_H_continuous"] = dict(zip(map(str, lams), Hc))
        # top-hat
        st, wt = K.tophat_weights(W, D)
        rec.update({"th_Lsr_target_mm": W, "th_Lsr_disc_mm": K.lsr_discrete(wt, D),
                    "th_half_nodes": int((len(wt) - 1) // 2), "th_weights": wt.tolist()})
        rec["th_H"] = dict(zip(map(str, lams), K.transfer(st, wt, lams).tolist()))
        # noise kernel
        n0, ng, nl0, nl1, nrecal, nreach = K.calibrate_sigma(K.LSR_NOISE * W, D)
        sn, wn = K.mhg_weights(ng, D)
        ac = K.autocorr_discrete(wn, 4)
        rec.update({"noise_sigma_nominal_mm": n0, "noise_sigma_used_mm": ng, "noise_sigma_nominal_over_W": n0 / W,
                    "noise_Lsr_target_mm": K.LSR_NOISE * W, "noise_Lsr_disc_nominal_mm": nl0,
                    "noise_Lsr_disc_nominal_err_pct": 100 * (nl0 / (K.LSR_NOISE * W) - 1),
                    "noise_Lsr_disc_used_mm": nl1, "noise_recalibrated": nrecal, "noise_reachable": nreach,
                    "noise_support_R_mm": K.TRUNC_SIGMA * ng, "noise_half_nodes": int((len(wn) - 1) // 2),
                    "noise_rho_lag0to4_nodes": ac.tolist(), "noise_rho_lag_mm": [m * D for m in range(5)]})
        rows.append(rec)

json.dump({"grid": grid_check, "rows": rows}, open(out / "kernel_registry.json", "w"), indent=1)

# compact CSV
import csv
cols = ["W_mm", "axis", "cell_mm", "W_over_cell", "margin_nodes", "mhg_sigma_nominal_mm", "mhg_sigma_used_mm",
        "mhg_Lsr_target_mm", "mhg_Lsr_disc_nominal_mm", "mhg_Lsr_disc_nominal_err_pct", "mhg_Lsr_disc_used_mm",
        "mhg_recalibrated", "mhg_support_R_mm", "mhg_half_nodes", "mhg_neg_mass", "th_Lsr_disc_mm", "th_half_nodes",
        "noise_sigma_nominal_mm", "noise_sigma_used_mm", "noise_Lsr_target_mm", "noise_Lsr_disc_nominal_mm",
        "noise_Lsr_disc_nominal_err_pct", "noise_Lsr_disc_used_mm", "noise_recalibrated", "noise_half_nodes"]
with open(out / "kernel_registry.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(cols)
    for r in rows: wr.writerow([f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
json.dump(K.verify(), open(out / "kernel_formula_verification.json", "w"), indent=1)
C.dump_provenance(out / "f12_kernel_registry_provenance.json", {"script": "f12_kernel_registry.py", "grid": grid_check})
print(json.dumps(grid_check))
for r in rows:
    print(f"W={r['W_mm']} {r['axis']} W/D={r['W_over_cell']:.2f} margin={r['margin_nodes']} "
          f"mhg sig {r['mhg_sigma_nominal_mm']:.3f}->{r['mhg_sigma_used_mm']:.3f} Ldisc err {r['mhg_Lsr_disc_nominal_err_pct']:+.1f}% "
          f"| TH Ldisc {r['th_Lsr_disc_mm']:.3f}/{r['th_Lsr_target_mm']} | noise err {r['noise_Lsr_disc_nominal_err_pct']:+.1f}% "
          f"rho1={r['noise_rho_lag0to4_nodes'][1]:.2f} rho2={r['noise_rho_lag0to4_nodes'][2]:.2f} rho3={r['noise_rho_lag0to4_nodes'][3]:.2f} "
          f"H(4mm)={r['mhg_H']['4']:.3f}")
