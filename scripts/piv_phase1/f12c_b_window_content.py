"""F1.2 supplement: what do the three PIV-B windows of the roadmap rule (x centred; y centre at y_mid-30, y_mid, y_mid+30 mm)
contain, compared with the full slice? TRAIN C149; T used only as a declared diagnostic (cold-gas fraction)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import common as C
out = C.REPO / "results/piv/phase1/f12"
g = C.load_geometry(); dx, dy = g["dx"], g["dy"]
d = C.load_role("train_fit", want_T=True)
T = d["T"]; u = np.hypot(d["vi"], d["vj"])
nx, ny = int(np.floor(30 / dx)) + 1, int(np.floor(30 / dy)) + 1   # nodes inside 30 mm
def window(c_row, c_col):
    r0 = int(round(c_row - (ny - 1) / 2)); c0 = int(round(c_col - (nx - 1) / 2))
    return r0, r0 + ny, c0, c0 + nx
mid = 63.5
off_rows = 30 / dy
res = {"window_nodes": [ny, nx], "valid_nodes_after_margin_W1.5": [(ny - 2), (nx - 4)], "n_valid": (ny - 2) * (nx - 4)}
def stats(win):
    r0, r1, c0, c1 = win
    t = T[:, r0:r1, c0:c1]; s = u[:, r0:r1, c0:c1]
    return {"rows": [r0, r1], "cols": [c0, c1], "frac_T_lt_1800": float((t < 1800).mean()), "mean_speed": float(s.mean()),
            "frac_speed_gt_20": float((s > 20).mean()), "rms_vj_minus_mean": float(d["vj"][:, r0:r1, c0:c1].std())}
res["full_slice"] = {"frac_T_lt_1800": float((T < 1800).mean()), "mean_speed": float(u.mean()), "frac_speed_gt_20": float((u > 20).mean())}
for name, (cr, cc) in {"B_center(y_mid)": (mid, 63.5), "B_y_mid-30mm": (mid - off_rows, 63.5), "B_y_mid+30mm": (mid + off_rows, 63.5),
                       "ALT_axial_near_jets(cols 100-127 side)": (mid, 127 - (nx - 1) / 2)}.items():
    res[name] = stats(window(cr, cc))
json.dump(res, open(out / "b_window_content.json", "w"), indent=1)
C.dump_provenance(out / "f12c_provenance.json", {"script": "f12c_b_window_content.py"})
for k, v in res.items(): print(k, v)
