"""F2.1 - write and freeze ``piv_observation_spec_v1`` (YAML + sha256 sidecar) from the F2.0 outputs.

The spec is the single source of truth for the generator: kernel scales, reference velocities, windows, configurations,
seeds and the classification of every choice. After writing, the sha256 of the file is stored next to it; the generator and
the tests refuse a spec whose hash differs.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd, yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piv_phase1"))
import common as C
sys.path.insert(0, str(C.REPO / "src"))
from combustion_recon.data import piv_observations as P

ap = argparse.ArgumentParser()
ap.add_argument("--f20", default=str(C.REPO / "results/piv/phase2/f20"))
ap.add_argument("--out", default=str(C.REPO / "configs/observations/piv_observation_spec_v1.yaml"))
a = ap.parse_args()
f20 = Path(a.f20)
vref = pd.read_csv(f20 / "vref_full_record.csv")
reg = json.load(open(f20 / "kernel_registry_v2.json"))
if reg["problems"]:
    raise SystemExit(f"kernel calibration problems, spec NOT written: {reg['problems']}")
st = reg["steps_mm"]
KEEP = ("param_mm", "param_nominal_mm", "lsr_target_mm", "lsr_disc_mm", "err_rel", "half_nodes", "support_mm", "neg_mass")
kernels = {}
for conv, byw in reg["kernels"].items():
    kernels[conv] = {}
    for W, ent in byw.items():
        kernels[conv][W] = {k: {ax: {kk: (round(float(v[kk]), 12) if isinstance(v[kk], float) else v[kk]) for kk in KEEP}
                                for ax, v in ent[k].items()} for k in ("response_mhg", "noise", "response_tophat")}
        kernels[conv][W]["window_margin_nodes"] = {ax: ent["window"][ax]["margin_nodes"] for ax in ("rows", "cols")}

# ---- configurations
def cfg(field, W, response, noise, sigma_px=0.0, rejection=None, level=None, steps="primary", edge_strict=False, role="", note=""):
    d = {"field": field, "W_mm": W, "response": response, "noise": noise, "sigma_px": sigma_px, "steps": steps,
         "role": role, "note": note}
    if rejection: d["rejection"] = rejection
    if level: d["level"] = level
    if edge_strict: d["edge_strict"] = True
    return d
SIG = 0.10
WB = {"A": 1.3, "B-in": 1.5, "B-down": 1.5, "B-off": 1.5}
configs = {}
for f, W in WB.items():
    configs[f"PIV-{f}"] = cfg(f, W, "mhg", "corr", SIG, role="main", note="training+evaluation" if f in ("A", "B-in") else "evaluation only")
    configs[f"IDEAL-{f}"] = cfg(f, W, "ideal", "none", 0.0, role="control", note="ideal LES values at the valid nodes of the matching PIV config")
    configs[f"FILT-{f}"] = cfg(f, W, "mhg", "none", 0.0, role="control", note="response only, no random error (contrast filter vs error)")
configs["IDEAL-full"] = cfg("full", 1.3, "ideal", "none", 0.0, role="control", note="all 16384 nodes, evaluation layout reference; never trained")
for f in ("A", "B-in"):
    W = WB[f]
    configs[f"SENS-TH-{f}"] = cfg(f, W, "tophat", "corr", SIG, role="sensitivity", note="single-pass top-hat calibrated to L_sr,disc = W (R1 sec. 5)")
    configs[f"SENS-SWAP-{f}"] = cfg(f, W, "mhg", "corr", SIG, steps="swap", role="sensitivity", note="swapped steps = historical convention (cols<->x)")
    configs[f"SENS-EDGE-{f}"] = cfg(f, W, "mhg", "corr", SIG, edge_strict=True, role="sensitivity", note="only vectors whose kernel support is fully inside the field")
    configs[f"SENS-SIGLO-{f}"] = cfg(f, W, "mhg", "corr", 0.06, role="sensitivity", note="sigma_px 0.06 (WIE17 measured)")
    configs[f"SENS-SIGHI-{f}"] = cfg(f, W, "mhg", "corr", 0.15, role="sensitivity", note="sigma_px 0.15 (WIE17 synthetic DaVis)")
    configs[f"SENS-RHO0-{f}"] = cfg(f, W, "mhg", "indep", SIG, role="sensitivity", note="independent error per node: modelling limit case, not the review result")
for W in (1.0, 1.9):
    configs[f"SENS-W{W:.1f}-A"] = cfg("A", W, "mhg", "corr", SIG, role="sensitivity", note="window not seen in the main configs (OBE15 / BOX10; KAR20 non-GTMC)")
for rej in ("iso", "clu"):
    for lev in ("flag", "sub"):
        configs[f"SENS-REJ-{rej}-{lev}-A"] = cfg("A", 1.3, "mhg", "corr", SIG, rejection=rej, level=lev, role="sensitivity",
                                                  note="10 % of the valid vectors rejected; NOT calibrated, NOT transferred")

# ---- parameter classification
D, T_, S_, M_ = "documentato", "trasferimento", "sensibilita", "scelta di modellazione (R1)"
params = [
 {"id": "grid_steps_mm", "value": f"rows {st['rows']:.6f}, cols {st['cols']:.6f}", "class": D, "source": "dataset geometry (INV B-data-audit-001); assignment rows<->x, cols<->y: ipotesi R1 'probabile'"},
 {"id": "assignment_rows_x_cols_y", "value": "rows<->x (lateral), cols<->y (axial)", "class": M_, "source": "decisione_R1 sec. 3; SENS-SWAP covers the historical convention"},
 {"id": "components", "value": "(vi, vj) planar; vi transverse, vj axial; vk out of plane, not used", "class": "ipotesi dichiarata (probabile)", "source": "orientation_note.md, decisione_R1 sec. 2"},
 {"id": "W_A_mm", "value": 1.3, "class": D, "source": "STO12 sec. 2.3.2 (GTMC 5 Hz, 1.3 mm, 50 % overlap)"},
 {"id": "W_B_mm", "value": 1.5, "class": D, "source": "STO12 sec. 2.3.3 (GTMC 5 kHz, 1.5 x 1.5 mm^2; BOX10 p.15 quotes 1.0 mm)"},
 {"id": "layout_one_vector_per_LES_node", "value": "step = LES cell, anisotropic", "class": T_, "source": "PIV step = W/2 (BOX10, SAI16, KAR20); LES cell 0.581/0.758 mm used instead of 0.5W (declared)"},
 {"id": "field_A", "value": "whole slice (extreme nodes)", "class": T_, "source": "STO12 'entire vertical section'; assumption slice = observed section"},
 {"id": "field_B_size", "value": "30 x 30 mm (52 rows x 40 cols)", "class": D, "source": "STO12 sec. 2.3.3"},
 {"id": "field_B_positions", "value": "B-in (train+eval), B-down, B-off (eval)", "class": M_, "source": "STO12 'above the nozzle'; decisione_R1 sec. 7; nozzle position relative to the slice unknown"},
 {"id": "window_rule", "value": "vector exists iff [-W/2, W/2]^2 inside the field", "class": T_, "source": "PIV practice; margin 2 rows / 1 col for W=1.3 and 1.5"},
 {"id": "response_kernel", "value": "MHG b=0.35, discrete L_sr = 0.93 W, calibrated per axis (1 %)", "class": T_, "source": "WIE17 Tab. 2.1/2.4 (synthetic DaVis multi-pass); GTMC weighting not declared"},
 {"id": "kernel_support", "value": "4 sigma, truncated and renormalised; truncated at the field edge", "class": M_, "source": "decisione_R1 sec. 4; edge behaviour is a declared approximation"},
 {"id": "field_discretisation", "value": "piecewise constant per LES cell, cell-integrated weights", "class": M_, "source": "roadmap sec. 2.3(3)"},
 {"id": "error_amplitude", "value": "sigma_u,c = sigma_px * v_ref,c / d_max, sigma_px = 0.10 px, d_max = 6 px", "class": T_, "source": "STO12 sec. 2.3.2 (+-0.1 px, Delta t per condition); WIE17 s4.4.2 (0.060 px measured) and Tab. 2.4 (0.146 px synthetic). '+-0.1 px' is NOT read as sigma"},
 {"id": "d_max_px", "value": 6, "class": T_, "source": "inference from STO12 (16 px = 1.3 mm, v ~ 80 m/s, 6 us)"},
 {"id": "v_ref", "value": "p99.9 of |u_planar|, all nodes, frames 0:2001:10 of each condition", "class": T_, "source": "STO12: Delta t chosen per condition on the instantaneous velocity scale; decisione_R1 sec. 6"},
 {"id": "error_distribution", "value": "Gaussian", "class": "ipotesi di modello", "source": "not established for flames"},
 {"id": "error_correlation", "value": "separate noise kernel g, MHG b=0.35, L_sr = 1.2 W, discrete, calibrated", "class": T_, "source": "WIE17 Tab. 2.4 / s4.3 (rho(2d)=0.11 at 50 % overlap vs 0 in the review: unresolved)"},
 {"id": "components_independent", "value": "u and v errors independent", "class": "ipotesi", "source": "WIE17 s4.3"},
 {"id": "error_stationarity", "value": "white noise on a grid padded by 16 nodes", "class": M_, "source": "numerical choice"},
 {"id": "bias", "value": "none", "class": M_, "source": "no transferable bias documented (SINT 2.2)"},
 {"id": "rejections_main", "value": "none in main/controls", "class": "approssimazione dichiarata", "source": "released GTMC data are complete after interpolation (BOX10 p.31); this does not prove the absence of rejections"},
 {"id": "rejection_fraction", "value": 0.10, "class": S_, "source": "upper bound quoted by BAU15 p.49 (micro-PIV H2): NOT calibrated, NOT transferred"},
 {"id": "rejection_clu", "value": "top-fraction nodes of a correlated Gaussian field on the noise-kernel scale", "class": S_, "source": "morphology not documented in swirl flames"},
 {"id": "substitution", "value": "iterative 3x3 mean of known neighbours, presented as measured (Mv=1)", "class": T_, "source": "BAU15 p.49, BOX10 p.31 practice; Cardiff F2.5 qualifies the rule, not the frequency"},
 {"id": "top_hat", "value": "width tuned so that discrete L_sr = W", "class": S_, "source": "decisione_R1 sec. 5"},
 {"id": "seeds", "value": "new namespace piv_observation_v1; roles train/selection/eval", "class": M_, "source": "distinct from 20261114/16/18 and 20261125/20261201/20261202"},
]
vref_tab = {r.condition: round(float(r["v_ref_p99.9"]), 6) for _, r in vref.iterrows()}
ci_tab = {r.condition: [round(float(r.boot95_lo), 4), round(float(r.boot95_hi), 4)] for _, r in vref.iterrows()}
f20_prov = json.load(open(f20 / "f20_provenance.json"))
spec = {
 "format": P.SPEC_FORMAT, "spec_version": 1, "date": "2026-11-27", "status": "frozen at R2 submission; changes require a new version",
 "roadmap": {"path": "mese3/02_virtual_piv/roadmap_tecnica_piv.md", "decision": "mese3/02_virtual_piv/decisione_R1.md", "commit_decision_R1": "05414c2"},
 "orientation": {"status": "probable, NOT confirmed", "components_observed": ["vi", "vj"], "vi": "transverse in-plane (probable)", "vj": "axial in-plane (probable)",
                 "vk": "out of plane, not used", "residual_uncertainty": "vi could be out of plane; sign of components and origin of coordinates unknown",
                 "oop_sensitivity": "SENS-OOP disabled"},
 "geometry": {"grid_hw": [128, 128], "steps_mm": {"rows": st["rows"], "cols": st["cols"]},
              "physical_convention_for_generator": "rows<->x (lateral, centred on the axis at row 63.5), cols<->y (axial, inlet at high columns)",
              "historical_convention_for_networks": "cols<->x, rows<->y: x_scaled/y_scaled, GLU coordinates and CanonicalSparseObservation.velocity_coords are UNCHANGED",
              "B_side_mm": 30.0},
 "normalization": {"stats_path": "data/manifests/train_norm_stats_month2_fit.json", "stats_sha256": C.STATS_SHA256_EXPECTED, "applied": "after the observation operators, frozen fit-only z-score"},
 "reference_velocity": {"definition": "p99.9 of |u_planar|=sqrt(vi^2+vj^2) over all nodes of frames 0:2001:10 (201 frames) of the raw record; velocity only",
                        "per_condition_m_s": vref_tab, "bootstrap95_blocks10": ci_tab, "source_table": "results/piv/phase2/f20/vref_full_record.csv",
                        "test_conditions": "not computed; only at the authorised opening (R5)"},
 "error": {"sigma_px": {"main": 0.10, "lo": 0.06, "hi": 0.15}, "d_max_px": 6.0, "distribution": "gaussian", "unit": "m/s",
           "formula": "sigma_u,c = sigma_px * v_ref,c / d_max, uniform over the field, per in-plane component"},
 "kernels": kernels,
 "kernel_definitions": {"mhg": "F(s) = (1 - b s^2/sig^2) exp(-s^2/2sig^2)/((1-b) sig sqrt(2 pi)), b=0.35; discrete weights = integral of F over the LES cell, clipped to +-4 sig, renormalised",
                        "tophat": "weights = overlap of [-W_eff/2, W_eff/2] with the cell, / W_eff", "lsr_discrete": "step / sum(w^2)",
                        "calibration": "param tuned per axis so that the DISCRETE L_sr equals the target (response 0.93 W, noise 1.2 W, top-hat W); tolerance 1 %; failure raises instead of forcing"},
 "windows_B": {"rule": "30 x 30 mm; B-in: rows centred on the axis (63.5), columns adjacent to the inlet (last column); B-down: same rows, adjacent window towards lower columns; B-off: columns as B-in, rows starting at row 64",
               "primary_steps": {f: list(P.field_rect(f, (st["rows"], st["cols"]))) for f in ("B-in", "B-down", "B-off")},
               "swap_steps": {f: list(P.field_rect(f, (st["cols"], st["rows"]))) for f in ("B-in", "B-down", "B-off")},
               "format": "[row0, row1, col0, col1] inclusive", "train_position": "B-in"},
 "configs": configs,
 "rejection_sensitivity": {"fraction": 0.10, "levels": {"flag": "rejected vectors absent (U-Net Mv=0; GLU token removed)", "sub": "rejected vectors replaced by the iterative 3x3 mean and presented as measured (Mv=1)"},
                           "applied_to": "PIV-A only", "same_positions_in_both_levels": True},
 "operation_order": ["LES velocity in m/s (frozen statistics)", "response convolution limited to the field, renormalised at the edge", "valid-vector selection (window rule)",
                      "correlated error", "rejections (SENS-REJ only)", "substitution (level sub only)", "frozen normalisation", "U-Net / GLU packet"],
 "no_temperature": "no generator argument, helper or test fixture reads T; temperature diagnostics are separate scripts",
 "seeds": {"namespace": P.NAMESPACE, "role_seeds": dict(P.ROLE_SEEDS),
           "key": "stable_seed(namespace, role_seed, operator, component, sim_id, frame_index, replica_or_epoch) via zlib.crc32; model family and model seed are NOT part of the key",
           "training": "one new error realisation per (frame, epoch)", "evaluation": "3 replicas", "excluded_historical": sorted(P.HISTORICAL_AND_MISSION00_SEEDS)},
 "parameter_classification": params,
 "sources": {"STO12": "Stohr, Boxx, Carter, Meier, Combust. Flame 159 (2012), DLR elib 77484", "WIE17": "Wieneke, TU Delft thesis 2017",
             "BOX10": "Boxx et al., Combust. Flame 157 (2010)", "BAU15": "Baumgartner, TU Munchen thesis 2015", "OBE15": "Oberleithner et al., Combust. Flame 162 (2015)",
             "SAI16": "Saini, Steinberg, Arndt, AIAA 2016-0436", "KAR20": "Karmarkar et al., arXiv:2011.14662", "SINT": "mese3/02_virtual_piv/sintesi_evidenze_modello_piv.md"},
 "provenance": {"phase2_f20_code_sha": f20_prov["code_sha"], "phase1_report_commit": "a1cb3a07ade3df624549b4d261ab815a6121f13d",
                "f20_files": {p.name: C.sha256_file(p) for p in sorted(f20.glob("*.csv")) + sorted(f20.glob("*.json")) if "provenance" not in p.name}},
}
spec["frozen"] = {"content_sha256": P.spec_content_hash(spec), "note": "sha256 of the canonical JSON of every key except 'frozen'; the file sha256 is stored in piv_observation_spec_v1.sha256"}
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=140))
sha = hashlib.sha256(out.read_bytes()).hexdigest()
(out.with_suffix(".sha256")).write_text(f"{sha}  {out.name}\n")
P.load_piv_spec(out, expected_sha256=sha)  # round trip
print("spec written", out, sha, "configs:", len(configs))
