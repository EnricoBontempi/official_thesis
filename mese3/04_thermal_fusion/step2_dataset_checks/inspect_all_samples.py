#!/usr/bin/env python3
"""
Reproducible inspection of representative experimental thermal dataset samples
for Mission 04 — Step 2 (Experimental Thermal Datasets).

Usage:
    python3 inspect_all_samples.py \
        --samples-dir /path/to/samples \
        --out-json inspection_summary.json

Dependencies:
    Python 3.10+, numpy (standard library xml/zipfile/csv/re/struct used for
    CSV/TXT/XLSX/ZIP/DOC); scipy.io is used for MATLAB v5 .mat files
    (CAM.94523 and Katurji).

Original sample files are opened strictly read-only and never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def r4(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        if not math.isfinite(float(x)):
            return str(x)
        return round(float(x), 4)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def read_xlsx_sheet_rows(path: Path, sheet_name: str | None = None) -> dict[str, list[dict[str, str | None]]]:
    """Pure-Python XLSX reader returning {sheet_name: [ {col_letter: value}, ... ]}."""
    out: dict[str, list[dict[str, str | None]]] = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                if si.tag.endswith("}si"):
                    shared.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        for s in wb.iter():
            if not s.tag.endswith("}sheet"):
                continue
            sname = s.attrib["name"]
            if sheet_name is not None and sname != sheet_name:
                continue
            rid = [v for k, v in s.attrib.items() if k.endswith("}id") or k == "r:id"][0]
            target = rmap[rid].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            if target not in names:
                continue
            sroot = ET.fromstring(z.read(target))
            rows: list[dict[str, str | None]] = []
            for row in sroot.iter():
                if not row.tag.endswith("}row"):
                    continue
                rmap_row: dict[str, str | None] = {}
                for c in row:
                    if not c.tag.endswith("}c"):
                        continue
                    ref = c.attrib.get("r", "")
                    col = re.sub(r"\d+", "", ref)
                    ctype = c.attrib.get("t", "")
                    v_el = next((ch for ch in c if ch.tag.endswith("}v")), None)
                    is_el = next((ch for ch in c if ch.tag.endswith("}is")), None)
                    val: str | None = None
                    if ctype == "s" and v_el is not None and v_el.text is not None:
                        idx = int(v_el.text)
                        val = shared[idx] if idx < len(shared) else v_el.text
                    elif ctype == "inlineStr" and is_el is not None:
                        val = "".join(t.text or "" for t in is_el.iter() if t.tag.endswith("}t"))
                    elif v_el is not None:
                        val = v_el.text
                    rmap_row[col] = val
                rows.append(rmap_row)
            out[sname] = rows
    return out


def parse_fds_csv(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Parse FDS/NIST 2-header-row CSV (names, units) into float matrix (NaN on blank/non-numeric)."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = [h.strip() for h in next(reader)]
        units = [u.strip() for u in next(reader)]
        rows = []
        for row in reader:
            if not row or all(x.strip() == "" for x in row):
                continue
            vals = []
            for i in range(len(headers)):
                s = row[i].strip() if i < len(row) else ""
                try:
                    vals.append(float(s))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(vals)
    return headers, units, np.asarray(rows, dtype=np.float64)


def check_time_vector(t: np.ndarray) -> dict[str, Any]:
    fin = np.isfinite(t)
    t_fin = t[fin]
    if t_fin.size < 2:
        return {"n": int(t.size), "n_finite": int(t_fin.size)}
    dt = np.diff(t_fin)
    med = float(np.median(dt))
    return {
        "n_rows": int(t.size),
        "n_nonfinite": int((~fin).sum()),
        "t_min_s": r4(t_fin[0]),
        "t_max_s": r4(t_fin[-1]),
        "duration_s": r4(t_fin[-1] - t_fin[0]),
        "dt_median_s": r4(med),
        "dt_min_s": r4(np.min(dt)),
        "dt_max_s": r4(np.max(dt)),
        "n_nonpositive_dt": int((dt <= 0).sum()),
        "n_irregular_dt_gt1pct": int((np.abs(dt - med) > 0.01 * abs(med)).sum()) if med > 0 else 0,
    }


# ---------------------------------------------------------------------------
# 1. NIST RSE 2007 (Bare-Bead vs Aspirated Type-K + Co-located Velocity + NaNs)
# ---------------------------------------------------------------------------
def inspect_nist_rse_2007(samples: Path) -> dict[str, Any]:
    d = samples / "nist_rse_2007"
    pairs = [
        ("TD70LA", "TD70LBB", "upper_left_70cm"),
        ("TD70CA", "TD70CBB", "upper_center_70cm"),
        ("TD50CA", "TD50CBB", "mid_center_50cm"),
        ("TD30LA", "TD30LBB", "lower_left_30cm"),
        ("TD30CA", "TD30CBB", "lower_center_30cm"),
    ]
    res: dict[str, Any] = {}
    for fn in ["NIST_RSE_2007_01.csv", "NIST_RSE_2007_02.csv", "NIST_RSE_2007_12.csv"]:
        hdr, units, arr = parse_fds_csv(d / fn)
        idx = {h: i for i, h in enumerate(hdr)}
        t = arr[:, idx["Time"]]
        nan_cols = {hdr[j]: int(np.isnan(arr[:, j]).sum()) for j in range(arr.shape[1]) if np.isnan(arr[:, j]).any()}
        pair_stats = {}
        for asp_c, bb_c, loc in pairs:
            if asp_c not in idx or bb_c not in idx:
                continue
            ta = arr[:, idx[asp_c]]
            tb = arr[:, idx[bb_c]]
            valid = np.isfinite(ta) & np.isfinite(tb)
            if not valid.any():
                pair_stats[loc] = {"asp": asp_c, "bb": bb_c, "n_valid": 0, "note": "all_nan_one_or_both"}
                continue
            diff = tb[valid] - ta[valid]  # bare_bead - aspirated (K or deg C)
            tv = t[valid]
            tav = ta[valid]
            # Pre-ignition / ambient baseline: t <= 10 s and T_asp < 40 C
            pre = (tv <= 10.0) & (tav < 40.0)
            # Active burn window: t >= 100 s and t <= (t_max - 30 s) where TD70CA > 200 C
            td70 = arr[:, idx["TD70CA"]] if "TD70CA" in idx else ta
            active = valid & (t >= 100.0) & (td70 > 200.0)
            diff_act = arr[active, idx[bb_c]] - arr[active, idx[asp_c]]
            ta_act = arr[active, idx[asp_c]]
            pair_stats[loc] = {
                "asp_col": asp_c,
                "bb_col": bb_c,
                "n_valid": int(valid.sum()),
                "n_total": int(len(t)),
                "pre_ign_n": int(pre.sum()),
                "pre_ign_mean_dT_K": r4(np.mean(diff[pre])) if pre.any() else None,
                "active_n": int(active.sum()),
                "active_T_asp_mean_C": r4(np.mean(ta_act)) if active.any() else None,
                "active_T_asp_max_C": r4(np.max(ta_act)) if active.any() else None,
                "active_dT_BB_minus_ASP_mean_K": r4(np.mean(diff_act)) if active.any() else None,
                "active_dT_BB_minus_ASP_std_K": r4(np.std(diff_act)) if active.any() else None,
                "active_dT_BB_minus_ASP_min_K": r4(np.min(diff_act)) if active.any() else None,
                "active_dT_BB_minus_ASP_max_K": r4(np.max(diff_act)) if active.any() else None,
            }
        # Check co-located doorway velocity probes
        vel_cols = [h for h in hdr if h.startswith("VD")]
        vel_summary = {}
        for vc in vel_cols:
            v = arr[:, idx[vc]]
            vf = v[np.isfinite(v)]
            vel_summary[vc] = {
                "unit": units[idx[vc]],
                "n_valid": int(vf.size),
                "min": r4(np.min(vf)) if vf.size else None,
                "max": r4(np.max(vf)) if vf.size else None,
            }
        res[fn] = {
            "n_cols": len(hdr),
            "time": check_time_vector(t),
            "nan_columns": nan_cols,
            "bare_vs_aspirated_pairs": pair_stats,
            "velocity_columns": vel_summary,
        }
    return res


# ---------------------------------------------------------------------------
# 2. NIST/NRC Compartment (70-ch Trees, Sentinels 999.0->0.0, Faulty 3320 C, Tree 4-9)
# ---------------------------------------------------------------------------
def inspect_nist_nrc(samples: Path) -> dict[str, Any]:
    d = samples / "nist_nrc"
    res: dict[str, Any] = {}
    for fn in ["NIST_NRC_Test_01_file_1.csv", "NIST_NRC_Test_01_file_2.csv", "NIST_NRC_Test_04_file_1.csv"]:
        # NIST_NRC CSVs have 1 header row (no separate units row); row 1 is t = -60 s!
        with open(d / fn, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            hdr = [h.strip() for h in next(reader)]
            rows = []
            for row in reader:
                if not row or all(x.strip() == "" for x in row):
                    continue
                vals = []
                for i in range(len(hdr)):
                    s = row[i].strip() if i < len(row) else ""
                    try:
                        vals.append(float(s))
                    except ValueError:
                        vals.append(float("nan"))
                rows.append(vals)
        arr = np.asarray(rows, dtype=np.float64)
        idx = {h: i for i, h in enumerate(hdr)}
        t = arr[:, idx["Time"]]
        n_rows = arr.shape[0]
        sentinel_999 = {}
        extreme_gt1500 = {}
        nan_cols = {}
        zero_after_999 = {}
        for j, h in enumerate(hdr):
            if j == 0:
                continue
            col = arr[:, j]
            n_nan = int(np.isnan(col).sum())
            if n_nan:
                nan_cols[h] = n_nan
            n_999 = int((np.abs(col - 999.0) < 1e-6).sum())
            if n_999:
                t_999 = t[np.abs(col - 999.0) < 1e-6]
                n_zero = int((col == 0.0).sum())
                sentinel_999[h] = {
                    "n_eq_999": n_999,
                    "n_eq_0": n_zero,
                    "t_first_999_s": r4(t_999[0]),
                    "t_last_999_s": r4(t_999[-1]),
                }
                if n_zero > 0:
                    zero_after_999[h] = {
                        "n_999": n_999,
                        "n_0": n_zero,
                        "t_first_0_s": r4(t[col == 0.0][0]),
                    }
            fin = col[np.isfinite(col)]
            if fin.size and np.max(fin) > 1500.0 and ("TC" in h or "Tree" in h or "ATC" in h or "Temp" in h):
                extreme_gt1500[h] = {
                    "min_C": r4(np.min(fin)),
                    "max_C": r4(np.max(fin)),
                    "n_gt1500": int((fin > 1500.0).sum()),
                    "n_total": n_rows,
                }
        t49 = None
        if "Tree 4-9" in idx:
            c = arr[:, idx["Tree 4-9"]]
            cf = c[np.isfinite(c)]
            t49 = {
                "val_at_t_minus60s": r4(c[0]),
                "val_at_t_minus50s": r4(c[1]),
                "n_eq_999": int((c == 999.0).sum()),
                "n_eq_0": int((c == 0.0).sum()),
                "neighbor_Tree_4_8_max_C": r4(np.nanmax(arr[:, idx["Tree 4-8"]])),
                "neighbor_Tree_4_10_max_C": r4(np.nanmax(arr[:, idx["Tree 4-10"]])),
            }
        atc_summary = {}
        for h in hdr:
            if h.startswith("ATC"):
                c = arr[:, idx[h]]
                cf = c[np.isfinite(c)]
                atc_summary[h] = {"min_C": r4(np.min(cf)), "max_C": r4(np.max(cf)), "n_eq_999": int((c == 999.0).sum())}
        res[fn] = {
            "n_rows": n_rows,
            "n_cols": len(hdr),
            "time": check_time_vector(t),
            "n_cols_with_999_sentinel": len(sentinel_999),
            "n_cols_transitioning_999_to_0": len(zero_after_999),
            "sentinel_999_examples": dict(list(sentinel_999.items())[:12]),
            "extreme_T_gt1500C": extreme_gt1500,
            "n_nan_cols": len(nan_cols),
            "nan_cols": nan_cols,
            "tree_4_9_check": t49,
            "atc_channels": atc_summary,
        }
    return res


# ---------------------------------------------------------------------------
# 3. NIST Pool Fires (Uncorrected TC/TC_RMS vs Rad+Inertia Corrected TG/TG_RMS)
# ---------------------------------------------------------------------------
def inspect_nist_pool_fires(samples: Path) -> dict[str, Any]:
    d = samples / "nist_pool_fires"
    # Known bead diameters from NIST README Table 4.1 (Sung et al. 2021, 2024)
    bead_map = {
        "Propane_37_cm_50_kW_TC_r=0_cm_Sung_2024.csv": "39 um",
        "Methanol_30_cm_TC_r=0_cm_Sung_2024.csv": "52 um",
        "Methane_37_cm_TC_r=0_cm_Sung_2024.csv": "52 um & 119 um",
        "Ethanol_30_cm_TC_r=0_cm_Sung_2024.csv": "125 um & 199 um",
        "Methanol_100_cm_TC_r=0_cm_Sung_2021.csv": "153 um",
        "Methanol_100_cm_TC_z=21_cm_Sung_2021.csv": "153 um",
        "Methanol_100_cm_TC_z=61_cm_Sung_2021.csv": "153 um",
        "Methanol_100_cm_TC_z=101_cm_Sung_2021.csv": "153 um",
    }
    res: dict[str, Any] = {"profiles": {}}
    for fn, bead_desc in bead_map.items():
        p = d / fn
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        hdr = [h.strip() for h in rows[0]]
        # Detect whether row 1 is a units row (Sung_2021) or already numeric data (Sung_2024)
        try:
            float(rows[1][0])
            has_units_row = False
        except (ValueError, IndexError):
            has_units_row = True
        units = [u.strip() for u in rows[1]] if has_units_row else [""] * len(hdr)
        data_rows = [r for r in rows[2 if has_units_row else 1:] if r and any(x.strip() for x in r)]
        idx = {h: i for i, h in enumerate(hdr)}

        def find_col(*candidates: str) -> str | None:
            for c in candidates:
                if c in idx:
                    return c
            return None

        def col_arr(name: str | None) -> np.ndarray | None:
            if name is None:
                return None
            j = idx[name]
            vals = []
            for r in data_rows:
                try:
                    vals.append(float(r[j]))
                except (ValueError, IndexError):
                    vals.append(float("nan"))
            return np.asarray(vals, dtype=np.float64)

        c_tc = find_col("TC", "TC_K")
        c_tg = find_col("TG", "TG_K")
        c_tcr = find_col("TC_RMS", "TC_RMS_K")
        c_tgr = find_col("TG_RMS", "TG_RMS_K")
        c_uc = find_col("UC_TG", "Uc_TG_K")
        c_db = find_col("D_bead", "D_bead_um", "Bead_D_um")

        tc = col_arr(c_tc)
        tg = col_arr(c_tg)
        tc_rms = col_arr(c_tcr)
        tg_rms = col_arr(c_tgr)
        uc_tg = col_arr(c_uc)
        d_bead = col_arr(c_db)

        unit_tc = units[idx[c_tc]] if has_units_row and c_tc else ("K" if c_tc and c_tc.endswith("_K") else "unknown")
        dt_rad = tg - tc
        valid_m = np.isfinite(dt_rad)
        info: dict[str, Any] = {
            "n_points": len(data_rows),
            "headers": hdr,
            "has_separate_units_row": has_units_row,
            "bead_readme": bead_desc,
            "TC_unit": unit_tc,
            "TC_min": r4(np.nanmin(tc)),
            "TC_max": r4(np.nanmax(tc)),
            "TG_max": r4(np.nanmax(tg)),
            "dT_TG_minus_TC_min_K": r4(np.nanmin(dt_rad[valid_m])) if valid_m.any() else None,
            "dT_TG_minus_TC_max_K": r4(np.nanmax(dt_rad[valid_m])) if valid_m.any() else None,
            "dT_TG_minus_TC_mean_K": r4(np.nanmean(dt_rad[valid_m])) if valid_m.any() else None,
        }
        if tc_rms is not None and tg_rms is not None:
            v = np.isfinite(tc_rms) & np.isfinite(tg_rms) & (tc_rms > 0)
            ratio = tg_rms[v] / tc_rms[v]
            info.update({
                "TC_RMS_max_K": r4(np.nanmax(tc_rms)),
                "TG_RMS_max_K": r4(np.nanmax(tg_rms)),
                "RMS_ratio_TG_over_TC_min": r4(np.min(ratio)) if ratio.size else None,
                "RMS_ratio_TG_over_TC_median": r4(np.median(ratio)) if ratio.size else None,
                "RMS_ratio_TG_over_TC_max": r4(np.max(ratio)) if ratio.size else None,
            })
        if uc_tg is not None and np.isfinite(uc_tg).any():
            info["UC_TG_min"] = r4(np.nanmin(uc_tg))
            info["UC_TG_max"] = r4(np.nanmax(uc_tg))
        if d_bead is not None and np.isfinite(d_bead).any():
            by_bead = {}
            for bd in sorted(set(d_bead[np.isfinite(d_bead)])):
                m = (d_bead == bd)
                v = m & np.isfinite(tc_rms) & np.isfinite(tg_rms) & (tc_rms > 0)
                by_bead[f"{bd:.0f}um"] = {
                    "n": int(m.sum()),
                    "dT_mean_K": r4(np.nanmean(dt_rad[m])),
                    "dT_max_K": r4(np.nanmax(dt_rad[m])),
                    "RMS_ratio_median": r4(np.median(tg_rms[v] / tc_rms[v])) if v.any() else None,
                    "RMS_ratio_max": r4(np.max(tg_rms[v] / tc_rms[v])) if v.any() else None,
                }
            info["by_bead_diameter"] = by_bead
        res["profiles"][fn] = info

    # Also check Acetone liquid-phase time series (1-row header!)
    with open(d / "Acetone_30_cm_TC_Sung_2024.csv", newline="", encoding="utf-8") as f:
        ac_rows = list(csv.reader(f))
    ac_hdr = [h.strip() for h in ac_rows[0]]
    ac_arr = np.asarray([[float(x) for x in r[:len(ac_hdr)]] for r in ac_rows[1:] if r and r[0].strip()], dtype=np.float64)
    res["acetone_liquid_timeseries"] = {
        "n_rows": int(ac_arr.shape[0]),
        "n_cols": len(ac_hdr),
        "headers": ac_hdr[:12],
        "time": check_time_vector(ac_arr[:, 0]),
        "n_nan_total": int(np.isnan(ac_arr).sum()),
        "T_min_max_K": [r4(np.nanmin(ac_arr[:, 1:])), r4(np.nanmax(ac_arr[:, 1:]))],
    }
    # Companion centerline velocity check (columns: z, r, U, Uc_U)
    u_hdr, u_units, u_arr = parse_fds_csv(d / "Methane_37_cm_U_r=0_Sung_2024.csv")
    u_idx = u_hdr.index("U")
    res["methane_velocity_r0"] = {
        "n_rows": int(u_arr.shape[0]),
        "headers": u_hdr,
        "units": u_units,
        "U_min_ms": r4(np.nanmin(u_arr[:, u_idx])),
        "U_max_ms": r4(np.nanmax(u_arr[:, u_idx])),
    }
    return res


# ---------------------------------------------------------------------------
# 4. HYLON Phase 2 (TNF Jan 2026: Double-bead TC vs 1D Raman + LIP Walls + PIV)
# ---------------------------------------------------------------------------
def inspect_hylon_phase2(samples: Path) -> dict[str, Any]:
    zp = samples / "tnf_hylon_phase2" / "HYLON_TNF_2026_01_Phase2-1.zip"
    res: dict[str, Any] = {"flames": {}}

    def parse_semicolon_csv(raw_bytes: bytes) -> tuple[list[str], np.ndarray]:
        lines = [l.strip() for l in raw_bytes.decode("utf-8", errors="replace").splitlines() if l.strip()]
        # Skip leading # comments
        data_lines = [l for l in lines if not l.startswith("#")]
        delim = ";" if ";" in data_lines[0] else ","
        hdr = [h.strip() for h in data_lines[0].split(delim) if h.strip() != ""]
        rows = []
        for l in data_lines[1:]:
            parts = [p.strip() for p in l.split(delim)]
            vals = []
            for j in range(len(hdr)):
                try:
                    vals.append(float(parts[j].replace(",", ".")) if j < len(parts) and parts[j] != "" else float("nan"))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(vals)
        return hdr, np.asarray(rows, dtype=np.float64)

    def parse_wall_txt(raw_bytes: bytes) -> dict[str, Any]:
        lines = [l.strip() for l in raw_bytes.decode("utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("#")]
        # Skip 2 header lines (names, units)
        pts = []
        for l in lines[2:]:
            parts = re.split(r"\s+", l)
            if len(parts) >= 3:
                try:
                    pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
        arr = np.asarray(pts, dtype=np.float64)
        return {
            "n_points": int(len(pts)),
            "coord_min_mm": r4(np.min(arr[:, 0])) if len(pts) else None,
            "coord_max_mm": r4(np.max(arr[:, 0])) if len(pts) else None,
            "T_min_K": r4(np.min(arr[:, 1])) if len(pts) else None,
            "T_max_K": r4(np.max(arr[:, 1])) if len(pts) else None,
            "unc_min_K": r4(np.min(arr[:, 2])) if len(pts) else None,
            "unc_max_K": r4(np.max(arr[:, 2])) if len(pts) else None,
            "rel_unc_pct_median": r4(100.0 * np.median(arr[:, 2] / arr[:, 1])) if len(pts) else None,
        }

    with zipfile.ZipFile(zp) as z:
        all_names = [n for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n and not n.endswith(".DS_Store") and "/._" not in n]
        res["total_real_files"] = len(all_names)
        for flame, tc_dir, fl_char in [
            ("FLAME_A1", "A1_T_flow_thermocouple", "A"),
            ("FLAME_L1", "L1_T_thermcouple", "L"),
        ]:
            fprefix = f"HYLON_TNF_2026_01_Phase2-1/{flame}/"
            short = flame.split("_")[1]
            # 1) TC files
            tc_files = sorted([n for n in all_names if n.startswith(fprefix + tc_dir + "/") and n.endswith(".csv")])
            raman_files = sorted([n for n in all_names if n.startswith(fprefix + f"{short}_Major_species_T_NO/") and n.endswith(".csv")])
            raman_by_z = {}
            for rf in raman_files:
                m = re.search(r"z\+([0-9.]+)\.csv$", rf)
                if m:
                    z_mm = round(float(m.group(1)) * 1000.0, 2)
                    rh, rarr = parse_semicolon_csv(z.read(rf))
                    ridx = {h: i for i, h in enumerate(rh)}
                    raman_by_z[z_mm] = (rh, rarr, ridx)

            tc_summary = {}
            overlap_comparisons = []
            replicate_diffs = []
            for tf in tc_files:
                m = re.search(r"z\+([0-9.]+)\.csv$", tf)
                z_mm = round(float(m.group(1)) * 1000.0, 2) if m else None
                th, tarr = parse_semicolon_csv(z.read(tf))
                # columns: x; T; error_T; error_x
                x = tarr[:, 0]
                T = tarr[:, 1]
                err_T = tarr[:, 2]
                err_x = tarr[:, 3]
                # Check replicates (same x)
                by_x: dict[float, list[float]] = {}
                for xi, Ti in zip(x, T):
                    if np.isfinite(xi) and np.isfinite(Ti):
                        by_x.setdefault(float(xi), []).append(float(Ti))
                reps = []
                for xi, vals in sorted(by_x.items()):
                    if len(vals) >= 2:
                        diff_rep = abs(vals[0] - vals[1])
                        reps.append({"x_mm": xi, "vals_K": [r4(v) for v in vals], "abs_diff_K": r4(diff_rep)})
                        replicate_diffs.append(diff_rep)
                # Check overlap with Raman at same z_mm
                ov_pts = []
                if z_mm in raman_by_z:
                    rh, rarr, ridx = raman_by_z[z_mm]
                    rx = rarr[:, ridx["x"]]
                    rT = rarr[:, ridx["Tpg"]]
                    rTrms = rarr[:, ridx["Tpg_rms"]]
                    rx_min, rx_max = float(np.nanmin(rx)), float(np.nanmax(rx))
                    for xi, vals in sorted(by_x.items()):
                        if rx_min <= xi <= rx_max:
                            j_near = int(np.argmin(np.abs(rx - xi)))
                            if abs(rx[j_near] - xi) <= 0.5:
                                t_tc = float(np.mean(vals))
                                t_ram = float(rT[j_near])
                                ov_pts.append({
                                    "z_mm": z_mm,
                                    "x_mm": xi,
                                    "T_TC_K": r4(t_tc),
                                    "T_Raman_K": r4(t_ram),
                                    "T_Raman_rms_K": r4(rTrms[j_near]),
                                    "dT_TC_minus_Raman_K": r4(t_tc - t_ram),
                                    "rel_diff_pct": r4(100.0 * (t_tc - t_ram) / t_ram),
                                })
                                overlap_comparisons.append(ov_pts[-1])
                tc_summary[f"z={z_mm}mm"] = {
                    "n_rows": int(tarr.shape[0]),
                    "n_unique_x": len(by_x),
                    "x_min_mm": r4(np.nanmin(x)),
                    "x_max_mm": r4(np.nanmax(x)),
                    "T_min_K": r4(np.nanmin(T)),
                    "T_max_K": r4(np.nanmax(T)),
                    "rel_err_T_pct": r4(100.0 * np.nanmedian(err_T / T)),
                    "err_x_mm": r4(np.nanmedian(err_x)),
                    "replicates": reps,
                    "overlap_with_raman": ov_pts,
                }
            # 2) Raman summary
            raman_summary = {}
            for z_mm, (rh, rarr, ridx) in sorted(raman_by_z.items()):
                rx = rarr[:, ridx["x"]]
                rT = rarr[:, ridx["Tpg"]]
                rTrms = rarr[:, ridx["Tpg_rms"]]
                raman_summary[f"z={z_mm}mm"] = {
                    "n_points": int(rarr.shape[0]),
                    "x_min_mm": r4(np.nanmin(rx)),
                    "x_max_mm": r4(np.nanmax(rx)),
                    "dx_median_mm": r4(np.median(np.abs(np.diff(rx)))),
                    "T_min_K": r4(np.nanmin(rT)),
                    "T_max_K": r4(np.nanmax(rT)),
                    "T_rms_max_K": r4(np.nanmax(rTrms)),
                    "n_nan_T": int(np.isnan(rT).sum()),
                }
            # 3) Wall LIP temps
            walls = {}
            for wf in sorted([n for n in all_names if n.startswith(fprefix + f"{short}_Wall_Temp/") and n.endswith(".txt")]):
                walls[Path(wf).name] = parse_wall_txt(z.read(wf))
            # 4) Hardware swap PIV comparison (2Steel2Quartz vs 4Quartz at z=12.7 and z=22.7 mm)
            piv_swap = {}
            for z_str in ["0.0127", "0.0227"]:
                f_2s2q = fprefix + f"{short}_Flow_velocity/{short}_Flow_velocity_2Steel2Quartz/{short}_Reactive_z+{z_str}.csv"
                f_4q = fprefix + f"{short}_Flow_velocity/{short}_Flow_velocity_4Quartz/{short}_Reactive_z+{z_str}.csv"
                if f_2s2q in all_names and f_4q in all_names:
                    h1, a1 = parse_semicolon_csv(z.read(f_2s2q))
                    h2, a2 = parse_semicolon_csv(z.read(f_4q))
                    # Interpolate U_z on overlapping x range
                    uz1_col = h1.index("U_z") if "U_z" in h1 else 3
                    uz2_col = h2.index("U_z") if "U_z" in h2 else 3
                    s1 = np.argsort(a1[:, 0])
                    s2 = np.argsort(a2[:, 0])
                    x_lo = max(float(np.nanmin(a1[:, 0])), float(np.nanmin(a2[:, 0])))
                    x_hi = min(float(np.nanmax(a1[:, 0])), float(np.nanmax(a2[:, 0])))
                    x_grid = np.linspace(x_lo, x_hi, 101)
                    u1_i = np.interp(x_grid, a1[s1, 0], a1[s1, uz1_col])
                    u2_i = np.interp(x_grid, a2[s2, 0], a2[s2, uz2_col])
                    piv_swap[f"z={float(z_str)*1000:.1f}mm"] = {
                        "headers_2s2q": h1,
                        "headers_4q": h2,
                        "n_2s2q": int(a1.shape[0]),
                        "n_4q": int(a2.shape[0]),
                        "x_overlap_mm": [r4(x_lo), r4(x_hi)],
                        "Uz_max_abs_diff_ms": r4(np.max(np.abs(u1_i - u2_i))),
                        "Uz_rms_diff_ms": r4(np.sqrt(np.mean((u1_i - u2_i) ** 2))),
                        "Uz_peak_2s2q_ms": r4(np.nanmax(a1[:, uz1_col])),
                        "Uz_peak_4q_ms": r4(np.nanmax(a2[:, uz2_col])),
                    }

            res["flames"][flame] = {
                "thermocouple_profiles": tc_summary,
                "n_tc_replicates": len(replicate_diffs),
                "tc_replicate_abs_diff_median_K": r4(np.median(replicate_diffs)) if replicate_diffs else None,
                "tc_replicate_abs_diff_max_K": r4(np.max(replicate_diffs)) if replicate_diffs else None,
                "raman_profiles": raman_summary,
                "tc_vs_raman_overlap_count": len(overlap_comparisons),
                "tc_vs_raman_overlap_points": overlap_comparisons,
                "lip_wall_temperatures": walls,
                "piv_hardware_swap_2s2q_vs_4q": piv_swap,
            }
    return res


# ---------------------------------------------------------------------------
# 5. Cambridge CAM.94523 (High-Pressure Swirl Burner LIGS — declared vs actual TC!)
# ---------------------------------------------------------------------------
def inspect_cam_94523(samples: Path) -> dict[str, Any]:
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    d = samples / "cam_94523_ligs_tc"
    fd = sio.loadmat(d / "FigureData.mat", squeeze_me=False)
    keys = sorted([k for k in fd.keys() if not k.startswith("__")])
    tc_like_keys = [k for k in keys if any(w in k.lower() for w in ["tc", "thermo", "couple", "probe"])]

    def summarize_profile(arr: np.ndarray) -> dict[str, Any]:
        # cols: r_mm, T_K, norm_hist_weight
        rs = sorted(set(arr[:, 0]))
        stations = []
        for rv in rs:
            sub = arr[arr[:, 0] == rv]
            t_bins = sub[:, 1]
            w = sub[:, 2]
            wsum = float(np.sum(w))
            t_mean = float(np.sum(t_bins * w) / wsum) if wsum > 0 else float("nan")
            t_std = float(np.sqrt(np.sum(w * (t_bins - t_mean) ** 2) / wsum)) if wsum > 0 else float("nan")
            t_mode = float(t_bins[int(np.argmax(w))]) if wsum > 0 else float("nan")
            stations.append({
                "r_mm": r4(rv),
                "n_bins": int(sub.shape[0]),
                "T_weighted_mean_K": r4(t_mean),
                "T_weighted_std_K": r4(t_std),
                "T_mode_K": r4(t_mode),
            })
        return {"n_rows": int(arr.shape[0]), "n_radial_stations": len(rs), "stations": stations}

    shot_dropouts = {}
    for fn in ["Signalss_thermal.mat", "Signalss_thermal2.mat", "Signalss_thermal3.mat", "Signalss_thermal4.mat"]:
        ms = sio.loadmat(d / fn, squeeze_me=False)["signalss"]
        lens = [x.size for x in ms.flat]
        n_total = len(lens)
        n_dropped = sum(1 for l in lens if l <= 1)
        valid_lens = sorted(set(l for l in lens if l > 1))
        shot_dropouts[fn] = {
            "n_shots": n_total,
            "n_dropped_scalar_zero": n_dropped,
            "dropout_pct": r4(100.0 * n_dropped / n_total),
            "valid_trace_len": valid_lens,
        }

    return {
        "n_variables_in_FigureData": len(keys),
        "thermocouple_named_variables_found": tc_like_keys,
        "has_thermocouple_data_in_deposit": False,
        "mean_vals_1_to_4": [r4(fd[f"mean_val_{i}"].flat[0]) for i in range(1, 5)],
        "std_vals_1_to_4": [r4(fd[f"std_val_{i}"].flat[0]) for i in range(1, 5)],
        "temperature30mm": summarize_profile(fd["temperature30mm"]),
        "temperature50mm": summarize_profile(fd["temperature50mm"]),
        "raw_ligs_shot_dropouts": shot_dropouts,
    }


# ---------------------------------------------------------------------------
# 6. NIST Smyth Wolfhard-Parker Slot Burner (Symmetrized vs Rad-Corrected Original + LDV)
# ---------------------------------------------------------------------------
def inspect_smyth_slot_burner(samples: Path) -> dict[str, Any]:
    d = samples / "smyth_slot_burner"

    def parse_smyth_file(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        txt = path.read_text(encoding="utf-8", errors="replace")
        blocks: dict[float, tuple[list[float], list[float]]] = {}
        cur_h: float | None = None
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            m = re.search(r"(?:height|HAB|z)\s*[:=]?\s*([0-9.]+)\s*mm", s, flags=re.I)
            if m:
                cur_h = float(m.group(1))
                blocks[cur_h] = ([], [])
                continue
            # Also handle tabular multi-column layout!
            parts = s.split()
            if cur_h is not None and len(parts) >= 2:
                try:
                    blocks[cur_h][0].append(float(parts[0]))
                    blocks[cur_h][1].append(float(parts[1]))
                except ValueError:
                    pass
        if blocks:
            return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in blocks.items()}
        # Fallback: inspect multi-column structure directly
        return {}

    raw_p = d / "Temperature-thermocouple.txt"
    orig_p = d / "Temperature-thermocouple-radiation-corrected-original.txt"
    v_p = d / "Vertical-velocity-LDV.txt"
    h_p = d / "Horizontal-Velocity-LDV.txt"

    def inspect_table(path: Path) -> dict[str, Any]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # Find header and numeric rows
        num_rows = []
        hdr_lines = []
        for l in lines:
            toks = l.strip().split()
            if not toks:
                continue
            try:
                floats = [float(x) for x in toks]
                num_rows.append(floats)
            except ValueError:
                if not num_rows:
                    hdr_lines.append(l.strip())
        # Check column counts
        col_counts = sorted(set(len(r) for r in num_rows))
        arr = np.asarray([r for r in num_rows if len(r) == col_counts[-1]], dtype=np.float64) if num_rows else np.empty((0, 0))
        out = {
            "n_lines": len(lines),
            "header_lines": hdr_lines[:10],
            "n_numeric_rows": len(num_rows),
            "col_counts": col_counts,
        }
        if arr.size:
            x = arr[:, 0]
            out["x_min"] = r4(np.min(x))
            out["x_max"] = r4(np.max(x))
            out["val_min"] = r4(np.min(arr[:, 1:]))
            out["val_max"] = r4(np.max(arr[:, 1:]))
            # Check exact symmetry x -> -x if x spans negative and positive
            sym_diffs = []
            for i in range(len(x)):
                j = int(np.argmin(np.abs(x + x[i])))
                if abs(x[j] + x[i]) < 1e-6 and x[i] > 0:
                    sym_diffs.append(float(np.max(np.abs(arr[i, 1:] - arr[j, 1:]))))
            out["max_left_right_asymmetry"] = r4(np.max(sym_diffs)) if sym_diffs else None
            out["median_left_right_asymmetry"] = r4(np.median(sym_diffs)) if sym_diffs else None
        return out

    t_sym = inspect_table(raw_p)
    t_orig = inspect_table(orig_p)
    v_ldv = inspect_table(v_p)
    h_ldv = inspect_table(h_p)

    # Also compute direct point-by-point diff between Temperature-thermocouple.txt and original if same columns
    # Let's parse both numeric arrays
    def load_num(path: Path) -> np.ndarray:
        rows = []
        for l in path.read_text(encoding="utf-8", errors="replace").splitlines():
            toks = l.strip().split()
            if not toks:
                continue
            try:
                rows.append([float(x) for x in toks])
            except ValueError:
                pass
        max_c = max(len(r) for r in rows)
        return np.asarray([r for r in rows if len(r) == max_c], dtype=np.float64)

    a_sym = load_num(raw_p)
    a_orig = load_num(orig_p)
    comp = {}
    n_ov = min(a_sym.shape[1], a_orig.shape[1]) - 1
    if n_ov >= 1:
        # Compare first n_ov overlapping height columns (H=3, 5, 7, 9, 11 mm)
        peaks_sym = [r4(np.max(a_sym[:, j])) for j in range(1, n_ov + 1)]
        peaks_orig = [r4(np.max(a_orig[:, j])) for j in range(1, n_ov + 1)]
        comp = {
            "n_height_cols_sym": int(a_sym.shape[1] - 1),
            "n_height_cols_orig": int(a_orig.shape[1] - 1),
            "n_overlapping_heights": int(n_ov),
            "peaks_symmetrized_K": peaks_sym,
            "peaks_original_K": peaks_orig,
            "peak_diff_orig_minus_sym_K": [r4(o - s) for o, s in zip(peaks_orig, peaks_sym)],
            "max_peak_diff_K": r4(np.max(np.abs(np.asarray(peaks_sym) - np.asarray(peaks_orig)))),
        }
        # Compare on overlapping x grid (interpolating orig onto sym x)
        diffs_all = []
        for j in range(1, n_ov + 1):
            y_orig_i = np.interp(a_sym[:, 0], a_orig[:, 0], a_orig[:, j])
            diffs_all.append(a_sym[:, j] - y_orig_i)
        da = np.concatenate(diffs_all)
        comp["interp_diff_sym_minus_orig_mean_K"] = r4(np.mean(da))
        comp["interp_diff_sym_minus_orig_max_abs_K"] = r4(np.max(np.abs(da)))
        comp["interp_diff_sym_minus_orig_rms_K"] = r4(np.sqrt(np.mean(da ** 2)))

    return {
        "Temperature_thermocouple_txt": t_sym,
        "Temperature_rad_corrected_original_txt": t_orig,
        "Vertical_velocity_LDV_txt": v_ldv,
        "Horizontal_Velocity_LDV_txt": h_ldv,
        "sym_vs_original_comparison": comp,
    }


# ---------------------------------------------------------------------------
# 7. ULB 20 kW Furnace (Air-Cooled Suction Pyrometer Type-B TC — README vs XLSX!)
# ---------------------------------------------------------------------------
def inspect_ulb_furnace(samples: Path) -> dict[str, Any]:
    p = samples / "ulb_furnace" / "Flame_Temperatures_data_V4.xlsx"
    sheets = read_xlsx_sheet_rows(p)
    data = sheets["data"]
    hdr = data[0]
    cols_present = sorted(set(k for row in data for k in row.keys()))
    c6h6_set, phi_set, x_set, y_set = set(), set(), set(), set()
    T_vals = []
    n_nonfinite = 0
    bad_examples = []
    by_key: dict[tuple[float, float, float, float], list[float]] = {}
    combos: dict[tuple[float, float], int] = {}
    for row in data[1:]:
        try:
            a, b, c, d_y, e = float(row["A"]), float(row["B"]), float(row["C"]), float(row["D"]), float(row["E"])
            if not all(math.isfinite(v) for v in (a, b, c, d_y, e)):
                raise ValueError("nonfinite")
            c6h6_set.add(a)
            phi_set.add(b)
            x_set.add(c)
            y_set.add(d_y)
            T_vals.append(e)
            by_key.setdefault((a, b, c, d_y), []).append(e)
            combos[(a, b)] = combos.get((a, b), 0) + 1
        except (KeyError, TypeError, ValueError):
            n_nonfinite += 1
            if len(bad_examples) < 5:
                bad_examples.append(row)
    T_arr = np.asarray(T_vals, dtype=np.float64)
    counts = list(combos.values())
    # Replicate repeatability across identical (C6H6, phi, x, y) keys
    rep_spans = [max(v) - min(v) for v in by_key.values() if len(v) >= 2]
    rep_stds = [float(np.std(v, ddof=1)) for v in by_key.values() if len(v) >= 2]
    return {
        "sheet_names": list(sheets.keys()),
        "data_header_row": hdr,
        "all_columns_found_in_data_sheet": cols_present,
        "units_sheet": sheets.get("units", []),
        "n_data_rows": len(data) - 1,
        "n_valid_5col_rows": int(T_arr.size),
        "n_missing_coord_or_meta_rows": n_nonfinite,
        "bad_row_examples": bad_examples,
        "unique_C6H6_pct": sorted(c6h6_set),
        "unique_phi": sorted(phi_set),
        "n_operating_conditions": len(combos),
        "unique_x_mm": sorted(x_set),
        "unique_y_mm": sorted(y_set),
        "rows_per_condition_min": min(counts) if counts else None,
        "rows_per_condition_max": max(counts) if counts else None,
        "n_unique_condition_coord_keys": len(by_key),
        "n_keys_with_replicates": len(rep_spans),
        "replicate_span_median_K": r4(np.median(rep_spans)) if rep_spans else None,
        "replicate_span_p95_K": r4(np.percentile(rep_spans, 95)) if rep_spans else None,
        "replicate_span_max_K": r4(np.max(rep_spans)) if rep_spans else None,
        "replicate_std_median_K": r4(np.median(rep_stds)) if rep_stds else None,
        "T_min_K": r4(np.min(T_arr)),
        "T_median_K": r4(np.median(T_arr)),
        "T_max_K": r4(np.max(T_arr)),
        "declared_vs_actual_discrepancies": [
            "README_V4.txt states 'expressed in degrees Celsius', whereas sheet 'data' header is 'T_mean [K]' and values span 1145-1927 (Kelvin), matching sheet 'units'.",
            "README_V4.txt states 'standard deviations... are also reported for all the measured temperatures' and sheet 'units' lists T_std/T_median/T_iqr, whereas sheet 'data' contains ONLY 5 columns (A-E: C6H6, phi, x, y, T_mean [K]) and NO T_std column.",
            f"Data contains {len(data)-1} rows including {n_nonfinite} rows with missing/non-numeric metadata cells and {len(rep_spans)} duplicated (C6H6, phi, x, y) entries (likely repeated runs/dates collapsed without the 'data (yyyy-mm-dd)' column listed in sheet 'units').",
        ],
    }


# ---------------------------------------------------------------------------
# 8. Cardiff NH3 Swirl (31141438 positive + 30234904 & 27277428 negative checks)
# ---------------------------------------------------------------------------
def inspect_cardiff(samples: Path) -> dict[str, Any]:
    d = samples / "cardiff_nh3_swirl"
    # Positive: Radiation_Supplementary.xlsx (10.17035/cardiff.31141438)
    rad = read_xlsx_sheet_rows(d / "Radiation_Supplementary.xlsx")["Data"]
    phis, tex_nh3, tex_cr, tin_nh3, tin_cr, tout_nh3, tout_cr = [], [], [], [], [], [], []
    n_blank_nh3_em = 0
    for row in rad[4:]:
        if "B" not in row or row["B"] is None:
            continue
        phis.append(float(row["B"]))
        tex_nh3.append(float(row["C"]))
        tex_cr.append(float(row["D"]))
        tin_nh3.append(float(row["E"]))
        tin_cr.append(float(row["F"]))
        tout_nh3.append(float(row["G"]))
        tout_cr.append(float(row["H"]))
        if row.get("K") is None:
            n_blank_nh3_em += 1
    # Negative check 1: COGNH3_Supplementary.xlsx (10.17035/cardiff.30234904)
    cog = read_xlsx_sheet_rows(d / "COGNH3_Supplementary.xlsx")["Data"]
    cog_headers_r2 = {k: v for k, v in cog[1].items() if v is not None} if len(cog) > 1 else {}
    cog_headers_r4 = {k: v for k, v in cog[3].items() if v is not None} if len(cog) > 3 else {}
    # Negative check 2: d.2022.0217184646.zip (Mashruk Energy 2022)
    mash_sheets = []
    with zipfile.ZipFile(d / "d.2022.0217184646.zip") as z:
        mash_files = z.namelist()
        for fn in mash_files:
            if fn.endswith(".xlsx"):
                tmp = d / "_tmp_check_mash.xlsx"
                tmp.write_bytes(z.read(fn))
                mash_sheets = list(read_xlsx_sheet_rows(tmp).keys())
                tmp.unlink()
    return {
        "cardiff_31141438_radiation": {
            "n_conditions": len(phis),
            "phi_range": [min(phis), max(phis)],
            "T_exhaust_NH3_H2_K": [r4(min(tex_nh3)), r4(max(tex_nh3))],
            "T_exhaust_cracked_NH3_K": [r4(min(tex_cr)), r4(max(tex_cr))],
            "T_inner_wall_NH3_H2_K": [r4(min(tin_nh3)), r4(max(tin_nh3))],
            "T_outer_wall_NH3_H2_K": [r4(min(tout_nh3)), r4(max(tout_nh3))],
            "wall_deltaT_inner_minus_outer_NH3_H2_K": [
                r4(min(a - b for a, b in zip(tin_nh3, tout_nh3))),
                r4(max(a - b for a, b in zip(tin_nh3, tout_nh3))),
            ],
            "n_blank_NH3_emission_cells_out_of_9": n_blank_nh3_em,
            "has_rms_or_uncertainty_cols": False,
        },
        "cardiff_30234904_cognh3_negative_check": {
            "row2_group_headers": cog_headers_r2,
            "row4_unit_headers_sample": dict(list(cog_headers_r4.items())[:12]),
            "contains_temperature_data": any("temp" in str(v).lower() or str(v).strip() in ("K", "C", "°C") for v in list(cog_headers_r2.values()) + list(cog_headers_r4.values())),
        },
        "cardiff_27277428_mashruk2022_negative_check": {
            "zip_files": mash_files,
            "xlsx_sheets": mash_sheets,
            "contains_temperature_sheet": any("temp" in s.lower() or "tc" in s.lower() for s in mash_sheets),
        },
    }


# ---------------------------------------------------------------------------
# 9. Katurji 50 Hz In-Fire Fine-Wire TC Array (EE1 & EE3 — Dropouts & Dynamics)
# ---------------------------------------------------------------------------
def inspect_katurji_50hz(samples: Path) -> dict[str, Any]:
    if not HAS_SCIPY:
        return {"error": "scipy not available"}
    d = samples / "katurji_50hz_tc"
    res = {}
    for fn, key in [
        ("TC_Array_50Hz_left_front_EE1.mat", "TC_1to32_matrix_50Hz"),
        ("TC_Array_50Hz_left_front_EE1.mat", "TbC_1to23_matrix_50Hz"),
        ("TC_Array_50Hz_left_front_EE3.mat", "TC_1to32_matrix"),
    ]:
        mat = sio.loadmat(d / fn, squeeze_me=False)[key]
        n_t, n_ch = mat.shape
        nan_per_ch = np.isnan(mat).sum(axis=0)
        dead_chs = [int(i + 1) for i in range(n_ch) if nan_per_ch[i] == n_t]
        partial_chs = {int(i + 1): int(nan_per_ch[i]) for i in range(n_ch) if 0 < nan_per_ch[i] < n_t}
        # For partial channels, locate exact first/last NaN index
        partial_detail = {}
        for ch1, cnt in partial_chs.items():
            nan_idx = np.where(np.isnan(mat[:, ch1 - 1]))[0]
            # Count contiguous NaN runs
            runs = int(1 + np.sum(np.diff(nan_idx) > 1)) if nan_idx.size else 0
            partial_detail[f"ch{ch1}"] = {
                "n_nan": cnt,
                "pct_nan": r4(100.0 * cnt / n_t),
                "first_nan_idx": int(nan_idx[0]),
                "last_nan_idx": int(nan_idx[-1]),
                "n_contiguous_nan_runs": runs,
            }
        # Dynamic signal properties on valid samples (without calling turbulence noise!)
        fin = mat[np.isfinite(mat)]
        diffs = np.diff(mat, axis=0)
        diffs_fin = diffs[np.isfinite(diffs)]
        # Max single-step jump (at 50 Hz = 20 ms)
        res[f"{fn}::{key}"] = {
            "shape_T_by_ch": [n_t, n_ch],
            "duration_s_at_50Hz": r4((n_t - 1) / 50.0),
            "total_cells": int(mat.size),
            "total_nan": int(np.isnan(mat).sum()),
            "nan_pct": r4(100.0 * np.isnan(mat).sum() / mat.size),
            "completely_dead_channels_1based": dead_chs,
            "partial_dropout_channels": partial_detail,
            "T_min_C": r4(np.min(fin)),
            "T_median_C": r4(np.median(fin)),
            "T_max_C": r4(np.max(fin)),
            "abs_dT_step_20ms_p50_K": r4(np.percentile(np.abs(diffs_fin), 50)),
            "abs_dT_step_20ms_p99_K": r4(np.percentile(np.abs(diffs_fin), 99)),
            "abs_dT_step_20ms_max_K": r4(np.max(np.abs(diffs_fin))),
            "n_exact_zero_steps_valid": int((diffs_fin == 0.0).sum()),
            "pct_exact_zero_steps_valid": r4(100.0 * (diffs_fin == 0.0).sum() / diffs_fin.size),
            "n_negative_values": int((fin < 0).sum()),
            "n_below_10C": int((fin < 10.0).sum()),
        }
    return res


# ---------------------------------------------------------------------------
# 10. Remaining datasets: Wang MWIR vs TC, Waterloo, FM Burner, UMD, Pelzmann,
#     MetForTC Drift, Brzosko 12-TC Grid, Sydney SM1 + Surface_Temp_BB, Sandia pmCDEF
# ---------------------------------------------------------------------------
def inspect_wang_mwir_tc(samples: Path) -> dict[str, Any]:
    zp = samples / "wang_mwir_tc" / "data_IJTS_R1.zip"
    with zipfile.ZipFile(zp) as z:
        c2 = list(csv.DictReader(z.read("campaign2_station_data.csv").decode("utf-8").splitlines()))
        rad = list(csv.DictReader(z.read("analysis_tc_radiation_error.csv").decode("utf-8").splitlines()))
    direct_0deg = [r for r in c2 if r["viewing_angle_deg"] == "0"]
    diffs = []
    for r in direct_0deg:
        mwir = float(r["mean_K"])
        tc = float(r["tc_reference_K"])
        diffs.append({
            "nozzle": r["nozzle"],
            "state": r["engine_state"],
            "MWIR_mean_K": mwir,
            "TC_ref_K": tc,
            "dT_MWIR_minus_TC_K": r4(mwir - tc),
            "declared_dev_pct": float(r["mean_deviation_pct"]),
        })
    return {
        "n_campaign2_rows": len(c2),
        "campaign2_0deg_vs_TC": diffs,
        "radiation_model_table_rows": len(rad),
        "radiation_model_h_range_W_m2_K": [float(rad[0]["h_W_m-2_K-1"]), float(rad[-1]["h_W_m-2_K-1"])],
        "note": "analysis_tc_radiation_error.csv is an analytical parametric radiation-bias sensitivity table, not raw TC measurements.",
    }


def inspect_waterloo_fm_umd(samples: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Waterloo: check upstream composite files (known zero-newline corruption!) + parse clean z=10 cm slice
    wm_comp = (samples / "waterloo_methanol" / "Mean_and_rms_values.csv").read_bytes()
    wc_comp = (samples / "waterloo_methanol" / "Estimated_correlations.csv").read_bytes()
    wm_z10_path = samples / "waterloo_methanol" / "Mean_and_rms_values_z10cm.csv"
    wc_z10_path = samples / "waterloo_methanol" / "Estimated_correlations_z10cm.csv"
    with open(wm_z10_path, newline="", encoding="utf-8") as f:
        wm_rows = list(csv.DictReader(f))
    with open(wc_z10_path, newline="", encoding="utf-8") as f:
        wc_rows = list(csv.DictReader(f))
    tbar = np.asarray([float(r["T-bar (K)"]) for r in wm_rows])
    trms = np.asarray([float(r["T-rms (K)"]) for r in wm_rows])
    ut = np.asarray([float(r["u'T' (m/s)K"]) for r in wc_rows])
    vt = np.asarray([float(r["v'T' (m/s)K"]) for r in wc_rows])
    r_vals = [float(r["r (cm)"]) for r in wm_rows]
    out["waterloo_methanol"] = {
        "composite_Mean_and_rms_bytes_LF_CR": [len(wm_comp), wm_comp.count(b"\n"), wm_comp.count(b"\r")],
        "composite_Estimated_correlations_bytes_LF_CR": [len(wc_comp), wc_comp.count(b"\n"), wc_comp.count(b"\r")],
        "packaging_anomaly": "Upstream composite CSVs in MaCFP/macfp-db have 0 LF and 0 CR bytes (all rows concatenated on 1 line); per-height slice CSVs (e.g. *_z10cm.csv) have valid LF delimiters.",
        "z10cm_n_radial_points": len(wm_rows),
        "z10cm_r_cm": r_vals,
        "z10cm_T_bar_min_max_K": [r4(np.min(tbar)), r4(np.max(tbar))],
        "z10cm_T_rms_min_max_K": [r4(np.min(trms)), r4(np.max(trms))],
        "z10cm_uT_min_max_mK_s": [r4(np.min(ut)), r4(np.max(ut))],
        "z10cm_vT_min_max_mK_s": [r4(np.min(vt)), r4(np.max(vt))],
        "surface_temp_file": (samples / "waterloo_methanol" / "Methanol_Surface_Temperature.csv").read_text().strip(),
    }

    # FM Burner (Oct 2025)
    fm_d = samples / "fm_burner"
    with open(fm_d / "temperature_mean.csv", encoding="utf-8") as f:
        fm_rows = list(csv.reader(f))
    fm_hdr = [h.strip() for h in fm_rows[0]]
    n_blank_mean = sum(1 for r in fm_rows[2:] for cell in r if cell.strip() == "")
    n_total_mean = sum(len(r) for r in fm_rows[2:])
    # Check PDF normalization on temperature_PDF_OC20.9_1.5D.csv
    with open(fm_d / "temperature_PDF_OC20.9_1.5D.csv", encoding="utf-8") as f:
        pdf_rows = list(csv.reader(f))
    pdf_hdr = [h.strip() for h in pdf_rows[0]]
    pdf_num = []
    for r in pdf_rows[1:]:
        vals = []
        for x in r:
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(float("nan"))
        if any(math.isfinite(v) for v in vals):
            pdf_num.append(vals)
    pdf_arr = np.asarray(pdf_num, dtype=np.float64)
    t_bins = pdf_arr[:, 0]
    dt_bin = float(np.nanmedian(np.diff(t_bins)))
    # Integrate first few radial PDF columns
    integrals = [r4(np.nansum(pdf_arr[:, j]) * dt_bin) for j in range(1, min(6, pdf_arr.shape[1]))]
    out["fm_burner_oct2025"] = {
        "temperature_mean_rows_excl_hdr": len(fm_rows) - 2,
        "temperature_mean_cols": len(fm_hdr),
        "temperature_mean_blank_cells": n_blank_mean,
        "temperature_mean_total_cells": n_total_mean,
        "temperature_mean_blank_pct": r4(100.0 * n_blank_mean / max(1, n_total_mean)),
        "pdf_1.5D_OC20.9_shape": list(pdf_arr.shape),
        "pdf_T_min_max_K": [r4(np.nanmin(t_bins)), r4(np.nanmax(t_bins))],
        "pdf_dT_bin_K": r4(dt_bin),
        "pdf_first5_radial_integrals": integrals,
        "pdf_cols_sample": pdf_hdr[:8],
    }

    # UMD Line Burner (1-row headers!)
    umd_d = samples / "umd_line_burner"
    def parse_1hdr(path: Path) -> tuple[list[str], np.ndarray]:
        with open(path, newline="", encoding="utf-8") as f:
            r = list(csv.reader(f))
        h = [x.strip() for x in r[0]]
        arr = np.asarray([[float(x) if x.strip() else float("nan") for x in row[:len(h)]] for row in r[1:] if row and any(c.strip() for c in row)], dtype=np.float64)
        return h, arr

    tcz_h, tcz_a = parse_1hdr(umd_d / "TCz_Data.csv")
    tcx_h, tcx_a = parse_1hdr(umd_d / "TCx_Data.csv")
    tc0_h, tc0_a = parse_1hdr(umd_d / "TC_Data.csv")
    out["umd_line_burner"] = {
        "TCz_rows": int(tcz_a.shape[0]),
        "TCz_headers": tcz_h,
        "TCz_T_min_max_K": [r4(np.nanmin(tcz_a[:, 1])), r4(np.nanmax(tcz_a[:, 1]))],
        "TCz_rms_max_K": r4(np.nanmax(tcz_a[:, 2])),
        "TCx_rows": int(tcx_a.shape[0]),
        "TCx_headers": tcx_h,
        "TCx_T_min_max_K": [r4(np.nanmin(tcx_a[:, 1:5])), r4(np.nanmax(tcx_a[:, 1:5]))],
        "TCx_n_nan": int(np.isnan(tcx_a).sum()),
        "TC_Data_rows": int(tc0_a.shape[0]),
        "TC_Data_headers": tc0_h,
        "TC_Data_T_min_max": [r4(np.nanmin(tc0_a[:, 2:])), r4(np.nanmax(tc0_a[:, 2:]))],
    }
    return out


def inspect_pelzmann_metfortc_brzosko(samples: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Pelzmann: check SBG file & TPH log backwards timestamp jump
    pd_dir = samples / "pelzmann_mckenna"
    with open(pd_dir / "20200511_SBG_FFC_condition01.csv", encoding="utf-8", errors="replace") as f:
        c1_rows = list(csv.reader(f))
    tph_lines = (pd_dir / "20201104_TPH_log.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    # Parse HH:MM:SS to seconds
    secs = []
    for l in tph_lines[1:]:
        parts = l.split()
        if len(parts) >= 5:
            hh, mm, ss = [int(x) for x in parts[1].split(":")]
            secs.append(hh * 3600 + mm * 60 + ss)
    dt_tph = np.diff(secs)
    out["pelzmann_mckenna"] = {
        "condition01_header": [x for x in c1_rows[0] if x != ""],
        "condition01_rows": len(c1_rows) - 1,
        "condition01_is_heatflux_SBG_not_TC": True,
        "tph_log_rows": len(secs),
        "tph_log_n_negative_dt_jumps": int((dt_tph < 0).sum()),
        "tph_log_n_zero_dt": int((dt_tph == 0).sum()),
        "tph_log_dt_min_max_s": [int(np.min(dt_tph)), int(np.max(dt_tph))],
        "tph_log_first_jump_row": int(np.where(dt_tph < 0)[0][0] + 2) if (dt_tph < 0).any() else None,
    }

    # MetForTC Drift (Zenodo 8189490)
    mf = read_xlsx_sheet_rows(samples / "metfortc_drift" / "MetForTC_Drift_TDMF_2023.xlsx")
    mf_sum = {}
    for sname, rows in mf.items():
        # Extract numeric voltages & temperatures where present
        # Inspect header row
        hdr_idx = next((i for i, r in enumerate(rows[:10]) if any("Voltage" in str(v) or "Temperature" in str(v) for v in r.values())), 0)
        hdr_row = rows[hdr_idx]
        # Find voltage / temp / std columns
        v_col = next((k for k, v in hdr_row.items() if v and "Voltage" in v), None)
        t_col = next((k for k, v in hdr_row.items() if v and "Temperature" in v), None)
        sd_col = next((k for k, v in hdr_row.items() if v and "StandDev V" in v), None)
        t_vals, v_vals, sd_vals = [], [], []
        if sname == "2022 K":
            # Comma-packed single-column CSV strings inside col A!
            for r in rows[6:]:
                s = r.get("A") or ""
                parts = s.split(",")
                if len(parts) >= 6:
                    try:
                        v_vals.append(float(parts[1]))
                        sd_vals.append(float(parts[2]))
                        t_vals.append(float(parts[5]))
                    except ValueError:
                        pass
        else:
            for r in rows[hdr_idx + 1:]:
                if v_col and r.get(v_col) is not None:
                    try:
                        v_vals.append(float(r[v_col]))
                    except ValueError:
                        pass
                if t_col and r.get(t_col) is not None:
                    try:
                        t_vals.append(float(r[t_col]))
                    except ValueError:
                        pass
                if sd_col and r.get(sd_col) is not None:
                    try:
                        sd_vals.append(float(r[sd_col]))
                    except ValueError:
                        pass
        mf_sum[sname] = {
            "total_rows": len(rows),
            "header_row": hdr_row,
            "n_voltage_vals": len(v_vals),
            "n_temp_vals": len(t_vals),
            "temp_min_max_C": [r4(min(t_vals)), r4(max(t_vals))] if t_vals else None,
            "voltage_min_max_mV": [r4(min(v_vals)), r4(max(v_vals))] if v_vals else None,
            "standdev_V_median": r4(np.median(sd_vals)) if sd_vals else None,
            "standdev_V_max": r4(np.max(sd_vals)) if sd_vals else None,
        }
    out["metfortc_drift"] = mf_sum

    # Brzosko 12-TC Grid (Zenodo 21371063)
    br = np.load(samples / "brzosko_tc_grid" / "experimental_pin_temperature_dataset.npz", allow_pickle=True)
    data = br["data"]  # (211, 532, 12)
    mask = br["mask"]  # (211, 532)
    valid_vals = data[mask == 1]
    masked_vals = data[mask == 0]
    seq_valid_lens = mask.sum(axis=1)
    # Check if masked entries are strictly zero-padded at tail
    all_masked_zero = bool(np.all(masked_vals == 0.0)) if masked_vals.size else True
    out["brzosko_tc_grid"] = {
        "shape": list(data.shape),
        "n_sequences": int(data.shape[0]),
        "max_steps": int(data.shape[1]),
        "n_channels": int(data.shape[2]),
        "feature_names": [str(x) for x in br["feature_names"]],
        "n_nan_total": int(np.isnan(data).sum()),
        "valid_step_len_min_max": [int(seq_valid_lens.min()), int(seq_valid_lens.max())],
        "masked_fraction_pct": r4(100.0 * (mask == 0).sum() / mask.size),
        "all_masked_cells_exact_zero": all_masked_zero,
        "valid_T_min_K": r4(np.min(valid_vals)),
        "valid_T_median_K": r4(np.median(valid_vals)),
        "valid_T_max_K": r4(np.max(valid_vals)),
        "n_exact_zero_in_valid_cells": int((valid_vals == 0.0).sum()),
    }
    return out


def inspect_sydney_and_sandia(samples: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # 1. Sydney Surface_Temp_BB.doc: extract all readable numeric table values (UTF-8/ASCII & UTF-16LE)
    raw = (samples / "sydney_thermofluids" / "Surface_Temp_BB.doc").read_bytes()
    # Extract full clean text block from Word 97-2003 stream
    ascii_blocks = [m.group(0).decode("latin1") for m in re.finditer(rb"[\x20-\x7e\r\n\t\x07]{15,}", raw)]
    bb_text = "\n".join(b.replace("\x07", " | ").replace("\r", "\n") for b in ascii_blocks if any(w in b for w in ["Ceramic", "HM1", "HM2", "HM3", "Mikron", "Bluff-Body"]))
    out["sydney_surface_temp_bb_doc"] = {
        "instrument_confirmed": "Mikron M90-D infrared pyrometer camera (500-1500 C, filters flame radiation) — NOT a thermocouple!",
        "extracted_text_excerpt": bb_text[:1800],
    }

    # 2. Sydney SM1.zip (Sandia Raman/Rayleigh/LIF single-shot scatter in Sydney Swirl SM1)
    with zipfile.ZipFile(samples / "sydney_thermofluids" / "SM1.zip") as z:
        sm1_files = [n for n in z.namelist() if not n.endswith("/")]
        # Parse SM1_10.YAL and SM1_40.YAL (or first & middle)
        sm1_stats = {}
        for fn in sm1_files:
            lines = z.read(fn).decode("utf-8", errors="replace").splitlines()
            # Find header containing T(K)
            hi = next((i for i, l in enumerate(lines[:10]) if "T(K)" in l), None)
            if hi is None:
                continue
            hdr = lines[hi].split()
            ti = hdr.index("T(K)")
            ri = hdr.index("r(mm)") if "r(mm)" in hdr else -1
            tdi = hdr.index("TDNR") if "TDNR" in hdr else -1
            Ts, rs, tds = [], [], []
            for l in lines[hi + 1:]:
                p = l.split()
                if len(p) >= len(hdr):
                    try:
                        Ts.append(float(p[ti]))
                        if ri >= 0:
                            rs.append(float(p[ri]))
                        if tdi >= 0:
                            tds.append(float(p[tdi]))
                    except ValueError:
                        pass
            Tarr = np.asarray(Ts)
            sm1_stats[Path(fn).name] = {
                "n_shots": int(Tarr.size),
                "n_radial_locs": len(set(rs)) if rs else None,
                "T_min_K": r4(np.min(Tarr)),
                "T_mean_K": r4(np.mean(Tarr)),
                "T_max_K": r4(np.max(Tarr)),
                "n_T_lt_250K": int((Tarr < 250.0).sum()),
                "TDNR_median": r4(np.median(tds)) if tds else None,
            }
        out["sydney_sm1_zip"] = {"n_files": len(sm1_files), "files": sm1_stats}

    # 3. Sandia pmCDEF.zip (Flame D single-shot D15.Yall + radial stats D15.Yave)
    with zipfile.ZipFile(samples / "tnf_sandia_def" / "pmCDEF.zip") as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        d_yall = [n for n in names if n.endswith("D15.Yall")]
        d_yave = [n for n in names if n.endswith("D15.Yave")]
        s_info: dict[str, Any] = {"n_files": len(names)}
        if d_yall:
            lines = z.read(d_yall[0]).decode("utf-8", errors="replace").splitlines()
            hi = next((i for i, l in enumerate(lines[:10]) if "T(K)" in l or "T" in l.split()), 0)
            hdr = lines[hi].split()
            ti = next((i for i, h in enumerate(hdr) if h in ("T(K)", "T")), 1)
            Ts = []
            for l in lines[hi + 1:]:
                p = l.split()
                if len(p) > ti:
                    try:
                        Ts.append(float(p[ti]))
                    except ValueError:
                        pass
            Tarr = np.asarray(Ts)
            s_info["D15_Yall"] = {
                "path": d_yall[0],
                "headers": hdr,
                "n_shots": int(Tarr.size),
                "T_min_K": r4(np.min(Tarr)),
                "T_mean_K": r4(np.mean(Tarr)),
                "T_max_K": r4(np.max(Tarr)),
                "n_nonfinite": int((~np.isfinite(Tarr)).sum()),
            }
        if d_yave:
            lines = z.read(d_yave[0]).decode("utf-8", errors="replace").splitlines()
            s_info["D15_Yave_head"] = lines[:8]
        out["sandia_pmcdef_zip"] = s_info
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect representative experimental thermal dataset samples.")
    ap.add_argument("--samples-dir", type=Path, required=True, help="Directory containing downloaded sample folders.")
    ap.add_argument("--out-json", type=Path, required=True, help="Output path for inspection summary JSON.")
    args = ap.parse_args()

    samples = args.samples_dir
    summary = {
        "nist_rse_2007": inspect_nist_rse_2007(samples),
        "nist_nrc": inspect_nist_nrc(samples),
        "nist_pool_fires": inspect_nist_pool_fires(samples),
        "hylon_phase2": inspect_hylon_phase2(samples),
        "cam_94523_ligs_tc": inspect_cam_94523(samples),
        "smyth_slot_burner": inspect_smyth_slot_burner(samples),
        "ulb_furnace": inspect_ulb_furnace(samples),
        "cardiff_nh3_swirl": inspect_cardiff(samples),
        "katurji_50hz_tc": inspect_katurji_50hz(samples),
        "wang_mwir_tc": inspect_wang_mwir_tc(samples),
        "waterloo_fm_umd": inspect_waterloo_fm_umd(samples),
        "pelzmann_metfortc_brzosko": inspect_pelzmann_metfortc_brzosko(samples),
        "sydney_and_sandia": inspect_sydney_and_sandia(samples),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_json} ({args.out_json.stat().st_size} B)")


if __name__ == "__main__":
    main()
