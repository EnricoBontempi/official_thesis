#!/usr/bin/env python3
"""
Generate 4 verifiable diagnostic figures directly from inspected experimental
thermal dataset samples (Mission 04 — Step 2).

Usage:
    python3 generate_verification_figures.py \
        --samples-dir /path/to/samples \
        --out-dir ./figures
"""
from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_fds_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        hdr = [h.strip() for h in next(reader)]
        _units = next(reader)
        rows = []
        for r in reader:
            if not r or all(x.strip() == "" for x in r):
                continue
            vals = []
            for j in range(len(hdr)):
                try:
                    vals.append(float(r[j]))
                except (ValueError, IndexError):
                    vals.append(float("nan"))
            rows.append(vals)
    return hdr, np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Figure 1: NIST RSE 2007 — Bare-Bead vs Aspirated Type-K Radiation Bias
# ---------------------------------------------------------------------------
def make_fig1_nist_rse2007(samples: Path, out_dir: Path) -> None:
    d = samples / "nist_rse_2007"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    # Panel A: Test 01 (NG 190 kW) time series of dT = T_BB - T_ASP in upper vs lower layer
    h1, a1 = parse_fds_csv(d / "NIST_RSE_2007_01.csv")
    i1 = {h: j for j, h in enumerate(h1)}
    t1 = a1[:, i1["Time"]]
    ax = axes[0]
    specs = [
        ("TD70CA", "TD70CBB", "Upper center 70 cm (hot layer)", "#c0392b"),
        ("TD70LA", "TD70LBB", "Upper left 70 cm (hot layer)", "#e67e22"),
        ("TD50CA", "TD50CBB", "Mid center 50 cm (interface)", "#2980b9"),
        ("TD30CA", "TD30CBB", "Lower center 30 cm (cool layer)", "#27ae60"),
    ]
    for asp, bb, lbl, col in specs:
        dt = a1[:, i1[bb]] - a1[:, i1[asp]]
        ax.plot(t1, dt, label=lbl, color=col, lw=1.4)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta T = T_{\mathrm{bare}} - T_{\mathrm{aspirated}}$ (K)")
    ax.set_title("(a) NIST RSE 2007 Test 01 (NG 190 kW): opposite-sign bias")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # Panel B: Scatter dT vs T_aspirated for Test 01 (NG) and Test 12 (Methanol 305 kW)
    h12, a12 = parse_fds_csv(d / "NIST_RSE_2007_12.csv")
    i12 = {h: j for j, h in enumerate(h12)}
    ax = axes[1]
    for (hdr, arr, idx, fuel_lbl, mk, alpha) in [
        (h1, a1, i1, "Test 01 NG (190 kW)", "o", 0.55),
        (h12, a12, i12, "Test 12 Methanol (305 kW)", "^", 0.55),
    ]:
        for asp, bb, z_lbl, col in [
            ("TD70CA", "TD70CBB", "70 cm", "#c0392b"),
            ("TD30CA", "TD30CBB", "30 cm", "#27ae60"),
        ]:
            ta = arr[:, idx[asp]]
            dt = arr[:, idx[bb]] - ta
            v = np.isfinite(ta) & np.isfinite(dt)
            ax.scatter(ta[v], dt[v], s=14, marker=mk, alpha=alpha, color=col,
                       label=f"{fuel_lbl}, {z_lbl} (n={int(v.sum())})")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"Double-shielded aspirated reference $T_{\mathrm{aspirated}}$ (°C)")
    ax.set_ylabel(r"$\Delta T = T_{\mathrm{bare}} - T_{\mathrm{aspirated}}$ (K)")
    ax.set_title("(b) Radiation bias vs. reference T (cooling hot / heating cool)")
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_nist_rse2007_radiation_bias.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Real Empirical Faults — NIST/NRC Sentinels + Katurji 50 Hz Dropouts
# ---------------------------------------------------------------------------
def make_fig2_faults(samples: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=160)

    # Panel A: NIST/NRC Test 01 file 1 real fault modes (1-header-row CSV!)
    with open(samples / "nist_nrc" / "NIST_NRC_Test_01_file_1.csv", newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        h = [c.strip() for c in next(rd)]
        rows = []
        for r in rd:
            if not r or not any(x.strip() for x in r):
                continue
            rows.append([float(r[j]) if j < len(r) and r[j].strip() else float("nan") for j in range(len(h))])
    a = np.asarray(rows, dtype=np.float64)
    idx = {c: j for j, c in enumerate(h)}
    t = a[:, idx["Time"]]
    ax = axes[0]
    ax.plot(t, a[:, idx["TC South U-3-1"]], color="#c0392b", lw=1.5, label="TC South U-3-1 (stuck 3320–3330 °C, open-circuit)")
    ax.plot(t, a[:, idx["TC South U-3-2"]], color="#27ae60", lw=1.4, label="TC South U-3-2 (adjacent valid wall TC)")
    ax.plot(t, a[:, idx["Tree 4-9"]], color="#8e44ad", lw=1.5, ls="--",
            label="Tree 4-9 (999.0 at t=-60 s -> 0.0 for t>=-50 s; faulty)")
    ax.plot(t, a[:, idx["Tree 4-8"]], color="#2980b9", lw=1.3, ls=":", label="Tree 4-8 (adjacent valid gas TC)")
    ax.scatter([t[0]], [a[0, idx["Tree 4-9"]]], color="#8e44ad", s=45, zorder=5)
    ax.annotate("999.0 sentinel\nat t = -60 s", xy=(t[0], 999.0), xytext=(320, 1250),
                arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.1), fontsize=8, color="#8e44ad")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Recorded temperature (°C)")
    ax.set_title("(a) NIST/NRC Test 01: open-circuit (3320 °C) & 999->0 sentinel")
    ax.legend(fontsize=7.4, loc="center right")
    ax.grid(alpha=0.3)

    # Panel B: Katurji 50 Hz EE1 32-channel NaN/valid availability heatmap
    ax = axes[1]
    if HAS_SCIPY:
        m1 = sio.loadmat(samples / "katurji_50hz_tc" / "TC_Array_50Hz_left_front_EE1.mat", squeeze_me=False)["TC_1to32_matrix_50Hz"]
        # 1 = finite, 0 = NaN
        avail = np.isfinite(m1).astype(float).T  # (32, 1607)
        t_s = np.arange(m1.shape[0]) / 50.0
        ax.imshow(avail, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[t_s[0], t_s[-1], 32.5, 0.5], interpolation="nearest")
        ax.set_xlabel("Time at 50 Hz (s)")
        ax.set_ylabel("Thermocouple channel (1–32)")
        n_nan = int(np.isnan(m1).sum())
        ax.set_title(f"(b) Katurji 50 Hz EE1: valid (green) vs NaN (red, {n_nan}/{m1.size}={100*n_nan/m1.size:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_nist_nrc_and_katurji_faults.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: NIST Pool Fires — Radiation Bias & Thermal-Inertia RMS Attenuation
# ---------------------------------------------------------------------------
def make_fig3_nist_pool(samples: Path, out_dir: Path) -> None:
    d = samples / "nist_pool_fires"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)

    cases = [
        ("Propane_37_cm_50_kW_TC_r=0_cm_Sung_2024.csv", "Propane 37 cm (39 µm)", 39, "#1f77b4", "o"),
        ("Methanol_30_cm_TC_r=0_cm_Sung_2024.csv", "Methanol 30 cm (52 µm)", 52, "#2ca02c", "s"),
        ("Methane_37_cm_TC_r=0_cm_Sung_2024.csv", "Methane 37 cm (52 / 119 µm)", None, "#ff7f0e", "^"),
        ("Ethanol_30_cm_TC_r=0_cm_Sung_2024.csv", "Ethanol 30 cm (125 / 199 µm)", None, "#9467bd", "D"),
        ("Methanol_100_cm_TC_r=0_cm_Sung_2021.csv", "Methanol 100 cm (153 µm)", 153, "#d62728", "v"),
    ]

    bead_x, ratio_y, bead_lbls = [], [], []
    ax0, ax1 = axes[0], axes[1]
    for fn, lbl, fixed_d, col, mk in cases:
        with open(d / fn, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        hdr = [h.strip() for h in rows[0]]
        try:
            float(rows[1][0])
            has_units_row = False
        except (ValueError, IndexError):
            has_units_row = True
        idx = {h: j for j, h in enumerate(hdr)}
        data = []
        for r in rows[2 if has_units_row else 1:]:
            if not r or not any(x.strip() for x in r):
                continue
            vals = []
            for j in range(len(hdr)):
                try:
                    vals.append(float(r[j]))
                except (ValueError, IndexError):
                    vals.append(float("nan"))
            data.append(vals)
        arr = np.asarray(data, dtype=np.float64)
        c_tc = "TC" if "TC" in idx else "TC_K"
        c_tg = "TG" if "TG" in idx else "TG_K"
        c_tcr = "TC_RMS" if "TC_RMS" in idx else "TC_RMS_K"
        c_tgr = "TG_RMS" if "TG_RMS" in idx else "TG_RMS_K"
        c_db = next((c for c in ("D_bead", "D_bead_um", "Bead_D_um") if c in idx), None)
        tc_K = arr[:, idx[c_tc]]
        tg_K = arr[:, idx[c_tg]]
        dt = tg_K - tc_K
        v = np.isfinite(tc_K) & np.isfinite(dt)
        ax0.scatter(tc_K[v], dt[v], color=col, marker=mk, s=36, alpha=0.85, label=lbl)

        if c_tcr in idx and c_tgr in idx:
            tcr = arr[:, idx[c_tcr]]
            tgr = arr[:, idx[c_tgr]]
            vr = np.isfinite(tcr) & np.isfinite(tgr) & (tcr > 0)
            ratios = tgr[vr] / tcr[vr]
            if c_db is not None:
                db = arr[vr, idx[c_db]]
            else:
                db = np.full(ratios.shape, float(fixed_d))
            ax1.scatter(db, ratios, color=col, marker=mk, s=36, alpha=0.85, label=lbl)

    ax0.set_xlabel(r"Uncorrected bead temperature $T_{\mathrm{C}}$ (K)")
    ax0.set_ylabel(r"Correction $T_{\mathrm{G}} - T_{\mathrm{C}}$ (K)")
    ax0.set_title("(a) NIST Pool Fires: mean radiation+inertia shift (TG - TC)")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(alpha=0.3)

    ax1.axhline(1.0, color="k", ls="--", lw=0.8)
    ax1.set_xlabel(r"Type-S thermocouple bead diameter $D_{\mathrm{bead}}$ (µm)")
    ax1.set_ylabel(r"RMS recovery factor $T_{\mathrm{G,RMS}} / T_{\mathrm{C,RMS}}$ (-)")
    ax1.set_title("(b) Thermal-inertia RMS attenuation vs. bead diameter")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_nist_pool_inertia_and_rad.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: HYLON Phase 2 (TNF Jan 2026) — Double-Bead TC vs 1D Raman + LIP Walls
# ---------------------------------------------------------------------------
def make_fig4_hylon2(samples: Path, out_dir: Path) -> None:
    zp = samples / "tnf_hylon_phase2" / "HYLON_TNF_2026_01_Phase2-1.zip"

    def read_semi(z: zipfile.ZipFile, name: str) -> tuple[list[str], np.ndarray]:
        lines = [l.strip() for l in z.read(name).decode("utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("#")]
        hdr = [h.strip() for h in lines[0].split(";") if h.strip() != ""]
        rows = []
        for l in lines[1:]:
            p = [x.strip() for x in l.split(";")]
            rows.append([float(p[j]) if j < len(p) and p[j] != "" else float("nan") for j in range(len(hdr))])
        return hdr, np.asarray(rows, dtype=np.float64)

    def read_wall(z: zipfile.ZipFile, name: str) -> np.ndarray:
        lines = [l.strip() for l in z.read(name).decode("utf-8", errors="replace").splitlines() if l.strip() and not l.startswith("#")]
        pts = []
        for l in lines[2:]:
            p = re.split(r"\s+", l)
            if len(p) >= 3:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
        return np.asarray(pts, dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=160)
    with zipfile.ZipFile(zp) as z:
        # Panel A: Flame A1 (attached) at z = 12.7 mm and z = 22.7 mm
        for ax, (flame, short, tc_dir, fl_ch, title) in zip(
            axes[:2],
            [
                ("FLAME_A1", "A1", "A1_T_flow_thermocouple", "A", "(a) HYLON2 Flame A.1 (anchored): TC vs 1D Raman"),
                ("FLAME_L1", "L1", "L1_T_thermcouple", "L", "(b) HYLON2 Flame L.1 (lifted): TC vs 1D Raman"),
            ],
        ):
            for z_str, col in [("0.0127", "#1f77b4"), ("0.0227", "#d62728")]:
                rh, ra = read_semi(z, f"HYLON_TNF_2026_01_Phase2-1/{flame}/{short}_Major_species_T_NO/{short}_Major_species_T_z+{z_str}.csv")
                ri = {h: j for j, h in enumerate(rh)}
                rx, rT, rR = ra[:, ri["x"]], ra[:, ri["Tpg"]], ra[:, ri["Tpg_rms"]]
                z_mm = float(z_str) * 1000.0
                ax.plot(rx, rT, color=col, lw=1.6, label=f"1D Raman z={z_mm:.1f} mm")
                ax.fill_between(rx, rT - rR, rT + rR, color=col, alpha=0.14)

                _, ta = read_semi(z, f"HYLON_TNF_2026_01_Phase2-1/{flame}/{tc_dir}/HYLON-KAUST_Isolated_Fla_{fl_ch}_EXP_z+{z_str}.csv")
                ax.errorbar(ta[:, 0], ta[:, 1], yerr=ta[:, 2], xerr=ta[:, 3], fmt="o",
                            color=col, ms=5, capsize=2.5, elinewidth=1.0,
                            label=f"Double-bead TC z={z_mm:.1f} mm (±10%, ±4 mm)")
            ax.set_xlabel("Radial position x (mm)")
            ax.set_ylabel("Gas temperature (K)")
            ax.set_title(title, fontsize=10)
            ax.set_ylim(250, 2350)
            ax.legend(fontsize=7.2, loc="best")
            ax.grid(alpha=0.3)

        # Panel C: LIP inner wall temperatures (quartz vs metal) for Flame A1 & L1
        ax = axes[2]
        for lbl, fn, col, mk in [
            ("A.1 quartz window", "HYLON_TNF_2026_01_Phase2-1/FLAME_A1/A1_Wall_Temp/A1_T_wall_chamber_quartz.txt", "#1f77b4", "o"),
            ("A.1 metallic wall", "HYLON_TNF_2026_01_Phase2-1/FLAME_A1/A1_Wall_Temp/A1_T_wall_chamber_metal.txt", "#1f77b4", "s"),
            ("L.1 quartz window", "HYLON_TNF_2026_01_Phase2-1/FLAME_L1/L1_Wall_Temp/L1_T_wall_chamber_quartz.txt", "#d62728", "o"),
            ("L.1 metallic wall", "HYLON_TNF_2026_01_Phase2-1/FLAME_L1/L1_Wall_Temp/L1_T_wall_chamber_metal.txt", "#d62728", "s"),
        ]:
            w = read_wall(z, fn)
            ls = "-" if "quartz" in lbl else "--"
            ax.errorbar(w[:, 0], w[:, 1], yerr=w[:, 2], color=col, marker=mk, ls=ls, ms=4.5, capsize=2.5, lw=1.3, label=lbl)
        ax.set_xlabel("Axial wall position z (mm)")
        ax.set_ylabel("Inner wall temperature (K)")
        ax.set_title("(c) HYLON2 LIP inner wall T (quartz vs metal)", fontsize=10)
        ax.legend(fontsize=7.8, loc="best")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_hylon2_tc_vs_raman_and_walls.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate verifiable diagnostic figures from experimental samples.")
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_fig1_nist_rse2007(args.samples_dir, args.out_dir)
    make_fig2_faults(args.samples_dir, args.out_dir)
    make_fig3_nist_pool(args.samples_dir, args.out_dir)
    make_fig4_hylon2(args.samples_dir, args.out_dir)
    print("Generated 4 figures in", args.out_dir)


if __name__ == "__main__":
    main()
