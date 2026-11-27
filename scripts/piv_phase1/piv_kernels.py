"""Diagnostic PIV kernels (roadmap_tecnica_piv.md §2.3 points 3 and 5). NOT the F2.2 generator.

Continuous kernel per axis (Wieneke 2017, Tab. 2.1; roadmap §2.3(3)):
    F(s) = (1 - b s^2/sig^2) exp(-s^2/(2 sig^2)) / ((1-b) sig sqrt(2 pi)),  int F ds = 1
    L_sr = 1/int F^2 ds = sig sqrt(4 pi) (1-b)^2 / (1 - b + 3 b^2/4)
Antiderivative (derived here, checked numerically in verify()):
    G(s) = 1/2 erf(s/(sig sqrt2)) + b s exp(-s^2/(2 sig^2)) / ((1-b) sig sqrt(2 pi))
Discretisation: field piecewise constant per LES cell; weight of node at offset s_i =
    int_{s_i - D/2}^{s_i + D/2} F ds  clipped to [-R, R], R = 4 sig; then sum-to-one.
Top-hat: F = 1/W on |s| <= W/2, cell weight = overlap length / W.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

B_MHG = 0.35
LSR_RESPONSE = 0.93   # L_sr / W, response kernel (WIE17 Tab 2.4)
LSR_NOISE = 1.2       # L_sr / W, noise autocorrelation kernel (WIE17 Tab 2.4)
TRUNC_SIGMA = 4.0
TOL_LSR = 0.05


def mhg_G(s, sig, b=B_MHG):
    s = np.asarray(s, dtype=np.float64)
    return 0.5 * erf(s / (sig * np.sqrt(2))) + b * s * np.exp(-s**2 / (2 * sig**2)) / ((1 - b) * sig * np.sqrt(2 * np.pi))


def mhg_F(s, sig, b=B_MHG):
    s = np.asarray(s, dtype=np.float64)
    return (1 - b * s**2 / sig**2) * np.exp(-s**2 / (2 * sig**2)) / ((1 - b) * sig * np.sqrt(2 * np.pi))


def lsr_continuous_formula(sig, b=B_MHG):
    return sig * np.sqrt(4 * np.pi) * (1 - b) ** 2 / (1 - b + 3 * b**2 / 4)


def sigma_for_lsr(lsr, b=B_MHG):
    return lsr / (np.sqrt(4 * np.pi) * (1 - b) ** 2 / (1 - b + 3 * b**2 / 4))


def n_half(R, D):
    """Number of nodes on each side that can overlap the support [-R, R]."""
    return int(np.ceil((R + D / 2) / D)) + 1


def mhg_weights(sig, D, b=B_MHG, trunc=TRUNC_SIGMA):
    R = trunc * sig
    n = n_half(R, D)
    s = np.arange(-n, n + 1) * D
    lo = np.clip(s - D / 2, -R, R)
    hi = np.clip(s + D / 2, -R, R)
    w = mhg_G(hi, sig, b) - mhg_G(lo, sig, b)
    keep = (hi > lo)
    s, w = s[keep], w[keep]
    return s, w / w.sum()


def tophat_weights(W, D):
    n = int(np.ceil((W / 2 + D / 2) / D)) + 1
    s = np.arange(-n, n + 1) * D
    lo = np.clip(s - D / 2, -W / 2, W / 2)
    hi = np.clip(s + D / 2, -W / 2, W / 2)
    w = np.clip(hi - lo, 0, None)
    keep = w > 0
    s, w = s[keep], w[keep]
    return s, w / w.sum()


def lsr_discrete(w, D):
    return D / float(np.sum(w**2))


def calibrate_sigma(lsr_target, D, b=B_MHG, tol=TOL_LSR):
    """Return (sigma_nominal, sigma_used, L_disc_at_nominal, L_disc_used, recalibrated, reachable)."""
    sig0 = sigma_for_lsr(lsr_target, b)
    l0 = lsr_discrete(mhg_weights(sig0, D, b)[1], D)
    if abs(l0 / lsr_target - 1) <= tol:
        return sig0, sig0, l0, l0, False, True
    f = lambda sg: lsr_discrete(mhg_weights(sg, D, b)[1], D) - lsr_target
    lo, hi = 0.05 * sig0, 5 * sig0
    if f(lo) * f(hi) > 0:
        return sig0, sig0, l0, l0, False, False
    sg = brentq(f, lo, hi, xtol=1e-10)
    return sig0, sg, l0, lsr_discrete(mhg_weights(sg, D, b)[1], D), True, True


def transfer(s, w, wavelength_mm):
    """Discrete transfer H(k) = sum_i w_i cos(k s_i), k = 2 pi / lambda."""
    k = 2 * np.pi / np.asarray(wavelength_mm, dtype=np.float64)
    return np.array([np.sum(w * np.cos(kk * s)) for kk in np.atleast_1d(k)])


def autocorr_discrete(w, max_lag=4):
    """rho(m) = sum_i g_i g_{i+m} / sum g^2 for the discrete noise kernel (white noise -> filter)."""
    g = w
    den = float(np.sum(g * g))
    return np.array([float(np.sum(g[: len(g) - m] * g[m:])) / den for m in range(max_lag + 1)])


def axis_matrix(N, w, s, D, valid_lo, valid_hi=None):
    """(N,N) matrix: row i = weights of node i, truncated to nodes in [0, N-1] and renormalised.

    Rows outside [valid_lo, N-1-valid_hi] are still built (edge approximation) but flagged by caller.
    """
    half = (len(w) - 1) // 2
    M = np.zeros((N, N))
    idx = np.rint(s / D).astype(int)
    for i in range(N):
        j = i + idx
        ok = (j >= 0) & (j < N)
        M[i, j[ok]] = w[ok] / w[ok].sum()
    return M


def valid_margin(W, D):
    """Window rule: node i valid iff i*D >= W/2 from the first/last node. Returns integer margin."""
    return int(np.ceil(W / (2 * D) - 1e-9))


def apply_separable(field, My, Mx):
    """field (..., Ny, Nx) -> My @ field @ Mx^T."""
    return np.einsum("ij,...jk,lk->...il", My, field, Mx, optimize=True)


def verify():
    out = {}
    for b in (0.35,):
        sig = 1.0
        s = np.linspace(-12, 12, 480001)
        F = mhg_F(s, sig, b)
        ds = s[1] - s[0]
        out["int_F"] = float(np.sum(F) * ds)
        out["Lsr_num"] = float(1 / (np.sum(F**2) * ds))
        out["Lsr_formula"] = float(lsr_continuous_formula(sig, b))
        G = mhg_G(s, sig, b)
        out["G_err"] = float(np.max(np.abs(np.gradient(G, s) - F)))
        out["sigma_for_0.93W"] = float(sigma_for_lsr(0.93, b))
        out["sigma_for_1.2W"] = float(sigma_for_lsr(1.2, b))
        out["lobe_start_sigma"] = float(1 / np.sqrt(b))
    return out


if __name__ == "__main__":
    print(verify())
