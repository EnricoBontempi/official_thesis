"""Virtual PIV observations of the LES velocity (mission 02, ``piv_observation_spec_v1``).

Observation model, in this order::

    LES velocity (m/s) -> response convolution (field-limited, renormalised at edges)
        -> valid-vector selection (interrogation-window rule)
        -> correlated random error (m/s)
        -> rejections (sensitivities only) -> substitution (level ``sub`` only)
        -> frozen normalisation -> U-Net / GLU packet

Design rules
------------
* **No temperature.** Neither the public signature nor any helper takes T.  Temperature
  diagnostics live in separate qualification scripts.
* **Two coordinate conventions are kept apart.**  The *physical* convention used by the
  generator (decision R1, "probable"): array rows <-> coordinate ``x`` (step 0.581 mm),
  array columns <-> coordinate ``y`` (step 0.758 mm).  The *historical* convention used by
  the networks (``x_scaled``/``y_scaled``, GLU geometry): columns <-> ``x``, rows <-> ``y``.
  ``CanonicalSparseObservation.velocity_coords`` is always historical (unchanged network
  input); the generator's physical coordinates are returned in ``PivObservation.phys_coords``.
* **Same physical packet for both families.**  U-Net and GLU receive the very same
  ``CanonicalSparseObservation`` (same sites, same values, same hash).
* Determinism: every random draw is keyed by
  ``(namespace, role seed, operator, component, sim_id, frame, replica_or_epoch)``
  through ``zlib.crc32`` (never Python ``hash``); neither the model family nor the model
  seed is part of the key.

This module does not modify ``nonideal_observations.py`` and does not read any file other
than the frozen spec and the frozen normalisation statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.ndimage import convolve, correlate1d, label
from scipy.optimize import brentq
from scipy.special import erf, ndtr
from scipy.stats import t as student_t

from combustion_recon.data.masks import stable_seed
from combustion_recon.data.month2_observations import mask_hash
from combustion_recon.data.nonideal_observations import VelocityScale
from combustion_recon.data.sparse_eval_masks import sites_to_mask
from combustion_recon.training.sparse_view import CanonicalSparseObservation

SPEC_FORMAT = "piv_observation_spec_v1"
NAMESPACE = "piv_observation_v1"
HISTORICAL_AND_MISSION00_SEEDS = frozenset({20261114, 20261116, 20261118, 20261125, 20261201, 20261202})
ROLE_SEEDS = {"train": 20261210, "selection": 20261211, "eval": 20261212}
FORBIDDEN_ROLE_PREFIXES = ("test",)
GRID_HW = (128, 128)
NOISE_PAD = 16  # nodes of white noise kept around the slice (> every kernel half-support)
B_MHG = 0.35
LSR_RESPONSE_OVER_W = 0.93
LSR_NOISE_OVER_W = 1.2
TRUNC_SIGMA = 4.0
CAL_TOL = 0.01  # R1: always calibrate, tolerance 1 %
VELOCITY_COMPONENTS = ("vi", "vj")


class PivError(ValueError):
    """Contract violation in the PIV observation model."""


# ===================================================================== kernels (continuous + discrete)

def mhg_cdf(s, sigma: float, b: float = B_MHG):
    """Antiderivative of F(s) = (1 - b s^2/sig^2) exp(-s^2/2sig^2) / ((1-b) sig sqrt(2 pi))."""
    s = np.asarray(s, dtype=np.float64)
    return 0.5 * erf(s / (sigma * math.sqrt(2.0))) + b * s * np.exp(-s**2 / (2 * sigma**2)) / (
        (1 - b) * sigma * math.sqrt(2 * math.pi)
    )


def mhg_lsr_continuous(sigma: float, b: float = B_MHG) -> float:
    return sigma * math.sqrt(4 * math.pi) * (1 - b) ** 2 / (1 - b + 3 * b**2 / 4)


def _half_nodes(radius: float, step: float) -> int:
    return int(math.ceil((radius + step / 2) / step)) + 1


def mhg_weights(sigma: float, step: float, b: float = B_MHG, trunc: float = TRUNC_SIGMA):
    """Integer node offsets and weights (sum 1) of the discrete MHG on a piecewise-constant field."""
    R = trunc * sigma
    n = _half_nodes(R, step)
    k = np.arange(-n, n + 1)
    s = k * step
    lo = np.clip(s - step / 2, -R, R)
    hi = np.clip(s + step / 2, -R, R)
    w = mhg_cdf(hi, sigma, b) - mhg_cdf(lo, sigma, b)
    keep = hi > lo
    k, w = k[keep], w[keep]
    return k.astype(int), w / w.sum()


def tophat_weights(w_eff: float, step: float):
    """Cell-overlap weights of a top-hat of width ``w_eff`` (sum 1)."""
    n = int(math.ceil((w_eff / 2 + step / 2) / step)) + 1
    k = np.arange(-n, n + 1)
    s = k * step
    lo = np.clip(s - step / 2, -w_eff / 2, w_eff / 2)
    hi = np.clip(s + step / 2, -w_eff / 2, w_eff / 2)
    w = np.clip(hi - lo, 0.0, None)
    keep = w > 0
    k, w = k[keep], w[keep]
    return k.astype(int), w / w.sum()


def lsr_discrete(w: np.ndarray, step: float) -> float:
    return float(step / np.sum(np.asarray(w) ** 2))


def calibrate(kind: str, lsr_target: float, step: float, tol: float = CAL_TOL) -> dict[str, Any]:
    """Tune the kernel scale so that the DISCRETE L_sr equals ``lsr_target`` (R1: always).

    kind = "mhg" -> tunes sigma; kind = "tophat" -> tunes the effective width.
    Raises ``PivError`` instead of returning a silently wrong kernel when the target is not
    attainable on this grid (e.g. target below the cell size) or the tuned kernel misses
    the target by more than ``tol``.
    """
    if kind == "mhg":
        nominal = lsr_target / (math.sqrt(4 * math.pi) * (1 - B_MHG) ** 2 / (1 - B_MHG + 3 * B_MHG**2 / 4))
        fn = lambda p: lsr_discrete(mhg_weights(p, step)[1], step) - lsr_target
        wfun = mhg_weights
        lo, hi = 0.02 * nominal, 6 * nominal
    elif kind == "tophat":
        nominal = lsr_target
        fn = lambda p: lsr_discrete(tophat_weights(p, step)[1], step) - lsr_target
        wfun = tophat_weights
        lo, hi = 1e-3 * lsr_target, 4 * lsr_target
    else:
        raise PivError(f"unknown kernel kind {kind!r}")
    if lsr_target <= step * (1 + 1e-9):
        raise PivError(f"{kind}: discrete target L_sr={lsr_target:.4f} mm is not above the cell size {step:.4f} mm")
    f_lo, f_hi = fn(lo), fn(hi)
    if f_lo * f_hi > 0:
        raise PivError(f"{kind}: target {lsr_target:.4f} mm not bracketed on step {step:.4f} mm")
    p = brentq(fn, lo, hi, xtol=1e-12, rtol=1e-13)
    k, w = wfun(p, step)
    lsr = lsr_discrete(w, step)
    err = lsr / lsr_target - 1
    if abs(err) > tol:
        raise PivError(f"{kind}: calibrated L_sr misses target by {100*err:.3f} % (> {100*tol:.1f} %)")
    nom_k, nom_w = wfun(nominal, step)
    return {"param": float(p), "param_nominal": float(nominal), "lsr_disc": lsr, "lsr_target": float(lsr_target),
            "lsr_disc_nominal": lsr_discrete(nom_w, step), "err_rel": err, "half_nodes": int(np.max(np.abs(k))),
            "n_weights": int(len(w)), "neg_mass": float(w[w < 0].sum())}


def kernel_weights(kind: str, param: float, step: float):
    if kind == "mhg":
        return mhg_weights(param, step)
    if kind == "tophat":
        return tophat_weights(param, step)
    raise PivError(kind)


def axis_matrix(n: int, offsets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n, n) operator: row i = weights placed at node i+offset, truncated to [0, n) and renormalised."""
    M = np.zeros((n, n))
    for i in range(n):
        j = i + offsets
        ok = (j >= 0) & (j < n)
        M[i, j[ok]] = weights[ok] / weights[ok].sum()
    return M


def window_margin(w_mm: float, step: float) -> int:
    """Nodes excluded at each side so that the interrogation window [-W/2, W/2] stays in the field."""
    return int(math.ceil(w_mm / (2 * step) - 1e-9))


def transfer_function(offsets: np.ndarray, weights: np.ndarray, step: float, wavelength_mm) -> np.ndarray:
    k = 2 * np.pi / np.atleast_1d(np.asarray(wavelength_mm, dtype=np.float64))
    return np.array([float(np.sum(weights * np.cos(kk * offsets * step))) for kk in k])


def autocorr_discrete(weights: np.ndarray, max_lag: int = 4) -> np.ndarray:
    g = np.asarray(weights)
    den = float(np.sum(g * g))
    return np.array([float(np.sum(g[: len(g) - m] * g[m:])) / den if m < len(g) else 0.0 for m in range(max_lag + 1)])


# ===================================================================== steps and fields

STEPS_PRIMARY = {"rows": 0.5810574316105278, "cols": 0.7581226848528722}  # replaced by the spec's values
B_SIDE_MM = 30.0


def _steps(spec_steps: Mapping[str, float], convention: str) -> tuple[float, float]:
    r, c = float(spec_steps["rows"]), float(spec_steps["cols"])
    if convention == "primary":
        return r, c
    if convention == "swap":
        return c, r
    raise PivError(f"unknown step convention {convention!r}")


def field_rect(field_id: str, steps: tuple[float, float], hw=GRID_HW) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1), all inclusive, of the observed field in node indices.

    Windows B are 30 x 30 mm (STO12 sec. 2.3.3) anchored by decision R1:
    B-in   rows centred on the axis (row 63.5), cols adjacent to the inlet (last column);
    B-down same rows, the adjacent window 30 mm downstream (towards lower columns);
    B-off  cols as B-in, rows starting at the axis (row 64).
    """
    H, W = hw
    if field_id in ("A", "full"):
        return 0, H - 1, 0, W - 1
    dr, dc = steps
    nr = int(math.floor(B_SIDE_MM / dr + 1e-9)) + 1
    nc = int(math.floor(B_SIDE_MM / dc + 1e-9)) + 1
    r_axis0 = int(round((H - 1) / 2 - (nr - 1) / 2))
    c_in0 = W - nc
    if field_id == "B-in":
        return r_axis0, r_axis0 + nr - 1, c_in0, W - 1
    if field_id == "B-down":
        return r_axis0, r_axis0 + nr - 1, c_in0 - nc, c_in0 - 1
    if field_id == "B-off":
        return H // 2, H // 2 + nr - 1, c_in0, W - 1
    raise PivError(f"unknown field {field_id!r}")


# ===================================================================== spec

@dataclass(frozen=True)
class PivSpec:
    raw: Mapping[str, Any]
    sha256: str  # sha256 of the YAML bytes

    @property
    def steps(self) -> Mapping[str, float]:
        return self.raw["geometry"]["steps_mm"]

    def vref(self, sim_id: str) -> float:
        tab = self.raw["reference_velocity"]["per_condition_m_s"]
        if sim_id not in tab:
            raise PivError(f"no reference velocity for {sim_id!r} (TEST conditions are never in the spec)")
        return float(tab[sim_id])

    def config(self, config_id: str) -> Mapping[str, Any]:
        cfgs = self.raw["configs"]
        if config_id not in cfgs:
            raise PivError(f"unknown config {config_id!r}")
        return cfgs[config_id]

    def kernel(self, which: str, convention: str, w_mm: float, axis: str) -> Mapping[str, Any]:
        key = f"{w_mm:.1f}"
        return self.raw["kernels"][convention][key][which][axis]


def spec_content_hash(raw: Mapping[str, Any]) -> str:
    """Canonical-JSON sha256 of the spec content (excluding the ``frozen`` block)."""
    body = {k: v for k, v in raw.items() if k != "frozen"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_piv_spec(path: str | Path, *, expected_sha256: str | None = None) -> PivSpec:
    p = Path(path)
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise PivError(f"spec sha256 mismatch: {sha} != {expected_sha256}")
    raw = yaml.safe_load(data)
    if raw.get("format") != SPEC_FORMAT:
        raise PivError(f"unexpected spec format {raw.get('format')!r}")
    seeds = raw["seeds"]
    if set(seeds["role_seeds"].values()) & HISTORICAL_AND_MISSION00_SEEDS:
        raise PivError("PIV seeds collide with historical / mission-00 namespaces")
    if dict(seeds["role_seeds"]) != ROLE_SEEDS:
        raise PivError("role seeds differ from the module constants")
    frozen = raw.get("frozen", {})
    if frozen.get("content_sha256") and frozen["content_sha256"] != spec_content_hash(raw):
        raise PivError("spec content hash does not match its frozen block")
    return PivSpec(raw=raw, sha256=sha)


# ===================================================================== generator

@dataclass(frozen=True)
class PivObservation:
    obs: CanonicalSparseObservation  # historical coordinates: the packet both families receive
    phys_coords: np.ndarray  # (N, 2) hypothesised physical (x_phys, y_phys) in mm; NOT used by the networks
    config_id: str
    meta: dict[str, Any]
    obs_hash: str
    intermediates: dict[str, np.ndarray] | None = None


def _hash_obs(sites: np.ndarray, values_phys: np.ndarray, level: str | None) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(sites, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(values_phys, dtype=np.float64).tobytes())
    h.update(str(level).encode())
    return h.hexdigest()


class PivObservationGenerator:
    """Applies one named configuration of the frozen spec to a pair of LES velocity fields."""

    def __init__(self, spec: PivSpec, scale: VelocityScale, *, x_vec_m: np.ndarray, y_vec_m: np.ndarray):
        self.spec = spec
        self.scale = scale
        if scale.stats_sha256 and scale.stats_sha256 != spec.raw["normalization"]["stats_sha256"]:
            raise PivError("normalisation statistics differ from the ones frozen in the spec")
        self.x_vec_mm = np.asarray(x_vec_m, dtype=np.float64) * 1e3
        self.y_vec_mm = np.asarray(y_vec_m, dtype=np.float64) * 1e3
        self.x_vec_m = np.asarray(x_vec_m, dtype=np.float64)
        self.y_vec_m = np.asarray(y_vec_m, dtype=np.float64)
        self._cache: dict[Any, Any] = {}

    # -- building blocks -------------------------------------------------------------
    def _kernel_axes(self, which: str, convention: str, w_mm: float):
        """(rows_offsets, rows_weights, cols_offsets, cols_weights) from the frozen spec."""
        key = (which, convention, w_mm)
        if key in self._cache:
            return self._cache[key]
        steps = _steps(self.spec.steps, convention)
        out = []
        for axis, step in zip(("rows", "cols"), steps):
            e = self.spec.kernel(which, convention, w_mm, axis)
            kind = {"response_mhg": "mhg", "response_tophat": "tophat", "noise": "mhg"}[which]
            out.append(kernel_weights(kind, float(e["param_mm"]), step))
        res = (out[0][0], out[0][1], out[1][0], out[1][1])
        self._cache[key] = res
        return res

    def _response(self, vel: np.ndarray, rect, which: str, convention: str, w_mm: float):
        r0, r1, c0, c1 = rect
        sub = vel[r0:r1 + 1, c0:c1 + 1]
        key = ("ops", which, convention, w_mm, sub.shape)
        if key not in self._cache:
            ko_r, kw_r, ko_c, kw_c = self._kernel_axes(which, convention, w_mm)
            self._cache[key] = (axis_matrix(sub.shape[0], ko_r, kw_r), axis_matrix(sub.shape[1], ko_c, kw_c))
        Mr, Mc = self._cache[key]
        return Mr @ sub @ Mc.T

    def _noise_field(self, role: str, comp: int, sim_id: str, frame: int, replica: int, convention: str, w_mm: float,
                     correlated: bool, stream: str = "error") -> np.ndarray:
        """Unit-variance noise on the whole slice (white on a padded grid, then filtered with g)."""
        seed = stable_seed(NAMESPACE, ROLE_SEEDS[role], stream, comp, sim_id, int(frame), int(replica))
        H, W = GRID_HW
        white = np.random.default_rng(seed).standard_normal((H + 2 * NOISE_PAD, W + 2 * NOISE_PAD))
        if not correlated:
            return white[NOISE_PAD:NOISE_PAD + H, NOISE_PAD:NOISE_PAD + W]
        ko_r, kw_r, ko_c, kw_c = self._kernel_axes("noise", convention, w_mm)
        if max(np.abs(ko_r).max(), np.abs(ko_c).max()) > NOISE_PAD:
            raise PivError("noise kernel exceeds the padded grid")
        # symmetric kernels: correlation == convolution; offsets are centred integers
        gr = np.zeros(2 * int(np.abs(ko_r).max()) + 1); gr[ko_r + int(np.abs(ko_r).max())] = kw_r
        gc = np.zeros(2 * int(np.abs(ko_c).max()) + 1); gc[ko_c + int(np.abs(ko_c).max())] = kw_c
        sm = correlate1d(white, gr, axis=0, mode="constant")
        sm = correlate1d(sm, gc, axis=1, mode="constant")
        sm = sm[NOISE_PAD:NOISE_PAD + H, NOISE_PAD:NOISE_PAD + W]
        return sm / math.sqrt(float(np.sum(kw_r**2)) * float(np.sum(kw_c**2)))

    @staticmethod
    def _valid_mask(rect, steps: tuple[float, float], w_mm: float, hw=GRID_HW) -> np.ndarray:
        r0, r1, c0, c1 = rect
        mr, mc = window_margin(w_mm, steps[0]), window_margin(w_mm, steps[1])
        m = np.zeros(hw, dtype=bool)
        m[r0 + mr:r1 - mr + 1, c0 + mc:c1 - mc + 1] = True
        return m

    def _edge_mask(self, rect, which: str, convention: str, w_mm: float) -> np.ndarray:
        """True where the kernel support leaves the field (vector near the field edge)."""
        r0, r1, c0, c1 = rect
        ko_r, _, ko_c, _ = self._kernel_axes(which, convention, w_mm)
        nr_, nc_ = int(np.abs(ko_r).max()), int(np.abs(ko_c).max())
        m = np.ones(GRID_HW, dtype=bool)
        m[r0 + nr_:r1 - nr_ + 1, c0 + nc_:c1 - nc_ + 1] = False
        return m

    # -- main entry ------------------------------------------------------------------
    def observe(self, config_id: str, vi_phys: np.ndarray, vj_phys: np.ndarray, *, sim_id: str, frame_index: int,
                role: str, replica: int = 0, return_intermediates: bool = False, restrict_sites: np.ndarray | None = None,
                cfg_override: Mapping[str, Any] | None = None) -> PivObservation:
        """One observation packet.  The signature has no temperature argument by design.

        ``restrict_sites`` (evaluation only): flat node indices; the observed set is intersected with them *after* the validity
        rule and before rejections (the E0v/E1/E2 contrasts at the historical sites). ``cfg_override`` (evaluation only) replaces the
        named configuration by an explicit mapping, labelled ``config_id`` (the E4*/E5* diagnostics on all 16 384 nodes).
        """
        if role not in ROLE_SEEDS:
            raise PivError(f"role {role!r} refused (allowed: {tuple(ROLE_SEEDS)}); TEST is never generated here")
        if any(sim_id.startswith(p) for p in FORBIDDEN_ROLE_PREFIXES):
            raise PivError("TEST identifiers refused")
        cfg = dict(cfg_override) if cfg_override is not None else self.spec.config(config_id)
        vref = self.spec.vref(sim_id)  # raises for any condition that is not TRAIN/VAL
        H, W = GRID_HW
        for a in (vi_phys, vj_phys):
            if a.shape != (H, W):
                raise PivError(f"velocity field must be {(H, W)}")
        conv = cfg.get("steps", "primary")
        steps = _steps(self.spec.steps, conv)
        rect = field_rect(cfg["field"], steps)
        w_mm = float(cfg["W_mm"])
        response = cfg["response"]            # ideal | mhg | tophat
        noise_mode = cfg["noise"]             # none | corr | indep
        sigma_px = float(cfg.get("sigma_px", 0.0))
        d_max = float(self.spec.raw["error"]["d_max_px"])
        rej = cfg.get("rejection")            # None | iso | clu
        level = cfg.get("level")              # None | flag | sub
        if cfg["field"] == "full":
            valid = np.ones((H, W), dtype=bool)
        else:
            valid = self._valid_mask(rect, steps, w_mm)
        which_edge = "response_tophat" if response == "tophat" else "response_mhg"
        if cfg.get("edge_strict", False):
            valid = valid & ~self._edge_mask(rect, which_edge, conv, w_mm)
        if restrict_sites is not None:
            rm = np.zeros(H * W, dtype=bool); rm[np.asarray(restrict_sites, dtype=np.int64)] = True
            valid = valid & rm.reshape(H, W)
        sites_all = np.flatnonzero(valid.ravel())

        fields = {"vi": np.asarray(vi_phys, dtype=np.float64), "vj": np.asarray(vj_phys, dtype=np.float64)}
        filtered, err = {}, {}
        for comp, name in enumerate(VELOCITY_COMPONENTS):
            if response == "ideal":
                f = fields[name]
            else:
                f = np.full((H, W), np.nan)
                r0, r1, c0, c1 = rect
                f[r0:r1 + 1, c0:c1 + 1] = self._response(fields[name], rect, "response_" + response, conv, w_mm)
            filtered[name] = f
            if noise_mode == "none" or sigma_px == 0.0:
                err[name] = np.zeros((H, W))
            else:
                z = self._noise_field(role, comp, sim_id, frame_index, replica, conv, w_mm, correlated=(noise_mode in ("corr", "corr_t")))
                if noise_mode == "corr_t":
                    # SENS-TAIL (v1.1, R2 sec. 4): monotone marginal map z -> F_t^-1(Phi(z); nu), rescaled to unit variance
                    nu = float(cfg["t_nu"])
                    z = student_t.ppf(ndtr(z), nu) / math.sqrt(nu / (nu - 2.0))
                err[name] = (sigma_px * vref / d_max) * z
        vals = {n: filtered[n] + err[n] for n in VELOCITY_COMPONENTS}

        # --- rejections (sensitivities only) and substitution --------------------------
        rejected = np.zeros((H, W), dtype=bool)
        if rej is not None:
            frac = float(self.spec.raw["rejection_sensitivity"]["fraction"])
            k = int(round(frac * sites_all.size))
            seed = stable_seed(NAMESPACE, ROLE_SEEDS[role], "rejection", rej, sim_id, int(frame_index), int(replica))
            if rej == "iso":
                pick = np.random.default_rng(seed).permutation(sites_all.size)[:k]
                rejected.ravel()[sites_all[pick]] = True
            elif rej == "clu":
                # correlated Gaussian field on scale ~W (noise kernel), thresholded at the (1-frac) quantile
                zf = self._noise_field(role, 0, sim_id, frame_index, replica, conv, w_mm, correlated=True, stream="rejclu")
                order = np.argsort(-zf.ravel()[sites_all], kind="stable")[:k]
                rejected.ravel()[sites_all[order]] = True
            else:
                raise PivError(f"unknown rejection {rej!r}")
        sub_info = None
        sub_rule = cfg.get("substitution", "mean3x3_iterative")
        if sub_rule not in SUBSTITUTION_HALF:
            raise PivError(f"unknown substitution rule {sub_rule!r}")
        if level == "sub":
            if rej is None:
                raise PivError("level 'sub' requires a rejection mask")
            passes = []
            for n in VELOCITY_COMPONENTS:
                vals[n], pn = substitute_iterative(vals[n], valid & ~rejected, valid, half=SUBSTITUTION_HALF[sub_rule])
                passes.append(pn)
            sub_info = max(passes)
            keep = valid
        elif level == "flag":
            keep = valid & ~rejected
        else:
            keep = valid
        sites = np.flatnonzero(keep.ravel())
        if sites.size == 0:
            raise PivError("no observed vectors")
        rr, cc = sites // W, sites % W
        phys = np.stack([vals["vi"][rr, cc], vals["vj"][rr, cc]], axis=1)
        norm = (phys - np.asarray(self.scale.mean)) / np.asarray(self.scale.std)
        coords_hist = np.stack([self.x_vec_m[cc], self.y_vec_m[rr]], axis=1)  # historical: cols<->x, rows<->y
        phys_coords = np.stack([self.x_vec_mm[rr], self.y_vec_mm[cc]], axis=1)  # R1 hypothesis: rows<->x, cols<->y
        mask = sites_to_mask(sites, height=H, width=W)
        obs_hash = _hash_obs(sites, phys, level)
        obs = CanonicalSparseObservation(
            sites=sites, mask=mask, velocity_coords=coords_hist, velocity_values_phys=phys, velocity_values_norm=norm,
            n_sites=int(sites.size), site_fraction=float(sites.size) / (H * W),
            layout_id=f"{SPEC_FORMAT}|{self.spec.sha256[:16]}|{config_id}", velocity_mask_seed=ROLE_SEEDS[role],
            mask_hash=mask_hash(mask), H=H, W=W,
        )
        meta = {
            "config_id": config_id, "spec_sha256": self.spec.sha256, "namespace": NAMESPACE, "role": role,
            "role_seed": ROLE_SEEDS[role], "sim_id": sim_id, "frame_index": int(frame_index), "replica_or_epoch": int(replica),
            "field": cfg["field"], "field_rect_rows_cols": list(rect), "step_convention": conv, "W_mm": w_mm,
            "response": response, "noise": noise_mode, "sigma_px": sigma_px, "sigma_u_m_s": sigma_px * vref / d_max if noise_mode != "none" else 0.0,
            "v_ref_m_s": vref, "rejection": rej, "level": level, "n_valid_vectors": int(valid.sum()),
            "n_observed": int(sites.size), "n_rejected": int(rejected.sum()), "n_substituted": int(rejected.sum()) if level == "sub" else 0,
            "substitution_rule": sub_rule if level == "sub" else None, "sub_passes": sub_info, "sub_passes_gt3": (sub_info > 3) if sub_info is not None else None,
            "valid_mask_hash": mask_hash(valid), "observed_mask_hash": obs.mask_hash, "obs_hash": obs_hash,
            "n_edge_vectors": int((valid & self._edge_mask(rect, which_edge, conv, w_mm)).sum()) if response != "ideal" else None,
            "temperature_used": False,
        }
        inter = None
        if return_intermediates:
            inter = {"filtered_vi": filtered["vi"], "filtered_vj": filtered["vj"], "err_vi": err["vi"], "err_vj": err["vj"],
                     "valid": valid, "rejected": rejected, "observed": keep,
                     "vals_vi": vals["vi"], "vals_vj": vals["vj"]}
            if sub_info is not None:
                inter["sub_passes"] = sub_info
        return PivObservation(obs=obs, phys_coords=phys_coords, config_id=config_id, meta=meta, obs_hash=obs_hash, intermediates=inter)


SUBSTITUTION_HALF = {"mean3x3_iterative": 1, "mean5x5_iterative": 2}


def substitute_iterative_3x3(values: np.ndarray, known: np.ndarray, domain: np.ndarray):
    """Iterative 3x3 mean (v1 rule). Thin wrapper kept so that v1 behaviour is untouched."""
    return substitute_iterative(values, known, domain, half=1)


def substitute_iterative(values: np.ndarray, known: np.ndarray, domain: np.ndarray, half: int = 1):
    """Replace unknown vectors of ``domain`` with the mean of their known (2*half+1)^2 neighbours, iteratively.

    half=1: 3x3 (8 neighbours, v1 rule); half=2: 5x5 (24 neighbours, v1.1 variant, area declared in the Cardiff XML).
    Each pass fills every unknown node having at least one known neighbour (simultaneous update);
    nodes filled in a pass count as known in the next one.  Returns (new_values, n_passes).
    """
    vals = np.where(known, values, 0.0)
    kn = known.copy()
    ker = np.ones((2 * half + 1, 2 * half + 1)); ker[half, half] = 0.0
    passes = 0
    while True:
        unknown = domain & ~kn
        if not unknown.any():
            break
        cnt = convolve(kn.astype(np.float64), ker, mode="constant", cval=0.0)
        tot = convolve(np.where(kn, vals, 0.0), ker, mode="constant", cval=0.0)
        fill = unknown & (cnt > 0)
        if not fill.any():
            raise PivError("substitution cannot reach some unknown vectors")
        vals = np.where(fill, tot / np.maximum(cnt, 1), vals)
        kn = kn | fill
        passes += 1
    out = np.where(domain, vals, np.nan)
    return out, passes


def cluster_sizes(mask: np.ndarray) -> np.ndarray:
    lab, n = label(mask, structure=np.ones((3, 3)))
    return np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)
