"""Non-neural reconstruction baselines for Month 1 feasibility checks.

All predictors that use observations consume sparse tokens / boolean masks
and never treat unobserved locations as physical zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor


# Stoichiometric reference for centered operating-parameter features
_NH3_REF = 0.30
_PHI_REF = 1.0
_PENALTY = np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # never regularize intercept
_RIDGE_GRID = (1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
_BASIS_CHOICES = ("quadratic", "piecewise_phi1")


def _feature_ops(
    nh3_fraction: float,
    phi: float,
    *,
    basis: str = "quadratic",
) -> np.ndarray:
    """Centered (or piecewise) operating-parameter features; col 0 = intercept."""
    n = float(nh3_fraction)
    p = float(phi)
    nc = n - _NH3_REF
    pc = p - _PHI_REF
    if basis == "quadratic":
        return np.array([1.0, nc, pc, nc * nc, pc * pc, nc * pc], dtype=np.float64)
    if basis == "piecewise_phi1":
        return np.array(
            [
                1.0,
                nc,
                max(p - 1.0, 0.0),
                max(1.0 - p, 0.0),
                nc * (p - 1.0),
                nc * nc,
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown basis={basis!r}")


def _poly_ops(nh3_fraction: float, phi: float) -> np.ndarray:
    """Backward-compatible alias → centered quadratic features."""
    return _feature_ops(nh3_fraction, phi, basis="quadratic")


def apply_anchor_offset(
    T0: np.ndarray,
    T_obs: np.ndarray,
    T_mask: np.ndarray,
    *,
    tau: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Calibrate a global thermal offset from K sparse temperature anchors.

    ``b_hat = 1/(K+τ) Σ_k (T(x_k) - T0(x_k))``, then return ``T0 + b_hat``.
    """
    ys, xs = np.where(T_mask)
    K = int(ys.size)
    if K == 0:
        return np.asarray(T0, dtype=np.float64).copy(), 0.0
    resid = np.asarray(T_obs, dtype=np.float64)[ys, xs] - np.asarray(T0, dtype=np.float64)[ys, xs]
    b = float(resid.sum() / (K + float(tau)))
    return np.asarray(T0, dtype=np.float64) + b, b


@dataclass
class GlobalTrainMeanPredictor:
    """Predict the single train-averaged 2D temperature field (zero sensors)."""

    mean_field: np.ndarray  # (H, W)

    @classmethod
    def fit(cls, train_T_fields: Sequence[np.ndarray]) -> "GlobalTrainMeanPredictor":
        stack = np.stack([np.asarray(t, dtype=np.float64) for t in train_T_fields], axis=0)
        return cls(mean_field=stack.mean(axis=0))

    def predict(self, shape: tuple[int, int] | None = None) -> np.ndarray:
        if shape is not None and shape != self.mean_field.shape:
            raise ValueError(f"shape mismatch: {shape} vs {self.mean_field.shape}")
        return self.mean_field.copy()


@dataclass
class ParamConditionedMeanPredictor:
    """Ridge map from centered (NH3, φ) features → condition-mean T field.

    Intercept is never ridge-penalized. With ``ridge_alpha="auto"`` (default),
    basis and α are selected by leave-one-condition-out CV on the train means.
    """

    coef: np.ndarray  # (6, H*W)
    H: int
    W: int
    ridge_alpha: float = 1e-4
    basis: str = "quadratic"
    selected_basis: str = "quadratic"
    selected_ridge_alpha: float = 1e-4
    loco_rmse_K: float = float("nan")

    @staticmethod
    def _solve(X: np.ndarray, Y: np.ndarray, ridge_alpha: float) -> np.ndarray:
        xtx = X.T @ X + float(ridge_alpha) * _PENALTY
        return np.linalg.solve(xtx, X.T @ Y)

    @classmethod
    def _loco_rmse(
        cls,
        X: np.ndarray,
        Y: np.ndarray,
        ridge_alpha: float,
    ) -> float:
        C = X.shape[0]
        errs = []
        for i in range(C):
            mask = np.ones(C, dtype=bool)
            mask[i] = False
            coef = cls._solve(X[mask], Y[mask], ridge_alpha)
            pred = X[i] @ coef
            errs.append(float(np.sqrt(np.mean((pred - Y[i]) ** 2))))
        return float(np.mean(errs))

    @classmethod
    def fit(
        cls,
        condition_mean_fields: Sequence[np.ndarray],
        op_params: Sequence[tuple[float, float]],
        *,
        ridge_alpha: float | str = "auto",
        basis: str | None = None,
    ) -> "ParamConditionedMeanPredictor":
        assert len(condition_mean_fields) == len(op_params)
        H, W = condition_mean_fields[0].shape
        Y = np.stack([np.asarray(f, dtype=np.float64).ravel() for f in condition_mean_fields], axis=0)

        if ridge_alpha == "auto":
            bases = (basis,) if basis is not None else _BASIS_CHOICES
            best_score = float("inf")
            best_basis = bases[0]
            best_alpha = 1e-4
            print("  ParamMean LOCO-CV over basis × α …", flush=True)
            for bname in bases:
                Xb = np.stack([_feature_ops(n, p, basis=bname) for n, p in op_params], axis=0)
                for alpha in _RIDGE_GRID:
                    score = cls._loco_rmse(Xb, Y, alpha)
                    print(f"    basis={bname:16s} α={alpha:g}  LOCO RMSE={score:.2f} K", flush=True)
                    if score < best_score:
                        best_score = score
                        best_basis = bname
                        best_alpha = float(alpha)
            basis_use = best_basis
            alpha_use = best_alpha
            loco = best_score
        else:
            basis_use = basis if basis is not None else "quadratic"
            alpha_use = float(ridge_alpha)
            Xb = np.stack([_feature_ops(n, p, basis=basis_use) for n, p in op_params], axis=0)
            loco = cls._loco_rmse(Xb, Y, alpha_use) if len(op_params) >= 2 else float("nan")

        X = np.stack([_feature_ops(n, p, basis=basis_use) for n, p in op_params], axis=0)
        coef = cls._solve(X, Y, alpha_use)
        return cls(
            coef=coef,
            H=H,
            W=W,
            ridge_alpha=alpha_use,
            basis=basis_use,
            selected_basis=basis_use,
            selected_ridge_alpha=alpha_use,
            loco_rmse_K=loco,
        )

    def predict(self, nh3_fraction: float, phi: float) -> np.ndarray:
        x = _feature_ops(nh3_fraction, phi, basis=self.basis)
        return (x @ self.coef).reshape(self.H, self.W)

    def predict_with_anchor_offset(
        self,
        nh3_fraction: float,
        phi: float,
        T_obs: np.ndarray,
        T_mask: np.ndarray,
        *,
        tau: float = 1.0,
    ) -> np.ndarray:
        T0 = self.predict(nh3_fraction, phi)
        T_cal, _ = apply_anchor_offset(T0, T_obs, T_mask, tau=tau)
        return T_cal


@dataclass
class LocalVelocityTemperatureRegressor:
    """Pixel-wise tabular regressor: local velocity (+ optional op params) → T."""

    model: Any
    conditioned: bool
    feature_names: list[str]
    include_vk: bool = True

    @staticmethod
    def _local_features(
        x: np.ndarray,
        y: np.ndarray,
        vi: np.ndarray,
        vj: np.ndarray,
        vk: np.ndarray,
        umag: np.ndarray,
        *,
        nh3: float | None = None,
        phi: float | None = None,
        include_vk: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        H, W = vi.shape
        pad = np.pad(umag, 1, mode="edge")
        neigh = np.stack(
            [pad[i : i + H, j : j + W] for i in range(3) for j in range(3)],
            axis=0,
        )
        umag_mean = neigh.mean(axis=0)
        umag_std = neigh.std(axis=0)
        dux = np.gradient(umag, axis=1)
        duy = np.gradient(umag, axis=0)
        cols = [
            x.ravel(),
            y.ravel(),
            vi.ravel(),
            vj.ravel(),
        ]
        names = ["x", "y", "vi", "vj"]
        if include_vk:
            cols.append(vk.ravel())
            names.append("vk")
        cols.extend(
            [
                umag.ravel(),
                umag_mean.ravel(),
                umag_std.ravel(),
                dux.ravel(),
                duy.ravel(),
            ]
        )
        names.extend(["umag", "umag_mean3", "umag_std3", "dumag_dx", "dumag_dy"])
        if nh3 is not None and phi is not None:
            cols.append(np.full(H * W, float(nh3)))
            cols.append(np.full(H * W, float(phi)))
            names.extend(["nh3_fraction", "phi"])
        return np.stack(cols, axis=1), names

    @classmethod
    def fit(
        cls,
        frames: Sequence[dict[str, Any]],
        *,
        conditioned: bool,
        max_pixels: int = 80_000,
        seed: int = 0,
        include_vk: bool = True,
    ) -> "LocalVelocityTemperatureRegressor":
        rng = np.random.default_rng(seed)
        X_list = []
        y_list = []
        names: list[str] | None = None
        per = max(1, max_pixels // max(1, len(frames)))
        for fr in frames:
            xx, yy = fr["x2d"], fr["y2d"]
            umag = fr["umag"]
            if not include_vk:
                umag = np.sqrt(fr["vi"] ** 2 + fr["vj"] ** 2)
            feats, names = cls._local_features(
                xx,
                yy,
                fr["vi"],
                fr["vj"],
                fr["vk"],
                umag,
                nh3=fr["nh3"] if conditioned else None,
                phi=fr["phi"] if conditioned else None,
                include_vk=include_vk,
            )
            T = fr["T"].ravel()
            if feats.shape[0] > per:
                idx = rng.choice(feats.shape[0], size=per, replace=False)
                feats = feats[idx]
                T = T[idx]
            X_list.append(feats)
            y_list.append(T)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=120,
            min_samples_leaf=40,
            random_state=seed,
        )
        model.fit(X, y)
        assert names is not None
        return cls(model=model, conditioned=conditioned, feature_names=names, include_vk=include_vk)

    def predict_field(
        self,
        vi: np.ndarray,
        vj: np.ndarray,
        vk: np.ndarray,
        umag: np.ndarray,
        x2d: np.ndarray,
        y2d: np.ndarray,
        nh3: float = 0.0,
        phi: float = 1.0,
    ) -> np.ndarray:
        if not self.include_vk:
            umag = np.sqrt(vi**2 + vj**2)
        feats, _ = self._local_features(
            x2d,
            y2d,
            vi,
            vj,
            vk,
            umag,
            nh3=nh3 if self.conditioned else None,
            phi=phi if self.conditioned else None,
            include_vk=self.include_vk,
        )
        pred = self.model.predict(feats)
        return pred.reshape(vi.shape)


@dataclass
class GappyPODReconstructor:
    """Cross-variable Gappy POD with block-normalized MAP / weighted ridge.

    Modes are built on concatenated normalized ``[vi, vj(, vk), T]`` fluctuations
    about a mean state (global or parameter-conditioned). Observation blocks are
    normalized by count so sparse temperature anchors are not drowned by ridge /
    velocity mass; the prior is scaled by empirical modal variances ``σ_r²``.
    """

    mean_vi: np.ndarray
    mean_vj: np.ndarray
    mean_vk: np.ndarray
    mean_T: np.ndarray
    modes: np.ndarray  # (n_state, R)
    modal_variances: np.ndarray  # (R,)
    n_pix: int
    R: int
    ridge_lambda: float
    w_T: float
    scale_vi: float
    scale_vj: float
    scale_vk: float
    scale_T: float
    use_param_mean: bool
    n_velocity_components: int = 3
    param_mean_predictor: ParamConditionedMeanPredictor | None = None
    global_mean_T: np.ndarray | None = None
    eps_var: float = 1e-8

    @property
    def n_vel_blocks(self) -> int:
        return int(self.n_velocity_components)

    @property
    def t_offset(self) -> int:
        return self.n_vel_blocks * self.n_pix

    @property
    def n_state(self) -> int:
        return (self.n_vel_blocks + 1) * self.n_pix

    @staticmethod
    def _pack(
        vi: np.ndarray,
        vj: np.ndarray,
        vk: np.ndarray | None,
        T: np.ndarray,
        *,
        mean_vi: np.ndarray,
        mean_vj: np.ndarray,
        mean_vk: np.ndarray,
        mean_T: np.ndarray,
        scale_vi: float,
        scale_vj: float,
        scale_vk: float,
        scale_T: float,
        n_velocity_components: int,
    ) -> np.ndarray:
        parts = [
            ((vi - mean_vi) / scale_vi).ravel(),
            ((vj - mean_vj) / scale_vj).ravel(),
        ]
        if n_velocity_components >= 3:
            assert vk is not None
            parts.append(((vk - mean_vk) / scale_vk).ravel())
        parts.append(((T - mean_T) / scale_T).ravel())
        return np.concatenate(parts)

    @classmethod
    def fit(
        cls,
        frames: Sequence[dict[str, Any]],
        *,
        R: int = 32,
        ridge_lambda: float = 1e-3,
        w_T: float = 20.0,
        use_param_mean: bool = False,
        param_mean_predictor: ParamConditionedMeanPredictor | None = None,
        seed: int = 0,
        max_train_frames: int | None = None,
        n_velocity_components: int = 3,
    ) -> "GappyPODReconstructor":
        if n_velocity_components not in (2, 3):
            raise ValueError("n_velocity_components must be 2 or 3")
        rng = np.random.default_rng(seed)
        frames_use = list(frames)
        if max_train_frames is not None and len(frames_use) > max_train_frames:
            idx = rng.choice(len(frames_use), size=max_train_frames, replace=False)
            frames_use = [frames_use[i] for i in idx]

        stack_vi = np.stack([f["vi"] for f in frames_use], axis=0)
        stack_vj = np.stack([f["vj"] for f in frames_use], axis=0)
        stack_vk = np.stack([f["vk"] for f in frames_use], axis=0)
        stack_T = np.stack([f["T"] for f in frames_use], axis=0)
        mean_vi = stack_vi.mean(axis=0)
        mean_vj = stack_vj.mean(axis=0)
        mean_vk = stack_vk.mean(axis=0)
        global_mean_T = stack_T.mean(axis=0)

        scale_vi = float(stack_vi.std() + 1e-8)
        scale_vj = float(stack_vj.std() + 1e-8)
        scale_vk = float(stack_vk.std() + 1e-8)
        scale_T = float(stack_T.std() + 1e-8)
        n_pix = int(np.prod(mean_vi.shape))

        snaps = []
        for f in frames_use:
            if use_param_mean:
                assert param_mean_predictor is not None
                mean_T = param_mean_predictor.predict(f["nh3"], f["phi"])
            else:
                mean_T = global_mean_T
            snaps.append(
                cls._pack(
                    f["vi"],
                    f["vj"],
                    f["vk"],
                    f["T"],
                    mean_vi=mean_vi,
                    mean_vj=mean_vj,
                    mean_vk=mean_vk,
                    mean_T=mean_T,
                    scale_vi=scale_vi,
                    scale_vj=scale_vj,
                    scale_vk=scale_vk,
                    scale_T=scale_T,
                    n_velocity_components=n_velocity_components,
                )
            )
        X = np.stack(snaps, axis=0)  # (N, n_state)
        svd = TruncatedSVD(n_components=R, random_state=seed)
        svd.fit(X)
        modes = svd.components_.T  # (n_state, R)
        # Empirical modal variances (covariance eigenvalues along POD coords)
        modal_variances = np.asarray(svd.explained_variance_, dtype=np.float64)
        modal_variances = np.maximum(modal_variances, 1e-12)
        return cls(
            mean_vi=mean_vi,
            mean_vj=mean_vj,
            mean_vk=mean_vk,
            mean_T=global_mean_T,
            modes=modes,
            modal_variances=modal_variances,
            n_pix=n_pix,
            R=R,
            ridge_lambda=ridge_lambda,
            w_T=w_T,
            scale_vi=scale_vi,
            scale_vj=scale_vj,
            scale_vk=scale_vk,
            scale_T=scale_T,
            use_param_mean=use_param_mean,
            n_velocity_components=n_velocity_components,
            param_mean_predictor=param_mean_predictor,
            global_mean_T=global_mean_T,
        )

    def with_hyperparams(self, *, ridge_lambda: float | None = None, w_T: float | None = None) -> "GappyPODReconstructor":
        """Shallow copy with updated MAP hyperparameters (shared modes)."""
        return GappyPODReconstructor(
            mean_vi=self.mean_vi,
            mean_vj=self.mean_vj,
            mean_vk=self.mean_vk,
            mean_T=self.mean_T,
            modes=self.modes,
            modal_variances=self.modal_variances,
            n_pix=self.n_pix,
            R=self.R,
            ridge_lambda=self.ridge_lambda if ridge_lambda is None else float(ridge_lambda),
            w_T=self.w_T if w_T is None else float(w_T),
            scale_vi=self.scale_vi,
            scale_vj=self.scale_vj,
            scale_vk=self.scale_vk,
            scale_T=self.scale_T,
            use_param_mean=self.use_param_mean,
            n_velocity_components=self.n_velocity_components,
            param_mean_predictor=self.param_mean_predictor,
            global_mean_T=self.global_mean_T,
            eps_var=self.eps_var,
        )

    def _mean_T_for(self, nh3: float, phi: float) -> np.ndarray:
        if self.use_param_mean and self.param_mean_predictor is not None:
            return self.param_mean_predictor.predict(nh3, phi)
        return self.mean_T

    def project_oracle_T(
        self,
        T_true: np.ndarray,
        nh3: float = 0.0,
        phi: float = 1.0,
    ) -> np.ndarray:
        """Oracle full-field projection of ``T_true`` onto the R temperature POD modes.

        Quantifies the linear POD representation floor (all pixels of T observed).
        """
        H, W = T_true.shape
        mean_T = self._mean_T_for(nh3, phi)
        resid = ((np.asarray(T_true, dtype=np.float64) - mean_T) / self.scale_T).ravel()
        Phi_T = self.modes[self.t_offset : self.t_offset + self.n_pix, :]  # (n_pix, R)
        a, *_ = np.linalg.lstsq(Phi_T, resid, rcond=None)
        T_fluc = (Phi_T @ a).reshape(H, W) * self.scale_T
        return mean_T + T_fluc

    def reconstruct(
        self,
        vi: np.ndarray,
        vj: np.ndarray,
        vk: np.ndarray,
        *,
        vel_mask: np.ndarray,
        T_obs: np.ndarray | None = None,
        T_mask: np.ndarray | None = None,
        nh3: float = 0.0,
        phi: float = 1.0,
        use_temperature_block: bool = True,
        anchor_offset: bool = False,
        anchor_tau: float = 1.0,
    ) -> np.ndarray:
        """Gappy POD reconstruction of T from observed velocities (± sparse T).

        Uses block-normalized likelihood + variance-scaled ridge prior::

            min  (1/M_u)||Φ_u a - y_u||²
               + (w_T / max(K,1)) ||Φ_T a - y_T||²
               + λ Σ_r a_r² / (σ_r² + ε)

        If ``anchor_offset=True``, a global thermal offset is calibrated from the
        K temperature anchors on top of the (temperature-free or assisted) field.
        """
        H, W = vi.shape
        mean_T = self._mean_T_for(nh3, phi)

        vel_rows: list[np.ndarray] = []
        vel_vals: list[np.ndarray] = []

        def add_vel(field, mean, scale, offset):
            ys, xs = np.where(vel_mask)
            if ys.size == 0:
                return
            resid = ((field - mean) / scale)[ys, xs]
            lin = offset + ys * W + xs
            vel_vals.append(resid)
            vel_rows.append(lin)

        add_vel(vi, self.mean_vi, self.scale_vi, 0)
        add_vel(vj, self.mean_vj, self.scale_vj, self.n_pix)
        if self.n_velocity_components >= 3:
            add_vel(vk, self.mean_vk, self.scale_vk, 2 * self.n_pix)

        T_rows = None
        T_vals = None
        K = 0
        if (
            use_temperature_block
            and T_obs is not None
            and T_mask is not None
            and np.any(T_mask)
        ):
            ys, xs = np.where(T_mask)
            K = int(ys.size)
            if K > 0:
                T_vals = ((T_obs - mean_T) / self.scale_T)[ys, xs]
                T_rows = self.t_offset + ys * W + xs

        if not vel_rows and T_rows is None:
            T0 = mean_T.copy()
            if anchor_offset and T_obs is not None and T_mask is not None:
                T0, _ = apply_anchor_offset(T0, T_obs, T_mask, tau=anchor_tau)
            return T0

        prior = self.ridge_lambda / (self.modal_variances + self.eps_var)
        A = np.diag(prior)
        b = np.zeros(self.R, dtype=np.float64)

        if vel_rows:
            rows_u = np.concatenate(vel_rows)
            y_u = np.concatenate(vel_vals)
            M_u = max(int(y_u.size), 1)
            Phi_u = self.modes[rows_u, :]
            wu = 1.0 / M_u
            A = A + wu * (Phi_u.T @ Phi_u)
            b = b + wu * (Phi_u.T @ y_u)

        if T_rows is not None and T_vals is not None and K > 0:
            Phi_T = self.modes[T_rows, :]
            wt = self.w_T / max(K, 1)
            A = A + wt * (Phi_T.T @ Phi_T)
            b = b + wt * (Phi_T.T @ T_vals)

        a = np.linalg.solve(A, b)
        full = self.modes @ a
        T_fluc = full[self.t_offset : self.t_offset + self.n_pix].reshape(H, W) * self.scale_T
        T0 = mean_T + T_fluc
        if anchor_offset and T_obs is not None and T_mask is not None and np.any(T_mask):
            # Calibrate residual level after temperature-free (or assisted) reconstruct
            if use_temperature_block:
                # recompute temperature-free base then offset (cleaner level fix)
                T_tf = self.reconstruct(
                    vi,
                    vj,
                    vk,
                    vel_mask=vel_mask,
                    T_obs=None,
                    T_mask=None,
                    nh3=nh3,
                    phi=phi,
                    use_temperature_block=False,
                    anchor_offset=False,
                )
                T0, _ = apply_anchor_offset(T_tf, T_obs, T_mask, tau=anchor_tau)
            else:
                T0, _ = apply_anchor_offset(T0, T_obs, T_mask, tau=anchor_tau)
        return T0
