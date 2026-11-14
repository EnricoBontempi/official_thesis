"""Non-neural reconstruction baselines for Month 1 feasibility checks.

All predictors that use observations consume sparse tokens / boolean masks
and never treat unobserved locations as physical zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD


def _poly_ops(nh3_fraction: float, phi: float) -> np.ndarray:
    n = float(nh3_fraction)
    p = float(phi)
    return np.array([1.0, n, p, n * n, p * p, n * p], dtype=np.float64)


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
    """Ridge map from (NH3, phi, …) polynomial features → condition-mean T field."""

    coef: np.ndarray  # (6, H*W)
    H: int
    W: int
    ridge_alpha: float = 1e-2

    @classmethod
    def fit(
        cls,
        condition_mean_fields: Sequence[np.ndarray],
        op_params: Sequence[tuple[float, float]],
        *,
        ridge_alpha: float = 1e-2,
    ) -> "ParamConditionedMeanPredictor":
        assert len(condition_mean_fields) == len(op_params)
        H, W = condition_mean_fields[0].shape
        X = np.stack([_poly_ops(n, p) for n, p in op_params], axis=0)  # (C, 6)
        Y = np.stack([np.asarray(f, dtype=np.float64).ravel() for f in condition_mean_fields], axis=0)
        # Solve min ||X B - Y|| + alpha ||B|| for each pixel via normal equations
        xtx = X.T @ X + ridge_alpha * np.eye(X.shape[1])
        xty = X.T @ Y
        coef = np.linalg.solve(xtx, xty)  # (6, H*W)
        return cls(coef=coef, H=H, W=W, ridge_alpha=ridge_alpha)

    def predict(self, nh3_fraction: float, phi: float) -> np.ndarray:
        x = _poly_ops(nh3_fraction, phi)
        return (x @ self.coef).reshape(self.H, self.W)


@dataclass
class LocalVelocityTemperatureRegressor:
    """Pixel-wise tabular regressor: local velocity (+ optional op params) → T."""

    model: Any
    conditioned: bool
    feature_names: list[str]

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
    ) -> np.ndarray:
        """Build per-pixel feature matrix for one frame (H,W,*) → (H*W, F)."""
        H, W = vi.shape
        # 3x3 neighborhood mean/std of |u| via pad
        pad = np.pad(umag, 1, mode="edge")
        neigh = np.stack(
            [pad[i : i + H, j : j + W] for i in range(3) for j in range(3)],
            axis=0,
        )
        umag_mean = neigh.mean(axis=0)
        umag_std = neigh.std(axis=0)
        # simple central gradients of |u|
        dux = np.gradient(umag, axis=1)
        duy = np.gradient(umag, axis=0)
        cols = [
            x.ravel(),
            y.ravel(),
            vi.ravel(),
            vj.ravel(),
            vk.ravel(),
            umag.ravel(),
            umag_mean.ravel(),
            umag_std.ravel(),
            dux.ravel(),
            duy.ravel(),
        ]
        names = [
            "x",
            "y",
            "vi",
            "vj",
            "vk",
            "umag",
            "umag_mean3",
            "umag_std3",
            "dumag_dx",
            "dumag_dy",
        ]
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
    ) -> "LocalVelocityTemperatureRegressor":
        """Fit on a list of frame dicts with keys vi,vj,vk,umag,T,x2d,y2d,nh3,phi."""
        rng = np.random.default_rng(seed)
        X_list = []
        y_list = []
        names: list[str] | None = None
        per = max(1, max_pixels // max(1, len(frames)))
        for fr in frames:
            xx, yy = fr["x2d"], fr["y2d"]
            feats, names = cls._local_features(
                xx,
                yy,
                fr["vi"],
                fr["vj"],
                fr["vk"],
                fr["umag"],
                nh3=fr["nh3"] if conditioned else None,
                phi=fr["phi"] if conditioned else None,
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
        return cls(model=model, conditioned=conditioned, feature_names=names)

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
        feats, _ = self._local_features(
            x2d,
            y2d,
            vi,
            vj,
            vk,
            umag,
            nh3=nh3 if self.conditioned else None,
            phi=phi if self.conditioned else None,
        )
        pred = self.model.predict(feats)
        return pred.reshape(vi.shape)


@dataclass
class GappyPODReconstructor:
    """Cross-variable Gappy POD for sparse velocity (± sparse T) → full T.

    Modes are built on concatenated normalized ``[vi, vj, vk, T]`` fluctuations
    about a mean state (global or parameter-conditioned).
    """

    mean_vi: np.ndarray
    mean_vj: np.ndarray
    mean_vk: np.ndarray
    mean_T: np.ndarray  # used only if mean_T_fn is None
    modes: np.ndarray  # (n_pix_total, R) — concatenated variables
    n_pix: int
    R: int
    ridge_lambda: float
    scale_vi: float
    scale_vj: float
    scale_vk: float
    scale_T: float
    use_param_mean: bool
    param_mean_predictor: ParamConditionedMeanPredictor | None = None
    global_mean_T: np.ndarray | None = None

    @staticmethod
    def _pack(
        vi: np.ndarray,
        vj: np.ndarray,
        vk: np.ndarray,
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
    ) -> np.ndarray:
        return np.concatenate(
            [
                ((vi - mean_vi) / scale_vi).ravel(),
                ((vj - mean_vj) / scale_vj).ravel(),
                ((vk - mean_vk) / scale_vk).ravel(),
                ((T - mean_T) / scale_T).ravel(),
            ]
        )

    @classmethod
    def fit(
        cls,
        frames: Sequence[dict[str, Any]],
        *,
        R: int = 32,
        ridge_lambda: float = 1e-2,
        use_param_mean: bool = False,
        param_mean_predictor: ParamConditionedMeanPredictor | None = None,
        seed: int = 0,
        max_train_frames: int | None = 400,
    ) -> "GappyPODReconstructor":
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
                )
            )
        X = np.stack(snaps, axis=0)  # (N, 4*n_pix)
        # Economy SVD via TruncatedSVD (randomized) — memory friendly
        svd = TruncatedSVD(n_components=R, random_state=seed)
        svd.fit(X)
        modes = svd.components_.T  # (4*n_pix, R)
        return cls(
            mean_vi=mean_vi,
            mean_vj=mean_vj,
            mean_vk=mean_vk,
            mean_T=global_mean_T,
            modes=modes,
            n_pix=n_pix,
            R=R,
            ridge_lambda=ridge_lambda,
            scale_vi=scale_vi,
            scale_vj=scale_vj,
            scale_vk=scale_vk,
            scale_T=scale_T,
            use_param_mean=use_param_mean,
            param_mean_predictor=param_mean_predictor,
            global_mean_T=global_mean_T,
        )

    def _mean_T_for(self, nh3: float, phi: float) -> np.ndarray:
        if self.use_param_mean and self.param_mean_predictor is not None:
            return self.param_mean_predictor.predict(nh3, phi)
        return self.mean_T

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
    ) -> np.ndarray:
        """Gappy POD reconstruction of T from observed velocities (± sparse T).

        ``vel_mask`` / ``T_mask`` are True where observations exist. Unobserved
        locations contribute no equations (no zero fill).
        """
        H, W = vi.shape
        mean_T = self._mean_T_for(nh3, phi)
        # Observation residual vector
        obs_vals = []
        obs_rows = []  # row indices into concatenated state

        def add_var(field, mean, scale, mask, offset):
            ys, xs = np.where(mask)
            if ys.size == 0:
                return
            resid = ((field - mean) / scale)[ys, xs]
            lin = offset + ys * W + xs
            obs_vals.append(resid)
            obs_rows.append(lin)

        add_var(vi, self.mean_vi, self.scale_vi, vel_mask, 0)
        add_var(vj, self.mean_vj, self.scale_vj, vel_mask, self.n_pix)
        add_var(vk, self.mean_vk, self.scale_vk, vel_mask, 2 * self.n_pix)
        if T_obs is not None and T_mask is not None and T_mask.any():
            add_var(T_obs, mean_T, self.scale_T, T_mask, 3 * self.n_pix)

        if not obs_rows:
            return mean_T.copy()

        rows = np.concatenate(obs_rows)
        y = np.concatenate(obs_vals)
        Phi = self.modes[rows, :]  # (M, R)
        A = Phi.T @ Phi + self.ridge_lambda * np.eye(self.R)
        b = Phi.T @ y
        a = np.linalg.solve(A, b)
        full = self.modes @ a  # (4*n_pix,)
        T_fluc = full[3 * self.n_pix :].reshape(H, W) * self.scale_T
        return mean_T + T_fluc
