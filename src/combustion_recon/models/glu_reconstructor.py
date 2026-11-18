"""GLU adattato — ablation deterministica (thesis wrapper; upstream frozen in external/).

Not a full paper reproduction. Reuses vendored ``DomainAdaptiveEncoder`` /
``SoftDomainAdaptiveReconstructor`` without editing ``external/GLU_Reconstruction/``.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLU_CFG = ROOT / "configs" / "models" / "glu_minimal.yaml"
_GLU_MODELS_PATH = ROOT / "external" / "GLU_Reconstruction" / "src" / "models.py"

# Tie-break for decoder top-k: ``torch.topk(..., largest=False)`` uses index order on
# ties. Sensors are canonically sorted by ``site_index`` (ascending), so equal
# distances prefer the lower site index — no distance perturbation.


def _load_glu_upstream_models():
    """Load upstream ``models.py`` once without clashing with ``combustion_recon.models``."""
    spec = importlib.util.spec_from_file_location("glu_upstream_models", _GLU_MODELS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load GLU upstream from {_GLU_MODELS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["glu_upstream_models"] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _glu_upstream():
    return _load_glu_upstream_models()


def phys_to_isotropic_q(
    coords: torch.Tensor,
    *,
    center: torch.Tensor,
    L: torch.Tensor | float,
) -> torch.Tensor:
    """Map physical (x, y) to isotropic coordinates q = 2 * (r - center) / L."""
    if coords.shape[-1] != 2:
        raise ValueError(f"coords last dim must be 2, got {coords.shape}")
    c = center.view(-1, 1, 2) if center.ndim == 2 else center.view(1, 1, 2)
    if isinstance(L, torch.Tensor):
        scale = L.view(-1, 1, 1) if L.ndim == 1 else L
    else:
        scale = L
    return 2.0 * (coords - c) / (scale + 1e-12)


def compute_isotropic_geometry(
    sensor_coords_phys: torch.Tensor,
    *,
    query_coords_phys: torch.Tensor | None = None,
    units: str = "physical_from_cache",
) -> dict[str, Any]:
    """Center / L for q-map shared by sensors and queries."""
    if sensor_coords_phys.ndim != 3 or sensor_coords_phys.shape[-1] != 2:
        raise ValueError("sensor_coords_phys must be (B, N, 2)")
    b_pts = [sensor_coords_phys.reshape(-1, 2)]
    if query_coords_phys is not None:
        if query_coords_phys.ndim != 3 or query_coords_phys.shape[-1] != 2:
            raise ValueError("query_coords_phys must be (B, P, 2)")
        b_pts.append(query_coords_phys.reshape(-1, 2))
    pts = torch.cat(b_pts, dim=0)
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    extent_x = xmax - xmin
    extent_y = ymax - ymin
    L = torch.maximum(extent_x, extent_y).clamp_min(1e-12)
    center = torch.stack([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
    return {
        "center": center,
        "L": L,
        "extent_x": extent_x,
        "extent_y": extent_y,
        "units": units,
        "convention": "isotropic_q",
        "formula": "q = 2 * (r - center) / L",
        "L_definition": "max(extent_x, extent_y)",
    }


def _sort_sensors_by_site(
    site_index: torch.Tensor,
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Stable ascending sort by site index; tie-break preserves prior order."""
    if site_index.ndim != 2:
        raise ValueError("site_index must be (B, N)")
    order = torch.argsort(site_index, dim=1, stable=True)
    out: list[torch.Tensor] = [torch.gather(site_index, 1, order)]
    for t in tensors:
        if t.shape[:2] != site_index.shape:
            raise ValueError(f"shape mismatch: {t.shape} vs {site_index.shape}")
        idx = order.unsqueeze(-1).expand_as(t)
        out.append(torch.gather(t, 1, idx))
    return tuple(out)


def _glu_classes():
    mod = _glu_upstream()
    return mod.DomainAdaptiveEncoder, mod.SoftDomainAdaptiveReconstructor


_DomainAdaptiveEncoder, _SoftDomainAdaptiveReconstructor = _glu_classes()


class ThesisDomainAdaptiveEncoder(_DomainAdaptiveEncoder):
    """Encoder with additive global ``p`` at the token-embedding stage (thesis-only)."""

    def __init__(self, *args: Any, p_dim: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        all_dim = int(kwargs.get("All_dim") if "All_dim" in kwargs else args[0])
        self.p_dim = int(p_dim)
        self.p_proj = nn.Linear(self.p_dim, all_dim, bias=False)

    def forward(
        self,
        coords_tuv: torch.Tensor,
        U: torch.Tensor,
        original_phi: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
    ):
        del U  # upstream contract; unused in encoder

        B, T, N_inp, _ = coords_tuv.shape

        xy = coords_tuv[..., 0:2]
        if self.channel_to_encode is None:
            c_in = self.embed["val_linear"].in_features
            vals = coords_tuv[..., 2 : 2 + c_in]
        else:
            value_channels_in_tensor = [c + 2 for c in self.channel_to_encode]
            vals = coords_tuv[..., value_channels_in_tensor]

        pos_tok = self.embed["pos_linear"](self.embed["pos_embed"](xy))
        pos_tok = self.embed["pos_norm"](pos_tok)

        val_tok = self.embed["val_linear"](vals)
        val_tok = self.embed["val_norm"](val_tok)

        if p is None:
            p_tok = 0.0
        else:
            if p.shape != (B, self.p_dim):
                raise ValueError(f"p expected ({B},{self.p_dim}), got {tuple(p.shape)}")
            p_tok = self.p_proj(p).unsqueeze(1).unsqueeze(1)

        tok = pos_tok + val_tok + p_tok
        tok_flat = tok.reshape(B * T, N_inp, -1)

        lat = self.latents["latent_param"].expand(B * T, -1, -1)
        if self.retain_cls:
            cls = self.latents["cls_token"].expand(B * T, 1, -1)
            lat = torch.cat([cls, lat], dim=1)

        lat = self.cross_norm(lat)
        lat = lat + self.cross_attn(lat, tok_flat, tok_flat)
        lat = lat + self.cross_latent_mixer(lat)
        final_latents = lat

        tok_flat_up = self.token_to_latent_norm(tok_flat)
        tok_flat_up = tok_flat_up + self.token_to_latent(tok_flat_up, lat, lat)
        tok_flat_out = tok_flat_up + self.token_to_latent_mixer(tok_flat_up)

        if self.retain_lat:
            tok_flat_out = torch.cat([final_latents, tok_flat_out], dim=1)
        elif self.retain_cls:
            cls_upd = lat[:, :1, :]
            tok_flat_out = torch.cat([cls_upd, tok_flat_out], dim=1)

        S = tok_flat_out.size(1)
        lat_vec = tok_flat_out.view(B, T, S, -1)

        mask = torch.ones(B, T, S, dtype=torch.bool, device=lat_vec.device)
        coords = coords_tuv[:, 0, :, :2]
        merged_phi = (
            original_phi
            if original_phi is not None
            else torch.ones(B, S, device=lat_vec.device)
        )

        return lat_vec, mask, coords, merged_phi


class AdaptedDeterministicGLU(nn.Module):
    """Single-frame sparse GLU reconstructor with isotropic coords and additive ``p``."""

    def __init__(
        self,
        *,
        height: int = 128,
        width: int = 128,
        All_dim: int = 32,
        num_heads: int = 4,
        latent_layers: int = 4,
        encoder_N_channels: int = 2,
        encoder_num_freqs: int = 256,
        latent_tokens: int = 64,
        retain_cls: bool = True,
        retain_lat: bool = False,
        decoder_num_freqs: int = 64,
        decoder_dropout: float = 0.0,
        decoder_N_channels: int = 1,
        top_k: int = 16,
        bandwidth_init: float = 0.05,
        per_sensor_sigma: bool = False,
        CalRecVar: bool = False,
        USE_FINAL_MLP: bool = False,
        importance_scale: float = 0.50,
        overlap_ratio: float = 0.02,
        use_weighted_fusion: bool = False,
        decode_chunk_size: int = 512,
        decode_chunked_eval: bool = True,
        geometry_units: str = "physical_from_cache",
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.decode_chunk_size = int(decode_chunk_size)
        self.decode_chunked_eval = bool(decode_chunked_eval)
        self._geometry_units = str(geometry_units)
        self._geometry_meta: dict[str, Any] | None = None

        self.encoder = ThesisDomainAdaptiveEncoder(
            All_dim=int(All_dim),
            num_heads=int(num_heads),
            latent_layers=int(latent_layers),
            N_channels=int(encoder_N_channels),
            num_freqs=int(encoder_num_freqs),
            latent_tokens=int(latent_tokens),
            pooling="none",
            retain_cls=bool(retain_cls),
            retain_lat=bool(retain_lat),
            channel_to_encode=None,
            p_dim=2,
        )
        self.decoder = _SoftDomainAdaptiveReconstructor(
            d_model=int(All_dim),
            num_heads=int(num_heads),
            N_channels=int(decoder_N_channels),
            latent_tokens=int(latent_tokens),
            num_freqs=int(decoder_num_freqs),
            dropout=float(decoder_dropout),
            overlap_ratio=float(overlap_ratio),
            importance_scale=float(importance_scale),
            bandwidth_init=float(bandwidth_init),
            top_k=int(top_k),
            per_sensor_sigma=bool(per_sensor_sigma),
            CalRecVar=bool(CalRecVar),
            retain_cls=bool(retain_cls),
            retain_lat=bool(retain_lat),
            use_checkpoint=False,
            USE_FINAL_MLP=bool(USE_FINAL_MLP),
            use_weighted_fusion=bool(use_weighted_fusion),
        )

    @property
    def geometry_meta(self) -> dict[str, Any] | None:
        return self._geometry_meta

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def decode_chunked(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        sensor_q: torch.Tensor,
        Y_q: torch.Tensor,
        *,
        phi_mean: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Query decoder in chunks via public ``SoftDomainAdaptiveReconstructor.forward``."""
        cs = int(chunk_size) if chunk_size is not None else self.decode_chunk_size
        P = Y_q.size(1)
        if P <= cs:
            _, out_mean, _ = self.decoder(z, Y_q, sensor_q, mask, phi_mean=phi_mean)
            return out_mean

        parts: list[torch.Tensor] = []
        for start in range(0, P, cs):
            y_chunk = Y_q[:, start : start + cs, :]
            _, out_mean, _ = self.decoder(z, y_chunk, sensor_q, mask, phi_mean=phi_mean)
            parts.append(out_mean)
        return torch.cat(parts, dim=2)

    def forward(
        self,
        sensor_coords_phys: torch.Tensor | Mapping[str, torch.Tensor],
        sensor_values_norm: torch.Tensor | None = None,
        site_index: torch.Tensor | None = None,
        p_norm: torch.Tensor | None = None,
        query_coords_phys: torch.Tensor | None = None,
        *,
        p_conditioning: bool = True,
        decode_chunked: bool | None = None,
        chunk_size: int | None = None,
        return_flat: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            raise ValueError("padding_mask is not supported; batch must have homogeneous N")

        if isinstance(sensor_coords_phys, Mapping):
            batch = sensor_coords_phys
            sensor_coords_phys = batch["sensor_coords_phys"]
            sensor_values_norm = batch["sensor_values_norm"]
            site_index = batch["site_index"]
            p_norm = batch.get("p_norm", p_norm)
            query_coords_phys = batch.get("query_coords_phys", query_coords_phys)
            if "p_conditioning" in batch:
                p_conditioning = bool(batch["p_conditioning"])
            if "padding_mask" in batch and batch["padding_mask"] is not None:
                raise ValueError("padding_mask is not supported; batch must have homogeneous N")

        if sensor_coords_phys.ndim != 3 or sensor_values_norm is None or site_index is None:
            raise ValueError("expected sensor_coords_phys (B,N,2), sensor_values_norm (B,N,2), site_index (B,N)")
        if p_norm is None:
            raise ValueError("p_norm (B, 2) is required")
        if query_coords_phys is None:
            raise ValueError("query_coords_phys (B, P, 2) is required")

        bsz, n_sens = sensor_coords_phys.shape[:2]
        if sensor_values_norm.shape[:2] != (bsz, n_sens):
            raise ValueError("homogeneous N required across batch")
        if site_index.shape != (bsz, n_sens):
            raise ValueError("site_index shape mismatch")

        site_index, sensor_coords_phys, sensor_values_norm = _sort_sensors_by_site(
            site_index, sensor_coords_phys, sensor_values_norm
        )

        if p_norm.shape != (bsz, 2):
            raise ValueError(f"p_norm expected ({bsz}, 2), got {tuple(p_norm.shape)}")

        if not p_conditioning:
            p_norm = torch.zeros_like(p_norm)

        geom = compute_isotropic_geometry(
            sensor_coords_phys,
            query_coords_phys=query_coords_phys,
            units=self._geometry_units,
        )
        center = geom["center"].detach().to(sensor_coords_phys.device, sensor_coords_phys.dtype)
        L = geom["L"].detach().to(sensor_coords_phys.device, sensor_coords_phys.dtype)
        self._geometry_meta = {
            "center": center.detach().cpu().tolist(),
            "L": float(L.detach().cpu()),
            "extent_x": float(geom["extent_x"].detach().cpu()),
            "extent_y": float(geom["extent_y"].detach().cpu()),
            "units": geom["units"],
            "convention": geom["convention"],
            "formula": geom["formula"],
            "L_definition": geom["L_definition"],
        }

        sensor_q = phys_to_isotropic_q(sensor_coords_phys, center=center, L=L)
        query_q = phys_to_isotropic_q(query_coords_phys, center=center, L=L)

        vi = sensor_values_norm[..., 0:1]
        vj = sensor_values_norm[..., 1:2]
        coords_tuv = torch.cat([sensor_q, vi, vj], dim=-1).unsqueeze(1)

        u_dummy = torch.zeros(bsz, device=sensor_coords_phys.device, dtype=sensor_coords_phys.dtype)
        z, mask, enc_sensor_q, _ = self.encoder(coords_tuv, u_dummy, original_phi=None, p=p_norm)

        phi_mean = torch.ones(bsz, n_sens, device=sensor_coords_phys.device, dtype=sensor_coords_phys.dtype)

        use_chunks = self.decode_chunked_eval if decode_chunked is None else bool(decode_chunked)
        if use_chunks:
            out_t = self.decode_chunked(
                z,
                mask,
                enc_sensor_q,
                query_q,
                phi_mean=phi_mean,
                chunk_size=chunk_size,
            )
        else:
            _, out_t, _ = self.decoder(z, query_q, enc_sensor_q, mask, phi_mean=phi_mean)

        flat = out_t.squeeze(1).squeeze(-1)
        if return_flat:
            return flat

        p_pts = flat.shape[1]
        hw = self.height * self.width
        if p_pts != hw:
            raise ValueError(
                f"cannot reshape P={p_pts} to (H,W)=({self.height},{self.width}); use return_flat=True"
            )
        return flat.view(bsz, 1, self.height, self.width)


def _load_glu_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("glu config must be a mapping")
    required = [
        "architecture",
        "geometry",
        "input",
        "output",
        "encoder",
        "decoder",
        "inference",
    ]
    for k in required:
        if k not in cfg:
            raise ValueError(f"glu config missing required key {k!r}")
    allowed = {
        "model_id",
        "architecture",
        "task",
        "geometry",
        "input",
        "output",
        "encoder",
        "decoder",
        "inference",
        "provenance",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"glu config has unsupported keys: {sorted(unknown)}")
    if cfg["architecture"] != "adapted_deterministic_glu":
        raise ValueError(f"unsupported architecture {cfg['architecture']!r}")
    return cfg


def load_glu_minimal_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_GLU_CFG
    return _load_glu_yaml(p)


def build_glu_from_config(
    path: str | Path | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
) -> AdaptedDeterministicGLU:
    cfg = load_glu_minimal_config(path)
    geom = cfg["geometry"]
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    inf = cfg["inference"]
    h = int(height) if height is not None else int(geom["spatial"][0])
    w = int(width) if width is not None else int(geom["spatial"][1])
    return AdaptedDeterministicGLU(
        height=h,
        width=w,
        All_dim=int(enc["All_dim"]),
        num_heads=int(enc["num_heads"]),
        latent_layers=int(enc["latent_layers"]),
        encoder_N_channels=int(enc["N_channels"]),
        encoder_num_freqs=int(enc["num_freqs"]),
        latent_tokens=int(enc["latent_tokens"]),
        retain_cls=bool(enc["retain_cls"]),
        retain_lat=bool(enc["retain_lat"]),
        decoder_num_freqs=int(dec["num_freqs"]),
        decoder_dropout=float(dec["dropout"]),
        decoder_N_channels=int(dec["N_channels"]),
        top_k=int(dec["top_k"]),
        bandwidth_init=float(dec["bandwidth_init"]),
        per_sensor_sigma=bool(dec["per_sensor_sigma"]),
        CalRecVar=bool(dec["CalRecVar"]),
        USE_FINAL_MLP=bool(dec["USE_FINAL_MLP"]),
        importance_scale=float(dec["importance_scale"]),
        overlap_ratio=float(dec["overlap_ratio"]),
        use_weighted_fusion=bool(dec["use_weighted_fusion"]),
        decode_chunk_size=int(inf.get("decode_chunk_size", 512)),
        decode_chunked_eval=bool(inf.get("decode_chunked_default", True)),
        geometry_units=str(geom.get("units", "physical_from_cache")),
    )


def make_synthetic_query_grid(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Row-major (H*W, 2) query coordinates from 2-D physical coord maps."""
    xx, yy = np.meshgrid(np.asarray(x_phys, dtype=np.float64), np.asarray(y_phys, dtype=np.float64), indexing="xy")
    q = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return torch.from_numpy(q).to(device=device, dtype=dtype)