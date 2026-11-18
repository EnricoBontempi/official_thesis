"""GLU adattato — ablation deterministica (thesis wrapper; upstream frozen in external/).

Not a full paper reproduction. Reuses vendored ``DomainAdaptiveEncoder`` /
``SoftDomainAdaptiveReconstructor`` without editing ``external/GLU_Reconstruction/``.

Decoder delta (thesis-only): top-k neighbor selection uses lexicographic order
``(distance ascending, site_index ascending)`` via stable double-argsort. Upstream
``torch.topk`` does **not** guarantee lower-index preference on ties; sorting sensors
alone is insufficient. Physical distances are not jittered.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLU_CFG = ROOT / "configs" / "models" / "glu_minimal.yaml"
DEFAULT_DOMAIN_GEOMETRY = ROOT / "data" / "processed" / "month2_dense_unet_cache" / "geometry.npz"
_GLU_MODELS_PATH = ROOT / "external" / "GLU_Reconstruction" / "src" / "models.py"


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
    c = center.to(device=coords.device, dtype=coords.dtype)
    if c.ndim == 1:
        c = c.view(1, 1, 2)
    elif c.ndim == 2:
        c = c.view(-1, 1, 2)
    if isinstance(L, torch.Tensor):
        scale = L.to(device=coords.device, dtype=coords.dtype)
        scale = scale.view(-1, 1, 1) if scale.ndim > 0 else scale
    else:
        scale = float(L)
    return 2.0 * (coords - c) / (scale + 1e-12)


def domain_geometry_from_axes(
    x_phys: np.ndarray | torch.Tensor | Sequence[float],
    y_phys: np.ndarray | torch.Tensor | Sequence[float],
    *,
    units: str = "physical_from_cache",
    source: str = "explicit_axes",
) -> dict[str, Any]:
    """Fixed domain center/L from full 1-D axis arrays (not from sparse sensors/queries)."""
    x = np.asarray(x_phys, dtype=np.float64).reshape(-1)
    y = np.asarray(y_phys, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        raise ValueError("domain axes must have length >= 2")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    extent_x = xmax - xmin
    extent_y = ymax - ymin
    if extent_x <= 0 or extent_y <= 0:
        raise ValueError("degenerate domain extents")
    L = max(extent_x, extent_y)
    center = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
    return {
        "center": center,
        "L": L,
        "extent_x": extent_x,
        "extent_y": extent_y,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "units": units,
        "convention": "isotropic_q",
        "formula": "q = 2 * (r - center) / L",
        "L_definition": "max(extent_x, extent_y)",
        "source": source,
        "n_x": int(x.size),
        "n_y": int(y.size),
    }


def domain_geometry_from_center_L(
    center: Sequence[float] | np.ndarray,
    L: float,
    *,
    extent_x: float | None = None,
    extent_y: float | None = None,
    units: str = "synthetic_explicit",
    source: str = "explicit_center_L",
) -> dict[str, Any]:
    c = tuple(float(v) for v in np.asarray(center, dtype=np.float64).reshape(-1))
    if len(c) != 2:
        raise ValueError("center must be length-2")
    Lf = float(L)
    if Lf <= 0:
        raise ValueError("L must be > 0")
    ex = float(extent_x) if extent_x is not None else Lf
    ey = float(extent_y) if extent_y is not None else Lf
    return {
        "center": c,
        "L": Lf,
        "extent_x": ex,
        "extent_y": ey,
        "units": units,
        "convention": "isotropic_q",
        "formula": "q = 2 * (r - center) / L",
        "L_definition": "max(extent_x, extent_y)",
        "source": source,
    }


def load_domain_geometry_npz(
    path: str | Path,
    *,
    units: str = "physical_from_cache",
) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"domain geometry file missing: {p}")
    data = np.load(p)
    if "x" not in data or "y" not in data:
        raise ValueError(f"geometry file must contain x/y axes: {p}")
    return domain_geometry_from_axes(
        data["x"], data["y"], units=units, source=str(p)
    )


def topk_distance_then_site_index(
    d_scaled: torch.Tensor,
    site_index: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Select k neighbors with criterion (distance↑, site_index↑).

    Uses stable double-argsort (lexicographic). Does **not** jitter distances.
    ``torch.topk`` is intentionally not used for ties.
    """
    if d_scaled.ndim != 3:
        raise ValueError("d_scaled must be (B,P,S)")
    if site_index.ndim != 2 or site_index.shape[0] != d_scaled.shape[0]:
        raise ValueError("site_index must be (B,S)")
    if site_index.shape[1] != d_scaled.shape[2]:
        raise ValueError("site_index S must match d_scaled")
    bsz, n_q, n_s = d_scaled.shape
    k_eff = min(int(k), n_s)
    site = site_index.to(dtype=d_scaled.dtype).unsqueeze(1).expand(bsz, n_q, n_s)
    order_site = torch.argsort(site, dim=2, stable=True)
    d_by_site = torch.gather(d_scaled, 2, order_site)
    order_d = torch.argsort(d_by_site, dim=2, stable=True)
    full = torch.gather(order_site, 2, order_d)
    return full[..., :k_eff].contiguous()


def _sort_sensors_by_site(
    site_index: torch.Tensor,
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Ascending sort by site index (canonical packet order)."""
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
        del U

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


class ThesisSoftDomainAdaptiveReconstructor(_SoftDomainAdaptiveReconstructor):
    """Upstream reconstructor with thesis top-k tie-break (distance, then site_index).

    Public forward path of the release is preserved except neighbor index selection:
    ``torch.topk`` → :func:`topk_distance_then_site_index`. No distance jitter.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_top_idx: torch.Tensor | None = None
        self._record_top_idx = False

    def forward(
        self,
        z: torch.Tensor,
        Y: torch.Tensor,
        sensor_coords: torch.Tensor,
        mask: torch.Tensor,
        phi_mean: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        site_index: torch.Tensor | None = None,
    ):
        # Body follows SoftDomainAdaptiveReconstructor.forward (release), with the
        # single selection delta documented above.
        B, T, S_or_Sp1, D_raw = z.shape
        del S_or_Sp1, D_raw
        P = Y.size(1)
        C = self.N_channels
        d_model = self.lat_proj.out_features
        S = sensor_coords.size(1)

        if phi_mean is None:
            phi_mean = torch.ones(B, S, device=z.device)

        if self.per_sensor_sigma:
            if (self.log_sigma is None) or (self.log_sigma.numel() != S):
                self.log_sigma = nn.Parameter(
                    torch.full((S,), math.log(self.bandwidth_init), device=z.device)
                )
            sigma = self.log_sigma.exp()
        else:
            sigma = torch.tensor(self.bandwidth_init, device=z.device)

        effective_mask = mask
        if padding_mask is not None:
            padding_bt = padding_mask.unsqueeze(1).expand(-1, T, -1)
            effective_mask = mask & padding_bt

        lat_proj = self.lat_proj(z)

        if self.retain_cls:
            assert lat_proj.size(2) == S + 1, "retain_cls=True requires z to include CLS at index 0."
            cls_proj = lat_proj[:, :, 0, :]
            sensor_proj = lat_proj[:, :, 1:, :]
            sensor_mask = effective_mask[:, :, 1:]
        else:
            sensor_proj = lat_proj
            sensor_mask = effective_mask

        coord_tok = self.coord_proj(self.pe(Y))
        coord_tok = self.coord_norm(coord_tok)
        coord_tok = coord_tok.unsqueeze(1).expand(B, T, P, d_model).reshape(B * T, P, d_model)

        d = torch.cdist(Y, sensor_coords)

        sensor_valid_bs = sensor_mask[:, 0, :]
        if padding_mask is not None and padding_mask.dim() == 2:
            sensor_valid_bs = sensor_valid_bs & padding_mask

        gamma = self.importance_scale
        d = d.masked_fill(~sensor_valid_bs.unsqueeze(1), float("inf"))
        phi = phi_mean.detach()
        d_scaled = d / (phi[:, None, :] ** gamma + 1e-6)

        local_S = sensor_proj.size(2)
        if self.top_k is None:
            k_eff = local_S
        else:
            k_eff = min(self.top_k, local_S)

        if site_index is None:
            site_index = torch.arange(S, device=z.device).unsqueeze(0).expand(B, -1)
        if site_index.shape != (B, S):
            raise ValueError(f"site_index expected {(B, S)}, got {tuple(site_index.shape)}")

        # DELTA vs release: lexicographic (distance, site_index), not torch.topk.
        top_idx = topk_distance_then_site_index(d_scaled, site_index, k_eff)
        if self._record_top_idx:
            self.last_top_idx = top_idx.detach()

        d_k = torch.gather(d_scaled, 2, top_idx)

        phi_expan = phi[:, None, :].expand(-1, P, -1)
        phi_k = torch.gather(phi_expan, 2, top_idx)

        if self.per_sensor_sigma:
            sigma_k = sigma[top_idx]
        else:
            sigma_k = sigma

        scores = -d_k / sigma_k
        scores -= scores.max(dim=-1, keepdim=True).values
        exp = torch.exp(scores)

        valid_k = torch.gather(
            sensor_valid_bs.unsqueeze(1).expand(-1, P, -1), 2, top_idx
        )
        exp = exp * valid_k.float()
        weights = exp / (exp.sum(dim=-1, keepdim=True) + 1e-6)

        if self.use_weighted_fusion:
            weights = weights * (phi_k * self.phi_scale)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        h, stats = self._topk_aggregate(
            sensor_proj, top_idx, weights, d_k, phi_k, valid_k, local_S
        )
        if not self.CalRecVar:
            stats = None

        h = self.agg_norm(h)
        lat = h.reshape(B * T, P, d_model)

        if hasattr(self.cross_attn, "to_v"):
            v_proj = self.cross_attn.to_v
            out_proj = self.cross_attn.out
        else:
            mha = self.cross_attn.attn
            dim = mha.embed_dim
            v_weight = mha.in_proj_weight[2 * dim : 3 * dim]
            v_bias = mha.in_proj_bias[2 * dim : 3 * dim] if mha.in_proj_bias is not None else None

            def v_proj(x):
                return F.linear(x, v_weight, v_bias)

            out_proj = mha.out_proj

        local_lat = out_proj(v_proj(lat))
        local_out_mean = None
        local_pre = coord_tok + local_lat

        if self.retain_cls:
            fused_pre = self.SpatiallyAwareCLSFusion(
                local_pre, cls_proj.reshape(B * T, 1, d_model)
            )
            fused_x = self.norm(fused_pre)
            if self.USE_FINAL_MLP:
                fused_x = fused_x + self.mlp(fused_x)
            out_mean = self.head(fused_x).view(B, T, P, C)
            x_for_var = fused_x
        else:
            local_x = self.norm(local_pre)
            if self.USE_FINAL_MLP:
                local_x = local_x + self.mlp(local_x)
            local_out_mean = self.head(local_x).view(B, T, P, C)
            out_mean = local_out_mean
            x_for_var = local_x

        out_logvar = None
        if self.CalRecVar:
            stats_emb = self.stats_proj(stats)
            stats_emb = stats_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, P, -1)
            x_var = torch.cat([x_for_var, stats_emb], dim=-1)
            logvar = self.var_head(x_var)
            out_logvar = logvar.view(B, T, P, C)

        return local_out_mean, out_mean, out_logvar


class AdaptedDeterministicGLU(nn.Module):
    """Single-frame sparse GLU reconstructor with fixed domain geometry and additive ``p``."""

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
        domain_geometry: dict[str, Any] | None = None,
        domain_x: np.ndarray | Sequence[float] | None = None,
        domain_y: np.ndarray | Sequence[float] | None = None,
        domain_center: Sequence[float] | None = None,
        domain_L: float | None = None,
        domain_geometry_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.decode_chunk_size = int(decode_chunk_size)
        self.decode_chunked_eval = bool(decode_chunked_eval)
        self._geometry_units = str(geometry_units)

        geom = self._resolve_domain_geometry(
            domain_geometry=domain_geometry,
            domain_x=domain_x,
            domain_y=domain_y,
            domain_center=domain_center,
            domain_L=domain_L,
            domain_geometry_path=domain_geometry_path,
            units=geometry_units,
        )
        self.register_buffer(
            "domain_center",
            torch.tensor(geom["center"], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "domain_L",
            torch.tensor(float(geom["L"]), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "domain_extent_x",
            torch.tensor(float(geom["extent_x"]), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "domain_extent_y",
            torch.tensor(float(geom["extent_y"]), dtype=torch.float32),
            persistent=True,
        )
        self._domain_source = str(geom.get("source", "unknown"))
        self._domain_units = str(geom.get("units", geometry_units))

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
        self.decoder = ThesisSoftDomainAdaptiveReconstructor(
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

    @staticmethod
    def _resolve_domain_geometry(
        *,
        domain_geometry: dict[str, Any] | None,
        domain_x,
        domain_y,
        domain_center,
        domain_L,
        domain_geometry_path,
        units: str,
    ) -> dict[str, Any]:
        if domain_geometry is not None:
            return dict(domain_geometry)
        if domain_x is not None and domain_y is not None:
            return domain_geometry_from_axes(domain_x, domain_y, units=units, source="ctor_axes")
        if domain_center is not None and domain_L is not None:
            return domain_geometry_from_center_L(
                domain_center, float(domain_L), units=units, source="ctor_center_L"
            )
        if domain_geometry_path is not None:
            return load_domain_geometry_npz(domain_geometry_path, units=units)
        raise ValueError(
            "domain geometry required: pass domain_geometry, domain_x/domain_y, "
            "domain_center/domain_L, or domain_geometry_path (no silent first-batch inference)"
        )

    @property
    def geometry_meta(self) -> dict[str, Any]:
        return {
            "center": self.domain_center.detach().cpu().tolist(),
            "L": float(self.domain_L.detach().cpu()),
            "extent_x": float(self.domain_extent_x.detach().cpu()),
            "extent_y": float(self.domain_extent_y.detach().cpu()),
            "units": self._domain_units,
            "convention": "isotropic_q",
            "formula": "q = 2 * (r - center) / L",
            "L_definition": "max(extent_x, extent_y)",
            "source": self._domain_source,
            "persistent_buffers": ["domain_center", "domain_L", "domain_extent_x", "domain_extent_y"],
        }

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
        site_index: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Query decoder in chunks; geometry already fixed on the module (not recomputed)."""
        cs = int(chunk_size) if chunk_size is not None else self.decode_chunk_size
        P = Y_q.size(1)
        if P <= cs:
            _, out_mean, _ = self.decoder(
                z, Y_q, sensor_q, mask, phi_mean=phi_mean, site_index=site_index
            )
            return out_mean

        parts: list[torch.Tensor] = []
        for start in range(0, P, cs):
            y_chunk = Y_q[:, start : start + cs, :]
            _, out_mean, _ = self.decoder(
                z, y_chunk, sensor_q, mask, phi_mean=phi_mean, site_index=site_index
            )
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
            raise ValueError(
                "expected sensor_coords_phys (B,N,2), sensor_values_norm (B,N,2), site_index (B,N)"
            )
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

        # Fixed domain geometry from buffers — independent of sensors/queries/batch mates.
        center = self.domain_center
        L = self.domain_L

        sensor_q = phys_to_isotropic_q(sensor_coords_phys, center=center, L=L)
        query_q = phys_to_isotropic_q(query_coords_phys, center=center, L=L)

        vi = sensor_values_norm[..., 0:1]
        vj = sensor_values_norm[..., 1:2]
        coords_tuv = torch.cat([sensor_q, vi, vj], dim=-1).unsqueeze(1)

        u_dummy = torch.zeros(bsz, device=sensor_coords_phys.device, dtype=sensor_coords_phys.dtype)
        z, mask, enc_sensor_q, _ = self.encoder(coords_tuv, u_dummy, original_phi=None, p=p_norm)

        phi_mean = torch.ones(
            bsz, n_sens, device=sensor_coords_phys.device, dtype=sensor_coords_phys.dtype
        )

        use_chunks = self.decode_chunked_eval if decode_chunked is None else bool(decode_chunked)
        if use_chunks:
            out_t = self.decode_chunked(
                z,
                mask,
                enc_sensor_q,
                query_q,
                phi_mean=phi_mean,
                site_index=site_index,
                chunk_size=chunk_size,
            )
        else:
            _, out_t, _ = self.decoder(
                z,
                query_q,
                enc_sensor_q,
                mask,
                phi_mean=phi_mean,
                site_index=site_index,
            )

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
    domain_geometry: dict[str, Any] | None = None,
    domain_x: np.ndarray | Sequence[float] | None = None,
    domain_y: np.ndarray | Sequence[float] | None = None,
    domain_center: Sequence[float] | None = None,
    domain_L: float | None = None,
    domain_geometry_path: str | Path | None = None,
) -> AdaptedDeterministicGLU:
    cfg = load_glu_minimal_config(path)
    geom = cfg["geometry"]
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    inf = cfg["inference"]
    h = int(height) if height is not None else int(geom["spatial"][0])
    w = int(width) if width is not None else int(geom["spatial"][1])

    # Resolve domain: explicit kwargs win; else YAML path; else default cache geometry.
    if (
        domain_geometry is None
        and domain_x is None
        and domain_y is None
        and domain_center is None
        and domain_L is None
        and domain_geometry_path is None
    ):
        yaml_path = geom.get("domain_geometry_path")
        if yaml_path:
            domain_geometry_path = ROOT / str(yaml_path)
        else:
            domain_geometry_path = DEFAULT_DOMAIN_GEOMETRY

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
        domain_geometry=domain_geometry,
        domain_x=domain_x,
        domain_y=domain_y,
        domain_center=domain_center,
        domain_L=domain_L,
        domain_geometry_path=domain_geometry_path,
    )


def make_synthetic_query_grid(
    x_phys: np.ndarray,
    y_phys: np.ndarray,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Row-major (H*W, 2) query coordinates from 1-D physical axes."""
    xx, yy = np.meshgrid(
        np.asarray(x_phys, dtype=np.float64),
        np.asarray(y_phys, dtype=np.float64),
        indexing="xy",
    )
    q = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return torch.from_numpy(q).to(device=device, dtype=dtype)


# Backward-compatible name kept for imports that referenced the old helper.
def compute_isotropic_geometry(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Deprecated call-site helper — raises; use domain geometry APIs instead."""
    raise RuntimeError(
        "compute_isotropic_geometry(from sensors∪queries) is removed: "
        "use domain_geometry_from_axes / load_domain_geometry_npz / "
        "domain_geometry_from_center_L and AdaptedDeterministicGLU buffers"
    )
