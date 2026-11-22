"""Temperature-assisted adapted GLU (temperature study). Minimal, documented adaptation of the free model.

Changes with respect to ``AdaptedDeterministicGLU`` (free GLU and its results are NOT modified; this is a new path):

1. Typed T tokens. Observed temperature anchors enter the encoder as extra tokens built like the velocity tokens
   (position embedding of the coordinate) with a 1-channel value embedding ``t_val_linear/t_val_norm`` and a learned
   type vector ``t_type``. Velocity tokens are unchanged. K=0 means no T token (pure velocity path).
2. Stable neighbour selection. The decoder's distance ``torch.cdist`` is replaced by the EXACT lattice distance
   sqrt((qx*dix)^2 + (qy*diy)^2) computed from the integer grid offsets of query and sensor sites. The grid is regular, so
   equal distances (mirror neighbours) are bit-identical and the existing (distance, site index) tie-break is
   deterministic and independent of batch shape; nothing depends on matmul-based distances.
Shared weights start from the free GLU init of the same seed (the extra modules are created afterwards).
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from combustion_recon.models import glu_reconstructor as G


def _lattice_cdist_factory(query_flat: torch.Tensor, site_flat: torch.Tensor, qstep: tuple[float, float], width: int):
    iy_q, ix_q = query_flat // width, query_flat % width                      # (P,)
    iy_s, ix_s = site_flat // width, site_flat % width                        # (B,S)

    def fn(Y, S, *args, **kwargs):  # signature of torch.cdist(Y, sensor_coords); coordinates are ignored on purpose
        if Y.shape[1] != iy_q.numel() or S.shape[:2] != site_flat.shape:
            raise RuntimeError("lattice distance: query/site shapes do not match the declared indices")
        dix = (ix_q[None, :, None] - ix_s[:, None, :]).to(torch.float32)
        diy = (iy_q[None, :, None] - iy_s[:, None, :]).to(torch.float32)
        d2 = (qstep[0] ** 2) * dix * dix + (qstep[1] ** 2) * diy * diy
        return torch.sqrt(d2).to(Y.dtype)

    return fn


@contextmanager
def _patched_cdist(fn):
    orig = G.torch.cdist
    G.torch.cdist = fn
    try:
        yield
    finally:
        G.torch.cdist = orig


class AssistedEncoder(G.ThesisDomainAdaptiveEncoder):
    def init_t(self, all_dim: int) -> None:
        self.t_val_linear = nn.Linear(1, all_dim)
        self.t_val_norm = copy.deepcopy(self.embed["val_norm"])
        self.t_type = nn.Parameter(torch.zeros(all_dim))
        nn.init.normal_(self.t_type, std=0.02)

    def forward(self, coords_tuv, U, original_phi=None, p=None, t_xy=None, t_vals=None):  # noqa: D401
        del U
        B, T, N_inp, _ = coords_tuv.shape
        xy = coords_tuv[..., 0:2]
        vals = coords_tuv[..., 2: 2 + self.embed["val_linear"].in_features]
        pos_tok = self.embed["pos_norm"](self.embed["pos_linear"](self.embed["pos_embed"](xy)))
        val_tok = self.embed["val_norm"](self.embed["val_linear"](vals))
        p_tok = 0.0 if p is None else self.p_proj(p).unsqueeze(1).unsqueeze(1)
        tok = pos_tok + val_tok + p_tok
        coords_all = xy
        if t_xy is not None and t_xy.shape[1] > 0:
            t_pos = self.embed["pos_norm"](self.embed["pos_linear"](self.embed["pos_embed"](t_xy.unsqueeze(1))))
            t_val = self.t_val_norm(self.t_val_linear(t_vals.unsqueeze(1)))
            t_tok = t_pos + t_val + self.t_type + p_tok
            tok = torch.cat([tok, t_tok], dim=2)
            coords_all = torch.cat([xy, t_xy.unsqueeze(1)], dim=2)
            N_inp = N_inp + t_xy.shape[1]
        tok_flat = tok.reshape(B * T, N_inp, -1)
        lat = self.latents["latent_param"].expand(B * T, -1, -1)
        if self.retain_cls:
            lat = torch.cat([self.latents["cls_token"].expand(B * T, 1, -1), lat], dim=1)
        lat = self.cross_norm(lat)
        lat = lat + self.cross_attn(lat, tok_flat, tok_flat)
        lat = lat + self.cross_latent_mixer(lat)
        final_latents = lat
        tok_up = self.token_to_latent_norm(tok_flat)
        tok_up = tok_up + self.token_to_latent(tok_up, lat, lat)
        tok_out = tok_up + self.token_to_latent_mixer(tok_up)
        if self.retain_lat:
            tok_out = torch.cat([final_latents, tok_out], dim=1)
        elif self.retain_cls:
            tok_out = torch.cat([lat[:, :1, :], tok_out], dim=1)
        S = tok_out.size(1)
        lat_vec = tok_out.view(B, T, S, -1)
        mask = torch.ones(B, T, S, dtype=torch.bool, device=lat_vec.device)
        coords = coords_all[:, 0, :, :2]
        merged_phi = original_phi if original_phi is not None else torch.ones(B, S, device=lat_vec.device)
        return lat_vec, mask, coords, merged_phi


class AssistedGLU(G.AdaptedDeterministicGLU):
    """Free GLU object converted in place: see :func:`to_assisted`."""

    def init_lattice(self, x_axis: np.ndarray, y_axis: np.ndarray) -> None:
        dx = float(np.mean(np.diff(x_axis)))
        dy = float(np.mean(np.diff(y_axis)))
        if np.ptp(np.diff(x_axis)) > 1e-9 * abs(dx) or np.ptp(np.diff(y_axis)) > 1e-9 * abs(dy):
            raise ValueError("grid must be regular for the lattice distance")
        L = float(self.domain_L)
        self.lattice_qstep = (2.0 * abs(dx) / L, 2.0 * abs(dy) / L)

    def forward(self, sensor_coords_phys, sensor_values_norm=None, site_index=None, p_norm=None, query_coords_phys=None, *,
                t_coords_phys=None, t_values_norm=None, t_site_index=None, p_conditioning: bool = True,
                decode_chunked: bool | None = None, chunk_size: int | None = None, return_flat: bool = False,
                padding_mask=None):
        if padding_mask is not None:
            raise ValueError("padding_mask is not supported")
        bsz, n = sensor_coords_phys.shape[:2]
        if site_index.shape != (bsz, n) or p_norm.shape != (bsz, 2):
            raise ValueError("shape mismatch")
        if query_coords_phys.shape[1] != self.height * self.width:
            raise ValueError("assisted GLU decodes the full grid only (lattice distances)")
        site_index, sensor_coords_phys, sensor_values_norm = G._sort_sensors_by_site(site_index, sensor_coords_phys, sensor_values_norm)
        if not p_conditioning:
            p_norm = torch.zeros_like(p_norm)
        sensor_q = G.phys_to_isotropic_q(sensor_coords_phys, center=self.domain_center, L=self.domain_L)
        query_q = G.phys_to_isotropic_q(query_coords_phys, center=self.domain_center, L=self.domain_L)
        coords_tuv = torch.cat([sensor_q, sensor_values_norm[..., 0:1], sensor_values_norm[..., 1:2]], dim=-1).unsqueeze(1)
        t_q = t_vals = None
        all_site = site_index
        if t_coords_phys is not None and t_coords_phys.shape[1] > 0:
            t_site_index, t_coords_phys, t_values_norm = G._sort_sensors_by_site(t_site_index, t_coords_phys, t_values_norm)
            t_q = G.phys_to_isotropic_q(t_coords_phys, center=self.domain_center, L=self.domain_L)
            t_vals = t_values_norm
            all_site = torch.cat([site_index, t_site_index], dim=1)
        u_dummy = torch.zeros(bsz, device=sensor_q.device, dtype=sensor_q.dtype)
        z, mask, enc_q, _ = self.encoder(coords_tuv, u_dummy, original_phi=None, p=p_norm, t_xy=t_q, t_vals=t_vals)
        n_all = enc_q.shape[1]
        phi_mean = torch.ones(bsz, n_all, device=sensor_q.device, dtype=sensor_q.dtype)
        use_chunks = self.decode_chunked_eval if decode_chunked is None else bool(decode_chunked)
        cs = int(chunk_size) if chunk_size is not None else self.decode_chunk_size
        P = query_q.size(1)
        flat_all = torch.arange(P, device=query_q.device)
        parts = []
        step = cs if (use_chunks and P > cs) else P
        for start in range(0, P, step):
            sl = slice(start, min(P, start + step))
            fn = _lattice_cdist_factory(flat_all[sl], all_site.to(torch.int64), self.lattice_qstep, self.width)
            with _patched_cdist(fn):
                _, out_mean, _ = self.decoder(z, query_q[:, sl, :], enc_q, mask, phi_mean=phi_mean, site_index=all_site)
            parts.append(out_mean)
        out_t = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        flat = out_t.squeeze(1).squeeze(-1)
        return flat if return_flat else flat.view(bsz, 1, self.height, self.width)


def to_assisted(model: G.AdaptedDeterministicGLU, x_axis: np.ndarray, y_axis: np.ndarray) -> AssistedGLU:
    """Convert a freshly built free GLU into the assisted model, keeping every shared weight bit-identical."""
    all_dim = model.encoder.p_proj.out_features
    model.encoder.__class__ = AssistedEncoder
    model.encoder.init_t(all_dim)
    model.__class__ = AssistedGLU
    model.init_lattice(x_axis, y_axis)
    return model  # type: ignore[return-value]


def build_assisted_glu_from_config(*, domain_geometry_path) -> AssistedGLU:
    m = G.build_glu_from_config(domain_geometry_path=domain_geometry_path)
    g = np.load(domain_geometry_path)
    return to_assisted(m, np.asarray(g["x"], dtype=np.float64), np.asarray(g["y"], dtype=np.float64))
