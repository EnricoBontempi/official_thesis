"""Structural (not scientific) check of what the existing GLU wrapper accepts.

Random-initialised tiny synthetic model, no checkpoint, no real field: this only documents the
contract the degradation spec relies on (variable N across frames needs homogeneous batches;
N < top_k is technically supported; N = 0 is not).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from combustion_recon.models.glu_reconstructor import build_glu_from_config
from combustion_recon.training.trainer import set_seed


def _model():
    x = np.linspace(-2, 2, 8)
    set_seed(0)
    m = build_glu_from_config(domain_x=x, domain_y=x, height=8, width=8)
    return m.eval()


def _call(m, n, b=1):
    coords = torch.randn(b, n, 2) * 0.5
    vals = torch.randn(b, n, 2)
    sites = torch.arange(n).unsqueeze(0).expand(b, -1)
    p = torch.zeros(b, 2)
    Y = torch.randn(b, 64, 2) * 0.5
    with torch.no_grad():
        return m(coords, vals, sites, p, Y, decode_chunked=False, return_flat=True)


@pytest.mark.parametrize("n", [1, 5, 16, 30])
def test_small_and_different_n_supported_with_batch_one(n):
    out = _call(_model(), n)
    assert out.shape == (1, 64) and torch.isfinite(out).all()


def test_zero_sites_not_supported():
    with pytest.raises(Exception):
        _call(_model(), 0)


def test_padding_mask_and_ragged_batches_are_refused():
    m = _model()
    with pytest.raises(ValueError, match="padding_mask"):
        m(torch.randn(1, 4, 2), torch.randn(1, 4, 2), torch.arange(4)[None], torch.zeros(1, 2),
          torch.randn(1, 8, 2), padding_mask=torch.ones(1, 4, dtype=torch.bool))
