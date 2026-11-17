"""Dense U-Net: full-grid planar velocity (+ coords + p) → normalized temperature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNET_CFG = ROOT / "configs" / "models" / "unet_small.yaml"

CHANNEL_ORDER = (
    "vi_norm",
    "vj_norm",
    "x_scaled",
    "y_scaled",
    "nh3_norm",
    "phi_norm",
)


class ConvGNSiLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8, bias: bool = True):
        super().__init__()
        g = min(int(groups), int(out_ch))
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNSiLU(in_ch, out_ch, groups=groups, bias=bias),
            ConvGNSiLU(out_ch, out_ch, groups=groups, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8, bias: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, *, groups: int = 8, bias: bool = True):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DenseUNet(nn.Module):
    """Encoder–decoder U-Net with channel-concatenated operating parameters.

    Forward consumes only the 6-channel tensor (B,6,128,128) and predicts
    normalized temperature (B,1,128,128). No FiLM / ParamMean prior.
    """

    def __init__(
        self,
        *,
        in_channels: int = 6,
        base_channels: int = 32,
        out_channels: int = 1,
        groups: int = 8,
        bias: bool = True,
        head_weight_std: float = 1e-3,
        head_bias: float = 0.0,
        init_seed: int | None = None,
    ):
        super().__init__()
        if in_channels != 6:
            raise ValueError("DenseUNet expects 6 input channels for this mission")
        b = int(base_channels)
        self.in_channels = in_channels
        self.base_channels = b
        self.channel_order = list(CHANNEL_ORDER)
        self.head_weight_std = float(head_weight_std)

        self.enc1 = DoubleConv(in_channels, b, groups=groups, bias=bias)  # 128: b
        self.down1 = Down(b, 2 * b, groups=groups, bias=bias)  # 64: 2b
        self.down2 = Down(2 * b, 4 * b, groups=groups, bias=bias)  # 32: 4b
        self.down3 = Down(4 * b, 8 * b, groups=groups, bias=bias)  # 16: 8b
        self.down4 = Down(8 * b, 8 * b, groups=groups, bias=bias)  # 8: 8b bottleneck

        self.up1 = Up(8 * b, 8 * b, 8 * b, groups=groups, bias=bias)  # →16: 8b
        self.up2 = Up(8 * b, 4 * b, 4 * b, groups=groups, bias=bias)  # →32: 4b
        self.up3 = Up(4 * b, 2 * b, 2 * b, groups=groups, bias=bias)  # →64: 2b
        self.up4 = Up(2 * b, b, b, groups=groups, bias=bias)  # →128: b

        self.head = nn.Conv2d(b, out_channels, kernel_size=1, bias=True)
        self._init_weights(init_seed=init_seed, head_bias=head_bias)

    def _init_weights(self, *, init_seed: int | None, head_bias: float) -> None:
        if init_seed is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(init_seed))
        else:
            g = None

        for m in self.modules():
            if m is self.head:
                continue
            if isinstance(m, nn.Conv2d):
                # Kaiming normal, fan_out, nonlinearity approximating SiLU via ReLU convention.
                if g is None:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                else:
                    # Manual kaiming with generator for reproducibility across devices.
                    fan_out = m.weight.size(0) * m.weight[0].numel()
                    std = (2.0 / fan_out) ** 0.5
                    with torch.no_grad():
                        m.weight.normal_(0.0, std, generator=g)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small non-zero head weights (not all-zero) so early blocks receive gradients.
        with torch.no_grad():
            if g is None:
                self.head.weight.normal_(0.0, self.head_weight_std)
            else:
                self.head.weight.normal_(0.0, self.head_weight_std, generator=g)
            if self.head.bias is not None:
                self.head.bias.fill_(float(head_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B,{self.in_channels},H,W), got {tuple(x.shape)}")
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        b = self.down4(e4)
        d1 = self.up1(b, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        return self.head(d4)

    def count_parameters(self, *, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def load_unet_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_UNET_CFG
    with p.open() as f:
        cfg = yaml.safe_load(f)
    required = [
        "architecture",
        "input",
        "output",
        "encoder",
        "block",
        "decoder",
        "init",
    ]
    for k in required:
        if k not in cfg:
            raise ValueError(f"unet config missing required key {k!r}")
    # Refuse unknown top-level experimental knobs silently ignored.
    allowed = {
        "model_id",
        "architecture",
        "task",
        "input",
        "output",
        "encoder",
        "block",
        "decoder",
        "init",
        "regularization_initial",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"unet config has unsupported keys: {sorted(unknown)}")
    if cfg["architecture"] != "dense_unet_concat_p":
        raise ValueError(f"unsupported architecture {cfg['architecture']!r}")
    if cfg["input"].get("conditioning") != "channel_concatenation_only":
        raise ValueError("only channel_concatenation_only conditioning is supported")
    if cfg["block"].get("activation") != "silu":
        raise ValueError("only silu activation is supported")
    if cfg["decoder"].get("upsample") != "bilinear":
        raise ValueError("only bilinear upsampling is supported")
    return cfg


def build_dense_unet_from_config(
    path: str | Path | None = None,
    *,
    base_channels: int | None = None,
    init_seed: int | None = 0,
) -> DenseUNet:
    cfg = load_unet_config(path)
    b = int(base_channels) if base_channels is not None else int(cfg["encoder"]["base_channels"])
    return DenseUNet(
        in_channels=int(cfg["input"]["channels"]),
        base_channels=b,
        out_channels=int(cfg["output"]["channels"]),
        groups=int(cfg["block"]["groups"]),
        bias=bool(cfg["block"].get("bias", True)),
        head_weight_std=float(cfg["init"]["head_weight_std"]),
        head_bias=float(cfg["init"]["head_bias"]),
        init_seed=init_seed,
    )
