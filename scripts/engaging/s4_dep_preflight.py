#!/usr/bin/env python3
"""Light dependency/interpreter preflight for S4 profiling launchers (no CUDA work).

Exits nonzero before GPU compute if required imports for the selected family fail.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("sparse_unet", "adapted_glu"), required=True)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    report: dict = {
        "executable": sys.executable,
        "version": sys.version,
        "family": args.family,
        "modules": {},
        "ok": True,
    }

    common = ["numpy", "yaml", "torch", "pandas"]
    for name in common:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", None)
            report["modules"][name] = {"present": True, "version": ver}
        except Exception as exc:  # noqa: BLE001
            report["modules"][name] = {"present": False, "error": f"{type(exc).__name__}: {exc}"}
            report["ok"] = False

    # Family-specific: U-Net must not require einops; GLU must.
    if args.family == "adapted_glu":
        try:
            import einops

            report["modules"]["einops"] = {"present": True, "version": einops.__version__}
        except Exception as exc:  # noqa: BLE001
            report["modules"]["einops"] = {
                "present": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["ok"] = False
    else:
        # Record einops presence for diagnostics only; not required for U-Net.
        try:
            import einops

            report["modules"]["einops"] = {
                "present": True,
                "version": einops.__version__,
                "required_for_family": False,
            }
        except Exception:
            report["modules"]["einops"] = {"present": False, "required_for_family": False}

    # Import path checks without triggering GPU kernels.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    try:
        if args.family == "sparse_unet":
            from combustion_recon.models.unet import build_sparse_unet_from_config  # noqa: F401
            from combustion_recon.training.sparse_scientific_loop import (  # noqa: F401
                SparseScientificTrainer,
            )
        else:
            from combustion_recon.models.glu_reconstructor import (  # noqa: F401
                build_glu_from_config,
            )
            from combustion_recon.training.sparse_scientific_loop import (  # noqa: F401
                SparseScientificTrainer,
            )
        report["package_imports"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["package_imports"] = f"{type(exc).__name__}: {exc}"
        report["ok"] = False

    text = json.dumps(report, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
