"""Gate access to Month-2 TEST roles behind a selection-freeze artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FREEZE_PATH = ROOT / "configs" / "experiments" / "month2_selection_freeze.json"


class FreezeGateError(PermissionError):
    """Raised when TEST payload access is requested without a valid freeze file."""


# Back-compat alias (not named Test* to avoid pytest collection).
TestAccessDenied = FreezeGateError


def load_freeze(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path is not None else DEFAULT_FREEZE_PATH
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def assert_test_access_allowed(
    *,
    freeze_path: str | Path | None = None,
    require_keys: tuple[str, ...] = ("freeze_id", "commit", "selection_metric"),
) -> dict[str, Any]:
    freeze = load_freeze(freeze_path)
    if freeze is None:
        raise FreezeGateError(
            "TEST payload access is disabled by default. "
            "Provide a valid month2_selection_freeze.json (created after selection lock)."
        )
    missing = [k for k in require_keys if k not in freeze]
    if missing:
        raise FreezeGateError(f"Freeze artifact missing required keys: {missing}")
    if freeze.get("allow_test_payload_access") is not True:
        raise FreezeGateError(
            "Freeze artifact present but allow_test_payload_access is not true."
        )
    return freeze
