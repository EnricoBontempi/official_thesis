"""Shared helpers for the PIV mission, Phase 1 (F1.1-F1.3) diagnostics.

Diagnostic code only: this is NOT the production observation generator (F2.2).

Data policy enforced here
-------------------------
* Only the dense caches ``train_fit`` / ``seen_train_eval`` / ``val_eval`` are read.
* The raw Arrow shards are read only for TRAIN sim_ids (F1.1 auxiliary check).
* Any ``test*`` role or any TEST sim_id is refused.
* Every file opened is recorded in ``READ_LOG`` and dumped with the outputs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/home/enric/projects/official_thesis")  # holds gitignored payloads
CACHE_C149 = "data/processed/month2_dense_unet_cache_c149"
CACHE_A38 = "data/processed/month2_dense_unet_cache"
STATS_REL = "data/manifests/train_norm_stats_month2_fit.json"
STATS_SHA256_EXPECTED = "678263fb4cb2598350b8fd69a71ce4843f747aa61c19556cf7f0f9c5c3ceac0e"
ALLOWED_ROLES = ("train_fit", "seen_train_eval", "val_eval")

READ_LOG: list[dict] = []


def sha256_file(path: Path, chunk: int = 1 << 23) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def load_splits() -> dict:
    with open(REPO / "configs/splits/splits_interp_extrap.yaml") as f:
        return yaml.safe_load(f)


def test_sim_ids() -> set[str]:
    s = load_splits()
    return set(s["test_interp"]["sim_ids"]) | set(s["test_extrap"]["sim_ids"])


def refuse_test(sim_id: str | None = None, role: str | None = None) -> None:
    if role is not None and (role.startswith("test") or role not in ALLOWED_ROLES):
        raise PermissionError(f"role {role!r} refused (TEST never read; allowed: {ALLOWED_ROLES})")
    if sim_id is not None and sim_id in test_sim_ids():
        raise PermissionError(f"sim_id {sim_id!r} is a TEST condition: refused")


def load_stats(data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    p = data_root / STATS_REL
    digest = sha256_file(p)
    if digest != STATS_SHA256_EXPECTED:
        raise RuntimeError(f"normalization stats sha256 mismatch: {digest}")
    d = json.load(open(p))["fields"]
    READ_LOG.append({"file": str(p), "sha256": digest})
    return {
        "vi": (d["Velocity_i"]["mean"], d["Velocity_i"]["std"]),
        "vj": (d["Velocity_j"]["mean"], d["Velocity_j"]["std"]),
        "sha256": digest,
    }


def load_geometry(data_root: Path = DEFAULT_DATA_ROOT, cache: str = CACHE_C149) -> dict:
    p = data_root / cache / "geometry.npz"
    g = np.load(p)
    READ_LOG.append({"file": str(p), "sha256": sha256_file(p)})
    x, y = g["x"].astype(np.float64) * 1e3, g["y"].astype(np.float64) * 1e3  # m -> mm
    return {"x_mm": x, "y_mm": y, "dx": float(np.mean(np.diff(x))), "dy": float(np.mean(np.diff(y)))}


def load_role(role: str, data_root: Path = DEFAULT_DATA_ROOT, cache: str = CACHE_C149,
              want_T: bool = False, mmap: bool = True) -> dict:
    """Velocity in m/s (float32) reconstructed with the frozen fit-only statistics."""
    refuse_test(role=role)
    st = load_stats(data_root)
    root = data_root / cache / role
    frames = json.load(open(root / "frames.json"))
    for r in frames:
        refuse_test(sim_id=r["sim_id"])
        if r["split"] == "test" or str(r["role"]).startswith("test"):
            raise PermissionError("TEST frame in cache listing")
    out = {"frames": frames}
    for k in ("vi", "vj"):
        p = root / f"{k}_norm.npy"
        READ_LOG.append({"file": str(p), "sha256": sha256_file(p)})
        a = np.load(p, mmap_mode="r" if mmap else None)
        mu, sd = st[k]
        out[k] = (np.asarray(a, dtype=np.float64) * sd + mu).astype(np.float32)
    if want_T:
        p = root / "target_T_K.npy"
        READ_LOG.append({"file": str(p), "sha256": sha256_file(p)})
        out["T"] = np.load(p)
    out["sim_ids"] = np.array([r["sim_id"] for r in frames])
    out["frame_idx"] = np.array([r["frame_index"] for r in frames])
    out["split"] = frames[0]["split"]
    return out


def git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--", "scripts/piv_phase1", "src", "configs"], text=True)
    return bool(out.strip())


def dump_provenance(path: Path, extra: dict | None = None) -> None:
    rec = {
        "code_sha": git_sha(),
        "code_dirty_in_scripts_src_configs": git_dirty(),
        "stats_sha256": STATS_SHA256_EXPECTED,
        "read_log": READ_LOG,
        "numpy": np.__version__,
    }
    import scipy, sys
    rec["scipy"] = scipy.__version__
    rec["python"] = sys.version.split()[0]
    if extra:
        rec.update(extra)
    json.dump(rec, open(path, "w"), indent=1)
