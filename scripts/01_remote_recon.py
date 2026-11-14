#!/usr/bin/env python3
"""Step 1.2 — Remote reconnaissance of RealPDEBench (no large-file downloads).

Queries Hugging Face Hub metadata for AI4Science-WestlakeU/RealPDEBench,
emphasizing combustion-related paths, and writes a structured inventory to
data/manifests/remote_file_inventory.json.

Usage:
    python scripts/01_remote_recon.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

REPO_ID = "AI4Science-WestlakeU/RealPDEBench"
REPO_TYPE = "dataset"

# Paths/names that indicate combustion content (case-insensitive).
COMBUSTION_PATTERN = re.compile(
    r"combust|flame|swirl|ammonia|nh3|chemilumines|oh\*|oh_star",
    re.IGNORECASE,
)

# Top-level metadata / license / readme filenames to always retain.
META_PATTERN = re.compile(
    r"(^|/)(readme|license|citation|dataset_card|gitattributes|gitignore|"
    r"meta|manifest|config|datasheet)(\.|$|_)",
    re.IGNORECASE,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "manifests" / "remote_file_inventory.json"


def human_bytes(n: int | None) -> str | None:
    if n is None:
        return None
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.3f} {unit}"
        size /= 1024.0
    return f"{n} B"


def extension_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.startswith(".") and name.count(".") == 1:
        return name.lower()
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def is_combustion_related(path: str) -> bool:
    return bool(COMBUSTION_PATTERN.search(path))


def is_metadata_file(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return bool(META_PATTERN.search(base)) or path.count("/") == 0 and bool(
        META_PATTERN.search(path)
    )


def serialize_entry(item: RepoFile | RepoFolder) -> dict[str, Any]:
    path = item.path
    entry: dict[str, Any] = {
        "path": path,
        "type": "folder" if isinstance(item, RepoFolder) else "file",
        "extension": extension_of(path) if isinstance(item, RepoFile) else None,
        "size_bytes": getattr(item, "size", None),
        "size_human": human_bytes(getattr(item, "size", None)),
        "blob_id": getattr(item, "blob_id", None),
        "lfs": None,
        "combustion_related": is_combustion_related(path),
        "metadata_related": is_metadata_file(path),
    }
    lfs = getattr(item, "lfs", None)
    if lfs is not None:
        # HfFileMetadata / dict-like LFS pointer
        if isinstance(lfs, dict):
            entry["lfs"] = {
                "sha256": lfs.get("sha256") or lfs.get("oid"),
                "size": lfs.get("size"),
                "pointer_size": lfs.get("pointer_size"),
            }
        else:
            entry["lfs"] = {
                "sha256": getattr(lfs, "sha256", None) or getattr(lfs, "oid", None),
                "size": getattr(lfs, "size", None),
                "pointer_size": getattr(lfs, "pointer_size", None),
            }
            if entry["lfs"]["sha256"] is None and hasattr(lfs, "model_dump"):
                dumped = lfs.model_dump()
                entry["lfs"] = {
                    "sha256": dumped.get("sha256") or dumped.get("oid"),
                    "size": dumped.get("size"),
                    "pointer_size": dumped.get("pointer_size"),
                }
    return entry


def summarize(files: list[dict[str, Any]]) -> dict[str, Any]:
    combustion_files = [f for f in files if f["type"] == "file" and f["combustion_related"]]
    meta_files = [f for f in files if f["type"] == "file" and f["metadata_related"]]
    all_files = [f for f in files if f["type"] == "file"]

    def total_bytes(items: list[dict[str, Any]]) -> int:
        return int(sum(f["size_bytes"] or 0 for f in items))

    by_ext: dict[str, dict[str, Any]] = {}
    for f in combustion_files:
        ext = f["extension"] or "(none)"
        bucket = by_ext.setdefault(ext, {"count": 0, "size_bytes": 0})
        bucket["count"] += 1
        bucket["size_bytes"] += f["size_bytes"] or 0
    for ext, bucket in by_ext.items():
        bucket["size_human"] = human_bytes(bucket["size_bytes"])

    # RealPDEBench layout: combustion/hf_dataset/{numerical,real}/...
    numerical = [
        f
        for f in combustion_files
        if "/hf_dataset/numerical/" in f["path"].replace("\\", "/")
        or f["path"].endswith("_numerical.json")
        or f["path"].endswith("params_numerical.json")
    ]
    experimental_real = [
        f
        for f in combustion_files
        if "/hf_dataset/real/" in f["path"].replace("\\", "/")
        or f["path"].endswith("_real.json")
        or f["path"].endswith("params_real.json")
    ]
    combustion_meta = [
        f
        for f in combustion_files
        if f not in numerical and f not in experimental_real
    ]

    return {
        "total_repo_entries": len(files),
        "total_files": len(all_files),
        "total_folders": sum(1 for f in files if f["type"] == "folder"),
        "total_repo_size_bytes": total_bytes(all_files),
        "total_repo_size_human": human_bytes(total_bytes(all_files)),
        "combustion_file_count": len(combustion_files),
        "combustion_size_bytes": total_bytes(combustion_files),
        "combustion_size_human": human_bytes(total_bytes(combustion_files)),
        "combustion_by_extension": by_ext,
        "numerical_file_count": len(numerical),
        "numerical_size_bytes": total_bytes(numerical),
        "numerical_size_human": human_bytes(total_bytes(numerical)),
        "experimental_real_file_count": len(experimental_real),
        "experimental_real_size_bytes": total_bytes(experimental_real),
        "experimental_real_size_human": human_bytes(total_bytes(experimental_real)),
        "combustion_other_meta_file_count": len(combustion_meta),
        "combustion_other_meta_size_bytes": total_bytes(combustion_meta),
        "combustion_other_meta_size_human": human_bytes(total_bytes(combustion_meta)),
        "metadata_file_count": len(meta_files),
        "combustion_paths": sorted(f["path"] for f in combustion_files),
        "numerical_paths": sorted(f["path"] for f in numerical),
        "experimental_real_paths": sorted(f["path"] for f in experimental_real),
        "metadata_paths": sorted(f["path"] for f in meta_files),
    }


def main() -> None:
    api = HfApi()
    print(f"Listing recursive tree for {REPO_ID} ({REPO_TYPE}) ...")
    tree = list(
        api.list_repo_tree(
            REPO_ID,
            repo_type=REPO_TYPE,
            recursive=True,
        )
    )
    entries = [serialize_entry(item) for item in tree]
    entries.sort(key=lambda e: e["path"])
    summary = summarize(entries)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
        "hub_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "summary": summary,
        "entries": entries,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_PATH}")
    print(f"  Total entries:     {summary['total_repo_entries']}")
    print(f"  Total files:       {summary['total_files']}")
    print(f"  Repo size:         {summary['total_repo_size_human']}")
    print(f"  Combustion files:  {summary['combustion_file_count']}")
    print(f"  Combustion size:   {summary['combustion_size_human']}")
    print(f"  Numerical:         {summary['numerical_size_human']}")
    print(f"  Experimental/real: {summary['experimental_real_size_human']}")
    print(f"  Other combustion:  {summary['combustion_other_meta_size_human']}")
    if summary["combustion_paths"]:
        print("  Combustion paths:")
        for p in summary["combustion_paths"]:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
