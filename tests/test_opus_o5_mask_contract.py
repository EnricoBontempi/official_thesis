"""Separate the eval-mask manifest file hash from the payload aggregate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from combustion_recon.data.sparse_eval_masks import (
    EvalMaskContractError,
    _aggregate_sha,
    check_eval_mask_contract,
)


def _write_fixture(root: Path, payload_bytes: bytes, aggregate: str | None = None) -> tuple[Path, Path, str, str]:
    payload_root = root / "payload"
    payload_root.mkdir(parents=True)
    (payload_root / "sites.npz").write_bytes(payload_bytes)
    files = {"sites.npz": hashlib.sha256(payload_bytes).hexdigest()}
    real_aggregate = _aggregate_sha(files)
    manifest = {
        "payload_files_sha256": files,
        "aggregate_payload_sha256": real_aggregate if aggregate is None else aggregate,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    file_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, payload_root, file_sha, real_aggregate


def test_correct_pair_is_accepted(tmp_path: Path):
    manifest, payload, file_sha, aggregate = _write_fixture(tmp_path, b"sites-v1")
    report = check_eval_mask_contract(
        manifest_path=manifest,
        payload_root=payload,
        expected_manifest_file_sha256=file_sha,
        expected_aggregate_payload_sha256=aggregate,
    )
    assert report["eval_mask_manifest_file_sha256"] == file_sha
    assert report["eval_mask_aggregate_payload_sha256"] == aggregate
    assert file_sha != aggregate


def test_swapped_hashes_are_rejected(tmp_path: Path):
    manifest, payload, file_sha, aggregate = _write_fixture(tmp_path, b"sites-v1")
    with pytest.raises(EvalMaskContractError) as swapped:
        check_eval_mask_contract(
            manifest_path=manifest,
            payload_root=payload,
            expected_manifest_file_sha256=aggregate,
            expected_aggregate_payload_sha256=file_sha,
        )
    assert swapped.value.reason == "manifest_file_sha256_mismatch"
    with pytest.raises(EvalMaskContractError) as aggregate_only:
        check_eval_mask_contract(
            manifest_path=manifest,
            payload_root=payload,
            expected_manifest_file_sha256=file_sha,
            expected_aggregate_payload_sha256=file_sha,
        )
    assert aggregate_only.value.reason == "aggregate_payload_sha256_mismatch"


def test_altered_manifest_or_payload_is_rejected(tmp_path: Path):
    manifest, payload, file_sha, aggregate = _write_fixture(tmp_path / "ok", b"sites-v1")
    altered = tmp_path / "manifest-altered"
    altered.mkdir()
    manifest_copy = altered / "manifest.json"
    manifest_copy.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(EvalMaskContractError) as manifest_error:
        check_eval_mask_contract(
            manifest_path=manifest_copy,
            payload_root=payload,
            expected_manifest_file_sha256=file_sha,
            expected_aggregate_payload_sha256=aggregate,
        )
    assert manifest_error.value.reason == "manifest_file_sha256_mismatch"

    payload_altered = tmp_path / "payload-altered"
    _, payload_bad, _, _ = _write_fixture(payload_altered, b"sites-v1")
    (payload_bad / "sites.npz").write_bytes(b"sites-changed")
    with pytest.raises(EvalMaskContractError) as payload_error:
        check_eval_mask_contract(
            manifest_path=manifest,
            payload_root=payload_bad,
            expected_manifest_file_sha256=file_sha,
            expected_aggregate_payload_sha256=aggregate,
        )
    assert payload_error.value.reason == "payload_file_mismatch"
