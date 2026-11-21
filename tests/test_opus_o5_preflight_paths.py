"""The launcher preflight lists only what each O5 task consumes, from the matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "experiments" / "opus_o5_matrix.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pre():
    return _load(ROOT / "scripts" / "engaging" / "o5_preflight_paths.py", "o5_preflight_under_test")


@pytest.fixture(scope="module")
def matrix():
    return json.loads(MATRIX.read_text())


def _c149_root(matrix):
    return matrix["shared_inputs"]["c149_cache_root"]


def _hist_root(matrix):
    return matrix["shared_inputs"]["historical_cache_root"]


@pytest.mark.parametrize("run_id", ["O-B19-on-s0", "O-B19-off-s0", "O-A38-on-s0", "O-A38-off-s0", "O-A38-on-s1", "O-A38-off-s1"])
def test_historical_runs_never_touch_the_c149_cache(pre, matrix, run_id):
    c149_hits: list[str] = []

    def stat(path: Path):
        if str(path).startswith(_c149_root(matrix)) or str(path) == matrix["shared_inputs"]["c149_hash_file"]:
            c149_hits.append(str(path))
            raise OSError("C149 unreachable")
        return None

    paths = pre.required_paths(matrix, run_id)
    assert pre.check_access(paths, stat) == []
    assert c149_hits == []
    assert any(str(p).startswith(_hist_root(matrix)) for p in paths)


@pytest.mark.parametrize("run_id", ["O-C149-on-s0", "O-C149-off-s0"])
def test_c149_runs_use_the_c149_cache_and_not_the_historical_one(pre, matrix, run_id):
    paths = [str(p) for p in pre.required_paths(matrix, run_id)]
    assert any(p.startswith(_c149_root(matrix)) for p in paths)
    assert not any(p.startswith(_hist_root(matrix)) for p in paths)
    assert matrix["shared_inputs"]["historical_hash_file"] not in paths


def test_an_unreachable_required_path_is_reported(pre, matrix):
    paths = pre.required_paths(matrix, "O-B19-off-s0")
    bad = paths[0]

    def stat(path: Path):
        if path == bad:
            raise OSError("down")

    assert pre.check_access(paths, stat) == [bad]


def test_preflight_matches_what_the_evaluator_reads(pre, matrix, tmp_path, monkeypatch):
    """Run the evaluator's input gate against recorded fakes for both pools."""
    o5 = _load(ROOT / "scripts" / "26_opus_o5_eval.py", "o5_eval_for_preflight")
    shared_template = matrix["shared_inputs"]
    for run_id in ("O-B19-off-s0", "O-C149-on-s0"):
        entry = next(r for r in matrix["runs"] if r["run_id"] == run_id)
        base = tmp_path / run_id
        shared = dict(shared_template)
        names = {
            "historical_cache_root": "hist",
            "c149_cache_root": "c149",
            "mask_root": "masks",
        }
        for key, name in names.items():
            (base / name).mkdir(parents=True)
            shared[key] = str(base / name)
        for key, name in (("historical_hash_file", "hist.json"), ("c149_hash_file", "c149.json"), ("stats", "stats.json")):
            (base / name).write_text("{}")
            shared[key] = str(base / name)
        for root in ("hist", "c149"):
            (base / root / "manifest.json").write_text(
                json.dumps({"cache_id": shared["c149_cache_id" if root == "c149" else "historical_cache_id"]})
            )
        (base / "masks" / "manifest.json").write_text("{}")
        train = base / "train"
        train.mkdir()
        (train / "run_identity.json").write_text(json.dumps({"git_sha": entry["training_git_sha"], "pool_id": entry["pool_id"]}))
        local = {**matrix, "shared_inputs": shared, "runs": [{**entry, "training_dir": str(train)}]}

        read: list[str] = []
        real_read = Path.read_text

        def recording_read(self, *a, **k):
            read.append(str(self))
            return real_read(self, *a, **k)

        def verify(expected, cache_root):
            read.append(str(cache_root) + "/<payload>")
            agg = shared["c149_cache_aggregate_sha256" if entry["pool_id"] == "C149" else "historical_cache_aggregate_sha256"]
            return {"ok": True, "actual_aggregate_sha256": agg, "mismatches": []}

        def sha(path):
            read.append(str(path))
            return shared["stats_sha256"]

        def contract(**kw):
            read.append(str(kw["manifest_path"]))
            return {
                "eval_mask_manifest_file_sha256": kw["expected_manifest_file_sha256"],
                "eval_mask_aggregate_payload_sha256": kw["expected_aggregate_payload_sha256"],
            }

        monkeypatch.setattr(Path, "read_text", recording_read)
        monkeypatch.setattr(o5, "verify_cache_payload_hashes", verify)
        monkeypatch.setattr(o5, "sha256_file", sha)
        monkeypatch.setattr(o5, "check_eval_mask_contract", contract)
        o5._verify_inputs({**entry, "training_dir": str(train)}, local)
        monkeypatch.setattr(Path, "read_text", real_read)

        listed = {str(p) for p in pre.required_paths(local, run_id)}
        used_cache = "c149" if entry["pool_id"] == "C149" else "hist"
        other = "hist" if used_cache == "c149" else "c149"
        assert any(r.startswith(str(base / used_cache)) for r in read)
        assert not any(r.startswith(str(base / other)) for r in read)
        assert str(base / used_cache / "manifest.json") in listed
        assert not any(p.startswith(str(base / other)) for p in listed)
        assert str(base / "masks" / "manifest.json") in listed and str(base / "masks" / "manifest.json") in read
        assert shared["stats"] in listed and shared["stats"] in read
        assert str(train / "run_identity.json") in listed and str(train / "run_identity.json") in read
