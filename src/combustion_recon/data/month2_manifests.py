"""Build compact Month-2 frame and observation manifests (recipes + hashes)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from combustion_recon.data.masks import (
    nested_temperature_anchor_permutation,
    random_sparse_mask,
    stable_seed,
)
from combustion_recon.data.month2_observations import mask_hash, observation_mask_seed
from combustion_recon.data.month2_protocol import (
    TEST_ROLES,
    FrameRole,
    build_index,
    frames_for_role,
    load_frame_protocol,
    load_yaml,
    sha256_file,
    split_name_for_role,
)

ROOT = Path(__file__).resolve().parents[3]


def build_frames_manifest(
    *,
    protocol_path: Path | None = None,
    pilot_path: Path | None = None,
) -> dict[str, Any]:
    protocol_path = protocol_path or (ROOT / "configs/splits/month2_frame_protocol.yaml")
    pilot_path = pilot_path or (ROOT / "configs/experiments/month2_pilot.yaml")
    protocol = load_frame_protocol(protocol_path)
    pilot = load_yaml(pilot_path)

    roles: list[FrameRole] = [
        "train_fit",
        "seen_train_eval",
        "val_eval",
        "test_interp_eval",
        "test_extrap_eval",
    ]
    entries = []
    counts = {}
    for role in roles:
        idx = build_index(role, protocol=protocol)
        counts[role] = {
            "n_samples": len(idx),
            "n_conditions": len({e["sim_id"] for e in idx}),
            "n_frames_per_condition": len(frames_for_role(protocol, role)),
            "split": split_name_for_role(role),
            "payload_access": (
                "requires_selection_freeze" if role in TEST_ROLES else "allowed"
            ),
        }
        # Compact recipe: store unique sim_ids + shared frame list, not duplicated fields
        entries.append(
            {
                "role": role,
                "split": split_name_for_role(role),
                "sim_ids": sorted({e["sim_id"] for e in idx}),
                "frame_indices": frames_for_role(protocol, role),
                "n_samples": len(idx),
            }
        )

    fit = set(frames_for_role(protocol, "train_fit"))
    seen = set(frames_for_role(protocol, "seen_train_eval"))
    return {
        "manifest_id": "month2_frames_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_path),
        "pilot_id": pilot.get("experiment_id"),
        "pilot_path": str(pilot_path.relative_to(ROOT)),
        "pilot_sha256": sha256_file(pilot_path),
        "fit_seen_disjoint": not bool(fit & seen),
        "gap_fit_to_seen": min(seen) - max(fit) if fit and seen else None,
        "counts": counts,
        "roles": entries,
        "notes": [
            "TEST roles list indices for future frozen evaluation only.",
            "This manifest does not materialize field payloads.",
            "seen_train_eval is a later temporal block of the same TRAIN trajectories.",
            "Temporal quality: Month-2 primary protocol is single-frame-by-index.",
            "Known packaging anomaly 40NH3_1.2.h5 (data-00022): interior t gap; "
            "numerical frame 2000 duplicates 1999; field↔time association is NOT "
            "verified for any index on that shard (see docs/month2/mission_baseline_timestamp_audit.md). "
            "Do not use that shard for multi-frame/forecasting without a new protocol decision.",
        ],
        "temporal_quality_policy": {
            "primary_protocol": "single_frame_by_index",
            "anomalous_sim_ids": ["40NH3_1.2.h5"],
            "flags_documented_in": "combustion_recon.data.temporal_quality",
            "note": (
                "Existing month2_frames.json on disk is historical; regenerating this "
                "builder updates notes for future manifests only."
            ),
        },
    }


def build_observations_manifest(
    *,
    protocol_path: Path | None = None,
    pilot_path: Path | None = None,
    verify_sample_sim_id: str = "0NH3_1.h5",
    verify_frame_index: int = 0,
) -> dict[str, Any]:
    """Compact deterministic observation recipes (no huge coordinate dumps)."""
    protocol_path = protocol_path or (ROOT / "configs/splits/month2_frame_protocol.yaml")
    pilot_path = pilot_path or (ROOT / "configs/experiments/month2_pilot.yaml")
    protocol = load_frame_protocol(protocol_path)
    pilot = load_yaml(pilot_path)

    mask_seed = 20261116
    replica_ids = list(pilot.get("seeds", {}).get("mask_replica_ids_proposed", [0, 1, 2]))
    fractions = list(pilot["sparsity"]["coverage_controls"]["fractions"])
    nested_K = list(pilot["temperature_anchors"]["nested_sweep_K"])
    layout_id = "random_sparse_primary"

    # Verify one sample's masks / nested inclusion and record hashes
    verified = []
    for frac in fractions:
        for replica in replica_ids:
            vel_seed = observation_mask_seed(
                mask_seed=mask_seed,
                sim_id=verify_sample_sim_id,
                frame_index=verify_frame_index,
                layout_id=layout_id,
                replica_id=int(replica),
            )
            vel_mask = random_sparse_mask(float(frac), seed=vel_seed, height=128, width=128)
            anchor_seed = stable_seed(vel_seed, "temp_anchors")
            perm = nested_temperature_anchor_permutation(
                seed=anchor_seed, height=128, width=128, max_k=32
            )
            nested_ok = True
            prev = set()
            k_hashes = {}
            for K in nested_K:
                cur = set(int(i) for i in perm[: int(K)])
                if not prev.issubset(cur):
                    nested_ok = False
                t_mask = np.zeros((128, 128), dtype=bool)
                if K > 0:
                    t_mask.ravel()[list(cur)] = True
                k_hashes[str(K)] = mask_hash(t_mask)
                prev = cur
            verified.append(
                {
                    "sim_id": verify_sample_sim_id,
                    "frame_index": verify_frame_index,
                    "layout_id": layout_id,
                    "site_fraction": float(frac),
                    "replica_id": int(replica),
                    "mask_seed": int(mask_seed),
                    "velocity_mask_seed": int(vel_seed),
                    "anchor_perm_seed": int(anchor_seed),
                    "n_velocity_sites": int(vel_mask.sum()),
                    "velocity_mask_hash": mask_hash(vel_mask),
                    "temperature_mask_hash_by_K": k_hashes,
                    "nested_K_ok": nested_ok,
                    "same_velocity_mask_for_all_K": True,
                }
            )

    return {
        "manifest_id": "month2_observations_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol.get("protocol_id"),
        "pilot_id": pilot.get("experiment_id"),
        "mask_seed_default": mask_seed,
        "layout_id_primary": layout_id,
        "site_fraction_primary": float(pilot["sparsity"]["primary"]["site_fraction"]),
        "coverage_fractions": fractions,
        "nested_K": nested_K,
        "mask_replica_ids": replica_ids,
        "seed_tuple_fields": ["mask_seed", "sim_id", "frame_index", "layout_id", "replica_id"],
        "training_mask_seed_fields": [
            "train_mask",
            "train_mask_seed",
            "epoch",
            "sim_id",
            "frame_index",
            "layout_id",
        ],
        "token_scheme": "co_located_velocity_vectors_plus_optional_T_anchors",
        "depends_on_target_T": False,
        "verified_examples": verified,
        "notes": [
            "Masks are regenerated from the numeric seeds above; full grids are not stored.",
            "Evaluation masks are independent of dataset order, worker id, and model choice.",
            "Training masks use a separate epoch-scoped seed path.",
        ],
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
