"""Temporal quality flags for Month-2 frame provenance.

Single-frame reconstruction uses field **indices**, not absolute time.
These flags describe whether a coordinate may be used for multi-frame /
forecasting protocols. They must never be merged into ``model_inputs``.

Flag meanings
-------------
``has_original_time_coordinate``
    An original ``t`` entry exists for this frame index in the shard payload
    (not invented). Does **not** imply the value is the physical time of the
    field at that index.

``timeline_regular``
    The shard's ``t`` array is approximately uniformly spaced (no documented
    interior gap / irregular step). False for the known ``40NH3_1.2`` packaging.

``field_time_association_verified``
    Evidence supports using ``t[i]`` as the physical time of numerical frame
    ``i``. False for the entire anomalous shard (index↔time not demonstrated).

``time_validated`` (compatibility)
    Alias of ``field_time_association_verified``. Kept so older callers that
    only check ``time_validated`` do not treat unassociated times as safe.

``duplicate_of``
    If not ``None``, the numerical field at this index is an exact duplicate of
    the named frame index (e.g. 2000 → 1999 on ``40NH3_1.2``).

``t`` / ``time`` / ``time_coordinate_original``
    May store an original coordinate for audit. Presence alone does **not**
    validate field↔time association.
"""

from __future__ import annotations

from typing import Any, Mapping


TEMPORAL_METADATA_KEYS: tuple[str, ...] = (
    "has_original_time_coordinate",
    "timeline_regular",
    "field_time_association_verified",
    "time_validated",
    "duplicate_of",
    "time_coordinate_original",
)


def temporal_quality_from_mapping(obj: Any) -> dict[str, Any]:
    """Extract temporal quality from a CompactFrame or metadata mapping."""
    if hasattr(obj, "field_time_association_verified"):
        return {
            "has_original_time_coordinate": bool(obj.has_original_time_coordinate),
            "timeline_regular": bool(obj.timeline_regular),
            "field_time_association_verified": bool(obj.field_time_association_verified),
            "time_validated": bool(obj.time_validated),
            "duplicate_of": obj.duplicate_of,
            "time_coordinate_original": float(obj.t),
        }
    if isinstance(obj, Mapping):
        verified = obj.get(
            "field_time_association_verified",
            obj.get("time_validated", True),
        )
        return {
            "has_original_time_coordinate": bool(
                obj.get("has_original_time_coordinate", True)
            ),
            "timeline_regular": bool(obj.get("timeline_regular", True)),
            "field_time_association_verified": bool(verified),
            "time_validated": bool(obj.get("time_validated", verified)),
            "duplicate_of": obj.get("duplicate_of"),
            "time_coordinate_original": obj.get(
                "time_coordinate_original", obj.get("time", obj.get("t"))
            ),
        }
    raise TypeError(f"Unsupported temporal quality source: {type(obj)!r}")


def require_verified_field_time_association(
    obj: Any,
    *,
    context: str = "",
) -> None:
    """Refuse temporal protocols when field↔time association is not verified.

    Single-frame-by-index use does **not** call this guard. Multi-frame windows,
    forecasting, and any code that treats ``t`` as physical time of the field
    must call it first.
    """
    from combustion_recon.data.shard_access import TimelineError

    q = temporal_quality_from_mapping(obj)
    if q["field_time_association_verified"]:
        return
    where = f" ({context})" if context else ""
    raise TimelineError(
        "Refusing temporal use: field↔time association is not verified"
        f"{where}. Single-frame-by-index remains allowed; multi-frame / "
        "forecasting requires a verified association."
    )
