"""Cheap label/metric access AFTER complete dataset intake, with no model work.

The freeze/init caller MUST first validate each sample with the complete
versioned ``narrative_qualification_v2.require_narrative_qualified`` (which
dispatches old Narrative samples to their original validator), or its original
legacy intake. This helper checks structure and consistency only. Neither an
``accepted`` boolean nor structurally plausible response/call dictionaries prove
that pixels, provider receipts or SQL calls were verified. Do not replace intake
with these functions or run expensive receipt verification inside a train step.

Retrieval uses past samples and their completed-history candidates. Version2's
raw ``positive_clip_ids`` stays an audit relevance union; only adopted, fully
verified support groups supply retrieval positives. Missing or unadopted support
never becomes a negative. All returned lists are copies; samples stay unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re


V2_QUALIFICATION_PROTOCOL = "narrative-answer-unique-v4-qualification-v2"
_CHECKS = {"reference_only", "evidence_only", "training_context", "no_concept"}


def _is_v2(sample: Mapping) -> bool:
    gate = sample.get("narrative_qualification", {})
    if isinstance(gate, Mapping) and gate.get("protocol") == V2_QUALIFICATION_PROTOCOL:
        return True
    if "verified_retrieval_positive_clip_ids" in sample:
        raise ValueError("A verified retrieval field requires its known version2 qualification protocol")
    return False


def _ids(value, name: str, *, nonempty: bool = True) -> list[str]:
    if (not isinstance(value, list) or nonempty and not value
            or any(not isinstance(cid, str) or not cid for cid in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}list of distinct clip IDs")
    return list(value)


def _candidate_ids(sample: Mapping) -> set[str]:
    clips = sample.get("candidates")
    if not isinstance(clips, list) or any(not isinstance(clip, Mapping) or "id" not in clip for clip in clips):
        raise ValueError("Historical retrieval candidates must contain explicit clip IDs")
    return set(_ids([clip["id"] for clip in clips], "Historical candidate IDs"))


def _bounded_support(ids: list[str], candidates: set[str], negatives: set[str]) -> None:
    if len(ids) > 3 or not set(ids) <= candidates:
        raise ValueError("A sufficient retrieval support group must use at most three historical candidates")
    if set(ids) & negatives:
        raise ValueError("Verified support and explicit retrieval negatives overlap")


def _v2_support(sample: Mapping) -> tuple[list[str], list[list[str]]]:
    if sample.get("task") != "past":
        raise ValueError("Version2 retrieval labels require a past sample; CURRENT is outside the history retrieval pool")
    candidates = _candidate_ids(sample)
    negatives = set(_ids(sample.get("negative_clip_ids", []), "Explicit negative IDs", nonempty=False))
    gate = sample["narrative_qualification"]
    groups = gate.get("alternative_support_sets")
    if (not isinstance(groups, list) or not groups or gate.get("support_union_is_required_conjunction") is not False):
        raise ValueError("Version2 needs explicit alternative verified sufficient groups")
    answer = sample.get("answer_index")
    if type(answer) is not int or not 0 <= answer < 4:
        raise ValueError("The fixed answer index is missing or invalid")
    result, seen = [], set()
    for group in groups:
        if (not isinstance(group, Mapping) or group.get("all_original_independent_quality_gates_passed") is not True
                or type(group.get("answer_index")) is not int or group["answer_index"] != answer):
            raise ValueError("Each adopted group must retain the fixed answer and all original quality gates")
        ids = _ids(group.get("required_clip_ids"), "Required support IDs")
        _bounded_support(ids, candidates, negatives)
        identity = tuple(sorted(ids))
        if identity in seen:
            raise ValueError("Duplicate sufficient support sets cannot inflate the group metric")
        seen.add(identity)
        checks = group.get("checks")
        if not isinstance(checks, Mapping) or set(checks) != _CHECKS:
            raise ValueError("A verified boolean alone is insufficient; each adopted group needs all four check records")
        for record in checks.values():
            if (not isinstance(record, Mapping) or not isinstance(record.get("response"), Mapping)
                    or not isinstance(record.get("call"), Mapping) or not record["response"] or not record["call"]):
                raise ValueError("Each quality check must retain its actual response/call structure; intake proves provenance")
        if _ids(group.get("negative_clip_ids"), "Group explicit negative IDs", nonempty=False) != sample.get("negative_clip_ids", []):
            raise ValueError("Adopted groups must retain the sample's explicit verified negatives")
        result.append(ids)
    expected = sorted({cid for group in result for cid in group})
    if sample.get("verified_retrieval_positive_clip_ids") != expected:
        raise ValueError("Retrieval positives must exactly equal the sorted adopted verified-group union")
    return expected, result


def retrieval_positive_ids(sample: Mapping) -> list[str]:
    """Return v2's verified union; preserve legacy positive values and order.

    This function never infers negative labels from any omitted/raw candidate.
    Complete versioned intake is a caller precondition, not a boolean argument.
    """
    if _is_v2(sample):
        return _v2_support(sample)[0]
    return list(sample["positive_clip_ids"])


def verified_support_sets(sample: Mapping) -> list[list[str]] | None:
    """Return known OR-of-AND groups, or None when legacy sufficiency is unknown.

    An old sample's actual EvidenceOnly call supplies one conservative known
    sufficient set after original intake. Its selected training context may be a
    different occurrence and is not a sufficiency receipt. Other flat positives
    are never guessed to be jointly necessary or individually sufficient. None
    means omit this metric for that sample, rather than scoring it false.
    """
    if _is_v2(sample):
        return _v2_support(sample)[1]
    screens = sample.get("quality_screens", {})
    call = screens.get("evidence_only_call") if isinstance(screens, Mapping) else None
    mapping = call.get("clip_mapping") if isinstance(call, Mapping) else None
    if not isinstance(mapping, Mapping):
        return None
    history = [cid for label, cid in mapping.items() if isinstance(label, str) and re.fullmatch(r"H\d+", label)]
    if not history:
        return None
    ids = _ids(history, "Legacy EvidenceOnly historical IDs")
    negatives = set(_ids(sample.get("negative_clip_ids", []), "Explicit negative IDs", nonempty=False))
    _bounded_support(ids, _candidate_ids(sample), negatives)
    if not set(ids) <= set(sample["positive_clip_ids"]):
        raise ValueError("Legacy EvidenceOnly support must be inside its original retrieval positives")
    return [ids]


def complete_support_hit(sample: Mapping, ranked_clip_ids: Sequence[str], k: int) -> bool | None:
    """True iff at least one known sufficient group is wholly in the top k.

    The ranking must contain distinct explicit IDs. The metric does not derive
    new labels from its misses. Legacy samples without a historical EvidenceOnly receipt
    return None so the caller can report the metric's eligible denominator.
    """
    if type(k) is not int or k < 0:
        raise ValueError("Retrieval k must be a nonnegative integer")
    if (not isinstance(ranked_clip_ids, Sequence) or isinstance(ranked_clip_ids, (str, bytes))
            or any(not isinstance(cid, str) or not cid for cid in ranked_clip_ids)
            or len(set(ranked_clip_ids)) != len(ranked_clip_ids)):
        raise ValueError("Ranked retrieval IDs must be distinct strings")
    groups = verified_support_sets(sample)
    if groups is None:
        return None
    shown = set(ranked_clip_ids[:k])
    return any(set(group) <= shown for group in groups)
