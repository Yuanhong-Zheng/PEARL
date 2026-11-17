"""Pure planning helpers for the isolated NarrativeMovies experiment.

These helpers neither generate nor qualify QA. In particular, source-clock
support mapping does not establish the joint visual evidence needed for a
before/after question. The original V4 checks and an additional independent
temporal gate remain necessary before accepting such a question.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


CLIP_SECONDS = 8
CATEGORY_UNITS = {
    "Presence": 2,
    "Behavior": 2,
    "Appearance": 2,
    "Location": 2,
    "Relation": 2,
    "Event-based": 5,
    "Time-based": 5,
}
CATEGORY_TASKS = {
    category: "current" if category not in {"Event-based", "Time-based"} else "past"
    for category in CATEGORY_UNITS
}


def category_targets(total: int) -> dict[str, int]:
    """Return exact aggregate quotas; do not invent per-split rounding rules."""
    denominator = sum(CATEGORY_UNITS.values())
    if type(total) is not int or total <= 0 or total % denominator:
        raise ValueError("Exact category quotas require a positive multiple of 20")
    return {category: total * units // denominator for category, units in CATEGORY_UNITS.items()}


def validate_category(category: str, task: str) -> None:
    """Keep the seven question categories separate from event attributes.

    Time-based means temporal order (before/after), not numeric timestamps.
    Behavior questions can query actions; that does not register an Action concept.
    """
    if not isinstance(category, str) or category not in CATEGORY_TASKS:
        raise ValueError("Unknown NarrativeMovies QA category")
    if task != CATEGORY_TASKS[category]:
        raise ValueError("QA category does not match its current/past task")


def _time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative time")
    return float(value)


def _interval(value: Sequence[float], name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a half-open [start, end) interval")
    start, end = (_time(value[0], name), _time(value[1], name))
    if start >= end:
        raise ValueError(f"{name} must have positive duration")
    return start, end


def annotation_interval_to_source(
    relative_interval: Sequence[float], annotation_start: float, annotation_duration: float = 64
) -> list[float]:
    """Map one annotation's local interval to the continuous stream clock."""
    start = _time(annotation_start, "annotation start")
    duration = _time(annotation_duration, "annotation duration")
    left, right = _interval(relative_interval, "relative interval")
    if duration <= 0 or right > duration:
        raise ValueError("Evidence extends beyond its annotation pixels")
    return [start + left, start + right]


def causal_clips(
    clips: Sequence[Mapping[str, Any]], question_time: float, *, visible_start: float = 0,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Return all completed history and the final eight-second current clip.

    Input may be the whole source manifest. Future clips are never returned.
    The retained prefix begins at explicit visible_start, with no clock reset.
    """
    query = _time(question_time, "question time")
    origin = _time(visible_start, "visible start")
    if query < origin + CLIP_SECONDS:
        raise ValueError("The query needs a complete current clip")
    ordered = sorted(clips, key=lambda clip: _time(clip["start"], "clip start"))
    seen = set()
    for clip in ordered:
        cid = clip["id"]
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("Clip IDs must be nonempty and unique")
        seen.add(cid)
    prefix = [clip for clip in ordered if _time(clip["start"], "clip start") < query
              and _time(clip["end"], "clip end") > origin]
    cursor = origin
    for clip in prefix:
        left, right = _interval([clip["start"], clip["end"]], "clip")
        if abs(left - cursor) > 1e-6 or abs(right - left - CLIP_SECONDS) > 1e-6:
            raise ValueError("Causal prefix must contain contiguous eight-second clips from visible_start")
        cursor = right
    if not prefix or abs(cursor - query) > 1e-6:
        raise ValueError("Question time must end a complete causal clip")
    return prefix[:-1], prefix[-1]


def evidence_clip_ids(
    evidence: Sequence[Sequence[float]], clips: Sequence[Mapping[str, Any]],
    question_time: float, category: str, *, max_support_clips: int = 3, visible_start: float = 0,
) -> list[str]:
    """Map source-clock evidence to clip IDs without limiting it to local64s.

    This produces proposed support IDs only. It never labels a negative, accepts
    a sample, or substitutes for independent blind discovery of visual support.
    """
    if not isinstance(category, str) or category not in CATEGORY_TASKS:
        raise ValueError("Unknown NarrativeMovies QA category")
    if type(max_support_clips) is not int or not 1 <= max_support_clips <= 3:
        raise ValueError("The support limit must preserve the original one-to-three-clip budget")
    if not isinstance(evidence, (list, tuple)) or not evidence:
        raise ValueError("At least one evidence interval is required")
    history, current = causal_clips(clips, question_time, visible_start=visible_start)
    task = CATEGORY_TASKS[category]
    eligible = history if task == "past" else [current]
    intervals = [_interval(value, "evidence") for value in evidence]
    left_bound = _time(visible_start, "visible start") if task == "past" else current["start"]
    right_bound = current["start"] if task == "past" else current["end"]
    if any(left < left_bound or right > right_bound for left, right in intervals):
        raise ValueError("Evidence crosses its permitted current/past causal boundary")
    support = [clip["id"] for clip in eligible
               if any(max(clip["start"], left) < min(clip["end"], right) for left, right in intervals)]
    if not support or len(support) > max_support_clips:
        raise ValueError("Evidence exceeds the supported visual media budget")
    return support


def validate_reference_time(known_at: float, question_time: float) -> None:
    """References may follow old evidence, but must precede the query."""
    if _time(known_at, "reference time") >= _time(question_time, "question time"):
        raise ValueError("A concept reference must precede the query")
