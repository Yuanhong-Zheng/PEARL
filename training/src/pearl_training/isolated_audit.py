"""Single-question teacher audits; no other QA can leak facts or alias bindings."""

from __future__ import annotations

import hashlib
import json

from .batch_generate import alias_names, references, shuffled_question, stage_call
from .batch_prompts import VERIFY, REFERENCE_ONLY, NO_HISTORY, NO_CONCEPT
from .generate import video_content
from .teacher import image_part


PROTOCOL_VERSION = "isolated-v3-explicit-clips"


def labelled_scene(window):
    result = []
    for index in range(0, 64, 8):
        label = "L" + str(index // 8) if index < 56 else "CURRENT"
        result.append({"type": "text", "text": f"BEGIN CLIP {label}: relative [{index},{index + 8}) seconds."})
        result += video_content({"frames": window["frames"][index:index + 8]})
        result.append({"type": "text", "text": f"END CLIP {label}."})
    return result


def audit_content(window, concepts, question, mode, extras=()):
    """Build one blind task, preserving the other inputs during each ablation."""
    if mode not in {"full", "reference_only", "no_concept"}:
        raise ValueError("Unknown audit mode")
    names = alias_names(window, concepts)
    display, order = shuffled_question(question, names, window["window_id"] + question["id"] + ":isolated:" + mode)
    # The actual model never gets dataset QA indices or physical concept IDs.
    display["id"] = "item"
    display.pop("concept_ids", None)
    if mode != "full":
        display.pop("evidence", None)
    if mode == "reference_only":
        prompt = NO_HISTORY if question["task"] == "past" else REFERENCE_ONLY
        content = [{"type": "text", "text": prompt}]
        content += references(window, concepts, names)
        if question["task"] == "past":
            content.append({"type": "text", "text": "For this past question, the CURRENT clip [56,64) is also available. Only the earlier history is removed. Answer if the references and current pixels suffice."})
            content += video_content({"frames": window["frames"][56:]})
        task = "visual_ablation"
    elif mode == "no_concept":
        content = [{"type": "text", "text": NO_CONCEPT}]
        content += labelled_scene(window)
        task = "visual_ablation"
    else:
        content = [{"type": "text", "text": VERIFY + f"\nScene source start: {window['start']}. The current clip is relative [56,64); [0,56) is historical. L0..L6 are its seven 8-second historical clips."}]
        content += references(window, concepts, names) + labelled_scene(window)
        task = "visual_verification"
    if mode != "reference_only":
        for i, clip in enumerate(extras):
            content.append({"type": "text", "text": f"Older clip H{i}: source [{clip['start']},{clip['end']}) seconds."})
            content.extend(image_part(frame) for frame in clip["frames"])
    content.append({"type": "text", "text": "The ONLY question to evaluate; return its id exactly as item:\n" + json.dumps(display, ensure_ascii=False)})
    return content, task, order


def single_question_audit(teacher, directory, window, concepts, question, mode, extras=()):
    content, task, order = audit_content(window, concepts, question, mode, extras)
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    name = question["id"] + "_" + PROTOCOL_VERSION + "_" + mode + "_" + fingerprint[:12]
    def validate_response(response):
        if not isinstance(response, dict):
            raise ValueError("Single-question response must be an object")
        rows = response.get("questions")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError("Single-question audit must return exactly one result object")
        row = rows[0]
        if row.get("id") != "item":
            raise ValueError("Single-question audit returned the wrong item ID")
        if "answer_index" not in row or not (row["answer_index"] is None or (type(row["answer_index"]) is int and 0 <= row["answer_index"] < 4)):
            raise ValueError("Audit answer_index must be null or an integer 0 through 3")
        if mode != "full" and ("sufficient_information" not in row or row["sufficient_information"] is not None and type(row["sufficient_information"]) is not bool):
            raise ValueError("Ablation sufficient_information must be boolean or null")
        if "confirmed_negative_labels" in row and (not isinstance(row["confirmed_negative_labels"], list) or not all(isinstance(x, str) for x in row["confirmed_negative_labels"])):
            raise ValueError("Negative labels must be a list of strings")
    response, call = stage_call(teacher, directory, name, task, content, validator=validate_response)
    result = response["questions"][0]
    return result, {**call, "option_permutation": order, "expected_display_index": order.index(question["answer_index"]),
                    "context_isolation": "single_question; no proposed correct label; dataset IDs hidden",
                    "protocol_version": PROTOCOL_VERSION,
                    "request_content_sha256": fingerprint,
                    "past_history_ablation_retains_current": question["task"] == "past" and mode == "reference_only"}
