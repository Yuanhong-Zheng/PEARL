from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path

import requests

from .batch_prompts import ANNOTATE, GENERATE, NO_CONCEPT, REFERENCE_ONLY, VERIFY
from .generate import compile_sample, video_content
from .io import read_json, write_json, write_jsonl
from .local_embeddings import LocalEmbeddings
from .schema import check_split_isolation, validate_proposal, validate_sample
from .teacher import Teacher, TeacherBudgetExhausted, TeacherRequestError, TeacherRetryExhausted, image_part


def alias_names(window, concepts):
    pool = ["Aster", "Nori", "Vela", "Miro", "Lumi", "Zavi", "Pavo", "Tavi"]
    seed = int(hashlib.sha256(window["window_id"].encode()).hexdigest()[:8], 16)
    return {concept["id"]: pool[(seed + i) % len(pool)] for i, concept in enumerate(concepts)}


def alias_text(text, names):
    for cid, name in names.items():
        text = text.replace("{" + cid + "}", "{" + name + "}")
    return text


def references(window, concepts, names):
    result = []
    for concept in concepts:
        result.extend([{"type": "text", "text": "Runtime concept {" + names[concept["id"]] + "}: " + concept["description"]},
                       image_part(window["frames"][concept["reference_frame"]]["path"])])
    return result


def shuffled_question(question, names, seed):
    order = list(range(4))
    random.Random(seed).shuffle(order)
    # Only the task itself is visible. Unexpected teacher rationale/label fields
    # must never leak through a blacklist into blind verification or ablations.
    visible = {key: question[key] for key in ("id", "task", "question", "options", "evidence")}
    visible["question"] = alias_text(question["question"], names)
    visible["concept_ids"] = [names[cid] for cid in question["concept_ids"]]
    visible["options"] = [alias_text(question["options"][i], names) for i in order]
    return visible, order


def stage_call(teacher, directory, name, task, content, validator=None):
    path = directory / (name + ".json")
    if path.exists():
        saved = read_json(path)
        try:
            if validator:
                validator(saved["response"])
        except (ValueError, KeyError, TypeError, IndexError) as error:
            request_id = saved["call"].get("request_id", f"legacy-{time.time_ns()}")
            path.replace(directory / (name + f"_invalid_response_{request_id}.json"))
            write_json(directory / (name + f"_error_request_{request_id}.json"), {
                "category": "response_schema", "request_id": request_id,
                "type": type(error).__name__, "error": str(error)[:800]})
        else:
            return saved["response"], saved["call"]
    write_json(directory / (name + "_request_summary.json"), {
        "text_parts": [part["text"] for part in content if part["type"] == "text"],
        "image_count": sum(part["type"] == "image_url" for part in content),
        "content_sha256": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(),
    })
    last = None
    max_attempts = getattr(teacher, "max_retries", 1) + 1
    # Retain failed attempts across process restarts; resume cannot create an
    # unbounded retry loop for the same stage. Existing two-attempt logs count.
    previous = list(directory.glob(name + "_error_*.json"))
    if len(previous) >= max_attempts:
        failure = read_json(max(previous, key=lambda p: p.stat().st_mtime))
        raise TeacherRetryExhausted(name, max_attempts, failure)
    for attempt in range(len(previous), max_attempts):
        try:
            response, call = teacher.chat_json(task, content)
            if validator:
                try:
                    validator(response)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    write_json(directory / (name + f"_invalid_response_{call['request_id']}.json"),
                               {"response": response, "call": call})
                    raise TeacherRequestError(f"Teacher response schema failure: {type(error).__name__}: {error}",
                        request_id=call["request_id"], category="response_schema",
                        elapsed_seconds=call.get("elapsed_seconds")) from None
            write_json(path, {"response": response, "call": call, "stage": name})
            return response, call
        except (ValueError, KeyError, RuntimeError) as error:
            last = error
            if isinstance(error, TeacherBudgetExhausted):
                raise
            details = getattr(error, "details", {})
            delay = max(min(2 ** (attempt + 1), 32), details.get("retry_after_seconds") or 0)
            delay += (int(details.get("request_id") or attempt + 1) % 13) / 13
            record = {"type": type(error).__name__, "error": str(error)[:1200], **details,
                      "attempt": attempt + 1, "max_attempts": max_attempts,
                      "next_retry_seconds": delay if attempt + 1 < max_attempts else None}
            request_id = details.get("request_id") or f"local-{time.time_ns()}"
            write_json(directory / (name + f"_error_request_{request_id}.json"), record)
            print("REQUEST_FAILED", name, "attempt", attempt + 1, "/", max_attempts,
                  details.get("category", type(error).__name__), str(error)[:200], flush=True)
            if attempt + 1 < max_attempts:
                time.sleep(delay)
    raise TeacherRetryExhausted(name, max_attempts, record)


def compile_global(window, proposal, question, verification, call, source, negatives):
    sample = compile_sample(window, proposal, question, verification, call)
    sample["source_window_start"] = window["start"]
    sample["visible_start"] = 0
    sample["question_time"] = window["start"] + 64
    sample["evidence"] = [[window["start"] + a, window["start"] + b] for a, b in question["evidence"]]
    for concept in sample["concepts"]:
        concept["known_at"] += window["start"]
    sample["candidates"] = [clip for clip in source["clips"] if clip["end"] <= window["start"] + 56]
    sample["current_clip"] = window["clips"][-1]
    sample["negative_clip_ids"] = negatives
    sample["compilation_version"] = "batch-source-clock-v1"
    family = {"source": window["source_id"], "window": window["start"], "events": sorted(question["event_ids"]),
              "attribute": question["attribute"], "task": question["task"]}
    sample["base_question_id"] = "batch500:" + hashlib.sha256(json.dumps(family, sort_keys=True).encode()).hexdigest()[:24]
    sample["question_family"] = family
    return sample


def normalize_references(proposal):
    """Use actual brace references, as the released inference code does.

    The teacher sometimes lists contextual entities that are not referenced in
    the question/options. This fixes metadata only, never wording or answers.
    """
    concepts = proposal["concepts"]
    known = {concept["id"] for concept in concepts}
    questions, rejected, changes = [], [], []
    for original in proposal["questions"]:
        question = dict(original)
        text = question["question"] + "\n" + "\n".join(question["options"])
        actual = sorted(set(re.findall(r"\{(C[0-9])\}", text)))
        if actual != sorted(question["concept_ids"]):
            changes.append({"id": question["id"], "declared": question["concept_ids"], "actual": actual})
            question["concept_ids"] = actual
        try:
            if not set(actual) <= known:
                raise ValueError("Undefined concept reference")
            validate_proposal({"concepts": concepts, "questions": [question]}, 64, 64, max_questions=8, max_concepts=4, allow_reference_until_query=True)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append({"id": question.get("id"), "reasons": ["invalid_question_schema"], "error": str(error)})
            continue
        questions.append(question)
    return {"concepts": concepts, "questions": questions}, rejected, changes


def process_window(path, args):
    window = read_json(path)
    directory = Path(args.output) / "items" / window["window_id"]
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "completed.json"
    if completed.exists():
        return read_json(completed)
    teacher = Teacher(args.credential_readme, args.approval, Path(args.output) / "calls.sqlite3", proxy=args.proxy)
    names = None
    annotation, annotation_call = stage_call(teacher, directory, "annotation", "visual_annotation", [{"type": "text", "text": ANNOTATE}] + video_content(window))
    if annotation.get("usable") is not True or not annotation.get("concepts") or not annotation.get("events"):
        result = {"window_id": window["window_id"], "accepted": [], "rejected": [], "reason": "no_supported_events"}
        write_json(completed, result)
        return result
    event_ids = {event["id"] for event in annotation["events"] if event.get("visually_clear") is True}
    proposal, generation_call = stage_call(teacher, directory, "generation", "visual_generation", [{"type": "text", "text": GENERATE + "\nAnnotation:\n" + json.dumps(annotation, ensure_ascii=False)}])
    # Definitions are an immutable registry, not another generation target.
    proposal = {"concepts": annotation["concepts"], "questions": proposal["questions"]}
    proposal, initial_rejections, changes = normalize_references(proposal)
    write_json(directory / "normalized_proposal.json", {"proposal": proposal, "metadata_normalizations": changes,
               "rejected_schema_items": initial_rejections, "question_wording_changed": False, "answers_changed": False})
    names = alias_names(window, proposal["concepts"])
    usable, rejections = [], initial_rejections
    for question in proposal["questions"]:
        reasons = []
        if not question.get("event_ids") or not set(question["event_ids"]) <= event_ids:
            reasons.append("unsupported_event_id")
        if question.get("attribute") not in {"object", "container", "action", "location", "order", "recipient", "relation"}:
            reasons.append("unsupported_attribute")
        concepts = [concept for concept in proposal["concepts"] if concept["id"] in question["concept_ids"]]
        if any(window["frames"][c["reference_frame"]]["timestamp"] >= window["start"] + 64 for c in concepts):
            reasons.append("reference_after_query")
        positives = [clip for clip in window["clips"] if any(max(clip["start"] - window["start"], a) < min(clip["end"] - window["start"], b) for a, b in question["evidence"])]
        if len(positives) > 3:
            reasons.append("evidence_exceeds_training_media_budget")
        if reasons:
            rejections.append({"id": question["id"], "reasons": reasons})
        else:
            usable.append(question)
    if not usable:
        result = {"window_id": window["window_id"], "accepted": [], "rejected": rejections}
        write_json(completed, result)
        return result
    source = read_json(window["source_manifest"])
    local = LocalEmbeddings(args.embeddings)
    prefix = [clip for clip in source["clips"] if clip["end"] <= window["start"]]
    extra = {}
    for question in usable:
        query = {"concepts": [{"name": names[c["id"]], "description": c["description"], "image": window["frames"][c["reference_frame"]]["path"]} for c in proposal["concepts"]],
                 "question": alias_text(question["question"], names)}
        for clip in local.retrieve(query, prefix, k=3):
            extra[clip["id"]] = clip
    extras = sorted(extra.values(), key=lambda clip: -clip["retrieval_score"])[:6]
    labels = {"L" + str(i): clip for i, clip in enumerate(window["clips"][:-1])}
    labels.update({"H" + str(i): clip for i, clip in enumerate(extras)})
    visible, permutations = [], {}
    for question in usable:
        item, order = shuffled_question(question, names, window["window_id"] + question["id"] + ":verify")
        visible.append(item)
        permutations[question["id"]] = order
    content = [{"type": "text", "text": VERIFY + f"\nScene source start {window['start']} seconds. L0..L6 refer to relative [0,8),...,[48,56)."}]
    content += references(window, proposal["concepts"], names) + video_content(window)
    for i, clip in enumerate(extras):
        content.append({"type": "text", "text": f"Older historical clip H{i}: source [{clip['start']},{clip['end']}) seconds."})
        for frame in clip["frames"]:
            content.append(image_part(frame))
    content.append({"type": "text", "text": "Questions, proposed evidence, shuffled options; answers hidden:\n" + json.dumps(visible, ensure_ascii=False)})
    checked, checked_call = stage_call(teacher, directory, "verification", "visual_verification", content)
    checked = {row["id"]: row for row in checked["questions"]}

    def reduced_stage(name, prompt, media):
        display, orders = [], {}
        for question in usable:
            row, order = shuffled_question(question, names, window["window_id"] + question["id"] + ":" + name)
            row.pop("evidence", None)
            display.append(row)
            orders[question["id"]] = order
        response, call = stage_call(teacher, directory, name, "visual_ablation", [{"type": "text", "text": prompt}] + media + [{"type": "text", "text": json.dumps(display, ensure_ascii=False)}])
        return {row["id"]: row for row in response["questions"]}, orders

    reference_only, reference_orders = reduced_stage("reference_only", REFERENCE_ONLY, references(window, proposal["concepts"], names))
    no_concept, no_concept_orders = reduced_stage("no_concept", NO_CONCEPT, video_content(window))
    accepted = []
    for question in usable:
        qid = question["id"]
        check = checked.get(qid, {})
        reasons = []
        required = ["concept_bindings_valid", "visually_answerable", "unique_answer", "evidence_sufficient", "requires_concept_binding", "no_reference_shortcut"]
        if not all(check.get(key) is True for key in required):
            reasons.append("visual_verification_failed")
        if question["task"] == "past" and check.get("requires_history_for_past") is not True:
            reasons.append("history_not_required")
        expected = permutations[qid].index(question["answer_index"])
        if type(check.get("answer_index")) is not int or check.get("answer_index") != expected:
            reasons.append("blind_answer_disagrees")
        for name, responses, orders in [("reference_only", reference_only, reference_orders), ("no_concept", no_concept, no_concept_orders)]:
            row = responses.get(qid)
            if row is None:
                reasons.append(name + "_missing")
            elif row.get("sufficient_information") is True and type(row.get("answer_index")) is int and row.get("answer_index") == orders[qid].index(question["answer_index"]):
                reasons.append(name + "_shortcut")
        positive_ids = {clip["id"] for clip in window["clips"] if any(max(clip["start"] - window["start"], a) < min(clip["end"] - window["start"], b) for a, b in question["evidence"])}
        negatives = []
        negative_labels = sorted(set(check.get("confirmed_negative_labels", [])), key=lambda label: (not label.startswith("H"), label))
        for label in negative_labels:
            if label not in labels:
                reasons.append("unknown_negative_label")
            elif labels[label]["id"] in positive_ids:
                reasons.append("conflicting_negative_label")
            else:
                negatives.append(labels[label]["id"])
        if reasons:
            rejections.append({"id": qid, "reasons": sorted(set(reasons)), "verification": check})
            continue
        sample = compile_global(window, proposal, question, check, checked_call, source, list(dict.fromkeys(negatives)))
        sample["data_round"] = "batch500"
        sample["teacher_annotation_call"] = annotation_call
        sample["teacher_generation_call"] = generation_call
        sample["quality_screens"] = {"reference_only": reference_only[qid], "no_concept": no_concept[qid],
                                     "independent_option_permutations": True, "teacher_rotations_per_screen": 1,
                                     "causal_rule": "reference_and_evidence_before_query; no_order_required_between_them"}
        validate_sample(sample)
        accepted.append(sample)
    # A confirmed swap pair is one family with two mapping variants.
    pairs = {}
    for question in usable:
        if question.get("swap_pair_id"):
            pairs.setdefault(question["swap_pair_id"], []).append(question)
    by_id = {sample["sample_id"].rsplit(":", 1)[-1]: sample for sample in accepted}
    for pair_id, questions in pairs.items():
        if len(questions) != 2 or any(q["id"] not in by_id for q in questions):
            continue
        first, second = questions
        if len(first["concept_ids"]) != 1 or len(second["concept_ids"]) != 1:
            continue
        a, b = first["concept_ids"][0], second["concept_ids"][0]
        if a == b or first["options"] != second["options"] or first["answer_index"] == second["answer_index"] or any(re.search(r"\{C[0-9]\}", option) for option in first["options"]):
            continue
        if first["question"].replace("{" + a + "}", "{TARGET}") != second["question"].replace("{" + b + "}", "{TARGET}"):
            continue
        left, right = by_id[first["id"]], by_id[second["id"]]
        swapped = {**names, a: names[b], b: names[a]}
        right["question"] = alias_text(second["question"], swapped)
        right["options"] = [alias_text(option, swapped) for option in second["options"]]
        for concept in right["concepts"]:
            concept["name"] = swapped[concept["id"]]
        right["base_question_id"] = left["base_question_id"]
        left["counterfactual_group"] = right["counterfactual_group"] = window["window_id"] + ":" + pair_id
        left["counterfactual_variant"], right["counterfactual_variant"] = 0, 1
        if left["question"] != right["question"]:
            raise ValueError("Swap pair did not preserve literal question wording")
    result = {"window_id": window["window_id"], "split": window["split"], "accepted": accepted,
              "rejected": rejections, "proposed_questions": len(proposal["questions"])}
    write_json(completed, result)
    return result


def family_counts(output, completed_name="completed.json"):
    families = {}
    for path in (Path(output) / "items").glob("*/" + completed_name):
        for sample in read_json(path)["accepted"]:
            families[sample["base_question_id"]] = (sample["split"], sample["task"])
    return Counter(families.values())


def export(output, completed_name="completed.json", filename="samples.jsonl"):
    rows = []
    for path in sorted((Path(output) / "items").glob("*/" + completed_name)):
        rows.extend(read_json(path)["accepted"])
    # Deterministic source/window/family selection; all paired variants stay together.
    families = {}
    for row in rows:
        families.setdefault(row["base_question_id"], []).append(row)
    selected = []
    counts = Counter()
    limits = {("train", "past"): 225, ("train", "current"): 225,
              ("validation", "past"): 25, ("validation", "current"): 25}
    for fid, variants in families.items():
        key = variants[0]["split"], variants[0]["task"]
        if counts[key] < limits[key]:
            selected.extend(variants)
            counts[key] += 1
    selected_ids = {row["base_question_id"] for row in selected}
    # Balance tasks where possible, then fill from other valid families. Task
    # balance is a sampling preference, not an unsupported data exclusion rule.
    for fid, variants in families.items():
        split = variants[0]["split"]
        target = 450 if split == "train" else 50
        if fid not in selected_ids and sum(n for (s, t), n in counts.items() if s == split) < target:
            selected.extend(variants)
            counts[(split, variants[0]["task"])] += 1
            selected_ids.add(fid)
    check_split_isolation(selected)
    write_jsonl(Path(output) / filename, selected)
    summary = {"selected_question_families": sum(counts.values()), "selected_variants": len(selected),
               "by_split_task": {s + ":" + t: n for (s, t), n in counts.items()},
               "all_accepted_families": len(families), "completed_windows": len(list((Path(output) / "items").glob("*/" + completed_name))),
               "counterfactual_families": len({row["base_question_id"] for row in selected if row.get("counterfactual_group")})}
    summary_name = "summary.json" if completed_name == "completed.json" else "summary_isolated.json"
    summary["qualification"] = "provisional_pooled" if completed_name == "completed.json" else "single_question_verified_and_ablated"
    write_json(Path(output) / summary_name, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--credential-readme", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--pilot-windows", required=True)
    parser.add_argument("--embeddings", default="http://127.0.0.1:28780")
    parser.add_argument("--proxy", default="http://your-proxy-host:8080")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-windows", type=int)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    excluded = read_json(args.pilot_windows)
    windows = []
    for path in sorted((Path(args.prepared) / "windows").glob("*.json")):
        window = read_json(path)
        if any(window["source_id"] == old["source_id"] and max(window["start"], old["start"]) < min(window["start"] + 64, old["start"] + 64) for old in excluded):
            continue
        windows.append((path, window))
    # Interleave sources and reserve validation coverage throughout generation.
    random.Random(42).shuffle(windows)
    windows.sort(key=lambda item: 0 if item[1]["split"] == "validation" else 1)
    validation = [item for item in windows if item[1]["split"] == "validation"]
    training = [item for item in windows if item[1]["split"] == "train"]
    plan = []
    while training or validation:
        if validation:
            plan.append(validation.pop())
        for _ in range(5):
            if training:
                plan.append(training.pop())
    if args.max_windows:
        plan = plan[:args.max_windows]
    failures = []
    teacher = Teacher(args.credential_readme, args.approval, Path(args.output) / "calls.sqlite3", proxy=args.proxy)
    iterator = iter(plan)
    pending = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        stopped = False
        while pending or not stopped:
            while not stopped and len(pending) < args.workers:
                counts = family_counts(args.output)
                if counts[("train", "past")] + counts[("train", "current")] >= 450 and counts[("validation", "past")] + counts[("validation", "current")] >= 50:
                    stopped = True
                    break
                try:
                    path, window = next(iterator)
                except StopIteration:
                    stopped = True
                    break
                if counts[(window["split"], "past")] + counts[(window["split"], "current")] >= (50 if window["split"] == "validation" else 450):
                    continue
                pending[pool.submit(process_window, path, args)] = window["window_id"]
                print("START", window["window_id"], window["split"], flush=True)
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                wid = pending.pop(future)
                try:
                    result = future.result()
                    print("DONE", wid, "accepted_variants", len(result["accepted"]), "rejected", len(result["rejected"]), flush=True)
                except Exception as error:
                    failures.append({"window": wid, "type": type(error).__name__, "error": str(error)[:300]})
                    write_json(Path(args.output) / "failures.json", failures)
                    print("ERROR", wid, type(error).__name__, str(error)[:150], flush=True)
                    # Fail closed for protocol/schema errors; retain all completed work.
                    stopped = True
                summary = export(args.output)
                write_json(Path(args.output) / "usage.json", teacher.usage())
                print("PROGRESS", json.dumps(summary), "requests", len(teacher.usage()), flush=True)
    print("FINISHED", json.dumps(export(args.output)), flush=True)


if __name__ == "__main__":
    main()
