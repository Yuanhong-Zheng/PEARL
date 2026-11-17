#!/usr/bin/env python3
"""Per-category PSVU-Bench table in the layout of the paper's main results table.

Run from the repository root, with one ``--run`` per evaluated split:

    python -m eval.main_table \
        --run output_results/frame-level/qwen3vl_k4_n1_fps1 data/frame-level/annotations \
        --run output_results/video-level/qwen3vl_k4_n1_fps1 data/video-level/annotations \
        --name "Qwen3-VL-8B+PEARL"

Each ``--run`` pairs a directory of ``*_evaluation.json`` files, written by
``video_qa_inference.py --enable_rotation`` (as in ``scripts/eval_qwen.sh``), with the
annotation directory it was run on. Every Real-Time ("current-time qa") and Past-Time
("past-time qa") question takes its category from the ``"category"`` field of its
annotation, one of Presence, Behavior, Appearance, Location, Relation, Action,
Event-based and Time-based.

Scoring follows the paper:

* cyclic option rotation: a question is correct only if all four rotations, with the
  correct answer at A, B, C and D, are answered correctly. This is the ``is_correct``
  flag of ``*_evaluation.json`` under ``--enable_rotation``. When the matching
  ``*_result.json`` is present, its per-rotation flags are checked as well;
* the accuracy of a category is the percentage of its questions answered correctly;
* Real-Time Avg and Past-Time Avg are the means of their category accuracies;
* Overall Avg is the mean of Real-Time Avg and Past-Time Avg.

Missing categories, unlabeled questions, missing predictions and questions not scored
under four rotations are reported instead of stopping the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REAL_TIME = ("Presence", "Behavior", "Appearance", "Location", "Relation", "Action")
PAST_TIME = ("Event-based", "Time-based")
CATEGORIES = REAL_TIME + PAST_TIME
# Column headers of the paper's table.
HEADERS = {"Event-based": "Event", "Time-based": "Time"}
# QA types that are scored, and the QA group their categories belong to.
SCORED_QA_TYPES = {"current-time qa": "Real-Time", "past-time qa": "Past-Time"}
GROUP_OF = {**{name: "Real-Time" for name in REAL_TIME}, **{name: "Past-Time" for name in PAST_TIME}}

_ALIASES = {name.lower(): name for name in CATEGORIES}
_ALIASES.update({"event": "Event-based", "time": "Time-based",
                 "event based": "Event-based", "time based": "Time-based"})

# Diagnostics, in the order they are printed.
ISSUES = {
    "missing_prediction": "questions without a prediction",
    "uncategorized": "questions without a category",
    "invalid_category": "questions with an unknown category",
    "category_qa_type_conflict": "questions whose category does not match their QA type",
    "qa_type_mismatch": "predictions whose QA type differs from the annotation",
    "single_pass": "questions not scored under four option rotations",
    "rotation_flag_mismatch": "questions whose rotation flags disagree with is_correct",
    "unmatched_prediction": "predictions without a matching annotated question",
    "duplicate_annotation": "repeated question ids in the annotations",
    "duplicate_prediction": "repeated question ids in the predictions",
}


def normalize_category(value):
    """Return the canonical category name, ``None`` if unset, or raise ``ValueError``."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"category must be a string, got {value!r}")
    key = " ".join(value.replace("_", " ").split()).lower()
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"unknown category {value!r}")


def strict_correct(rotation_flags):
    """A question counts as correct only if all four option rotations are correct."""
    flags = list(rotation_flags)
    return len(flags) == 4 and all(flag is True or flag == 1 for flag in flags)


def _videos(data, path):
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of annotated videos")
    return data


def _read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_annotations(annotation_dir):
    """Return ``{(file_stem, qa_id): {"qa_type", "category"}}`` and repeated keys."""
    directory = Path(annotation_dir)
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation files (*.json) in {directory}")
    questions, duplicates = {}, []
    for path in files:
        for video in _videos(_read_json(path), path):
            for qa in video.get("timestamps", []):
                if qa.get("qa_type") not in SCORED_QA_TYPES:
                    continue
                key = (path.stem, qa.get("id"))
                if key in questions:
                    duplicates.append(key)
                    continue
                questions[key] = {"qa_type": qa["qa_type"], "category": qa.get("category")}
    return questions, duplicates


def _rotation_flags(result_path):
    """``{qa_id: flags}`` from a ``*_result.json`` (``None`` flags: rotation disabled)."""
    if not result_path.is_file():
        return None
    flags = {}
    for video in _videos(_read_json(result_path), result_path):
        for qa in video.get("timestamps", []):
            if qa.get("rotation_enabled"):
                values = qa.get("rotation_correct_flags")
                if values is None:
                    values = [row.get("is_correct") for row in qa.get("rotation_details", [])]
                flags[qa.get("id")] = list(values)
            else:
                flags[qa.get("id")] = None
    return flags


def load_predictions(result_dir):
    """Return ``{(file_stem, qa_id): {"qa_type", "correct", "protocol"}}`` and repeated keys.

    ``protocol`` is ``four_rotations`` (checked against ``*_result.json``),
    ``single_pass`` (rotation was disabled for this question), ``inconsistent``
    (rotation flags disagree with ``is_correct``) or ``unchecked`` (no ``*_result.json``).
    """
    directory = Path(result_dir)
    files = sorted(directory.glob("*_evaluation.json"))
    if not files:
        raise FileNotFoundError(f"No *_evaluation.json files in {directory}")
    predictions, duplicates = {}, []
    for path in files:
        stem = path.name[: -len("_evaluation.json")]
        rotations = _rotation_flags(directory / f"{stem}_result.json")
        for item in _read_json(path).get("details", []):
            if item.get("qa_type") not in SCORED_QA_TYPES:
                continue
            key = (stem, item.get("id"))
            if key in predictions:
                duplicates.append(key)
                continue
            correct = item.get("is_correct") is True
            protocol = "unchecked"
            if rotations is not None and key[1] in rotations:
                flags = rotations[key[1]]
                if flags is None:
                    protocol = "single_pass"
                elif strict_correct(flags) == correct and len(flags) == 4:
                    protocol = "four_rotations"
                else:
                    protocol = "inconsistent"
            predictions[key] = {"qa_type": item["qa_type"], "correct": correct, "protocol": protocol}
    return predictions, duplicates


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def score(runs, missing="wrong"):
    """Score ``[(result_dir, annotation_dir), ...]`` into the main-table layout.

    ``missing="wrong"`` counts annotated questions without a prediction as incorrect,
    ``missing="skip"`` leaves them out. Both are reported.
    """
    if missing not in {"wrong", "skip"}:
        raise ValueError("missing must be 'wrong' or 'skip'")
    cells = {name: {"correct": 0, "total": 0} for name in CATEGORIES}
    issues = defaultdict(list)
    protocols = Counter()
    for result_dir, annotation_dir in runs:
        questions, repeated_questions = load_annotations(annotation_dir)
        predictions, repeated_predictions = load_predictions(result_dir)
        source = Path(annotation_dir).as_posix()

        def label(key):
            return f"{source}/{key[0]}.json#{key[1]}"

        issues["duplicate_annotation"].extend(label(key) for key in repeated_questions)
        issues["duplicate_prediction"].extend(label(key) for key in repeated_predictions)
        issues["unmatched_prediction"].extend(
            label(key) for key in sorted(predictions.keys() - questions.keys(), key=str))
        for key in sorted(questions, key=str):
            question = questions[key]
            try:
                category = normalize_category(question["category"])
            except ValueError:
                issues["invalid_category"].append(f"{label(key)} ({question['category']!r})")
                continue
            if category is None:
                issues["uncategorized"].append(label(key))
                continue
            if SCORED_QA_TYPES[question["qa_type"]] != GROUP_OF[category]:
                issues["category_qa_type_conflict"].append(
                    f"{label(key)} ({category}, {question['qa_type']})")
            prediction = predictions.get(key)
            if prediction is None:
                issues["missing_prediction"].append(label(key))
                if missing == "skip":
                    continue
                correct = False
            else:
                correct = prediction["correct"]
                protocols[prediction["protocol"]] += 1
                if prediction["qa_type"] != question["qa_type"]:
                    issues["qa_type_mismatch"].append(label(key))
                if prediction["protocol"] == "single_pass":
                    issues["single_pass"].append(label(key))
                elif prediction["protocol"] == "inconsistent":
                    issues["rotation_flag_mismatch"].append(label(key))
            cells[category]["total"] += 1
            cells[category]["correct"] += int(correct)

    categories = {}
    for name, cell in cells.items():
        accuracy = 100.0 * cell["correct"] / cell["total"] if cell["total"] else None
        categories[name] = {**cell, "accuracy": accuracy}
    missing_categories = [name for name in CATEGORIES if categories[name]["total"] == 0]
    real_time = _mean(categories[name]["accuracy"] for name in REAL_TIME)
    past_time = _mean(categories[name]["accuracy"] for name in PAST_TIME)
    overall = (real_time + past_time) / 2 if real_time is not None and past_time is not None else None
    return {
        "categories": categories,
        "real_time_avg": real_time,
        "past_time_avg": past_time,
        "overall_avg": overall,
        "real_time_complete": not any(name in missing_categories for name in REAL_TIME),
        "past_time_complete": not any(name in missing_categories for name in PAST_TIME),
        "missing_categories": missing_categories,
        "missing_predictions": missing,
        "protocol_counts": dict(protocols),
        "issues": {name: issues[name] for name in ISSUES if issues[name]},
        "runs": [{"results": str(result), "annotations": str(annotation)} for result, annotation in runs],
    }


def _cell(value, complete=True):
    if value is None:
        return "--"
    return f"{value:.2f}" + ("" if complete else "*")


def row_values(report):
    """The eleven metric cells of one main-table row, as printed strings."""
    categories = report["categories"]
    values = [_cell(categories[name]["accuracy"]) for name in REAL_TIME]
    values.append(_cell(report["real_time_avg"], report["real_time_complete"]))
    values += [_cell(categories[name]["accuracy"]) for name in PAST_TIME]
    values.append(_cell(report["past_time_avg"], report["past_time_complete"]))
    complete = report["real_time_complete"] and report["past_time_complete"]
    values.append(_cell(report["overall_avg"], complete))
    return values


COLUMNS = ([HEADERS.get(name, name) for name in REAL_TIME] + ["Real-Time Avg"]
           + [HEADERS.get(name, name) for name in PAST_TIME] + ["Past-Time Avg", "Overall Avg"])


def to_markdown(report, name):
    lines = ["| Method | " + " | ".join(COLUMNS) + " |",
             "|---|" + "---:|" * len(COLUMNS),
             f"| {name} | " + " | ".join(row_values(report)) + " |"]
    return "\n".join(lines)


def to_latex(report, name):
    return f"{name} & " + " & ".join(row_values(report)) + r" \\"


def counts_table(report):
    lines = ["| Category | Correct | Total | Accuracy |", "|---|---:|---:|---:|"]
    for name in CATEGORIES:
        cell = report["categories"][name]
        lines.append(f"| {name} | {cell['correct']} | {cell['total']} | {_cell(cell['accuracy'])} |")
    return "\n".join(lines)


def notes(report, limit=5):
    lines = []
    if report["missing_categories"]:
        lines.append("Missing categories (no scored questions): " + ", ".join(report["missing_categories"])
                     + ". Averages marked * are taken over the categories present.")
    for name, description in ISSUES.items():
        items = report["issues"].get(name)
        if items:
            shown = ", ".join(items[:limit]) + (", ..." if len(items) > limit else "")
            lines.append(f"{len(items)} {description}: {shown}")
    if "missing_prediction" in report["issues"]:
        rule = "counted as incorrect" if report["missing_predictions"] == "wrong" else "left out"
        lines.append(f"Questions without a prediction are {rule} (--missing).")
    if report["protocol_counts"].get("unchecked"):
        lines.append(f"{report['protocol_counts']['unchecked']} predictions have no *_result.json; "
                     "their four-rotation flags were not checked.")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("RESULT_DIR", "ANNOTATION_DIR"),
                        help="Directory of *_evaluation.json files and the annotation directory it was run on; "
                             "repeat for each split (e.g. frame-level and video-level)")
    parser.add_argument("--name", default="PEARL", help="Row label")
    parser.add_argument("--missing", choices=("wrong", "skip"), default="wrong",
                        help="Count annotated questions without a prediction as incorrect (default) or skip them")
    parser.add_argument("--json", dest="json_path", help="Also write the full report to this JSON file")
    parser.add_argument("--latex", action="store_true", help="Also print the row in LaTeX")
    parser.add_argument("--fail-on-issues", action="store_true",
                        help="Exit with status 1 if a category is missing or any issue is reported")
    args = parser.parse_args(argv)

    report = score([tuple(run) for run in args.run], missing=args.missing)
    print(to_markdown(report, args.name))
    if args.latex:
        print()
        print(to_latex(report, args.name))
    print()
    print(counts_table(report))
    messages = notes(report)
    if messages:
        print()
        for message in messages:
            print("Note: " + message, file=sys.stderr)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": args.name, **report}, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    if args.fail_on_issues and (report["missing_categories"] or report["issues"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
