#!/usr/bin/env python3
"""Audit completed fixed-checkpoint ablations and count paired QA changes.

Read-only model evaluation: no new inference, filtering, checkpoint selection or
metric changes. Readers with the same retriever must receive identical ranked
and expanded evidence, and every arm must retain the same question rotations.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pearl_training.io import read_json, write_json
from pearl_training.public112_adapter import summarize_results


ARMS = ("baseline", "embedding_only", "reader_only", "joint")


def load_rows(root, arm, sources):
    qa, traces = {}, {}
    for source in sources:
        name = Path(source["annotation"]).stem
        for video in read_json(root / arm / "results" / (name + "_result.json")):
            for row in video["timestamps"]:
                key = (name, row["id"])
                if key in qa:
                    raise ValueError("Repeated QA result")
                qa[key] = row
        for row in read_json(root / arm / "traces" / (name + ".json")):
            key = (name, row["qa_id"])
            if key in traces:
                raise ValueError("Repeated retrieval trace")
            traces[key] = row
    if len(qa) != 1316 or set(qa) != set(traces):
        raise ValueError("Full matching public112 QA/traces are required")
    return qa, traces


def question_inputs(row):
    return {"question": row["question"], "qa_type": row["qa_type"], "gt": row["gt"],
            "rotations": [{key: rotation[key] for key in ("rotated_question", "target_gt")}
                          for rotation in row["rotation_details"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve(strict=True)
    plan = read_json(root / "plan.json")
    # Validation also rechecks every completed receipt and shared cache hash.
    summaries = {arm: summarize_results(plan, arm) for arm in ARMS}
    rows = {arm: load_rows(root, arm, plan["corpus"]["sources"]) for arm in ARMS}
    qa_inputs, retrieval_inputs, rankings = {}, {}, {}
    baseline_qa, baseline_traces = rows["baseline"]
    for arm, (qa, traces) in rows.items():
        if set(qa) != set(baseline_qa):
            raise ValueError("Compared arms have different questions")
        qa_inputs[arm] = all(question_inputs(row) == question_inputs(baseline_qa[key])
                             for key, row in qa.items())
        retrieval_inputs[arm] = all(all(trace[field] == baseline_traces[key][field]
            for field in ("query", "eligible_clip_count", "qa_type")) for key, trace in traces.items())
    for before, after in (("baseline", "reader_only"), ("embedding_only", "joint")):
        left, right = rows[before][1], rows[after][1]
        rankings[before + "_vs_" + after] = all(
            [item["clip_id"] for item in left[key]["ranked_top10"]]
                == [item["clip_id"] for item in right[key]["ranked_top10"]]
            and left[key]["expanded_top4"] == right[key]["expanded_top4"] for key in left)
    paired = {}
    for before, after in (("baseline", "embedding_only"), ("baseline", "reader_only"),
                          ("baseline", "joint"), ("embedding_only", "joint"), ("reader_only", "joint")):
        counts = defaultdict(Counter)
        for key, old in rows[before][0].items():
            new = rows[after][0][key]
            a, b = old["rotation_all_correct"], new["rotation_all_correct"]
            counts[old["qa_type"]]["both_correct" if a and b else "both_wrong" if not a and not b
                                   else "gain" if b else "loss"] += 1
        paired[before + "_vs_" + after] = {task: dict(value) for task, value in counts.items()}
    invariants = {"identical_question_rotations": qa_inputs, "identical_retrieval_queries_and_pools": retrieval_inputs,
                  "reader_training_preserves_retrieved_evidence": rankings}
    passed = all(all(value.values()) for value in invariants.values())
    result = {"status": "passed" if passed else "input_mismatch_needs_review", "model_calls": 0,
              "qualified_training_groups": 500, "input_invariants": invariants,
              "summaries": summaries, "paired_strict_qa": paired}
    write_json(root / "paired_comparison.json", result)
    print({"status": result["status"], "completed_arms": list(summaries), "model_calls": 0}, flush=True)
    if not passed:
        raise ValueError("Compared reader arms differ in supposedly shared inputs; inspect paired_comparison.json")


if __name__ == "__main__":
    main()
