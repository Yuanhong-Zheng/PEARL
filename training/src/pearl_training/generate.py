from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .io import read_json, read_jsonl, write_json, write_jsonl
from .schema import check_split_isolation, validate_proposal, validate_sample
from .teacher import Teacher, image_part


GENERATION_PROMPT = """Create visual training questions for personalized streaming video understanding.
You see a 64-second video window sampled at one frame per second, with exact frame
indices and relative times. The stream starts at this window, and the user asks
at relative time 64. Clips 0–6 are completed history [0,8),...,[48,56); clip 7
is the current clip [56,64). Refer only to what is visually verifiable in the
provided frames. Do not use subtitles, dialogue, actual character names, movie
knowledge, or events outside this window.

Select 1–3 clearly visible entities (C0, C1, C2). Each concept is defined by a
reference frame, preferably in the first 24 seconds, and a short appearance-only
description uniquely identifying that entity IN THAT REFERENCE FRAME. Do not
put later events, answers, or the character's real name in this description.
The concept may persist across views. References must precede the current clip.

Write up to four four-option questions: ideally two current questions about the
last clip and two past questions that require a completed historical clip.
Use placeholders such as {C0} for these runtime-defined names. The options
must have exactly one visually supported answer, with plausible distractors.
Prefer entity/action distinctions, objects handled, locations, and short event
order relations. Make questions depend on the concept binding, not generic scene
knowledge. When two entities are suitable, include a pair of identically worded
questions differing only in {C0}/{C1}, with different supported answers. Do not
invent such a pair if the video does not support it.

Evidence intervals are relative seconds, half-open [start,end), supported by
visible sampled frames. Past evidence must end by 56; current evidence must
start at or after 56. Include all needed evidence intervals, not just a salient
timestamp. For past questions, list historical clip indices that you can confirm
do not provide the requested evidence. Do not mark an unexamined or potentially
valid clip as a negative. An empty negatives list is allowed.

Return ONLY a JSON object with this schema (answer_index is zero-based):
{
 "concepts":[{"id":"C0","reference_frame":5,"description":"the person on the left wearing a plain red jacket"}],
 "questions":[{"id":"q0","task":"past","concept_ids":["C0"],
 "question":"What did {C0} pick up earlier in this stream?",
 "options":["A cup","A book","A key","A phone"],"answer_index":0,
 "evidence":[[24,27]],"negative_clip_indices":[1,2]}]
}
Return fewer questions when necessary rather than unsupported questions.
"""


def video_content(window):
    parts = []
    for frame in window["frames"]:
        parts.append({"type": "text", "text": f"Frame {frame['index']}; relative time {frame['relative_time']:.3f} seconds."})
        parts.append(image_part(frame["path"]))
    return parts


def verification_content(window, proposal, question):
    parts = [{"type": "text", "text": "You independently audit a personalized video QA item. You are NOT given its proposed correct answer. Answer from the pixels and runtime concept definitions below. The stream is this 64-second window; the question is asked at 64 seconds. Clips 0–6 are historical 8-second clips and clip 7 [56,64) is current. Check entity identity across views carefully. A description identifies the target in its reference image, not every later frame."}]
    for concept in proposal["concepts"]:
        parts.append({"type": "text", "text": f"Runtime concept {{{concept['id']}}}: {concept['description']}. Reference follows."})
        parts.append(image_part(window["frames"][concept["reference_frame"]]["path"]))
    parts.extend(video_content(window))
    blinded = {key: value for key, value in question.items() if key not in {"answer_index"}}
    parts.append({"type": "text", "text": "Question and proposed evidence (the correct answer is hidden):\n" + json.dumps(blinded, ensure_ascii=False) + '\nReturn ONLY JSON: {"answer_index":0,"concept_bindings_valid":true,"question_visually_answerable":true,"unique_answer":true,"evidence_sufficient":true,"negative_clips_valid":true,"brief_reason":"one short factual explanation"}. Use false for any unsupported check; guessing is not verification. Evidence_sufficient means ALL information necessary to answer is in the proposed intervals with the concept references. Negative_clips_valid means the specified negatives do not contain alternate valid evidence.'})
    return parts


def compile_sample(window, proposal, question, verification, call_info):
    alias_pool = ["Aster", "Nori", "Vela", "Miro", "Lumi", "Zavi", "Pavo", "Tavi"]
    seed = int(hashlib.sha256(window["window_id"].encode()).hexdigest()[:8], 16)
    aliases = {concept["id"]: alias_pool[(seed + index) % len(alias_pool)] for index, concept in enumerate(proposal["concepts"])}

    def replace(text):
        for cid, name in aliases.items():
            text = text.replace("{" + cid + "}", "{" + name + "}")
        return text

    concepts = []
    for concept in proposal["concepts"]:
        frame = window["frames"][concept["reference_frame"]]
        concepts.append({"id": concept["id"], "name": aliases[concept["id"]], "image": frame["path"], "description": concept["description"], "known_at": frame["relative_time"]})
    start = window["start"]
    # Episode time is relative to stream start; source offsets remain provenance.
    clips = [{**clip, "source_start": clip["start"], "source_end": clip["end"],
              "start": clip["start"] - start, "end": clip["end"] - start} for clip in window["clips"]]
    evidence = question["evidence"]
    positive_ids = [clip["id"] for clip in clips if any(max(clip["start"], a) < min(clip["end"], b) for a, b in evidence)]
    sid = window["window_id"] + ":" + question["id"]
    return {
        "sample_id": sid, "base_question_id": sid, "source_id": window["source_id"],
        "source_group": window["source_group"], "source_sha256": window["sha256"],
        "source_overlap_with_original_benchmark": True,
        "split": window["split"], "source_window_start": start, "visible_start": 0, "question_time": window["duration"],
        "task": question["task"], "concepts": concepts,
        "question": replace(question["question"]), "options": [replace(option) for option in question["options"]],
        "answer_index": question["answer_index"], "evidence": evidence,
        "candidates": clips[:-1], "current_clip": clips[-1],
        "positive_clip_ids": positive_ids,
        "negative_clip_ids": [window["clips"][i]["id"] for i in question.get("negative_clip_indices", [])],
        "verification": {**verification, "accepted": True, **call_info},
        "compilation_version": "relative-clock-v2",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True)
    parser.add_argument("--credential-readme", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-id", action="append")
    parser.add_argument("--past-only", action="store_true")
    parser.add_argument("--proxy", default="http://your-proxy-host:8080")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    teacher = Teacher(args.credential_readme, args.approval, output / "calls.sqlite3", proxy=args.proxy)
    windows = read_json(args.windows)
    if args.window_id:
        windows = [window for window in windows if window["window_id"] in args.window_id]
    for window in windows:
        directory = output / window["window_id"]
        directory.mkdir(exist_ok=True)
        proposal_file = directory / "proposal.json"
        if proposal_file.exists():
            proposal = read_json(proposal_file)["proposal"]
        else:
            print("GENERATE", window["window_id"], flush=True)
            prompt = GENERATION_PROMPT
            if args.past_only:
                prompt += "\nFor this supplemental window return at most THREE questions, ALL of task past. Each target concept reference must be at least EIGHT SECONDS before the first supporting evidence interval. Avoid any question whose answer is already visible in its concept reference image or written in the reference subtitles. Prefer questions about later actions, later objects handled, and later spatial relations. Give all four options in a balanced order, not always with the correct answer first. If supported, include a concept-swap pair with identical question wording and identical options but different target concepts and different correct answers."
            proposal, info = teacher.chat_json("visual_generation", [{"type": "text", "text": prompt}] + video_content(window))
            if args.past_only and (len(proposal.get("questions", [])) > 3 or any(q.get("task") != "past" for q in proposal.get("questions", []))):
                write_json(directory / "invalid_supplemental_proposal.json", {"proposal": proposal, "call": info})
                raise ValueError("Supplemental proposal violates the approved remaining-call allocation")
            write_json(proposal_file, {"proposal": proposal, "call": info, "prompt_version": "entity-evidence-past-v2" if args.past_only else "entity-evidence-v1"})
        validate_proposal(proposal, len(window["frames"]), window["duration"])
        for question in proposal["questions"]:
            verification_file = directory / (question["id"] + "_verification.json")
            if verification_file.exists():
                continue
            print("VERIFY", window["window_id"], question["id"], flush=True)
            verification, info = teacher.chat_json("visual_verification", verification_content(window, proposal, question))
            required = ["concept_bindings_valid", "question_visually_answerable", "unique_answer", "evidence_sufficient", "negative_clips_valid"]
            accepted = all(verification.get(key) is True for key in required) and verification.get("answer_index") == question["answer_index"]
            write_json(verification_file, {"verification": verification, "call": info, "accepted": accepted})
            if accepted:
                sample = compile_sample(window, proposal, question, verification, info)
                validate_sample(sample)
                write_json(directory / (question["id"] + "_sample.json"), sample)
            print("RESULT", window["window_id"], question["id"], "accepted" if accepted else "rejected", flush=True)
            write_json(output / "usage.json", teacher.usage())
    samples = [read_json(path) for path in sorted(output.glob("*/q*_sample.json"))]
    check_split_isolation(samples)
    write_jsonl(output / "samples.jsonl", samples)
    write_json(output / "usage.json", teacher.usage())
    print("ACCEPTED_TOTAL", len(samples), flush=True)


if __name__ == "__main__":
    main()
