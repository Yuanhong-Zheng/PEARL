"""Restore the successful Flash500 window workflow on measured movie streams.

Only input clock/provenance and empty-start orchestration are adapted. Annotation
and generation use the old64s prompts/compiler, and qualification uses the old
V4 gates with measured timestamps. No temporal discovery, fixed global reference
bank, CURRENT bridge, alternative-group review or seven-category quota is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import runpy
import shutil
import sqlite3
import threading
from urllib.parse import urlparse

from .batch_prompts import ANNOTATE
from .generate import video_content
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .local_production import can_dispatch
from .local_teacher import LocalTeacher
from .narrative_qualification import persistent_stage_call
from .replenishment import NEW_QUESTIONS, make_candidates
from .schema import check_split_isolation, validate_sample
from .strict_generate import annotation_format
from .teacher import TeacherBudgetExhausted, TeacherRequestError, TeacherRetryExhausted


PROTOCOL = "restored-movie-window-v4-production-v1"
ROOT = Path(__file__).resolve().parents[2]
TARGET = {"train": 450, "validation": 50}
SCALE_TARGET = {"train": 9000, "validation": 1000}
_OLD = runpy.run_path(str(ROOT / "scripts/replenish_v4.py"), run_name="restored_window_generation_library")
reserve_fact = _OLD["reserve_fact"]
generation_format = _OLD["generation_format"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def bound(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def checked_target(value=None):
    """Only the two user-selected stage boundaries; never trim imported rows."""
    value = TARGET if value is None else value
    if (not isinstance(value, dict) or set(value) != set(TARGET)
            or any(type(number) is not int for number in value.values())
            or value not in (TARGET, SCALE_TARGET)):
        raise ValueError("Preserve the approved450/50 pilot or9000/1000 cumulative target")
    return dict(value)


def remaining_families(target, counts):
    target = checked_target(target)
    if set(counts) - set(target) or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Invalid qualified-family split counts")
    remaining = {split: number-counts.get(split, 0) for split, number in target.items()}
    if any(number < 0 for number in remaining.values()):
        raise ValueError(f"Qualified families exceed{target['train']}/{target['validation']}; never silently trim")
    return remaining


def ledger_snapshot(path):
    path = Path(path).resolve(strict=True)
    with sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=30) as db:
        rows = db.execute("SELECT id,model,task,prompt_sha256,status,elapsed,usage FROM calls ORDER BY id").fetchall()
    if any(row[1] != "Qwen3.8-Flash-Next" or not row[4] or row[4] == "reserved" for row in rows):
        raise ValueError("Only completed local Flash ledgers may be frozen as prior calls")
    return {"path": str(path), "calls": len(rows), "rows_sha256": digest(rows)}


def source_rotation(windows, seed=42):
    """The old source-first fair rotation, earliest retained windows first."""
    by_source = defaultdict(list)
    for window in windows:
        by_source[window["source_id"]].append(window)
    queues = {sid: deque(sorted(items, key=lambda item: item["start"])) for sid, items in by_source.items()}
    order = sorted(queues)
    random.Random(seed).shuffle(order)
    result = []
    while any(queues.values()):
        for sid in order:
            if queues[sid]:
                result.append(queues[sid].popleft())
    return result


def imported_rows(inputs):
    """Retain explicit previously qualified rows unchanged; no seed conversion."""
    from .restored_v4_qualification import require_qualified
    rows, ids, registry = [], set(), {}
    for record in inputs.get("imports", []):
        if bound(record["path"]) != {key: record[key] for key in ("path", "sha256")}:
            raise ValueError("An explicit qualified import changed")
        for sample in read_jsonl(record["path"]):
            require_qualified(sample)
            if (not isinstance(sample.get("sample_id"), str) or sample["sample_id"] in ids
                    or not isinstance(sample.get("base_question_id"), str)
                    or any(old["base_question_id"] == sample["base_question_id"] for old in rows)
                    or not reserve_fact(sample, registry)):
                raise ValueError("Qualified imports duplicate a sample or old-style fact family")
            ids.add(sample["sample_id"])
            rows.append(sample)
    check_split_isolation(rows)
    counts = Counter(row["split"] for row in rows)
    remaining_families(inputs.get("target", TARGET), counts)
    return rows


def _continued_run(directory):
    """Only a closed restored run can supply imports, spent calls and attempts."""
    directory = Path(directory).resolve(strict=True)
    plan_path, summary_path, samples_path = [directory / name for name in ("plan.json", "summary.json", "samples.jsonl")]
    plan, summary = read_json(plan_path), read_json(summary_path)
    if (plan.get("protocol") != PROTOCOL or summary.get("protocol") != PROTOCOL
            or summary.get("status") not in {"complete", "frozen_input_limit_reached"}
            or summary.get("pending_jobs") != 0 or summary.get("errors")):
        raise ValueError("Wait for the previous restored pilot to drain and close before continuation")
    inputs = plan["inputs"]
    if bound(plan["manifest"]["path"]) != plan["manifest"]:
        raise ValueError("The closed run's frozen input manifest changed")
    priors = list(inputs["prior_ledgers"])
    if any(ledger_snapshot(row["path"]) != row for row in priors):
        raise ValueError("An inherited completed prior ledger changed")
    own = ledger_snapshot(directory / "calls.sqlite3")
    prior_calls = sum(row["calls"] for row in priors)
    if (summary.get("actual_new_calls") != own["calls"] or summary.get("prior_calls") != prior_calls
            or summary.get("cumulative_calls") != prior_calls + own["calls"]):
        raise ValueError("Closed pilot summary and cumulative SQL call counts disagree")
    excluded = list(inputs.get("excluded_windows", []))
    for item in inputs["windows"]:
        window_dir = directory / "windows" / item["window_id"]
        completed = window_dir / "completed.json"
        attempts = sorted(window_dir.glob("persistent_stages/*/attempts.json"))
        if not completed.exists() and not attempts:
            continue
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("A previously attempted window changed")
        window = read_json(item["path"])
        excluded.append({"source_sha256": window["sha256"], "start": window["start"],
            "end": window["start"] + window["duration"], "window_id": window["window_id"],
            "attempt_bindings": [bound(path) for path in ([completed] if completed.exists() else []) + attempts]})
    return {"run": str(directory), "plan": bound(plan_path), "summary": bound(summary_path),
            "samples": bound(samples_path), "ledger": own, "prior_ledgers": priors,
            "excluded_windows": excluded}


def _previously_attempted(window, excluded):
    return any(window["sha256"] == row["source_sha256"]
               and max(window["start"], row["start"]) < min(window["start"] + window["duration"], row["end"])
               for row in excluded)


def prepare_inputs(source_paths, directory, prior_ledgers, *, archive=None, per_source=4, continue_from=None, target=None):
    """Freeze compatible manifests/windows, reusing existing decoded JPEGs only."""
    directory = Path(directory).resolve()
    target_counts = checked_target(target)
    if directory.exists():
        raise FileExistsError("Use a new input directory; never rewrite prepared inputs")
    if per_source is not None and (type(per_source) is not int or per_source < 1):
        raise ValueError("Use a positive per-source limit or None for all remaining windows")
    previous = _continued_run(continue_from) if continue_from is not None else None
    excluded = previous["excluded_windows"] if previous else []
    inherited_priors = previous["prior_ledgers"] + [previous["ledger"]] if previous else []
    imports = []
    if previous:
        imports = [{**previous["samples"], "closed_run": previous["run"], "closed_run_ledger": previous["ledger"]}]
    imported = imported_rows({"imports": imports, "target": target_counts})
    directory.mkdir(parents=True)
    if imports:
        target = directory / "imports" / "samples.jsonl"
        target.parent.mkdir()
        shutil.copyfile(imports[0]["path"], target)
        imports[0] = {**bound(target), "original": previous["samples"], "closed_run": previous["run"], "closed_run_ledger": previous["ledger"]}
    windows, sources, groups, raw_splits = [], [], {}, {}
    for sample in imported:
        groups[sample["source_group"]] = sample["split"]
        raw_splits[sample["source_sha256"]] = sample["split"]
    for original in map(lambda path: Path(path).resolve(strict=True), source_paths):
        source = read_json(original)
        if source["split"] not in TARGET or source.get("source_sha256", source.get("sha256")) != source["sha256"]:
            raise ValueError("Source identity/split is inconsistent")
        if groups.setdefault(source["source_group"], source["split"]) != source["split"] or raw_splits.setdefault(source["sha256"], source["split"]) != source["split"]:
            raise ValueError("Source work or raw content crosses splits")
        sid = source["stream_id"]
        target = directory / "prepared/sources" / sid / "source.json"
        write_json(target, source)
        inventory = read_json(original.parent / "windows.json")
        candidates = sorted((read_json(path) for path in inventory), key=lambda window: window["start"])
        limit = len(candidates) if per_source is None else per_source
        if not candidates:
            continue
        if len(candidates) < limit:
            raise ValueError("Insufficient existing complete64s windows for the requested pilot")
        indices = [round(index*(len(candidates)-1)/(limit-1)) for index in range(limit)] if limit > 1 else [0]
        retained, previous_end = [], None
        for index in indices:
            window = copy.deepcopy(candidates[index])
            if _previously_attempted(window, excluded):
                continue
            if (window["source_id"], window["source_group"], window["split"], window["sha256"]) != (source["source_id"], source["source_group"], source["split"], source["sha256"]):
                raise ValueError("Window/source identity changed")
            if len(window["frames"]) != 64 or len(window["clips"]) != 8 or previous_end is not None and window["start"] < previous_end:
                raise ValueError("Pilot windows must be complete64s and nonoverlapping")
            if any(not window["start"] <= frame["timestamp"] < window["start"]+64 for frame in window["frames"]):
                raise ValueError("A prepared annotation window contains noncausal pixels")
            previous_end = window["start"]+64
            window["source_manifest"] = str(target)
            path = directory / "prepared/windows" / (window["window_id"]+".json")
            write_json(path, window)
            item = {**bound(path), **{key: window[key] for key in ["window_id", "source_id", "source_group", "split", "start"]}}
            windows.append(item)
            retained.append(item)
        sources.append({"source": bound(target), "original_source": bound(original), "stream_id": sid,
            "source_id": source["source_id"], "source_group": source["source_group"], "source_sha256": source["sha256"],
            "split": source["split"], "visible_start": source.get("visible_start", source["stream_start"]),
            "windows": retained})
    if len({row["window_id"] for row in windows}) != len(windows):
        raise ValueError("Duplicate retained window IDs")
    prior_paths = {str(Path(path).resolve(strict=True)) for path in prior_ledgers}
    prior_paths.update(row["path"] for row in inherited_priors)
    priors = [ledger_snapshot(path) for path in sorted(prior_paths)]
    if any(next(row for row in priors if row["path"] == frozen["path"]) != frozen for frozen in inherited_priors):
        raise ValueError("A closed-run ledger changed while preparing continuation")
    archived = None
    if archive is not None:
        original_archive = Path(archive).resolve(strict=True)
        target = directory / "archive" / original_archive.name
        target.parent.mkdir()
        shutil.copyfile(original_archive, target)
        archived = {**bound(target), "original": bound(original_archive), "rows": len(read_jsonl(target)), "imported_as_seeds": False}
    manifest = {"protocol": PROTOCOL, "prepared": str(directory / "prepared"), "sources": sources,
        "windows": source_rotation(windows), "prior_ledgers": priors, "archive": archived,
        "per_source_windows": per_source, "seed": 42, "target": target_counts,
        "sampling_clock": "original measured source PTS; no reset/redecode", "imports": imports,
        "continued_from": previous, "excluded_windows": excluded,
        "input_limit_is_not_target_completion": True, "model_calls": 0}
    write_json(directory / "manifest.json", manifest)
    return directory / "manifest.json"


def checked_inputs(path):
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    if value.get("protocol") != PROTOCOL or not isinstance(value.get("imports"), list) or value.get("target") is None:
        raise ValueError("Expected explicit restored inputs/imports")
    checked_target(value.get("target"))
    previous = value.get("continued_from")
    if previous and _continued_run(previous["run"]) != previous:
        raise ValueError("The closed source run or its spent attempts changed")
    if previous and (len(value["imports"]) != 1 or value.get("excluded_windows") != previous["excluded_windows"]):
        raise ValueError("Continuation must preserve the exact closed-run imports and attempted-window exclusions")
    imported_rows(value)
    inherited = previous["prior_ledgers"] + [previous["ledger"]] if previous else []
    by_prior = {row["path"]: row for row in value["prior_ledgers"]}
    if any(by_prior.get(row["path"]) != row for row in inherited):
        raise ValueError("Continuation cannot omit/reset inherited spent-call ledgers")
    for record in value.get("imports", []):
        if (not previous or record.get("closed_run_ledger") != previous["ledger"]
                or record.get("original") != previous["samples"] or record.get("sha256") != previous["samples"]["sha256"]):
            raise ValueError("Imports must name the frozen closed restored run")
    for source in value["sources"]:
        for field in ["source", "original_source"]:
            if bound(source[field]["path"]) != source[field]:
                raise ValueError("Frozen source manifest changed")
    for window in value["windows"]:
        if sha256_file(window["path"]) != window["sha256"]:
            raise ValueError("Frozen64s window changed")
        if _previously_attempted(read_json(window["path"]), value.get("excluded_windows", [])):
            raise ValueError("A continued input overlaps an already attempted raw-SHA/global-time window")
    for prior in value["prior_ledgers"]:
        if ledger_snapshot(prior["path"]) != prior:
            raise ValueError("A completed prior ledger changed")
    if value["archive"] and bound(value["archive"]["path"]) != {key: value["archive"][key] for key in ["path", "sha256"]}:
        raise ValueError("Archived previous rows changed")
    return value


def checked_approval(path, *, target=None):
    value = read_json(path)
    parsed = urlparse(value.get("base_url", ""))
    tasks = {"visual_annotation", "visual_generation", "visual_verification", "visual_ablation"}
    if (value.get("provider") != "local_vllm" or value.get("model") != "Qwen3.8-Flash-Next"
            or parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.path != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment
            or value.get("enable_thinking") is not True or not tasks <= set(value.get("allowed_tasks", []))):
        raise ValueError("Only the approved loopback local Flash teacher is supported")
    target = checked_target(target)
    limit = 20000 if target == TARGET else 200000
    if target == SCALE_TARGET and value.get("dataset_target") != target:
        raise ValueError("A10k run needs a separate explicit local-request configuration")
    if (type(value.get("max_vision_requests")) is not int or not 1 <= value["max_vision_requests"] <= limit
            or type(value.get("max_concurrent_requests")) is not int or not 1 <= value["max_concurrent_requests"] <= 256
            or type(value.get("request_retries")) is not int or not 0 <= value["request_retries"] <= 5):
        raise ValueError("Preserve the stage's cumulative request,256worker,six-attempt ceilings")
    for task in tasks:
        if value.get("max_output_tokens_by_task", {}).get(task) != 32768 or value.get("reasoning_effort_by_task", {}).get(task, value.get("reasoning_effort")) != "xhigh":
            raise ValueError("Every stage must remain xhigh/32768")
    return value


def implementations():
    names = ["legacy_movie_generation", "restored_v4_qualification", "replenishment", "batch_generate", "batch_prompts", "generate",
             "stream_audit", "narrative_qualification", "narrative_temporal", "strict_generate", "multimodal", "local_teacher", "local_production", "teacher", "schema", "io"]
    paths = [Path(__file__).with_name(name+".py") for name in names]
    paths += [ROOT / "scripts" / name for name in ["repair_dataset.py", "replenish_v4.py", "generate_movies_legacy_v4.py"]]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def freeze_plan(manifest, output, approval, workers):
    inputs = checked_inputs(manifest)
    settings = checked_approval(approval, target=inputs["target"])
    if type(workers) is not int or not 1 <= workers <= settings["max_concurrent_requests"]:
        raise ValueError("Workers exceed the approved limit")
    output = Path(output).resolve()
    if any(Path(row["path"]) == output / "calls.sqlite3" for row in inputs["prior_ledgers"]):
        raise ValueError("The new ledger cannot also be a prior ledger")
    retained = imported_rows(inputs)
    plan = {"protocol": PROTOCOL, "manifest": bound(manifest), "inputs": inputs, "approval": bound(approval),
        "settings": settings, "workers": workers, "target": inputs["target"], "implementation_sha256": implementations(),
        "prior_calls": sum(row["calls"] for row in inputs["prior_ledgers"]), "seed_families": len(retained),
        "annotation_windows": len(inputs["windows"]), "no_temporal_or_alternative_group_extensions": True}
    if plan["prior_calls"] >= settings["max_vision_requests"]:
        raise ValueError("Completed prior calls exhaust the cumulative budget")
    path = output / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Frozen restored-run inputs/settings/code changed")
    else:
        if any(path.name != ".production.lock" for path in output.iterdir()):
            raise ValueError("Choose a new empty restored experiment output")
        write_json(path, plan)
    return plan


def adapt_candidate(sample, window, source, annotation):
    """Preserve all generated labels/text/evidence while fixing input metadata."""
    result = copy.deepcopy(sample)
    result["visible_start"] = source.get("visible_start", source["stream_start"])
    paths = {path for clip in result["candidates"]+[result["current_clip"]] for path in clip["frames"]}
    paths.update(concept["image"] for concept in result["concepts"])
    result["frame_timestamps"] = {path: source["frame_timestamps"][path] for path in sorted(paths)}
    result["benchmark_overlap"] = copy.deepcopy(source.get("benchmark_overlap", {"status":"unknown"}))
    result["benchmark_source_policy"] = "overlap_allowed_by_user"
    result.pop("source_overlap_with_original_benchmark", None)
    if result["benchmark_overlap"]["status"] != "unknown":
        result["source_overlap_with_original_benchmark"] = result["benchmark_overlap"]["status"] == "known_shared"
    result["restored_movie_adapter"] = {"protocol": PROTOCOL, "source_manifest": window["source_manifest"],
        "stream_id": source["stream_id"], "source_clock": source["source_clock"], "generated_labels_unchanged": True}
    validate_sample(result)
    return result


def api_deferred(error):
    # Match the successful old pilot: an exhausted bounded request/schema stage
    # is an unlabelled defer, while budget and ordinary code failures propagate.
    return isinstance(error, (TeacherRequestError, TeacherRetryExhausted))


def deferred(error):
    return {"type":type(error).__name__, "error":str(error), "attempts":getattr(error,"attempts",None),
            "last_failure":getattr(error,"last_failure",getattr(error,"details",{})), "label_assigned":False,
            "retry_budget_reset":False}


def run_window(item, output, teacher):
    if sha256_file(item["path"]) != item["sha256"]:
        raise ValueError("Frozen window changed")
    window = read_json(item["path"])
    source = read_json(window["source_manifest"])
    directory = Path(output) / "windows" / window["window_id"]
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "completed.json"
    binding = digest({"window":item,"source":bound(window["source_manifest"]),"protocol":PROTOCOL})
    if completed.exists():
        result = read_json(completed)
        if result["binding_sha256"] != binding:
            raise ValueError("Cached window belongs to different frozen inputs")
        return result

    def validate_annotation(response):
        annotation_format(response)
        if response["usable"]:
            ids = set()
            for concept in response["concepts"]:
                if (not isinstance(concept, dict) or not isinstance(concept.get("id"), str) or concept["id"] in ids
                        or type(concept.get("reference_frame")) is not int or not 0 <= concept["reference_frame"] < 64
                        or not isinstance(concept.get("description"), str)):
                    raise ValueError("Invalid local entity reference schema")
                ids.add(concept["id"])

    try:
        annotation, call = persistent_stage_call(teacher, directory, "annotation", "visual_annotation",
            [{"type":"text","text":ANNOTATE+"\nRegister visible people or physical objects; never register actions/motions as concepts. A person's visible actions remain allowed events/Behavior."}]+video_content(window),
            validate_annotation, binding_context={"protocol":PROTOCOL,"window_sha256":item["sha256"]})
        if not annotation["usable"] or not annotation.get("events"):
            result = {"kind":"window","key":window["window_id"],"status":"unusable","candidates":[]}
        else:
            annotation_path = directory / "annotation.json"
            write_json(annotation_path, {"response":annotation,"call":call})
            provenance = {**bound(annotation_path),"call":call}
            prompt = NEW_QUESTIONS.replace("source time zero", f"source time {source.get('visible_start',source['stream_start']):.6f}")
            content = [{"type":"text","text":prompt+"\nAlready retained fact keys:\n[]\nFixed annotation (relative times):\n"+json.dumps(annotation,ensure_ascii=False)}]

            def validate_generation(response):
                generation_format(response)
                make_candidates(window, annotation, response, {"request_id":0}, provenance)

            generation, generation_call = persistent_stage_call(teacher,directory,"new_questions","visual_generation",content,
                validate_generation,binding_context={"protocol":PROTOCOL,"window_sha256":item["sha256"]})
            proposed, invalid, corrections = make_candidates(window,annotation,generation,generation_call,provenance)
            candidates = [adapt_candidate(sample,window,source,annotation) for sample in proposed]
            for sample in candidates:
                sample["sample_id"] = Path(output).name+":"+sample["sample_id"]
            result = {"kind":"window","key":window["window_id"],"status":"generated","candidates":candidates,
                      "invalid_proposals":invalid,"metadata_normalizations":corrections}
    except (TeacherRequestError,TeacherRetryExhausted) as error:
        if not api_deferred(error):
            raise
        result = {"kind":"window","key":window["window_id"],"status":"api_deferred","candidates":[],"failure":deferred(error)}
    result["binding_sha256"] = binding
    write_json(completed,result)
    return result


def run_question(sample, output, teacher):
    from .restored_v4_qualification import qualify_candidate, require_qualified
    key = sample["sample_id"]
    directory = Path(output) / "questions" / hashlib.sha256(key.encode()).hexdigest()[:24]
    directory.mkdir(parents=True,exist_ok=True)
    completed = directory / "completed.json"
    if completed.exists():
        result = read_json(completed)
        if result["candidate_sha256"] != digest(sample):
            raise ValueError("Cached question belongs to another proposal")
        if result["accepted"] is not None:
            require_qualified(result["accepted"])
        return result
    try:
        accepted,reasons = qualify_candidate(teacher,directory,sample)
        if accepted is not None:
            require_qualified(accepted)
        result = {"kind":"question","key":key,"status":"accepted" if accepted else "rejected","accepted":accepted,"reasons":reasons}
    except (TeacherRequestError,TeacherRetryExhausted) as error:
        if not api_deferred(error):
            raise
        result = {"kind":"question","key":key,"status":"api_deferred","accepted":None,"reasons":[],"failure":deferred(error)}
    result["candidate_sha256"] = digest(sample)
    write_json(completed,result)
    return result


def create_teacher(approval,output,plan,stop):
    class BudgetedTeacher(LocalTeacher):
        def __init__(self):
            super().__init__(approval,Path(output)/"calls.sqlite3")
            self.budget_lock = threading.Lock()
        def reserve(self,task,content):
            with self.budget_lock:
                if stop.is_set():
                    raise TeacherBudgetExhausted("Restored production dispatch is stopped")
                for prior in plan["inputs"]["prior_ledgers"]:
                    if ledger_snapshot(prior["path"]) != prior:
                        stop.set()
                        raise TeacherBudgetExhausted("A frozen completed prior ledger changed")
                with sqlite3.connect(self.ledger,timeout=30) as db:
                    own = db.execute("SELECT count(*) FROM calls").fetchone()[0]
                if plan["prior_calls"]+own >= self.approval["max_vision_requests"]:
                    stop.set()
                    raise TeacherBudgetExhausted(f"Cumulative{self.approval['max_vision_requests']} request budget exhausted")
                return super().reserve(task,content)
    return BudgetedTeacher()


def run(manifest,output,approval,*,workers=256,plan_only=False):
    from .restored_v4_qualification import require_qualified
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/".production.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = freeze_plan(manifest,output,approval,workers)
        if plan_only:
            return {"plan_only":True,"windows":plan["annotation_windows"],"workers":workers,"target":plan["target"],
                "prior_calls":plan["prior_calls"],"remaining_calls":plan["settings"]["max_vision_requests"]-plan["prior_calls"],
                "max_proposals_from_frozen_inputs":6*plan["annotation_windows"],"seed_families":plan["seed_families"]}
        retained = imported_rows(plan["inputs"])
        rows = {sample["base_question_id"]: sample for sample in retained}
        registry, seen, finished, errors = {}, set(rows), {sample["sample_id"] for sample in retained}, []
        for sample in retained:
            if not reserve_fact(sample, registry):
                raise ValueError("Imported old-style facts collide before dispatch")
        for path in sorted((output/"questions").glob("*/completed.json")):
            result = read_json(path)
            finished.add(result["key"])
            if result["accepted"] is not None:
                sample = result["accepted"]
                require_qualified(sample)
                if sample["base_question_id"] in rows or not reserve_fact(sample,registry):
                    raise ValueError("Duplicate accepted old-style source/window/event/attribute/task family")
                rows[sample["base_question_id"]] = sample
                seen.add(sample["base_question_id"])
        stop = threading.Event()
        teacher = create_teacher(approval,output,plan,stop)
        queue = deque(("window",item) for item in plan["inputs"]["windows"])
        pending, statuses = {}, Counter()
        exported_families = -1
        # Qualified per-question receipts are already durable before each future
        # returns. Avoid repeatedly rewriting multi-GB exports while waiting.
        snapshot_every = 50 if plan["target"] == SCALE_TARGET else 1

        def export(*, force=False):
            nonlocal exported_families
            samples = sorted(rows.values(),key=lambda row:row["base_question_id"])
            check_split_isolation(samples)
            counts = Counter(row["split"] for row in samples)
            remaining = remaining_families(plan["target"], counts)
            if force or exported_families < 0 or len(samples)-exported_families >= snapshot_every:
                write_jsonl(output/"samples.jsonl",samples)
                exported_families = len(samples)
            with sqlite3.connect(teacher.ledger) as db:
                own = db.execute("SELECT count(*) FROM calls").fetchone()[0]
            summary = {"protocol":PROTOCOL,"families":len(samples),"family_splits":dict(counts),"remaining":remaining,
                "target_met":not any(remaining.values()),"job_statuses":dict(statuses),"pending_jobs":len(pending),
                "errors":errors,"prior_calls":plan["prior_calls"],"actual_new_calls":own,"cumulative_calls":own+plan["prior_calls"],
                "limited_pilot_windows":plan["annotation_windows"],"old_archived_rows_imported":0,
                "retained_qualified_imports":len(retained),"new_qualified_families":len(samples)-len(retained),
                "exported_families":exported_families,"snapshot_every_groups":snapshot_every}
            write_json(output/"summary.json",summary)
            return summary

        with ThreadPoolExecutor(max_workers=workers) as pool:
            summary = export()
            while (queue or pending) and not stop.is_set():
                held = deque()
                while queue and len(pending)<workers:
                    kind,payload = queue.popleft()
                    if not can_dispatch(kind,payload["split"],summary["remaining"],pending):
                        held.append((kind,payload));continue
                    fn = run_window if kind=="window" else run_question
                    pending[pool.submit(fn,payload,output,teacher)] = (kind,payload)
                queue.extendleft(reversed(held))
                if not pending:
                    break
                done,_ = wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    kind,payload = pending.pop(future)
                    try:
                        result = future.result()
                        statuses[kind+":"+result["status"]] += 1
                        if kind=="window":
                            for sample in reversed(result.get("candidates",[])):
                                fid = sample["base_question_id"]
                                if fid not in seen and reserve_fact(sample,registry):
                                    seen.add(fid)
                                    if sample["sample_id"] not in finished:
                                        queue.appendleft(("question",sample))
                        elif result["accepted"] is not None:
                            sample = result["accepted"]
                            require_qualified(sample)
                            if sample["base_question_id"] in rows:
                                raise ValueError("Duplicate accepted family")
                            rows[sample["base_question_id"]] = sample
                    except Exception as error:
                        errors.append({"kind":kind,"type":type(error).__name__,"error":str(error)})
                        stop.set()
                summary = export()
                if summary["target_met"]:
                    stop.set()
            for future,(kind,payload) in list(pending.items()):
                try:
                    result = future.result()
                    statuses[kind+":"+result["status"]] += 1
                    if kind=="question" and result["accepted"] is not None:
                        sample = result["accepted"]
                        require_qualified(sample)
                        if sample["base_question_id"] in rows:
                            raise ValueError("Duplicate accepted family while draining")
                        rows[sample["base_question_id"]] = sample
                except Exception as error:
                    errors.append({"kind":kind,"type":type(error).__name__,"error":str(error)})
                finally:
                    pending.pop(future, None)
        summary = export(force=True)
        summary["status"] = "halted_error" if errors else "complete" if summary["target_met"] else "frozen_input_limit_reached"
        write_json(output/"summary.json",summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--approval",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=256)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only",action="store_true")
    mode.add_argument("--run",action="store_true")
    args=parser.parse_args()
    result=run(args.manifest,args.output,args.approval,workers=args.workers,plan_only=args.plan_only)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result.get("errors"):
        raise SystemExit(2)
