"""Distribute questions without changing their inputs or per-question scoring."""

import os
from pathlib import Path
import subprocess
import time

from .io import read_json, write_json


def ordered_records(samples, records):
    expected = [sample["sample_id"] for sample in samples]
    indexed = {record["sample_id"]: record for record in records}
    if len(indexed) != len(records) or len(set(expected)) != len(expected) or set(indexed) != set(expected):
        raise ValueError("Parallel evaluation omitted, duplicated or substituted questions")
    return [indexed[key] for key in expected]


def gather_records(accelerator, samples, records):
    if accelerator.num_processes > 1:
        import torch.distributed as distributed
        parts = [None] * accelerator.num_processes
        distributed.all_gather_object(parts, records)
        records = [record for part in parts for record in part]
    return ordered_records(samples, records)


def joint_summary(records):
    if not records:
        raise ValueError("Cannot score an empty joint evaluation")
    families = {}
    for record in records:
        families.setdefault(record["base_question_id"], []).append(record["correct"])
    return {"questions": len(records), "strict_correct": sum(r["correct"] for r in records),
            "accuracy": sum(r["correct"] for r in records) / len(records),
            "question_families": len(families), "family_strict_correct": sum(all(v) for v in families.values()),
            "by_task": {task: {"questions": sum(r["task"] == task for r in records),
                "strict_correct": sum(r["correct"] for r in records if r["task"] == task)}
                for task in ["current", "past"]}}


def run_gpu_commands(jobs, log_dir):
    """Children inherit this process group, so the owning coordinator can stop all."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    try:
        for label, command, gpu in jobs:
            log = (log_dir / (label + ".log")).open("a")
            logs.append(log)
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                     stdout=log, stderr=subprocess.STDOUT)
            children.append((label, child))
        pending = dict(children)
        while pending:
            for label, child in list(pending.items()):
                code = child.poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"{label} exited {code}; inspect {log_dir / (label + '.log')}")
                print("COMPLETED", label, flush=True)
                del pending[label]
            if pending:
                time.sleep(1)
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


def merge_joint_shards(output, samples, count):
    output = Path(output)
    configs = [read_json(output / f"shard{i}" / "run_config.json") for i in range(count)]
    fields = ["samples_sha256", "adapter_sha256", "reader_input_protocol", "max_length", "top_k", "neighbors",
              "model", "context_truncation", "rankings_before", "rankings_after", "num_shards",
              "hostname", "device_name", "torch_version", "cuda_version"]
    for i, config in enumerate(configs):
        if config["shard_index"] != i or config["num_shards"] != count:
            raise ValueError("Joint shard identity differs")
        if any(config[field] != configs[0][field] for field in fields):
            raise ValueError("Joint shards used different inputs, checkpoints or policies")
    summaries = {}
    for reader in ["frozen_reader", "trained_reader"]:
        for retrieval in ["frozen_retriever", "trained_retriever"]:
            filename = reader + "_" + retrieval + ".json"
            combined = []
            for i in range(count):
                data = read_json(output / f"shard{i}" / filename)
                if not data["complete"]:
                    raise ValueError("Incomplete joint evaluation shard")
                combined.extend(ordered_records(samples[i::count], data["records"]))
            records = ordered_records(samples, combined)
            write_json(output / filename, {"records": records, "complete": True})
            summaries[reader + "+" + retrieval] = joint_summary(records)
    config = {**configs[0], "output": str(output), "shard_index": None, "evaluation_parallelism": count,
              "evaluation_gpus": [{"shard_index": c["shard_index"], "visible_gpu_ids": c["visible_gpu_ids"],
                                   "device_name": c["device_name"]} for c in configs]}
    write_json(output / "run_config.json", config)
    write_json(output / "summary.json", summaries)
    return summaries
