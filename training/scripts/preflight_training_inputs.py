#!/usr/bin/env python3
"""Check every training/validation option rotation on CPU without truncation."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
from pathlib import Path

from pearl_training.io import read_json, read_jsonl, sha256_file, write_json


PROCESSOR = None
DATA = None


def initialize(model, sources):
    global PROCESSOR, DATA
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    PROCESSOR = AutoProcessor.from_pretrained(model, local_files_only=True)
    DATA = {teacher: read_jsonl(path) for teacher, path in sources.items()}


def check(job):
    from pearl_training.multimodal import answer_text, process_messages, sft_messages
    teacher, index = job
    sample = DATA[teacher][index]
    lengths = []
    for rotation in range(4):
        batch = process_messages(PROCESSOR, sft_messages(sample, rotation), 8192,
                                 assistant=answer_text(sample, rotation))
        lengths.append(int(batch["input_ids"].shape[1]))
    return teacher, index, sample["sample_id"], lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sources = read_json(args.sources)
    counts = {teacher: len(read_jsonl(path)) for teacher, path in sources.items()}
    jobs = [(teacher, index) for teacher, count in counts.items() for index in range(count)]
    iterator = iter(jobs)
    completed, lengths, seen = Counter(), {t: [] for t in sources}, set()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=initialize, initargs=(args.model, sources)) as pool:
        pending = {pool.submit(check, job) for _, job in zip(range(args.workers * 2), iterator)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                teacher, index, sample_id, values = future.result()
                if (teacher, index) in seen or len(values) != 4 or not all(0 < v <= 8192 for v in values):
                    raise ValueError("Invalid, duplicated or over-budget preflight input")
                seen.add((teacher, index))
                completed[teacher] += 1
                lengths[teacher].extend(values)
                if sum(completed.values()) % 10 == 0:
                    progress = {"completed": dict(completed), "expected": counts, "cpu_workers": args.workers}
                    write_json(args.output.with_suffix(".progress.json"), progress)
                    print("PREFLIGHT", progress, flush=True)
                job = next(iterator, None)
                if job is not None:
                    pending.add(pool.submit(check, job))
    if dict(completed) != counts or len(seen) != len(jobs):
        raise ValueError("Preflight omitted dataset rows")
    report = {teacher: {"samples_sha256": sha256_file(path), "rotations_checked": len(lengths[teacher]),
        "min_tokens": min(lengths[teacher]), "max_tokens": max(lengths[teacher]),
        "truncation": False, "cpu_workers": args.workers} for teacher, path in sources.items()}
    write_json(args.output, report)
    print("COMPLETE", report, flush=True)


if __name__ == "__main__":
    main()
