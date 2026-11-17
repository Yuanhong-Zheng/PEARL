from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .io import read_jsonl, write_json
from .evaluation_parallel import gather_records
from .multimodal import answer_text, process_messages, sft_messages
from .schema import check_split_isolation, validate_sample
from .scoring import extract_answer
from .stream_audit import require_repaired_sample
from .train_common import add_arguments, initialize, train_loop
from .raw_token_backends import generate_hf


def evaluate(model, processor, samples, device, max_length):
    model.eval()
    records = []
    for sample in samples:
        rotations = []
        for rotation in range(4):
            messages = sft_messages(sample, rotation)
            inputs = process_messages(processor, messages, max_length=max_length).to(device)
            with torch.inference_mode():
                output = generate_hf(model, processor, inputs, messages=messages,
                    metadata={"sample_id": sample["sample_id"], "rotation": rotation,
                              "scope": "held_out_reader_validation"},
                    max_new_tokens=1024, do_sample=False, use_cache=True)
            text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            predicted = extract_answer(text).strip().upper()
            expected = "ABCD"[(sample["answer_index"] - rotation) % 4]
            rotations.append({"rotation": rotation, "response": text, "expected": expected,
                              "correct": predicted == expected, "predicted": predicted,
                              "input_tokens": inputs["input_ids"].shape[1]})
        records.append({"sample_id": sample["sample_id"], "base_question_id": sample["base_question_id"],
                        "source_group": sample["source_group"], "task": sample["task"], "correct": all(row["correct"] for row in rotations), "rotations": rotations})
    return score_records(records)


def score_records(records):
    families = {}
    for row in records:
        families.setdefault(row["base_question_id"], []).append(row["correct"])
    return {"questions": len(records), "strict_correct": sum(row["correct"] for row in records), "records": records,
            "question_families": len(families), "family_strict_correct": sum(all(v) for v in families.values()),
            "context": "teacher_verified_evidence_plus_at_most_two_verified_negatives"}


def evaluate_distributed(accelerator, model, processor, samples, max_length):
    local = samples[accelerator.process_index::accelerator.num_processes]
    result = evaluate(model, processor, local, accelerator.device, max_length)
    return score_records(gather_records(accelerator, samples, result["records"]))


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    samples = read_jsonl(args.samples)
    for sample in samples:
        validate_sample(sample)
        require_repaired_sample(sample)
    check_split_isolation(samples)
    training = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    if not validation:
        raise ValueError("Need a held-out validation split before training")
    accelerator, model, optimizer = initialize(args, Qwen3VLForConditionalGeneration)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    unwrapped = accelerator.unwrap_model(model)
    with unwrapped.disable_adapter():
        baseline = evaluate_distributed(accelerator, unwrapped, processor, validation, args.max_length)
    if accelerator.is_main_process:
        write_json(Path(args.output) / "validation_before.json", baseline)
        print("VALIDATION_BEFORE", baseline["strict_correct"], "/", baseline["questions"], flush=True)
    accelerator.wait_for_everyone()

    def loss_function(wrapped, sample, step):
        rotation = sample["training_rotation"]
        inputs = process_messages(processor, sft_messages(sample, rotation), args.max_length, assistant=answer_text(sample, rotation)).to(accelerator.device)
        return wrapped(**inputs).loss, {"input_tokens": inputs["input_ids"].shape[1]}

    augmented = [{**sample, "training_rotation": rotation} for sample in training for rotation in range(4)]
    trace = train_loop(args, accelerator, model, optimizer, augmented, loss_function)
    after = evaluate_distributed(accelerator, unwrapped, processor, validation, args.max_length)
    if accelerator.is_main_process:
        write_json(Path(args.output) / "validation_after.json", after)
        processor.save_pretrained(args.output)
        write_json(Path(args.output) / "summary.json", {"training_questions": len(training), "validation_questions": len(validation),
                   "before_strict_correct": baseline["strict_correct"], "after_strict_correct": after["strict_correct"],
                   "optimizer_updates": len(trace), "pilot_only": len(training) < 10,
                   "source_group_disjoint_new_validation": True})
        print("VALIDATION_AFTER", after["strict_correct"], "/", after["questions"], flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
