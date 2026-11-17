from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor

from .io import read_json, read_jsonl, sha256_file, write_json
from .evaluation_parallel import gather_records
from .multimodal import document_messages, embedding_model_class, pool_embedding, process_messages, query_messages
from .narrative_training_labels import complete_support_hit, retrieval_positive_ids
from .narrative_qualification_v2 import require_narrative_qualified
from .schema import check_split_isolation, validate_sample
from .stream_audit import require_repaired_sample
from .text_query_cache import load_text_query_cache
from .train_common import add_arguments, initialize, train_loop
from .raw_token_backends import forward_hf


def multi_positive_loss(scores, positive_mask):
    if not positive_mask.any() or positive_mask.all():
        raise ValueError("Retrieval loss needs positives and verified negatives")
    # Train coverage of every positive, rather than rewarding only the easiest one.
    return -torch.log_softmax(scores.float(), dim=-1)[positive_mask].mean()


def query_content(sample, query_cache=None):
    return query_messages(sample) if query_cache is None else query_cache.messages(sample)


def evaluate(model, processor, samples, cache, device, max_length, query_cache=None):
    model.eval()
    records = []
    for sample in samples:
        messages = query_content(sample, query_cache)
        inputs = process_messages(processor, messages, max_length).to(device)
        with torch.inference_mode():
            query = forward_hf(inputs, processor, lambda: pool_embedding(model, inputs), messages=messages,
                metadata={"sample_id": sample["sample_id"], "scope": "held_out_retrieval_validation"})[0]
            candidates = torch.stack([cache[clip["id"]] for clip in sample["candidates"]]).to(device)
            order = (candidates @ query).argsort(descending=True).tolist()
        ids = [sample["candidates"][i]["id"] for i in order]
        positives = set(retrieval_positive_ids(sample))
        row = {"sample_id": sample["sample_id"], "ranked_clip_ids": ids, "candidate_count": len(ids)}
        for k in [1, 2, 4]:
            row[f"recall_any@{k}"] = bool(positives & set(ids[:k]))
            row[f"evidence_coverage@{k}"] = len(positives & set(ids[:k])) / len(positives)
            row[f"complete_support_set@{k}"] = complete_support_hit(sample, ids, k)
        records.append(row)
    return score_records(records, query_cache=query_cache)


def score_records(records, query_cache=None):
    result = {"questions": len(records), "records": records,
            "query": "reference_images_plus_original_question", "pool": "all supplied completed-history candidates",
            **{f"recall_any@{k}": sum(row[f"recall_any@{k}"] for row in records) / len(records) if records else None
               for k in [1, 2, 4]}}
    for k in [1, 2, 4]:
        known = [row[f"complete_support_set@{k}"] for row in records if row.get(f"complete_support_set@{k}") is not None]
        result[f"complete_support_set_questions@{k}"] = len(known)
        result[f"complete_support_set@{k}"] = sum(known) / len(known) if known else None
    if query_cache is not None:
        result["query"] = query_cache.metadata["query_mode"]
        result["query_cache"] = dict(query_cache.metadata)
    return result


def evaluate_distributed(accelerator, model, processor, samples, cache, max_length, query_cache=None):
    local = samples[accelerator.process_index::accelerator.num_processes]
    result = evaluate(model, processor, local, cache, accelerator.device, max_length, query_cache=query_cache)
    return score_records(gather_records(accelerator, samples, result["records"]), query_cache=query_cache)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--document-cache", help="Frozen document vectors exported by freeze_training_data.py")
    parser.add_argument("--query-cache", help="Completed base-helper release-text cache, bound to this samples file")
    args = parser.parse_args()
    all_samples = read_jsonl(args.samples)
    for sample in all_samples:
        validate_sample(sample)
        require_repaired_sample(sample)
        if sample.get("narrative_qualification"):
            require_narrative_qualified(sample)
    check_split_isolation(all_samples)
    # CPU-only validation precedes initialization/model loading on every rank.
    # This one immutable text map is shared by loss and before/after validation.
    query_cache = load_text_query_cache(args.query_cache, args.samples, samples=all_samples) if args.query_cache else None
    samples = [sample for sample in all_samples if sample["task"] == "past"]
    training = [sample for sample in samples if sample["split"] == "train" and sample["negative_clip_ids"]]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    if not validation:
        raise ValueError("Need verified past questions in the validation split")
    accelerator, model, optimizer = initialize(args, embedding_model_class())
    processor = AutoProcessor.from_pretrained(args.model, padding_side="right", local_files_only=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()
    cache = {}
    clips = {clip["id"]: clip for sample in samples for clip in sample["candidates"]}
    if args.document_cache:
        cache_path = Path(args.document_cache)
        manifest = read_json(cache_path.parent / "manifest.json")
        identity = manifest["identity"]
        if identity["model"] != str(Path(args.model).resolve()) or identity["config_sha256"] != sha256_file(Path(args.model) / "config.json"):
            raise ValueError("Document cache was built with a different base checkpoint")
        if manifest["samples_sha256"] != sha256_file(args.samples) or manifest["cache_sha256"] != sha256_file(cache_path):
            raise ValueError("Document cache or frozen dataset hash changed")
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if not set(clips) <= set(cache) or any(not torch.isfinite(cache[cid]).all() for cid in clips):
            raise ValueError("Document cache is incomplete or non-finite")
        # The service and disabled-adapter training encoder must really agree.
        with unwrapped.disable_adapter(), torch.inference_mode():
            for cid in list(clips)[:3]:
                inputs = process_messages(processor, document_messages(clips[cid]), args.max_length).to(accelerator.device)
                vector = pool_embedding(unwrapped, inputs)[0].cpu()
                deviation = (vector - cache[cid]).abs().max().item()
                if deviation > 1e-6:
                    raise ValueError(f"Cached and training document encoders differ: {deviation}")
    else:
        # Small pilot fallback, with the same disabled-adapter document encoder.
        with unwrapped.disable_adapter(), torch.inference_mode():
            for clip_id, clip in clips.items():
                inputs = process_messages(processor, document_messages(clip), args.max_length).to(accelerator.device)
                cache[clip_id] = pool_embedding(unwrapped, inputs)[0].cpu()
    with unwrapped.disable_adapter():
        before = evaluate_distributed(accelerator, unwrapped, processor, validation, cache, args.max_length, query_cache=query_cache)
    if accelerator.is_main_process:
        torch.save(cache, Path(args.output) / "frozen_document_embeddings.pt")
        write_json(Path(args.output) / "validation_before.json", before)
        print("VALIDATION_BEFORE", before["recall_any@1"], flush=True)
    accelerator.wait_for_everyone()

    def loss_function(wrapped, sample, step):
        inputs = process_messages(processor, query_content(sample, query_cache), args.max_length).to(accelerator.device)
        query = pool_embedding(wrapped, inputs)[0]
        positives = retrieval_positive_ids(sample)
        ids = positives + sample["negative_clip_ids"]
        documents = torch.stack([cache[clip_id] for clip_id in ids]).to(accelerator.device)
        scores = documents @ query / args.temperature
        mask = torch.tensor([clip_id in positives for clip_id in ids], dtype=torch.bool, device=accelerator.device)
        return multi_positive_loss(scores, mask), {"input_tokens": inputs["input_ids"].shape[1], "labelled_candidates": len(ids)}

    trace = train_loop(args, accelerator, model, optimizer, training, loss_function)
    unwrapped.eval()
    with unwrapped.disable_adapter(), torch.inference_mode():
        first_id, first_clip = next(iter(clips.items()))
        inputs = process_messages(processor, document_messages(first_clip), args.max_length).to(accelerator.device)
        recomputed = pool_embedding(unwrapped, inputs)[0].cpu()
        deviation = (recomputed - cache[first_id]).abs().max().item()
    if deviation > 1e-6:
        raise ValueError(f"Frozen document encoder changed: {deviation}")
    after = evaluate_distributed(accelerator, unwrapped, processor, validation, cache, args.max_length, query_cache=query_cache)
    if accelerator.is_main_process:
        processor.save_pretrained(args.output)
        write_json(Path(args.output) / "validation_after.json", after)
        write_json(Path(args.output) / "summary.json", {"training_questions": len(training), "validation_questions": len(validation),
                   "optimizer_updates": len(trace), "before_recall@1": before["recall_any@1"], "after_recall@1": after["recall_any@1"],
                   "frozen_document_embedding_max_abs_change": deviation, "pilot_only": len(training) < 10,
                   "source_group_disjoint_new_validation": True,
                   "query": before["query"], "query_cache": dict(query_cache.metadata) if query_cache else None,
                   "query_alignment_scope": "release_text_only" if query_cache else "legacy_refs_plus_question",
                   "training_model_dtype": "bfloat16", "training_normalization_dtype": "float32",
                   "document_frame_counts": sorted({len(clip["frames"]) for clip in clips.values()}),
                   "document_adapter_disabled": True,
                   "training_input_budget_policy": "reject_over_budget_without_truncation"})
        print("VALIDATION_AFTER", after["recall_any@1"], flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
