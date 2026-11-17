#!/usr/bin/env python3
"""Evaluate one saved query adapter on a fixed validation set and frozen corpus."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

from pearl_training.io import code_revision, read_json, read_jsonl, sha256_file, write_json
from pearl_training.multimodal import document_messages, embedding_model_class, pool_embedding, process_messages
from pearl_training.patch_embed import enable_torch29_patch_embed_workaround
from pearl_training.schema import check_split_isolation
from pearl_training.stream_audit import require_repaired_sample
from pearl_training.train_retrieval import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["samples", "model", "adapter", "document-cache", "output"]:
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("A complete retrieval evaluation already exists")
    rows = read_jsonl(args.samples)
    for row in rows:
        require_repaired_sample(row)
    check_split_isolation(rows)
    validation = [s for s in rows if s["split"] == "validation" and s["task"] == "past"]
    if not validation:
        raise ValueError("No fixed historical validation questions")
    cache_path = Path(args.document_cache)
    manifest = read_json(cache_path.parent / "manifest.json")
    if (manifest["samples_sha256"] != sha256_file(args.samples)
            or manifest["cache_sha256"] != sha256_file(cache_path)
            or manifest["identity"]["model"] != str(Path(args.model).resolve())
            or manifest["identity"]["config_sha256"] != sha256_file(Path(args.model) / "config.json")):
        raise ValueError("Frozen evaluation corpus/checkpoint identity changed")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    clips = {c["id"]: c for s in validation for c in s["candidates"]}
    if not set(clips) <= set(cache) or any(not torch.isfinite(cache[cid]).all() for cid in clips):
        raise ValueError("Evaluation document cache is incomplete")
    base = embedding_model_class().from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa", local_files_only=True)
    enable_torch29_patch_embed_workaround(base)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to("cuda:0").eval()
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(args.model, padding_side="right", local_files_only=True)
    deviations = []
    with model.disable_adapter(), torch.inference_mode():
        for cid in list(clips)[:3]:
            vector = pool_embedding(model, process_messages(processor, document_messages(clips[cid]),
                                                             args.max_length).to("cuda:0"))[0].cpu()
            deviations.append((vector - cache[cid]).abs().max().item())
        if max(deviations) > 1e-6:
            raise ValueError(f"Frozen document encoder differs from evaluation cache: {max(deviations)}")
        before = evaluate(model, processor, validation, cache, "cuda:0", args.max_length)
    after = evaluate(model, processor, validation, cache, "cuda:0", args.max_length)
    write_json(output / "validation_before.json", before)
    write_json(output / "validation_after.json", after)
    summary = {"questions": len(validation), "before_recall@1": before["recall_any@1"],
               "after_recall@1": after["recall_any@1"], "before_recall@4": before["recall_any@4"],
               "after_recall@4": after["recall_any@4"], "max_document_deviation": max(deviations)}
    write_json(output / "run_config.json", {**vars(args), "code": code_revision(),
        "samples_sha256": sha256_file(args.samples), "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
        "hardware": torch.cuda.get_device_name(0), "query": "reference_images_plus_original_question"})
    write_json(output / "summary.json", summary)
    print("COMPLETE", summary, flush=True)


if __name__ == "__main__":
    main()
