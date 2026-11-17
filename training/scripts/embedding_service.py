#!/usr/bin/env python3
"""Local frozen embedding service and content-addressed document cache."""

import argparse
import hashlib
import json
import threading
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import AutoProcessor

from pearl_training.io import sha256_file
from pearl_training.multimodal import embedding_model_class, pool_embedding, process_messages
from pearl_training.patch_embed import enable_torch29_patch_embed_workaround


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--port", type=int, default=28780)
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    model = embedding_model_class().from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    enable_torch29_patch_embed_workaround(model)
    model.to("cuda:0").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, padding_side="right")
    identity = {"model": str(Path(args.model).resolve()), "config_sha256": sha256_file(Path(args.model) / "config.json"),
                "pooling": "normalized_last_token", "dtype": "bf16", "image_max_pixels": 256 * 32 * 32}
    lock = threading.Lock()
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(identity)

    @app.post("/encode")
    def encode():
        body = request.get_json()
        messages = body["messages"]
        key = hashlib.sha256(json.dumps({"identity": identity, "messages": messages}, sort_keys=True).encode()).hexdigest()
        path = cache / (key + ".pt")
        with lock:
            if path.exists():
                vector = torch.load(path, map_location="cpu", weights_only=True)
            else:
                inputs = process_messages(processor, messages).to("cuda:0")
                with torch.inference_mode():
                    vector = pool_embedding(model, inputs)[0].cpu()
                temporary = path.with_suffix(".tmp")
                torch.save(vector, temporary)
                temporary.replace(path)
        return jsonify({"vector": vector.tolist(), "cache_key": key, "identity": identity})

    print("READY", json.dumps(identity), flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
