from __future__ import annotations

import json
import random
import socket
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model

from .io import code_revision, sha256_file, write_json
from .multimodal import READER_INPUT_PROTOCOL, lora_targets
from .patch_embed import enable_torch29_patch_embed_workaround
from .checkpoints import save_periodic_weights


def add_arguments(parser):
    parser.add_argument("--samples", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-updates", type=int, default=10)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1000)


def initialize(args, model_class):
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.accumulation)
    if accelerator.num_processes != args.expected_gpus:
        raise ValueError(f"Expected {args.expected_gpus} GPU processes, got {accelerator.num_processes}")
    if accelerator.device.type != "cuda":
        raise ValueError("Training requires allocated CUDA devices")
    set_seed(args.seed)
    model = model_class.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    patch_count = enable_torch29_patch_embed_workaround(model)
    targets = lora_targets(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, target_modules=targets, bias="none"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if any(parameter.requires_grad and "lora_" not in name for name, parameter in model.named_parameters()):
        raise ValueError("Non-adapter parameters unexpectedly require gradients")
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    if accelerator.is_main_process:
        output = Path(args.output)
        if (output / "adapter_config.json").exists():
            raise FileExistsError("Refusing to overwrite an existing trained adapter")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run_config.json", {**vars(args), "code": code_revision(), "world_size": accelerator.num_processes,
                  "effective_batch_size": accelerator.num_processes * args.accumulation,
                  "hostname": socket.gethostname(), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "reader_input_protocol": READER_INPUT_PROTOCOL,
                  "samples_sha256": sha256_file(args.samples),
                  "torch29_equivalent_linear_patch_embed_count": patch_count,
                  "hardware": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "total_parameters": sum(p.numel() for p in model.parameters())})
    accelerator.wait_for_everyone()
    return accelerator, model, optimizer


def train_loop(args, accelerator, model, optimizer, samples, loss_function):
    if len(samples) < accelerator.num_processes:
        raise ValueError("Too few training examples for distinct per-GPU work")
    randomizer = random.Random(args.seed)
    order = list(range(len(samples)))
    randomizer.shuffle(order)
    # Same fixed list on every rank; pad its end instead of dropping samples.
    while len(order) % accelerator.num_processes:
        order.append(order[len(order) % len(samples)])
    local = order[accelerator.process_index::accelerator.num_processes]
    model.train()
    trace = []
    started = time.monotonic()
    microsteps = args.max_updates * args.accumulation
    for step in range(microsteps):
        sample = samples[local[step % len(local)]]
        with accelerator.accumulate(model):
            loss, metadata = loss_function(model, sample, step)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Non-finite gradient norm")
                if step < args.accumulation and norm.item() <= 0:
                    raise ValueError("No adapter gradients on the first optimizer update")
            optimizer.step()
            optimizer.zero_grad()
        loss_value = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
        if accelerator.sync_gradients:
            record = {"update": (step + 1) // args.accumulation, "loss": loss_value, "elapsed_seconds": time.monotonic() - started, **metadata}
            trace.append(record)
            if accelerator.is_main_process:
                print(json.dumps(record), flush=True)
                write_json(Path(args.output) / "training_trace.json", trace)
            save_periodic_weights(args, accelerator, model, record["update"])
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(args.output, safe_serialization=True)
    accelerator.wait_for_everyone()
    return trace
