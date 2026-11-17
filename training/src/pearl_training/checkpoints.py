"""Periodic weight snapshots without changing optimizer state or RNG streams."""
from contextlib import contextmanager
import os
from pathlib import Path
import random

import numpy as np
import torch

from .io import write_json


@contextmanager
def preserve_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def save_periodic_weights(args, accelerator, model, update):
    every = getattr(args, "save_every", 1000)
    if every < 0:
        raise ValueError("save_every must be nonnegative")
    if not every or update % every:
        return
    accelerator.wait_for_everyone()
    destination = Path(args.output) / f"checkpoint-{update}"
    if accelerator.is_main_process:
        if destination.exists():
            raise FileExistsError(f"Preserve the existing weight checkpoint: {destination}")
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with preserve_rng(accelerator.device):
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(temporary, safe_serialization=True)
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(args.model, local_files_only=True).save_pretrained(temporary)
            write_json(temporary / "checkpoint_metadata.json", {
                "optimizer_update": update, "save_every": every,
                "samples": args.samples, "base_model": args.model,
                "world_size": accelerator.num_processes, "seed": args.seed,
                "contents": "model_weights_and_processor", "rng_preserved": True,
            })
        temporary.rename(destination)
        print(f"WEIGHT_CHECKPOINT_SAVED {update} {destination}", flush=True)
    accelerator.wait_for_everyone()
