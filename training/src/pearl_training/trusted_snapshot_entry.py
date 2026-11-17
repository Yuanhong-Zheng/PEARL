"""Explicitly reuse completed frozen-data validation before unchanged training.

The operator requested skipping repeated initialization validation. This entry
avoids media filesystem scans and rebuilding immutable helper requests on each
DDP rank. It does not change samples, cached query strings, models or training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import MappingProxyType

from .io import read_json, write_json
from .schema import validate_sample
from .text_query_cache import PROTOCOL, QUERY_MODE, TextQueryCache, _light_binding


TARGETS = {
    "pearl_training.train_retrieval", "pearl_training.train_sft",
    "pearl_training.train_full_retrieval", "pearl_training.train_full_sft",
}


def load_completed_queries(path, samples_path, *, samples, snapshot,
                           release_root=None, allow_test_cache=False):
    """Read the exact completed query map without rerunning its construction."""
    root = Path(snapshot).resolve()
    path = Path(path).resolve(strict=True)
    if (Path(samples_path).resolve() != root / "samples.jsonl"
            or path != root / "release_text_queries.json"):
        raise ValueError("Trusted loading is restricted to the selected frozen snapshot")
    frozen = read_json(root / "dataset_manifest.json")
    cache_bytes = path.read_bytes()
    artifact = json.loads(cache_bytes)
    manifest = artifact["manifest"]
    if manifest.get("status") != "complete" or manifest.get("protocol") != PROTOCOL:
        raise ValueError("Only a completed original query cache may be reused")
    kind = manifest.get("execution_kind")
    if kind != "loopback_base_helper" and not (allow_test_cache and kind == "injected_cpu_test"):
        raise ValueError("CPU-test query outputs cannot be used for actual training")
    if manifest["samples_sha256"] != frozen["samples_sha256"]:
        raise ValueError("Frozen dataset and query cache identities differ")
    rows = {s["sample_id"]: s for s in samples if s["task"] == "past"}
    indexed = {}
    for record in artifact["records"]:
        sid, query = record["sample_id"], record["rewrite"]["text"]
        if sid in indexed or sid not in rows or not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid completed query entry")
        indexed[sid] = query
    if set(indexed) != set(rows) or manifest["queries"] != len(indexed):
        raise ValueError("Completed cache must cover every Past question")
    metadata = {
        "query_mode": QUERY_MODE, "cache_path": str(path),
        "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "samples_sha256": manifest["samples_sha256"],
        "base_helper_sha256": manifest["base_helper_sha256"],
        "helper_contract_sha256": manifest["helper_contract_sha256"],
        "queries": len(indexed),
    }
    return TextQueryCache(MappingProxyType(indexed),
                          MappingProxyType({sid: _light_binding(s) for sid, s in rows.items()}),
                          MappingProxyType(metadata))


def install(module, snapshot, reader_baseline=None):
    """Replace initialization-only checks; leave numerical functions untouched."""
    def structural_check(sample, **kwargs):
        return validate_sample(sample, check_files=False)

    def structural_checks(samples, workers=8):
        for sample in samples:
            structural_check(sample)

    if hasattr(module, "validate_sample"):
        module.validate_sample = structural_check
    if hasattr(module, "validate_schema_samples"):
        module.validate_schema_samples = structural_checks
    if hasattr(module, "load_text_query_cache"):
        def load(path, samples_path, **kwargs):
            # The command-line training entry never permits injected test data.
            kwargs.pop("allow_test_cache", None)
            return load_completed_queries(path, samples_path, snapshot=snapshot, **kwargs)
        module.load_text_query_cache = load
    runtime_args = {}
    if reader_baseline is not None:
        if module.__name__ != "pearl_training.train_sft":
            raise ValueError("A reader baseline can only be reused for the matching LoRA reader")
        original_evaluate = module.evaluate_distributed
        first_evaluation = True

        def evaluate(accelerator, model, processor, samples, max_length):
            nonlocal first_evaluation
            if not first_evaluation:
                return original_evaluate(accelerator, model, processor, samples, max_length)
            previous = Path(reader_baseline).resolve(strict=True)
            config = read_json(previous / "run_config.json")
            current = runtime_args["args"]
            manifest = read_json(Path(snapshot) / "dataset_manifest.json")
            current_config = read_json(Path(current.output) / "run_config.json")
            if (config["samples_sha256"] != manifest["samples_sha256"]
                    or Path(config["model"]).resolve() != Path(current.model).resolve()
                    or config["max_length"] != max_length
                    or config["reader_input_protocol"] != current_config["reader_input_protocol"]):
                raise ValueError("Cached reader baseline has different data/model/inputs")
            before = read_json(previous / "validation_before.json")
            if (len(before["records"]) != len(samples)
                    or {row["sample_id"] for row in before["records"]} != {s["sample_id"] for s in samples}):
                raise ValueError("Cached reader baseline questions differ")
            first_evaluation = False
            model.eval()
            if accelerator.is_main_process:
                write_json(Path(current.output) / "baseline_reuse.json", {
                    "previous_run": str(previous), "questions": len(samples),
                    "purpose": "preserve completed original base evaluation during checkpoint-feature restart",
                })
                print("REUSING_COMPLETED_READER_BASELINE", previous, flush=True)
            accelerator.wait_for_everyone()
            return before

        module.evaluate_distributed = evaluate
    for name in ("initialize", "initialize_full"):
        if not hasattr(module, name):
            continue
        original = getattr(module, name)

        def initialize(*args, _original=original, **kwargs):
            runtime_args["args"] = args[0]
            result = _original(*args, **kwargs)
            if result[0].is_main_process:
                receipt = {
                    "mode": "reuse_previously_validated_frozen_snapshot",
                    "snapshot": str(snapshot), "target_module": module.__name__,
                    "operator_instruction": "Skip repeated initialization validation",
                    "skipped": ["media_existence_rescan", "query_request_reconstruction"],
                    "unchanged": ["samples", "cached_query_text", "training_and_scoring_functions"],
                }
                write_json(Path(args[0].output) / "initialization_reuse.json", receipt)
                print("TRUSTED_SNAPSHOT_MODEL_READY", module.__name__, flush=True)
            return result

        setattr(module, name, initialize)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--target-module", choices=sorted(TARGETS), required=True)
    parser.add_argument("--trusted-snapshot", required=True)
    parser.add_argument("--reader-baseline-run")
    args, remaining = parser.parse_known_args()
    snapshot = Path(args.trusted_snapshot).resolve(strict=True)
    print("REUSING_COMPLETED_FROZEN_VALIDATION", snapshot, flush=True)
    module = importlib.import_module(args.target_module)
    install(module, snapshot, reader_baseline=args.reader_baseline_run)
    sys.argv = [args.target_module, *remaining]
    module.main()


if __name__ == "__main__":
    main()
