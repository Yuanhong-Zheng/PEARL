#!/usr/bin/env python3
"""Plan, or explicitly load one isolated public112 reader/embedding service."""
import argparse
import json

from pearl_training.public112_services import (load_embedding_app, load_reader_app,
                                               service_identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("reader", "embedding"), required=True)
    parser.add_argument("--arm", choices=("baseline", "embedding_only", "reader_only", "joint"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--helper-cache")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    identity = service_identity(args.kind, args.arm, args.model, args.adapter, args.release_root)
    if args.kind == "reader" and not args.helper_cache:
        parser.error("All reader arms need the same explicit --helper-cache directory")
    if not args.execute:
        print(json.dumps({"status": "plan_only", "model_calls": 0, "identity": identity}, indent=2))
        return
    app = (load_reader_app(identity, args.release_root, args.helper_cache, args.tensor_parallel_size)
           if args.kind == "reader" else load_embedding_app(identity, args.release_root))
    app.run(host="127.0.0.1", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
