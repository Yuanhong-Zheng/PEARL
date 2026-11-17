#!/usr/bin/env python3
"""Plan public112 baseline/joint evaluation; execution needs explicit services."""
import argparse
import json
from pathlib import Path

from pearl_training.public112_adapter import make_plan
from pearl_training.public112_runner import freeze_plan, run_arm, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release-root", "embedding-base", "reader-base", "embedding-adapter", "reader-adapter", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--arms", nargs="+", default=["baseline", "joint"])
    parser.add_argument("--arm", choices=("baseline", "embedding_only", "reader_only", "joint"))
    parser.add_argument("--reader-url")
    parser.add_argument("--embedding-url")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    plan = make_plan(release_root=args.release_root, embedding_base=args.embedding_base, reader_base=args.reader_base,
        embedding_adapter=args.embedding_adapter, reader_adapter=args.reader_adapter, output=args.output, arms=args.arms)
    if args.execute:
        if args.summarize or not all((args.arm, args.reader_url, args.embedding_url)):
            parser.error("--execute requires --arm, --reader-url and --embedding-url, without --summarize")
        result = run_arm(plan, args.arm, args.reader_url, args.embedding_url,
                         shard_index=args.shard_index, shard_count=args.shard_count)
    elif args.summarize:
        freeze_plan(plan, Path(args.output)/"plan.json")
        result = summarize(plan)
    else:
        plan_path = Path(args.output)/("plan.json" if all(item["ready"] for item in plan["adapters"].values())
                                      else "preflight_plan.json")
        freeze_plan(plan, plan_path)
        result = {"status": "plan_only", "plan": str(plan_path), "model_calls": 0,
                  "counts": plan["corpus"]["counts"], "recall_questions": plan["corpus"]["recall_eligible_count"],
                  "adapters_ready": all(item["ready"] for item in plan["adapters"].values())}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
