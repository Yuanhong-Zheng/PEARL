#!/usr/bin/env python3
"""Forward shared teacher requests to this node's existing local model."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pearl_training.fleet_transport import run_worker

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pool", required=True, type=Path)
parser.add_argument("--endpoint", default="http://127.0.0.1:28790/v1")
parser.add_argument("--workers", type=int, default=128)
args = parser.parse_args()
run_worker(args.pool, os.environ["FLEET_NODE_NAME"], args.endpoint, args.workers)
