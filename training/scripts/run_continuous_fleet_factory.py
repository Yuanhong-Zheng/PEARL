#!/usr/bin/env python3
"""Continuously feed the unchanged V4 factory across ready teacher nodes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from pearl_training.fleet_continuous import main

if __name__ == "__main__":
    main()
