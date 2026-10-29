"""Rebuild version-2 aggregate reports without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller.rl.bc_pretraining_v2 import rebuild_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("moment_arm_dir", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    summary = rebuild_reports(args.output_dir, args.moment_arm_dir, args.config)
    print(json.dumps({"run_count": summary["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
