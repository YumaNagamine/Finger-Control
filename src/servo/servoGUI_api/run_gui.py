from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    base_file = Path(__file__).resolve()
    src_dir = base_file.parents[2]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    runpy.run_module("servo.servoGUI_api.main", run_name="__main__")


if __name__ == "__main__":
    main()
