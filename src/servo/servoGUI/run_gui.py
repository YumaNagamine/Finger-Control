import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    app_dir = base_dir / "app"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    os.chdir(app_dir)
    runpy.run_path(str(app_dir / "GUIMain.py"), run_name="__main__")


if __name__ == "__main__":
    main()
