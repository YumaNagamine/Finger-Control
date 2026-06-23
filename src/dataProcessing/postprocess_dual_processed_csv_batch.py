from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure src directory is importable when running as a script.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from dataProcessing.postprocess_dual_processed_csv import DEFAULT_OUTPUT_ROOT, process_processed_csv
from utils.path_utils import resolve_path


PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "processed_csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch post-process dual-camera processed CSV files.")
    parser.add_argument("--csv-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _resolve_existing_dir(raw_path: str, base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, base_dir)
    if path is None:
        raise ValueError(f"Invalid directory path: {raw_path!r}")
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    return path


def _resolve_output_dir(raw_path: str, base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, base_dir)
    if path is None:
        raise ValueError(f"Invalid output directory path: {raw_path!r}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_csv_files(csv_dir: Path) -> list[Path]:
    return sorted(csv_dir.glob("*.csv"), key=lambda path: path.name.lower())


def main() -> None:
    args = parse_args()
    csv_dir = _resolve_existing_dir(args.csv_dir, Path.cwd())
    output_root = _resolve_output_dir(args.output_root, Path.cwd())
    csv_files = find_csv_files(csv_dir)

    print(f"[batch] csv dir: {csv_dir}")
    print(f"[batch] output root: {output_root}")
    print(f"[batch] found {len(csv_files)} csv files")

    successes: list[Path] = []
    failures: list[tuple[Path, Exception]] = []

    for csv_path in csv_files:
        print(f"[batch] processing: {csv_path.name}")
        try:
            process_processed_csv(csv_path, output_root)
        except Exception as exc:
            failures.append((csv_path, exc))
            print(f"[batch] failed: {csv_path.name}: {exc}")
            continue
        successes.append(csv_path)
        print(f"[batch] finished: {csv_path.name}")

    print(f"[batch] completed {len(successes)} / {len(csv_files)} csv files")
    if failures:
        for csv_path, exc in failures:
            print(f"[batch] failure detail: {csv_path.name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
