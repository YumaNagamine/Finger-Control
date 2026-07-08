### Batch processing script for dual-camera recording directories.

import argparse
import copy
import sys
from pathlib import Path

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.vision.process_dual_recording import DEFAULT_CONFIG, process_dual_recording_config
from utils.config_loader import load_config
from utils.path_utils import resolve_path


PROJECT_ROOT = ROOT.parent
DEFAULT_RECORDINGS_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "recordings"
REQUIRED_RECORDING_FILES = ("cam0.mp4", "cam1.mp4", "pair_timestamps.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch offline post-processing for dual-camera recording directories."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--recordings-root", type=str, default=str(DEFAULT_RECORDINGS_ROOT))
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N pair rows per recording (0 = unlimited).")
    parser.add_argument("--dry-run", action="store_true", help="List target directories without processing them.")
    return parser.parse_args()


def _resolve_existing_dir(raw_path: str, config_base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, config_base_dir)
    if path is None:
        raise ValueError(f"Invalid directory path: {raw_path!r}")
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory path: {path}")
    return path


def is_recording_dir(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_RECORDING_FILES)


def find_recording_dirs(recordings_root: Path) -> list[Path]:
    return sorted(
        [child for child in recordings_root.iterdir() if is_recording_dir(child)],
        key=lambda p: p.name.lower(),
    )


def build_session_config(base_cfg: dict, session_dir: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    input_cfg = dict(cfg["input"])
    input_cfg["cam0_video_path"] = str(session_dir / "cam0.mp4")
    input_cfg["cam1_video_path"] = str(session_dir / "cam1.mp4")
    input_cfg["pair_timestamps_csv"] = str(session_dir / "pair_timestamps.csv")
    cfg["input"] = input_cfg
    return cfg


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_cfg = load_config(
        config_path,
        required_keys=("input", "processors", "output"),
    )
    recordings_root = _resolve_existing_dir(args.recordings_root, config_path.parent)
    recording_dirs = find_recording_dirs(recordings_root)

    print(f"[batch] recordings root: {recordings_root}")
    print(f"[batch] found {len(recording_dirs)} recording directories")
    for session_dir in recording_dirs:
        print(f"[batch] target: {session_dir}")

    if args.dry_run:
        return

    successes: list[tuple[Path, dict[str, object]]] = []
    failures: list[tuple[Path, Exception]] = []

    for session_dir in recording_dirs:
        print(f"[batch] processing: {session_dir.name}")
        session_cfg = build_session_config(base_cfg, session_dir)
        try:
            result = process_dual_recording_config(
                session_cfg,
                config_base_dir=config_path.parent,
                frame_limit=args.frame_limit,
            )
        except Exception as exc:
            failures.append((session_dir, exc))
            print(f"[batch] failed: {session_dir.name}: {exc}")
            continue
        successes.append((session_dir, result))
        print(f"[batch] finished: {session_dir.name}")

    print(f"[batch] completed {len(successes)} / {len(recording_dirs)} recordings")
    if failures:
        print(f"[batch] failures: {len(failures)}")
        for session_dir, exc in failures:
            print(f"[batch] failure detail: {session_dir.name}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

