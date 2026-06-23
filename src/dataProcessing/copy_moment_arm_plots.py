from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "processed_data"
DESTINATION_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "moment arm"
MOMENT_ARM_GLOB = "*_moment_arm.png"


def iter_moment_arm_plots(source_root: Path) -> list[Path]:
    return sorted(source_root.rglob(MOMENT_ARM_GLOB), key=lambda path: str(path).lower())


def make_unique_destination_path(destination_dir: Path, file_name: str) -> Path:
    candidate = destination_dir / file_name
    if not candidate.exists():
        return candidate

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    index = 2
    while True:
        candidate = destination_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_moment_arm_plots() -> int:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_ROOT}")

    DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
    plot_paths = iter_moment_arm_plots(SOURCE_ROOT)

    print(f"Source root: {SOURCE_ROOT}")
    print(f"Destination root: {DESTINATION_ROOT}")
    print(f"Found {len(plot_paths)} moment arm plot(s)")

    copied_count = 0
    for plot_path in plot_paths:
        destination_path = make_unique_destination_path(DESTINATION_ROOT, plot_path.name)
        shutil.copy2(plot_path, destination_path)
        copied_count += 1
        print(f"Copied: {plot_path.name} -> {destination_path.name}")

    print(f"Copied {copied_count} file(s)")
    return copied_count


def main() -> None:
    copy_moment_arm_plots()


if __name__ == "__main__":
    main()
