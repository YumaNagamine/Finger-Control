from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2

# Ensure repository src root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import fourcc_from_str, load_chessboard_calibration


DEFAULT_DLC_CALIBRATION = resolve_param_path("camera_calibration_dlc.json")
DEFAULT_WEIGHT_CALIBRATION = resolve_param_path("camera_calibration_weight.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Undistort cam0/cam1 in one dual-recording directory. Original videos "
            "are renamed to cam0_distorted/cam1_distorted after both outputs validate."
        )
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--cam0-calibration", type=Path, default=DEFAULT_DLC_CALIBRATION)
    parser.add_argument("--cam1-calibration", type=Path, default=DEFAULT_WEIGHT_CALIBRATION)
    parser.add_argument("--fourcc", default="MJPG")
    return parser.parse_args()


def _video_metadata(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {path}: "
            f"size={width}x{height}, fps={fps}, frames={frame_count}"
        )
    return width, height, fps, frame_count


def _calibration_image_size(path: Path) -> tuple[int, int] | None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    value = data.get("image_size")
    if not isinstance(value, list) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _undistort_video(
    source: Path,
    destination: Path,
    calibration_path: Path,
    fourcc_text: str,
) -> dict[str, int | float | str]:
    width, height, fps, expected_frames = _video_metadata(source)
    calibration_size = _calibration_image_size(calibration_path)
    if calibration_size is not None and calibration_size != (width, height):
        raise ValueError(
            f"Calibration/video size mismatch for {source.name}: "
            f"calibration={calibration_size[0]}x{calibration_size[1]}, "
            f"video={width}x{height}"
        )

    camera_matrix, distortion_coefficients = load_chessboard_calibration(calibration_path)
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    fourcc = fourcc_from_str(fourcc_text)
    if fourcc is None:
        raise ValueError(f"FOURCC must contain exactly four characters: {fourcc_text!r}")

    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), fourcc, fps, (width, height))
    if not capture.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video: {source}")
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {destination}")

    written_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR))
            written_frames += 1
    finally:
        capture.release()
        writer.release()

    if written_frames != expected_frames:
        raise RuntimeError(
            f"Frame count changed while decoding {source.name}: "
            f"metadata={expected_frames}, written={written_frames}"
        )
    out_width, out_height, out_fps, out_frames = _video_metadata(destination)
    if (out_width, out_height, out_frames) != (width, height, expected_frames):
        raise RuntimeError(
            f"Output validation failed for {destination.name}: "
            f"expected={width}x{height}/{expected_frames}, "
            f"actual={out_width}x{out_height}/{out_frames}"
        )
    if abs(out_fps - fps) > 0.01:
        raise RuntimeError(
            f"Output FPS validation failed for {destination.name}: expected={fps}, actual={out_fps}"
        )
    return {
        "source": str(source),
        "calibration": str(calibration_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": written_frames,
    }


def undistort_session(
    session_dir: Path,
    calibration_paths: tuple[Path, Path],
    fourcc_text: str = "MJPG",
) -> dict[str, object]:
    session_dir = session_dir.resolve()
    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session_dir}")

    source_paths = [session_dir / "cam0.avi", session_dir / "cam1.avi"]
    backup_paths = [session_dir / "cam0_distorted.avi", session_dir / "cam1_distorted.avi"]
    temp_paths = [session_dir / "cam0_undistorted_tmp.avi", session_dir / "cam1_undistorted_tmp.avi"]
    for path in [*source_paths, *calibration_paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    for path in [*backup_paths, *temp_paths]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    results: dict[str, object] = {}
    try:
        for index in range(2):
            camera_id = f"cam{index}"
            print(f"[{camera_id}] undistorting {source_paths[index].name}")
            results[camera_id] = _undistort_video(
                source_paths[index], temp_paths[index], calibration_paths[index], fourcc_text
            )

        source_paths[0].rename(backup_paths[0])
        try:
            source_paths[1].rename(backup_paths[1])
            temp_paths[0].rename(source_paths[0])
            temp_paths[1].rename(source_paths[1])
        except Exception:
            for source_path, backup_path in zip(source_paths, backup_paths):
                if backup_path.exists():
                    if source_path.exists():
                        source_path.unlink()
                    backup_path.rename(source_path)
            raise
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest["session_dir"] = str(session_dir)
        manifest["video_paths"] = {
            "cam0": str(source_paths[0]),
            "cam1": str(source_paths[1]),
        }
        manifest["distorted_video_paths"] = {
            "cam0": str(backup_paths[0]),
            "cam1": str(backup_paths[1]),
        }
        manifest["undistortion"] = {
            "applied_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "fourcc": fourcc_text,
            "cameras": results,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Undistortion complete: {session_dir}")
    return results


def main() -> None:
    args = parse_args()
    undistort_session(
        args.session_dir,
        (args.cam0_calibration.resolve(), args.cam1_calibration.resolve()),
        args.fourcc,
    )


if __name__ == "__main__":
    main()
