"""Undistort an 800x600 video and upscale it to 1600x1200 in one pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import load_chessboard_calibration
from observation.vision.undistort_video import _calibration_image_size


SOURCE_SIZE = (800, 600)
TARGET_SIZE = (1600, 1200)


def upscale_and_undistort_video(
    input_path: Path,
    output_path: Path,
    calibration_path: Path,
) -> None:
    """Undistort at the calibrated size, then upscale with Lanczos interpolation."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")

    writer: cv2.VideoWriter | None = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (width, height) != SOURCE_SIZE:
            raise ValueError(
                f"入力解像度は {width}x{height} です。"
                f"{SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} の動画を指定してください。"
            )
        if fps <= 0 or expected_frames <= 0:
            raise RuntimeError("入力動画のFPSまたはフレーム数を取得できません。")

        calibration_size = _calibration_image_size(calibration_path)
        if calibration_size is not None and calibration_size != SOURCE_SIZE:
            raise ValueError(
                f"キャリブレーションは {calibration_size[0]}x{calibration_size[1]} 用です。"
                f"{SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} 用を指定してください。"
            )

        camera_matrix, distortion_coefficients = load_chessboard_calibration(calibration_path)
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion_coefficients,
            None,
            camera_matrix,
            SOURCE_SIZE,
            cv2.CV_32FC1,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            TARGET_SIZE,
        )
        if not writer.isOpened():
            raise RuntimeError(f"出力動画を作成できません: {output_path}")

        written_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            undistorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            upscaled = cv2.resize(undistorted, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            writer.write(upscaled)
            written_frames += 1

        if written_frames != expected_frames:
            raise RuntimeError(
                f"フレーム数が一致しません: input={expected_frames}, output={written_frames}"
            )
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="800x600動画を歪み補正し、1600x1200へアップスケールします。"
    )
    parser.add_argument("input", type=Path, help="入力動画のパス")
    parser.add_argument("output", type=Path, help="出力MP4のパス")
    parser.add_argument("calibration", type=Path, help="800x600用キャリブレーションJSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upscale_and_undistort_video(args.input, args.output, args.calibration)
    print(f"歪み補正・アップスケール完了: {args.output}")


if __name__ == "__main__":
    main()
