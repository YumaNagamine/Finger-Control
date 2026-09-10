"""Undistort one video without replacing the source file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import load_chessboard_calibration


def _calibration_image_size(path: Path) -> tuple[int, int] | None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    value = data.get("image_size")
    if not isinstance(value, list) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def undistort_video(input_path: Path, output_path: Path, calibration_path: Path) -> None:
    """Undistort a video while preserving its dimensions and FPS."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")

    writer: cv2.VideoWriter | None = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or expected_frames <= 0:
            raise RuntimeError("入力動画のメタデータを取得できません。")

        calibration_size = _calibration_image_size(calibration_path)
        if calibration_size is not None and calibration_size != (width, height):
            raise ValueError(
                f"動画は {width}x{height}、キャリブレーションは "
                f"{calibration_size[0]}x{calibration_size[1]} です。"
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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"出力動画を作成できません: {output_path}")

        written_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR))
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
    parser = argparse.ArgumentParser(description="動画1本のレンズ歪みを補正します。")
    parser.add_argument("input", type=Path, help="入力動画のパス")
    parser.add_argument("output", type=Path, help="出力MP4のパス")
    parser.add_argument("calibration", type=Path, help="キャリブレーションJSONのパス")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    undistort_video(args.input, args.output, args.calibration)
    print(f"歪み補正完了: {args.output}")


if __name__ == "__main__":
    main()
