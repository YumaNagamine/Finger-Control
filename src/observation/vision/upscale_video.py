"""Upscale an 800x600 video to 1600x1200."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


SOURCE_SIZE = (800, 600)
TARGET_SIZE = (1600, 1200)


def upscale_video(input_path: Path, output_path: Path) -> None:
    """Upscale a video with Lanczos interpolation while preserving its FPS."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {input_path}")

    writer: cv2.VideoWriter | None = None
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != SOURCE_SIZE:
            raise ValueError(
                f"入力解像度は {width}x{height} です。"
                f"{SOURCE_SIZE[0]}x{SOURCE_SIZE[1]} の動画を指定してください。"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError("入力動画のFPSを取得できません。")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, TARGET_SIZE)
        if not writer.isOpened():
            raise RuntimeError(f"出力動画を作成できません: {output_path}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            upscaled = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            writer.write(upscaled)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="800x600の動画を1600x1200へアップスケールします。"
    )
    parser.add_argument("input", type=Path, help="入力動画のパス")
    parser.add_argument("output", type=Path, help="出力MP4のパス")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upscale_video(args.input, args.output)
    print(f"アップスケール完了: {args.output}")


if __name__ == "__main__":
    main()
