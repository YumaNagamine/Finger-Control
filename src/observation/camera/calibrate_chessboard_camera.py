from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Set params such as cols and rows etc. before doing this!!!!

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = SCRIPT_DIR / "chessboard_imgs"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "camera_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics and distortion from chessboard images."
    )
    parser.add_argument("--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of inner corners along chessboard columns (x).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of inner corners along chessboard rows (y).",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=15.0,
        help="Chessboard square size in millimeters.",
    )
    parser.add_argument(
        "--min-valid-images",
        type=int,
        default=10,
        help="Minimum number of valid chessboard detections required to calibrate.",
    )
    return parser.parse_args()


def _list_images(images_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(images_dir.glob(pattern))
    return sorted(files)


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if ok:
            return ok, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, pattern_size, flags)


def _mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += float(err * err)
        total_points += int(len(objp))

    if total_points == 0:
        return float("nan")
    return float((total_error / total_points) ** 0.5)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    pattern_size = (int(args.cols), int(args.rows))
    square_size_mm = float(args.square_size_mm)
    min_valid_images = int(args.min_valid_images)

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    image_paths = _list_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        ok, corners = _find_corners(gray, pattern_size)
        if not ok:
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        object_points.append(objp.copy())
        image_points.append(corners)
        used_images.append(path.name)

    if image_size is None:
        raise RuntimeError("Could not read any valid image.")

    if len(object_points) < min_valid_images:
        raise RuntimeError(
            f"Valid chessboard detections are too few: {len(object_points)} < {min_valid_images}. "
            "Collect more views with varied positions and angles."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    mean_reproj_error = _mean_reprojection_error(
        object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    result = {
        "camera_model": "opencv_pinhole",
        "image_size": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error": mean_reproj_error,
        "chessboard": {
            "inner_corners": [pattern_size[0], pattern_size[1]],
            "square_size_mm": square_size_mm,
        },
        "num_images_found": len(image_paths),
        "num_images_used": len(object_points),
        "used_image_files": used_images,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved camera calibration to {output_path}")
    print(f"Images used: {len(object_points)}/{len(image_paths)}")
    print(f"RMS reprojection error: {rms:.4f}")
    print(f"Mean reprojection error: {mean_reproj_error:.4f}")


if __name__ == "__main__":
    main()
