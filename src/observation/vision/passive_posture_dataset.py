"""Capture and process passive finger postures for P2P goal selection.

The tool deliberately does not import or operate any servo/encoder code.  Images
are captured first and DLC inference is run later so the camera is owned by only
one process at a time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from observation.vision.dlc_angle_processor import DLCAngleProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DLC_CONFIG = SCRIPT_DIR / "config_deeplabcut_angle_control.json"
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config_dlc.json")
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "logs" / "posture_dataset"
JOINTS = ("DIP", "PIP", "MCP")
CSV_FIELDS = (
    "posture_id",
    "image_path",
    "annotated_image_path",
    "dip_deg",
    "pip_deg",
    "mcp_deg",
    "min_likelihood",
    "mean_likelihood",
    "status",
    "failure_reason",
    "approved",
    "captured_at",
    "analyzed_at",
    "dlc_config_path",
    "dlc_config_sha256",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _next_dataset_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for candidate in root.glob("dataset_*"):
        suffix = candidate.name.removeprefix("dataset_")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return root / f"dataset_{max(numbers, default=0) + 1:03d}"


def _open_camera(camera_cfg: dict[str, Any]) -> cv2.VideoCapture:
    camera_index = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    capture = (
        cv2.VideoCapture(camera_index, backend)
        if backend is not None
        else cv2.VideoCapture(camera_index)
    )
    if not capture.isOpened():
        raise RuntimeError(f"Camera not available: index={camera_index}")
    apply_camera_settings(capture, camera_cfg)
    return capture


def capture_postures(
    dataset_dir: Path,
    camera_config_path: Path,
    capture_factory: Callable[[dict[str, Any]], Any] = _open_camera,
    show_window: bool = True,
    key_provider: Callable[[], int] | None = None,
) -> int:
    """Capture Enter-triggered still images without running DLC inference."""
    dataset_dir = dataset_dir.resolve()
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    camera_config_path = camera_config_path.resolve()
    camera_cfg = load_config(camera_config_path)
    calibration = setup_undistortion_from_config(
        camera_cfg, log_prefix="[posture-capture]"
    )
    capture = capture_factory(camera_cfg)
    records: list[dict[str, Any]] = []
    metadata_path = dataset_dir / "capture_metadata.json"
    existing = _load_json(metadata_path) if metadata_path.exists() else {}
    records.extend(existing.get("images", []))
    next_index = max(
        (int(item["posture_id"].split("_")[-1]) for item in records),
        default=0,
    ) + 1
    window_name = "Passive Posture Capture - Enter: save, Esc/Q: quit"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                raise RuntimeError("Failed to read a frame from the camera")
            frame = undistort_frame(raw_frame, calibration)
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Next: posture_{next_index:04d} | Enter: capture | Esc/Q: quit",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if show_window:
                cv2.imshow(window_name, preview)
            if key_provider is not None:
                key = key_provider() & 0xFF
            elif show_window:
                key = cv2.waitKey(1) & 0xFF
            else:
                raise ValueError("key_provider is required when show_window is false")
            if key in (27, ord("q"), ord("Q")):
                break
            if key not in (10, 13):
                continue

            posture_id = f"posture_{next_index:04d}"
            image_path = images_dir / f"{posture_id}.png"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Failed to save image: {image_path}")
            height, width = frame.shape[:2]
            records.append(
                {
                    "posture_id": posture_id,
                    "image_path": _relative(image_path, dataset_dir),
                    "captured_at": _now_iso(),
                    "width": int(width),
                    "height": int(height),
                    "color_order": "BGR",
                    "undistorted": calibration is not None,
                }
            )
            next_index += 1
            print(f"Saved {image_path}")
    finally:
        capture.release()
        if show_window:
            cv2.destroyWindow(window_name)
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "dataset_dir": str(dataset_dir),
                "camera_config_path": str(camera_config_path),
                "camera_config": camera_cfg,
                "updated_at": _now_iso(),
                "images": records,
            },
        )
    return len(records)


def _runtime_dlc_config(config_path: Path) -> dict[str, Any]:
    config = load_config(
        config_path, required_keys=("input", "dlc", "keypoints", "output")
    )
    runtime = dict(config)
    dlc_cfg = dict(config["dlc"])
    for key in ("third_party_path", "model_path"):
        resolved = resolve_path(dlc_cfg.get(key), config_path.parent)
        if resolved is not None:
            dlc_cfg[key] = str(resolved)
    runtime["dlc"] = dlc_cfg
    return runtime


def _flexion_angles(raw_angles: dict[str, Any]) -> dict[str, float]:
    return {joint: 180.0 - float(raw_angles.get(joint, math.nan)) for joint in JOINTS}


def _quality(result: dict[str, Any], minimum_likelihood: float) -> tuple[str, str, float, float]:
    keypoints = list(result.get("keypoints", {}).values())
    likelihoods = [float(item.get("likelihood", 0.0)) for item in keypoints]
    statuses = [str(item.get("status", "missing")) for item in keypoints]
    minimum = min(likelihoods, default=0.0)
    mean = float(np.mean(likelihoods)) if likelihoods else 0.0
    flexion = _flexion_angles(result.get("angles", {}))
    if any(not math.isfinite(value) for value in flexion.values()):
        return "failed", "one or more joint angles are not finite", minimum, mean
    if any(status != "detected" for status in statuses):
        return "failed", "one or more DLC keypoints were not detected", minimum, mean
    if minimum < minimum_likelihood:
        return (
            "failed",
            f"minimum DLC likelihood {minimum:.3f} is below {minimum_likelihood:.3f}",
            minimum,
            mean,
        )
    return "pending_review", "", minimum, mean


def _draw_dataset_labels(
    overlay: np.ndarray,
    posture_id: str,
    flexion: dict[str, float],
    status: str,
    reason: str,
) -> np.ndarray:
    out = overlay.copy()
    height, width = out.shape[:2]
    box_width = min(430, max(240, width - 20))
    x0 = max(10, width - box_width - 10)
    lines = [posture_id, "P2P flexion angles"]
    lines.extend(f"{joint}: {flexion[joint]:.1f} deg" for joint in JOINTS)
    lines.append(f"status: {status}")
    if reason:
        lines.append(reason[:55])
    box_height = min(height - 20, 32 * len(lines) + 15)
    cv2.rectangle(out, (x0, 10), (width - 10, 10 + box_height), (20, 20, 20), -1)
    color = (0, 220, 0) if status == "pending_review" else (0, 0, 255)
    for index, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x0 + 10, 37 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color if index in (0, len(lines) - 1) else (255, 255, 255),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    return out


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _capture_times(dataset_dir: Path) -> dict[str, str]:
    path = dataset_dir / "capture_metadata.json"
    if not path.exists():
        return {}
    return {
        str(item.get("posture_id")): str(item.get("captured_at", ""))
        for item in _load_json(path).get("images", [])
    }


def process_dataset(
    dataset_dir: Path,
    dlc_config_path: Path,
    minimum_likelihood: float = 0.3,
    processor: Any | None = None,
) -> list[dict[str, Any]]:
    """Run DLC over captured PNG files and write review artifacts."""
    if not 0.0 <= minimum_likelihood <= 1.0:
        raise ValueError("minimum_likelihood must be in the range 0-1")
    dataset_dir = dataset_dir.resolve()
    dlc_config_path = dlc_config_path.resolve()
    images = sorted((dataset_dir / "images").glob("posture_*.png"))
    if not images:
        raise FileNotFoundError(f"No posture PNG files found in {dataset_dir / 'images'}")
    annotated_dir = dataset_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    if processor is None:
        runtime = _runtime_dlc_config(dlc_config_path)
        processor = DLCAngleProcessor(runtime, dlc_config_path.parent, enable_live=True)
    config_hash = hashlib.sha256(dlc_config_path.read_bytes()).hexdigest()
    captured_at = _capture_times(dataset_dir)
    rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(images):
        posture_id = image_path.stem
        analyzed_at = _now_iso()
        processor.reset()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            result: dict[str, Any] = {"angles": {}, "keypoints": {}}
            overlay = np.zeros((480, 640, 3), dtype=np.uint8)
            status, reason, minimum, mean = "failed", "image could not be read", 0.0, 0.0
        else:
            try:
                result, overlay = processor.process_frame(image, index)
                status, reason, minimum, mean = _quality(result, minimum_likelihood)
            except Exception as exc:
                result = {"angles": {}, "keypoints": {}}
                overlay = image.copy()
                status, reason, minimum, mean = (
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    0.0,
                    0.0,
                )
        flexion = _flexion_angles(result.get("angles", {}))
        annotated = _draw_dataset_labels(overlay, posture_id, flexion, status, reason)
        annotated_path = annotated_dir / image_path.name
        if not cv2.imwrite(str(annotated_path), annotated):
            raise RuntimeError(f"Failed to save annotated image: {annotated_path}")
        rows.append(
            {
                "posture_id": posture_id,
                "image_path": _relative(image_path, dataset_dir),
                "annotated_image_path": _relative(annotated_path, dataset_dir),
                "dip_deg": flexion["DIP"],
                "pip_deg": flexion["PIP"],
                "mcp_deg": flexion["MCP"],
                "min_likelihood": minimum,
                "mean_likelihood": mean,
                "status": status,
                "failure_reason": reason,
                "approved": "",
                "captured_at": captured_at.get(posture_id, ""),
                "analyzed_at": analyzed_at,
                "dlc_config_path": str(dlc_config_path),
                "dlc_config_sha256": config_hash,
            }
        )
        print(f"{posture_id}: {status}{' - ' + reason if reason else ''}")

    _write_csv(dataset_dir / "dlc_predictions.csv", rows)
    _write_csv(dataset_dir / "posture_goals.csv", rows)
    create_overview(dataset_dir, rows)
    return rows


def create_overview(dataset_dir: Path, rows: list[dict[str, Any]], columns: int = 4) -> Path:
    thumbnails: list[np.ndarray] = []
    tile_width, tile_height = 400, 320
    for row in rows:
        image = cv2.imread(str(dataset_dir / str(row["annotated_image_path"])))
        if image is None:
            continue
        scale = min(tile_width / image.shape[1], (tile_height - 35) / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile = np.full((tile_height, tile_width, 3), 32, dtype=np.uint8)
        y = (tile_height - 35 - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            tile,
            f"{row['posture_id']} | {row['status']}",
            (8, tile_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        thumbnails.append(tile)
    if not thumbnails:
        raise RuntimeError("No annotated images were available for overview")
    blank = np.full_like(thumbnails[0], 32)
    row_count = math.ceil(len(thumbnails) / columns)
    thumbnails.extend(blank.copy() for _ in range(row_count * columns - len(thumbnails)))
    overview = np.vstack(
        [np.hstack(thumbnails[offset : offset + columns]) for offset in range(0, len(thumbnails), columns)]
    )
    output = dataset_dir / "overview.png"
    if not cv2.imwrite(str(output), overview):
        raise RuntimeError(f"Failed to save overview: {output}")
    return output


def export_approved(dataset_dir: Path) -> int:
    source = dataset_dir / "posture_goals.csv"
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    true_values = {"1", "true", "yes", "y", "approved"}
    approved = [
        row
        for row in rows
        if row.get("status") == "pending_review"
        and str(row.get("approved", "")).strip().lower() in true_values
    ]
    for row in approved:
        row["status"] = "approved"
    _write_csv(dataset_dir / "approved_goals.csv", approved)
    return len(approved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Capture passive-posture images")
    capture.add_argument("--dataset-dir", type=Path)
    capture.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    capture.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)

    process = subparsers.add_parser("process", help="Run offline DLC and create review files")
    process.add_argument("dataset_dir", type=Path)
    process.add_argument("--dlc-config", type=Path, default=DEFAULT_DLC_CONFIG)
    process.add_argument("--min-likelihood", type=float, default=0.3)

    approve = subparsers.add_parser("export-approved", help="Export manually approved goals")
    approve.add_argument("dataset_dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "capture":
        dataset_dir = args.dataset_dir or _next_dataset_dir(args.dataset_root.resolve())
        count = capture_postures(dataset_dir, args.camera_config)
        print(f"Captured image count: {count}\nDataset: {dataset_dir.resolve()}")
    elif args.command == "process":
        rows = process_dataset(
            args.dataset_dir, args.dlc_config, minimum_likelihood=args.min_likelihood
        )
        print(f"Processed image count: {len(rows)}")
    elif args.command == "export-approved":
        count = export_approved(args.dataset_dir.resolve())
        print(f"Exported approved goal count: {count}")


if __name__ == "__main__":
    main()
