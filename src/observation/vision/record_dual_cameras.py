from __future__ import annotations

import argparse
import csv
import datetime
import json
import multiprocessing as mp
import queue
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

# Ensure src is importable when running as a script.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import fourcc_from_str
from observation.vision.dual_camera_recording_workers import capture_worker, writer_worker
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_dual_camera_record.json"
DEFAULT_DLC_CAMERA_CONFIG = resolve_param_path("camera_config_dlc.json")
DEFAULT_WEIGHT_CAMERA_CONFIG = resolve_param_path("camera_config_weight.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record two cameras in separate processes.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=None)
    parser.add_argument("--dlc-camera-config", type=str, default=str(DEFAULT_DLC_CAMERA_CONFIG))
    parser.add_argument("--weight-camera-config", type=str, default=str(DEFAULT_WEIGHT_CAMERA_CONFIG))
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def _optional_list_value(values: object, index: int) -> object | None:
    if isinstance(values, list) and len(values) == 2:
        return values[index]
    return None


def _configure_camera(
    base_config: dict,
    camera_index: int,
    camera_name: str,
    calibration_path: object | None,
    calibration_enabled: object | None,
) -> dict:
    config = dict(base_config)
    config["index"] = camera_index
    config["name"] = camera_name
    if calibration_path is not None:
        config["calibration_path"] = str(calibration_path)
    if calibration_enabled is not None:
        config["use_chessboard_calibration"] = bool(calibration_enabled)
    return config


def _load_timestamp_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_paired_timestamps(cam0_path: Path, cam1_path: Path, pair_path: Path) -> int:
    cam0_rows = _load_timestamp_rows(cam0_path)
    cam1_rows = _load_timestamp_rows(cam1_path)
    if not cam0_rows or not cam1_rows:
        raise RuntimeError("Cannot pair empty camera timestamp data")

    cam1_elapsed = [float(row["elapsed_s"]) for row in cam1_rows]
    cam1_index = 0
    with pair_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair_idx",
                "timestamp_iso",
                "elapsed_s",
                "cam0_frame_idx",
                "cam1_frame_idx",
                "cam0_ok",
                "cam1_ok",
            ]
        )
        for pair_index, cam0_row in enumerate(cam0_rows):
            cam0_elapsed = float(cam0_row["elapsed_s"])
            while (
                cam1_index + 1 < len(cam1_rows)
                and abs(cam1_elapsed[cam1_index + 1] - cam0_elapsed)
                < abs(cam1_elapsed[cam1_index] - cam0_elapsed)
            ):
                cam1_index += 1
            writer.writerow(
                [
                    pair_index,
                    cam0_row["timestamp_iso"],
                    f"{cam0_elapsed:.9f}",
                    cam0_row["frame_idx"],
                    cam1_rows[cam1_index]["frame_idx"],
                    1,
                    1,
                ]
            )
    return len(cam0_rows)


def _drain_latest(preview_queue, latest: np.ndarray | None) -> np.ndarray | None:
    while True:
        try:
            latest = preview_queue.get_nowait()
        except queue.Empty:
            return latest


def _join_preview(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    if frame0.shape[0] != frame1.shape[0]:
        frame1 = cv2.resize(frame1, (frame1.shape[1], frame0.shape[0]))
    return cv2.hconcat([frame0, frame1])


def _collect_results(result_queue, expected_complete: int) -> list[dict]:
    results = []
    deadline = time.monotonic() + 5.0
    while sum(item.get("status") == "complete" for item in results) < expected_complete:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        try:
            results.append(result_queue.get(timeout=remaining))
        except queue.Empty:
            break
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            return results


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path, required_keys=("camera", "capture", "output"))
    if args.camera_config:
        camera_config_paths = [Path(args.camera_config).resolve()] * 2
    else:
        camera_config_paths = [
            Path(args.dlc_camera_config).resolve(),
            Path(args.weight_camera_config).resolve(),
        ]

    dual_camera_config = dict(config["camera"])
    camera_indices = dual_camera_config.get("indices", [0, 1])
    camera_names = dual_camera_config.get("names", ["cam0", "cam1"])
    if not isinstance(camera_indices, list) or len(camera_indices) != 2:
        raise ValueError("camera.indices must contain exactly two integers")
    if not isinstance(camera_names, list) or len(camera_names) != 2:
        raise ValueError("camera.names must contain exactly two strings")

    calibration_paths = dual_camera_config.get("calibration_paths")
    calibration_enabled = dual_camera_config.get("use_chessboard_calibration")
    camera_configs = [
        _configure_camera(
            load_config(camera_config_paths[index]),
            int(camera_indices[index]),
            str(camera_names[index]),
            _optional_list_value(calibration_paths, index),
            _optional_list_value(calibration_enabled, index),
        )
        for index in range(2)
    ]

    capture_config = dict(config["capture"])
    output_config = dict(config["output"])
    if args.duration_s is not None:
        capture_config["duration_s"] = float(args.duration_s)
    if args.frame_limit is not None:
        capture_config["frame_limit"] = int(args.frame_limit)
    if args.target_fps is not None:
        capture_config["target_fps"] = float(args.target_fps)

    target_fps = float(capture_config.get("target_fps", 90.0))
    duration_s = float(capture_config.get("duration_s", 0.0))
    frame_limit = int(capture_config.get("frame_limit", 0))
    warmup_frames = max(0, int(capture_config.get("warmup_frames", 5)))
    ring_buffer_frames = int(capture_config.get("ring_buffer_frames", 16))
    preview_fps = float(capture_config.get("preview_fps", 10.0))
    if target_fps <= 0.0:
        raise ValueError("capture.target_fps must be greater than zero")
    if ring_buffer_frames < 2:
        raise ValueError("capture.ring_buffer_frames must be at least 2")

    record_dir = resolve_path(
        output_config.get("record_dir", "logs/dual_camera/recordings"), PROJECT_ROOT
    )
    if record_dir is None:
        raise ValueError("output.record_dir must be provided")
    record_dir.mkdir(parents=True, exist_ok=True)
    session_dir = record_dir / (
        f"{output_config.get('filename_prefix', 'dual_record')}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=False)

    writer_fourcc = str(output_config.get("writer_fourcc", "mp4v"))
    if fourcc_from_str(writer_fourcc) is None:
        raise ValueError(f"Invalid output.writer_fourcc: {writer_fourcc}")
    video_extension = str(output_config.get("video_extension", ".mp4")).strip().lower()
    if video_extension not in {".mp4", ".avi"}:
        raise ValueError("output.video_extension must be '.mp4' or '.avi'")
    show_window = (not args.no_window) and bool(output_config.get("show_window", True))

    context = mp.get_context("spawn")
    start_event = context.Event()
    stop_event = context.Event()
    start_ns = context.Value("q", 0)
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    shared_blocks = []
    preview_queues = []
    camera_paths = []

    for index, camera_id in enumerate(("cam0", "cam1")):
        camera_config = camera_configs[index]
        height = int(camera_config.get("height", 1200))
        width = int(camera_config.get("width", 1600))
        frame_shape = (ring_buffer_frames, height, width, 3)
        shared_block = shared_memory.SharedMemory(
            create=True, size=int(np.prod(frame_shape, dtype=np.int64))
        )
        shared_blocks.append(shared_block)
        free_slots = context.Queue(maxsize=ring_buffer_frames)
        filled_slots = context.Queue(maxsize=ring_buffer_frames)
        for slot in range(ring_buffer_frames):
            free_slots.put(slot)
        preview_queue = context.Queue(maxsize=1) if show_window else None
        preview_queues.append(preview_queue)

        video_path = session_dir / f"{camera_id}{video_extension}"
        timestamp_path = session_dir / f"{camera_id}_timestamps.csv"
        camera_paths.append((video_path, timestamp_path))
        processes.append(
            context.Process(
                target=writer_worker,
                name=f"writer-{camera_id}",
                args=(
                    camera_id,
                    str(video_path),
                    str(timestamp_path),
                    shared_block.name,
                    frame_shape,
                    free_slots,
                    filled_slots,
                    result_queue,
                    stop_event,
                    start_ns,
                    target_fps,
                    writer_fourcc,
                ),
            )
        )
        processes.append(
            context.Process(
                target=capture_worker,
                name=f"capture-{camera_id}",
                args=(
                    camera_id,
                    camera_config,
                    shared_block.name,
                    frame_shape,
                    free_slots,
                    filled_slots,
                    ready_queue,
                    result_queue,
                    preview_queue,
                    start_event,
                    stop_event,
                    start_ns,
                    target_fps,
                    warmup_frames,
                    duration_s,
                    frame_limit,
                    preview_fps,
                ),
            )
        )

    worker_results = []
    camera_properties = {}
    preview_name = "Dual Camera Recorder"
    try:
        for process in processes:
            process.start()
        while len(camera_properties) < 2:
            try:
                ready = ready_queue.get(timeout=15.0)
            except queue.Empty as exc:
                stop_event.set()
                raise RuntimeError("Timed out waiting for camera workers") from exc
            if ready.get("status") == "error":
                stop_event.set()
                raise RuntimeError(
                    f"{ready['camera_id']} failed during startup: {ready['error']}"
                )
            camera_properties[str(ready["camera_id"])] = dict(ready["properties"])

        start_ns.value = time.perf_counter_ns()
        start_event.set()
        print(f"Recording started at target {target_fps:.3f} FPS")
        if show_window:
            cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
            print("Press ESC in the preview window to stop recording.")

        latest_previews = [None, None]
        while not stop_event.is_set():
            if show_window:
                latest_previews[0] = _drain_latest(preview_queues[0], latest_previews[0])
                latest_previews[1] = _drain_latest(preview_queues[1], latest_previews[1])
                if latest_previews[0] is not None and latest_previews[1] is not None:
                    cv2.imshow(preview_name, _join_preview(latest_previews[0], latest_previews[1]))
                    if cv2.waitKey(1) & 0xFF == 27:
                        stop_event.set()
            time.sleep(0.002)

        capture_processes = [
            process for process in processes if process.name.startswith("capture-")
        ]
        writer_processes = [
            process for process in processes if process.name.startswith("writer-")
        ]
        for process in [*capture_processes, *writer_processes]:
            process.join(timeout=30.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
                worker_results.append(
                    {"worker": process.name, "status": "error", "error": "join timeout"}
                )
        worker_results.extend(_collect_results(result_queue, expected_complete=4))
        errors = [result for result in worker_results if result.get("status") == "error"]
        if errors:
            details = "\n".join(f"{item['worker']}: {item['error']}" for item in errors)
            raise RuntimeError(f"Recording worker failed:\n{details}")

        results = {item["worker"]: item for item in worker_results}
        pair_path = session_dir / "pair_timestamps.csv"
        pair_rows = _write_paired_timestamps(
            camera_paths[0][1], camera_paths[1][1], pair_path
        )
        manifest = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_dir": str(session_dir),
            "video_paths": {"cam0": str(camera_paths[0][0]), "cam1": str(camera_paths[1][0])},
            "timestamp_csv_paths": {
                "cam0": str(camera_paths[0][1]),
                "cam1": str(camera_paths[1][1]),
                "pair": str(pair_path),
            },
            "capture": {
                "duration_s": duration_s,
                "frame_limit": frame_limit,
                "target_fps": target_fps,
                "warmup_frames": warmup_frames,
                "ring_buffer_frames": ring_buffer_frames,
                "preview_fps": preview_fps,
            },
            "output": {
                "writer_fourcc": writer_fourcc,
                "video_extension": video_extension,
                "show_window": show_window,
            },
            "camera_names": {"cam0": camera_names[0], "cam1": camera_names[1]},
            "camera_config_paths": {
                "cam0": str(camera_config_paths[0]),
                "cam1": str(camera_config_paths[1]),
            },
            "camera_configs": {"cam0": camera_configs[0], "cam1": camera_configs[1]},
            "camera_reported_properties": camera_properties,
            "saved_frames": {
                "cam0": results["writer-cam0"]["saved_frames"],
                "cam1": results["writer-cam1"]["saved_frames"],
                "pair_rows": pair_rows,
            },
            "actual_fps": {
                "cam0": results["writer-cam0"]["actual_fps"],
                "cam1": results["writer-cam1"]["actual_fps"],
            },
            "dropped_frames": {
                "cam0": results["capture-cam0"]["dropped_frames"],
                "cam1": results["capture-cam1"]["dropped_frames"],
            },
            "undistortion_during_recording": False,
            "worker_results": worker_results,
        }
        manifest_path = session_dir / "session_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved session to {session_dir}")
        print(f"cam0 actual FPS: {manifest['actual_fps']['cam0']:.3f}")
        print(f"cam1 actual FPS: {manifest['actual_fps']['cam1']:.3f}")
    finally:
        stop_event.set()
        start_event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        for shared_block in shared_blocks:
            shared_block.close()
            shared_block.unlink()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.freeze_support()
    main()
