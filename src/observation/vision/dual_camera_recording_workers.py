from __future__ import annotations

import csv
import datetime
import queue
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

from observation.camera.camera_utils import apply_camera_settings, fourcc_from_str, resolve_backend


TIMESTAMP_HEADER = ["frame_idx", "timestamp_iso", "elapsed_s"]


def camera_properties(cap: cv2.VideoCapture) -> dict[str, float | int | str]:
    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "fourcc": fourcc,
    }


def capture_worker(
    camera_id: str,
    camera_cfg: dict,
    shared_memory_name: str,
    frame_shape: tuple[int, int, int, int],
    free_slots,
    filled_slots,
    ready_queue,
    result_queue,
    preview_queue,
    start_event,
    stop_event,
    start_ns,
    target_fps: float,
    warmup_frames: int,
    duration_s: float,
    frame_limit: int,
    preview_fps: float,
) -> None:
    cap = None
    shm = None
    captured_frames = 0
    dropped_frames = 0
    error: dict | None = None
    ready_sent = False
    if preview_queue is not None:
        # Preview frames are disposable. Do not keep this process alive while the
        # Queue feeder flushes a frame after the coordinator has stopped reading.
        preview_queue.cancel_join_thread()
    try:
        backend = resolve_backend(camera_cfg.get("backend"))
        camera_index = int(camera_cfg.get("index", 0))
        cap = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not cap.isOpened():
            raise RuntimeError(f"Camera not available: {camera_id}, index={camera_index}")
        apply_camera_settings(cap, camera_cfg)
        for _ in range(warmup_frames):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError(f"Warmup frame read failed: {camera_id}")

        properties = camera_properties(cap)
        expected_height, expected_width = frame_shape[1:3]
        if properties["width"] != expected_width or properties["height"] != expected_height:
            raise RuntimeError(
                f"{camera_id} returned {properties['width']}x{properties['height']}; "
                f"expected {expected_width}x{expected_height}"
            )

        shm = shared_memory.SharedMemory(name=shared_memory_name)
        ring = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
        ready_queue.put({"camera_id": camera_id, "status": "ready", "properties": properties})
        ready_sent = True
        start_event.wait()

        preview_stride = max(1, int(round(target_fps / preview_fps))) if preview_fps > 0 else 1
        while not stop_event.is_set():
            ok, frame = cap.read()
            capture_ns = time.perf_counter_ns()
            wall_time_ns = time.time_ns()
            if not ok or frame is None:
                raise RuntimeError(f"Frame read failed: {camera_id}")
            if frame.shape != frame_shape[1:]:
                raise RuntimeError(f"{camera_id} frame shape changed to {frame.shape}")

            try:
                slot = free_slots.get_nowait()
            except queue.Empty as exc:
                dropped_frames += 1
                raise RuntimeError(f"Shared frame buffer overflow: {camera_id}") from exc

            ring[slot] = frame
            filled_slots.put((slot, captured_frames, capture_ns, wall_time_ns))
            if preview_queue is not None and preview_fps > 0 and captured_frames % preview_stride == 0:
                try:
                    preview_queue.put_nowait(frame[::4, ::4].copy())
                except queue.Full:
                    pass

            captured_frames += 1
            elapsed_s = (capture_ns - start_ns.value) / 1_000_000_000.0
            if frame_limit > 0 and captured_frames >= frame_limit:
                stop_event.set()
            if duration_s > 0.0 and elapsed_s >= duration_s:
                stop_event.set()
    except Exception as exc:
        stop_event.set()
        error = {
            "worker": f"capture-{camera_id}",
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not ready_sent:
            ready_queue.put(
                {"camera_id": camera_id, "status": "error", "error": str(exc)}
            )
    finally:
        try:
            filled_slots.put(None, timeout=2.0)
        except queue.Full:
            pass
        if cap is not None:
            cap.release()
        if shm is not None:
            shm.close()
        if error is not None:
            result_queue.put(error)
        result_queue.put(
            {
                "worker": f"capture-{camera_id}",
                "status": "complete",
                "captured_frames": captured_frames,
                "dropped_frames": dropped_frames,
            }
        )


def writer_worker(
    camera_id: str,
    video_path: str,
    timestamp_csv_path: str,
    shared_memory_name: str,
    frame_shape: tuple[int, int, int, int],
    free_slots,
    filled_slots,
    result_queue,
    stop_event,
    start_ns,
    target_fps: float,
    writer_fourcc_str: str,
) -> None:
    shm = None
    writer = None
    saved_frames = 0
    first_elapsed_s = None
    last_elapsed_s = None
    error: dict | None = None
    try:
        shm = shared_memory.SharedMemory(name=shared_memory_name)
        ring = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)
        fourcc = fourcc_from_str(writer_fourcc_str)
        if fourcc is None:
            raise ValueError(f"Invalid writer FOURCC: {writer_fourcc_str}")
        width, height = frame_shape[2], frame_shape[1]
        writer = cv2.VideoWriter(video_path, fourcc, target_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        with Path(timestamp_csv_path).open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(TIMESTAMP_HEADER)
            while True:
                item = filled_slots.get()
                if item is None:
                    break
                slot, frame_index, capture_ns, wall_time_ns = item
                try:
                    writer.write(ring[slot])
                    elapsed_s = (capture_ns - start_ns.value) / 1_000_000_000.0
                    timestamp_iso = datetime.datetime.fromtimestamp(
                        wall_time_ns / 1_000_000_000.0
                    ).isoformat(timespec="milliseconds")
                    csv_writer.writerow([frame_index, timestamp_iso, f"{elapsed_s:.9f}"])
                    first_elapsed_s = elapsed_s if first_elapsed_s is None else first_elapsed_s
                    last_elapsed_s = elapsed_s
                    saved_frames += 1
                finally:
                    free_slots.put(slot)
    except Exception as exc:
        stop_event.set()
        error = {
            "worker": f"writer-{camera_id}",
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if writer is not None:
            writer.release()
        if shm is not None:
            shm.close()
        duration_s = (
            last_elapsed_s - first_elapsed_s
            if first_elapsed_s is not None and last_elapsed_s is not None
            else 0.0
        )
        actual_fps = (saved_frames - 1) / duration_s if saved_frames > 1 and duration_s > 0 else 0.0
        if error is not None:
            result_queue.put(error)
        result_queue.put(
            {
                "worker": f"writer-{camera_id}",
                "status": "complete",
                "saved_frames": saved_frames,
                "duration_s": duration_s,
                "actual_fps": actual_fps,
            }
        )
