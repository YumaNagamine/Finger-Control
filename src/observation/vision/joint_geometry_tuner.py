from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Any

import cv2

# Ensure the repository src directory is importable when this file is run directly.
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
DEFAULT_DLC_CONFIG = SCRIPT_DIR / "config_deeplabcut_angle.json"
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
WINDOW_TITLE = "Joint Geometry Tuner"
PREVIEW_MAX_WIDTH = 1050
PREVIEW_MAX_HEIGHT = 760

SLIDER_FIELDS: tuple[dict[str, Any], ...] = (
    {
        "key": "theta_rad",
        "label": "Medial rotation [rad]",
        "min": -1.5,
        "max": 1.5,
        "resolution": 0.01,
        "group": "Segment corrections",
    },
    {
        "key": "distance_shift",
        "label": "Proximal shift [px]",
        "min": -200.0,
        "max": 200.0,
        "resolution": 1.0,
        "group": "Segment corrections",
    },
    {
        "key": "palm_horizontal_offset_px",
        "label": "Palm line length [px]",
        "min": 1.0,
        "max": 300.0,
        "resolution": 1.0,
        "group": "Segment corrections",
    },
    {
        "key": "dip_shifter",
        "label": "DIP shift [px]",
        "min": -200.0,
        "max": 200.0,
        "resolution": 1.0,
        "group": "Joint positions",
    },
    {
        "key": "pip_shifter",
        "label": "PIP shift [px]",
        "min": -200.0,
        "max": 200.0,
        "resolution": 1.0,
        "group": "Joint positions",
    },
    {
        "key": "mcp_offset_x",
        "label": "MCP offset X [px]",
        "min": -300.0,
        "max": 300.0,
        "resolution": 1.0,
        "group": "Joint positions",
    },
    {
        "key": "mcp_offset_y",
        "label": "MCP offset Y [px]",
        "min": -300.0,
        "max": 300.0,
        "resolution": 1.0,
        "group": "Joint positions",
    },
)


def _resolve_dlc_runtime_config(config_path: Path) -> dict:
    config = load_config(
        config_path,
        required_keys=("input", "dlc", "keypoints", "output"),
    )
    dlc_cfg = dict(config["dlc"])
    for key in ("third_party_path", "model_path"):
        if dlc_cfg.get(key):
            resolved = resolve_path(str(dlc_cfg[key]), config_path.parent)
            if resolved is not None:
                dlc_cfg[key] = str(resolved)

    runtime_config = dict(config)
    runtime_config["dlc"] = dlc_cfg
    return runtime_config


def save_adjustments_snapshot(
    source_config_path: Path,
    adjustments: dict[str, float | list[float]],
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    stem = f"{source_config_path.stem}_adjustments_{timestamp}"
    output_path = source_config_path.parent / f"{stem}.json"
    suffix = 2
    while output_path.exists():
        output_path = source_config_path.parent / f"{stem}_{suffix}.json"
        suffix += 1

    payload = {"processing": {"adjustments": adjustments}}
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return output_path


def _to_photo_image_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + frame_rgb.tobytes(), format="PPM")


def _resize_preview(frame_bgr, max_width: int, max_height: int):
    height, width = frame_bgr.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame_bgr
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)


class JointGeometryTunerApp:
    def __init__(self, root: tk.Tk, config_path: Path, camera_config_path: Path):
        self.root = root
        self.config_path = config_path.expanduser().resolve()
        self.camera_config_path = camera_config_path.expanduser().resolve()
        self.runtime_config = _resolve_dlc_runtime_config(self.config_path)
        self.camera_config = load_config(self.camera_config_path)
        self.processor = DLCAngleProcessor(
            self.runtime_config,
            self.config_path.parent,
            enable_live=True,
        )
        self.initial_adjustments = self.processor.get_adjustments()
        self.cap: cv2.VideoCapture | None = None
        self.calibration = None
        self.photo = None
        self.frame_index = 0
        self._after_id: str | None = None
        self._closing = False
        self.status_var = tk.StringVar(value=f"Loaded: {self.config_path}")
        self.slider_vars: dict[str, tk.DoubleVar] = {}

        self.root.title(WINDOW_TITLE)
        self.root.geometry("1500x920")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self._open_camera()
        self._schedule_next_frame()

    def _build_ui(self) -> None:
        content = tk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=8, pady=8)

        preview_frame = tk.Frame(content)
        preview_frame.pack(side="left", fill="both", expand=True)
        self.preview_label = tk.Label(preview_frame, text="Starting camera and DLC inference...")
        self.preview_label.pack(fill="both", expand=True)

        controls = tk.Frame(content, width=410)
        controls.pack(side="right", fill="y", padx=(10, 0))
        controls.pack_propagate(False)

        tk.Label(
            controls,
            text=f"DLC config:\n{self.config_path}\n\nCamera config:\n{self.camera_config_path}",
            justify="left",
            anchor="w",
            wraplength=390,
        ).pack(fill="x", pady=(0, 10))

        flat_values = self._flatten_adjustments(self.initial_adjustments)
        current_group = None
        for field in SLIDER_FIELDS:
            if field["group"] != current_group:
                current_group = field["group"]
                tk.Label(controls, text=current_group, font=("TkDefaultFont", 10, "bold"), anchor="w").pack(
                    fill="x", pady=(8, 2)
                )
            self._build_slider(controls, field, flat_values[field["key"]])

        tk.Label(
            controls,
            text=(
                "Medial rotation affects measured angles. Joint position offsets primarily "
                "change the displayed joint locations."
            ),
            justify="left",
            anchor="w",
            wraplength=390,
        ).pack(fill="x", pady=(10, 8))

        buttons = tk.Frame(controls)
        buttons.pack(fill="x", pady=(4, 8))
        tk.Button(buttons, text="Reset", width=12, command=self.reset_adjustments).pack(side="left")
        tk.Button(buttons, text="Save Parameters", width=18, command=self.save_parameters).pack(
            side="left", padx=(8, 0)
        )

        tk.Label(controls, textvariable=self.status_var, justify="left", anchor="w", wraplength=390).pack(
            fill="x", pady=(4, 0)
        )

    def _build_slider(self, parent: tk.Frame, field: dict[str, Any], initial_value: float) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=field["label"], width=23, anchor="w").pack(side="left")

        variable = tk.DoubleVar(value=initial_value)
        self.slider_vars[field["key"]] = variable
        scale = tk.Scale(
            row,
            from_=field["min"],
            to=field["max"],
            resolution=field["resolution"],
            orient="horizontal",
            variable=variable,
            showvalue=False,
            length=180,
            command=lambda _value: self._apply_adjustments_from_controls(),
        )
        scale.pack(side="left", fill="x", expand=True)

        entry = tk.Entry(row, textvariable=variable, width=8)
        entry.pack(side="left", padx=(6, 0))
        entry.bind("<Return>", lambda _event: self._apply_adjustments_from_controls())
        entry.bind("<FocusOut>", lambda _event: self._apply_adjustments_from_controls())

    @staticmethod
    def _flatten_adjustments(adjustments: dict[str, float | list[float]]) -> dict[str, float]:
        joint_shifters = adjustments["joint_shifters"]
        mcp_offset = adjustments["mcp_offset"]
        assert isinstance(joint_shifters, list)
        assert isinstance(mcp_offset, list)
        return {
            "theta_rad": float(adjustments["theta_rad"]),
            "distance_shift": float(adjustments["distance_shift"]),
            "palm_horizontal_offset_px": float(adjustments["palm_horizontal_offset_px"]),
            "dip_shifter": float(joint_shifters[0]),
            "pip_shifter": float(joint_shifters[1]),
            "mcp_offset_x": float(mcp_offset[0]),
            "mcp_offset_y": float(mcp_offset[1]),
        }

    def _adjustments_from_controls(self) -> dict[str, float | list[float]]:
        values = {key: float(variable.get()) for key, variable in self.slider_vars.items()}
        return {
            "theta_rad": values["theta_rad"],
            "distance_shift": values["distance_shift"],
            "joint_shifters": [values["dip_shifter"], values["pip_shifter"]],
            "mcp_offset": [values["mcp_offset_x"], values["mcp_offset_y"]],
            "palm_horizontal_offset_px": values["palm_horizontal_offset_px"],
        }

    def _apply_adjustments_from_controls(self) -> None:
        if len(self.slider_vars) != len(SLIDER_FIELDS):
            return
        try:
            self.processor.update_adjustments(self._adjustments_from_controls())
            self.status_var.set("Adjusted values are active in the live preview.")
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(f"Invalid adjustment value: {exc}")

    def reset_adjustments(self) -> None:
        flat_values = self._flatten_adjustments(self.initial_adjustments)
        for key, value in flat_values.items():
            self.slider_vars[key].set(value)
        self.processor.update_adjustments(self.initial_adjustments)
        self.status_var.set("Reset to the values loaded from the source configuration.")

    def save_parameters(self) -> None:
        try:
            adjustments = self._adjustments_from_controls()
            self.processor.update_adjustments(adjustments)
            saved_path = save_adjustments_snapshot(self.config_path, adjustments)
        except (OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Save Parameters Error", str(exc))
            self.status_var.set(f"Failed to save parameters: {exc}")
            return

        print("Saved tuned parameters:", flush=True)
        print(f"  {saved_path}", flush=True)
        print("", flush=True)
        print("Source configuration:", flush=True)
        print(f"  {self.config_path}", flush=True)
        print("", flush=True)
        print("The source configuration was not modified.", flush=True)
        print("To apply these parameters, copy processing.adjustments from the saved file", flush=True)
        print("into processing.adjustments in the source configuration.", flush=True)
        self.status_var.set(f"Saved parameters: {saved_path.name}")

    def _open_camera(self) -> None:
        camera_index = int(self.camera_config.get("index", 0))
        backend = resolve_backend(self.camera_config.get("backend"))
        cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera not available: index={camera_index}")
        apply_camera_settings(cap, self.camera_config)
        self.cap = cap
        self.calibration = setup_undistortion_from_config(self.camera_config, log_prefix="[joint-tuner]")

    def _schedule_next_frame(self, delay_ms: int = 1) -> None:
        if not self._closing:
            self._after_id = self.root.after(delay_ms, self._update_frame)

    def _update_frame(self) -> None:
        if self._closing or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_var.set("Failed to read a camera frame. Retrying...")
            self._schedule_next_frame(100)
            return

        try:
            frame = undistort_frame(frame, self.calibration)
            _, overlay = self.processor.process_frame(frame, self.frame_index)
            display_frame = _resize_preview(overlay, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
            self.photo = _to_photo_image_bgr(display_frame)
            self.preview_label.configure(image=self.photo, text="")
            self.frame_index += 1
        except Exception as exc:
            self.status_var.set(f"Frame processing failed: {exc}")
            self._schedule_next_frame(250)
            return

        self._schedule_next_frame()

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DLC joint geometry parameters with a realtime camera preview."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_DLC_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    JointGeometryTunerApp(
        root,
        config_path=Path(args.config),
        camera_config_path=Path(args.camera_config),
    )
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        messagebox.showerror("Joint Geometry Tuner Error", str(exc))
        raise
