from __future__ import annotations

import json
import os
from typing import Dict, List


class RecordService:
    def __init__(self, motion_files_dir: str = "recorded_motions_api") -> None:
        self.motion_files_dir = motion_files_dir

    def _ensure_dir(self) -> None:
        if not os.path.exists(self.motion_files_dir):
            os.makedirs(self.motion_files_dir)

    def list_motions(self) -> List[str]:
        if not os.path.exists(self.motion_files_dir):
            return []
        files = sorted([f for f in os.listdir(self.motion_files_dir) if f.endswith(".json")])
        return [f.replace(".json", "") for f in files]

    def save_motion_auto(self, frames: List[Dict]) -> str:
        if not frames:
            raise ValueError("No frames to save")

        self._ensure_dir()

        idx = 0
        while True:
            name = f"Motion {chr(65 + idx)}" if idx < 26 else f"Motion {idx}"
            filename = os.path.join(self.motion_files_dir, f"{name}.json")
            if not os.path.exists(filename):
                break
            idx += 1

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(frames, f)

        return filename

    def load_motion(self, motion_name: str) -> List[Dict]:
        filename = os.path.join(self.motion_files_dir, f"{motion_name}.json")
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)

    def delete_motion(self, motion_name: str) -> bool:
        filename = os.path.join(self.motion_files_dir, f"{motion_name}.json")
        if not os.path.exists(filename):
            return False
        os.remove(filename)
        return True
