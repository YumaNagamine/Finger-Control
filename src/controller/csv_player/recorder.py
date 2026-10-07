"""CSV recording for trajectory control and calibrated force results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from controller.servo_mapping import TENDONS


def _format(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.9g}"
    return value


class ControlTraceWriter:
    """Write one wide control/force row per control tick."""

    def __init__(self, path: Path, servo_ids_by_tendon: Sequence[int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self.servo_ids_by_tendon = tuple(int(value) for value in servo_ids_by_tendon)
        header = ["phase", "elapsed_s", "dt_s", "servo_telemetry_age_s"]
        for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon):
            header.extend(
                [
                    f"{tendon}_servo_id",
                    f"{tendon}_target_excursion_mm",
                    f"{tendon}_target_position",
                    f"{tendon}_actual_position",
                    f"{tendon}_position_error",
                    f"{tendon}_speed_command",
                    f"{tendon}_force_n",
                    f"{tendon}_torque_nmm",
                    f"{tendon}_angle_delta_deg",
                    f"{tendon}_servo_age_s",
                    f"{tendon}_encoder_age_s",
                    f"{tendon}_sync_age_s",
                    f"{tendon}_force_status",
                ]
            )
        self._writer.writerow(header)
        self._file.flush()

    def write(
        self,
        *,
        phase: str,
        elapsed_s: float,
        dt_s: float,
        servo_telemetry_age_s: float,
        target_excursions_mm: Sequence[float],
        target_positions: Mapping[int, float],
        actual_positions: Mapping[int, float],
        speeds: Mapping[int, int],
        force_readings: Mapping[int, Mapping[str, object]] | None = None,
    ) -> None:
        force_readings = force_readings or {}
        row: list[object] = [phase, _format(elapsed_s), _format(dt_s), _format(servo_telemetry_age_s)]
        for tendon_index, (tendon, servo_id) in enumerate(
            zip(TENDONS, self.servo_ids_by_tendon)
        ):
            force = force_readings.get(servo_id, {})
            target = target_positions[servo_id]
            actual = actual_positions[servo_id]
            row.extend(
                [
                    servo_id,
                    _format(target_excursions_mm[tendon_index]),
                    _format(target),
                    _format(actual),
                    _format(target - actual),
                    speeds.get(servo_id, 0),
                    _format(force.get("force_n")),
                    _format(force.get("torque_nmm")),
                    _format(force.get("angle_delta_deg")),
                    _format(force.get("servo_age_s")),
                    _format(force.get("encoder_age_s")),
                    _format(force.get("sync_age_s")),
                    force.get("status", ""),
                ]
            )
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
