from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

ID_EVENT_KEYWORDS = (
    "FOUND_ID",
    "SCAN_",
    "CHANGED_ID",
    "VERIFY",
    "NVS_",
    "EXECUTING",
    "CMD_",
    "SUCCESS",
    "FAIL",
)


@dataclass
class ParsedLine:
    kind: str
    line: str
    telemetry: Optional["TelemetryData"] = None


@dataclass
class TelemetryData:
    timestamp_ms: int
    positions: List[int]
    loads: List[int]
    speeds: Optional[List[int]]
    raw: str


def _parse_telemetry(parts: List[str], num_motors: int, line: str) -> Optional[TelemetryData]:
    if not parts or not parts[0].isdigit():
        return None

    if len(parts) == 1 + (num_motors * 2):
        ts = int(parts[0])
        positions: List[int] = []
        loads: List[int] = []
        for i in range(num_motors):
            base = 1 + (i * 2)
            positions.append(int(parts[base]))
            loads.append(int(parts[base + 1]))
        return TelemetryData(ts, positions, loads, None, line)

    if len(parts) == 1 + (num_motors * 3):
        ts = int(parts[0])
        positions = []
        loads = []
        speeds: List[int] = []
        for i in range(num_motors):
            base = 1 + (i * 3)
            positions.append(int(parts[base]))
            loads.append(int(parts[base + 1]))
            speeds.append(int(parts[base + 2]))
        return TelemetryData(ts, positions, loads, speeds, line)

    return None


def parse_line(line: str, num_motors: int) -> ParsedLine:
    stripped = line.strip()
    if not stripped:
        return ParsedLine(kind="empty", line="")

    if any(keyword in stripped for keyword in ID_EVENT_KEYWORDS):
        return ParsedLine(kind="id_event", line=stripped)

    parts = stripped.split(",")
    try:
        telemetry = _parse_telemetry(parts, num_motors, stripped)
    except (ValueError, IndexError):
        telemetry = None

    if telemetry is not None:
        return ParsedLine(kind="telemetry", line=stripped, telemetry=telemetry)

    return ParsedLine(kind="raw", line=stripped)
