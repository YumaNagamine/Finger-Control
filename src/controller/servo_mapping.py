"""Canonical tendon-to-servo mapping shared by controller tools."""

from __future__ import annotations


TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")

# Current ServoHardwareAPI uses physical STS3215 IDs 1-6.  Keep this mapping
# independent of GUI row order and legacy internal indexes.
TENDON_TO_SERVO_ID = {
    "FDP": 4,
    "FDS": 3,
    "ED": 1,
    "DI": 2,
    "PI": 6,
    "LUM": 5,
}
SERVO_IDS_BY_TENDON = tuple(TENDON_TO_SERVO_ID[tendon] for tendon in TENDONS)

# Names and encoder channels in physical-ID order used by the current API.
SERVO_IDS = (1, 2, 3, 4, 5, 6)
SERVO_NAMES_BY_ID = ("ED", "DI", "FDS", "FDP", "LUM", "PI")
ENCODER_SENSOR_IDS_BY_SERVO_ID = {
    servo_id: servo_id for servo_id in SERVO_IDS
}
