"""Canonical tendon-to-servo mapping shared by controller tools."""

from __future__ import annotations


TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")

# GUI motor order for servo IDs 0-5 is LU, PI, ED, DI, FDS, FDP.
# Controller name LUM corresponds to the GUI label LU; ED is already the GUI label.
TENDON_TO_SERVO_ID = {
    "FDP": 5,
    "FDS": 4,
    "ED": 2,
    "DI": 3,
    "PI": 1,
    "LUM": 0,
}
SERVO_IDS_BY_TENDON = tuple(TENDON_TO_SERVO_ID[tendon] for tendon in TENDONS)
