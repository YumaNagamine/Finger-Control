from __future__ import annotations

from servo.core.session import ServoSession


class IDService:
    def __init__(self, session: ServoSession) -> None:
        self._session = session

    def scan(self, start_id: int, end_id: int) -> None:
        self._session.scan_ids(int(start_id), int(end_id))

    def change_id(self, old_id: int, new_id: int) -> None:
        self._session.change_id(int(old_id), int(new_id))

    def reset_ids(self) -> None:
        self._session.reset_ids()

    def list_ids(self) -> None:
        self._session.list_ids()
