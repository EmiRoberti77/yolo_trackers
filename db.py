import os
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence, Tuple


DEFAULT_DB_FILENAME = "tracking.sqlite3"


class TrackingDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_dir()
        self._initialize()

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video TEXT NOT NULL,
                    tracker TEXT NOT NULL,
                    model TEXT,
                    frame_idx INTEGER NOT NULL,
                    timestamp_ms REAL,
                    track_id INTEGER,
                    class_id INTEGER,
                    conf REAL,
                    x1 REAL,
                    y1 REAL,
                    x2 REAL,
                    y2 REAL,
                    cx REAL,
                    cy REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_track ON events (video, tracker, track_id);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_frame ON events (video, frame_idx);
                """
            )

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_event(
        self,
        *,
        video: str,
        tracker: str,
        model: Optional[str],
        frame_idx: int,
        timestamp_ms: Optional[float],
        track_id: Optional[int],
        class_id: Optional[int],
        conf: Optional[float],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (
                    video, tracker, model, frame_idx, timestamp_ms,
                    track_id, class_id, conf, x1, y1, x2, y2, cx, cy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video,
                    tracker,
                    model,
                    frame_idx,
                    timestamp_ms,
                    track_id,
                    class_id,
                    conf,
                    x1,
                    y1,
                    x2,
                    y2,
                    cx,
                    cy,
                ),
            )

    def log_events_bulk(self, rows: Sequence[Tuple]) -> None:
        """
        rows: sequence of tuples matching log_event insert order
        (video, tracker, model, frame_idx, timestamp_ms, track_id, class_id, conf, x1, y1, x2, y2)
        """
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO events (
                    video, tracker, model, frame_idx, timestamp_ms,
                    track_id, class_id, conf, x1, y1, x2, y2, cx, cy
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ( (? + ?) / 2.0 ),
                    ( (? + ?) / 2.0 )
                )
                """,
                rows,
            )


def compute_timestamp_ms(frame_idx: int, fps: float) -> float:
    if fps and fps > 0:
        return (frame_idx / fps) * 1000.0
    return float(frame_idx) * 33.3333


