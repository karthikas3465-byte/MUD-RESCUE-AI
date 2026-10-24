"""Small SQLite incident log with one connection per operation."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MUDRESCUE_DATA_DIR", ROOT / ".runtime")).resolve()
DB_PATH = DATA_DIR / "mudrescue.sqlite3"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS missions (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL CHECK (media_type IN ('image','video')),
            is_demo INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source_path TEXT NOT NULL,
            annotated_video_path TEXT,
            current_frame_path TEXT,
            current_frame_number INTEGER,
            current_timestamp REAL,
            width INTEGER,
            height INTEGER,
            fps REAL,
            frame_count INTEGER,
            duration REAL,
            confidence_threshold REAL NOT NULL,
            sample_interval REAL NOT NULL,
            tracking_requested INTEGER NOT NULL,
            tracking_available INTEGER NOT NULL DEFAULT 0,
            frames_analyzed INTEGER NOT NULL DEFAULT 0,
            frames_with_detections INTEGER NOT NULL DEFAULT 0,
            processing_seconds REAL NOT NULL DEFAULT 0,
            progress REAL NOT NULL DEFAULT 0,
            pause_requested INTEGER NOT NULL DEFAULT 0,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            mission_id TEXT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
            frame_number INTEGER NOT NULL,
            timestamp_seconds REAL NOT NULL,
            confidence REAL NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            x2 INTEGER NOT NULL,
            y2 INTEGER NOT NULL,
            track_id INTEGER,
            model_version TEXT NOT NULL,
            source_filename TEXT,
            evidence_path TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'Needs review',
            reviewer_notes TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'Standard review',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_detections_mission ON detections(mission_id);
        CREATE INDEX IF NOT EXISTS idx_detections_review ON detections(review_status);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(missions)")}
        if "is_demo" not in columns:
            db.execute("ALTER TABLE missions ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0")
        detection_columns = {row[1] for row in db.execute("PRAGMA table_info(detections)")}
        if "source_filename" not in detection_columns:
            db.execute("ALTER TABLE detections ADD COLUMN source_filename TEXT")


def one(query: str, args: tuple = ()) -> dict | None:
    with connect() as db:
        row = db.execute(query, args).fetchone()
        return dict(row) if row else None


def all_rows(query: str, args: tuple = ()) -> list[dict]:
    with connect() as db:
        return [dict(row) for row in db.execute(query, args).fetchall()]


def execute(query: str, args: tuple = ()) -> None:
    with connect() as db:
        db.execute(query, args)


def mission(mission_id: str) -> dict | None:
    record = one("SELECT * FROM missions WHERE id=?", (mission_id,))
    if record is None:
        return None
    stats = one(
        "SELECT COUNT(*) AS detection_count, COUNT(DISTINCT CASE WHEN track_id IS NOT NULL THEN track_id END) AS unique_tracks,"
        " COALESCE(SUM(CASE WHEN review_status='Needs review' THEN 1 ELSE 0 END),0) AS pending_reviews"
        " FROM detections WHERE mission_id=?", (mission_id,),
    )
    record.update(stats or {})
    return record


def missions() -> list[dict]:
    rows = all_rows("SELECT id FROM missions ORDER BY created_at DESC")
    return [mission(row["id"]) for row in rows]


def add_detection(row: dict) -> None:
    keys = tuple(row)
    with connect() as db:
        db.execute(
            f"INSERT INTO detections ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",
            tuple(row.values()),
        )


def review(detection_id: str, status: str, notes: str) -> dict | None:
    with connect() as db:
        db.execute(
            "UPDATE detections SET review_status=?, reviewer_notes=? WHERE id=?",
            (status, notes, detection_id),
        )
        row = db.execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
        return dict(row) if row else None
