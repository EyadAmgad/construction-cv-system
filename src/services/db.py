"""
Database helpers for the construction CV system.

Connection settings are read from environment variables with sensible defaults
that match the docker-compose.yml configuration.

  DB_HOST     (default: localhost)
  DB_PORT     (default: 5432)
  DB_NAME     (default: construction_cv)
  DB_USER     (default: cvuser)
  DB_PASSWORD (default: cvpass)
"""

import os
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "construction_cv"),
        user=os.getenv("DB_USER", "cvuser"),
        password=os.getenv("DB_PASSWORD", "cvpass"),
    )


def start_run(video_path: str, fps: float, total_frames: int) -> int:
    """Insert a new analysis_runs row and return its id."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO analysis_runs (video_path, fps, total_frames)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (video_path, fps, total_frames),
        )
        run_id = cur.fetchone()[0]
        conn.commit()
    return run_id


def finish_run(run_id: int) -> None:
    """Stamp finished_at on the run row."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE analysis_runs SET finished_at = %s WHERE id = %s",
            (datetime.now(timezone.utc), run_id),
        )
        conn.commit()


def insert_detections(run_id: int, rows: list[dict]) -> None:
    """
    Bulk-insert detection rows.
    Each dict must have: frame, equipment_id, label, confidence,
                         state, active_sec, idle_sec, utilization_pct
    """
    if not rows:
        return
    records = [
        (
            run_id,
            r["frame"],
            r["equipment_id"],
            r["label"],
            r["confidence"],
            r["state"],
            r["active_sec"],
            r["idle_sec"],
            r["utilization_pct"],
        )
        for r in rows
    ]
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO detections
              (run_id, frame, equipment_id, label, confidence,
               state, active_sec, idle_sec, utilization_pct)
            VALUES %s
            """,
            records,
        )
        conn.commit()
