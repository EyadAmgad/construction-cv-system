-- Construction CV System — PostgreSQL Schema

CREATE TABLE IF NOT EXISTS analysis_runs (
    id          SERIAL PRIMARY KEY,
    video_path  TEXT        NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    fps         REAL,
    total_frames INTEGER
);

CREATE TABLE IF NOT EXISTS detections (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER     NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    frame           INTEGER     NOT NULL,
    equipment_id    TEXT        NOT NULL,
    label           TEXT        NOT NULL,
    confidence      REAL        NOT NULL,
    state           TEXT        NOT NULL,
    active_sec      REAL        NOT NULL,
    idle_sec        REAL        NOT NULL,
    utilization_pct REAL        NOT NULL
);

-- Kafka consumer writes into this table (one row per Kafka message)
CREATE TABLE IF NOT EXISTS equipment_events (
    id                    SERIAL PRIMARY KEY,
    received_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    frame_id              INTEGER     NOT NULL,
    equipment_id          TEXT        NOT NULL,
    equipment_class       TEXT        NOT NULL,
    video_timestamp       TEXT        NOT NULL,
    current_state         TEXT        NOT NULL,
    current_activity      TEXT        NOT NULL,
    total_tracked_seconds REAL        NOT NULL,
    total_active_seconds  REAL        NOT NULL,
    total_idle_seconds    REAL        NOT NULL,
    utilization_percent   REAL        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_run_id        ON detections(run_id);
CREATE INDEX IF NOT EXISTS idx_detections_equipment_id  ON detections(equipment_id);
CREATE INDEX IF NOT EXISTS idx_events_equipment_id      ON equipment_events(equipment_id);
CREATE INDEX IF NOT EXISTS idx_events_frame_id          ON equipment_events(frame_id);
