-- Safety Eye — PostgreSQL initialization
-- Runs automatically on first container start

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS video_sources (
    id           SERIAL PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    file_path    VARCHAR(500) NOT NULL,
    location     VARCHAR(255),
    fps          FLOAT,
    total_frames INTEGER,
    duration_sec FLOAT,
    processed_at TIMESTAMP,
    status       VARCHAR(50) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS detection_sessions (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_source_id      INTEGER REFERENCES video_sources(id),
    frame_number         INTEGER NOT NULL,
    timestamp_in_video   FLOAT NOT NULL,
    captured_at          TIMESTAMP DEFAULT NOW(),
    total_detections     INTEGER DEFAULT 0,
    person_count         INTEGER DEFAULT 0,
    violation_count      INTEGER DEFAULT 0,
    compliance_rate      FLOAT DEFAULT 100.0,
    inference_time_ms    FLOAT,
    annotated_frame_path VARCHAR(500)
);

CREATE TABLE IF NOT EXISTS detections (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID REFERENCES detection_sessions(id),
    class_id      INTEGER NOT NULL,
    class_name    VARCHAR(100) NOT NULL,
    confidence    FLOAT NOT NULL,
    bbox_x1       FLOAT NOT NULL,
    bbox_y1       FLOAT NOT NULL,
    bbox_x2       FLOAT NOT NULL,
    bbox_y2       FLOAT NOT NULL,
    bbox_area     FLOAT,
    bbox_center_x FLOAT,
    bbox_center_y FLOAT,
    is_violation  BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS violations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID REFERENCES detection_sessions(id),
    detection_id    UUID REFERENCES detections(id),
    video_source_id INTEGER REFERENCES video_sources(id),
    violation_type  VARCHAR(100) NOT NULL,
    severity        VARCHAR(20) NOT NULL,
    frame_timestamp TIMESTAMP NOT NULL,
    acknowledged    BOOLEAN DEFAULT FALSE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_detection_sessions_captured_at  ON detection_sessions(captured_at);
CREATE INDEX IF NOT EXISTS idx_detection_sessions_source       ON detection_sessions(video_source_id);
CREATE INDEX IF NOT EXISTS idx_detections_session              ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_class                ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp            ON violations(frame_timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_source               ON violations(video_source_id);
CREATE INDEX IF NOT EXISTS idx_violations_acknowledged         ON violations(acknowledged);
