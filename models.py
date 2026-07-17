import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Integer, String, Text, TIMESTAMP
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class VideoSource(Base):
    __tablename__ = "video_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    location = Column(String(255))           # e.g. "Zone A", "Main Entrance"
    fps = Column(Float)
    total_frames = Column(Integer)
    duration_sec = Column(Float)
    processed_at = Column(TIMESTAMP)
    status = Column(String(50), default="pending")  # pending/processing/done/failed

    sessions = relationship("DetectionSession", back_populates="video_source")
    violations = relationship("Violation", back_populates="video_source")


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_in_video = Column(Float, nullable=False)  # seconds from video start
    captured_at = Column(TIMESTAMP, default=datetime.utcnow)
    total_detections = Column(Integer, default=0)
    person_count = Column(Integer, default=0)
    violation_count = Column(Integer, default=0)
    compliance_rate = Column(Float, default=100.0)      # 0-100%
    inference_time_ms = Column(Float)
    annotated_frame_path = Column(String(500))           # path to saved JPEG

    video_source = relationship("VideoSource", back_populates="sessions")
    detections = relationship("Detection", back_populates="session")
    violations = relationship("Violation", back_populates="session")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("detection_sessions.id"), nullable=False)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    bbox_area = Column(Float)        # (x2-x1)*(y2-y1)
    bbox_center_x = Column(Float)    # (x1+x2)/2
    bbox_center_y = Column(Float)    # (y1+y2)/2
    is_violation = Column(Boolean, default=False)

    session = relationship("DetectionSession", back_populates="detections")
    violation = relationship("Violation", back_populates="detection", uselist=False)


class Violation(Base):
    __tablename__ = "violations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("detection_sessions.id"), nullable=False)
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False)
    video_source_id = Column(Integer, ForeignKey("video_sources.id"), nullable=False)
    violation_type = Column(String(100), nullable=False)  # no_helmet, no_goggle, etc.
    severity = Column(String(20), nullable=False)          # high / medium
    frame_timestamp = Column(TIMESTAMP, nullable=False)
    acknowledged = Column(Boolean, default=False)

    session = relationship("DetectionSession", back_populates="violations")
    detection = relationship("Detection", back_populates="violation")
    video_source = relationship("VideoSource", back_populates="violations")
