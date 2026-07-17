"""
video_processor.py

Processes a video file frame-by-frame using the YOLO safety detection model,
stores detection data in PostgreSQL, and saves annotated frames as JPEGs.

Usage:
    python video_processor.py --video path/to/video.mp4 --location "Zone A"
    python video_processor.py --video path/to/video.mp4 --interval 15
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

from inference import SafetyEyeModel
from config import (
    FRAME_SAMPLE_INTERVAL,
    FRAMES_OUTPUT_DIR,
    MODEL_PATH,
    SEVERITY_MAP,
    VIOLATION_CLASSES,
)
from database import SessionLocal, create_tables
from models import Detection, DetectionSession, VideoSource, Violation


def process_video(video_path: str, location: str = None, interval: int = None, on_frame=None, cancel_event=None):
    interval = interval or FRAME_SAMPLE_INTERVAL
    frames_dir = Path(FRAMES_OUTPUT_DIR)
    frames_dir.mkdir(exist_ok=True)

    model = SafetyEyeModel(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    db = SessionLocal()
    try:
        # Register the video source
        source = VideoSource(
            name=Path(video_path).name,
            file_path=str(Path(video_path).resolve()),
            location=location,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            status="processing",
        )
        db.add(source)
        db.commit()
        db.refresh(source)

        print(f"Processing: {source.name}")
        print(f"  Frames: {total_frames} @ {fps:.1f}fps | Duration: {duration_sec:.1f}s")
        print(f"  Sampling every {interval} frames")

        frame_number = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if cancel_event and cancel_event.is_set():
                break

            if frame_number % interval != 0:
                frame_number += 1
                continue

            timestamp_in_video = frame_number / fps

            # Run ONNX inference
            t0 = time.perf_counter()
            raw_detections = model(frame)
            inference_ms = (time.perf_counter() - t0) * 1000

            detections_data = []
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox"]
                area = (x2 - x1) * (y2 - y1)
                is_viol = det["class_name"] in VIOLATION_CLASSES
                detections_data.append({
                    "class_id":     det["class_id"],
                    "class_name":   det["class_name"],
                    "confidence":   det["confidence"],
                    "bbox_x1": x1, "bbox_y1": y1,
                    "bbox_x2": x2, "bbox_y2": y2,
                    "bbox_area":    area,
                    "bbox_center_x": (x1 + x2) / 2,
                    "bbox_center_y": (y1 + y2) / 2,
                    "is_violation": is_viol,
                })

            person_count = sum(1 for d in detections_data if d["class_name"] == "Person")
            violation_count = sum(1 for d in detections_data if d["is_violation"])
            total_det = len(detections_data)

            if total_det > 0:
                compliance_rate = ((total_det - violation_count) / total_det) * 100
            else:
                compliance_rate = 100.0

            # Save annotated frame
            annotated = model.plot(frame, raw_detections)   # returns BGR numpy
            frame_filename = f"{source.id}_f{frame_number:07d}.jpg"
            frame_path = str(frames_dir / frame_filename)
            cv2.imwrite(frame_path, annotated)

            if on_frame:
                on_frame(
                    source_id=source.id,
                    frame_path=frame_path,
                    frame_number=frame_number,
                    total_frames=total_frames,
                )

            # Write detection session
            session_obj = DetectionSession(
                video_source_id=source.id,
                frame_number=frame_number,
                timestamp_in_video=timestamp_in_video,
                captured_at=datetime.utcnow(),
                total_detections=total_det,
                person_count=person_count,
                violation_count=violation_count,
                compliance_rate=compliance_rate,
                inference_time_ms=inference_ms,
                annotated_frame_path=frame_path,
            )
            db.add(session_obj)
            db.flush()  # get session_obj.id without committing

            # Write individual detections
            for d in detections_data:
                det = Detection(
                    session_id=session_obj.id,
                    **d,
                )
                db.add(det)
                db.flush()

                # Write violation record if applicable
                if d["is_violation"]:
                    viol = Violation(
                        session_id=session_obj.id,
                        detection_id=det.id,
                        video_source_id=source.id,
                        violation_type=d["class_name"],
                        severity=SEVERITY_MAP.get(d["class_name"], "medium"),
                        frame_timestamp=session_obj.captured_at,
                    )
                    db.add(viol)

            db.commit()
            processed += 1

            if processed % 10 == 0:
                pct = (frame_number / total_frames) * 100
                print(f"  [{pct:.1f}%] frame {frame_number} | "
                      f"detections={total_det} violations={violation_count} "
                      f"compliance={compliance_rate:.1f}%")

            frame_number += 1

        # Mark source as done or cancelled
        source.status = "cancelled" if (cancel_event and cancel_event.is_set()) else "done"
        source.processed_at = datetime.utcnow()
        db.commit()

        print(f"\nDone. Processed {processed} frames from {source.name}.")
        print(f"Annotated frames saved to: {frames_dir.resolve()}")

    except Exception as e:
        source.status = "failed"
        db.commit()
        raise e
    finally:
        cap.release()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Eye video processor")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--location", default=None, help="Camera/zone location label")
    parser.add_argument(
        "--interval", type=int, default=None,
        help=f"Process every Nth frame (default: {FRAME_SAMPLE_INTERVAL})"
    )
    args = parser.parse_args()

    create_tables()
    process_video(args.video, location=args.location, interval=args.interval)
