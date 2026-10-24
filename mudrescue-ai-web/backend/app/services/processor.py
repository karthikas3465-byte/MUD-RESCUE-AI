"""CPU-friendly sampled inference and mission progress."""

from __future__ import annotations

import math
import threading
import time
import uuid
from pathlib import Path

import cv2

from .. import database as db
from .detector import PersonDetector
from .media import MediaError, annotate, decode_image, inspect_video


INFERENCE_LOCK = threading.Lock()


def _record_boxes(mission_id: str, frame_no: int, seconds: float, boxes, evidence: Path, model: str,
                  source_filename: str | None = None) -> None:
    for box in boxes:
        db.add_detection({
            "id": uuid.uuid4().hex,
            "mission_id": mission_id,
            "frame_number": frame_no,
            "timestamp_seconds": round(seconds, 3),
            "confidence": round(box.confidence, 5),
            "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
            "track_id": box.track_id,
            "model_version": model,
            "source_filename": source_filename,
            "evidence_path": str(evidence),
            "review_status": "Needs review",
            "reviewer_notes": "",
            "priority": "Standard review",
            "created_at": db.now(),
        })


def process_image(mission_id: str) -> None:
    started = time.perf_counter()
    mission = db.mission(mission_id)
    source = Path(mission["source_path"])
    run_dir = db.DATA_DIR / "evidence" / mission_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with INFERENCE_LOCK:
            frame = decode_image(source)
            detector = PersonDetector()
            boxes = detector.detect(frame, mission["confidence_threshold"], high_res_retry=True)
            evidence = run_dir / "frame_00000000.jpg"
            if not cv2.imwrite(str(evidence), annotate(frame, boxes, 0, 0, "IMAGE ANALYSIS")):
                raise MediaError("Could not write annotated image.")
            _record_boxes(mission_id, 0, 0, boxes, evidence, detector.version)
            h, w = frame.shape[:2]
            db.execute(
                "UPDATE missions SET status='complete', width=?, height=?, frame_count=1, "
                "frames_analyzed=1, frames_with_detections=?, current_frame_path=?, "
                "current_frame_number=0, current_timestamp=0, progress=1, processing_seconds=? WHERE id=?",
                (w, h, int(bool(boxes)), str(evidence), round(time.perf_counter() - started, 2), mission_id),
            )
    except Exception as exc:
        db.execute(
            "UPDATE missions SET status='failed', error=?, processing_seconds=? WHERE id=?",
            (f"{type(exc).__name__}: {exc}", round(time.perf_counter() - started, 2), mission_id),
        )
        raise


def process_image_set(mission_id: str, names: list[str]) -> None:
    """Run genuine inference on independent stills; no invented video time or tracking."""
    started = time.perf_counter()
    mission = db.mission(mission_id)
    source_dir = Path(mission["source_path"]).parent
    run_dir = db.DATA_DIR / "evidence" / mission_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with INFERENCE_LOCK:
            db.execute("UPDATE missions SET status='processing',frame_count=? WHERE id=?", (len(names), mission_id))
            detector = PersonDetector()
            analyzed = hits = 0
            stopped = False
            for index, original_name in enumerate(names, start=1):
                while True:
                    state = db.one("SELECT pause_requested,cancel_requested FROM missions WHERE id=?", (mission_id,))
                    if state["cancel_requested"]:
                        stopped = True
                        break
                    if not state["pause_requested"]:
                        break
                    db.execute("UPDATE missions SET status='paused' WHERE id=?", (mission_id,))
                    time.sleep(0.15)
                if stopped:
                    break
                db.execute("UPDATE missions SET status='processing' WHERE id=?", (mission_id,))
                frame = decode_image(source_dir / f"{index:03d}.jpg")
                boxes = detector.detect(frame, mission["confidence_threshold"], high_res_retry=True)
                evidence = run_dir / f"image_{index:03d}.jpg"
                annotated = annotate(frame, boxes, index, None, "DRIVE STILL IMAGE - SYNTHETIC")
                if not cv2.imwrite(str(evidence), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise MediaError("Could not save annotated still image.")
                _record_boxes(mission_id, index, 0, boxes, evidence, detector.version, original_name)
                analyzed += 1
                hits += int(bool(boxes))
                height, width = frame.shape[:2]
                db.execute(
                    "UPDATE missions SET width=?,height=?,frames_analyzed=?,frames_with_detections=?,"
                    "current_frame_path=?,current_frame_number=?,current_timestamp=NULL,progress=?,processing_seconds=? WHERE id=?",
                    (width, height, analyzed, hits, str(evidence), index, analyzed / len(names),
                     round(time.perf_counter() - started, 2), mission_id),
                )
            db.execute("UPDATE missions SET status=?,processing_seconds=? WHERE id=?",
                       ("stopped" if stopped else "complete", round(time.perf_counter() - started, 2), mission_id))
    except Exception as exc:
        db.execute("UPDATE missions SET status='failed',error=?,processing_seconds=? WHERE id=?",
                   (f"{type(exc).__name__}: {exc}", round(time.perf_counter() - started, 2), mission_id))


def _set_repeated_priorities(mission_id: str) -> None:
    with db.connect() as conn:
        conn.execute(
            """UPDATE detections SET priority='Repeated sighting'
               WHERE mission_id=? AND track_id IN (
                 SELECT track_id FROM detections
                 WHERE mission_id=? AND track_id IS NOT NULL
                 GROUP BY track_id HAVING COUNT(DISTINCT frame_number) >= 2
               )""",
            (mission_id, mission_id),
        )


def process_video(mission_id: str) -> None:
    started = time.perf_counter()
    mission = db.mission(mission_id)
    source = Path(mission["source_path"])
    run_dir = db.DATA_DIR / "evidence" / mission_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cap = None
    writer = None
    try:
        with INFERENCE_LOCK:
            info = inspect_video(source)
            db.execute(
                "UPDATE missions SET status='processing',width=?,height=?,fps=?,frame_count=?,duration=? WHERE id=?",
                (info.width, info.height, info.fps, info.frame_count, info.duration, mission_id),
            )
            detector = PersonDetector()
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise MediaError("Video could not be reopened for analysis.")
            stride = max(1, round(info.fps * mission["sample_interval"]))
            sampled_fps = min(info.fps, 1 / mission["sample_interval"])
            clip_path = run_dir / "sampled_review.webm"
            writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"VP80"), sampled_fps,
                                     (info.width, info.height))
            if not writer.isOpened():
                writer.release()
                writer = None
            frame_no = 0
            analyzed = 0
            hit_frames = 0
            any_track = False
            stopped = False
            progress = 0.0
            while True:
                if frame_no % stride == 0:
                    # Operator controls are checked before every sampled inference.
                    while True:
                        state = db.one("SELECT pause_requested,cancel_requested FROM missions WHERE id=?", (mission_id,))
                        if state["cancel_requested"]:
                            stopped = True
                            break
                        if not state["pause_requested"]:
                            break
                        db.execute("UPDATE missions SET status='paused' WHERE id=?", (mission_id,))
                        time.sleep(0.15)
                    if stopped:
                        break
                    db.execute("UPDATE missions SET status='processing' WHERE id=?", (mission_id,))
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_no % stride == 0:
                    pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                    seconds = pos_ms / 1000 if math.isfinite(pos_ms) and pos_ms > 0 else frame_no / info.fps
                    boxes = detector.detect(frame, mission["confidence_threshold"], bool(mission["tracking_requested"]))
                    annotated = annotate(
                        frame, boxes, frame_no, seconds,
                        "STILL-PHOTO DEMO" if mission["is_demo"] else "VIDEO ANALYSIS",
                    )
                    evidence = run_dir / f"frame_{frame_no:08d}.jpg"
                    if not cv2.imwrite(str(evidence), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                        raise MediaError("Could not save evidence frame.")
                    if writer is not None:
                        writer.write(annotated)
                    _record_boxes(mission_id, frame_no, seconds, boxes, evidence, detector.version)
                    analyzed += 1
                    hit_frames += int(bool(boxes))
                    any_track |= any(box.track_id is not None for box in boxes)
                    progress = min(.99, (frame_no + 1) / info.frame_count) if info.frame_count > 0 else 0.5
                    db.execute(
                        "UPDATE missions SET frames_analyzed=?,frames_with_detections=?,tracking_available=?,"
                        "current_frame_path=?,current_frame_number=?,current_timestamp=?,progress=?,processing_seconds=? WHERE id=?",
                        (analyzed, hit_frames, int(any_track), str(evidence), frame_no, round(seconds, 3),
                         progress, round(time.perf_counter() - started, 2), mission_id),
                    )
                frame_no += 1
            if analyzed == 0 and not stopped:
                raise MediaError("No readable frames were analyzed.")
            if writer is not None:
                writer.release()
                writer = None
            _set_repeated_priorities(mission_id)
            db.execute(
                "UPDATE missions SET status=?, progress=?, annotated_video_path=?, processing_seconds=? WHERE id=?",
                ("stopped" if stopped else "complete", min(.99, progress) if stopped else 1,
                 str(clip_path) if clip_path.is_file() and clip_path.stat().st_size > 0 else None,
                 round(time.perf_counter() - started, 2), mission_id),
            )
    except Exception as exc:
        db.execute(
            "UPDATE missions SET status='failed',error=?,processing_seconds=? WHERE id=?",
            (f"{type(exc).__name__}: {exc}", round(time.perf_counter() - started, 2), mission_id),
        )
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
