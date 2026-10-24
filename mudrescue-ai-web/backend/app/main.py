"""Local-only FastAPI service for real model inference and review."""

from __future__ import annotations

import csv
import io
import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database as db
from .services.media import (
    IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, VIDEO_EXTENSIONS,
    MediaError, decode_image, inspect_video, validate_extension,
)
from .services.processor import process_image, process_image_set, process_video


ROOT = Path(__file__).resolve().parents[2]
DEMO_CLIP = ROOT / "demo" / "sample_review_clip.mp4"
DRIVE_SET = ROOT / "demo" / "drive_test_images"
REVIEW_STATUSES = {
    "Needs review", "Confirmed visible person", "False detection", "Investigate on site"
}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="MudRescue AI", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


class ReviewUpdate(BaseModel):
    review_status: str
    reviewer_notes: str = Field(default="", max_length=1000)


class Control(BaseModel):
    action: str


def _mission_or_404(mission_id: str) -> dict:
    mission = db.mission(mission_id)
    if mission is None:
        raise HTTPException(404, "Mission not found.")
    return mission


def _make_mission(
    mission_id: str, filename: str, media_type: str, source: Path,
    confidence: float, interval: float, tracking: bool, is_demo: bool = False,
) -> None:
    db.execute(
        "INSERT INTO missions (id,filename,media_type,status,created_at,source_path,"
        "confidence_threshold,sample_interval,tracking_requested,is_demo) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (mission_id, Path(filename).name, media_type, "queued", db.now(), str(source),
         confidence, interval, int(tracking), int(is_demo)),
    )


def _validate_settings(confidence: float, interval: float) -> None:
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(422, "Confidence threshold must be between 0.05 and 0.95.")
    if not 0.2 <= interval <= 10:
        raise HTTPException(422, "Sampling interval must be between 0.2 and 10 seconds.")


async def _save_upload(upload: UploadFile, path: Path, max_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Upload exceeds {max_bytes // 1024 // 1024} MB limit.")
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.get("/api/health")
def health():
    model = ROOT / "models" / "yolo11n.pt"
    return {
        "status": "ok", "inference": "local", "model": "YOLO11n pretrained COCO",
        "bundled_model_available": model.is_file(),
        "demo_clip_available": DEMO_CLIP.is_file(),
        "drive_image_set_available": (DRIVE_SET / "manifest.json").is_file(),
    }


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(.25),
):
    _validate_settings(confidence_threshold, 1)
    try:
        suffix = validate_extension(file.filename or "", "image")
    except MediaError as exc:
        raise HTTPException(415, str(exc)) from exc
    mission_id = uuid.uuid4().hex
    source = db.DATA_DIR / "uploads" / f"{mission_id}{suffix}"
    await _save_upload(file, source, MAX_IMAGE_BYTES)
    try:
        await run_in_threadpool(decode_image, source)
    except MediaError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    _make_mission(mission_id, file.filename or "image", "image", source, confidence_threshold, 1, False)
    try:
        await run_in_threadpool(process_image, mission_id)
    except Exception as exc:
        raise HTTPException(503, f"Image analysis failed: {exc}") from exc
    return db.mission(mission_id)


@app.post("/api/analyze/video", status_code=202)
async def analyze_video(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(.25),
    sample_interval: float = Form(1.0),
    tracking: bool = Form(True),
):
    _validate_settings(confidence_threshold, sample_interval)
    try:
        suffix = validate_extension(file.filename or "", "video")
    except MediaError as exc:
        raise HTTPException(415, str(exc)) from exc
    mission_id = uuid.uuid4().hex
    source = db.DATA_DIR / "uploads" / f"{mission_id}{suffix}"
    await _save_upload(file, source, MAX_VIDEO_BYTES)
    try:
        info = await run_in_threadpool(inspect_video, source)
    except MediaError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    _make_mission(mission_id, file.filename or "video", "video", source,
                  confidence_threshold, sample_interval, tracking)
    db.execute(
        "UPDATE missions SET width=?,height=?,fps=?,frame_count=?,duration=? WHERE id=?",
        (info.width, info.height, info.fps, info.frame_count, info.duration, mission_id),
    )
    threading.Thread(target=process_video, args=(mission_id,), daemon=True).start()
    return db.mission(mission_id)


@app.post("/api/analyze/demo", status_code=202)
def analyze_demo(
    confidence_threshold: float = Form(.25),
    sample_interval: float = Form(1.0),
    tracking: bool = Form(True),
):
    _validate_settings(confidence_threshold, sample_interval)
    if not DEMO_CLIP.is_file():
        raise HTTPException(404, "Bundled demo clip is unavailable.")
    mission_id = uuid.uuid4().hex
    source = db.DATA_DIR / "uploads" / f"{mission_id}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEMO_CLIP, source)
    info = inspect_video(source)
    _make_mission(mission_id, "sample_review_clip.mp4", "video", source,
                  confidence_threshold, sample_interval, tracking, is_demo=True)
    db.execute(
        "UPDATE missions SET width=?,height=?,fps=?,frame_count=?,duration=? WHERE id=?",
        (info.width, info.height, info.fps, info.frame_count, info.duration, mission_id),
    )
    threading.Thread(target=process_video, args=(mission_id,), daemon=True).start()
    return db.mission(mission_id)


@app.get("/api/missions")
def list_missions():
    return db.missions()


@app.post("/api/analyze/drive-set", status_code=202)
def analyze_drive_set(confidence_threshold: float = Form(.25)):
    """Analyze selected user-supplied Drive still images with original names preserved."""
    _validate_settings(confidence_threshold, 1)
    manifest_path = DRIVE_SET / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "The supplied Drive still-image set is unavailable.")
    names = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    if not names:
        raise HTTPException(404, "The supplied Drive still-image set is empty.")
    mission_id = uuid.uuid4().hex
    source_dir = db.DATA_DIR / "uploads" / mission_id
    source_dir.mkdir(parents=True, exist_ok=True)
    for index, _name in enumerate(names, start=1):
        source_image = DRIVE_SET / f"{index:03d}.jpg"
        if not source_image.is_file():
            raise HTTPException(404, f"Supplied image {index} is unavailable.")
        shutil.copyfile(source_image, source_dir / f"{index:03d}.jpg")
    _make_mission(mission_id, "Supplied Drive labelled test images", "image",
                  source_dir / "001.jpg", confidence_threshold, 1, False, is_demo=True)
    db.execute("UPDATE missions SET is_demo=2,frame_count=? WHERE id=?", (len(names), mission_id))
    threading.Thread(target=process_image_set, args=(mission_id, names), daemon=True).start()
    return db.mission(mission_id)


@app.get("/api/missions/{mission_id}")
def get_mission(mission_id: str):
    return _mission_or_404(mission_id)


@app.post("/api/missions/{mission_id}/control")
def control_mission(mission_id: str, control: Control):
    mission = _mission_or_404(mission_id)
    if (mission["media_type"] != "video" and mission["is_demo"] != 2) or mission["status"] not in {"queued", "processing", "paused"}:
        raise HTTPException(409, "This mission is not processing.")
    if control.action == "pause":
        db.execute("UPDATE missions SET pause_requested=1 WHERE id=?", (mission_id,))
    elif control.action == "resume":
        db.execute("UPDATE missions SET pause_requested=0 WHERE id=?", (mission_id,))
    elif control.action == "stop":
        db.execute("UPDATE missions SET cancel_requested=1,pause_requested=0 WHERE id=?", (mission_id,))
    else:
        raise HTTPException(422, "Action must be pause, resume, or stop.")
    return db.mission(mission_id)


@app.get("/api/detections")
def list_detections(
    mission_id: str | None = None,
    review_status: str | None = None,
    min_confidence: float = Query(0, ge=0, le=1),
):
    clauses = ["confidence>=?"]
    args: list = [min_confidence]
    if mission_id:
        clauses.append("mission_id=?")
        args.append(mission_id)
    if review_status:
        clauses.append("review_status=?")
        args.append(review_status)
    return db.all_rows(
        "SELECT * FROM detections WHERE " + " AND ".join(clauses)
        + " ORDER BY created_at DESC, frame_number DESC", tuple(args),
    )


@app.get("/api/detections/{detection_id}")
def get_detection(detection_id: str):
    record = db.one("SELECT * FROM detections WHERE id=?", (detection_id,))
    if record is None:
        raise HTTPException(404, "Detection not found.")
    return record


@app.patch("/api/detections/{detection_id}")
def update_detection(detection_id: str, update: ReviewUpdate):
    if update.review_status not in REVIEW_STATUSES:
        raise HTTPException(422, "Invalid review status.")
    record = db.review(detection_id, update.review_status, update.reviewer_notes.strip())
    if record is None:
        raise HTTPException(404, "Detection not found.")
    return record


@app.get("/api/missions/{mission_id}/media/{kind}")
def mission_media(mission_id: str, kind: str):
    mission = _mission_or_404(mission_id)
    fields = {"source": "source_path", "frame": "current_frame_path", "clip": "annotated_video_path"}
    if kind not in fields:
        raise HTTPException(404, "Media type not found.")
    path = Path(mission[fields[kind]]) if mission[fields[kind]] else None
    if path is None or not path.is_file() or not path.resolve().is_relative_to(db.DATA_DIR):
        raise HTTPException(404, "Media is unavailable.")
    return FileResponse(path, filename=path.name if kind == "clip" else None)


@app.get("/api/missions/{mission_id}/source-frame/{frame_number}")
def image_set_source(mission_id: str, frame_number: int):
    mission = _mission_or_404(mission_id)
    if mission["is_demo"] != 2 or frame_number < 1 or frame_number > (mission["frame_count"] or 0):
        raise HTTPException(404, "Source image is unavailable.")
    path = Path(mission["source_path"]).parent / f"{frame_number:03d}.jpg"
    if not path.is_file() or not path.resolve().is_relative_to(db.DATA_DIR):
        raise HTTPException(404, "Source image is unavailable.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/detections/{detection_id}/evidence")
def evidence(detection_id: str):
    record = get_detection(detection_id)
    path = Path(record["evidence_path"])
    if not path.is_file() or not path.resolve().is_relative_to(db.DATA_DIR):
        raise HTTPException(404, "Evidence image is unavailable.")
    return FileResponse(path, media_type="image/jpeg")


def _summary(mission_id: str | None = None) -> dict:
    missions = [db.mission(mission_id)] if mission_id else db.missions()
    missions = [m for m in missions if m]
    detections = db.all_rows(
        "SELECT * FROM detections WHERE mission_id=? ORDER BY timestamp_seconds" if mission_id
        else "SELECT * FROM detections ORDER BY created_at DESC",
        (mission_id,) if mission_id else (),
    )
    return {
        "analyzed_missions": sum(m["status"] in {"complete", "stopped"} for m in missions),
        "total_detections": len(detections),
        "pending_reviews": sum(d["review_status"] == "Needs review" for d in detections),
        "unique_tracks": sum(m["unique_tracks"] for m in missions),
        "frames_analyzed": sum(m["frames_analyzed"] for m in missions),
        "missions": missions,
        "detections": detections,
    }


@app.get("/api/reports/summary")
def report_summary(mission_id: str | None = None):
    if mission_id:
        _mission_or_404(mission_id)
    return _summary(mission_id)


@app.get("/api/missions/{mission_id}/report.csv")
def report_csv(mission_id: str):
    _mission_or_404(mission_id)
    rows = db.all_rows(
        "SELECT * FROM detections WHERE mission_id=? ORDER BY timestamp_seconds,frame_number",
        (mission_id,),
    )
    columns = [
        "id", "mission_id", "frame_number", "timestamp_seconds", "confidence",
        "x1", "y1", "x2", "y2", "track_id", "model_version", "source_filename",
        "review_status", "reviewer_notes", "priority",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        content=stream.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="mudrescue-{mission_id[:8]}.csv"'},
    )


@app.get("/api/missions/{mission_id}/summary.txt")
def report_text(mission_id: str):
    mission = _mission_or_404(mission_id)
    rows = db.all_rows(
        "SELECT * FROM detections WHERE mission_id=? ORDER BY timestamp_seconds,frame_number",
        (mission_id,),
    )
    lines = [
        "MUDRESCUE AI — SEARCH REVIEW SUMMARY",
        "AI-assisted search prototype. Human verification required.",
        "",
        f"Source: {mission['filename']}",
        f"Bundled still-photo clip demo: {'Yes' if mission['is_demo'] == 1 else 'No'}",
        f"Supplied Drive synthetic still-image set: {'Yes' if mission['is_demo'] == 2 else 'No'}",
        f"Status: {mission['status']}",
        f"Model: YOLO11n pretrained COCO",
        f"Confidence threshold: {mission['confidence_threshold']}",
        ("Sample interval: N/A (independent still images)" if mission['is_demo'] == 2
         else f"Sample interval: {mission['sample_interval']} seconds"),
        f"Tracking requested: {bool(mission['tracking_requested'])}",
        f"Duration: {mission['duration'] if mission['duration'] is not None else 'N/A'} seconds",
        f"Frames analyzed: {mission['frames_analyzed']}",
        f"Frames with detections: {mission['frames_with_detections']}",
        f"Total detections: {len(rows)}",
        f"Estimated distinct tracks: {mission['unique_tracks'] if mission['tracking_available'] else 'Unavailable'}",
        f"Processing time: {mission['processing_seconds']} seconds",
        "",
        "Candidate detections:",
    ]
    for r in rows:
        location = (f"image {r['frame_number']} ({r['source_filename']})" if mission['is_demo'] == 2
                    else f"frame {r['frame_number']} at {r['timestamp_seconds']:.3f}s")
        lines.append(
            f"- {r['id'][:8]} {location}; confidence {r['confidence']:.3f}; "
            f"box ({r['x1']},{r['y1']},{r['x2']},{r['y2']}); "
            f"track {r['track_id'] if r['track_id'] is not None else 'N/A'}; "
            f"review {r['review_status']}; priority {r['priority']}"
        )
    if not rows:
        lines.append("- None. No detections does not establish that no person is present.")
    lines += [
        "",
        "Limitations: This baseline has not been validated for partial occlusion in landslides. "
        "RGB cannot reveal a fully buried person or determine life, injury or rescue need. "
        "Small targets, mud, debris, poor light, blur, camera angle, and occlusion can cause misses. "
        "Model confidence is not a survival probability. Track IDs are temporary associations. "
        "No geographic positions are inferred without reliable metadata and georeferencing.",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", headers={
        "Content-Disposition": f'attachment; filename="mudrescue-{mission_id[:8]}-summary.txt"'
    })


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
