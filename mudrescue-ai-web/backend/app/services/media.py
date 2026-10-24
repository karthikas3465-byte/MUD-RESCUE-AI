"""Upload validation, sampling and evidence annotation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_VIDEO_BYTES = 350 * 1024 * 1024
MAX_VIDEO_SECONDS = 30 * 60


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


def validate_extension(filename: str, kind: str) -> str:
    suffix = Path(filename).suffix.lower()
    allowed = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
    if suffix not in allowed:
        raise MediaError(f"Unsupported {kind} type. Accepted: {', '.join(sorted(allowed))}")
    return suffix


def decode_image(path: Path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise MediaError("The image cannot be decoded or is corrupt.")
    h, w = frame.shape[:2]
    if h > 9000 or w > 9000 or h * w > 45_000_000:
        raise MediaError("Image dimensions exceed the prototype limit.")
    return frame


def inspect_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError("OpenCV could not open the video. Check its codec.")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            raise MediaError("The video has no readable frames or is corrupt.")
        h, w = frame.shape[:2]
        if not math.isfinite(fps) or fps <= 0:
            raise MediaError("Video frame rate is unavailable; timestamps cannot be trusted.")
        if w > 9000 or h > 9000 or w * h > 45_000_000:
            raise MediaError("Video dimensions exceed the prototype limit.")
        duration = frame_count / fps if frame_count > 0 else 0.0
        if duration > MAX_VIDEO_SECONDS:
            raise MediaError("Videos must be 30 minutes or shorter.")
        return VideoInfo(w, h, fps, frame_count, duration)
    finally:
        cap.release()


def timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    return f"{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def annotate(frame, boxes, frame_number: int, seconds: float | None, source_label: str = "VIDEO ANALYSIS"):
    """Draw true model boxes over the actual source frame."""
    image = frame.copy()
    height, width = image.shape[:2]
    thickness = max(2, round(min(width, height) / 280))
    scale = max(0.5, min(width, height) / 1250)
    red = (49, 59, 239)
    for box in boxes:
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), red, thickness)
        title = "POSSIBLE PERSON - REVIEW"
        detail = f"CONFIDENCE {box.confidence * 100:.1f}%"
        if box.track_id is not None:
            detail += f"  TRACK {box.track_id}"
        title_y = max(45, box.y1)
        label_w = min(width - box.x1, max(255, int(390 * scale)))
        cv2.rectangle(image, (box.x1, title_y - 42), (box.x1 + label_w, title_y), red, -1)
        cv2.putText(image, title, (box.x1 + 7, title_y - 23),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * .75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, detail, (box.x1 + 7, title_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * .67, (255, 255, 255), 1, cv2.LINE_AA)
    # The small HUD is evidence context, including on frames with no detections.
    hud = (f"{source_label}  |  IMAGE {frame_number}" if seconds is None
           else f"{source_label}  |  FRAME {frame_number}  |  {timestamp(seconds)}")
    cv2.rectangle(image, (0, height - 38), (min(width, max(350, int(660 * scale))), height), (17, 29, 38), -1)
    cv2.putText(image, hud, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                scale * .75, (240, 244, 247), 1, cv2.LINE_AA)
    return image
