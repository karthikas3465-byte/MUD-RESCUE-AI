"""YOLO11n baseline. Only model outputs become detection records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int | None = None


class ModelLoadError(RuntimeError):
    pass


def _deduplicate_tiles(boxes: list[PersonBox]) -> list[PersonBox]:
    """Merge overlapping predictions of the same candidate from adjacent tiles."""
    kept = []
    for box in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        area = max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)
        duplicate = False
        for prior in kept:
            width = max(0, min(box.x2, prior.x2) - max(box.x1, prior.x1))
            height = max(0, min(box.y2, prior.y2) - max(box.y1, prior.y1))
            intersection = width * height
            prior_area = max(0, prior.x2 - prior.x1) * max(0, prior.y2 - prior.y1)
            if intersection / max(1, area + prior_area - intersection) > .5:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


class PersonDetector:
    """Keep this interface when evaluating a future aerial/occlusion model."""

    version = "YOLO11n pretrained COCO"

    def __init__(self) -> None:
        bundled = Path(__file__).resolve().parents[3] / "models" / "yolo11n.pt"
        model_path = os.environ.get("MUDRESCUE_MODEL_PATH", str(bundled) if bundled.exists() else "yolo11n.pt")
        data_dir = Path(os.environ.get("MUDRESCUE_DATA_DIR", Path(__file__).resolve().parents[3] / ".runtime"))
        config_dir = data_dir / "ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        try:
            import torch
            from ultralytics import YOLO

            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.model = YOLO(model_path)
            self.person_class = next(
                (int(key) for key, value in self.model.names.items() if str(value).lower() == "person"),
                None,
            )
            if self.person_class is None:
                raise ValueError("Model does not contain a person class")
        except Exception as exc:
            raise ModelLoadError(f"Could not load person model '{model_path}': {exc}") from exc

    def detect(self, frame, threshold: float, track: bool = False,
               high_res_retry: bool = False) -> list[PersonBox]:
        options = dict(classes=[self.person_class], conf=threshold, device=self.device, verbose=False)
        result = (
            self.model.track(frame, persist=True, tracker="bytetrack.yaml", **options)[0]
            if track else self.model.predict(frame, **options)[0]
        )
        if high_res_retry and not track and (result.boxes is None or len(result.boxes) == 0):
            if max(frame.shape[:2]) >= 1280:
                result = self.model.predict(frame, imgsz=1280, **options)[0]
            else:
                result = self.model.predict(frame, augment=True, **options)[0]
        if high_res_retry and not track and (result.boxes is None or len(result.boxes) == 0):
            height, width = frame.shape[:2]
            tile_width = min(900, max(1, round(width * .625)))
            tile_height = min(900, max(1, round(height * .625)))
            tile_boxes = []
            for y in sorted({0, height - tile_height}):
                for x in sorted({0, width - tile_width}):
                    tile = frame[y:y + tile_height, x:x + tile_width]
                    tiled = self.model.predict(tile, imgsz=640, **options)[0]
                    if tiled.boxes is None:
                        continue
                    for xyxy, score in zip(tiled.boxes.xyxy.cpu().numpy(),
                                           tiled.boxes.conf.cpu().numpy()):
                        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                        tile_boxes.append(PersonBox(
                            max(0, min(x + x1, width - 1)), max(0, min(y + y1, height - 1)),
                            max(0, min(x + x2, width - 1)), max(0, min(y + y2, height - 1)),
                            float(score),
                        ))
            if tile_boxes:
                return _deduplicate_tiles(tile_boxes)
        if result.boxes is None or len(result.boxes) == 0:
            return []
        coordinates = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if track and result.boxes.id is not None else []
        height, width = frame.shape[:2]
        boxes = []
        for index, (xyxy, score) in enumerate(zip(coordinates, scores)):
            x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
            boxes.append(PersonBox(
                max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)), max(0, min(y2, height - 1)),
                float(score), ids[index] if index < len(ids) else None,
            ))
        return boxes
