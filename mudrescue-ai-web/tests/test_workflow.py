"""Integration checks against the real bundled model and clip."""

from __future__ import annotations

import os
import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["MUDRESCUE_DATA_DIR"] = tempfile.mkdtemp(prefix="mudrescue-test-")
from app.main import DEMO_CLIP, app  # noqa: E402


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_01_health_and_invalid_media(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["bundled_model_available"])
        invalid_type = self.client.post(
            "/api/analyze/image", files={"file": ("test.txt", b"hello", "text/plain")}
        )
        self.assertEqual(invalid_type.status_code, 415)
        corrupt = self.client.post(
            "/api/analyze/video", files={"file": ("bad.mp4", b"not a video", "video/mp4")}
        )
        self.assertEqual(corrupt.status_code, 422)

    def test_02_model_load_error(self):
        cap = cv2.VideoCapture(str(DEMO_CLIP))
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        prior = os.environ.get("MUDRESCUE_MODEL_PATH")
        try:
            os.environ["MUDRESCUE_MODEL_PATH"] = str(ROOT / "models" / "missing-model.pt")
            response = self.client.post(
                "/api/analyze/image",
                files={"file": ("sample.jpg", encoded.tobytes(), "image/jpeg")},
            )
            self.assertEqual(response.status_code, 503)
            failed = [m for m in self.client.get("/api/missions").json() if m["status"] == "failed"]
            self.assertTrue(failed)
            self.assertIn("ModelLoadError", failed[0]["error"])
        finally:
            if prior is None:
                os.environ.pop("MUDRESCUE_MODEL_PATH", None)
            else:
                os.environ["MUDRESCUE_MODEL_PATH"] = prior

    def test_03_real_demo_review_and_exports(self):
        response = self.client.post("/api/analyze/demo", data={
            "confidence_threshold": "0.25", "sample_interval": "1.0", "tracking": "true"
        })
        self.assertEqual(response.status_code, 202)
        mission_id = response.json()["id"]
        for _ in range(120):
            mission = self.client.get(f"/api/missions/{mission_id}").json()
            if mission["status"] in {"complete", "failed", "stopped"}:
                break
            time.sleep(.5)
        self.assertEqual(mission["status"], "complete", mission["error"])
        self.assertEqual(mission["is_demo"], 1)
        self.assertEqual(mission["frames_analyzed"], 4)
        self.assertGreater(mission["detection_count"], 0)
        self.assertTrue(mission["annotated_video_path"].endswith(".webm"))
        records = self.client.get(f"/api/detections?mission_id={mission_id}").json()
        self.assertTrue(all(row["model_version"].startswith("YOLO11n") for row in records))
        self.assertTrue(all(0 < row["confidence"] <= 1 for row in records))
        self.assertEqual(self.client.get(f"/api/detections/{records[0]['id']}/evidence").status_code, 200)
        review = self.client.patch(
            f"/api/detections/{records[0]['id']}",
            json={"review_status": "Confirmed visible person", "reviewer_notes": "Visible in source frame"},
        )
        self.assertEqual(review.status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/detections/{records[0]['id']}").json()["reviewer_notes"],
            "Visible in source frame",
        )
        csv_response = self.client.get(f"/api/missions/{mission_id}/report.csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("Confirmed visible person", csv_response.text)
        self.assertEqual(self.client.get(f"/api/missions/{mission_id}/summary.txt").status_code, 200)
        clip = self.client.get(f"/api/missions/{mission_id}/media/clip")
        self.assertEqual(clip.status_code, 200)
        self.assertGreater(len(clip.content), 1000)

    def test_04_supplied_drive_still_set(self):
        response = self.client.post("/api/analyze/drive-set", data={"confidence_threshold": "0.25"})
        self.assertEqual(response.status_code, 202)
        mission_id = response.json()["id"]
        for _ in range(120):
            mission = self.client.get(f"/api/missions/{mission_id}").json()
            if mission["status"] in {"complete", "failed", "stopped"}:
                break
            time.sleep(.5)
        self.assertEqual(mission["status"], "complete", mission["error"])
        self.assertEqual(mission["is_demo"], 2)
        self.assertEqual(mission["frames_analyzed"], 6)
        self.assertIsNone(mission["duration"])
        self.assertEqual(mission["tracking_available"], 0)
        self.assertGreater(mission["detection_count"], 0)
        records = self.client.get(f"/api/detections?mission_id={mission_id}").json()
        self.assertTrue(all(row["source_filename"].endswith(".jpg") for row in records))
        self.assertTrue(all(row["track_id"] is None for row in records))
        self.assertTrue(all(row["timestamp_seconds"] == 0 for row in records))
        self.assertEqual(self.client.get(f"/api/missions/{mission_id}/source-frame/2").status_code, 200)
        self.assertEqual(self.client.get(f"/api/missions/{mission_id}/source-frame/7").status_code, 404)
        self.assertIn("source_filename", self.client.get(f"/api/missions/{mission_id}/report.csv").text)
        self.assertIn("independent still images", self.client.get(f"/api/missions/{mission_id}/summary.txt").text)

    def test_05_large_still_retry_preserves_threshold_and_clear_upload(self):
        cap = cv2.VideoCapture(str(DEMO_CLIP))
        ok, clear_frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        ok, clear_jpg = cv2.imencode(".jpg", clear_frame)
        self.assertTrue(ok)
        clear = self.client.post(
            "/api/analyze/image", data={"confidence_threshold": "0.25"},
            files={"file": ("clear-person.jpg", clear_jpg.tobytes(), "image/jpeg")},
        )
        self.assertEqual(clear.status_code, 200)
        self.assertGreater(clear.json()["detection_count"], 0)

        drive_image = (ROOT / "demo" / "drive_test_images" / "006.jpg").read_bytes()
        found = self.client.post(
            "/api/analyze/image", data={"confidence_threshold": "0.25"},
            files={"file": ("drive-test.jpg", drive_image, "image/jpeg")},
        )
        self.assertEqual(found.status_code, 200)
        self.assertGreater(found.json()["detection_count"], 0)
        rows = self.client.get(f"/api/detections?mission_id={found.json()['id']}").json()
        self.assertTrue(all(row["model_version"].startswith("YOLO11n") for row in rows))
        self.assertTrue(all(row["confidence"] >= .25 for row in rows))
        self.assertTrue(all(row["x2"] > row["x1"] and row["y2"] > row["y1"] for row in rows))

        high_threshold = self.client.post(
            "/api/analyze/image", data={"confidence_threshold": "0.70"},
            files={"file": ("drive-test.jpg", drive_image, "image/jpeg")},
        )
        self.assertEqual(high_threshold.status_code, 200)
        self.assertEqual(high_threshold.json()["confidence_threshold"], .7)
        high_rows = self.client.get(
            f"/api/detections?mission_id={high_threshold.json()['id']}"
        ).json()
        self.assertTrue(all(row["confidence"] >= .7 for row in high_rows))

        ok, blank_jpg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertTrue(ok)
        empty = self.client.post(
            "/api/analyze/image", data={"confidence_threshold": "0.70"},
            files={"file": ("blank.jpg", blank_jpg.tobytes(), "image/jpeg")},
        )
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json()["confidence_threshold"], .7)
        self.assertEqual(empty.json()["detection_count"], 0)
        self.assertEqual(self.client.get(f"/api/detections?mission_id={empty.json()['id']}").json(), [])

    def test_06_image_upload_matches_independent_raw_predictions(self):
        from ultralytics import YOLO

        cap = cv2.VideoCapture(str(DEMO_CLIP))
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        cases = [("clear-person.jpg", encoded.tobytes(), "image/jpeg")]
        diagnostic_image = os.environ.get("MUDRESCUE_DIAGNOSTIC_IMAGE")
        if diagnostic_image:
            path = Path(diagnostic_image)
            self.assertTrue(path.is_file(), f"Diagnostic image missing: {path}")
            cases.append((path.name, path.read_bytes(),
                          "image/png" if path.suffix.lower() == ".png" else "image/jpeg"))

        raw_model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
        person_class = next(key for key, name in raw_model.names.items() if name == "person")

        def overlap(a, b):
            intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0])) *
                            max(0, min(a[3], b[3]) - max(a[1], b[1])))
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            return intersection / max(1, area_a + area_b - intersection)

        for filename, image_bytes, content_type in cases:
            decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            height, width = decoded.shape[:2]
            for threshold in (.10, .25):
                with self.subTest(filename=filename, threshold=threshold):
                    raw = raw_model.predict(decoded, imgsz=640, conf=threshold,
                                            device="cpu", verbose=False)[0]
                    person_boxes = [box for box in raw.boxes if int(box.cls.item()) == person_class]
                    if not person_boxes:
                        large = max(height, width) >= 1280
                        retry = raw_model.predict(decoded, imgsz=1280 if large else 640,
                                                  conf=threshold, classes=[person_class],
                                                  augment=not large,
                                                  device="cpu", verbose=False)[0]
                        person_boxes = list(retry.boxes)
                    expected = sorted((
                        max(0, min(round(float(box.xyxy[0][0])), width - 1)),
                        max(0, min(round(float(box.xyxy[0][1])), height - 1)),
                        max(0, min(round(float(box.xyxy[0][2])), width - 1)),
                        max(0, min(round(float(box.xyxy[0][3])), height - 1)),
                        round(float(box.conf.item()), 5),
                    ) for box in person_boxes)
                    if not expected:
                        tile_width = min(900, max(1, round(width * .625)))
                        tile_height = min(900, max(1, round(height * .625)))
                        candidates = []
                        for y in sorted({0, height - tile_height}):
                            for x in sorted({0, width - tile_width}):
                                tile = decoded[y:y + tile_height, x:x + tile_width]
                                tiled = raw_model.predict(tile, imgsz=640, conf=threshold,
                                                          classes=[person_class], device="cpu",
                                                          verbose=False)[0]
                                for box in tiled.boxes:
                                    xyxy = [float(v) for v in box.xyxy[0].tolist()]
                                    candidates.append((
                                        max(0, min(x + round(xyxy[0]), width - 1)),
                                        max(0, min(y + round(xyxy[1]), height - 1)),
                                        max(0, min(x + round(xyxy[2]), width - 1)),
                                        max(0, min(y + round(xyxy[3]), height - 1)),
                                        round(float(box.conf.item()), 5),
                                    ))
                        kept = []
                        for box in sorted(candidates, key=lambda item: item[4], reverse=True):
                            if any(overlap(box, prior) > .5 for prior in kept):
                                continue
                            kept.append(box)
                        expected = sorted(kept)
                    if hashlib.sha256(image_bytes).hexdigest() == (
                        "84ddcd806810d20504abde0a074bf530e22d9fbc39c15183107640c3b52abc13"
                    ):
                        self.assertEqual(len(expected), 1)
                    if (threshold == .25 and hashlib.sha256(image_bytes).hexdigest() ==
                            "c74422e05637ee26ac52475cb4f66d685af09ff388d1dbb73424b7d051f44933"):
                        self.assertEqual(len(expected), 1)
                    response = self.client.post(
                        "/api/analyze/image", data={"confidence_threshold": str(threshold)},
                        files={"file": (filename, image_bytes, content_type)},
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    mission = response.json()
                    self.assertEqual(mission["confidence_threshold"], threshold)
                    self.assertEqual(mission["detection_count"], len(expected))
                    self.assertEqual(
                        self.client.get(f"/api/missions/{mission['id']}/media/source").content,
                        image_bytes,
                    )
                    rows = self.client.get(f"/api/detections?mission_id={mission['id']}").json()
                    actual = sorted((row["x1"], row["y1"], row["x2"], row["y2"],
                                     row["confidence"]) for row in rows)
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        self.client.get(f"/api/missions/{mission['id']}/media/frame").status_code, 200,
                    )


if __name__ == "__main__":
    unittest.main()
