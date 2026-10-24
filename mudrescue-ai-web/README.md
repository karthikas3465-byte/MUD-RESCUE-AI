# MudRescue AI — search review prototype

**Every Detection Matters. Every Second Counts.**

MudRescue AI is a local FastAPI + React application for human review of visible person detections in disaster imagery. It accepts images and videos, samples video frames, runs real YOLO11n person inference, stores evidence and review decisions in SQLite, and exports reports. The interface draws red boxes only for actual model detections.

> **AI-assisted search prototype. Human verification required.**

The pretrained COCO detector is a **baseline**, not a validated landslide or occlusion detector. Ordinary RGB video cannot see a completely buried person, establish whether someone is alive or injured, or infer a geographic location from pixel coordinates.

## Quick start (Windows PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. API documentation is at **http://127.0.0.1:8000/docs**. Node.js 20.19+ or 22.12+ is required for the Vite build; Node 24 was tested. Python 3.14 was used in development, but Python 3.11 is the recommended starting point for broad PyTorch wheel availability.

The supplied ZIP includes `models/yolo11n.pt` and a built `frontend/dist`. After Python dependency installation, the last `uvicorn` command can run without Node. If obtaining source without `dist`, run `npm ci` and `npm run build` first.

For frontend development, run the backend command above in one terminal, then `cd frontend; npm run dev` in another and open **http://127.0.0.1:5173/**. Vite proxies `/api` to the local backend. No cloud account or API key is needed.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cd frontend && npm ci && npm run build && cd ..
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

The server binds to loopback by default. Do not expose it on a public network; this hackathon build has no authentication. Uploads, SQLite, and generated evidence are written to `.runtime/` under the project. Set `MUDRESCUE_DATA_DIR` to choose another local data directory. Set `MUDRESCUE_MODEL_PATH` to select a compatible model weight file; a future detector can replace `PersonDetector` in `backend/app/services/detector.py`.

## Tested example workflow

1. Open **Analyse Footage** and click **Analyse supplied Drive still images**. The six selected synthetic stills are bundled for an offline run. You can also upload an image/video or use the separate public-domain demo clip.
2. Choose a confidence threshold and a video sample interval, then start analysis.
3. Watch actual model boxes and progress. The Drive image set shows image indices and original filenames; it has no video timestamps or tracking. Uploaded videos show source timestamps and frame numbers. Pause/resume/stop is checked between inferences.
4. Open **Detection Results** to inspect annotated evidence and the unchanged original source. The WebM review clip plays sampled annotated frames as a **timelapse**, not a frame-perfect copy of the original video.
5. Open **Review Queue**, compare each box with the source, save a status and optional notes.
6. Download the mission CSV, text summary, and annotated image. Video missions also provide a sampled annotated WebM clip.

On the bundled four-second public-domain clip, the packaged YOLO11n model produced **8 boxes across 4 sampled frames**, with **2 temporary track IDs** at threshold 0.25 and a one-second interval. These are real detections of rescue volunteers in a still-photo demonstration; the sample is **not drone footage**, and they are **not survivors**. Results can vary with package/model changes. See [demo/README.md](demo/README.md) for provenance.

## Linked landslide data

The user supplied a [Google Drive Landslide folder](https://drive.google.com/drive/folders/18bcTLqfcTtHg3ZK3pcyVyZiLRYJXHs3c) containing `Labelled data` and `Synthetic + Real`. The supplied folders contain still images and text labels; **no video file was present in the inspected folder listings**. The labelled branch has paired `tested images` / `txt test` and `trained images` / `txt train` folders. One inspected label used YOLO's `class center_x center_y width height` normalized format.

At a 0.25 confidence threshold, the original 640-pixel YOLO11n pass produced **19 boxes across 10 of the 36 images** in `tested images`. This is a box count, **not recall or accuracy**: ground-truth annotations were not audited, boxes may be false or duplicated, and visible people were missed. For example, `1.jpg` produced no person box at 640 pixels. The current app retries large still images at 1280 pixels only when the first pass is empty; the unchanged model then produces one 25.8% box on that image. Some visible figures still produce no box. See [Drive Test Findings](DRIVE_TEST_FINDINGS.md) and the [original-pass per-image CSV](demo/drive_test_evaluation.csv).

The offline **Analyse supplied Drive still images** button runs fresh model inference on six selected `Lucid_Origin...` synthetic images copied from that test folder. At threshold 0.25, the original 640-pixel pass produced 11 boxes across 4 images; the current retry produced **14 boxes across 6 images** in an isolated test. Original filenames, model confidences, coordinates, annotated evidence, and review decisions are retained. These independent images have **no source timestamps, track IDs, or geolocation**. No model training or simulated detection is involved.

The Drive folder mixes generated images with real photographs whose redistribution rights are unclear. Only the six synthetic test images are bundled. The remaining images were evaluated locally and are not redistributed. Before training or publication, establish image rights, verify annotations and class definitions, remove duplicates, and separate synthetic from real evaluation data.

## API and persisted records

Required endpoints: `GET /api/health`, `POST /api/analyze/image`, `POST /api/analyze/video`, `GET /api/detections`, `GET /api/detections/{id}`, `PATCH /api/detections/{id}`, `GET /api/missions`, and `GET /api/reports/summary`.

Additional endpoints provide mission status, pause/resume/stop, source/evidence media, CSV, and a text report. Image analysis returns when inference finishes. Video and Drive-set analysis return a mission ID immediately and run in a local worker; the frontend polls progress. Accepted images: JPG/JPEG/PNG/WEBP/BMP, up to 30 MB. Accepted videos: MP4/MOV/AVI/MKV/WEBM/M4V, up to 350 MB and 30 minutes, provided OpenCV can decode the actual codec. No uploaded footage is sent to an external inference API.

SQLite stores mission metadata, each person box, confidence, temporary track ID when available, evidence path, priority cue, review status, and notes. `Repeated sighting` means one temporary track ID appeared in at least two sampled frames. This is a queue cue, **not** proof of a unique person or a rescue urgency rating. Review status `Confirmed visible person` is a human judgement of visible imagery, **not** a survivor or life-status claim. The interface does not display a map without reliable georeferencing.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm run build
```

Integration tests cover health, unsupported/corrupt media, model-load failure, real bundled inference, frame sampling, evidence, persisted review, CSV, summary, and WebM export. Browser verification checked the working overview, analysis progress, detected-frame viewer, review queue, source playback, analytics, and responsive layout.

## Troubleshooting

- **Model cannot load:** verify `models/yolo11n.pt` exists, or set `MUDRESCUE_MODEL_PATH` to a compatible YOLO person model. On a first install without bundled weights, initialize `YOLO('yolo11n.pt')` once while online.
- **Install fails:** use a clean virtual environment and a Python version with available PyTorch wheels. If an existing environment has a mismatched FastAPI and Starlette pair, reinstall from `backend/requirements.txt`.
- **Video rejected:** convert to a conventional H.264 MP4 with FFmpeg if OpenCV cannot decode the codec. The original file is never overwritten.
- **Annotated clip will not play:** the WebM export depends on OpenCV's VP8 encoder. Download the source evidence frames or open the WebM in another player if the browser lacks VP8 support.
- **No detections:** try reviewing the original frames and, when appropriate, a lower threshold. A blank result is not a safety clearance.
- **Slow CPU:** use a longer sampling interval and shorter clip. No GPU is required; CUDA is selected only when available.
- **Port in use:** change `--port 8000` and open the matching localhost URL.

## Research path

After data licensing and annotation QA, create train/validation/test splits by source scene and capture sequence so near-duplicate frames do not leak across splits. Maintain a separate real, heavily occluded holdout set and negative examples of rocks, branches, clothing, mud and debris. A YOLO dataset YAML should point to `train/images`, `val/images`, and `test/images` with `names: {0: person}`. Fine-tune only then, for example:

```bash
yolo detect train model=models/yolo11n.pt data=path/to/verified-dataset.yaml imgsz=960 epochs=50 device=cpu
yolo detect val model=runs/detect/train/weights/best.pt data=path/to/verified-dataset.yaml split=test
```

Report measured precision, recall, mAP, false positives per image, and occlusion-specific recall separately for real and synthetic imagery. Evaluate aerial altitude, blur, lighting, and camera motion. Research buried-person sensing through other validated modalities (for example thermal where exposed heat is measurable or radar where physically suitable); RGB alone cannot establish such capability.
