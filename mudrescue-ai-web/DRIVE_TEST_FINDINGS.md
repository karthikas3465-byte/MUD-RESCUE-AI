# Supplied Drive image test — baseline findings

Source: [Landslide Google Drive folder](https://drive.google.com/drive/folders/18bcTLqfcTtHg3ZK3pcyVyZiLRYJXHs3c), specifically `Labelled data/tested images`.

The inspected folder listings contain JPEG still images and text labels, with no video file. Image filenames and prompts indicate that many scenes are synthetic. Their apparent aerial viewpoint does not turn them into actual drone footage. The six images packaged with the app are selected `Lucid_Origin...` synthetic stills.

## What was run

- Model: bundled, pretrained YOLO11n COCO person class; no fine-tuning. The 36-image figures below are from the original 640-pixel inference pass.
- Threshold: 0.25; CPU inference on each of the 36 test JPEGs.
- Output: 19 model boxes across 10 images. The other 26 images produced no box.
- Offline app subset: 6 selected synthetic images, 11 boxes across 4 images; 2 images with no box.
- Original images were unchanged. The app creates separate annotated evidence and review records.

These are **detection event counts**, not accuracy, precision, or recall. The annotations have not been audited. One visible figure may generate multiple boxes, and some boxes may be false detections. Conversely, several visible figures were missed. In `1.jpg`, a partially obscured figure produced no box in the original 640-pixel pass. The current app retries large still images at 1280 pixels when the first pass is empty; it produces one 25.8% box on `1.jpg`. On the six-image synthetic subset, the current app produced 14 boxes across all 6 images in an isolated test. A blank model result still cannot clear a search area.

Independent still images have **no source video time, motion continuity, track IDs, or geographic coordinates**. The app displays image numbers and original filenames for this set. It does not infer a person's condition or an on-site rescue outcome. Ordinary RGB imagery cannot reveal a completely buried person.

See [per-image outputs](demo/drive_test_evaluation.csv). The CSV records only actual YOLO boxes from the local test. The app's downloadable mission CSV adds reviewer decisions for the selected six-image subset.

## Next validation step

Audit ground-truth labels and usage rights, separate real and synthetic imagery, and establish a held-out real aerial test set. Score matched boxes using a documented IoU threshold, then report precision, recall, false positives per image, and occlusion-specific recall. Do not train and evaluate on near-duplicate images from one source scene.
