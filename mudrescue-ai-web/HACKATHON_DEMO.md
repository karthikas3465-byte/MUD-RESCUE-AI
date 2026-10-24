# Hackathon presentation

## Five-slide outline

1. **Problem:** disaster imagery takes time to review, and missed visual clues matter.
2. **Prototype:** local image/video upload, sampled YOLO11n inference, red evidence boxes and timestamps.
3. **Operator workflow:** original source comparison, review statuses, notes, CSV/summary/WebM export.
4. **Measured Drive data:** the original 640-pixel pass on 36 supplied test stills produced 19 model boxes across 10 images at a 0.25 threshold. The current large-still retry produced 14 boxes across all 6 selected synthetic images, including a formerly missed partially occluded figure.
5. **Limits and research:** no buried-person claims or fake geolocation; audit data rights, train and evaluate on real aerial occlusions, test other physical sensing modalities.

## Two-minute demonstration script

**0:00–0:20** — “MudRescue AI helps a trained operator review disaster footage. It runs locally and every red box comes from a real model prediction. The persistent notice reminds us that human verification is required.”

**0:20–0:45** — “I can upload a video or run the still images from the supplied Drive folder. The folder has labelled images, but no video file in the inspected listings. This offline run uses six synthetic stills. Each is analysed independently, with no invented timestamp or tracker.”

**0:45–1:10** — “YOLO11n runs locally on CPU and records each predicted person box, confidence, image number, and original filename. On these six stills it recorded 14 boxes across 6 images at a 0.25 threshold. That is a count of model outputs, not 14 distinct people or a measure of accuracy.”

**1:10–1:35** — “In Detection Results I compare an annotated image with its unchanged source, then record a human review decision and notes. The database keeps those decisions. I can export a CSV, a summary, and annotated image evidence. Uploaded video also supports timestamp jumps and a sampled WebM clip.”

**1:35–2:00** — “The hardest challenge is partial occlusion. The original 640-pixel pass missed a visible figure that a 1280-pixel retry now marks, but other figures can still be missed. Box counts are not recall because the labels are unaudited. This remains a review aid, not a deployed rescue sensor. Next we need rights-cleared real aerial data, rigorous occlusion testing, and validated sensing beyond RGB for buried-person research.”
