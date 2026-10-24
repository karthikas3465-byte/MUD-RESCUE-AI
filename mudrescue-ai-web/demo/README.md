# Bundled demo provenance

`sample_review_clip.mp4` is a four-second H.264 video generated from one public-domain FEMA still photograph. Its pixels include the visible label **STILL-PHOTO DEMO | NOT DRONE FOOTAGE**. The people pictured are search-and-rescue volunteers. This file is a workflow demonstration, not evidence of survivors, a landslide, or buried people.

Photograph: Andrea Booher/FEMA, “Search and Rescue volunteers in Minnesota,” 30 March 2009. [Wikimedia Commons source and public-domain information](https://commons.wikimedia.org/wiki/File:FEMA_-_40381_-_Search_and_Rescue_volunteers_in_Minnesota.jpg).

The app does not ship fictional boxes. Clicking **Run bundled demo clip** performs fresh YOLO inference and creates new local database and evidence records.

`example_annotated_frame.jpg` is an actual YOLO11n result from this clip at a 0.25 confidence threshold. It is a static preview of model output, not a preloaded incident or a claimed survivor detection.

`drive_test_images/` contains six synthetic still images selected from the user's supplied [labelled test folder](https://drive.google.com/drive/folders/111xBVshbgHrCJz3CpMAida5P8w9C3YIB). `manifest.json` preserves their original filenames and source. Clicking **Analyse supplied Drive still images** runs fresh inference on these independent images. They have no real video timestamps or track continuity. The rest of the mixed-origin Drive image set is not bundled.
