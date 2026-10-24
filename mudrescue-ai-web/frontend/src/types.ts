export type Mission = {
  id: string;
  filename: string;
  media_type: "image" | "video";
  is_demo: number;
  status: "queued" | "processing" | "paused" | "stopped" | "complete" | "failed";
  created_at: string;
  width: number | null;
  height: number | null;
  fps: number | null;
  frame_count: number | null;
  duration: number | null;
  current_frame_number: number | null;
  current_timestamp: number | null;
  current_frame_path: string | null;
  annotated_video_path: string | null;
  confidence_threshold: number;
  sample_interval: number;
  tracking_requested: number;
  tracking_available: number;
  frames_analyzed: number;
  frames_with_detections: number;
  detection_count: number;
  unique_tracks: number;
  pending_reviews: number;
  processing_seconds: number;
  progress: number;
  error: string | null;
};

export type Detection = {
  id: string;
  mission_id: string;
  frame_number: number;
  timestamp_seconds: number;
  confidence: number;
  x1: number; y1: number; x2: number; y2: number;
  track_id: number | null;
  model_version: string;
  source_filename: string | null;
  review_status: string;
  reviewer_notes: string;
  priority: string;
};

export type Summary = {
  analyzed_missions: number;
  total_detections: number;
  pending_reviews: number;
  unique_tracks: number;
  frames_analyzed: number;
  missions: Mission[];
  detections: Detection[];
};
