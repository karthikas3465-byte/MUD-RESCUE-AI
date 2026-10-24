import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDownToLine, BarChart3, CheckCircle2, ChevronRight,
  Clock3, Crosshair, FileVideo2, FolderOpen, Image as ImageIcon, LayoutDashboard,
  MapPinOff, Pause, Play, Search, ShieldAlert, SlidersHorizontal, Square, UploadCloud,
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, evidence, media } from "./api";
import type { Detection, Mission, Summary } from "./types";

type Page = "Overview" | "Analyse Footage" | "Detection Results" | "Review Queue" | "Analytics" | "About";
const pages: { title: Page; icon: typeof LayoutDashboard }[] = [
  { title: "Overview", icon: LayoutDashboard },
  { title: "Analyse Footage", icon: Crosshair },
  { title: "Detection Results", icon: Search },
  { title: "Review Queue", icon: FolderOpen },
  { title: "Analytics", icon: BarChart3 },
  { title: "About", icon: ShieldAlert },
];
const reviewOptions = ["Needs review", "Confirmed visible person", "False detection", "Investigate on site"];
const emptySummary: Summary = {
  analyzed_missions: 0, total_detections: 0, pending_reviews: 0,
  unique_tracks: 0, frames_analyzed: 0, missions: [], detections: [],
};

function time(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const value = Math.floor(seconds);
  return [Math.floor(value / 3600), Math.floor(value / 60) % 60, value % 60]
    .map((part) => String(part).padStart(2, "0")).join(":");
}
function date(iso: string) {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
function statusName(status: Mission["status"]) {
  return status === "complete" ? "ANALYSIS COMPLETE" : status.toUpperCase();
}
function StatCard({ label, value, caption, tone = "" }: {
  label: string; value: string | number; caption?: string; tone?: string;
}) {
  return <div className="stat-card">
    <div className="stat-top"><span>{label}</span><span className={`stat-dot ${tone}`} /></div>
    <strong>{value}</strong><small>{caption || "Actual recorded data"}</small>
  </div>;
}

export default function App() {
  const [page, setPage] = useState<Page>("Overview");
  const [missions, setMissions] = useState<Mission[]>([]);
  const [mission, setMission] = useState<Mission | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selected, setSelected] = useState<Detection | null>(null);
  const [summary, setSummary] = useState<Summary>(emptySummary);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(.25);
  const [interval, setIntervalSeconds] = useState(1);
  const [tracking, setTracking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [minConfidence, setMinConfidence] = useState(0);
  const [statusFilter, setStatusFilter] = useState("All");
  const [notes, setNotes] = useState("");
  const [refreshingReview, setRefreshingReview] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  async function refreshOverview() {
    const [missionRows, totals] = await Promise.all([
      api<Mission[]>("/api/missions"), api<Summary>("/api/reports/summary"),
    ]);
    setMissions(missionRows);
    setSummary(totals);
  }
  async function refreshMission(id: string) {
    const [current, rows] = await Promise.all([
      api<Mission>(`/api/missions/${id}`),
      api<Detection[]>(`/api/detections?mission_id=${id}`),
    ]);
    setMission(current);
    setDetections(rows);
    setSelected((prior) => rows.find((row) => row.id === prior?.id) || rows[0] || null);
    return current;
  }
  useEffect(() => {
    refreshOverview().catch((exc) => setError(exc.message));
  }, []);
  useEffect(() => {
    if (!mission || !["queued", "processing", "paused"].includes(mission.status)) return;
    const id = window.setInterval(async () => {
      try {
        const updated = await refreshMission(mission.id);
        if (["complete", "stopped", "failed"].includes(updated.status)) await refreshOverview();
      } catch (exc) { setError((exc as Error).message); }
    }, 900);
    return () => window.clearInterval(id);
  }, [mission?.id, mission?.status]);
  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => { setNotes(selected?.reviewer_notes || ""); }, [selected?.id, selected?.reviewer_notes]);

  function acceptFile(candidate: File | undefined) {
    if (!candidate) return;
    const extension = candidate.name.split(".").pop()?.toLowerCase();
    const isImage = ["jpg", "jpeg", "png", "webp", "bmp"].includes(extension || "");
    const isVideo = ["mp4", "mov", "avi", "mkv", "webm", "m4v"].includes(extension || "");
    if (!isImage && !isVideo) { setError("Use JPG, PNG, WEBP, BMP, MP4, MOV, AVI, MKV, WEBM, or M4V."); return; }
    if (candidate.size > (isImage ? 30 : 350) * 1024 * 1024) {
      setError(`File exceeds the ${isImage ? 30 : 350} MB limit.`); return;
    }
    setError(null);
    setFile(candidate);
    setMission(null);
    setDetections([]);
    setSelected(null);
    setPage("Analyse Footage");
  }
  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    acceptFile(event.dataTransfer.files[0]);
  }
  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }
  async function startAnalysis(mode: "upload" | "clip" | "drive" = "upload") {
    setBusy(true); setError(null);
    if (mode !== "upload") setFile(null);
    try {
      const body = new FormData();
      if (mode === "upload" && file) body.append("file", file);
      body.append("confidence_threshold", String(confidence));
      body.append("sample_interval", String(interval));
      body.append("tracking", String(tracking));
      const isImage = mode === "upload" && file?.type.startsWith("image/");
      const endpoint = mode === "clip" ? "/api/analyze/demo" : mode === "drive" ? "/api/analyze/drive-set"
        : isImage ? "/api/analyze/image" : "/api/analyze/video";
      const created = await api<Mission>(endpoint, { method: "POST", body });
      setMission(created); setDetections([]); setSelected(null);
      setPage("Analyse Footage");
      await refreshOverview();
      if (created.status === "complete") await refreshMission(created.id);
    } catch (exc) { setError((exc as Error).message); }
    finally { setBusy(false); }
  }
  async function openMission(id: string) {
    setError(null);
    setFile(null);
    try { await refreshMission(id); setPage("Analyse Footage"); }
    catch (exc) { setError((exc as Error).message); }
  }
  async function control(action: "pause" | "resume" | "stop") {
    if (!mission) return;
    try {
      const updated = await api<Mission>(`/api/missions/${mission.id}/control`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }),
      });
      setMission(updated);
    } catch (exc) { setError((exc as Error).message); }
  }
  async function updateReview(detection: Detection, status: string) {
    setRefreshingReview(true); setError(null);
    try {
      await api<Detection>(`/api/detections/${detection.id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_status: status, reviewer_notes: notes }),
      });
      if (mission) await refreshMission(mission.id);
      await refreshOverview();
    } catch (exc) { setError((exc as Error).message); }
    finally { setRefreshingReview(false); }
  }
  function jumpToDetection(detection: Detection) {
    setSelected(detection);
    setPage("Detection Results");
    window.setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.currentTime = detection.timestamp_seconds;
        videoRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }, 100);
  }

  const filtered = detections
    .filter((row) => row.confidence >= minConfidence)
    .filter((row) => statusFilter === "All" || row.review_status === statusFilter)
    .sort((a, b) => a.timestamp_seconds - b.timestamp_seconds);
  const active = selected || detections[0];
  const stageFrame = mission?.current_frame_path
    ? media(mission.id, "frame", mission.frames_analyzed) : null;
  const timeline = (() => {
    const map = new Map<number, number>();
    if (mission?.is_demo === 2) {
      for (let image = 1; image <= (mission.frame_count || 0); image++) map.set(image, 0);
    }
    for (const row of detections) {
      const second = mission?.is_demo === 2 ? row.frame_number : Math.floor(row.timestamp_seconds);
      map.set(second, (map.get(second) || 0) + 1);
    }
    return [...map.entries()].sort((a, b) => a[0] - b[0])
      .map(([second, count]) => ({ time: mission?.is_demo === 2 ? `Image ${second}` : time(second), detections: count }));
  })();
  const histogram = [0, .2, .4, .6, .8].map((lower) => ({
    range: `${Math.round(lower * 100)}–${Math.round((lower + .2) * 100)}%`,
    count: detections.filter((row) => row.confidence >= lower && row.confidence < lower + .2).length,
  }));

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Crosshair size={23} strokeWidth={2.4} /></div>
        <div><strong>MUDRESCUE <span>AI</span></strong><small>SEARCH OPERATIONS</small></div>
      </div>
      <div className="sidebar-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        {pages.map(({ title, icon: Icon }) =>
          <button key={title} className={`nav-item ${page === title ? "active" : ""}`}
            onClick={() => setPage(title)}><Icon size={18} /><span>{title}</span>
            {title === "Review Queue" && summary.pending_reviews > 0 &&
              <em>{summary.pending_reviews}</em>}</button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="system-status"><span className="status-light" /> LOCAL INFERENCE <strong>READY</strong></div>
        <p>Footage stays on this computer. Model outputs require human verification.</p>
        <div className="sidebar-version">YOLO11n BASELINE <span>v0.2</span></div>
      </div>
    </aside>

    <div className="workspace">
      <header className="topbar">
        <div className="topbar-location"><span>OPERATIONS CENTER</span><ChevronRight size={14} /><strong>{page.toUpperCase()}</strong></div>
        <div className="topbar-right"><span className="topbar-clock"><Clock3 size={15} /> LOCAL SESSION</span>
          <span className="operator-badge">MR</span></div>
      </header>
      <main>
        <div className="notice"><ShieldAlert size={18} />
          <span>AI-assisted search prototype. Human verification required.</span>
          <span className="notice-end">RGB / VISIBLE PERSONS ONLY</span>
        </div>
        {error && <div className="error-banner" role="alert"><AlertTriangle size={18} /> {error}
          <button aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div>}

        {page === "Overview" && <>
          <div className="page-heading">
            <div><div className="eyebrow">MISSION CONTROL / OVERVIEW</div>
              <h1>Search operations</h1><p>Review real model evidence from locally analyzed imagery.</p></div>
            <button className="button-primary" onClick={() => setPage("Analyse Footage")}><UploadCloud size={18} /> Analyse footage</button>
          </div>
          <div className="stats-grid">
            <StatCard label="ANALYZED MISSIONS" value={summary.analyzed_missions} caption="Completed or stopped analyses" />
            <StatCard label="PERSON DETECTIONS" value={summary.total_detections} caption="Individual model boxes" tone="red" />
            <StatCard label="PENDING REVIEW" value={summary.pending_reviews} caption="Awaiting an operator" tone="amber" />
            <StatCard label="TEMPORARY TRACKS" value={summary.unique_tracks} caption="Estimates, not identities" tone="green" />
          </div>
          <div className="overview-grid">
            <section className="panel">
              <div className="section-header"><div><span className="eyebrow">RECENT ACTIVITY</span><h2>Mission history</h2></div>
                <span className="section-count">{missions.length} missions</span></div>
              {missions.length ? <div className="mission-list">{missions.map((row) =>
                <button className="mission-row" key={row.id} onClick={() => openMission(row.id)}>
                  <div className="mission-icon">{row.media_type === "image" ? <ImageIcon size={20} /> : <FileVideo2 size={20} />}</div>
                  <div className="mission-name"><strong>{row.is_demo === 2 ? "DRIVE STILL SET · " : row.is_demo ? "DEMO · " : ""}{row.filename}</strong><small>{date(row.created_at)} · {row.frames_analyzed} {row.is_demo === 2 ? "images" : "frames"} analyzed</small></div>
                  <span className={`status-pill ${row.status}`}>{row.status}</span><ChevronRight size={17} />
                </button>)}</div> :
                <div className="empty-state"><FolderOpen size={34} /><strong>No missions yet</strong>
                  <p>Upload an image or video to create your first real analysis record.</p></div>}
            </section>
            <section className="panel readiness-panel">
              <div className="section-header"><div><span className="eyebrow">OPERATING NOTES</span><h2>Review protocol</h2></div></div>
              <div className="protocol-item"><span>01</span><div><strong>Inspect the source</strong><p>Compare every red box with the original frame.</p></div></div>
              <div className="protocol-item"><span>02</span><div><strong>Record a decision</strong><p>Mark visible evidence and write reviewer notes.</p></div></div>
              <div className="protocol-item"><span>03</span><div><strong>Export the record</strong><p>Download CSV, summary, and annotated evidence.</p></div></div>
              <div className="capability-note"><MapPinOff size={19} /><span>No location is inferred from image position. Mapping requires reliable georeferencing.</span></div>
            </section>
          </div>
        </>}

        {page === "Analyse Footage" && <>
          <div className="page-heading"><div><div className="eyebrow">ANALYSIS / SOURCE MATERIAL</div>
            <h1>Footage analysis</h1><p>Visible person detection in aerial and disaster imagery.</p></div>
            {mission && <span className={`status-pill large ${mission.status}`}>{statusName(mission.status)}</span>}
          </div>
          <div className="analysis-grid">
            <section className="stage-panel">
              <div className="stage-toolbar"><span className="live-dot" /> {mission ? mission.filename : file ? file.name : "AWAITING SOURCE"}
                <span>{mission?.is_demo === 2 ? "SUPPLIED DRIVE SYNTHETIC STILLS · NO VIDEO" : mission?.is_demo ? "STILL-PHOTO DEMO · NOT DRONE FOOTAGE" : mission?.media_type === "video" ? "VIDEO ANALYSIS" : mission?.media_type === "image" ? "IMAGE ANALYSIS" : "LOCAL PREVIEW"}</span></div>
              <div className="stage" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
                {stageFrame ? <img src={stageFrame} alt="Current sampled frame with actual model bounding boxes" /> :
                  preview ? file?.type.startsWith("image/") ?
                    <img src={preview} alt="Selected source image before analysis" /> :
                    <video src={preview} controls aria-label="Selected source video" /> :
                    <div className="stage-empty"><Crosshair size={62} strokeWidth={1} /><strong>AWAITING SOURCE IMAGERY</strong>
                      <span>Upload an image or video, or run the bundled demonstration.</span></div>}
                <div className="stage-brand"><span className="stage-brand-mark"><Crosshair size={19} /></span>
                  <div><strong>MUDRESCUE AI</strong><small>VISIBLE PERSON SEARCH REVIEW</small></div></div>
                <div className="stage-status"><span>{mission?.is_demo ? "DEMO ANALYSIS" : mission?.status === "processing" ? "VIDEO ANALYSIS" : mission?.status === "paused" ? "PAUSED" : "LOCAL ANALYSIS"}</span>
                  <small>{mission?.is_demo === 2 ? "IMAGE" : "FRAME"} {mission?.current_frame_number ?? "—"}{mission?.is_demo === 2 ? "" : <> <b>·</b> {time(mission?.current_timestamp)}</>}</small></div>
                {mission && <div className="stage-evidence"><div className="evidence-flag"><span /> DETECTION EVIDENCE</div>
                  <strong>{mission.detection_count ? "POSSIBLE PERSON DETECTED — HUMAN REVIEW REQUIRED" : "NO MODEL PERSON BOX IN CURRENT EVIDENCE"}</strong>
                  <div className="evidence-meta"><span>Model detections <b>{mission.detection_count}</b></span>
                    <span>{mission.is_demo === 2 ? "Image" : "Frame"} <b>{mission.current_frame_number ?? "—"}</b></span>
                    {mission.is_demo !== 2 && <span>Timestamp <b>{time(mission.current_timestamp)}</b></span>}</div></div>}
              </div>
              {mission && <div className="stage-footer">
                <div><span>PROGRESS</span><strong>{Math.round(mission.progress * 100)}%</strong></div>
                <div className="progress-track"><span style={{ width: `${Math.round(mission.progress * 100)}%` }} /></div>
                <span>{mission.frames_analyzed} {mission.is_demo === 2 ? "independent images" : "sampled frames"} · {mission.detection_count} detections · {mission.processing_seconds.toFixed(1)}s processing</span>
                {mission.error && <p className="inline-error">{mission.error}</p>}
              </div>}
            </section>
            <aside className="control-panel panel">
              <div className="section-header"><div><span className="eyebrow">INPUT & PARAMETERS</span><h2>Analysis controls</h2></div><SlidersHorizontal size={18} /></div>
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.avi,.mkv,.webm,.m4v"
                onChange={onFileChange} className="hidden-input" aria-label="Select image or video" />
              <div className="upload-zone" onDrop={onDrop} onDragOver={(e) => e.preventDefault()}
                onClick={() => inputRef.current?.click()} role="button" tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter") inputRef.current?.click(); }}>
                <UploadCloud size={27} /><strong>{file ? file.name : "Drop footage here"}</strong>
                <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · click to replace` : "or click to browse local files"}</span>
                <small>Images ≤30 MB · videos ≤350 MB · local processing</small>
              </div>
              <div className="control-field"><label htmlFor="confidence">CONFIDENCE THRESHOLD <strong>{Math.round(confidence * 100)}%</strong></label>
                <input id="confidence" type="range" min=".05" max=".95" step=".01" value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))} />
                <small>Lower values show more candidates and may add false detections.</small></div>
              <div className="control-field"><label htmlFor="interval">FRAME SAMPLING <strong>Every {interval.toFixed(1)}s</strong></label>
                <input id="interval" type="range" min=".2" max="5" step=".2" value={interval}
                  onChange={(e) => setIntervalSeconds(Number(e.target.value))} />
                <small>Video only. Each sampled frame receives real inference.</small></div>
              <label className="toggle-line"><span><strong>Temporary tracking</strong><small>Associate boxes across sampled frames</small></span>
                <input type="checkbox" checked={tracking} onChange={(e) => setTracking(e.target.checked)} /></label>
              <button className="button-primary full" onClick={() => startAnalysis()} disabled={!file || busy}>
                <Crosshair size={18} /> {busy ? "Starting analysis…" : "Start analysis"}</button>
              <button className="button-secondary full" onClick={() => startAnalysis("drive")} disabled={busy}>
                <ImageIcon size={17} /> Analyse supplied Drive still images</button>
              <div className="control-note"><AlertTriangle size={17} /> The supplied Drive folder contains labelled still images. This run uses six synthetic test images and reports real model outputs, including misses. It is not drone video.</div>
              <button className="button-secondary full" onClick={() => startAnalysis("clip")} disabled={busy}>
                <Play size={17} /> Run bundled demo clip</button>
              {mission && ["queued", "processing", "paused"].includes(mission.status) &&
                <div className="job-controls">
                  <button onClick={() => control(mission.status === "paused" ? "resume" : "pause")}>
                    {mission.status === "paused" ? <Play size={16} /> : <Pause size={16} />}
                    {mission.status === "paused" ? "Resume" : "Pause"}</button>
                  <button onClick={() => control("stop")}><Square size={15} /> Stop</button>
                </div>}
              <div className="control-note"><AlertTriangle size={17} /> A missed detection does not rule out a person. Review original footage.</div>
            </aside>
          </div>
          {mission && <div className="quick-stats">
            <StatCard label={mission.is_demo === 2 ? "IMAGES ANALYZED" : "FRAMES ANALYZED"} value={mission.frames_analyzed} caption={mission.is_demo === 2 ? "Independent synthetic stills" : mission.media_type === "video" ? `Sampling every ${mission.sample_interval}s` : "Single image"} />
            <StatCard label={mission.is_demo === 2 ? "IMAGES WITH BOXES" : "FRAMES WITH DETECTIONS"} value={mission.frames_with_detections} tone="red" />
            <StatCard label="TOTAL PERSON BOXES" value={mission.detection_count} tone="amber" />
            <StatCard label="DISTINCT TRACKS" value={mission.tracking_available ? mission.unique_tracks : "—"} caption="Temporary estimate" tone="green" />
          </div>}
        </>}

        {page === "Detection Results" && <>
          <div className="page-heading"><div><div className="eyebrow">EVIDENCE / DETECTION RESULTS</div>
            <h1>Detection evidence</h1><p>Inspect source frames and actual model outputs before review.</p></div>
            {mission && <div className="heading-actions">
              <a className="button-secondary" href={`/api/missions/${mission.id}/report.csv`}><ArrowDownToLine size={16} /> CSV</a>
              <a className="button-secondary" href={`/api/missions/${mission.id}/summary.txt`}><ArrowDownToLine size={16} /> Summary</a>
            </div>}
          </div>
          {!mission ? <div className="empty-state panel"><Search size={38} /><strong>No mission selected</strong>
            <p>Analyze footage or open a mission from the overview.</p></div> :
          <div className="results-grid">
            <section className="panel result-view">
              <div className="section-header"><div><span className="eyebrow">ANNOTATED EVIDENCE</span>
                <h2>{active ? `${mission.is_demo === 2 ? "Image" : "Frame"} ${active.frame_number}` : "No detections recorded"}</h2></div>
                <span className="section-count">{mission.is_demo === 2 ? (active?.source_filename || "Still image") : active ? time(active.timestamp_seconds) : time(mission.current_timestamp)}</span></div>
              {active ? <img className="evidence-image" src={evidence(active.id)} alt={`Annotated frame ${active.frame_number} showing a model detection`} /> :
                <div className="empty-frame"><Search size={40} /><strong>No model boxes recorded</strong>
                  <p>Review the original source. No detection is not evidence of absence.</p></div>}
              {active && <div className="evidence-detail">
                <div><span>MODEL CONFIDENCE</span><strong>{(active.confidence * 100).toFixed(1)}%</strong></div>
                <div><span>BOUNDING BOX</span><strong>{active.x1}, {active.y1} → {active.x2}, {active.y2}</strong></div>
                <div><span>TRACK ID</span><strong>{active.track_id ?? "Unavailable"}</strong></div>
                <div><span>REVIEW</span><strong>{active.review_status}</strong></div>
              </div>}
              {active && <a className="button-secondary" href={evidence(active.id)} download={`mudrescue-${active.frame_number}-annotated.jpg`}><ArrowDownToLine size={16} /> Download annotated image</a>}
              <div className="source-preview"><div className="section-header"><div><span className="eyebrow">SOURCE CHECK</span><h2>{mission.is_demo === 2 ? "Original still image" : "Original footage"}</h2></div>
                {active && mission.media_type === "video" && <button className="text-button" onClick={() => {
                  if (videoRef.current) videoRef.current.currentTime = active.timestamp_seconds;
                }}>Jump to {time(active.timestamp_seconds)}</button>}</div>
                {mission.media_type === "video" ? <video ref={videoRef} src={media(mission.id, "source")} controls preload="metadata" aria-label="Original uploaded video" /> :
                  <img src={mission.is_demo === 2 && active ? `/api/missions/${mission.id}/source-frame/${active.frame_number}` : media(mission.id, "source")} alt="Original source image" />}
                <p>{mission.is_demo === 2 ? "Supplied synthetic still image. No source timestamp or tracking exists. " : mission.is_demo ? "Still-photo demo, not drone footage. " : ""}Original source is unchanged; model boxes appear only on the annotated evidence.</p></div>
              <div className="exports">
                {mission.annotated_video_path && <a className="button-secondary" href={media(mission.id, "clip")} download><ArrowDownToLine size={16} /> Sampled review clip</a>}
                {mission.annotated_video_path && <video src={media(mission.id, "clip")} controls preload="metadata" aria-label="Annotated sampled frame review clip" />}
              </div>
            </section>
            <aside className="panel detections-list">
              <div className="section-header"><div><span className="eyebrow">TIMELINE</span><h2>Candidate sightings</h2></div>
                <span className="section-count">{filtered.length}</span></div>
              <div className="filter-row"><label>MIN CONFIDENCE <b>{Math.round(minConfidence * 100)}%</b>
                <input type="range" min="0" max=".95" step=".05" value={minConfidence}
                  onChange={(e) => setMinConfidence(Number(e.target.value))} /></label>
                <select aria-label="Filter review status" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
                  <option>All</option>{reviewOptions.map((option) => <option key={option}>{option}</option>)}</select></div>
              <div className="detection-scroll">{filtered.length ? filtered.map((row) =>
                <button key={row.id} className={`detection-row ${active?.id === row.id ? "selected" : ""}`}
                  onClick={() => setSelected(row)}><img src={evidence(row.id)} alt="" />
                  <span><strong>{mission.is_demo === 2 ? `IMAGE ${row.frame_number}` : time(row.timestamp_seconds)} <em>{mission.is_demo === 2 ? row.source_filename : `FRAME ${row.frame_number}`}</em></strong>
                    <small>{(row.confidence * 100).toFixed(1)}% confidence · {row.review_status}</small>
                    <small>{row.priority}{row.track_id != null ? ` · Track ${row.track_id}` : ""}</small></span>
                    <ChevronRight size={15} /></button>) :
                <div className="empty-list">No detections match these filters.</div>}</div>
              {active && <button className="button-primary full" onClick={() => setPage("Review Queue")}>Review selected candidate <ChevronRight size={17} /></button>}
            </aside>
          </div>}
        </>}

        {page === "Review Queue" && <>
          <div className="page-heading"><div><div className="eyebrow">HUMAN VERIFICATION / QUEUE</div>
            <h1>Review candidates</h1><p>Model detections are potential sightings, never confirmed rescue outcomes.</p></div>
            <span className="section-count">{mission?.pending_reviews ?? 0} pending in current mission</span></div>
          {!mission || !detections.length ? <div className="empty-state panel"><CheckCircle2 size={37} /><strong>No candidates to review</strong>
            <p>When the model records detections, each will appear here with its evidence frame.</p></div> :
          <div className="review-grid"><div className="panel review-list">
            <div className="section-header"><div><span className="eyebrow">CURRENT MISSION</span><h2>{mission.filename}</h2></div></div>
            {detections.map((row) => <button key={row.id} className={`review-row ${active?.id === row.id ? "selected" : ""}`}
              onClick={() => setSelected(row)}><span className="review-time">{mission.is_demo === 2 ? `IMG ${row.frame_number}` : time(row.timestamp_seconds)}</span>
              <span><strong>{mission.is_demo === 2 ? row.source_filename : `Frame ${row.frame_number}`}</strong><small>{(row.confidence * 100).toFixed(1)}% model confidence</small></span>
              <em className={row.review_status === "Needs review" ? "pending" : "reviewed"}>{row.review_status}</em></button>)}
          </div><div className="panel review-editor">{active && <>
            <div className="section-header"><div><span className="eyebrow">DETECTION {active.id.slice(0, 8).toUpperCase()}</span><h2>Evidence review</h2></div></div>
            <img src={evidence(active.id)} alt="Evidence frame for human review" />
            <div className="review-facts"><span>{mission.is_demo === 2 ? "IMAGE" : "FRAME"} <b>{active.frame_number}</b></span>{mission.is_demo !== 2 && <span>TIME <b>{time(active.timestamp_seconds)}</b></span>}
              <span>CONFIDENCE <b>{(active.confidence * 100).toFixed(1)}%</b></span></div>
            <p className="priority-explain">{active.priority === "Repeated sighting"
              ? "Repeated sighting: the same temporary track ID appeared in at least two sampled frames. This is a review cue, not a rescue priority."
              : "Standard review: no repeated temporary track evidence was recorded."}</p>
            <label className="notes-label" htmlFor="review-notes">REVIEWER NOTES</label>
            <textarea id="review-notes" value={notes} maxLength={1000} onChange={(e) => setNotes(e.target.value)}
              placeholder="Describe what is visible and any uncertainty…" />
            <div className="review-actions">{reviewOptions.map((option) =>
              <button disabled={refreshingReview} key={option} onClick={() => updateReview(active, option)}
                className={`review-action ${active.review_status === option ? "current" : ""}`}>{option}</button>)}</div>
            <button className="text-button" onClick={() => jumpToDetection(active)}>View alongside original source <ChevronRight size={15} /></button>
          </>}</div></div>}
        </>}

        {page === "Analytics" && <>
          <div className="page-heading"><div><div className="eyebrow">MEASURED OUTPUT / ANALYTICS</div>
            <h1>Analysis metrics</h1><p>Charts use stored inference records from the selected mission.</p></div></div>
          <div className="stats-grid">
            <StatCard label="FRAMES PROCESSED" value={mission?.frames_analyzed ?? summary.frames_analyzed} />
            <StatCard label="DETECTION EVENTS" value={mission?.detection_count ?? summary.total_detections} tone="red" />
            <StatCard label="FRAMES WITH BOXES" value={mission?.frames_with_detections ?? "—"} tone="amber" />
            <StatCard label="PROCESSING TIME" value={mission ? `${mission.processing_seconds.toFixed(1)}s` : "—"} tone="green" />
          </div>
          <div className="charts-grid"><section className="panel chart-panel">
            <div className="section-header"><div><span className="eyebrow">{mission?.is_demo === 2 ? "IMAGE SET" : "TEMPORAL ACTIVITY"}</span><h2>{mission?.is_demo === 2 ? "Detections by image" : "Detections over time"}</h2></div></div>
            {timeline.length ? <ResponsiveContainer width="100%" height={270}><AreaChart data={timeline}>
              <defs><linearGradient id="redArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ed4650" stopOpacity={.35} /><stop offset="100%" stopColor="#ed4650" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="#263543" vertical={false} /><XAxis dataKey="time" stroke="#8294a4" tickLine={false} />
              <YAxis allowDecimals={false} stroke="#8294a4" tickLine={false} />
              <Tooltip contentStyle={{ background: "#172431", border: "1px solid #354555", color: "#fff" }} />
              <Area dataKey="detections" stroke="#ed4650" fill="url(#redArea)" strokeWidth={2} />
            </AreaChart></ResponsiveContainer> : <div className="empty-chart">No detections to chart.</div>}</section>
            <section className="panel chart-panel"><div className="section-header"><div><span className="eyebrow">MODEL OUTPUT</span><h2>Confidence distribution</h2></div></div>
              {detections.length ? <ResponsiveContainer width="100%" height={270}><BarChart data={histogram}>
                <CartesianGrid stroke="#263543" vertical={false} /><XAxis dataKey="range" stroke="#8294a4" tickLine={false} />
                <YAxis allowDecimals={false} stroke="#8294a4" tickLine={false} />
                <Tooltip contentStyle={{ background: "#172431", border: "1px solid #354555", color: "#fff" }} />
                <Bar dataKey="count" fill="#d8a448" radius={[4, 4, 0, 0]} />
              </BarChart></ResponsiveContainer> : <div className="empty-chart">No confidence values to chart.</div>}</section></div>
          <div className="analytics-note"><Activity size={18} /> Tracking estimates can fragment when the camera moves or people are occluded. Confidence is a model estimate, not the probability of survival.</div>
        </>}

        {page === "About" && <>
          <div className="page-heading"><div><div className="eyebrow">PROJECT / METHOD</div><h1>About MudRescue AI</h1>
            <p>Every Detection Matters. Every Second Counts.</p></div></div>
          <div className="about-grid"><section className="panel about-card"><span className="eyebrow">WHAT THIS PROTOTYPE DOES</span><h2>Human-reviewed visual triage</h2>
            <p>YOLO11n pretrained on COCO predicts visible person boxes in uploaded images and sampled video frames. OpenCV records actual coordinates, confidence estimates, frame numbers, and timestamps. SQLite preserves missions and human review decisions locally.</p>
            <p>The current detector has not been validated as occlusion robust on aerial landslide imagery. A missed box does not establish that nobody is present.</p></section>
            <section className="panel about-card"><span className="eyebrow">SAFETY BOUNDARIES</span><h2>Operational limitations</h2>
              <ul><li>Ordinary RGB imagery cannot see through opaque mud or debris.</li>
                <li>The model cannot determine whether a person is alive, injured, or in need of rescue.</li>
                <li>Small targets, unusual poses, occlusion, poor light, blur, and aerial angles can cause misses or false detections.</li>
                <li>Track IDs are temporary box associations, not personal identities.</li>
                <li>Geolocation needs trustworthy drone metadata, calibration, altitude, and georeferencing.</li></ul></section>
            <section className="panel about-card"><span className="eyebrow">DATA & PRIVACY</span><h2>Local by design</h2>
              <p>Uploads, evidence frames, and the SQLite incident log remain on the computer running the backend. The application does not call a cloud inference service or perform face recognition.</p>
              <p>The supplied Drive test set contains independent synthetic stills, not video. The bundled clip is a separate, labelled public-domain sample. All detection boxes come from the same real model pipeline.</p></section>
            <section className="panel about-card"><span className="eyebrow">RESEARCH ROADMAP</span><h2>Better evidence, measured first</h2>
              <p>Develop a licensed aerial landslide dataset with held-out occluded examples. Fine-tune a replaceable detector, measure precision, recall, mAP, and occlusion-specific recall, and test false positives against rocks, clothing, and branches. Buried-person sensing requires other modalities and field validation.</p></section></div>
        </>}
      </main>
      <footer className="footer"><span>© MUDRESCUE AI · HACKATHON PROTOTYPE</span><span>HUMAN VERIFICATION REQUIRED · NO GEOGRAPHIC INFERENCE</span></footer>
    </div>
  </div>;
}
