"""
Padel Smart Reframer — Intelligent 9:16 Vertical Crop
======================================================
Upload a landscape padel video, get a broadcast-quality
vertical reframe that follows ball + players like a real
camera operator.

TRACKING PHILOSOPHY:
  - During rallies → follow the ball (tight, responsive)
  - Ball lost / between points → frame the players (wide, stable)
  - Serve / setup → anticipate, drift to receiver side
  - Always → smooth, never cut, never jitter

Run:  python reframer.py
Open: http://localhost:5001
"""

import os
import cv2
import time
import shutil
import logging
import subprocess
import numpy as np
from collections import deque
from datetime import datetime

from flask import Flask, request, render_template_string, send_from_directory, jsonify

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")

# =============================================================
# CONFIG
# =============================================================
UPLOAD_DIR   = "reframer_uploads"
OUTPUT_DIR   = "reframer_outputs"
TEMP_DIR     = "reframer_temp"
MODEL_PATH   = "yolo11n.pt"

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("reframer")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# =============================================================
# FFMPEG
# =============================================================
def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    for p in [r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
              r"C:\ffmpeg\bin\ffmpeg.exe",
              "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.isfile(p):
            return p
    return None

FFMPEG = find_ffmpeg()


def transcode_h264(inp, out):
    if not FFMPEG:
        raise RuntimeError("FFmpeg not found")
    cmd = [
        FFMPEG, "-y", "-i", inp,
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an", out
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# =============================================================
# SMART REFRAMING ENGINE
# =============================================================
class PadelReframer:
    """
    Broadcast-style reframing with game-state awareness.

    Tracking modes:
      BALL_TRACKING  — ball detected, follow it smoothly
      PLAYER_HOLD    — ball lost, hold on players
      CENTER_DRIFT   — nothing detected, drift to center
      ANTICIPATE     — ball near edge, pre-move toward likely return

    The camera operator "personality":
      - Patient: holds position for a beat before reacting
      - Smooth: never snaps, always eases
      - Smart: weights ball vs player positions based on context
      - Cinematic: slight overshoot on fast pans, then settles
    """

    # YOLO COCO classes
    BALL_CLASS   = 32   # sports ball
    PERSON_CLASS = 0    # person

    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        if YOLO_AVAILABLE and os.path.isfile(model_path):
            self.model = YOLO(model_path)
            log.info("YOLO model loaded: %s", model_path)

    def reframe(self, input_path, output_path, progress_callback=None):
        """
        Gravity Core reframing.
        
        All detected objects (ball + up to 4 players) form a single
        weighted cluster. The camera targets the center of mass of 
        this cluster, where:
          - Ball weight: 2.5x (it's where the eyes go)
          - Each player weight: 1.0x (equal anchors, no area bias)
        
        When the ball is missing, players alone define the core.
        The camera always shows the maximum amount of the action.
        """
        if self.model is None:
            log.error("No YOLO model available")
            return None

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log.error("Cannot open video: %s", input_path)
            return None

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or fps > 120:
            fps = 15.0
        if total_frames <= 0:
            total_frames = 999999

        crop_w = int(H * (9 / 16))
        crop_w = min(crop_w, W)

        log.info("Input: %dx%d @ %.1ffps, %d frames", W, H, fps, total_frames)
        log.info("Crop: %dx%d", crop_w, H)

        # ── Weights ──
        BALL_WEIGHT_BASE = 2.5  # ball pulls camera 2.5x more than one player (when mid-frame)
        PLAYER_WEIGHT    = 1.0  # all players pull equally (no area bias)

        # Minimum detection confidence to trust a ball (filters false positives: lights, logos)
        BALL_CONF_FLOOR = 0.30

        # ── Smoothing ──
        # Camera speed is driven by ball velocity (rhythm of the game).
        # The faster the ball moves, the faster the camera follows.
        ALPHA_GENTLE = 0.06   # ball slow / between points
        ALPHA_NORMAL = 0.18   # ball at moderate rally speed
        ALPHA_FAST   = 0.38   # ball moving fast (aggressive rally)
        ALPHA_SPRINT = 0.55   # camera far behind — sprint to catch up (reduced from 0.72 to avoid snaps)
        ALPHA_DRIFT  = 0.03   # nothing detected, drift to center
        SPEED_SMOOTH = 0.50   # how quickly ball-speed estimate adapts (higher = faster reaction)

        # Speed thresholds (as fraction of frame width per frame)
        SPEED_FAST   = 0.040  # ball crosses >4% of frame width per frame
        SPEED_NORMAL = 0.014  # ball crosses >1.4% of frame width per frame

        # Maximum camera movement per frame (fraction of crop width) — prevents jarring snaps
        MAX_STEP_FRAC = 0.10  # camera moves at most 10% of crop width per frame

        # ── State ──
        x_smooth = W / 2.0
        last_core_x = W / 2.0  # remember last good core position
        last_player_centroid = W / 2.0  # fallback when ball is lost
        prev_ball_x = None        # for computing ball velocity
        prev_ball_y = None        # for computing ball vertical velocity
        ball_speed_smooth = 0.0   # smoothed ball speed (px/frame)
        ball_vy_smooth = 0.0      # smoothed vertical velocity (negative = going up)

        # Write temp then transcode
        temp_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (crop_w, H))

        if not writer.isOpened():
            log.error("Cannot create VideoWriter")
            cap.release()
            return None

        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── Detect everything ──
            ball_x = None
            ball_y = None
            player_xs = []   # list of x-positions for each player

            try:
                results = self.model.track(
                    frame, persist=True,
                    classes=[self.BALL_CLASS, self.PERSON_CLASS],
                    verbose=False
                )

                if results and len(results[0].boxes) > 0:
                    det_classes = results[0].boxes.cls.cpu().numpy()
                    det_xywh = results[0].boxes.xywh.cpu().numpy()
                    det_confs = results[0].boxes.conf.cpu().numpy()

                    # Ball — take highest confidence detection above floor
                    ball_mask = det_classes == self.BALL_CLASS
                    if ball_mask.any():
                        ball_boxes = det_xywh[ball_mask]
                        ball_confs = det_confs[ball_mask]
                        best_idx = np.argmax(ball_confs)
                        if ball_confs[best_idx] >= BALL_CONF_FLOOR:
                            ball_x = float(ball_boxes[best_idx][0])
                            ball_y = float(ball_boxes[best_idx][1])

                    # Players — collect ALL their x-positions equally
                    person_mask = det_classes == self.PERSON_CLASS
                    if person_mask.any():
                        person_boxes = det_xywh[person_mask]
                        player_xs = [float(px) for px in person_boxes[:, 0]]

            except Exception as e:
                if frame_idx < 3:
                    log.warning("Detection error frame %d: %s", frame_idx, e)

            # ── Track player centroid (always available fallback) ──
            if player_xs:
                last_player_centroid = float(np.mean(player_xs))

            # ── Track ball velocity (drives camera speed) ──
            if ball_x is not None:
                if prev_ball_x is not None:
                    raw_speed = abs(ball_x - prev_ball_x)
                    ball_speed_smooth = ball_speed_smooth * (1 - SPEED_SMOOTH) + raw_speed * SPEED_SMOOTH
                    # Use the peak of raw vs smoothed so sudden fast moves register instantly
                    effective_ball_speed = max(raw_speed, ball_speed_smooth)
                else:
                    effective_ball_speed = ball_speed_smooth
                prev_ball_x = ball_x

                # Track vertical velocity: negative = ball going up
                if prev_ball_y is not None:
                    raw_vy = ball_y - prev_ball_y
                    ball_vy_smooth = ball_vy_smooth * 0.6 + raw_vy * 0.4
                prev_ball_y = ball_y
            else:
                ball_speed_smooth *= 0.88  # decay speed gradually when ball lost
                effective_ball_speed = ball_speed_smooth
                ball_vy_smooth *= 0.80     # decay vertical velocity when ball lost

            # ── Ball weight: reduce when ball is near top of frame (about to exit) ──
            # A ball in the top 20% of the frame is likely mid-arc / leaving view.
            # Reducing its pull prevents the camera from chasing it off-screen.
            if ball_x is not None and ball_y is not None:
                if ball_y < H * 0.15:
                    effective_ball_weight = 0.4   # near top edge — barely follow
                elif ball_y < H * 0.25:
                    frac = (ball_y - H * 0.15) / (H * 0.10)
                    effective_ball_weight = 0.4 + frac * (BALL_WEIGHT_BASE - 0.4)
                else:
                    effective_ball_weight = BALL_WEIGHT_BASE
            else:
                effective_ball_weight = 0.0

            # ── Compute gravity core ──
            # Every detected object is a point with a weight.
            # Core = weighted average of all points.
            points = []
            weights = []

            if ball_x is not None:
                points.append(ball_x)
                weights.append(effective_ball_weight)

            for px in player_xs:
                points.append(px)
                weights.append(PLAYER_WEIGHT)

            if len(points) > 0:
                # Weighted center of mass
                points = np.array(points)
                weights = np.array(weights)
                core_x = float(np.average(points, weights=weights))
                last_core_x = core_x

                # Alpha driven by ball speed — fast ball = fast camera
                speed_ratio = effective_ball_speed / W
                if speed_ratio > SPEED_FAST:
                    speed_alpha = ALPHA_FAST
                elif speed_ratio > SPEED_NORMAL:
                    speed_alpha = ALPHA_NORMAL
                else:
                    speed_alpha = ALPHA_GENTLE

                # Position shift — sprint when camera is far behind
                shift = abs(core_x - x_smooth) / W
                if shift > 0.25:
                    shift_alpha = ALPHA_SPRINT  # too far behind — race to catch up
                elif shift > 0.12:
                    shift_alpha = ALPHA_FAST
                elif shift > 0.06:
                    shift_alpha = ALPHA_NORMAL
                else:
                    shift_alpha = ALPHA_GENTLE

                alpha = max(speed_alpha, shift_alpha)

            else:
                # Nothing detected — if ball was going up when lost, snap back to
                # players quickly (ball left the frame). Otherwise drift gently.
                ball_was_rising = ball_vy_smooth < -(H * 0.008)
                if ball_was_rising and player_xs:
                    core_x = last_player_centroid
                    alpha = ALPHA_NORMAL  # return to players promptly
                else:
                    core_x = last_core_x * 0.9 + (W / 2.0) * 0.1
                    alpha = ALPHA_DRIFT
                last_core_x = core_x

            # ── Smooth camera movement with per-frame movement cap ──
            target_x = x_smooth * (1 - alpha) + core_x * alpha
            max_step = crop_w * MAX_STEP_FRAC
            x_smooth = x_smooth + float(np.clip(target_x - x_smooth, -max_step, max_step))

            # ── Clamp and crop ──
            x1 = int(max(0, min(W - crop_w, x_smooth - crop_w / 2)))
            writer.write(frame[0:H, x1:x1 + crop_w])
            frame_idx += 1

            # Progress
            if progress_callback and frame_idx % 30 == 0:
                pct = min(99, int(frame_idx / total_frames * 100))
                progress_callback(pct, frame_idx, total_frames)

        cap.release()
        writer.release()

        elapsed = time.time() - start_time
        processing_fps = frame_idx / elapsed if elapsed > 0 else 0

        log.info("Reframe done: %d frames in %.1fs (%.1f fps)",
                 frame_idx, elapsed, processing_fps)

        # Transcode to H.264
        try:
            transcode_h264(temp_path, output_path)
            os.remove(temp_path)
            log.info("Output: %s", output_path)
        except Exception as e:
            log.error("Transcode failed: %s", e)
            if os.path.exists(temp_path):
                os.rename(temp_path, output_path)

        return {
            "frames": frame_idx,
            "duration": round(frame_idx / fps, 1),
            "processing_time": round(elapsed, 1),
            "processing_fps": round(processing_fps, 1),
            "output_resolution": f"{crop_w}x{H}"
        }


# Global reframer instance
reframer = PadelReframer()

# Track processing jobs
jobs = {}


# =============================================================
# API ROUTES
# =============================================================

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    safe_name = f"upload_{ts}{ext}"
    input_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(input_path)

    output_name = f"vertical_{ts}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Create job
    job_id = ts
    jobs[job_id] = {"status": "processing", "progress": 0, "result": None}

    import threading

    def process():
        def on_progress(pct, current, total):
            jobs[job_id]["progress"] = pct

        result = reframer.reframe(input_path, output_path, progress_callback=on_progress)

        if result:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"] = {
                **result,
                "filename": output_name,
                "download_url": f"/download/{output_name}",
                "preview_url": f"/preview/{output_name}"
            }
        else:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Processing failed"

    t = threading.Thread(target=process, daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "status": "processing"})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/preview/<filename>")
def preview_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4")


# =============================================================
# FRONTEND
# =============================================================

PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Padel Reframer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

  :root {
    --bg: #08090a;
    --surface: #111214;
    --surface-2: #191b1e;
    --border: #222427;
    --border-hover: #333639;
    --text: #e8e8e8;
    --text-dim: #6b7280;
    --text-muted: #3d4149;
    --accent: #10b981;
    --accent-soft: rgba(16,185,129,0.08);
    --accent-glow: rgba(16,185,129,0.15);
    --warn: #f59e0b;
    --radius: 14px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Instrument Sans', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Grain overlay ── */
  body::after {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    z-index: 9999;
  }

  .layout {
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 24px;
  }

  /* ── Header ── */
  header {
    padding: 32px 0 48px;
    border-bottom: 1px solid var(--border);
  }

  .logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 20px;
  }

  .logo span { color: var(--accent); }

  header h1 {
    font-size: clamp(28px, 5vw, 42px);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.5px;
  }

  header p {
    margin-top: 12px;
    color: var(--text-dim);
    font-size: 15px;
    max-width: 520px;
    line-height: 1.6;
  }

  /* ── Upload zone ── */
  .main-area {
    padding: 48px 0 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: start;
  }

  @media (max-width: 768px) {
    .main-area { grid-template-columns: 1fr; }
  }

  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 56px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    background: var(--surface);
  }

  .upload-zone:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
  }

  .upload-zone.dragging {
    border-color: var(--accent);
    background: var(--accent-soft);
    transform: scale(1.01);
  }

  .upload-zone.has-file {
    border-style: solid;
    border-color: var(--border-hover);
    padding: 24px;
    cursor: default;
  }

  .upload-icon {
    width: 48px; height: 48px;
    margin: 0 auto 16px;
    color: var(--text-dim);
    transition: color 0.3s;
  }

  .upload-zone:hover .upload-icon { color: var(--accent); }

  .upload-title {
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 6px;
  }

  .upload-hint {
    font-size: 13px;
    color: var(--text-dim);
  }

  .file-info {
    display: none;
    text-align: left;
  }

  .upload-zone.has-file .file-info { display: block; }
  .upload-zone.has-file .upload-placeholder { display: none; }

  .file-preview {
    width: 100%;
    border-radius: 10px;
    background: #000;
    margin-bottom: 14px;
  }

  .file-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-dim);
    word-break: break-all;
    margin-bottom: 16px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 28px;
    border: none;
    border-radius: 10px;
    font-family: 'Instrument Sans', sans-serif;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
  }

  .btn-primary {
    background: var(--accent);
    color: #000;
    width: 100%;
    justify-content: center;
  }

  .btn-primary:hover { filter: brightness(1.1); transform: translateY(-1px); }
  .btn-primary:active { transform: translateY(0); }

  .btn-primary:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transform: none;
    filter: none;
  }

  .btn-secondary {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

  .change-file {
    display: inline-block;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    margin-top: 10px;
    border-bottom: 1px solid transparent;
    transition: all 0.2s;
  }

  .change-file:hover { color: var(--accent); border-bottom-color: var(--accent); }

  /* ── Result panel ── */
  .result-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    min-height: 300px;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    padding: 18px 20px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-dim);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .panel-body {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px;
  }

  .idle-state {
    text-align: center;
    color: var(--text-muted);
  }

  .idle-state svg {
    width: 40px; height: 40px;
    margin-bottom: 12px;
    opacity: 0.3;
  }

  .idle-state p {
    font-size: 13px;
  }

  /* ── Progress ── */
  .progress-container {
    width: 100%;
    display: none;
  }

  .progress-container.active { display: block; }

  .progress-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
  }

  .progress-bar-outer {
    width: 100%;
    height: 6px;
    background: var(--surface-2);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-bar-inner {
    height: 100%;
    width: 0%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.4s ease;
  }

  .progress-detail {
    margin-top: 16px;
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    line-height: 2;
  }

  /* ── Result video ── */
  .result-video-container {
    width: 100%;
    display: none;
  }

  .result-video-container.active { display: block; }

  .result-video {
    width: 100%;
    max-height: 60vh;
    border-radius: 8px;
    background: #000;
    display: block;
  }

  .result-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    margin-top: 16px;
    border-radius: 10px;
    overflow: hidden;
  }

  .stat {
    background: var(--surface-2);
    padding: 14px 16px;
    text-align: center;
  }

  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 700;
    color: var(--accent);
  }

  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-top: 4px;
  }

  .result-actions {
    display: flex;
    gap: 10px;
    margin-top: 16px;
  }

  .result-actions .btn { flex: 1; justify-content: center; }

  /* ── How it works ── */
  .how-section {
    border-top: 1px solid var(--border);
    padding: 48px 0;
  }

  .how-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--text-dim);
    margin-bottom: 32px;
  }

  .how-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
  }

  .how-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    transition: border-color 0.3s;
  }

  .how-card:hover { border-color: var(--border-hover); }

  .how-card-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    color: var(--accent);
    opacity: 0.5;
    margin-bottom: 12px;
  }

  .how-card h3 {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .how-card p {
    font-size: 13px;
    color: var(--text-dim);
    line-height: 1.6;
  }

  /* ── Animations ── */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .animate-in {
    animation: fadeUp 0.5s ease forwards;
    opacity: 0;
  }

  .animate-in:nth-child(1) { animation-delay: 0.05s; }
  .animate-in:nth-child(2) { animation-delay: 0.1s; }
  .animate-in:nth-child(3) { animation-delay: 0.15s; }
  .animate-in:nth-child(4) { animation-delay: 0.2s; }
</style>
</head>
<body>
<div class="layout">

  <header>
    <div class="logo animate-in">PADEL<span>FRAME</span></div>
    <h1 class="animate-in">Smart Vertical Reframe</h1>
    <p class="animate-in">Upload padel footage and get a broadcast-quality 9:16 vertical crop. The AI tracks ball movement and player positions to follow the action like a real camera operator.</p>
  </header>

  <div class="main-area">

    <!-- Left: Upload -->
    <div>
      <input type="file" id="fileInput" accept="video/*" style="display:none">

      <!-- Drop zone — only used for selecting a file -->
      <div class="upload-zone" id="uploadZone">
        <div class="upload-placeholder">
          <svg class="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <div class="upload-title">Drop your video here</div>
          <div class="upload-hint">MP4, MOV, AVI — up to 500MB</div>
        </div>

        <div class="file-info">
          <video class="file-preview" id="previewVideo" muted playsinline></video>
          <div class="file-name" id="fileName"></div>
        </div>
      </div>

      <!-- Action buttons OUTSIDE the zone to prevent event conflicts -->
      <div class="upload-actions" id="uploadActions" style="display:none; margin-top: 14px;">
        <button class="btn btn-primary" id="processBtn" type="button">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polygon points="5,3 19,12 5,21" fill="currentColor" stroke="none"/>
          </svg>
          Reframe to Vertical
        </button>
        <div class="change-file" id="changeFile">choose a different file</div>
      </div>
    </div>

    <!-- Right: Result -->
    <div class="result-panel">
      <div class="panel-header">
        <span>Output</span>
        <span id="panelStatus">waiting</span>
      </div>

      <div class="panel-body" id="panelBody">
        <div class="idle-state" id="idleState">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="2" y="2" width="20" height="20" rx="2"/>
            <line x1="2" y1="14" x2="22" y2="14"/>
            <line x1="8" y1="2" x2="8" y2="14"/>
          </svg>
          <p>Upload a video to begin</p>
        </div>

        <div class="progress-container" id="progressContainer">
          <div class="progress-label">
            <span id="progressText">Processing...</span>
            <span id="progressPct">0%</span>
          </div>
          <div class="progress-bar-outer">
            <div class="progress-bar-inner" id="progressBar"></div>
          </div>
          <div class="progress-detail" id="progressDetail">
            Detecting ball &amp; players...<br>
            Calculating camera path...
          </div>
        </div>

        <div class="result-video-container" id="resultContainer">
          <video class="result-video" id="resultVideo" controls playsinline></video>
          <div class="result-stats" id="resultStats"></div>
          <div class="result-actions" id="resultActions"></div>
        </div>
      </div>
    </div>

  </div>

  <!-- How it works -->
  <div class="how-section">
    <div class="how-title">How the AI camera works</div>
    <div class="how-grid">
      <div class="how-card animate-in">
        <div class="how-card-num">01</div>
        <h3>Ball Tracking</h3>
        <p>YOLO11 detects the ball 12× per second. Adaptive smoothing follows fast shots tightly, slow rallies gently.</p>
      </div>
      <div class="how-card animate-in">
        <div class="how-card-num">02</div>
        <h3>Player Awareness</h3>
        <p>When the ball is lost, the camera smoothly shifts to frame the players — weighted by proximity, not just position.</p>
      </div>
      <div class="how-card animate-in">
        <div class="how-card-num">03</div>
        <h3>Momentum Physics</h3>
        <p>Slight overshoot on fast pans then settle — the same inertia a real camera operator has when swinging a lens.</p>
      </div>
      <div class="how-card animate-in">
        <div class="how-card-num">04</div>
        <h3>No Cuts. Ever.</h3>
        <p>Every transition is smoothed. The camera holds, drifts, or eases — it never snaps from one position to another.</p>
      </div>
    </div>
  </div>

</div>

<script>
  const zone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const previewVideo = document.getElementById('previewVideo');
  const fileName = document.getElementById('fileName');
  const uploadActions = document.getElementById('uploadActions');
  const processBtn = document.getElementById('processBtn');
  const changeFile = document.getElementById('changeFile');
  const panelStatus = document.getElementById('panelStatus');
  const panelBody = document.getElementById('panelBody');
  const idleState = document.getElementById('idleState');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  const progressPct = document.getElementById('progressPct');
  const progressDetail = document.getElementById('progressDetail');
  const resultContainer = document.getElementById('resultContainer');
  const resultVideo = document.getElementById('resultVideo');
  const resultStats = document.getElementById('resultStats');
  const resultActions = document.getElementById('resultActions');

  let selectedFile = null;

  // Drag & drop on the zone
  ['dragenter', 'dragover'].forEach(e =>
    zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.add('dragging'); })
  );
  ['dragleave', 'drop'].forEach(e =>
    zone.addEventListener(e, ev => { ev.preventDefault(); zone.classList.remove('dragging'); })
  );
  zone.addEventListener('drop', ev => {
    const files = ev.dataTransfer.files;
    if (files.length) handleFile(files[0]);
  });

  // Click on zone opens file picker (always — zone is just for selection)
  zone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  changeFile.addEventListener('click', () => {
    resetUpload();
    fileInput.click();
  });

  function handleFile(file) {
    if (!file.type.startsWith('video/') && !file.name.match(/\.(mp4|mov|avi|mkv|webm)$/i)) {
      alert('Please select a video file');
      return;
    }
    selectedFile = file;
    zone.classList.add('has-file');
    uploadActions.style.display = 'block';

    const url = URL.createObjectURL(file);
    previewVideo.src = url;
    previewVideo.play().catch(() => {});
    setTimeout(() => { previewVideo.pause(); previewVideo.currentTime = 1; }, 1500);

    const sizeMB = (file.size / (1024*1024)).toFixed(1);
    fileName.textContent = `${file.name} (${sizeMB} MB)`;
    processBtn.disabled = false;
  }

  function resetUpload() {
    selectedFile = null;
    zone.classList.remove('has-file');
    uploadActions.style.display = 'none';
    previewVideo.src = '';
    fileInput.value = '';
    processBtn.disabled = true;
  }

  // Process — button is OUTSIDE the zone, no event conflicts
  processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    processBtn.disabled = true;

    // Show progress
    idleState.style.display = 'none';
    resultContainer.classList.remove('active');
    progressContainer.classList.add('active');
    panelStatus.textContent = 'processing';
    progressBar.style.width = '0%';

    const phases = [
      'Analyzing video dimensions...',
      'Detecting ball &amp; players...',
      'Computing camera trajectory...',
      'Applying smooth transitions...',
      'Encoding H.264 output...'
    ];

    // Upload
    const form = new FormData();
    form.append('video', selectedFile);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();

      if (data.error) {
        alert(data.error);
        processBtn.disabled = false;
        return;
      }

      // Poll job status
      const jobId = data.job_id;
      const poll = setInterval(async () => {
        try {
          const jr = await fetch(`/api/job/${jobId}`);
          const job = await jr.json();

          const pct = job.progress || 0;
          progressBar.style.width = pct + '%';
          progressPct.textContent = pct + '%';

          const phase = phases[Math.min(Math.floor(pct / 22), phases.length - 1)];
          progressText.innerHTML = phase;

          if (pct > 60) {
            progressDetail.innerHTML = 'Smoothing camera movements...<br>Almost there...';
          }

          if (job.status === 'done') {
            clearInterval(poll);
            showResult(job.result);
          } else if (job.status === 'error') {
            clearInterval(poll);
            panelStatus.textContent = 'error';
            progressText.textContent = 'Processing failed';
            processBtn.disabled = false;
          }
        } catch (e) {}
      }, 800);

    } catch (e) {
      alert('Upload failed: ' + e.message);
      processBtn.disabled = false;
    }
  });

  function showResult(result) {
    progressContainer.classList.remove('active');
    resultContainer.classList.add('active');
    panelStatus.textContent = 'done';

    resultVideo.src = result.preview_url;

    resultStats.innerHTML = `
      <div class="stat">
        <div class="stat-value">${result.duration}s</div>
        <div class="stat-label">Duration</div>
      </div>
      <div class="stat">
        <div class="stat-value">${result.processing_fps}</div>
        <div class="stat-label">Proc. FPS</div>
      </div>
      <div class="stat">
        <div class="stat-value">${result.output_resolution}</div>
        <div class="stat-label">Resolution</div>
      </div>
    `;

    resultActions.innerHTML = `
      <a class="btn btn-primary" href="${result.download_url}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        Download
      </a>
      <button class="btn btn-secondary" onclick="resetAll()">
        New Video
      </button>
    `;

    processBtn.disabled = false;
  }

  function resetAll() {
    resetUpload();
    resultContainer.classList.remove('active');
    progressContainer.classList.remove('active');
    idleState.style.display = '';
    panelStatus.textContent = 'waiting';
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE_HTML


# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    if not YOLO_AVAILABLE:
        print("\n⚠  ultralytics not installed!")
        print("   pip install ultralytics\n")
    if not FFMPEG:
        print("\n⚠  FFmpeg not found!")
        print("   Install: winget install -e --id Gyan.FFmpeg\n")
    if not os.path.isfile(MODEL_PATH):
        print(f"\n⚠  YOLO model not found at {MODEL_PATH}")
        print("   Place yolo11n.pt in the same folder\n")

    print("\n" + "="*50)
    print("  PADEL SMART REFRAMER")
    print("  http://localhost:5001")
    print("="*50 + "\n")

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)