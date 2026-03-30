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
        Broadcast-style reframing with game-state awareness.

        Four game states drive all camera decisions:
          RALLY    — ball detected and moving  → predictive tracking, ball leads camera
          LOFT     — ball going upward fast    → hold on players, ignore arc position
          RECOVERY — ball just disappeared     → physics prediction fades into players
          IDLE     — no ball for a while       → stable wide shot on players

        Camera movement uses a spring-damper model (position + velocity) so it
        naturally accelerates into a pan and decelerates smoothly into the target,
        with no snapping or exponential-smoothing artifacts.
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

        # ══════════════════════════════════════════════════════════
        #  CONSTANTS
        # ══════════════════════════════════════════════════════════

        # Detection
        BALL_CONF_FLOOR  = 0.30     # ignore ball detections below this confidence
        MAX_BALL_JUMP    = W * 0.18 # max believable ball jump per frame (outlier rejection)

        # Gravity weights (RALLY state)
        BALL_WEIGHT_BASE = 2.5
        PLAYER_WEIGHT    = 1.0

        # Center bias — always pull target toward W/2 by this fraction.
        # Prevents camera from drifting into glass/walls between points.
        CENTER_BIAS = 0.22

        # Spring-damper camera (deliberately soft — smoothness > speed):
        #   cam_vel += spring_k * (target_smooth - cam_pos)
        #   cam_vel *= DAMPING
        #   cam_pos += cam_vel
        SPRING_RALLY  = 0.11        # rally tracking
        SPRING_IDLE   = 0.05        # between points / loft / recovery
        SPRING_SPRINT = 0.20        # camera significantly behind (was 0.38 — reduced to kill lurches)
        DAMPING       = 0.82        # higher = smoother stop (was 0.72)
        MAX_STEP_FRAC = 0.055       # hard cap per frame — 5.5% of crop_w (was 10%)

        # Target smoothing — the target itself is low-pass filtered before the spring
        # sees it. This is the main fix for jitter: even a sudden ball jump creates
        # a gradual target shift, so the camera can't lunge.
        # T_* = retention per frame (higher = slower target movement)
        T_RALLY    = 0.80           # target updates at moderate pace during rally
        T_IDLE     = 0.94           # target barely moves between points
        T_LOFT     = 0.92           # hold steady during ball arc
        T_RECOVERY = 0.90           # gentle fade back to players

        # Transition rate limiter — track direction reversals; if too many in a short
        # window, penalise spring_k so the camera commits to one direction.
        REVERSAL_DECAY    = 0.92    # reversal score decays each frame
        REVERSAL_PENALISE = 3.0     # score above this → apply penalty
        REVERSAL_FACTOR   = 0.40    # spring_k multiplied by this when oscillating

        # Predictive lookahead (RALLY state only)
        LOOKAHEAD_MIN = 3
        LOOKAHEAD_MAX = 10          # reduced from 14 — less overshoot
        SPEED_FAST    = W * 0.040

        # Ball velocity smoothing
        VX_SMOOTH = 0.30            # slower adaptation → less reaction to single-frame noise
        VY_SMOOTH = 0.35

        # Ball physics prediction (RECOVERY state)
        PRED_VX_DECAY = 0.88
        PRED_VY_DECAY = 0.95
        GRAVITY       = 0.6
        BALL_LOST_MAX = 35          # longer prediction window before giving up (was 22)

        # LOFT detection
        LOFT_VY_THRESH = -(H * 0.010)

        # Y-position ball weight
        LOFT_Y_TOP   = 0.15
        LOFT_Y_TRANS = 0.28

        # Edge-proximity boost (gentle — no more SPRINT lurch)
        EDGE_MARGIN_FRAC  = 0.14
        EDGE_SPRING_BOOST = 1.6     # multiply spring_k (not replace with SPRINT)

        # Player containment
        # Stable bounds update slowly so missed detections don't suddenly drift the camera.
        STABLE_PLAYER_ALPHA = 0.06  # slow EMA — persists through detection gaps
        PLAYER_MARGIN       = 25    # px: outermost player must stay this far from crop edge

        # ══════════════════════════════════════════════════════════
        #  STATE
        # ══════════════════════════════════════════════════════════

        cam_pos       = float(W / 2)
        cam_vel       = 0.0
        target_smooth = float(W / 2)   # smoothed target — what the spring actually chases

        # Ball state estimator
        est_x  = None
        est_y  = None
        est_vx = 0.0
        est_vy = 0.0
        ball_lost_frames = 0

        # Stable player bounds — updated slowly each frame
        stable_p_left  = float(W * 0.25)
        stable_p_right = float(W * 0.75)

        # Reversal tracker
        prev_vel_sign  = 0
        reversal_score = 0.0

        # ══════════════════════════════════════════════════════════
        #  VIDEO WRITER
        # ══════════════════════════════════════════════════════════

        temp_path = output_path + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (crop_w, H))

        if not writer.isOpened():
            log.error("Cannot create VideoWriter")
            cap.release()
            return None

        frame_idx = 0
        start_time = time.time()

        # ══════════════════════════════════════════════════════════
        #  MAIN LOOP
        # ══════════════════════════════════════════════════════════

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── 1. Detect ──────────────────────────────────────────
            det_ball_x = None
            det_ball_y = None
            player_xs  = []

            try:
                results = self.model.track(
                    frame, persist=True,
                    classes=[self.BALL_CLASS, self.PERSON_CLASS],
                    verbose=False
                )
                if results and len(results[0].boxes) > 0:
                    cls   = results[0].boxes.cls.cpu().numpy()
                    xywh  = results[0].boxes.xywh.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()

                    ball_mask = cls == self.BALL_CLASS
                    if ball_mask.any():
                        bboxes = xywh[ball_mask]
                        bconfs = confs[ball_mask]
                        best   = np.argmax(bconfs)
                        if bconfs[best] >= BALL_CONF_FLOOR:
                            det_ball_x = float(bboxes[best][0])
                            det_ball_y = float(bboxes[best][1])

                    person_mask = cls == self.PERSON_CLASS
                    if person_mask.any():
                        player_xs = list(xywh[person_mask][:, 0].astype(float))

            except Exception as e:
                if frame_idx < 3:
                    log.warning("Detection error frame %d: %s", frame_idx, e)

            # ── 2. Player bounds tracker ───────────────────────────
            # Stable bounds update slowly — if YOLO misses a player for a few frames,
            # the camera doesn't suddenly drift toward the empty side.
            if player_xs:
                p_left  = min(player_xs)
                p_right = max(player_xs)
                stable_p_left  = stable_p_left  * (1 - STABLE_PLAYER_ALPHA) + p_left  * STABLE_PLAYER_ALPHA
                stable_p_right = stable_p_right * (1 - STABLE_PLAYER_ALPHA) + p_right * STABLE_PLAYER_ALPHA

            player_midpoint = (stable_p_left + stable_p_right) / 2.0
            player_span     = stable_p_right - stable_p_left
            # Slack = how much room remains in the crop after fitting all players.
            # Ball can only shift the camera within this budget.
            slack = max(0.0, crop_w - player_span - 2 * PLAYER_MARGIN)

            # ── 3. Ball state estimator ────────────────────────────
            if det_ball_x is not None:
                if est_x is not None:
                    # Outlier rejection: if detection is far from predicted position,
                    # it's likely a false positive or tracking glitch — partially trust it.
                    predicted_x = est_x + est_vx
                    predicted_y = est_y + est_vy
                    jump = abs(det_ball_x - predicted_x)
                    if jump > MAX_BALL_JUMP:
                        trust = max(0.25, 1.0 - (jump - MAX_BALL_JUMP) / (W * 0.30))
                        det_ball_x = predicted_x * (1 - trust) + det_ball_x * trust
                        det_ball_y = predicted_y * (1 - trust) + det_ball_y * trust

                    raw_vx = det_ball_x - est_x
                    raw_vy = det_ball_y - est_y
                    est_vx = est_vx * (1 - VX_SMOOTH) + raw_vx * VX_SMOOTH
                    est_vy = est_vy * (1 - VY_SMOOTH) + raw_vy * VY_SMOOTH

                est_x = det_ball_x
                est_y = det_ball_y
                ball_lost_frames = 0
            else:
                # Ball not detected — run physics prediction
                ball_lost_frames += 1
                if est_x is not None and ball_lost_frames <= BALL_LOST_MAX:
                    est_x  += est_vx
                    est_y  += est_vy
                    est_vx *= PRED_VX_DECAY
                    est_vy  = est_vy * PRED_VY_DECAY + GRAVITY
                    if est_x < 0 or est_x > W or est_y < 0 or est_y > H * 1.15:
                        est_x = None
                elif ball_lost_frames > BALL_LOST_MAX:
                    est_x = None

            # ── 4. Game state ──────────────────────────────────────
            ball_rising   = est_vy < LOFT_VY_THRESH
            ball_near_top = est_y is not None and est_y < H * LOFT_Y_TOP

            if est_x is None:
                game_state = "IDLE"
            elif ball_rising or ball_near_top:
                game_state = "LOFT"
            elif ball_lost_frames > 0:
                game_state = "RECOVERY"
            else:
                game_state = "RALLY"

            # ── 5. Compute raw target and spring_k per state ───────
            if game_state == "IDLE":
                raw_target = player_midpoint
                spring_k   = SPRING_IDLE
                t_retain   = T_IDLE

            elif game_state == "LOFT":
                # Hold on player midpoint; tiny hint of ball X so we're not blind to it
                raw_target = player_midpoint * 0.90 + est_x * 0.10
                spring_k   = SPRING_IDLE
                t_retain   = T_LOFT

            elif game_state == "RECOVERY":
                fade       = 1.0 - (ball_lost_frames / BALL_LOST_MAX)
                raw_target = est_x * fade + player_midpoint * (1.0 - fade)
                spring_k   = SPRING_IDLE
                t_retain   = T_RECOVERY

            else:  # RALLY
                speed_abs = abs(est_vx)
                lookahead = LOOKAHEAD_MIN + (LOOKAHEAD_MAX - LOOKAHEAD_MIN) * min(1.0, speed_abs / SPEED_FAST)
                pred_x    = float(np.clip(est_x + est_vx * lookahead, 0, W))

                if est_y < H * LOFT_Y_TOP:
                    eff_ball_w = 0.3
                elif est_y < H * LOFT_Y_TRANS:
                    frac       = (est_y - H * LOFT_Y_TOP) / (H * (LOFT_Y_TRANS - LOFT_Y_TOP))
                    eff_ball_w = 0.3 + frac * (BALL_WEIGHT_BASE - 0.3)
                else:
                    eff_ball_w = BALL_WEIGHT_BASE

                # Ball shifts camera from player_midpoint, but only within slack budget.
                # This guarantees all players stay in frame while still following the ball.
                ball_shift = (pred_x - player_midpoint) * (eff_ball_w / (eff_ball_w + PLAYER_WEIGHT * max(1, len(player_xs))))
                ball_shift = float(np.clip(ball_shift, -slack / 2, slack / 2))
                raw_target = player_midpoint + ball_shift

                gap_frac = abs(raw_target - cam_pos) / W
                if gap_frac > 0.25:
                    spring_k = SPRING_SPRINT
                elif gap_frac > 0.10:
                    spring_k = SPRING_RALLY * 1.20
                else:
                    spring_k = SPRING_RALLY

                # Edge-proximity: boost spring gently (not replace with sprint)
                crop_left    = cam_pos - crop_w / 2
                ball_in_crop = est_x - crop_left
                edge_margin  = crop_w * EDGE_MARGIN_FRAC
                if ball_in_crop < edge_margin or ball_in_crop > (crop_w - edge_margin):
                    spring_k = min(SPRING_SPRINT, spring_k * EDGE_SPRING_BOOST)

                t_retain = T_RALLY

            # ── 5b. Player-midpoint bias — always drift toward player_midpoint ──
            # Replaces old W/2 center bias: camera naturally rests where the players are,
            # not at the geometric center of the original frame.
            raw_target = raw_target * (1.0 - CENTER_BIAS) + player_midpoint * CENTER_BIAS

            # ── 5c. Smooth the target (low-pass filter) ────────────
            # The spring never sees raw target jumps — only the smoothed version.
            # This is the primary fix for jitter and hard transitions.
            target_smooth = target_smooth * t_retain + raw_target * (1.0 - t_retain)

            # ── 5d. Transition rate limiter ────────────────────────
            # If the camera has been reversing direction frequently, penalise
            # spring_k so it commits to one direction instead of oscillating.
            reversal_score *= REVERSAL_DECAY
            if cam_vel != 0:
                new_sign = 1 if cam_vel > 0 else -1
                if prev_vel_sign != 0 and new_sign != prev_vel_sign:
                    reversal_score += 1.0
                prev_vel_sign = new_sign
            if reversal_score > REVERSAL_PENALISE:
                spring_k *= REVERSAL_FACTOR

            # ── 6. Spring-damper camera update ────────────────────
            cam_vel += spring_k * (target_smooth - cam_pos)
            cam_vel *= DAMPING
            max_step = crop_w * MAX_STEP_FRAC
            cam_vel  = float(np.clip(cam_vel, -max_step, max_step))
            cam_pos += cam_vel

            # ── 6b. Hard player-visibility clamp ──────────────────
            # After all smoothing, guarantee outermost players are inside the crop.
            # Only applied when players actually fit within crop_w.
            if player_span <= crop_w - 2 * PLAYER_MARGIN:
                cam_min = stable_p_right - crop_w / 2 + PLAYER_MARGIN
                cam_max = stable_p_left  + crop_w / 2 - PLAYER_MARGIN
                if cam_min <= cam_max:  # sanity check
                    cam_pos = float(np.clip(cam_pos, cam_min, cam_max))

            # ── 7. Clamp and crop ──────────────────────────────────
            x1 = int(max(0, min(W - crop_w, cam_pos - crop_w / 2)))
            writer.write(frame[0:H, x1:x1 + crop_w])
            frame_idx += 1

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