"""
process_video.py — Offline Video Processing & Detection Pipeline
================================================================
Processes any uploaded video file through the full AI drone detection stack:
  • Slicing Aided Hyper Inference (SAHI) + TTA for small/distant drone detection
  • DeepSORT tracking with persistent Track IDs (#1, #2, etc.)
  • Multi-Frame Decision Engine (4-6 frames confirming, 8 frames locked)
  • Tactical HUD rendering with bounding boxes, crosshairs, and confirmation telemetry
  • Output video saved with bounding boxes & track IDs

Usage:
  1. Put your video in the 'test_videos' folder as 'input.mp4' (or any name).
  2. Run:
       python process_video.py
     Or specify a custom file:
       python process_video.py --input test_videos/my_drone_video.mp4
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console stdout reconfigure for Windows
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from detection.detector import Detection
from detection.tracker import DeepSORTTracker
from detection.decision_engine import DecisionEngine, DecisionResult, ThreatLevel, TargetThreatState
from detection.sahi_engine import SahiTTAEngine
from logging_system.event_logger import EventLogger
from logging_system.evidence_recorder import EvidenceRecorder
from logging_system.snapshot_saver import SnapshotSaver
from utils.bearing_calculator import BearingCalculator
from utils.distance_gps_estimator import DistanceGPSEstimator, DRONE_REFERENCE_SIZES


# ══════════════════════════════════════════════════════════════════
# Tactical HUD Color Palette
# ══════════════════════════════════════════════════════════════════
CLR_ACCENT     = (173, 165, 243)   # Soft lavender accent
CLR_ALERT      = (30, 30, 230)     # Tactical alert red
CLR_CRIMSON    = (35, 30, 154)     # Dark crimson
CLR_YELLOW     = (0, 215, 255)     # Suspicious candidate yellow (4-6 frames)
CLR_GREEN      = (50, 205, 50)     # Secure airspace green
CLR_CYAN       = (255, 220, 0)     # Acquiring cyan (1-3 frames)
CLR_WHITE      = (245, 245, 250)
CLR_HUD_BG     = (18, 12, 28)


def find_input_video(default_path: Path) -> Optional[Path]:
    """
    Search for a valid input video file at the default path or discover any supported video file
    (.mp4, .avi, .mov, .mkv) residing inside the 'test_videos' directory.

    Args:
        default_path: Initial candidate path to test for existence.

    Returns:
        Optional[Path]: Path to located video file if found, otherwise None.
    """
    if default_path.exists() and default_path.is_file():
        return default_path

    # Check test_videos folder for any video file
    search_dir = PROJECT_ROOT / "test_videos"
    if search_dir.exists():
        candidates = list(search_dir.glob("*.mp4")) + \
                     list(search_dir.glob("*.avi")) + \
                     list(search_dir.glob("*.mov")) + \
                     list(search_dir.glob("*.mkv"))
        if candidates:
            return candidates[0]

    return None


def process_video_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    model_path: str = "model/best.onnx",
    conf_thresh: float = 0.35,
    slice_size: int = 256,
    camera_heading: float = 90.0,
    cam_lat: float = 33.6844,
    cam_lon: float = 73.0479,
    drone_type: str = "generic_medium",
    drone_width_m: Optional[float] = None,
) -> None:
    """
    Execute the complete end-to-end drone detection, tracking, bearing calculation,
    distance/GPS estimation, multi-frame threat verification, and tactical HUD rendering on a video file.

    Args:
        input_path: Path to input video file.
        output_path: Destination path for annotated MP4 video (default: output_videos/<stem>_detected.mp4).
        model_path: Path to trained YOLO model weights (.onnx or .pt).
        conf_thresh: Minimum confidence score for raw candidate detections.
        slice_size: Pixel dimension for SAHI window slicing (e.g., 256).
        camera_heading: Compass azimuth heading of camera optical axis in degrees (0=North, 90=East).
        cam_lat: Geographic latitude of camera sensor in decimal degrees.
        cam_lon: Geographic longitude of camera sensor in decimal degrees.
        drone_type: Model key from DRONE_REFERENCE_SIZES for optical pinhole distance estimation.
        drone_width_m: Optional manual override of drone diagonal physical dimension in metres.
    """
    cam_card = BearingCalculator.degrees_to_cardinal(camera_heading)
    print("\n" + "=" * 65)
    print("🚁 OFFLINE VIDEO DRONE DETECTION & TRUE COMPASS BEARING")
    print("=" * 65)
    print(f" [INPUT]         {input_path}")
    print(f" [COMPASS HDG]   Camera facing {camera_heading:.1f} deg ({cam_card})")

    if not input_path.exists():
        print(f"\n [ERROR] Input video not found at: {input_path}")
        print(" Please place your video inside the 'test_videos' folder as 'input.mp4'")
        return

    # Determine output path
    if output_path is None:
        out_dir = PROJECT_ROOT / "output_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        output_path = out_dir / f"{stem}_detected.mp4"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f" [OUTPUT]        {output_path}")

    # Load YOLO model
    from ultralytics import YOLO
    model_file = PROJECT_ROOT / model_path
    if not model_file.exists():
        model_file = PROJECT_ROOT / "yolo11n.pt"

    print(f" [MODEL]         Loading weights: {model_file}")
    yolo_model = YOLO(str(model_file))
    print(f" [MODEL]         Ready! Classes: {yolo_model.names}")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"\n [ERROR] Could not open video file: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f" [VIDEO INFO]    Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        print(" [WARN] 'mp4v' codec failed, falling back to 'XVID'...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_path.with_suffix(".avi")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize subsystems
    config = {
        "logging": {
            "level": "INFO",
            "event_log_dir": str(PROJECT_ROOT / "logs" / "events"),
            "snapshot_dir":  str(PROJECT_ROOT / "logs" / "snapshots"),
            "max_log_days":  30,
            "event_ring_buffer_size": 100,
        },
        "tracker": {
            "confirmation_frames":   2,
            "max_age":               30,
            "n_init":                2,
            "max_cosine_distance":   0.40,
            "max_iou_distance":      0.90,
            "max_coast_frames":      5,
            "nn_budget":             100,
            "cross_camera_check":    False,
            "embedder":              None,
        },
        "bearing": {
            "camera_fov_horizontal": 85.0,
            "camera_fov_vertical": 60.0,
            "camera_heading_deg": camera_heading,
        },
        "cameras": {
            "video_file": {
                "source": str(input_path),
                "label": input_path.name,
                "latitude": cam_lat,
                "longitude": cam_lon,
                "azimuth_deg": camera_heading,
                "city": "Video Analysis Station",
            }
        }
    }

    event_logger = EventLogger(config)
    snapshot_saver = SnapshotSaver(config)
    evidence_recorder = EvidenceRecorder(PROJECT_ROOT / "logs" / "evidence_videos")
    tracker = DeepSORTTracker(config)
    bearing_calculator = BearingCalculator(config.get("bearing", {}))

    # Step 2: Distance Estimation + GPS Projection
    dist_cfg = {
        "camera_fov_horizontal": 85.0,
        "camera_fov_vertical":   60.0,
        "frame_width":  width,
        "frame_height": height,
        "drone_type":  drone_type,
        "camera_latitude":  cam_lat,
        "camera_longitude": cam_lon,
    }
    if drone_width_m is not None:
        dist_cfg["drone_real_width_m"] = drone_width_m
    dist_estimator = DistanceGPSEstimator(dist_cfg)
    print(f" [STEP 2]        DistanceGPSEstimator | Drone ref: {dist_estimator._drone_width_m:.3f}m | CAM GPS: ({cam_lat:.6f},{cam_lon:.6f})")

    # 4-6 frames verifying, 8 frames locked
    decision_engine = DecisionEngine(
        min_consecutive_hits=8,
        window_size=12,
        min_hits_in_window=7,
        min_confidence=conf_thresh,
        target_cooldown_sec=10.0,
    )

    # SAHI + TTA Engine for high resolution
    sahi_engine = SahiTTAEngine(
        slice_size=slice_size,
        overlap_ratio=0.25,
        iou_threshold=0.45,
        use_tta=True,
    )

    print("\n" + "=" * 65)
    print(" [PROCESSING] Starting detection, tracking & annotation...")
    print("              - Frames 1-3: Acquiring (Cyan)")
    print("              - Frames 4-6: Confirming Lock (Yellow)")
    print("              - Frame 8:    Target Confirmed Locked (Red)")
    print(f"              - Compass:    Camera Heading = {camera_heading:.1f} deg ({cam_card})")
    print("              - Angles:     True Compass Bearing (0-360 deg) & Relative Offsets")
    print("              - STEP 2:     Distance Estimation + GPS Lat/Long (Vincenty WGS-84)")
    print("=" * 65 + "\n")

    frame_idx = 0
    t_start = time.time()
    total_detections = 0
    confirmed_threat_events = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # 1. SAHI + TTA Detection
        raw_detections = sahi_engine.detect(
            yolo_model,
            frame,
            conf_threshold=conf_thresh,
            drone_classes=[0],
        )
        total_detections += len(raw_detections)

        # 2. DeepSORT Tracking (persistent Track IDs)
        confirmed_tracks = tracker.update(
            raw_detections,
            camera_id="video_file",
            frame_shape=(h, w),
            frame=frame,
        )

        active_targets = confirmed_tracks if confirmed_tracks else raw_detections

        # 3. Decision Engine (4-6 frames confirming, 8 frames locked)
        decision = decision_engine.evaluate(
            active_targets,
            camera_id="video_file"
        )

        # Check for alert trigger on frame 8 lock
        if decision.should_trigger_alert and decision.confirmed_threats:
            confirmed_threat_events += 1
            snap_path = snapshot_saver.save(frame, decision.confirmed_threats, camera_id="video_file")
            event_logger.log_detection(decision.confirmed_threats, camera_id="video_file", snapshot_path=snap_path)
            primary_ang = bearing_calculator.calculate_for_detection(decision.confirmed_threats[0])
            print(
                f"\n [ALERT @ Frame {frame_idx}] 🎯 TARGET LOCKED (8 Frames Passed): {len(decision.confirmed_threats)} drone(s) "
                f"| True Bearing: {primary_ang['true_bearing_deg']:.1f} deg ({primary_ang['cardinal_direction']}) "
                f"[Relative: {primary_ang['azimuth_str']}, {primary_ang['elevation_str']}]"
            )

        # 4. Render Tactical HUD with True Compass Bearing + GPS
        annotated_frame = render_hud(
            frame.copy(),
            active_targets,
            decision,
            bearing_calculator,
            dist_estimator,
            frame_idx,
            total_frames,
            fps,
            w, h
        )

        # Write to output video
        out.write(annotated_frame)

        # Progress indicator
        if frame_idx % 10 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            elapsed = time.time() - t_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r [PROGRESS] Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                f"Speed: {proc_fps:.1f} FPS | Targets Locked: {confirmed_threat_events}   "
            )
            sys.stdout.flush()

    cap.release()
    out.release()

    # Universal H.264 encoding for 100% media player compatibility
    temp_h264 = output_path.with_name(f"{output_path.stem}_compat.mp4")
    try:
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", str(output_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", str(temp_h264)]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if res.returncode == 0 and temp_h264.exists() and temp_h264.stat().st_size > 0:
            import shutil
            shutil.move(str(temp_h264), str(output_path))
            print(" [CODEC]         Universal H.264 MP4 encoded successfully!")
    except Exception:
        pass

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\n\n" + "=" * 65)
    print(" ✅ PROCESSING COMPLETE!")
    print("=" * 65)
    print(f" [OUTPUT FILE]   {output_path}")
    print(f" [FILE SIZE]     {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f" [FRAMES DONE]   {frame_idx} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f" [TOTAL DETS]    {total_detections} raw detections across video")
    print(f" [TARGET LOCKS]  {confirmed_threat_events} target lock confirmations")
    print(f" [SNAPSHOTS]     Saved in: logs/snapshots/")
    print("=" * 65 + "\n")


def render_hud(
    frame: np.ndarray,
    detections: List[Detection],
    decision: Optional[DecisionResult],
    bearing_calculator: Optional[BearingCalculator],
    dist_estimator: Optional[DistanceGPSEstimator],
    frame_idx: int,
    total_frames: int,
    video_fps: float,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Render tactical HUD overlay on the video frame, including boresight crosshair, target bounding boxes,
    corner brackets, bearing ray lines, 8-frame lock indicators, telemetry banners (bearing, ground distance,
    slant range, altitude, speed, motion state, and GPS coordinates), and status headers/footers.

    Args:
        frame: Source BGR image frame (numpy.ndarray).
        detections: List of current target detections or active DeepSORT tracks.
        decision: DecisionEngine evaluation result containing threat states and confirmations.
        bearing_calculator: Utility instance configured for relative angles and true compass bearings.
        dist_estimator: Utility instance computing pinhole distances and WGS-84 Vincenty GPS coordinates.
        frame_idx: Current frame index (1-based).
        total_frames: Total number of frames in video file.
        video_fps: Video frame rate.
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        numpy.ndarray: Annotated BGR frame with tactical HUD overlays.
    """
    target_states = decision.target_states if decision else {}

    # Center boresight (optical camera center 0.0 deg relative)
    cx_center = w // 2
    cy_center = h // 2
    cam_hdg = bearing_calculator.camera_heading if bearing_calculator else 90.0
    cam_card = BearingCalculator.degrees_to_cardinal(cam_hdg)
    cv2.drawMarker(frame, (cx_center, cy_center), (120, 120, 140), cv2.MARKER_CROSS, 14, 1)
    cv2.circle(frame, (cx_center, cy_center), 6, (120, 120, 140), 1)
    cv2.putText(frame, f"BORESIGHT HDG: {cam_hdg:.0f} deg ({cam_card})", (cx_center + 10, cy_center + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 170), 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        px1 = int(x1 * w); py1 = int(y1 * h)
        px2 = int(x2 * w); py2 = int(y2 * h)

        tid = getattr(det, "track_id", None)
        m = target_states.get(tid) if tid is not None else None

        # Step 1: True Compass Bearing & angles
        ang = bearing_calculator.calculate_for_detection(det) if bearing_calculator else None
        true_brg = ang["true_bearing_deg"] if ang else 0.0
        cardinal = ang["cardinal_direction"] if ang else "N"
        rel_az = ang["rel_azimuth_deg"] if ang else 0.0
        el_val = ang["elevation_deg"] if ang else 0.0
        los_ang = ang["total_angle_deg"] if ang else 0.0

        # Step 2: Distance Estimation + GPS Lat/Long + Speed & Motion
        gps_data = None
        if dist_estimator and ang:
            frame_time = frame_idx / max(1.0, video_fps)
            gps_data = dist_estimator.estimate_from_detection(
                det, ang, track_id=str(tid) if tid is not None else None, timestamp=frame_time
            )

        # Color & State Text based on confirmation frames
        if m and m.state == TargetThreatState.CONFIRMED:
            box_color = CLR_ALERT       # Red for 8+ frames locked
            state_text = f"TARGET LOCKED [8/8 frames]"
        elif m and m.state == TargetThreatState.COOLDOWN:
            box_color = CLR_ALERT
            state_text = f"LOCKED (Active Target)"
        elif m and m.state == TargetThreatState.SUSPICIOUS:
            box_color = CLR_YELLOW      # Yellow for 4-6 frames verifying
            state_text = f"CONFIRMING LOCK ({m.consecutive_hits}/8 frames) (LOS: {los_ang} deg)"
        else:
            box_color = CLR_CYAN        # Cyan for 1-3 frames acquiring
            hits = m.consecutive_hits if m else 1
            state_text = f"ACQUIRING ({hits}/8 frames) (LOS: {los_ang} deg)"

        # Bounding box
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

        # Corner brackets
        k = min(16, (px2 - px1) // 3, (py2 - py1) // 3)
        cv2.line(frame, (px1, py1), (px1 + k, py1), CLR_ACCENT, 3)
        cv2.line(frame, (px1, py1), (px1, py1 + k), CLR_ACCENT, 3)
        cv2.line(frame, (px2, py1), (px2 - k, py1), CLR_ACCENT, 3)
        cv2.line(frame, (px2, py1), (px2 - k, py1 + k), CLR_ACCENT, 3)
        cv2.line(frame, (px1, py2), (px1 + k, py2), CLR_ACCENT, 3)
        cv2.line(frame, (px1, py2), (px1, py2 - k), CLR_ACCENT, 3)
        cv2.line(frame, (px2, py2), (px2 - k, py2), CLR_ACCENT, 3)
        cv2.line(frame, (px2, py2), (px2 - k, py2 - k), CLR_ACCENT, 3)

        # Center target reticle
        cen_x = (px1 + px2) // 2
        cen_y = (py1 + py2) // 2
        cv2.drawMarker(frame, (cen_x, cen_y), CLR_CRIMSON, cv2.MARKER_CROSS, 14, 2)

        # Tactical Bearing Ray line from boresight center to target center
        cv2.line(frame, (cx_center, cy_center), (cen_x, cen_y), box_color, 1, cv2.LINE_AA)
        mid_x = (cx_center + cen_x) // 2
        mid_y = (cy_center + cen_y) // 2
        cv2.circle(frame, (mid_x, mid_y), 3, box_color, -1)
        cv2.putText(frame, f"BRG {true_brg:.1f} deg {cardinal}", (mid_x + 6, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_WHITE, 1, cv2.LINE_AA)

        # Label banner: Step 1 Bearing + Step 2 Distance/GPS + Speed & Motion State
        id_str = f"#{tid}" if tid is not None else "NEW"
        tag_title = f"{det.class_name.upper()} | ID: {id_str} | {det.confidence*100:.1f}% | BRG: {true_brg:.1f} deg ({cardinal})"
        tag_sub   = f"{state_text} [Az: {rel_az:+.1f} deg | El: {el_val:+.1f} deg | LOS: {los_ang} deg]"

        # Telemetry lines below bbox
        if gps_data:
            dist_line  = f"{gps_data['dist_str']} | {gps_data['grd_str']}"
            speed_line = gps_data.get("speed_str", "SPD: -- | STATUS: HOVERING")
            gps_line   = gps_data['gps_str']
            m_state    = gps_data.get("motion_state", "HOVERING")
            eta_val    = gps_data.get("eta_seconds")
            time_to_base_str = gps_data.get("time_to_base_str", "TIME TO BASE: N/A")
            closing_kmh = gps_data.get("closing_speed_kmh", 0.0)
        else:
            dist_line  = "DST: -- | GND: --"
            speed_line = "SPD: -- | STATUS: --"
            gps_line   = "GPS: NOT CONFIGURED"
            m_state    = "UNKNOWN"
            eta_val    = None
            time_to_base_str = "TIME TO BASE: N/A"
            closing_kmh = 0.0

        # Flight motion status dynamic color code
        if m_state == "INBOUND":
            spd_color = CLR_ALERT       # Red (Threat Closing In / Approaching)
        elif m_state == "OUTBOUND":
            spd_color = CLR_GREEN       # Green (Leaving / Receding)
        elif m_state == "HOVERING":
            spd_color = CLR_YELLOW      # Yellow (Stationary / Hovering Threat)
        elif m_state == "CROSSING":
            spd_color = CLR_CYAN        # Cyan (Crossing Sideways / Transit)
        else:
            spd_color = CLR_WHITE

        # Dynamic ETA to Base line and color code
        if m_state == "INBOUND" and eta_val is not None:
            if eta_val <= 10.0:
                eta_line  = f"BASE ETA: {eta_val:.1f}s [CRITICAL CLOSE]"
                eta_color = CLR_CRIMSON
            elif eta_val <= 30.0:
                eta_line  = f"BASE ETA: {eta_val:.1f}s [CLOSING @ {closing_kmh:.1f} km/h]"
                eta_color = CLR_ALERT
            else:
                eta_line  = f"BASE ETA: {gps_data.get('eta_str', '')} [APPROACHING BASE]"
                eta_color = CLR_YELLOW
        elif gps_data and gps_data.get("slant_distance_m", 999) <= 3.0:
            eta_line  = "BASE ETA: ARRIVED (<3m AT CAMERA BASE)"
            eta_color = CLR_ALERT
        else:
            eta_line  = time_to_base_str
            eta_color = CLR_WHITE

        tag_w = max(340, max(len(tag_title), len(tag_sub), len(dist_line), len(speed_line), len(eta_line), len(gps_line)) * 7 + 10)
        banner_top = max(42, py1 - 82)
        cv2.rectangle(frame, (px1, banner_top), (min(w - 5, px1 + tag_w), max(82, py1)), CLR_HUD_BG, -1)
        cv2.putText(frame, tag_title,  (px1 + 4, banner_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, tag_sub,    (px1 + 4, banner_top + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color,  1, cv2.LINE_AA)
        cv2.putText(frame, dist_line,  (px1 + 4, banner_top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, speed_line, (px1 + 4, banner_top + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.34, spd_color,   1, cv2.LINE_AA)
        cv2.putText(frame, eta_line,   (px1 + 4, banner_top + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, eta_color,   1, cv2.LINE_AA)
        cv2.putText(frame, gps_line,   (px1 + 4, banner_top + 77), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_GREEN,  1, cv2.LINE_AA)

        # Broadcast telemetry to live dashboard SSE if active
        if gps_data and m and m.state in (TargetThreatState.CONFIRMED, TargetThreatState.COOLDOWN):
            try:
                from dashboard.app import push_sse_event
                push_sse_event({
                    "event": "detection",
                    "track_id": f"TGT-#{tid}",
                    "target_id": f"TGT-#{tid}",
                    "camera_id": "primary",
                    "bearing_deg": true_brg,
                    "azimuth_deg": rel_az,
                    "elevation_deg": el_val,
                    "distance_m": gps_data["slant_distance_m"],
                    "slant_distance_m": gps_data["slant_distance_m"],
                    "ground_distance_m": gps_data["ground_distance_m"],
                    "drone_altitude_m": gps_data["drone_altitude_m"],
                    "speed_kmh": gps_data.get("speed_kmh", 0.0),
                    "speed_mps": gps_data.get("speed_mps", 0.0),
                    "motion_state": gps_data.get("motion_state", "HOVERING"),
                    "motion_label": gps_data.get("motion_label", "HOVERING (STATIONARY)"),
                    "closing_speed_kmh": gps_data.get("closing_speed_kmh", 0.0),
                    "closing_speed_mps": gps_data.get("closing_speed_mps", 0.0),
                    "eta_seconds": gps_data.get("eta_seconds"),
                    "eta_str": gps_data.get("eta_str", "N/A"),
                    "time_to_base_str": gps_data.get("time_to_base_str", "TIME TO BASE: N/A"),
                    "altitude_m": gps_data["drone_altitude_m"],
                    "drone_lat": gps_data["drone_lat"],
                    "drone_lon": gps_data["drone_lon"],
                    "confidence": f"{gps_data['confidence']} ({det.confidence*100:.1f}%)",
                    "status": "LOCKED",
                })
            except Exception:
                pass

    # ── Top HUD Header ───────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 36), CLR_HUD_BG, -1)
    threat_str = decision.threat_level.value if decision else "NONE"
    status_clr = CLR_ALERT if threat_str in ["HIGH", "CRITICAL"] else (CLR_YELLOW if threat_str == "MEDIUM" else CLR_GREEN)

    header_left = f"FRAME: {frame_idx}/{total_frames} | CAM HDG: {cam_hdg:.0f} deg ({cam_card})"
    if detections and bearing_calculator:
        p_ang = bearing_calculator.calculate_for_detection(detections[0])
        header_left += f" | BRG: {p_ang['true_bearing_deg']:.1f} deg ({p_ang['cardinal_direction']}) Az: {p_ang['rel_azimuth_deg']:+.1f} deg"
        if dist_estimator:
            gd = dist_estimator.estimate_from_detection(detections[0], p_ang, track_id="hud_hdr")
            header_left += f" | {gd['dist_str']} | {gd['gps_str']}"
            if gd.get("motion_state") == "INBOUND" and gd.get("eta_seconds") is not None:
                header_left += f" | BASE ETA: {gd['eta_str']}"

    header_right = f"THREAT: {threat_str} | 8F-LOCK"
    (tw, _), _ = cv2.getTextSize(header_right, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    max_left_w = w - tw - 24

    # Ensure header_left fits cleanly without colliding with header_right
    (lw, _), _ = cv2.getTextSize(header_left, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    if lw > max_left_w:
        header_left = f"F: {frame_idx}/{total_frames} | HDG: {cam_hdg:.0f} deg"
        if detections and bearing_calculator:
            header_left += f" | BRG: {p_ang['true_bearing_deg']:.0f} deg"
            if dist_estimator and gd.get("motion_state") == "INBOUND" and gd.get("eta_seconds") is not None:
                header_left += f" | ETA: {gd['eta_str']}"

    cv2.putText(frame, header_left, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, header_right, (w - tw - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_clr, 2, cv2.LINE_AA)

    # ── Bottom Telemetry Footer ──────────────────────────────────────
    cv2.rectangle(frame, (0, h - 26), (w, h), CLR_HUD_BG, -1)
    footer_text = f"TACTICAL AIRSPACE RADAR  |  S1: TRUE BEARING (0-360 deg)  |  S2: DISTANCE + GPS LAT/LON (WGS-84 Vincenty)  |  8F LOCK"
    cv2.putText(frame, footer_text, (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_ACCENT, 1, cv2.LINE_AA)

    return frame


def main():
    """
    CLI entry point for offline video processing.
    Parses command-line arguments, locates the target video file, and executes the processing pipeline.
    """
    parser = argparse.ArgumentParser(description="Process uploaded video file with AI Drone Detection & Tracking")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "test_videos" / "input.mp4"),
        help="Path to input video file (default: test_videos/input.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output video (default: output_videos/<input_name>_detected.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/best.onnx",
        help="Path to YOLO model (default: model/best.onnx)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=90.0,
        help="Camera Compass Heading in degrees (0=North, 90=East, 180=South, 270=West). Default: 90.0 (East)",
    )
    parser.add_argument(
        "--cam-lat",
        type=float,
        default=33.6844,
        help="Camera GPS Latitude (WGS-84). Default: 33.6844 (Islamabad)",
    )
    parser.add_argument(
        "--cam-lon",
        type=float,
        default=73.0479,
        help="Camera GPS Longitude (WGS-84). Default: 73.0479 (Islamabad)",
    )
    parser.add_argument(
        "--drone-type",
        type=str,
        default="generic_medium",
        choices=list(DRONE_REFERENCE_SIZES.keys()),
        help="Reference drone size for distance estimation. Default: generic_medium (0.5m diagonal)",
    )
    parser.add_argument(
        "--drone-width",
        type=float,
        default=None,
        help="Override drone real-world diagonal width in metres (e.g. 0.59 for DJI Phantom)",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=256,
        help="SAHI slice size for high-resolution processing (default: 256)",
    )
    args = parser.parse_args()

    # Locate input video
    in_path = Path(args.input)
    found = find_input_video(in_path)

    if found is None:
        print("\n" + "=" * 65)
        print(" [!] NO INPUT VIDEO FOUND")
        print("=" * 65)
        print(" Bhai, aap apni video is folder me daal dein:")
        print(f" Folder:   {PROJECT_ROOT / 'test_videos'}")
        print(" File Name: input.mp4   (ya koi bhi .mp4 file)")
        print("\n Video rakhne ke baad yeh command run karein:")
        print("   python process_video.py")
        print("=" * 65 + "\n")
        return

    out_path = Path(args.output) if args.output else None
    process_video_file(
        input_path=found,
        output_path=out_path,
        model_path=args.model,
        conf_thresh=args.conf,
        slice_size=args.slice_size,
        camera_heading=args.heading,
        cam_lat=args.cam_lat,
        cam_lon=args.cam_lon,
        drone_type=args.drone_type,
        drone_width_m=args.drone_width,
    )


if __name__ == "__main__":
    main()
