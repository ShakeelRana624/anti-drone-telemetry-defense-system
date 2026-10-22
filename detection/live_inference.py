"""
live_inference.py — Real-Time Live Stream AI Drone Detection & Telemetry Engine
==============================================================================
Processes live camera / video stream frames through the complete AI stack:
  • YOLO11 inference (model/best.onnx)
  • DeepSORT tracking with persistent Track IDs
  • Multi-frame threat verification (Acquiring -> Confirming -> Locked)
  • True compass bearing (0-360 deg) and relative azimuth/elevation
  • Step 2: Slant & ground distance estimation, altitude, WGS-84 Vincenty GPS coordinates
  • Flight speed (km/h & m/s), motion vector classification (INBOUND/OUTBOUND/HOVERING/CROSSING)
  • Dynamic Base ETA calculation & urgency indicators
  • Real-time SSE dispatch to the C2 Dashboard Radar and Telemetry Feed
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from detection.detector import Detection
from detection.tracker import DeepSORTTracker
from detection.decision_engine import DecisionEngine, DecisionResult, ThreatLevel, TargetThreatState
from utils.bearing_calculator import BearingCalculator
from utils.distance_gps_estimator import DistanceGPSEstimator

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tactical HUD Palette
CLR_ACCENT     = (173, 165, 243)   # Soft rose lavender
CLR_ALERT      = (30, 30, 230)     # Tactical alert red
CLR_CRIMSON    = (35, 30, 154)     # Deep crimson
CLR_YELLOW     = (0, 215, 255)     # Yellow (Verifying)
CLR_GREEN      = (50, 205, 50)     # Green (Clear / Receding)
CLR_CYAN       = (255, 220, 0)     # Cyan (Acquiring)
CLR_WHITE      = (245, 245, 250)
CLR_HUD_BG     = (18, 12, 28)


class LiveDroneInferenceEngine:
    """
    Central real-time detection & tracking engine for live camera feeds and dashboard MJPEG streams.
    """

    def __init__(
        self,
        model_path: str = "model/best.onnx",
        conf_thresh: float = 0.35,
        camera_heading: float = 45.0,
        cam_lat: float = 33.6844,
        cam_lon: float = 73.0479,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.camera_heading = camera_heading
        self.cam_lat = cam_lat
        self.cam_lon = cam_lon
        self.frame_idx = 0
        self._lock = threading.Lock()

        # Load YOLO model
        m_file = PROJECT_ROOT / model_path
        if not m_file.exists():
            m_file = PROJECT_ROOT / "yolo11n.pt"

        logger.info("Initializing LiveDroneInferenceEngine with weights: %s", m_file)
        self.yolo_model = YOLO(str(m_file))

        # DeepSORT Tracker
        tracker_cfg = {
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
            }
        }
        self.tracker = DeepSORTTracker(tracker_cfg)

        # Bearing & Distance/GPS estimators
        self.bearing_calculator = BearingCalculator({
            "camera_fov_horizontal": 85.0,
            "camera_fov_vertical":   60.0,
            "camera_heading_deg":    self.camera_heading,
        })

        self.dist_estimator = DistanceGPSEstimator({
            "camera_fov_horizontal": 85.0,
            "camera_fov_vertical":   60.0,
            "frame_width":           1280,
            "frame_height":          720,
            "drone_type":            "generic_medium",
            "camera_latitude":       self.cam_lat,
            "camera_longitude":      self.cam_lon,
        })

        # Decision Engine
        self.decision_engine = DecisionEngine(
            min_consecutive_hits=4,
            window_size=8,
            min_hits_in_window=4,
            min_confidence=conf_thresh,
            target_cooldown_sec=10.0,
        )

        self.last_active_targets: List[Dict[str, Any]] = []
        self.last_process_time = time.time()
        self.infer_fps = 0.0

        self._telemetry_callback = None
        self._last_push_time: Dict[str, float] = {}
        self._locked_announced: Dict[str, bool] = {}

        # Event-triggered Forensic Evidence Video Recording (ONLY on drone detection)
        from collections import deque
        from logging_system.evidence_recorder import EvidenceRecorder
        self.evidence_recorder = EvidenceRecorder()
        self._frame_buffer = deque(maxlen=75)
        self._last_evidence_time = 0.0
        self._evidence_cooldown_sec = 20.0

    def set_telemetry_callback(self, callback) -> None:
        """Register a callback function to receive real-time target telemetry payloads."""
        self._telemetry_callback = callback

    def update_camera_params(
        self,
        heading: Optional[float] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        fov_h: Optional[float] = None,
    ) -> None:
        """
        Dynamically update camera compass heading, geo-coordinates, and FOV across all tracking submodules.
        """
        with self._lock:
            if heading is not None:
                try:
                    self.camera_heading = float(heading) % 360.0
                    self.bearing_calculator.camera_heading = self.camera_heading
                    logger.info("LiveInference updated camera heading: %.1f deg", self.camera_heading)
                except (ValueError, TypeError):
                    pass
            if lat is not None or lon is not None:
                try:
                    if lat is not None:
                        self.cam_lat = float(lat)
                    if lon is not None:
                        self.cam_lon = float(lon)
                    self.dist_estimator.set_camera_gps(self.cam_lat, self.cam_lon)
                    logger.info("LiveInference updated camera GPS: (%.6f, %.6f)", self.cam_lat, self.cam_lon)
                except (ValueError, TypeError):
                    pass
            if fov_h is not None:
                try:
                    self.bearing_calculator._fov_h = float(fov_h)
                    self.dist_estimator._fov_h = float(fov_h)
                    self.dist_estimator._focal_px = self.dist_estimator._compute_focal_px()
                except (ValueError, TypeError):
                    pass

    def process_frame(self, frame: np.ndarray, camera_id: str = "primary") -> np.ndarray:
        """
        Run inference on frame, update tracker and distance/GPS, render tactical HUD annotations,
        and broadcast real-time telemetry events via SSE.
        """
        if frame is None:
            return frame

        with self._lock:
            self.frame_idx += 1
            self._frame_buffer.append(frame.copy())
            now = time.time()
            t0 = time.perf_counter()

            h, w = frame.shape[:2]
            out_frame = frame.copy()

            # Center boresight reference marker
            cx = w // 2
            cy = h // 2
            cam_card = BearingCalculator.degrees_to_cardinal(self.camera_heading)
            cv2.drawMarker(out_frame, (cx, cy), (100, 100, 130), cv2.MARKER_CROSS, 16, 1)
            cv2.circle(out_frame, (cx, cy), 6, (100, 100, 130), 1)

            # 1. Run YOLO inference on frame
            try:
                results = self.yolo_model.predict(
                    frame,
                    conf=self.conf_thresh,
                    verbose=False,
                    imgsz=640,
                )
            except Exception as err:
                logger.error("YOLO live inference error: %s", err)
                results = []

            raw_detections: List[Detection] = []
            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = self.yolo_model.names.get(cls_id, "drone")

                        # Normalize bbox to 0..1
                        norm_box = [
                            float(xyxy[0] / w),
                            float(xyxy[1] / h),
                            float(xyxy[2] / w),
                            float(xyxy[3] / h),
                        ]
                        raw_detections.append(Detection(
                            bbox=norm_box,
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        ))

            # 2. DeepSORT Tracking
            confirmed_tracks = self.tracker.update(
                raw_detections,
                camera_id=camera_id,
                frame_shape=(h, w),
                frame=frame,
            )
            active_targets = confirmed_tracks if confirmed_tracks else raw_detections

            # 3. Decision Engine
            decision = self.decision_engine.evaluate(
                active_targets,
                camera_id=camera_id,
            )
            target_states = decision.target_states if decision else {}

            # 4. Render Tactical Annotations ONLY IF TARGETS ARE FOUND
            for det in active_targets:
                x1, y1, x2, y2 = det.bbox
                px1 = int(x1 * w); py1 = int(y1 * h)
                px2 = int(x2 * w); py2 = int(y2 * h)

                tid = getattr(det, "track_id", None)
                m = target_states.get(tid) if tid is not None else None

                # Calculate Bearing & Distance/GPS
                ang = self.bearing_calculator.calculate_for_detection(det)
                true_brg = ang["true_bearing_deg"] if ang else self.camera_heading
                cardinal = ang["cardinal_direction"] if ang else cam_card
                rel_az = ang["rel_azimuth_deg"] if ang else 0.0
                el_val = ang["elevation_deg"] if ang else 0.0
                los_ang = ang["total_angle_deg"] if ang else 0.0

                frame_time = self.frame_idx / 25.0
                gps_data = self.dist_estimator.estimate_from_detection(
                    det, ang, track_id=str(tid) if tid is not None else "TGT-01", timestamp=frame_time
                )

                # State color code
                if m and m.state in (TargetThreatState.CONFIRMED, TargetThreatState.COOLDOWN):
                    box_color = CLR_ALERT       # Red for confirmed lock
                    state_text = "TARGET LOCKED"
                elif m and m.state == TargetThreatState.SUSPICIOUS:
                    box_color = CLR_YELLOW      # Yellow for verifying
                    state_text = f"CONFIRMING LOCK ({m.consecutive_hits} frames)"
                else:
                    box_color = CLR_CYAN        # Cyan for acquiring
                    hits = m.consecutive_hits if m else 1
                    state_text = f"ACQUIRING ({hits} frames)"

                # Bounding box
                cv2.rectangle(out_frame, (px1, py1), (px2, py2), box_color, 2)

                # Corner brackets
                k = min(16, max(6, (px2 - px1) // 3), max(6, (py2 - py1) // 3))
                cv2.line(out_frame, (px1, py1), (px1 + k, py1), CLR_ACCENT, 3)
                cv2.line(out_frame, (px1, py1), (px1, py1 + k), CLR_ACCENT, 3)
                cv2.line(out_frame, (px2, py1), (px2 - k, py1), CLR_ACCENT, 3)
                cv2.line(out_frame, (px2, py1), (px2, py1 + k), CLR_ACCENT, 3)
                cv2.line(out_frame, (px1, py2), (px1 + k, py2), CLR_ACCENT, 3)
                cv2.line(out_frame, (px1, py2), (px1, py2 - k), CLR_ACCENT, 3)
                cv2.line(out_frame, (px2, py2), (px2 - k, py2), CLR_ACCENT, 3)
                cv2.line(out_frame, (px2, py2), (px2 - k, py2 - k), CLR_ACCENT, 3)

                # Center marker
                cen_x = (px1 + px2) // 2
                cen_y = (py1 + py2) // 2
                cv2.drawMarker(out_frame, (cen_x, cen_y), CLR_CRIMSON, cv2.MARKER_CROSS, 14, 2)

                # Bearing ray line from boresight center to target center
                cv2.line(out_frame, (cx, cy), (cen_x, cen_y), box_color, 1, cv2.LINE_AA)
                mid_x = (cx + cen_x) // 2
                mid_y = (cy + cen_y) // 2
                cv2.circle(out_frame, (mid_x, mid_y), 3, box_color, -1)
                cv2.putText(out_frame, f"BRG {true_brg:.1f}° {cardinal}", (mid_x + 6, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_WHITE, 1, cv2.LINE_AA)

                # Telemetry banner
                id_str = f"#{tid}" if tid is not None else "LIVE"
                tag_title = f"{det.class_name.upper()} | ID: {id_str} | {det.confidence*100:.1f}% | BRG: {true_brg:.1f}° ({cardinal})"
                tag_sub   = f"{state_text} [Az: {rel_az:+.1f}° | El: {el_val:+.1f}°]"

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
                    gps_line   = "GPS: ESTIMATING"
                    m_state    = "UNKNOWN"
                    eta_val    = None
                    time_to_base_str = "TIME TO BASE: N/A"
                    closing_kmh = 0.0

                # Motion state color
                if m_state == "INBOUND":
                    spd_color = CLR_ALERT
                elif m_state == "OUTBOUND":
                    spd_color = CLR_GREEN
                elif m_state == "CROSSING":
                    spd_color = CLR_CYAN
                else:
                    spd_color = CLR_YELLOW

                # ETA line
                if m_state == "INBOUND" and eta_val is not None:
                    if eta_val <= 10.0:
                        eta_line  = f"BASE ETA: {eta_val:.1f}s [CRITICAL CLOSE]"
                        eta_color = CLR_CRIMSON
                    elif eta_val <= 30.0:
                        eta_line  = f"BASE ETA: {eta_val:.1f}s [CLOSING @ {closing_kmh:.1f} km/h]"
                        eta_color = CLR_ALERT
                    else:
                        eta_line  = f"BASE ETA: {gps_data.get('eta_str', '')} [APPROACHING]"
                        eta_color = CLR_YELLOW
                else:
                    eta_line  = time_to_base_str
                    eta_color = CLR_WHITE

                tag_w = max(330, max(len(tag_title), len(tag_sub), len(dist_line), len(speed_line), len(eta_line), len(gps_line)) * 7 + 10)
                banner_top = max(38, py1 - 82)
                cv2.rectangle(out_frame, (px1, banner_top), (min(w - 5, px1 + tag_w), max(82, py1)), CLR_HUD_BG, -1)
                cv2.putText(out_frame, tag_title,  (px1 + 4, banner_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_ACCENT, 1, cv2.LINE_AA)
                cv2.putText(out_frame, tag_sub,    (px1 + 4, banner_top + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color,  1, cv2.LINE_AA)
                cv2.putText(out_frame, dist_line,  (px1 + 4, banner_top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_YELLOW, 1, cv2.LINE_AA)
                cv2.putText(out_frame, speed_line, (px1 + 4, banner_top + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.34, spd_color,   1, cv2.LINE_AA)
                cv2.putText(out_frame, eta_line,   (px1 + 4, banner_top + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, eta_color,   1, cv2.LINE_AA)
                cv2.putText(out_frame, gps_line,   (px1 + 4, banner_top + 77), cv2.FONT_HERSHEY_SIMPLEX, 0.34, CLR_GREEN,  1, cv2.LINE_AA)

                # Push live telemetry to C2 Dashboard (Radar Scope & Telemetry Feed Table)
                if gps_data:
                    tgt_id_str = f"TGT-#{tid}" if tid is not None else "TGT-#01"
                    now_t = time.time()
                    last_push = self._last_push_time.get(tgt_id_str, 0.0)
                    is_locked = (m and m.state in (TargetThreatState.CONFIRMED, TargetThreatState.COOLDOWN))
                    first_lock = is_locked and not self._locked_announced.get(tgt_id_str, False)
                    if first_lock:
                        self._locked_announced[tgt_id_str] = True

                        # Trigger forensic evidence video clip ONLY on drone detection lock
                        now_ev = time.time()
                        if now_ev - self._last_evidence_time >= self._evidence_cooldown_sec:
                            self._last_evidence_time = now_ev
                            buffered_snap = list(self._frame_buffer)
                            det_snap = list(active_targets)
                            brg_snap = float(true_brg)
                            c_id_snap = str(camera_id)
                            gps_snap = (self.cam_lat, self.cam_lon)
                            out_frame_snap = out_frame.copy()

                            def _record_task():
                                try:
                                    logger.warning("🎬 Drone target %s confirmed: Recording forensic MP4 evidence clip...", tgt_id_str)
                                    self.evidence_recorder.record_evidence_clip(
                                        base_frame=out_frame_snap,
                                        detections=det_snap,
                                        camera_id=c_id_snap,
                                        bearing_deg=brg_snap,
                                        gps_coords=gps_snap,
                                        city="Sector Alpha",
                                        duration_sec=5.0,
                                        fps=20,
                                        frames=buffered_snap if len(buffered_snap) >= 15 else None
                                    )
                                except Exception as ev_err:
                                    logger.error("Failed to record evidence clip: %s", ev_err)

                            threading.Thread(target=_record_task, daemon=True, name=f"evid-{tgt_id_str}").start()

                    # Throttle regular telemetry to ~7 Hz (every 0.14s) to keep SSE smooth and responsive
                    if first_lock or (now_t - last_push) >= 0.14:
                        self._last_push_time[tgt_id_str] = now_t
                        status_str = "LOCKED [8/8]" if is_locked else ("CONFIRMING" if (m and m.state == TargetThreatState.SUSPICIOUS) else "ACQUIRING")
                        payload = {
                            "event": "detection" if first_lock else "telemetry",
                            "trigger_alert": bool(first_lock),
                            "track_id": tgt_id_str,
                            "target_id": tgt_id_str,
                            "camera_id": "CAM 01 (Optical HD)",
                            "drone_type": "Commercial Drone (Quadcopter)",
                            "bearing_deg": round(float(true_brg), 1),
                            "azimuth_deg": round(float(rel_az), 1),
                            "elevation_deg": round(float(el_val), 1),
                            "distance_m": round(float(gps_data["slant_distance_m"]), 1),
                            "slant_distance_m": round(float(gps_data["slant_distance_m"]), 1),
                            "ground_distance_m": round(float(gps_data["ground_distance_m"]), 1),
                            "drone_altitude_m": round(float(gps_data["drone_altitude_m"]), 1),
                            "altitude_m": round(float(gps_data["drone_altitude_m"]), 1),
                            "drone_lat": round(float(gps_data.get("drone_lat", gps_data.get("drone_latitude", 0.0))), 6),
                            "drone_lon": round(float(gps_data.get("drone_lon", gps_data.get("drone_longitude", 0.0))), 6),
                            "speed_kmh": round(float(gps_data.get("speed_kmh", 0.0)), 1),
                            "speed_mps": round(float(gps_data.get("speed_mps", 0.0)), 1),
                            "motion_state": gps_data.get("motion_state", "HOVERING"),
                            "motion_label": gps_data.get("motion_label", "HOVERING"),
                            "eta_seconds": gps_data.get("eta_seconds"),
                            "eta_str": gps_data.get("eta_str", "N/A"),
                            "time_to_base_str": time_to_base_str,
                            "closing_speed_kmh": round(float(closing_kmh), 1),
                            "confidence": f"{det.confidence * 100:.1f}%",
                            "status": status_str,
                            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        }

                        if self._telemetry_callback is not None:
                            try:
                                self._telemetry_callback(payload)
                            except Exception as cb_err:
                                logger.debug("Telemetry callback error: %s", cb_err)
                        else:
                            try:
                                from dashboard.app import push_sse_event
                                push_sse_event(payload)
                            except Exception as sse_err:
                                logger.debug("Fallback push_sse_event error: %s", sse_err)

            # Bottom optical watermark (always visible, subtle)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            fps_text = f"AI INFERENCE: {elapsed_ms:.1f}ms ({1000/max(1, elapsed_ms):.1f} FPS) | TARGETS: {len(active_targets)}"
            cv2.putText(out_frame, fps_text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 190), 1, cv2.LINE_AA)

            return out_frame


_live_engine: Optional[LiveDroneInferenceEngine] = None
_engine_lock = threading.Lock()


def get_live_inference_engine() -> LiveDroneInferenceEngine:
    """Singleton provider for LiveDroneInferenceEngine."""
    global _live_engine
    with _engine_lock:
        if _live_engine is None:
            _live_engine = LiveDroneInferenceEngine()
        return _live_engine
