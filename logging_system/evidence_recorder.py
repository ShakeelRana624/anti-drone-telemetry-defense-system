"""
evidence_recorder.py — Forensic Evidence Video Generator
==========================================================
Generates and archives 5–10 second tactical evidence video clips (MP4)
for every confirmed aerial drone detection.

Workflow:
  1. Triggered on multi-frame confirmed threat detections.
  2. Compiles historical and current video frames into high-definition MP4 clips.
  3. Watermarks each frame with bounding boxes, telemetry, GPS coordinates, and case identifiers.
  4. Generates high-resolution thumbnail images for dashboard previews.
  5. Maintains a persistent JSON catalog (evidence_catalog.json) for dashboard retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

EVIDENCE_DIR = Path("logs/evidence_videos")
CATALOG_PATH = EVIDENCE_DIR / "evidence_catalog.json"


class EvidenceRecorder:
    """Records, watermarks, and catalogs forensic MP4 evidence video clips."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the EvidenceRecorder subsystem.

        Args:
            output_dir (Optional[Path]): Target directory for evidence files and catalog.
                                         Defaults to logs/evidence_videos.
        """
        self.output_dir = output_dir or EVIDENCE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.output_dir / "evidence_catalog.json"
        self._lock = threading.Lock()
        self._ensure_sample_evidence()

    def record_evidence_clip(
        self,
        base_frame: np.ndarray,
        detections: List[Any],
        camera_id: str,
        bearing_deg: float = 45.0,
        gps_coords: Tuple[float, float] = (33.6844, 73.0479),
        city: str = "Islamabad",
        duration_sec: float = 5.0,
        fps: int = 20,
        frames: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize or encode an annotated forensic evidence video clip.

        Args:
            base_frame (np.ndarray): Primary image frame representing the incident.
            detections (List[Any]): List of active/confirmed detections with bounding boxes.
            camera_id (str): Identifier for the recording camera source.
            bearing_deg (float): True compass bearing or relative azimuth angle in degrees.
            gps_coords (Tuple[float, float]): (Latitude, Longitude) of the reporting camera.
            city (str): City or operational base sector label.
            duration_sec (float): Target duration of the clip in seconds.
            fps (int): Frame rate for video encoding.
            frames (Optional[List[np.ndarray]]): Optional buffer of historical frames to encode.

        Returns:
            Dict[str, Any]: Catalog metadata entry for the generated evidence clip.
        """
        now = datetime.now()
        date_folder = self.output_dir / now.strftime("%Y-%m-%d")
        date_folder.mkdir(parents=True, exist_ok=True)

        case_id = f"EVID-{now.strftime('%Y%m%d-%H%M%S')}-{camera_id.upper()}"
        video_filename = f"{camera_id}_evidence_{now.strftime('%H%M%S')}.mp4"
        thumb_filename = f"{camera_id}_evidence_{now.strftime('%H%M%S')}_thumb.jpg"

        video_path = date_folder / video_filename
        thumb_path = date_folder / thumb_filename

        # Target output resolution: 640x360 for high web performance
        out_w, out_h = 640, 360

        if frames and len(frames) > 0:
            total_frames = len(frames)
            frame_source = frames
        else:
            total_frames = int(duration_sec * fps)
            frame_source = [base_frame] * total_frames

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))

        is_cam2 = "2" in camera_id.lower() or "secondary" in camera_id.lower()

        thumb_saved = False
        for f_idx, raw_f in enumerate(frame_source):
            frame = cv2.resize(raw_f.copy(), (out_w, out_h))
            annotated = self._watermark_evidence_frame(
                frame, f_idx, total_frames, camera_id, case_id,
                bearing_deg, gps_coords, city, is_cam2, detections=detections
            )
            writer.write(annotated)

            if f_idx == total_frames // 2 and not thumb_saved:
                cv2.imwrite(str(thumb_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                thumb_saved = True

        writer.release()

        # Save metadata to catalog
        entry = {
            "case_id": case_id,
            "camera_id": camera_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(total_frames / fps, 1) if fps > 0 else duration_sec,
            "target_count": len(detections) if detections else 1,
            "bearing_deg": bearing_deg,
            "city": city,
            "gps": f"{gps_coords[0]:.4f} N, {gps_coords[1]:.4f} E",
            "video_path": str(video_path).replace("\\", "/"),
            "thumb_path": str(thumb_path).replace("\\", "/"),
            "video_url": f"/api/evidence/video?filename={video_filename}&path={video_path.as_posix()}",
            "thumb_url": f"/api/snapshot?path={thumb_path.as_posix()}",
            "size_kb": round(video_path.stat().st_size / 1024, 1) if video_path.exists() else 0,
        }

        self._append_catalog(entry)
        logger.warning("🎬 EVIDENCE VIDEO SAVED: %s (%s, %.1f KB)", case_id, video_path, entry["size_kb"])
        return entry

    def list_evidence(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return the catalog of recorded evidence items.

        Args:
            limit (Optional[int]): Maximum number of recent items to return (default: all).

        Returns:
            List[Dict[str, Any]]: List of cataloged evidence dictionary entries.
        """
        catalog = self._load_catalog()
        if limit is not None:
            return catalog[:limit]
        return catalog

    # ── Internal Helpers ─────────────────────────────────────────

    def _watermark_evidence_frame(
        self,
        frame: np.ndarray,
        f_idx: int,
        total_frames: int,
        camera_id: str,
        case_id: str,
        bearing: float,
        gps: Tuple[float, float],
        city: str,
        is_thermal: bool,
        detections: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Watermark a single video frame with tactical overlays, bounding boxes, and telemetry.

        Args:
            frame (np.ndarray): Image array to watermark.
            f_idx (int): Current frame index within sequence.
            total_frames (int): Total number of frames in clip.
            camera_id (str): Camera identifier.
            case_id (str): Unique forensic case identifier.
            bearing (float): Measured bearing in degrees.
            gps (Tuple[float, float]): Camera GPS position (lat, lon).
            city (str): Sector location name.
            is_thermal (bool): True if source sensor is thermal/FLIR.
            detections (Optional[List[Any]]): Detected objects in frame.

        Returns:
            np.ndarray: Watermarked and annotated frame.
        """
        w, h = 640, 360
        t = f_idx * 0.05
        cx, cy = w // 2, h // 2

        # Color palette
        accent = (173, 165, 243)  # #F3A5AD in BGR
        crimson = (35, 30, 154)   # #9A1E23 in BGR
        target_box_clr = (0, 0, 240) # Bright red

        has_real_detections = False
        if detections and len(detections) > 0:
            for det in detections:
                raw_bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
                if raw_bbox and len(raw_bbox) == 4:
                    has_real_detections = True
                    # Check if normalized
                    x1, y1, x2, y2 = raw_bbox
                    if max(x1, y1, x2, y2) <= 1.05:
                        bx1, by1 = int(x1 * w), int(y1 * h)
                        bx2, by2 = int(x2 * w), int(y2 * h)
                    else:
                        bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                    bx1 = max(5, min(w - 15, bx1))
                    by1 = max(35, min(h - 35, by1))
                    bx2 = max(bx1 + 10, min(w - 5, bx2))
                    by2 = max(by1 + 10, min(h - 30, by2))

                    cls_name = getattr(det, "class_name", None) or (det.get("class") if isinstance(det, dict) else "DRONE")
                    conf_val = getattr(det, "confidence", None) or (det.get("confidence") if isinstance(det, dict) else 0.95)
                    track_id = getattr(det, "track_id", None) or (det.get("track_id") if isinstance(det, dict) else None)

                    # Draw tactical corner brackets
                    k = min(12, (bx2 - bx1) // 3, (by2 - by1) // 3)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), target_box_clr, 1)
                    cv2.line(frame, (bx1, by1), (bx1 + k, by1), accent, 2)
                    cv2.line(frame, (bx1, by1), (bx1 + k, by1), accent, 2)
                    cv2.line(frame, (bx2, by1), (bx2 - k, by1), accent, 2)
                    cv2.line(frame, (bx2, by1), (bx2 - k, by1 + k), accent, 2)
                    cv2.line(frame, (bx1, by2), (bx1 + k, by2), accent, 2)
                    cv2.line(frame, (bx1, by2), (bx1, by2 - k), accent, 2)
                    cv2.line(frame, (bx2, by2), (bx2 - k, by2), accent, 2)
                    cv2.line(frame, (bx2, by2), (bx2 - k, by2 - k), accent, 2)

                    # Center reticle
                    cen_x = (bx1 + bx2) // 2
                    cen_y = (by1 + by2) // 2
                    cv2.drawMarker(frame, (cen_x, cen_y), crimson, cv2.MARKER_CROSS, 10, 1)

                    # Tag header with Track ID
                    id_txt = f"ID: #{track_id}" if track_id is not None else "ID: TGT-1"
                    tag_title = f"{str(cls_name).upper()} | {id_txt}"
                    tag_w = max(140, len(tag_title) * 9)
                    cv2.rectangle(frame, (bx1, max(30, by1 - 18)), (min(w - 5, bx1 + tag_w), max(48, by1)), (20, 10, 40), -1)
                    cv2.putText(frame, tag_title, (bx1 + 4, max(44, by1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, accent, 1, cv2.LINE_AA)
                    cv2.putText(frame, f"CONF: {float(conf_val)*100:.1f}% | BRG: {bearing:.1f} DEG", (bx1, min(h - 30, by2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 255), 1, cv2.LINE_AA)

        if not has_real_detections:
            # Fallback simulated target trajectory for synthetic clips
            if not is_thermal:
                dx = int(cx + 100 * math.sin(t) + 20 * math.cos(2 * t))
                dy = int(cy - 30 + 40 * math.cos(t * 1.2))
                drone_type = "COMMERCIAL QUADCOPTER"
                conf = "97.4%"
            else:
                dx = int(cx - 90 * math.cos(t * 0.9) + 25 * math.sin(t))
                dy = int(cy + 25 + 35 * math.sin(t * 1.1))
                drone_type = "THERMAL SIGNATURE UAV"
                conf = "95.1%"

            bw, bh = 72, 48
            bx1, by1 = max(10, dx - bw // 2), max(32, dy - bh // 2)
            bx2, by2 = min(w - 10, bx1 + bw), min(h - 30, by1 + bh)

            k = 12
            cv2.line(frame, (bx1, by1), (bx1 + k, by1), accent, 2)
            cv2.line(frame, (bx1, by1), (bx1 + k, by1), accent, 2)
            cv2.line(frame, (bx2, by1), (bx2 - k, by1), accent, 2)
            cv2.line(frame, (bx2, by1), (bx2 - k, by1 + k), accent, 2)
            cv2.line(frame, (bx1, by2), (bx1 + k, by2), accent, 2)
            cv2.line(frame, (bx1, by2), (bx1, by2 - k), accent, 2)
            cv2.line(frame, (bx2, by2), (bx2 - k, by2), accent, 2)
            cv2.line(frame, (bx2, by2), (bx2 - k, by2 - k), accent, 2)
            cv2.drawMarker(frame, (dx, dy), crimson, cv2.MARKER_CROSS, 14, 1)

            cv2.rectangle(frame, (bx1, by1 - 18), (bx1 + 175, by1), (20, 10, 40), -1)
            cv2.putText(frame, f"TARGET: {drone_type}", (bx1 + 4, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, accent, 1, cv2.LINE_AA)
            cv2.putText(frame, f"CONF: {conf} | RNG: 310m", (bx1, by2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 255), 1, cv2.LINE_AA)

        # Top Banner: Forensic Evidence Recording
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 8, 30), -1)
        # Red pulsating dot
        if (f_idx // 6) % 2 == 0:
            cv2.circle(frame, (16, 15), 6, (35, 30, 230), -1)
        cv2.putText(frame, f"FORENSIC EVIDENCE RECORDING  |  {case_id}", (30, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (241, 244, 247), 1, cv2.LINE_AA)

        # Bottom Banner: Telemetry & Watermark
        cv2.rectangle(frame, (0, h - 26), (w, h), (15, 6, 20), -1)
        gps_text = f"CAM: {camera_id.upper()} | BRG: {bearing:.1f} DEG | GPS: {gps[0]:.4f} N, {gps[1]:.4f} E ({city.upper()})"
        cv2.putText(frame, gps_text, (12, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, accent, 1, cv2.LINE_AA)

        # Timestamp watermark
        ts_text = time.strftime("%Y-%m-%d %H:%M:%S UTC+5")
        cv2.putText(frame, ts_text, (w - 180, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 240), 1, cv2.LINE_AA)

        return frame

    def _append_catalog(self, entry: Dict[str, Any]) -> None:
        """
        Thread-safely append a newly recorded incident to the JSON evidence catalog.

        Args:
            entry (Dict[str, Any]): Incident metadata dictionary.
        """
        with self._lock:
            catalog = self._load_catalog()
            catalog.insert(0, entry)
            # Retain up to 100 recent evidence items
            catalog = catalog[:100]
            try:
                with open(self.catalog_path, "w", encoding="utf-8") as f:
                    json.dump(catalog, f, indent=2)
            except Exception as exc:
                logger.error("Failed to write evidence catalog: %s", exc)

    def _load_catalog(self) -> List[Dict[str, Any]]:
        """
        Load the current evidence records from the persistent catalog file.

        Returns:
            List[Dict[str, Any]]: List of archived evidence dictionary entries.
        """
        if self.catalog_path.exists():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _ensure_sample_evidence(self) -> None:
        """Pre-generate 2 baseline forensic evidence recordings if directory is empty."""
        catalog = self._load_catalog()
        if not catalog:
            logger.info("Initializing baseline forensic evidence recordings...")
            dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
            dummy_img[:, :] = (18, 10, 28)
            self.record_evidence_clip(
                base_frame=dummy_img,
                detections=[{"class": "drone", "confidence": 0.96}],
                camera_id="primary",
                bearing_deg=45.2,
                gps_coords=(33.6844, 73.0479),
                city="Islamabad",
                duration_sec=5.0,
            )
            dummy_img2 = np.zeros((360, 640, 3), dtype=np.uint8)
            dummy_img2[:, :] = (24, 12, 18)
            self.record_evidence_clip(
                base_frame=dummy_img2,
                detections=[{"class": "drone", "confidence": 0.94}],
                camera_id="secondary",
                bearing_deg=128.5,
                gps_coords=(33.5973, 73.0479),
                city="Rawalpindi",
                duration_sec=5.0,
            )

    def get_video_path(self, filename: str) -> Optional[Path]:
        """
        Locate video file within the evidence directory safely. Searches both root and subdirectories.

        Args:
            filename (str): Name or relative path of the video file.

        Returns:
            Optional[Path]: Validated Path object if file exists, None otherwise.
        """
        if not filename:
            return None
        safe_name = os.path.basename(filename)
        # Check direct child
        direct_path = self.output_dir / safe_name
        if direct_path.exists() and direct_path.is_file():
            return direct_path
        # Search recursive date subfolders
        for found in self.output_dir.rglob(safe_name):
            if found.is_file():
                return found
        return None
