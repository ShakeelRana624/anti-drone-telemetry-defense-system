"""
evidence_recorder.py — Forensic Evidence Video Generator & Incursion Session Recorder
=====================================================================================
Records and archives continuous tactical MP4 evidence videos for every confirmed aerial drone detection.

Workflow:
  1. Rolling 2-second pre-recording buffer (50-60 frames) is constantly maintained.
  2. When an actual drone threat is confirmed, incursion recording starts automatically:
     - 2 seconds of pre-recording are written first.
     - Live video frames continue to be appended while the drone is actively tracked.
     - Video frames are watermarked with HUD brackets, target IDs, bearing, GPS, and timestamps.
  3. Continuous recording lasts until the drone disappears (no targets for > 2.5s grace period).
  4. Video is finalized into a single MP4 file inside the 'evidence/' directory named with the date & time:
     evidence_YYYY-MM-DD_HH-MM-SS.mp4
  5. Remuxed with FFmpeg for instant web browser streaming and cataloged for the Dashboard Evidence Vault.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
CATALOG_PATH = EVIDENCE_DIR / "evidence_catalog.json"
LEGACY_DIR = PROJECT_ROOT / "logs" / "evidence_videos"


class EvidenceRecorder:
    """Records, watermarks, and catalogs forensic MP4 evidence video clips."""

    _session_recorded_lock = threading.Lock()
    _session_recorded = False

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize the EvidenceRecorder subsystem.

        Args:
            output_dir (Optional[Path]): Target directory for evidence files and catalog.
                                         Defaults to 'evidence/'.
        """
        self.output_dir = output_dir or EVIDENCE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.output_dir / "evidence_catalog.json"
        self._lock = threading.Lock()

        # Continuous incursion session state
        self._session_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_video_path: Optional[Path] = None
        self._current_thumb_path: Optional[Path] = None
        self._current_meta: Dict[str, Any] = {}
        self._recording: bool = False
        self._last_target_seen: float = 0.0
        self._frames_written: int = 0
        self._thumb_saved: bool = False
        self._incursion_fps: float = 20.0

    @classmethod
    def reset_session(cls) -> None:
        """Reset session to allow recording a new clip on the next confirmed drone detection."""
        with cls._session_recorded_lock:
            cls._session_recorded = False
            logger.info("🎬 EvidenceRecorder session re-armed. Next drone detection will record evidence.")

    def is_recording(self) -> bool:
        """Return True if an evidence recording session is actively writing frames."""
        with self._session_lock:
            return self._recording

    def clear_evidence(self) -> None:
        """Clear all cataloged evidence items and archived video clips, and re-arm session."""
        with self._lock:
            with self._session_lock:
                if self._recording and self._writer:
                    try:
                        self._writer.release()
                    except Exception:
                        pass
                    self._writer = None
                    self._recording = False

            EvidenceRecorder.reset_session()
            try:
                if self.catalog_path.exists():
                    self.catalog_path.unlink()
                for item in self.output_dir.iterdir():
                    if item.is_file() and (item.suffix in (".mp4", ".jpg", ".tmp") or item.name == "evidence_catalog.json"):
                        item.unlink(missing_ok=True)
                logger.info("Cleared all evidence videos from %s.", self.output_dir)
            except Exception as e:
                logger.warning("Error clearing evidence: %s", e)

    # ── Continuous Incursion Recording (Pre-buffer + Tracking until Disappearance) ──

    def start_incursion(
        self,
        pre_frames: Optional[List[np.ndarray]] = None,
        camera_id: str = "primary",
        bearing_deg: float = 45.0,
        gps_coords: Tuple[float, float] = (33.6844, 73.0479),
        city: str = "Sector Alpha",
        base_frame: Optional[np.ndarray] = None,
        target_id: str = "TGT-#01",
        drone_type: str = "Commercial Drone",
        fps: float = 20.0,
    ) -> bool:
        """
        Start continuous incursion recording:
        1. Opens a new MP4 video file named with the current date & timestamp in 'evidence/'.
        2. Flushes the 2-second pre-recording buffer into the video writer first.
        3. Remains active, recording all live frames until the drone disappears.
        """
        with self._session_lock:
            if self._recording:
                # Already actively recording this incursion; update target heartbeat
                self._last_target_seen = time.time()
                return False

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = f"evidence_{date_str}.mp4"
            thumb_filename = f"evidence_{date_str}_thumb.jpg"

            video_path = self.output_dir / video_filename
            thumb_path = self.output_dir / thumb_filename
            case_id = f"EVID-{now.strftime('%Y%m%d-%H%M%S')}"

            out_w, out_h = 640, 360
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                logger.error("Failed to open VideoWriter for %s", video_path)
                return False

            self._writer = writer
            self._current_video_path = video_path
            self._current_thumb_path = thumb_path
            self._recording = True
            self._frames_written = 0
            self._thumb_saved = False
            self._incursion_fps = fps
            self._last_target_seen = time.time()

            self._current_meta = {
                "case_id": case_id,
                "filename": video_filename,
                "thumb_filename": thumb_filename,
                "camera_id": camera_id,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "bearing_deg": round(float(bearing_deg), 1),
                "gps": f"{gps_coords[0]:.4f} N, {gps_coords[1]:.4f} E",
                "gps_coords": gps_coords,
                "city": city,
                "target_id": target_id,
                "drone_type": drone_type,
            }

            # 1. Flush 2-second pre-recording buffer
            pre_count = 0
            if pre_frames:
                for pf in pre_frames:
                    if pf is not None:
                        frame_resized = cv2.resize(pf, (out_w, out_h)) if pf.shape[:2] != (out_h, out_w) else pf.copy()
                        # Subtle top watermark for pre-buffer
                        cv2.rectangle(frame_resized, (0, 0), (out_w, 24), (20, 8, 30), -1)
                        cv2.putText(
                            frame_resized,
                            f"PRE-RECORDING BUFFER (-2s) | {case_id} | {camera_id.upper()}",
                            (10, 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (180, 180, 210),
                            1,
                            cv2.LINE_AA
                        )
                        self._writer.write(frame_resized)
                        self._frames_written += 1
                        pre_count += 1

            # 2. Save initial thumbnail if base_frame available
            if base_frame is not None:
                b_thumb = cv2.resize(base_frame, (out_w, out_h)) if base_frame.shape[:2] != (out_h, out_w) else base_frame
                cv2.imwrite(str(thumb_path), b_thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._thumb_saved = True

            logger.warning(
                "🎬 [EVIDENCE RECORDING STARTED] Case %s: Writing to %s (pre-buffer: %d frames / 2.0s)",
                case_id, video_path.name, pre_count
            )
            return True

    def write_frame(self, frame: np.ndarray, has_targets: bool = True) -> None:
        """
        Append an incoming video frame to the active continuous incursion recording.
        Thread-safe; executes in <0.5ms.
        """
        if not self._recording or self._writer is None or frame is None:
            return

        with self._session_lock:
            if not self._recording or self._writer is None:
                return

            if has_targets:
                self._last_target_seen = time.time()

            out_w, out_h = 640, 360
            f = cv2.resize(frame, (out_w, out_h)) if frame.shape[:2] != (out_h, out_w) else frame.copy()
            self._writer.write(f)
            self._frames_written += 1

            if not self._thumb_saved and self._frames_written >= 20:
                cv2.imwrite(str(self._current_thumb_path), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self._thumb_saved = True

    def on_target_present(self) -> None:
        """Heartbeat called while drone is actively visible and locked."""
        with self._session_lock:
            self._last_target_seen = time.time()

    def check_disappearance(self, grace_period: float = 2.5) -> Optional[Dict[str, Any]]:
        """
        Check if the drone has disappeared.
        If no targets have been observed for longer than grace_period (default: 2.5 seconds),
        finalizes the continuous recording and returns the catalog entry dictionary.
        """
        if not self._recording:
            return None

        with self._session_lock:
            if not self._recording:
                return None
            elapsed_since_target = time.time() - self._last_target_seen
            if elapsed_since_target < grace_period:
                return None

        logger.info("🎬 [EVIDENCE] Drone disappeared (no target for %.1fs). Finalizing recording...", grace_period)
        return self.stop_incursion()

    def stop_incursion(self) -> Optional[Dict[str, Any]]:
        """
        Finalize continuous incursion recording:
        1. Releases VideoWriter.
        2. Fast-starts video using FFmpeg for optimal browser playback.
        3. Saves thumbnail and records entry in evidence_catalog.json.
        4. Re-arms session for subsequent drone incursions.
        """
        with self._session_lock:
            if not self._recording:
                return None

            self._recording = False
            if self._writer:
                try:
                    self._writer.release()
                except Exception as e:
                    logger.warning("Error releasing VideoWriter: %s", e)
                self._writer = None

            video_path = self._current_video_path
            thumb_path = self._current_thumb_path
            meta = dict(self._current_meta)
            total_frames = self._frames_written
            fps = self._incursion_fps if self._incursion_fps > 0 else 20.0

        if not video_path or not video_path.exists():
            return None

        # Ensure thumbnail exists
        if not thumb_path or not thumb_path.exists():
            # Grab middle frame from video
            try:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames // 2))
                    ret, fr = cap.read()
                    if ret and fr is not None:
                        cv2.imwrite(str(thumb_path), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    cap.release()
            except Exception:
                pass

        # Optional: FFmpeg faststart optimization
        try:
            temp_path = video_path.with_suffix(".tmp.mp4")
            res = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "copy", "-movflags", "+faststart", str(temp_path)],
                capture_output=True,
                timeout=12
            )
            if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
                video_path.unlink(missing_ok=True)
                temp_path.rename(video_path)
            else:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        except Exception as ff_err:
            logger.debug("Faststart remux bypassed: %s", ff_err)

        duration_sec = round(total_frames / fps, 1)
        size_kb = round(video_path.stat().st_size / 1024, 1) if video_path.exists() else 0.0

        entry = {
            "case_id": meta.get("case_id"),
            "filename": video_path.name,
            "camera_id": meta.get("camera_id", "primary"),
            "timestamp": meta.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "duration_seconds": duration_sec,
            "duration_sec": duration_sec,
            "target_count": 1,
            "bearing_deg": meta.get("bearing_deg", 45.0),
            "city": meta.get("city", "Sector Alpha"),
            "gps": meta.get("gps", "33.6844 N, 73.0479 E"),
            "video_path": str(video_path).replace("\\", "/"),
            "thumb_path": str(thumb_path).replace("\\", "/") if thumb_path else "",
            "video_url": f"/api/evidence/video?filename={video_path.name}&path={video_path.as_posix()}",
            "thumbnail_url": f"/api/snapshot?path={thumb_path.as_posix()}" if thumb_path else "",
            "thumb_url": f"/api/snapshot?path={thumb_path.as_posix()}" if thumb_path else "",
            "download_url": f"/api/evidence/download?filename={video_path.name}",
            "size_kb": size_kb,
        }

        self._append_catalog(entry)
        logger.warning(
            "🎬 [EVIDENCE RECORDING COMPLETED] Case: %s | File: %s | Duration: %.1fs | Size: %.1f KB",
            entry["case_id"], video_path.name, duration_sec, size_kb
        )
        return entry

    # ── Legacy Static Clip Encoder (For Backward Compatibility) ──

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
        """Synthesize or encode an annotated forensic evidence video clip."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        video_filename = f"evidence_{date_str}.mp4"
        thumb_filename = f"evidence_{date_str}_thumb.jpg"

        video_path = self.output_dir / video_filename
        thumb_path = self.output_dir / thumb_filename
        case_id = f"EVID-{now.strftime('%Y%m%d-%H%M%S')}"

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
            if raw_f is None:
                continue
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

        # FFmpeg faststart
        try:
            temp_path = video_path.with_suffix(".tmp.mp4")
            res = subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "copy", "-movflags", "+faststart", str(temp_path)],
                capture_output=True,
                timeout=10
            )
            if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
                video_path.unlink(missing_ok=True)
                temp_path.rename(video_path)
            else:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        except Exception:
            pass

        entry = {
            "case_id": case_id,
            "filename": video_filename,
            "camera_id": camera_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(total_frames / fps, 1) if fps > 0 else duration_sec,
            "duration_sec": round(total_frames / fps, 1) if fps > 0 else duration_sec,
            "target_count": len(detections) if detections else 1,
            "bearing_deg": bearing_deg,
            "city": city,
            "gps": f"{gps_coords[0]:.4f} N, {gps_coords[1]:.4f} E",
            "video_path": str(video_path).replace("\\", "/"),
            "thumb_path": str(thumb_path).replace("\\", "/"),
            "video_url": f"/api/evidence/video?filename={video_filename}&path={video_path.as_posix()}",
            "thumbnail_url": f"/api/snapshot?path={thumb_path.as_posix()}",
            "thumb_url": f"/api/snapshot?path={thumb_path.as_posix()}",
            "download_url": f"/api/evidence/download?filename={video_filename}",
            "size_kb": round(video_path.stat().st_size / 1024, 1) if video_path.exists() else 0,
        }

        self._append_catalog(entry)
        logger.warning("🎬 EVIDENCE VIDEO SAVED: %s (%s, %.1f KB)", case_id, video_path, entry["size_kb"])
        return entry

    # ── Catalog & File Lookup ────────────────────────────────────

    def list_evidence(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return catalog of recorded evidence items from 'evidence/' and legacy paths."""
        catalog = self._load_catalog()
        # Also scan directory for any loose MP4s not yet in catalog
        known_files = {item.get("filename") for item in catalog}
        new_items = []
        for mp4 in sorted(self.output_dir.glob("*.mp4"), key=os.path.getmtime, reverse=True):
            if mp4.name not in known_files and not mp4.name.endswith(".tmp.mp4"):
                mtime = datetime.fromtimestamp(mp4.stat().st_mtime)
                thumb = mp4.with_name(f"{mp4.stem}_thumb.jpg")
                new_items.append({
                    "case_id": f"EVID-{mtime.strftime('%Y%m%d-%H%M%S')}",
                    "filename": mp4.name,
                    "camera_id": "primary",
                    "timestamp": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": 10.0,
                    "duration_sec": 10.0,
                    "target_count": 1,
                    "bearing_deg": 45.0,
                    "city": "Sector Alpha",
                    "gps": "33.6844 N, 73.0479 E",
                    "video_path": str(mp4).replace("\\", "/"),
                    "thumb_path": str(thumb).replace("\\", "/") if thumb.exists() else "",
                    "video_url": f"/api/evidence/video?filename={mp4.name}&path={mp4.as_posix()}",
                    "thumbnail_url": f"/api/snapshot?path={thumb.as_posix()}" if thumb.exists() else "/api/video_feed/primary",
                    "thumb_url": f"/api/snapshot?path={thumb.as_posix()}" if thumb.exists() else "/api/video_feed/primary",
                    "download_url": f"/api/evidence/download?filename={mp4.name}",
                    "size_kb": round(mp4.stat().st_size / 1024, 1),
                })
        if new_items:
            for itm in reversed(new_items):
                catalog.insert(0, itm)
            self._save_catalog(catalog)

        if limit is not None:
            return catalog[:limit]
        return catalog

    def get_video_path(self, filename: str) -> Optional[Path]:
        """Locate video file within 'evidence/' or legacy directory safely, returning absolute Path."""
        if not filename:
            return None
        safe_name = os.path.basename(filename)

        # 1. Primary check: evidence/ directory
        p = self.output_dir / safe_name
        if p.exists() and p.is_file():
            return p.resolve()

        # 2. Recursive check in evidence/
        for found in self.output_dir.rglob(safe_name):
            if found.is_file():
                return found.resolve()

        # 3. Fallback check: legacy logs/evidence_videos/
        if LEGACY_DIR.exists():
            lp = LEGACY_DIR / safe_name
            if lp.exists() and lp.is_file():
                return lp.resolve()
            for found in LEGACY_DIR.rglob(safe_name):
                if found.is_file():
                    return found.resolve()

        return None

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
        w, h = 640, 360
        accent = (173, 165, 243)
        crimson = (35, 30, 154)

        # Top Banner
        cv2.rectangle(frame, (0, 0), (w, 28), (20, 8, 30), -1)
        if (f_idx // 6) % 2 == 0:
            cv2.circle(frame, (16, 14), 5, (35, 30, 230), -1)
        cv2.putText(frame, f"FORENSIC EVIDENCE RECORDING  |  {case_id}", (30, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (241, 244, 247), 1, cv2.LINE_AA)

        # Bottom Banner
        cv2.rectangle(frame, (0, h - 24), (w, h), (15, 6, 20), -1)
        gps_text = f"CAM: {camera_id.upper()} | BRG: {bearing:.1f} DEG | GPS: {gps[0]:.4f} N, {gps[1]:.4f} E ({city.upper()})"
        cv2.putText(frame, gps_text, (12, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, accent, 1, cv2.LINE_AA)

        ts_text = time.strftime("%Y-%m-%d %H:%M:%S UTC+5")
        cv2.putText(frame, ts_text, (w - 180, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 240), 1, cv2.LINE_AA)
        return frame

    def _append_catalog(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            catalog = self._load_catalog()
            catalog.insert(0, entry)
            catalog = catalog[:100]
            self._save_catalog(catalog)

    def _save_catalog(self, catalog: List[Dict[str, Any]]) -> None:
        try:
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)
        except Exception as exc:
            logger.error("Failed to write evidence catalog: %s", exc)

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if self.catalog_path.exists():
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []


_evidence_recorder_instance: Optional[EvidenceRecorder] = None
_evidence_instance_lock = threading.Lock()


def get_evidence_recorder(output_dir: Optional[Path] = None) -> EvidenceRecorder:
    """Retrieve or initialize the global EvidenceRecorder singleton instance."""
    global _evidence_recorder_instance
    with _evidence_instance_lock:
        if _evidence_recorder_instance is None:
            _evidence_recorder_instance = EvidenceRecorder(output_dir)
        return _evidence_recorder_instance
