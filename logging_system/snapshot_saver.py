"""
snapshot_saver.py — Detection Snapshot Saver
=============================================
Saves annotated JPEG snapshots for every confirmed detection.

Workflow:
  1. Triggered on confirmed drone detections by DroneDetector.
  2. Renders tactical annotations (red bounding boxes, IDs, confidence, bearing).
  3. Watermarks frame with timestamp and operational banners.
  4. Automatically structures saved images into date-partitioned folders:
     logs/snapshots/<YYYY-MM-DD>/<camera_id>/<HHMMSSmmm>.jpg
  5. Returns absolute Path to the saved JPEG image for event logging and gallery display.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Annotation style
_BOX_CLR   = (0, 0, 220)       # BGR red
_TXT_CLR   = (255, 255, 255)   # white
_BRG_CLR   = (0, 220, 255)     # yellow for bearing text
_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


class SnapshotSaver:
    """
    Annotates video frames with tactical bounding boxes and saves high-quality JPEGs to disk.

    Config keys (under 'logging'):
        snapshot_dir (str)  : root directory for snapshots
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the SnapshotSaver subsystem.

        Args:
            config (dict): Configuration dictionary containing 'logging' section.
        """
        log_cfg = config.get("logging", {})
        self._root: Path = Path(log_cfg.get("snapshot_dir", "logs/snapshots"))
        self._root.mkdir(parents=True, exist_ok=True)
        self._jpeg_quality: int = 92

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def save(
        self,
        frame: np.ndarray,
        detections: List,
        camera_id: str,
        bearing_deg: Optional[float] = None,
    ) -> Path:
        """
        Annotate `frame` with target detections and write JPEG to disk.

        Args:
            frame (np.ndarray): Original image frame.
            detections (List): List of active Detection objects to annotate.
            camera_id (str): Camera identifier reporting the detection.
            bearing_deg (Optional[float]): Computed bearing angle in degrees.

        Returns:
            Path: Absolute Path to the saved JPEG snapshot file.
        """
        if bearing_deg is None and detections:
            cx = sum(d.center[0] for d in detections) / len(detections)
            bearing_deg = (cx - 0.5) * 90.0

        annotated    = self._annotate(frame.copy(), detections, bearing_deg)
        out_path     = self._build_path(camera_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(
            str(out_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if ok:
            logger.info("Snapshot saved → %s", out_path)
        else:
            logger.error("Failed to write snapshot → %s", out_path)

        return out_path

    # ──────────────────────────────────────────────────────────────
    # Annotation
    # ──────────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List,
        bearing_deg: Optional[float],
    ) -> np.ndarray:
        """
        Render tactical overlays, bounding boxes, labels, and watermarks onto the frame.

        Args:
            frame (np.ndarray): Image array copy to annotate.
            detections (List): Detections to draw onto the frame.
            bearing_deg (Optional[float]): Bearing angle to display in banner.

        Returns:
            np.ndarray: Annotated image array.
        """
        h, w = frame.shape[:2]

        # --- bounding boxes ---
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            px1 = int(x1 * w); py1 = int(y1 * h)
            px2 = int(x2 * w); py2 = int(y2 * h)

            cv2.rectangle(frame, (px1, py1), (px2, py2), _BOX_CLR, 2)

            tid_str = f" [ID:{det.track_id}]" if getattr(det, "track_id", None) is not None else ""
            label   = f"{det.class_name}{tid_str}  {det.confidence:.2f}"
            ly      = py1 - 8 if py1 > 20 else py2 + 18
            # shadow
            cv2.putText(frame, label, (px1 + 1, ly + 1), _FONT, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, label, (px1,     ly),     _FONT, 0.55, _TXT_CLR, 1, cv2.LINE_AA)

        # --- bearing overlay ---
        if bearing_deg is not None:
            direction = _bearing_to_cardinal(bearing_deg)
            brg_text  = f"Bearing: {bearing_deg:+.1f}°  ({direction})"
            cv2.putText(frame, brg_text, (10, 30), _FONT_BOLD, 0.7, (0,0,0),   2, cv2.LINE_AA)
            cv2.putText(frame, brg_text, (10, 30), _FONT_BOLD, 0.7, _BRG_CLR, 1, cv2.LINE_AA)

        # --- timestamp watermark ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(frame, ts, (8, h - 10), _FONT, 0.45, (0,0,0),       1, cv2.LINE_AA)
        cv2.putText(frame, ts, (8, h - 10), _FONT, 0.45, (190,190,190), 1, cv2.LINE_AA)

        # --- "DRONE DETECTED" banner ---
        banner = f"DRONE DETECTED  x{len(detections)}"
        cv2.rectangle(frame, (0, h - 50), (w, h - 32), (0, 0, 180), -1)
        cv2.putText(frame, banner, (10, h - 37), _FONT_BOLD, 0.55, (255,255,255), 1, cv2.LINE_AA)

        return frame

    # ──────────────────────────────────────────────────────────────
    # Path builder
    # ──────────────────────────────────────────────────────────────

    def _build_path(self, camera_id: str) -> Path:
        """
        Construct standard date-partitioned file path for saving a snapshot.

        Args:
            camera_id (str): Camera identifier.

        Returns:
            Path: Constructed target Path for the snapshot image.
        """
        now  = datetime.now(timezone.utc)
        date = now.strftime("%Y-%m-%d")
        ts   = now.strftime("%H%M%S%f")[:11]      # HHMMSSmmm
        return self._root / date / camera_id / f"{ts}.jpg"


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _bearing_to_cardinal(deg: float) -> str:
    """
    Convert a horizontal bearing offset (-90 to +90) into a human-readable direction phrase.

    Args:
        deg (float): Horizontal bearing angle in degrees.

    Returns:
        str: Descriptive string (e.g., 'FAR LEFT', 'CENTRE', 'RIGHT').
    """
    if   deg < -60: return "FAR LEFT"
    elif deg < -30: return "LEFT"
    elif deg < -10: return "SLIGHT LEFT"
    elif deg <  10: return "CENTRE"
    elif deg <  30: return "SLIGHT RIGHT"
    elif deg <  60: return "RIGHT"
    else:           return "FAR RIGHT"
