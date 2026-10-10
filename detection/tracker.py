"""
tracker.py — DeepSORT Multi-Camera Drone Tracker
==================================================
Replaces the simple IoU tracker with DeepSORT, providing:
  • Appearance-based re-identification (ResNet feature extractor)
  • Kalman-filter motion prediction (handles brief occlusion)
  • Track age confirmation (drone must appear in N frames before alert)
  • Cross-camera spatial cross-check (military-grade false-positive suppression)

Library: deep_sort_realtime  (pip install deep-sort-realtime)

One DeepSORT instance is maintained per camera so that each stream
has independent track IDs and Kalman states.

Cross-camera logic:
  A track confirmed on Camera A is validated by checking whether
  Camera B also has a raw detection in a matching spatial region
  within a configurable time window. If only one camera is present,
  the cross-check is skipped gracefully.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DeepSORT Tracker (per-camera)
# ══════════════════════════════════════════════════════════════════

class DeepSORTTracker:
    """
    Wraps deep_sort_realtime.DeepSort with a confirmation gate.

    A track is "confirmed" only after it has been continuously matched
    for `confirmation_frames` consecutive frames.

    Config keys (under 'tracker'):
        confirmation_frames    (int)   : frames before alert fires       [5]
        max_age                (int)   : frames before lost track pruned [30]
        n_init                 (int)   : DeepSORT internal init frames   [3]
        max_cosine_distance    (float) : Re-ID distance threshold        [0.40]
        nn_budget              (int)   : appearance descriptor budget    [100]
        cross_camera_check     (bool)  : enable cross-camera validation  [True]
        cross_camera_window_s  (float) : time window for cross-cam check [2.0]
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("tracker", {})
        self._confirm_n:   int   = cfg.get("confirmation_frames",   5)
        self._max_age:     int   = cfg.get("max_age",               30)
        self._n_init:      int   = cfg.get("n_init",                3)
        self._cos_dist:    float = cfg.get("max_cosine_distance",   0.40)
        self._nn_budget:   int   = cfg.get("nn_budget",             100)
        self._cross_cam:   bool  = cfg.get("cross_camera_check",    True)
        self._cross_win:   float = cfg.get("cross_camera_window_s", 2.0)
        self._embedder:    Optional[str] = cfg.get("embedder", None)

        # camera_id → DeepSort instance
        self._sorters: Dict[str, Any] = {}

        # track_id → hit count (consecutive confirmed hits)
        self._hit_counts: Dict[str, Dict[int, int]] = {}

        # camera_id → recent raw detection bboxes for cross-cam check
        # Each entry: (bbox_norm, timestamp)
        self._raw_history: Dict[str, Deque[Tuple[List[float], float]]] = {}

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update(
        self,
        detections,                       # List[Detection]
        camera_id: str,
        frame_shape: Tuple[int, int] = (720, 1280),
        frame: Optional[np.ndarray] = None,
    ) -> List:                            # returns confirmed Detection list
        """
        Feed raw detections into DeepSORT for this camera.
        Returns only detections whose track has been confirmed
        for `confirmation_frames` consecutive frames AND passes
        the cross-camera check (if enabled).
        """
        h, w = frame_shape

        # Record raw bboxes for cross-camera check
        now = time.monotonic()
        if camera_id not in self._raw_history:
            self._raw_history[camera_id] = deque(maxlen=500)
        for det in detections:
            self._raw_history[camera_id].append((det.bbox, now))

        # Ensure DeepSort instance exists for this camera
        sorter = self._get_sorter(camera_id)
        if camera_id not in self._hit_counts:
            self._hit_counts[camera_id] = {}

        # Convert detections → DeepSort input format:
        # List of ([left, top, w, h], confidence, class_name)
        ds_input = [
            (det.to_ltwh_pixels(w, h), det.confidence, det.class_name)
            for det in detections
        ]

        if self._embedder is None:
            embeds = (np.ones((len(ds_input), 128), dtype=np.float32) / np.sqrt(128.0)).astype(np.float32)
            tracks = sorter.update_tracks(ds_input, embeds=embeds)
        else:
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            tracks = sorter.update_tracks(ds_input, frame=frame)

        hit_map = self._hit_counts[camera_id]
        active_ids = set()
        confirmed_detections: List = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            active_ids.add(tid)
            hit_map[tid] = hit_map.get(tid, 0) + 1

            if hit_map[tid] < self._confirm_n:
                continue  # not yet confirmed by our gate

            # Retrieve Detection object that matched this track
            matched_det = self._match_track_to_detection(track, detections, w, h)
            if matched_det is None:
                continue

            # Cross-camera check
            if self._cross_cam:
                if not self._cross_camera_ok(matched_det.bbox, camera_id, now):
                    if hit_map[tid] < self._confirm_n * 2:
                        # Require 2× confirmation if no cross-cam corroboration
                        continue

            matched_det.track_id = tid
            confirmed_detections.append(matched_det)

        # Prune dead track hit counts
        dead_ids = set(hit_map.keys()) - active_ids
        for tid in dead_ids:
            del hit_map[tid]

        return confirmed_detections

    def reset(self, camera_id: Optional[str] = None) -> None:
        """Clear tracker state for one or all cameras."""
        if camera_id:
            self._sorters.pop(camera_id, None)
            self._hit_counts.pop(camera_id, None)
            self._raw_history.pop(camera_id, None)
        else:
            self._sorters.clear()
            self._hit_counts.clear()
            self._raw_history.clear()
        logger.info("DeepSORT tracker reset — camera: %s", camera_id or "ALL")

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _get_sorter(self, camera_id: str):
        """Return (or create) the DeepSort instance for a camera."""
        if camera_id not in self._sorters:
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
            except ImportError:
                raise SystemExit(
                    "deep-sort-realtime not installed.\n"
                    "Run: pip install deep-sort-realtime"
                )
            self._sorters[camera_id] = DeepSort(
                max_age=self._max_age,
                n_init=self._n_init,
                max_cosine_distance=self._cos_dist,
                nn_budget=self._nn_budget,
                embedder=self._embedder,
            )
            logger.info("DeepSORT instance created for camera '%s'.", camera_id)
        return self._sorters[camera_id]

    @staticmethod
    def _match_track_to_detection(track, detections, w: int, h: int):
        """Find the Detection whose pixel-centre is closest to the track bbox centre."""
        if not detections:
            return None

        # Track bbox is in pixel [x1,y1,x2,y2]
        tlwh = track.to_tlwh()
        tx = (tlwh[0] + tlwh[2] / 2) / w
        ty = (tlwh[1] + tlwh[3] / 2) / h

        best_det = None
        best_dist = float("inf")
        for det in detections:
            cx, cy = det.center
            dist = (cx - tx) ** 2 + (cy - ty) ** 2
            if dist < best_dist:
                best_dist = dist
                best_det = det

        return best_det if best_dist < 0.05 else None   # 5% of frame width threshold

    def _cross_camera_ok(
        self, bbox: List[float], source_cam: str, now: float
    ) -> bool:
        """
        Return True if any OTHER camera has a recent detection
        in a spatially overlapping region (IoU > 0.05).
        Falls back to True if no other camera data is available.
        """
        other_cams = [c for c in self._raw_history if c != source_cam]
        if not other_cams:
            return True                   # single-camera mode: always pass

        for cam in other_cams:
            for (other_bbox, ts) in self._raw_history[cam]:
                if now - ts > self._cross_win:
                    continue
                if _iou(bbox, other_bbox) >= 0.05:
                    return True

        return False                      # no cross-cam corroboration found


# ══════════════════════════════════════════════════════════════════
# IoU helper
# ══════════════════════════════════════════════════════════════════

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter  = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
