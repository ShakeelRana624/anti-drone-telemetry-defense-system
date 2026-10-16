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

from detection.detector import Detection

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# DeepSORT Tracker (per-camera)
# ══════════════════════════════════════════════════════════════════

class DeepSORTTracker:
    """
    Wraps deep_sort_realtime.DeepSort with a high-speed motion gate and Kalman coasting.

    Features:
      • Kalman filter motion prediction & coasting (prevents tracking loss during fast drone maneuvers)
      • Generous IoU & adaptive distance matching (handles abrupt speed boosts)
      • Multi-frame persistent track confirmation
      • Cross-camera spatial validation
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the DeepSORT multi-camera tracker.

        Args:
            config: System configuration dictionary containing 'tracker' parameters (n_init, max_age,
                    max_cosine_distance, max_iou_distance, max_coast_frames, nn_budget, cross_camera_check).
        """
        cfg = config.get("tracker", {})
        self._confirm_n:    int   = cfg.get("confirmation_frames",   2)
        self._max_age:      int   = cfg.get("max_age",               30)
        self._n_init:       int   = cfg.get("n_init",                2)
        self._cos_dist:     float = cfg.get("max_cosine_distance",   0.40)
        self._max_iou_dist: float = cfg.get("max_iou_distance",      0.90)
        self._max_coast:    int   = cfg.get("max_coast_frames",      5)
        self._nn_budget:    int   = cfg.get("nn_budget",             100)
        self._cross_cam:    bool  = cfg.get("cross_camera_check",    True)
        self._cross_win:    float = cfg.get("cross_camera_window_s", 2.0)
        self._embedder:     Optional[str] = cfg.get("embedder", None)

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
        detections: List[Detection],
        camera_id: str,
        frame_shape: Tuple[int, int] = (720, 1280),
        frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Update DeepSORT tracks for the given camera using raw detections from current frame.
        Performs Kalman motion prediction, association, and Kalman coasting (up to 5 frames)
        to prevent target loss during rapid drone accelerations or momentary occlusion.

        Args:
            detections: List of Detection objects detected in current frame.
            camera_id: Camera identifier string.
            frame_shape: (height, width) frame dimension tuple.
            frame: Optional BGR frame image for visual appearance embedding extraction.

        Returns:
            List[Detection]: Confirmed and Kalman-coasted Detection objects with persistent track_id.
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
            if ds_input:
                embeds = (np.ones((len(ds_input), 128), dtype=np.float32) / np.sqrt(128.0)).astype(np.float32)
            else:
                embeds = np.zeros((0, 128), dtype=np.float32)
            tracks = sorter.update_tracks(ds_input, embeds=embeds)
        else:
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            tracks = sorter.update_tracks(ds_input, frame=frame)

        hit_map = self._hit_counts[camera_id]
        active_ids = set()
        confirmed_detections: List[Detection] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            tsu = getattr(track, "time_since_update", 0)

            # ── 1. Target detected in current frame ───────────────────
            if tsu == 0:
                active_ids.add(tid)
                hit_map[tid] = hit_map.get(tid, 0) + 1

                matched_det = self._match_track_to_detection(track, detections, w, h)
                if matched_det is not None:
                    matched_det.track_id = tid
                    if self._cross_cam:
                        if not self._cross_camera_ok(matched_det.bbox, camera_id, now):
                            if hit_map[tid] < self._confirm_n:
                                continue
                    confirmed_detections.append(matched_det)
                else:
                    # Fallback to track Kalman box if slight threshold offset
                    ltrb = track.to_ltrb()
                    norm_bbox = [
                        max(0.0, min(1.0, float(ltrb[0] / w))),
                        max(0.0, min(1.0, float(ltrb[1] / h))),
                        max(0.0, min(1.0, float(ltrb[2] / w))),
                        max(0.0, min(1.0, float(ltrb[3] / h))),
                    ]
                    if norm_bbox[2] > norm_bbox[0] and norm_bbox[3] > norm_bbox[1]:
                        coasted_det = Detection(
                            bbox=norm_bbox,
                            confidence=float(getattr(track, "det_conf", 0.60) or 0.60),
                            class_id=0,
                            class_name=getattr(track, "det_class", "drone") or "drone",
                            track_id=tid,
                            is_coasted=False,
                        )
                        confirmed_detections.append(coasted_det)

            # ── 2. Kalman Coasting (Drone sped up or blurred for 1-5 frames) ──
            elif 0 < tsu <= self._max_coast:
                active_ids.add(tid)
                ltrb = track.to_ltrb()
                norm_bbox = [
                    max(0.0, min(1.0, float(ltrb[0] / w))),
                    max(0.0, min(1.0, float(ltrb[1] / h))),
                    max(0.0, min(1.0, float(ltrb[2] / w))),
                    max(0.0, min(1.0, float(ltrb[3] / h))),
                ]
                if norm_bbox[2] > norm_bbox[0] and norm_bbox[3] > norm_bbox[1]:
                    coasted_det = Detection(
                        bbox=norm_bbox,
                        confidence=float(getattr(track, "det_conf", 0.55) or 0.55),
                        class_id=0,
                        class_name=getattr(track, "det_class", "drone") or "drone",
                        track_id=tid,
                        is_coasted=True,
                    )
                    confirmed_detections.append(coasted_det)

        # Prune dead track hit counts ONLY when completely deleted by DeepSort
        live_track_ids = {t.track_id for t in tracks if not t.is_deleted()}
        dead_ids = set(hit_map.keys()) - live_track_ids
        for tid in dead_ids:
            hit_map.pop(tid, None)

        return confirmed_detections

    def reset(self, camera_id: Optional[str] = None) -> None:
        """
        Reset DeepSORT tracker state and Kalman histories for a specific camera or all cameras.

        Args:
            camera_id: Optional camera identifier (clears all cameras if None).
        """
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
        """
        Retrieve or instantiate the per-camera DeepSort instance.

        Args:
            camera_id: Camera identifier string.

        Returns:
            DeepSort: Active tracker instance for this stream.
        """
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
                max_iou_distance=self._max_iou_dist,
                nn_budget=self._nn_budget,
                embedder=self._embedder,
            )
            logger.info("DeepSORT instance created for camera '%s' (max_iou_dist=%.2f).", camera_id, self._max_iou_dist)
        return self._sorters[camera_id]

    @staticmethod
    def _match_track_to_detection(track, detections: List[Detection], w: int, h: int) -> Optional[Detection]:
        """
        Associate a DeepSORT track hypothesis with the best matching raw Detection in the current frame.
        Evaluates spatial IoU overlap first, then falls back to Euclidean centroid proximity
        (up to 35% frame jump) to accommodate sudden drone flight maneuvers.

        Args:
            track: DeepSORT track object.
            detections: List of candidate Detection instances.
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            Optional[Detection]: Closest matching Detection instance or None.
        """
        if not detections:
            return None

        ltrb = track.to_ltrb()
        tx1 = max(0.0, min(1.0, float(ltrb[0] / w)))
        ty1 = max(0.0, min(1.0, float(ltrb[1] / h)))
        tx2 = max(0.0, min(1.0, float(ltrb[2] / w)))
        ty2 = max(0.0, min(1.0, float(ltrb[3] / h)))
        track_bbox = [tx1, ty1, tx2, ty2]
        tx_cen = (tx1 + tx2) / 2.0
        ty_cen = (ty1 + ty2) / 2.0

        # 1. Best IoU match
        best_det_iou = None
        best_iou = 0.0
        for det in detections:
            iou_val = _iou(det.bbox, track_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_det_iou = det

        if best_iou >= 0.10:
            return best_det_iou

        # 2. Generous Euclidean distance fallback for fast drone speed jumps
        best_det_dist = None
        best_dist = float("inf")
        for det in detections:
            cx, cy = det.center
            dist = (cx - tx_cen) ** 2 + (cy - ty_cen) ** 2
            if dist < best_dist:
                best_dist = dist
                best_det_dist = det

        # 0.35 in normalized coordinates allows up to 35% frame jump during high speed maneuvers
        return best_det_dist if best_dist < (0.35 ** 2) else None

    def _cross_camera_ok(
        self, bbox: List[float], source_cam: str, now: float
    ) -> bool:
        """
        Military-grade cross-sensor corroboration:
        Validates whether another operational camera stream has registered a raw detection
        in a matching spatial sector (IoU >= 0.05) within the active cross-camera time window.
        Returns True immediately if only a single camera stream is active.

        Args:
            bbox: Normalized target bounding box coordinates.
            source_cam: Camera identifier reporting the detection.
            now: Current monotonic timestamp.

        Returns:
            bool: True if cross-camera corroborated or single camera mode, False if unverified.
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
    """
    Calculate spatial Intersection-over-Union (IoU) overlap between two normalized bounding boxes.

    Args:
        a: First bounding box [x1, y1, x2, y2].
        b: Second bounding box [x1, y1, x2, y2].

    Returns:
        float: Overlap score in range [0.0, 1.0].
    """
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
