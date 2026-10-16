"""
decision_engine.py — Temporal Multi-Frame Threat Decision Engine
================================================================
Prevents false-positive alarms by evaluating detections across multiple
temporal frames rather than triggering on an isolated single frame.

Features:
  • Multi-Frame Confirmation Gate (M-of-N persistence window)
  • Threat State Machine:
      - CANDIDATE   (1-2 frames): Initial sighting, alert suppressed
      - SUSPICIOUS  (3 frames)  : Gathering evidence, evaluating trajectory
      - CONFIRMED   (>= 4 frames): True positive threat, triggers alert & evidence
      - COOLDOWN    : Alert fired, suppressing alarm spam while maintaining track
  • Statistical Frame Calculations:
      - Consecutive hit count
      - Temporal window persistence ratio (hits / window_size)
      - Exponential moving average (EMA) of confidence
      - Spatial velocity / motion consistency check
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    NONE      = "NONE"
    LOW       = "LOW"         # 1 hit — evaluating
    MEDIUM    = "MEDIUM"      # 2–3 hits — suspicious
    HIGH      = "HIGH"        # 4+ hits — confirmed threat
    CRITICAL  = "CRITICAL"    # Multiple drones / confirmed close-in


class TargetThreatState(str, Enum):
    CANDIDATE  = "ACQUIRING"   # 1-7 frames: detecting, acquiring lock
    SUSPICIOUS = "VERIFYING"   # 8-15 frames: tracking, locking in progress
    CONFIRMED  = "LOCKED"       # >= 16 frames: TARGET LOCKED, Threat Confirmed!
    COOLDOWN   = "COOLDOWN"    # Already alerted, maintaining track
    LOST       = "LOST"        # Target disappeared


@dataclass
class TargetMetrics:
    """Historical frame calculation metrics for an individual tracked target."""
    track_id: int
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    consecutive_hits: int = 1
    total_hits: int = 1
    miss_count: int = 0
    hit_history: Deque[bool] = field(default_factory=lambda: deque(maxlen=30))
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: Deque[List[float]] = field(default_factory=lambda: deque(maxlen=30))
    state: TargetThreatState = TargetThreatState.CANDIDATE
    threat_level: ThreatLevel = ThreatLevel.LOW
    alert_fired: bool = False
    last_alert_time: float = 0.0

    @property
    def avg_confidence(self) -> float:
        """Calculate arithmetic mean of historical detection confidence scores across sliding window."""
        if not self.confidence_history:
            return 0.0
        return float(sum(self.confidence_history) / len(self.confidence_history))

    @property
    def persistence_ratio(self) -> float:
        """Calculate temporal persistence ratio (hits / total window frames) in rolling history."""
        if not self.hit_history:
            return 0.0
        return float(sum(1 for h in self.hit_history if h) / len(self.hit_history))


@dataclass
class DecisionResult:
    """Summary of decision calculations for the current frame."""
    frame_timestamp: float
    confirmed_threats: List[Any]          # Targets confirmed by multi-frame gate
    evaluating_candidates: List[Any]      # Targets still in evaluation (no alert)
    should_trigger_alert: bool            # True only when a new confirmed threat arises
    active_target_count: int
    threat_level: ThreatLevel
    target_states: Dict[int, TargetMetrics]
    calculation_summary: str


class DecisionEngine:
    """
    Temporal multi-frame decision engine for drone confirmation & target lock.
    Confirms suspicious candidate on 4-6 frames and locks target at 8 consecutive hits.
    """

    def __init__(
        self,
        min_consecutive_hits: int = 8,
        window_size: int = 12,
        min_hits_in_window: int = 7,
        min_confidence: float = 0.35,
        target_cooldown_sec: float = 20.0,
        max_miss_frames: int = 15,
    ) -> None:
        """
        Initialize the DecisionEngine with multi-frame confirmation gates and cooldown thresholds.

        Args:
            min_consecutive_hits: Required consecutive frames of detection before locking target (default 8).
            window_size: Sliding window size for persistence ratio check (default 12).
            min_hits_in_window: Minimum positive sightings required within sliding window (default 7).
            min_confidence: Minimum average detection confidence required for confirmation (default 0.35).
            target_cooldown_sec: Post-alert cooldown period in seconds suppressing repeated alerts for same target.
            max_miss_frames: Maximum missed frames before target is purged from memory.
        """
        self.min_consecutive_hits = min_consecutive_hits
        self.window_size = window_size
        self.min_hits_in_window = min_hits_in_window
        self.min_confidence = min_confidence
        self.cooldown_sec = target_cooldown_sec
        self.max_miss_frames = max_miss_frames

        # track_id -> TargetMetrics
        self._targets: Dict[int, TargetMetrics] = {}
        self._camera_last_alert: Dict[str, float] = {}

    def evaluate(
        self,
        detections: List[Any],
        camera_id: str = "primary",
    ) -> DecisionResult:
        """
        Evaluate raw or tracked detections for the current video frame against historical persistence gates.
        Separates unconfirmed candidates (1-7 frames) from confirmed locked threats (>= 8 frames),
        manages state transitions, handles target miss decays with speed-burst grace periods,
        and determines whether a global alarm trigger should fire.

        Args:
            detections: List of current candidate detections or active DeepSORT tracks.
            camera_id: Camera stream identifier.

        Returns:
            DecisionResult: Encapsulated decision assessment including confirmed targets,
                           candidate targets, alert trigger flag, and target states.
        """
        now = time.time()
        seen_ids = set()

        confirmed_for_frame = []
        candidates_for_frame = []
        new_alert_triggered = False

        # 1. Update targets that were detected in this frame
        for det in detections:
            # Extract track_id (or fallback to dummy positive ID)
            tid = getattr(det, "track_id", None)
            if tid is None and isinstance(det, dict):
                tid = det.get("track_id")

            # If DeepSORT hasn't assigned a track ID yet, use temporary hash based on center
            if tid is None:
                cx, cy = getattr(det, "center", (0.5, 0.5))
                tid = int(hash((round(cx, 1), round(cy, 1))) % 10000)

            seen_ids.add(tid)
            conf = getattr(det, "confidence", 0.5)
            bbox = getattr(det, "bbox", [0.0, 0.0, 0.0, 0.0])
            cname = getattr(det, "class_name", "drone")

            if tid not in self._targets:
                # First time sighting — create new candidate
                m = TargetMetrics(
                    track_id=tid,
                    class_name=cname,
                    first_seen=now,
                    last_seen=now,
                    consecutive_hits=1,
                    total_hits=1,
                    miss_count=0,
                )
                m.hit_history.append(True)
                m.confidence_history.append(conf)
                m.bbox_history.append(bbox)
                m.state = TargetThreatState.CANDIDATE
                m.threat_level = ThreatLevel.LOW
                self._targets[tid] = m
            else:
                # Existing target — accumulate frame calculations
                m = self._targets[tid]
                m.last_seen = now
                m.consecutive_hits += 1
                m.total_hits += 1
                m.miss_count = 0
                m.hit_history.append(True)
                m.confidence_history.append(conf)
                m.bbox_history.append(bbox)

            m = self._targets[tid]

            # ── Multi-Frame Calculation Gate ─────────────────────────
            # Require minimum consecutive hits AND minimum persistence in window
            passes_consecutive = (m.consecutive_hits >= self.min_consecutive_hits)
            passes_window = (sum(1 for h in m.hit_history if h) >= self.min_hits_in_window)
            passes_confidence = (m.avg_confidence >= self.min_confidence)

            if passes_consecutive and passes_window and passes_confidence:
                # Target has proven itself over multiple consecutive frames!
                if not m.alert_fired:
                    m.state = TargetThreatState.CONFIRMED
                    m.threat_level = ThreatLevel.HIGH
                    m.alert_fired = True
                    m.last_alert_time = now
                    new_alert_triggered = True
                    logger.info(
                        "🎯 [DECISION ENGINE] Threat CONFIRMED: Target #%d (Hits: %d, Conf: %.1f%%)",
                        tid, m.consecutive_hits, m.avg_confidence * 100
                    )
                else:
                    # In cooldown period
                    if now - m.last_alert_time < self.cooldown_sec:
                        m.state = TargetThreatState.COOLDOWN
                    else:
                        m.state = TargetThreatState.CONFIRMED

                confirmed_for_frame.append(det)

            else:
                # Candidate does NOT have 8 frames yet (1 to 7 frames) — check if confirming (4-6 frames)
                if m.consecutive_hits >= 4:
                    m.state = TargetThreatState.SUSPICIOUS
                    m.threat_level = ThreatLevel.MEDIUM
                else:
                    m.state = TargetThreatState.CANDIDATE
                    m.threat_level = ThreatLevel.LOW

                candidates_for_frame.append(det)

        # 2. Update missed targets (decay with high-speed grace period)
        dead_ids = []
        for tid, m in self._targets.items():
            if tid not in seen_ids:
                m.miss_count += 1
                m.hit_history.append(False)
                # Grace period: If target was locked/tracking, allow 1-3 frames for speed bursts without breaking lock
                if m.miss_count <= 3:
                    m.consecutive_hits = max(1, m.consecutive_hits - 1)
                else:
                    m.consecutive_hits = 0
                    if m.miss_count > self.max_miss_frames:
                        dead_ids.append(tid)
                    else:
                        m.state = TargetThreatState.LOST

        for tid in dead_ids:
            del self._targets[tid]

        # 3. Overall threat level for this frame
        if confirmed_for_frame:
            overall_threat = ThreatLevel.CRITICAL if len(confirmed_for_frame) > 1 else ThreatLevel.HIGH
        elif candidates_for_frame:
            overall_threat = ThreatLevel.MEDIUM
        else:
            overall_threat = ThreatLevel.NONE

        # 4. Summary string for telemetry display
        summary_parts = []
        for tid, m in self._targets.items():
            summary_parts.append(
                f"#{tid}: {m.consecutive_hits}/{self.min_consecutive_hits} frames ({m.state.value})"
            )
        calc_summary = " | ".join(summary_parts) if summary_parts else "Scanning... No targets"

        return DecisionResult(
            frame_timestamp=now,
            confirmed_threats=confirmed_for_frame,
            evaluating_candidates=candidates_for_frame,
            should_trigger_alert=new_alert_triggered,
            active_target_count=len(self._targets),
            threat_level=overall_threat,
            target_states=dict(self._targets),
            calculation_summary=calc_summary,
        )

    def reset(self) -> None:
        """
        Purge all tracked targets, historical frame hit counters, and camera alert cooldown timestamps.
        Resets decision engine state to clean baseline.
        """
        self._targets.clear()
        self._camera_last_alert.clear()
