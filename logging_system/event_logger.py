"""
event_logger.py — Timestamped Event Logger
===========================================
Writes structured JSON lines to a daily rotating log file AND
maintains an in-memory ring buffer for the live dashboard.

Workflow:
  1. Receives confirmed detection events and system lifecycle events.
  2. Formats events with UTC ISO-8601 timestamps and spatial parameters.
  3. Appends records to a thread-safe in-memory ring buffer (for /api/recent).
  4. Automatically pushes events to active Server-Sent Events (SSE) web subscribers.
  5. Rotates log files daily (events_YYYY-MM-DD.jsonl) and purges files older than max_days.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventLogger:
    """
    Thread-safe event logger with daily file rotation and in-memory ring buffer.

    Config keys (under 'logging'):
        event_log_dir (str)              : directory for .jsonl files
        max_log_days  (int)              : days to retain logs
        event_ring_buffer_size (int)     : events in memory for dashboard
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the EventLogger subsystem.

        Args:
            config (dict): Full system configuration dictionary containing 'logging' section.
        """
        log_cfg = config.get("logging", {})
        self.log_dir: Path  = Path(log_cfg.get("event_log_dir", "logs/events"))
        self.max_days: int  = log_cfg.get("max_log_days", 30)
        buf_size: int       = log_cfg.get("event_ring_buffer_size", 200)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._ring: Deque[Dict[str, Any]] = deque(maxlen=buf_size)
        self._file = None
        self._current_date: str = ""

        self._open_today()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def log_detection(
        self,
        detections: List,
        camera_id: str,
        snapshot_path: Optional[Path] = None,
    ) -> None:
        """
        Log a confirmed multi-frame drone detection incident.

        Args:
            detections (List): List of confirmed Detection objects in the frame.
            camera_id (str): Camera identifier reporting the detection.
            snapshot_path (Optional[Path]): File path to the saved annotated snapshot JPEG.

        Returns:
            None
        """
        from utils.bearing_calculator import BearingCalculator  # lazy to avoid circular
        # bearing is already computed upstream; we receive it via snapshot_path metadata
        # We compute a simple centre-based bearing here for the log record.
        bearing_deg: float = 0.0
        if detections:
            cx = sum((d.center[0] for d in detections), 0.0) / len(detections)
            # simple linear bearing from FOV centre (logged; true calc in BearingCalculator)
            bearing_deg = round((cx - 0.5) * 90.0, 2)

        record: Dict[str, Any] = {
            "event":       "detection",
            "timestamp":   _utc_now(),
            "camera_id":   camera_id,
            "count":       len(detections),
            "bearing_deg": bearing_deg,
            "snapshot":    str(snapshot_path) if snapshot_path else None,
            "detections": [
                {
                    "class":      d.class_name,
                    "confidence": round(d.confidence, 4),
                    "track_id":   getattr(d, "track_id", None),
                    "bbox":       [round(v, 4) for v in d.bbox],
                    "center":     [round(v, 4) for v in d.center],
                }
                for d in detections
            ],
        }
        self._write(record)

    def log_system(self, event_type: str, details: dict) -> None:
        """
        Log a system-level lifecycle or operational event.

        Args:
            event_type (str): Category of event (e.g., 'startup', 'shutdown', 'health_warning').
            details (dict): Additional event-specific key-value telemetry data.

        Returns:
            None
        """
        record = {
            "event":     event_type,
            "timestamp": _utc_now(),
            **details,
        }
        self._write(record)

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve recent in-memory events from the ring buffer.

        Args:
            n (int): Number of most recent records to return (default: 50).

        Returns:
            List[Dict[str, Any]]: Chronological list of recent event dictionaries.
        """
        events = list(self._ring)
        return events[-n:]

    def close(self) -> None:
        """
        Flush buffers and close the open log file stream.

        Returns:
            None
        """
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        """
        Internal write dispatcher: appends to ring buffer and writes to daily file.

        Args:
            record (Dict[str, Any]): Structured JSON event object.

        Returns:
            None
        """
        self._ring.append(record)
        self._push_sse(record)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._open_today()

        try:
            if self._file:
                self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._file.flush()
        except Exception as exc:
            logger.error("EventLogger write failed: %s", exc)

    def _open_today(self) -> None:
        """
        Rotate to or initialize today's JSONL event log file.

        Returns:
            None
        """
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path  = self.log_dir / f"events_{today}.jsonl"
        self._file = open(path, "a", encoding="utf-8")
        self._current_date = today
        logger.info("EventLogger: log file → %s", path)
        self._purge_old_logs()

    def _purge_old_logs(self) -> None:
        """
        Scan log directory and remove files exceeding retention limit (max_days).

        Returns:
            None
        """
        cutoff = datetime.now(timezone.utc).timestamp() - self.max_days * 86400
        for f in self.log_dir.glob("events_*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info("EventLogger: purged old log %s", f.name)
            except OSError:
                pass

    @staticmethod
    def _push_sse(record: Dict[str, Any]) -> None:
        """
        Forward the record to the live dashboard SSE stream if active.

        Args:
            record (Dict[str, Any]): Event record dictionary.

        Returns:
            None
        """
        try:
            from dashboard.app import push_sse_event  # type: ignore
            push_sse_event(record)
        except Exception:
            pass


def _utc_now() -> str:
    """
    Generate an ISO-8601 UTC timestamp string with millisecond precision.

    Returns:
        str: Formatted UTC timestamp (e.g. '2026-10-16T13:30:00.123Z').
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
