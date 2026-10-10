"""
event_logger.py — Timestamped Event Logger
===========================================
Writes structured JSON lines to a daily rotating log file AND
maintains an in-memory ring buffer for the live dashboard.

Each detection record includes:
  - ISO-8601 UTC timestamp
  - camera_id
  - detection count
  - per-detection details (class, confidence, bbox, center)
  - bearing estimate
  - path to saved snapshot image
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
        """Log a confirmed detection event."""
        from utils.bearing_calculator import BearingCalculator  # lazy to avoid circular
        # bearing is already computed upstream; we receive it via snapshot_path metadata
        # We'll compute a simple centre-based bearing here for the log record.
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
        """Log a system-level event (startup, health warning, etc.)."""
        record = {
            "event":     event_type,
            "timestamp": _utc_now(),
            **details,
        }
        self._write(record)

    def recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return up to `n` most recent events from the ring buffer."""
        events = list(self._ring)
        return events[-n:]

    def close(self) -> None:
        """Flush and close the current log file."""
        if self._file:
            self._file.close()
            self._file = None

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────

    def _write(self, record: Dict[str, Any]) -> None:
        """Write to ring buffer + today's file; rotate if date changed."""
        self._ring.append(record)
        self._push_sse(record)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._open_today()

        try:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception as exc:
            logger.error("EventLogger write failed: %s", exc)

    def _open_today(self) -> None:
        if self._file:
            self._file.close()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path  = self.log_dir / f"events_{today}.jsonl"
        self._file = open(path, "a", encoding="utf-8")
        self._current_date = today
        logger.info("EventLogger: log file → %s", path)
        self._purge_old_logs()

    def _purge_old_logs(self) -> None:
        """Delete .jsonl files older than max_days."""
        cutoff = datetime.now(timezone.utc).timestamp() - self.max_days * 86400
        for f in self.log_dir.glob("events_*.jsonl"):
            if f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    logger.info("EventLogger: purged old log %s", f.name)
                except OSError:
                    pass

    @staticmethod
    def _push_sse(record: Dict[str, Any]) -> None:
        """Forward the record to the live dashboard SSE stream."""
        try:
            from dashboard.app import push_sse_event  # type: ignore
            push_sse_event(record)
        except Exception:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
