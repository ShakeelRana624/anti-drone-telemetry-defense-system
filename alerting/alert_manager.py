"""
alert_manager.py — Alert Orchestrator
=======================================
Coordinates all four alerting channels simultaneously when a
confirmed drone detection is received.

Key features:
  • Per-camera cooldown timer (default 60 s) to prevent alert spam
  • Each channel fires in its own daemon thread (channel failure
    cannot block others)
  • Bearing estimate included in all messages
  • Dashboard SSE push on every alert
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Dict, List

from alerting.siren    import SirenController
from alerting.gsm_sms  import GsmSmsNotifier
from alerting.landline import LandlineDialer
from alerting.rf_comset import RfComsetInterface
from utils.bearing_calculator import BearingCalculator

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Orchestrates all alerting subsystems with per-camera cooldown.

    Config section: config['alerting']
        cooldown_seconds (int) : seconds between consecutive alerts per camera
    """

    def __init__(self, config: dict) -> None:
        self._cfg  = config.get("alerting", {})
        self._cooldown: float = float(self._cfg.get("cooldown_seconds", 60))

        self.siren    = SirenController  (self._cfg.get("siren",    {}))
        self.gsm      = GsmSmsNotifier   (self._cfg.get("gsm_sms",  {}))
        self.landline = LandlineDialer   (self._cfg.get("landline", {}))
        self.rf       = RfComsetInterface(self._cfg.get("rf_comset",{}))
        self._bearing = BearingCalculator(config.get("bearing",     {}))

        # camera_id → last alert epoch
        self._last_alert: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def trigger(self, detections: List, camera_id: str) -> None:
        """
        Trigger all alert channels for a confirmed detection.
        Respects per-camera cooldown.
        """
        now = time.monotonic()
        with self._lock:
            last = self._last_alert.get(camera_id, 0.0)
            if now - last < self._cooldown:
                remaining = self._cooldown - (now - last)
                logger.debug(
                    "Alert suppressed for camera '%s' — cooldown %.0fs remaining.",
                    camera_id, remaining,
                )
                return
            self._last_alert[camera_id] = now

        # Build context dict for all channels
        bearing = self._bearing.estimate(detections, camera_id)
        ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context = {
            "timestamp": ts,
            "camera_id": camera_id,
            "bearing":   f"{bearing:.1f}",
            "count":     len(detections),
        }

        logger.warning(
            "🚨 ALERT [%s] camera=%s  bearing=%.1f°  targets=%d",
            ts, camera_id, bearing, len(detections),
        )

        # Launch all channels in parallel daemon threads
        threads = [
            threading.Thread(target=self._fire_siren,    daemon=True),
            threading.Thread(target=self._fire_gsm,      args=(context,), daemon=True),
            threading.Thread(target=self._fire_landline, daemon=True),
            threading.Thread(target=self._fire_rf,       args=(context,), daemon=True),
            threading.Thread(target=self._push_dashboard,args=(context, detections), daemon=True),
        ]
        for t in threads:
            t.start()
        # Non-blocking: we deliberately do not join here so the detector
        # thread can continue processing frames immediately.

    def cooldown_remaining(self, camera_id: str) -> float:
        """Seconds remaining on this camera's cooldown (0 if ready)."""
        elapsed = time.monotonic() - self._last_alert.get(camera_id, 0.0)
        return max(0.0, self._cooldown - elapsed)

    # ──────────────────────────────────────────────────────────────
    # Channel dispatchers
    # ──────────────────────────────────────────────────────────────

    def _fire_siren(self) -> None:
        try:
            self.siren.activate()
        except Exception as exc:
            logger.error("Siren failed: %s", exc)

    def _fire_gsm(self, context: dict) -> None:
        try:
            self.gsm.send_alert(context)
        except Exception as exc:
            logger.error("GSM/SMS failed: %s", exc)

    def _fire_landline(self) -> None:
        try:
            self.landline.dial()
        except Exception as exc:
            logger.error("Landline failed: %s", exc)

    def _fire_rf(self, context: dict) -> None:
        try:
            self.rf.transmit(context)
        except Exception as exc:
            logger.error("RF comset failed: %s", exc)

    def _push_dashboard(self, context: dict, detections: List) -> None:
        """Push alert event to Flask SSE stream."""
        try:
            from dashboard.app import push_sse_event  # type: ignore
            push_sse_event({
                "event":     "alert",
                "timestamp": context["timestamp"],
                "camera_id": context["camera_id"],
                "bearing":   context["bearing"],
                "count":     context["count"],
            })
        except Exception:
            pass  # Dashboard may not be running
