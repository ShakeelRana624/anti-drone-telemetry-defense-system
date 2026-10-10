"""
system_health.py — Hardware Health Monitor
===========================================
Periodically samples CPU, RAM, disk, and temperature via psutil.
Emits warnings when thresholds are exceeded and pushes metrics
to the live dashboard SSE stream for real-time display.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthSnapshot:
    """Point-in-time system resource snapshot."""
    cpu_percent:         float
    memory_percent:      float
    disk_percent:        float
    temperature_celsius: Optional[float]
    timestamp:           float


class SystemHealthMonitor:
    """
    Polls system health at a configurable interval.

    Config keys (under 'system_health'):
        check_interval_seconds       (int)   [30]
        cpu_threshold_percent        (float) [85]
        memory_threshold_percent     (float) [80]
        disk_threshold_percent       (float) [90]
        temperature_threshold_celsius(float) [75]
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("system_health", {})
        self._interval:  int   = cfg.get("check_interval_seconds",        30)
        self._cpu_thr:   float = cfg.get("cpu_threshold_percent",          85.0)
        self._mem_thr:   float = cfg.get("memory_threshold_percent",       80.0)
        self._disk_thr:  float = cfg.get("disk_threshold_percent",         90.0)
        self._temp_thr:  float = cfg.get("temperature_threshold_celsius",  75.0)

        self._thread:  Optional[threading.Thread] = None
        self._running: bool = False
        self._latest:  Optional[HealthSnapshot] = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="health-monitor"
        )
        self._thread.start()
        logger.info("SystemHealthMonitor started (interval=%ds).", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("SystemHealthMonitor stopped.")

    @property
    def latest(self) -> Optional[HealthSnapshot]:
        return self._latest

    # ── Internal ─────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            snap = self._sample()
            self._latest = snap
            self._check_thresholds(snap)
            self._push_sse(snap)
            time.sleep(self._interval)

    def _sample(self) -> HealthSnapshot:
        try:
            import psutil  # type: ignore
            cpu  = psutil.cpu_percent(interval=1)
            mem  = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent
            temp = self._read_temperature(psutil)
            return HealthSnapshot(cpu, mem, disk, temp, time.time())
        except ImportError:
            logger.warning("psutil not installed — health monitoring inactive.")
            return HealthSnapshot(0.0, 0.0, 0.0, None, time.time())

    @staticmethod
    def _read_temperature(psutil_module) -> Optional[float]:
        try:
            sensors = psutil_module.sensors_temperatures()
            if not sensors:
                return None
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in sensors and sensors[key]:
                    return sensors[key][0].current
            first = next(iter(sensors.values()))
            return first[0].current if first else None
        except Exception:
            return None

    def _check_thresholds(self, snap: HealthSnapshot) -> None:
        if snap.cpu_percent  >= self._cpu_thr:
            logger.warning("⚠️  CPU critical: %.1f%%", snap.cpu_percent)
        if snap.memory_percent >= self._mem_thr:
            logger.warning("⚠️  RAM critical: %.1f%%", snap.memory_percent)
        if snap.disk_percent >= self._disk_thr:
            logger.warning("⚠️  Disk critical: %.1f%%", snap.disk_percent)
        if snap.temperature_celsius is not None and snap.temperature_celsius >= self._temp_thr:
            logger.warning("⚠️  Temperature critical: %.1f°C", snap.temperature_celsius)

    @staticmethod
    def _push_sse(snap: HealthSnapshot) -> None:
        try:
            from dashboard.app import push_sse_event  # type: ignore
            push_sse_event({
                "event": "health",
                "cpu":   round(snap.cpu_percent,   1),
                "ram":   round(snap.memory_percent, 1),
                "disk":  round(snap.disk_percent,   1),
                "temp":  round(snap.temperature_celsius, 1) if snap.temperature_celsius else None,
            })
        except Exception:
            pass
