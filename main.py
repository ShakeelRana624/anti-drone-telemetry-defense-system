"""
main.py — System Entry Point
==============================
AI-Based Autonomous Drone Detection & Early Warning System
Military-Grade | Edge-Deployed | Fully Offline

Startup sequence:
  1. Load config
  2. Initialise logging subsystem
  3. Create logs/models directories
  4. Wire all components
  5. Start health monitor, dashboard, camera streams
  6. Block on signal (SIGINT/SIGTERM) → graceful shutdown
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

import yaml


# ══════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ══════════════════════════════════════════════════════════════════

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level   = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "system.log", encoding="utf-8"),
        ],
    )
    # Suppress noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def ensure_directories(config: dict) -> None:
    """Create required runtime directories if they don't exist."""
    paths = [
        "models",
        config["logging"]["event_log_dir"],
        config["logging"]["snapshot_dir"],
        "logs",
    ]
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Config & logging ───────────────────────────────────────
    config = load_config()
    setup_logging(config)
    ensure_directories(config)

    log = logging.getLogger("main")
    log.info("═" * 65)
    log.info("  AI-Based Autonomous Drone Detection & Early Warning System")
    log.info("  Military-Grade | YOLO11m + SAHI + TTA + DeepSORT")
    log.info("═" * 65)

    # ── Import components after logging is configured ──────────
    from detection.detector       import DroneDetector
    from detection.tracker        import DeepSORTTracker
    from detection.camera_manager import CameraManager
    from alerting.alert_manager   import AlertManager
    from logging_system.event_logger   import EventLogger
    from logging_system.snapshot_saver import SnapshotSaver
    from dashboard.app            import DashboardApp
    from utils.system_health      import SystemHealthMonitor

    # ── Instantiate subsystems ─────────────────────────────────
    log.info("Initialising subsystems …")

    event_logger   = EventLogger(config)
    snapshot_saver = SnapshotSaver(config)
    alert_manager  = AlertManager(config)
    tracker        = DeepSORTTracker(config)
    detector       = DroneDetector(
        config, tracker, alert_manager, event_logger, snapshot_saver
    )
    camera_manager = CameraManager(config, detector)
    health_monitor = SystemHealthMonitor(config)
    dashboard      = DashboardApp(config, event_logger)

    # Inject shared state into dashboard for /api/status
    dashboard.model_path = detector.model_path

    # ── Log startup event ──────────────────────────────────────
    event_logger.log_system("startup", {
        "model":     detector.model_path,
        "sahi":      detector.using_sahi,
        "tta":       detector.using_tta,
        "cameras":   list(config.get("cameras", {}).keys()),
    })

    # ── Graceful shutdown handler ──────────────────────────────
    def shutdown(signum, frame):  # noqa: ANN001
        log.info("Shutdown signal received (%s). Stopping …", signum)
        camera_manager.stop()
        health_monitor.stop()
        event_logger.log_system("shutdown", {})
        event_logger.close()
        log.info("System stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Start all subsystems ───────────────────────────────────
    log.info("Starting dashboard …")
    dashboard.start()

    log.info("Starting health monitor …")
    health_monitor.start()

    log.info("Starting camera streams …")
    camera_manager.start()

    # Keep dashboard camera status updated
    import threading, time
    def _status_sync():
        while True:
            dashboard.camera_status = camera_manager.status()
            time.sleep(5)
    threading.Thread(target=_status_sync, daemon=True, name="status-sync").start()

    log.info("✅ System is ONLINE — Dashboard: http://localhost:%d",
             config["dashboard"]["port"])
    log.info("Press Ctrl+C to stop.")

    # Block main thread
    signal.pause() if hasattr(signal, "pause") else _windows_block()


def _windows_block() -> None:
    """signal.pause() is not available on Windows — use a sleep loop."""
    import time
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
