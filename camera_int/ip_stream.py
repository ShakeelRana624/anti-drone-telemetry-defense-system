"""
ip_stream.py — Robust IP Camera Stream Handler
================================================
Handles real-time RTSP, HTTP (MJPEG/H.264), and device video streaming.
Features:
  - Background double-buffered capture thread
  - TCP transport option for stable RTSP over LAN/WAN
  - Automatic reconnection logic
  - Connection probing for non-technical setup validation
  - Synthetic tactical HUD fallback with live GPS watermark when camera is offline
"""

from __future__ import annotations

import logging
import math
import os
import socket
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Force TCP transport for RTSP in OpenCV ffmpeg backend
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


def probe_ip_camera(
    ip: str,
    port: int = 554,
    timeout_sec: float = 2.0
) -> Tuple[bool, str, float]:
    """
    Test whether a target IP camera address and port are reachable on the network.
    Performs a non-blocking TCP socket connect attempt.

    Args:
        ip: Target IP address or hostname.
        port: Destination TCP port number (typically 554 for RTSP, 80/8080 for HTTP).
        timeout_sec: Maximum wait duration in seconds before timing out.

    Returns:
        Tuple[bool, str, float]: (is_reachable, status_message, latency_milliseconds).
    """
    if not ip or ip.strip() in ["", "0.0.0.0", "localhost"]:
        return False, "Invalid or empty IP address provided.", 0.0

    t_start = time.perf_counter()
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_sec)
        result = sock.connect_ex((ip.strip(), int(port)))
        sock.close()
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        if result == 0:
            return True, f"Host reachable at {ip}:{port} (Response time: {elapsed_ms}ms)", elapsed_ms
        else:
            return False, f"Cannot connect to {ip}:{port} (Port closed or host unreachable). Error code: {result}", elapsed_ms
    except Exception as exc:
        return False, f"Network probe failed: {exc}", 0.0


class IPCameraStream:
    """
    Background worker thread reading frames from an IP (RTSP/HTTP) source.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the IPCameraStream with profile parameters and geographic coordinates.

        Args:
            config: Camera configuration dictionary containing stream parameters, resolution,
                    FPS, sensor type, and base GPS coordinates.
        """
        self.camera_id = config.get("id", "camera")
        self.config = config
        self.source = self._resolve_source(config)
        self.target_fps = config.get("fps", 25)
        self.resolution = tuple(config.get("resolution", [1280, 720]))
        self.sensor_type = config.get("sensor_type", "optical")

        # Telemetry info
        self.city = config.get("city", "Station Alpha")
        self.latitude = float(config.get("latitude", 33.6844))
        self.longitude = float(config.get("longitude", 73.0479))
        self.azimuth_deg = float(config.get("azimuth_deg", 45.0))

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.healthy = False
        self.latency_ms = 12.0
        self.current_fps = 0.0
        self._frame_count = 0
        self._synthetic_idx = 0

    def _resolve_source(self, cfg: Dict[str, Any]) -> Any:
        """
        Derive the OpenCV VideoCapture source descriptor (RTSP/HTTP URL string or USB index).

        Args:
            cfg: Camera configuration dictionary.

        Returns:
            str | int: Stream connection URI or USB camera index.
        """
        mode = cfg.get("connection_mode", "ip_rtsp")
        custom = (cfg.get("custom_url") or "").strip()
        if custom:
            return custom

        if mode in ("usb", "webcam"):
            try:
                return int(cfg.get("ip", 0))
            except (ValueError, TypeError):
                return 0

        if mode in ("file", "video"):
            fpath = cfg.get("file_path") or cfg.get("ip") or "test_videos/input.mp4"
            return str(fpath)

        ip = (cfg.get("ip") or "").strip()
        # Direct video file support
        if any(ip.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
            return ip

        # Strict user-specified RTSP / IP stream without auto-switching
        ip = (cfg.get("ip") or "").strip()
        if not ip:
            return ""

        port = cfg.get("port", 554)
        user = (cfg.get("username") or "").strip()
        pwd = (cfg.get("password") or "").strip()
        path = (cfg.get("rtsp_path") or "").strip()
        if path and not path.startswith("/"):
            path = "/" + path

        auth = f"{user}:{pwd}@" if user and pwd else (f"{user}@" if user else "")
        if mode == "ip_http":
            return f"http://{auth}{ip}:{port}{path}"
        return f"rtsp://{auth}{ip}:{port}{path}"

    def start(self) -> None:
        """Launch asynchronous background frame capture thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"ip-cam-{self.camera_id}"
        )
        self._thread.start()
        logger.info("IPCameraStream [%s] initialized with source: %s", self.camera_id, self.source)

    def stop(self) -> None:
        """Stop asynchronous capture thread and release hardware video capture device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        logger.info("IPCameraStream [%s] stopped.", self.camera_id)

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Dynamically update streaming and geographical parameters; restarts stream if connection URL changes.

        Args:
            new_config: Dictionary containing updated camera profile values.
        """
        self.config.update(new_config)
        self.city = new_config.get("city") or self.city
        try:
            if "latitude" in new_config and new_config["latitude"] not in (None, ""):
                self.latitude = float(new_config["latitude"])
        except (ValueError, TypeError):
            pass
        try:
            if "longitude" in new_config and new_config["longitude"] not in (None, ""):
                self.longitude = float(new_config["longitude"])
        except (ValueError, TypeError):
            pass
        try:
            if "azimuth_deg" in new_config and new_config["azimuth_deg"] not in (None, ""):
                self.azimuth_deg = float(new_config["azimuth_deg"]) % 360.0
        except (ValueError, TypeError):
            pass
        new_source = self._resolve_source(self.config)

        if new_source != self.source:
            logger.info("IP Camera [%s] source changed to: %s. Reconnecting...", self.camera_id, new_source)
            self.source = new_source
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self.healthy = False

    def read(self) -> np.ndarray:
        """
        Fetch the most recent video frame.
        Returns latest hardware frame if connected, or renders synthetic tactical HUD frame if offline.

        Returns:
            numpy.ndarray: BGR image frame (640x360 or configured resolution).
        """
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()

        # Fallback to high-definition simulated tactical frame with real GPS watermark
        self._synthetic_idx += 1
        return self._render_synthetic_tactical_frame(self._synthetic_idx)

    def _worker_loop(self) -> None:
        """
        Continuous background worker loop for frame polling, health monitoring,
        FPS calculation, and automated network reconnect handling with socket fast-probing.
        """
        interval = 1.0 / max(1, self.target_fps)
        reconnect_delay = 5.0
        last_reconnect_attempt = 0.0

        fps_timer = time.time()
        fps_counter = 0

        while self._running:
            now = time.time()

            # Attempt capture if available
            if self._cap is None:
                if now - last_reconnect_attempt > reconnect_delay:
                    last_reconnect_attempt = now
                    try:
                        cam_ip = self.config.get("ip", "").strip()
                        cam_port = int(self.config.get("port", 554))
                        is_usb = self.config.get("connection_mode") in ("usb", "webcam") or isinstance(self.source, int)
                        is_file = isinstance(self.source, str) and any(str(self.source).lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"])
                        is_reachable = True

                        if not is_usb and not is_file and cam_ip and cam_ip not in ("192.168.1.101", "localhost", "127.0.0.1", ""):
                            is_reachable, _, _ = probe_ip_camera(cam_ip, cam_port, timeout_sec=0.8)

                        if is_reachable:
                            t0 = time.perf_counter()
                            import sys
                            if is_usb and sys.platform == "win32":
                                try:
                                    dev_i = int(self.source)
                                except (ValueError, TypeError):
                                    dev_i = 0
                                cap = cv2.VideoCapture(dev_i, cv2.CAP_DSHOW)
                                if not cap.isOpened():
                                    cap = cv2.VideoCapture(dev_i)
                            else:
                                cap = cv2.VideoCapture(self.source)
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                self._cap = cap
                                self.healthy = True
                                self.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                                logger.info("Connected to camera stream [%s]: %s", self.camera_id, self.source)
                            else:
                                self.healthy = False
                        else:
                            self.healthy = False
                    except Exception as err:
                        self.healthy = False
                        self._cap = None
            else:
                # Read physical or video frame
                try:
                    ret, frame = self._cap.read()
                    # Auto-loop video files for continuous testing
                    is_file = isinstance(self.source, str) and any(str(self.source).lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"])
                    if (not ret or frame is None) and is_file:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self._cap.read()

                    if ret and frame is not None:
                        self.healthy = True
                        fps_counter += 1
                        with self._lock:
                            self._frame = frame
                    else:
                        self.healthy = False
                        self._cap.release()
                        self._cap = None
                        last_reconnect_attempt = now
                except Exception:
                    self.healthy = False
                    self._cap = None

            # Calculate actual FPS
            if now - fps_timer >= 1.0:
                self.current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            time.sleep(interval)

    def _render_synthetic_tactical_frame(self, frame_idx: int) -> np.ndarray:
        """
        Synthesize high-aesthetic tactical HUD preview with live simulated targets, reticle brackets,
        and geographic telemetry (GPS latitude, longitude, and azimuth) when hardware camera is offline.

        Args:
            frame_idx: Frame animation step counter.

        Returns:
            numpy.ndarray: Rendered BGR preview image frame (640x360).
        """
        w, h = 640, 360
        is_thermal = self.sensor_type == "thermal"

        if is_thermal:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = (18, 12, 28)  # Deep wine thermal base
            noise = np.random.randint(0, 14, (h, w, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = (15, 10, 24)
            noise = np.random.randint(0, 10, (h, w, 3), dtype=np.uint8)
            img = cv2.add(img, noise)

        grid_color = (40, 25, 60)
        for x in range(40, w, 80):
            cv2.line(img, (x, 0), (x, h), grid_color, 1)
        for y in range(30, h, 60):
            cv2.line(img, (0, y), (w, y), grid_color, 1)

        # Crosshair & reticle
        cx, cy = w // 2, h // 2
        accent_bgr = (173, 165, 243)
        crimson_bgr = (35, 30, 154)

        cv2.circle(img, (cx, cy), 45, grid_color, 1)
        cv2.circle(img, (cx, cy), 4, accent_bgr, -1)
        cv2.line(img, (cx - 20, cy), (cx - 8, cy), accent_bgr, 1)
        cv2.line(img, (cx + 8, cy), (cx + 20, cy), accent_bgr, 1)
        cv2.line(img, (cx, cy - 20), (cx, cy - 8), accent_bgr, 1)
        cv2.line(img, (cx, cy + 8), (cx, cy + 20), accent_bgr, 1)

        # Clean tactical surveillance standby status (NO FAKE TARGETS)
        cv2.putText(img, "AIRSPACE SECURE - OPTICAL SURVEILLANCE ACTIVE", (cx - 190, cy + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 210), 1, cv2.LINE_AA)
        cv2.putText(img, "AWAITING AERIAL TARGET DETECTION", (cx - 130, cy + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 150), 1, cv2.LINE_AA)

        # Top Bar: Camera Label & Status
        status_text = "LIVE [IP CONNECTED]" if self.healthy else "STANDBY [SIMULATOR MODE]"
        cam_label = self.config.get("name", self.camera_id).replace("—", "-").replace("–", "-").upper()
        cv2.putText(img, f"{status_text}  {cam_label}", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (241, 244, 247), 1, cv2.LINE_AA)

        # Blinking REC dot
        if (frame_idx // 10) % 2 == 0:
            cv2.circle(img, (w - 24, 20), 5, (35, 30, 230), -1)

        # Bottom Bar: GPS Geolocation Telemetry
        gps_text = f"GPS: {self.latitude:.4f} N, {self.longitude:.4f} E | {self.city.upper()} | AZ: {self.azimuth_deg:.1f} DEG"
        cv2.putText(img, gps_text, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, accent_bgr, 1, cv2.LINE_AA)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, time_str, (w - 170, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 220), 1, cv2.LINE_AA)

        return img
