"""
camera_config.py — Camera Configuration & Persistence
======================================================
Stores and manages IP camera parameters:
  - IP address, Port, Stream protocol, RTSP path, Username, Password
  - GPS Coordinates: Latitude & Longitude
  - City name, Installation Sector, Azimuth Bearing, Sensor Type
Persists settings to camera_int/cameras_config.json and syncs with config/config.yaml.
Designed for non-technical operators to configure cameras via Web UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "cameras_config.json"

DEFAULT_CAMERAS = {
    "primary": {
        "id": "primary",
        "name": "Camera 01 — North Perimeter",
        "sensor_type": "optical",
        "connection_mode": "ip_rtsp",
        "ip": "192.168.1.101",
        "port": 554,
        "username": "admin",
        "password": "",
        "rtsp_path": "/stream1",
        "custom_url": "",
        "city": "Islamabad",
        "latitude": 33.6844,
        "longitude": 73.0479,
        "azimuth_deg": 45.0,
        "fov_deg": 90.0,
        "fps": 25,
        "resolution": [1280, 720],
        "enabled": True,
        "notes": "Main optical zoom covering northern airspace perimeter."
    },
    "secondary": {
        "id": "secondary",
        "name": "Camera 02 — East Sector (Thermal)",
        "sensor_type": "thermal",
        "connection_mode": "ip_rtsp",
        "ip": "192.168.1.102",
        "port": 554,
        "username": "admin",
        "password": "",
        "rtsp_path": "/stream2",
        "custom_url": "",
        "city": "Rawalpindi",
        "latitude": 33.5973,
        "longitude": 73.0479,
        "azimuth_deg": 128.5,
        "fov_deg": 60.0,
        "fps": 25,
        "resolution": [1280, 720],
        "enabled": True,
        "notes": "Long-wave FLIR thermal camera covering eastern vector."
    }
}


class CameraConfigManager:
    """Manages loading, updating, and saving IP camera configurations."""

    def __init__(self, filepath: Optional[Path] = None) -> None:
        self.filepath = filepath or CONFIG_PATH
        self.cameras: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> Dict[str, Dict[str, Any]]:
        """Load cameras from JSON file; create defaults if file does not exist."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.cameras = json.load(f)
                logger.info("Loaded camera configurations from %s", self.filepath)
                return self.cameras
            except Exception as exc:
                logger.error("Failed to parse %s: %s. Using default configs.", self.filepath, exc)

        # Fallback to default
        self.cameras = json.loads(json.dumps(DEFAULT_CAMERAS))
        self.save()
        return self.cameras

    def save(self) -> bool:
        """Save active configurations to disk."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.cameras, f, indent=2, ensure_ascii=False)
            logger.info("Saved camera configurations to %s", self.filepath)
            return True
        except Exception as exc:
            logger.error("Failed to save camera configurations: %s", exc)
            return False

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        return self.cameras.get(camera_id)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        return self.cameras

    def update_camera(self, camera_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields for a camera and persist changes."""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = {"id": camera_id}

        target = self.cameras[camera_id]
        for key, value in new_data.items():
            if key in ["latitude", "longitude", "azimuth_deg", "fov_deg"]:
                try:
                    target[key] = float(value)
                except (ValueError, TypeError):
                    pass
            elif key in ["port", "fps"]:
                try:
                    target[key] = int(value)
                except (ValueError, TypeError):
                    pass
            elif key == "enabled":
                target[key] = bool(value)
            else:
                target[key] = value

        self.save()
        return target

    def build_stream_url(self, camera_id: str) -> str:
        """
        Construct standard RTSP/HTTP URL based on configured parameters.
        Example: rtsp://user:pass@192.168.1.100:554/stream1
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return ""

        custom = (cam.get("custom_url") or "").strip()
        if custom:
            return custom

        mode = cam.get("connection_mode", "ip_rtsp")
        ip = (cam.get("ip") or "").strip()
        port = cam.get("port", 554)
        user = (cam.get("username") or "").strip()
        pwd = (cam.get("password") or "").strip()
        path = (cam.get("rtsp_path") or "").strip()
        if path and not path.startswith("/"):
            path = "/" + path

        if mode == "usb":
            # USB device index
            try:
                return int(ip)
            except ValueError:
                return 0

        auth = ""
        if user:
            auth = f"{user}:{pwd}@" if pwd else f"{user}@"

        if mode == "ip_http":
            return f"http://{auth}{ip}:{port}{path}"
        else:
            # Default to RTSP
            return f"rtsp://{auth}{ip}:{port}{path}"
