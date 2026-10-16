"""
ip_manager.py — Central IP Camera System Manager
=================================================
Manages all active IP camera streams, dynamic configuration updates,
connection testing, and GPS target calculation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from camera_int.camera_config import CameraConfigManager
from camera_int.ip_stream import IPCameraStream, probe_ip_camera

logger = logging.getLogger(__name__)

_instance: Optional[IPCameraManager] = None


class IPCameraManager:
    """Central controller for dual IP cameras."""

    def __init__(self) -> None:
        """
        Initialize the central IPCameraManager instance and load active streams.
        """
        self.config_manager = CameraConfigManager()
        self._streams: Dict[str, IPCameraStream] = {}
        self._running = False
        self._init_streams()

    def _init_streams(self) -> None:
        """Instantiate IPCameraStream workers for all enabled camera configurations."""
        cameras = self.config_manager.get_all()
        for cid, cfg in cameras.items():
            if cfg.get("enabled", True):
                self._streams[cid] = IPCameraStream(cfg)

    def start(self) -> None:
        """Start all configured camera streams."""
        self._running = True
        for cid, stream in self._streams.items():
            try:
                stream.start()
            except Exception as exc:
                logger.error("Failed to start stream [%s]: %s", cid, exc)
        logger.info("IPCameraManager started %d stream(s).", len(self._streams))

    def stop(self) -> None:
        """Stop all streams gracefully."""
        self._running = False
        for cid, stream in self._streams.items():
            stream.stop()
        logger.info("IPCameraManager stopped.")

    def get_frame(self, camera_id: str) -> np.ndarray:
        """
        Get latest frame for specified camera; fallbacks to synthetic frame if stream does not exist.

        Args:
            camera_id: Camera stream identifier ('primary', 'secondary').

        Returns:
            numpy.ndarray: BGR image frame.
        """
        stream = self._streams.get(camera_id)
        if stream:
            return stream.read()

        # If camera not initialized yet, create temporary stream
        cam_cfg = self.config_manager.get_camera(camera_id) or {"id": camera_id}
        temp_stream = IPCameraStream(cam_cfg)
        self._streams[camera_id] = temp_stream
        temp_stream.start()
        return temp_stream.read()

    def get_camera_statuses(self) -> Dict[str, Any]:
        """
        Return full camera metadata, connection states, FPS, and latencies for Web UI dashboard.

        Returns:
            Dict[str, Any]: Nested dictionary keyed by camera ID containing configuration and live health.
        """
        configs = self.config_manager.get_all()
        statuses = {}
        for cid, cfg in configs.items():
            stream = self._streams.get(cid)
            healthy = stream.healthy if stream else False
            fps = round(stream.current_fps, 1) if stream else 0.0
            latency = stream.latency_ms if stream else 0.0

            statuses[cid] = {
                **cfg,
                "is_online": healthy,
                "current_fps": fps,
                "latency_ms": latency,
                "status_label": "CONNECTED" if healthy else "SIMULATION / STANDBY",
            }
        return statuses

    def update_camera(self, camera_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update camera configuration, persist changes to disk, and hot-reload its active streaming worker.

        Args:
            camera_id: Unique camera profile key.
            new_data: Dictionary of modified parameters.

        Returns:
            Dict[str, Any]: Updated configuration dictionary.
        """
        updated = self.config_manager.update_camera(camera_id, new_data)

        if camera_id in self._streams:
            self._streams[camera_id].update_config(updated)
        else:
            new_stream = IPCameraStream(updated)
            self._streams[camera_id] = new_stream
            if self._running:
                new_stream.start()

        logger.info("Updated camera [%s] config: IP=%s, City=%s, Lat=%s, Lon=%s",
                    camera_id, updated.get("ip"), updated.get("city"),
                    updated.get("latitude"), updated.get("longitude"))
        return updated

    def test_connection(self, data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Probe an IP camera endpoint before saving or applying configuration changes.

        Args:
            data: Dictionary containing 'ip' and 'port'.

        Returns:
            Tuple[bool, str, float]: Reachability boolean, status message, and roundtrip response latency in ms.
        """
        ip = data.get("ip", "").strip()
        port = int(data.get("port", 554))
        return probe_ip_camera(ip, port, timeout_sec=2.5)

    def calculate_drone_gps(
        self,
        camera_id: str,
        bearing_deg: float,
        distance_meters: float
    ) -> Tuple[float, float]:
        """
        Spherical forward geodesic projection:
        Given camera base GPS coordinates (lat, lon) and target bearing & distance,
        computes the estimated latitude and longitude of the aerial drone.

        Args:
            camera_id: Identifier of reference camera base.
            bearing_deg: Compass bearing angle in degrees from true North.
            distance_meters: Ground distance from camera to target in meters.

        Returns:
            Tuple[float, float]: (drone_latitude, drone_longitude) rounded to 6 decimal places.
        """
        cam = self.config_manager.get_camera(camera_id)
        if not cam:
            return 33.6844, 73.0479

        lat1 = math.radians(float(cam.get("latitude", 33.6844)))
        lon1 = math.radians(float(cam.get("longitude", 73.0479)))
        brng = math.radians(bearing_deg)

        R = 6371000.0  # Earth's radius in meters
        d_div_r = distance_meters / R

        lat2 = math.asin(
            math.sin(lat1) * math.cos(d_div_r) +
            math.cos(lat1) * math.sin(d_div_r) * math.cos(brng)
        )
        lon2 = lon1 + math.atan2(
            math.sin(brng) * math.sin(d_div_r) * math.cos(lat1),
            math.cos(d_div_r) - math.sin(lat1) * math.sin(lat2)
        )

        return round(math.degrees(lat2), 6), round(math.degrees(lon2), 6)


def get_ip_camera_manager() -> IPCameraManager:
    """
    Retrieve or initialize the global IPCameraManager singleton instance and begin streaming.

    Returns:
        IPCameraManager: Shared singleton manager.
    """
    global _instance
    if _instance is None:
        _instance = IPCameraManager()
        _instance.start()
    return _instance
