"""
ip_manager.py — Central IP Camera System Manager
=================================================
Manages all active IP camera streams, dynamic configuration updates,
connection testing, and GPS target calculation.
"""

from __future__ import annotations

import logging
import math
import sys
import time
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

            ptz_data = None
            try:
                from camera_int.ptz_controller import get_ptz_controller
                ptz_ctrl = get_ptz_controller(cid, cfg)
                ptz_data = ptz_ctrl.get_status()
            except Exception:
                pass

            statuses[cid] = {
                **cfg,
                "is_online": healthy,
                "current_fps": fps,
                "latency_ms": latency,
                "status_label": "CONNECTED" if healthy else "SIMULATION / STANDBY",
                "ptz": ptz_data,
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

        # Update PTZ Controller if available
        try:
            from camera_int.ptz_controller import get_ptz_controller
            ptz = get_ptz_controller(camera_id, updated)
        except Exception as e:
            logger.debug("Could not update PTZ controller: %s", e)

        # Update Live Inference Engine if active
        try:
            from detection.live_inference import get_live_inference_engine
            engine = get_live_inference_engine()
            azimuth = updated.get("azimuth_deg")
            lat = updated.get("latitude")
            lon = updated.get("longitude")
            fov = updated.get("fov_deg", 90.0)
            engine.update_camera_params(heading=azimuth, lat=lat, lon=lon, fov_h=fov)
        except Exception as e:
            logger.debug("Could not update live inference engine: %s", e)

        # Sync to config/config.yaml
        try:
            self._sync_to_yaml_config(camera_id, updated)
        except Exception as e:
            logger.debug("Could not sync to config.yaml: %s", e)

        logger.info("Updated camera [%s] config: Mode=%s, Source=%s, Azimuth=%.1f deg, Lat=%s, Lon=%s",
                    camera_id, updated.get("connection_mode"), updated.get("ip"),
                    float(updated.get("azimuth_deg", 0.0)),
                    updated.get("latitude"), updated.get("longitude"))
        return updated

    def _sync_to_yaml_config(self, camera_id: str, updated: Dict[str, Any]) -> None:
        """Keep config/config.yaml in sync with camera updates."""
        try:
            import yaml
            from pathlib import Path
            cfg_file = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
            if not cfg_file.exists():
                return
            with open(cfg_file, "r", encoding="utf-8") as fh:
                root_cfg = yaml.safe_load(fh) or {}

            cams = root_cfg.setdefault("cameras", {})
            target_cam = cams.setdefault(camera_id, {})
            mode = updated.get("connection_mode", "ip_rtsp")
            if mode in ("usb", "webcam"):
                try:
                    target_cam["source"] = int(updated.get("ip", 0))
                except (ValueError, TypeError):
                    target_cam["source"] = 0
            else:
                ip = updated.get("ip", "")
                port = updated.get("port", 554)
                path = updated.get("rtsp_path", "/stream1")
                user = updated.get("username", "")
                pwd = updated.get("password", "")
                auth = f"{user}:{pwd}@" if user and pwd else (f"{user}@" if user else "")
                target_cam["source"] = f"rtsp://{auth}{ip}:{port}{path}"

            target_cam["label"] = updated.get("name", target_cam.get("label", "Primary Camera"))
            target_cam["azimuth_deg"] = float(updated.get("azimuth_deg", 90.0))
            target_cam["camera_heading_deg"] = float(updated.get("azimuth_deg", 90.0))

            bearing_cfg = root_cfg.setdefault("bearing", {})
            bearing_cfg["camera_heading_deg"] = float(updated.get("azimuth_deg", 90.0))
            if "fov_deg" in updated:
                bearing_cfg["camera_fov_horizontal"] = float(updated["fov_deg"])

            with open(cfg_file, "w", encoding="utf-8") as fh:
                yaml.safe_dump(root_cfg, fh, sort_keys=False)
            logger.info("Synchronized camera [%s] config into config/config.yaml", camera_id)
        except Exception as exc:
            logger.warning("Failed to sync config.yaml: %s", exc)

    def test_connection(self, data: Dict[str, Any]) -> Tuple[bool, str, float]:
        """
        Probe an IP or USB camera endpoint before saving or applying configuration changes.

        Args:
            data: Dictionary containing 'ip', 'port', and 'connection_mode'.

        Returns:
            Tuple[bool, str, float]: Reachability boolean, status message, and roundtrip response latency in ms.
        """
        mode = str(data.get("connection_mode", "")).lower()
        ip = str(data.get("ip", "")).strip()
        port = int(data.get("port", 554))

        # Check USB camera
        if mode in ("usb", "webcam") or (ip.isdigit() and int(ip) < 10 and mode != "ip_rtsp"):
            dev_idx = int(ip) if ip.isdigit() else 0
            cam_id = data.get("id", "primary")
            active_st = self._streams.get(cam_id)
            if active_st and active_st.source == dev_idx:
                fps = getattr(active_st, "current_fps", 0.0)
                lat = getattr(active_st, "latency_ms", 12.0)
                return True, f"USB Camera (Device Index {dev_idx}) is currently active and assigned to this stream ({fps:.1f} FPS)!", lat

            t0 = time.perf_counter()
            try:
                import cv2
                cap = cv2.VideoCapture(dev_idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        return True, f"USB Camera (Device Index {dev_idx}) verified & active! Delivering {w}x{h} px video stream.", elapsed_ms
                    return True, f"USB Camera (Device Index {dev_idx}) hardware device opened successfully.", elapsed_ms
                else:
                    return False, f"USB Device Index {dev_idx} could not be opened. Ensure camera is plugged into USB port.", 0.0
            except Exception as e:
                return False, f"USB camera probe error: {e}", 0.0

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
