"""
Camera Integration & Telemetry Subsystem Package
================================================
Provides enterprise IP camera ingestion, configuration persistence, RTSP/HTTP streaming,
and coordinate tracking:
  • CameraConfigManager: Reads/writes camera JSON configs (RTSP URLs, GPS coordinates, azimuth).
  • IPCameraStream: Asynchronous threaded frame capture with auto-reconnection and FPS throttling.
  • probe_ip_camera: Network connectivity and RTSP/HTTP feed probe helper.
  • IPCameraManager: High-level dual-camera manager managing primary/secondary video feeds.
"""

from camera_int.camera_config import CameraConfigManager
from camera_int.ip_stream import IPCameraStream, probe_ip_camera
from camera_int.ip_manager import IPCameraManager, get_ip_camera_manager

__all__ = [
    "CameraConfigManager",
    "IPCameraStream",
    "probe_ip_camera",
    "IPCameraManager",
    "get_ip_camera_manager",
]
