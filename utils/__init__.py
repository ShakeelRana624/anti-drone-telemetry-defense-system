"""
Utilities Subsystem Package
===========================
Provides mathematical, geometric, and system monitoring utilities:
  • BearingCalculator: Computes camera boresight relative angles and True Compass Bearings (0-360°).
  • DistanceGPSEstimator: Computes optical pinhole slant/ground distances and WGS-84 Vincenty geodesic GPS projection.
  • SystemHealthMonitor: Background hardware resource telemetry (CPU, RAM, GPU, Disk).
"""

from utils.bearing_calculator import BearingCalculator
from utils.distance_gps_estimator import DistanceGPSEstimator, DRONE_REFERENCE_SIZES
from utils.system_health import SystemHealthMonitor

__all__ = [
    "BearingCalculator",
    "DistanceGPSEstimator",
    "DRONE_REFERENCE_SIZES",
    "SystemHealthMonitor",
]
