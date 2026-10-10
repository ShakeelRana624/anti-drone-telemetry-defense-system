"""
distance_gps_estimator.py  --  Step 2: Distance Estimation and GPS Projection
===========================================================================
Implements two-stage pipeline:

  STAGE A - Monocular Distance Estimation (Pinhole Camera Model)
  Uses the fundamental triangle-similarity formula:
      focal_length_px  =  (frame_width_px) / (2 * tan(FOV_H / 2))
      slant_distance   =  (drone_real_width_m * focal_length_px) / bbox_width_px
      ground_distance  =  slant_distance * cos(elevation_deg)
      drone_altitude   =  slant_distance * sin(elevation_deg)

  STAGE B - WGS-84 Vincenty Direct Formula (sub-meter accuracy)
  Given camera GPS (lat, lon), True Compass Bearing (Step 1), and
  ground distance (Stage A), computes drone GPS coordinates.

  WGS-84 constants:
    a  = 6 378 137.000 m  (equatorial radius)
    b  = 6 356 752.314 m  (polar radius)
    f  = 1 / 298.257 223 563  (flattening)

References:
  [1] Vincenty, T. (1975). Survey Review 23(176):88-93.
  [2] Pinhole camera model, Wikipedia.
  [3] WGS-84 standard (NIMA TR8350.2).
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# WGS-84 Ellipsoid Constants
_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.2572235630
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)

# Reference Drone Sizes (motor-to-motor diagonal, metres)
# Diagonal is orientation-invariant unlike width/height
DRONE_REFERENCE_SIZES = {
    "dji_phantom":    0.590,
    "dji_mini":       0.289,
    "dji_mavic":      0.350,
    "generic_small":  0.300,
    "generic_medium": 0.500,
    "generic_large":  0.800,
    "custom":         0.400,
}

_EMA_ALPHA = 0.25  # 0.25 = ~4-frame lag at 25 FPS
_MIN_BBOX_PX = 8


def _vincenty_direct(lat1_deg, lon1_deg, bearing_deg, distance_m):
    """
    Vincenty Direct Formula on WGS-84 ellipsoid.
    Given start point, initial bearing and distance, compute destination GPS.
    Accuracy: sub-millimetre over any distance on Earth.
    """
    if distance_m < 0.01:
        return lat1_deg, lon1_deg

    a, f, b = _WGS84_A, _WGS84_F, _WGS84_B
    lat1   = math.radians(lat1_deg)
    lon1   = math.radians(lon1_deg)
    alpha1 = math.radians(bearing_deg)

    sin_a1 = math.sin(alpha1)
    cos_a1 = math.cos(alpha1)

    tan_U1 = (1.0 - f) * math.tan(lat1)
    cos_U1 = 1.0 / math.sqrt(1.0 + tan_U1 ** 2)
    sin_U1 = tan_U1 * cos_U1
    sigma1 = math.atan2(tan_U1, cos_a1)

    sin_alpha  = cos_U1 * sin_a1
    cos2_alpha = 1.0 - sin_alpha ** 2

    u2     = cos2_alpha * (a**2 - b**2) / b**2
    A_coef = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B_coef = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))

    sigma = distance_m / (b * A_coef)
    for _ in range(100):
        cos2sm  = math.cos(2.0 * sigma1 + sigma)
        sin_sig = math.sin(sigma)
        cos_sig = math.cos(sigma)
        dsig = B_coef * sin_sig * (
            cos2sm + B_coef / 4.0 * (
                cos_sig * (-1.0 + 2.0 * cos2sm**2)
                - B_coef / 6.0 * cos2sm * (-3.0 + 4.0 * sin_sig**2) * (-3.0 + 4.0 * cos2sm**2)
            )
        )
        new_sigma = distance_m / (b * A_coef) + dsig
        if abs(new_sigma - sigma) < 1e-12:
            sigma = new_sigma
            break
        sigma = new_sigma

    cos2sm  = math.cos(2.0 * sigma1 + sigma)
    sin_sig = math.sin(sigma)
    cos_sig = math.cos(sigma)

    lat2 = math.atan2(
        sin_U1 * cos_sig + cos_U1 * sin_sig * cos_a1,
        (1.0 - f) * math.sqrt(sin_alpha**2 + (sin_U1 * sin_sig - cos_U1 * cos_sig * cos_a1)**2)
    )
    lam = math.atan2(sin_sig * sin_a1, cos_U1 * cos_sig - sin_U1 * sin_sig * cos_a1)
    C   = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha))
    L   = lam - (1.0 - C) * f * sin_alpha * (
        sigma + C * sin_sig * (cos2sm + C * cos_sig * (-1.0 + 2.0 * cos2sm**2))
    )
    lon2 = (lon1 + L + 3.0 * math.pi) % (2.0 * math.pi) - math.pi
    return round(math.degrees(lat2), 7), round(math.degrees(lon2), 7)


class DistanceGPSEstimator:
    """
    Step 2: Monocular Distance Estimation + WGS-84 Vincenty GPS Projection.

    Config keys:
        camera_fov_horizontal  (float): horizontal FOV in degrees  [80.0]
        camera_fov_vertical    (float): vertical FOV in degrees    [60.0]
        frame_width            (int)  : frame width in pixels      [1280]
        frame_height           (int)  : frame height in pixels     [ 720]
        drone_type             (str)  : key from DRONE_REFERENCE_SIZES
        drone_real_width_m     (float): override real drone width (m)
        camera_latitude        (float): camera GPS latitude        [0.0]
        camera_longitude       (float): camera GPS longitude       [0.0]
        distance_ema_alpha     (float): EMA smoothing factor       [0.25]
    """

    def __init__(self, config: dict) -> None:
        self._fov_h = float(config.get("camera_fov_horizontal", 80.0))
        self._fov_v = float(config.get("camera_fov_vertical",   60.0))
        self._fw    = int(config.get("frame_width",  1280))
        self._fh    = int(config.get("frame_height",  720))

        drone_type = config.get("drone_type", "generic_medium")
        default_sz = DRONE_REFERENCE_SIZES.get(drone_type, 0.5)
        self._drone_width_m = float(config.get("drone_real_width_m", default_sz))

        self._cam_lat = float(config.get("camera_latitude",  0.0))
        self._cam_lon = float(config.get("camera_longitude", 0.0))

        self._ema_alpha = float(config.get("distance_ema_alpha", _EMA_ALPHA))
        self._ema_state: Dict[str, float] = {}

        self._focal_px = self._compute_focal_px()
        logger.info(
            "DistanceGPSEstimator | FOV=%.1f deg | focal=%.1f px | "
            "drone_ref=%.3f m | cam=(%.6f, %.6f)",
            self._fov_h, self._focal_px, self._drone_width_m,
            self._cam_lat, self._cam_lon
        )

    def set_camera_gps(self, lat: float, lon: float) -> None:
        self._cam_lat = float(lat)
        self._cam_lon = float(lon)

    def set_drone_reference_width(self, width_m: float) -> None:
        self._drone_width_m = float(width_m)

    def reset_ema(self, track_id=None) -> None:
        if track_id is None:
            self._ema_state.clear()
        else:
            self._ema_state.pop(str(track_id), None)

    def estimate(
        self,
        bbox_width_px: float,
        bbox_height_px: float,
        elevation_deg: float,
        true_bearing_deg: float,
        track_id=None,
    ) -> Dict[str, Any]:
        """
        Full Step 2 pipeline for one detection.

        Returns dict with:
            slant_distance_m  - straight 3D camera-to-drone distance (m)
            ground_distance_m - horizontal ground projection (m)
            drone_altitude_m  - estimated drone height above camera (m)
            drone_lat         - drone latitude  (WGS-84, 7 decimal places)
            drone_lon         - drone longitude (WGS-84, 7 decimal places)
            confidence        - HIGH | MEDIUM | LOW
            dist_str          - HUD-ready distance string
            grd_str           - HUD-ready ground+alt string
            gps_str           - HUD-ready GPS string
        """
        use_px     = max(bbox_width_px, bbox_height_px)
        low_conf   = use_px < _MIN_BBOX_PX

        # Stage A: Pinhole distance
        slant_raw  = (self._drone_width_m * self._focal_px / use_px) if use_px > 0 else 9999.0
        slant_raw  = max(1.0, slant_raw)

        # EMA temporal smoothing per track
        tid = str(track_id) if track_id is not None else "__global__"
        if tid in self._ema_state:
            slant_s = self._ema_alpha * slant_raw + (1.0 - self._ema_alpha) * self._ema_state[tid]
        else:
            slant_s = slant_raw
        self._ema_state[tid] = slant_s

        # Decompose into ground distance + altitude
        el_rad      = math.radians(elevation_deg)
        ground_dist = round(slant_s * math.cos(el_rad), 2)
        altitude    = round(slant_s * math.sin(el_rad), 2)
        slant_m     = round(slant_s, 2)

        # Stage B: Vincenty GPS
        if self._cam_lat == 0.0 and self._cam_lon == 0.0:
            drone_lat, drone_lon, gps_valid = 0.0, 0.0, False
        else:
            drone_lat, drone_lon = _vincenty_direct(
                self._cam_lat, self._cam_lon, true_bearing_deg, ground_dist
            )
            gps_valid = True

        # Confidence band
        if low_conf or use_px < 15:
            conf = "LOW"
        elif use_px < 40:
            conf = "MEDIUM"
        else:
            conf = "HIGH"

        return {
            "slant_distance_m":  slant_m,
            "ground_distance_m": ground_dist,
            "drone_altitude_m":  altitude,
            "drone_lat":         drone_lat,
            "drone_lon":         drone_lon,
            "gps_valid":         gps_valid,
            "confidence":        conf,
            "bbox_width_px":     bbox_width_px,
            "bbox_height_px":    bbox_height_px,
            "focal_px":          round(self._focal_px, 1),
            "ema_smoothed":      tid in self._ema_state,
            "dist_str":  f"DST: {slant_m:.0f}m [{conf[0]}]",
            "grd_str":   f"GND {ground_dist:.0f}m | ALT {altitude:.0f}m",
            "gps_str":   f"GPS {drone_lat:.6f},{drone_lon:.6f}" if gps_valid else "GPS: SET CAM LAT/LON",
        }

    def estimate_from_detection(self, det: Any, bearing_data: Dict[str, Any], track_id=None) -> Dict[str, Any]:
        """
        Convenience wrapper: accepts Detection object + Step 1 bearing_data dict.
        det.bbox must be (x1,y1,x2,y2) normalised [0,1].
        """
        x1, y1, x2, y2 = det.bbox
        return self.estimate(
            bbox_width_px  = abs(x2 - x1) * self._fw,
            bbox_height_px = abs(y2 - y1) * self._fh,
            elevation_deg  = bearing_data.get("elevation_deg",   0.0),
            true_bearing_deg = bearing_data.get("true_bearing_deg", 0.0),
            track_id       = track_id,
        )

    def _compute_focal_px(self) -> float:
        """f_px = (W_px/2) / tan(FOV_H/2)"""
        return (self._fw / 2.0) / math.tan(math.radians(self._fov_h / 2.0))
