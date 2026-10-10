"""
bearing_calculator.py — Drone Bearing & Elevation Estimator
=============================================================
Estimates horizontal bearing (azimuth) and vertical elevation angle
to a detected drone based on its pixel position within the frame
and the camera's configured field of view.

Bearing convention:
    0°   = straight ahead (frame centre)
  -90°   = hard left edge
  +90°   = hard right edge

Elevation convention:
  +ve  = above horizon (drone climbing)
  -ve  = below horizon
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class BearingCalculator:
    """
    Computes relative bearing, vertical elevation, and real-world True Compass Bearing (0°-360°).

    Config keys (under 'bearing'):
        camera_fov_horizontal (float) : horizontal FOV in degrees [90.0]
        camera_fov_vertical   (float) : vertical FOV in degrees   [60.0]
        camera_heading_deg    (float) : camera compass heading [0.0 = North, 90.0 = East, 180.0 = South]
        frame_width  (int)            : frame width in pixels      [1280]
        frame_height (int)            : frame height in pixels     [720]
    """

    CARDINAL_POINTS_16 = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]

    CARDINAL_NAMES = {
        "N": "North", "NNE": "North-North-East", "NE": "North-East", "ENE": "East-North-East",
        "E": "East", "ESE": "East-South-East", "SE": "South-East", "SSE": "South-South-East",
        "S": "South", "SSW": "South-South-West", "SW": "South-West", "WSW": "West-South-West",
        "W": "West", "WNW": "West-North-West", "NW": "North-West", "NNW": "North-North-West"
    }

    def __init__(self, config: dict) -> None:
        self._fov_h: float = float(config.get("camera_fov_horizontal", 90.0))
        self._fov_v: float = float(config.get("camera_fov_vertical",   60.0))
        # Camera physical heading in real world (0° = North, 90° = East, 180° = South, 270° = West)
        self._camera_heading: float = float(config.get("camera_heading_deg", config.get("azimuth_deg", 0.0)))
        self._fw:    int   = int(config.get("frame_width",  1280))
        self._fh:    int   = int(config.get("frame_height",  720))

    # ── Configuration Getters & Setters ──────────────────────────

    @property
    def camera_heading(self) -> float:
        """Get currently configured camera compass heading (0° - 360°)."""
        return self._camera_heading

    @camera_heading.setter
    def camera_heading(self, heading_deg: float) -> None:
        """Set camera compass heading (0° = North, 90° = East, 180° = South, 270° = West)."""
        self._camera_heading = float(heading_deg) % 360.0

    def set_camera_heading(self, heading_deg: float) -> None:
        """Set camera compass heading (0° - 360°)."""
        self.camera_heading = heading_deg

    @classmethod
    def degrees_to_cardinal(cls, heading_deg: float) -> str:
        """Convert a 0°-360° compass heading to 16-point cardinal abbreviation (e.g. 'N', 'NE', 'ESE')."""
        norm_deg = float(heading_deg) % 360.0
        idx = int(round(norm_deg / 22.5)) % 16
        return cls.CARDINAL_POINTS_16[idx]

    @classmethod
    def get_cardinal_full_name(cls, cardinal_abbr: str) -> str:
        """Get full descriptive name (e.g. 'ESE' -> 'East-South-East')."""
        return cls.CARDINAL_NAMES.get(cardinal_abbr, cardinal_abbr)

    # ── Public API ────────────────────────────────────────────────

    def estimate(
        self,
        detections: List,
        camera_id: str = "primary",
    ) -> float:
        """
        Horizontal bearing to the primary (largest) confirmed drone relative to boresight.
        Returns bearing in degrees [-fov/2 to +fov/2].
        """
        if not detections:
            return 0.0
        primary   = max(detections, key=lambda d: d.area)
        cx, _     = primary.center
        bearing   = round((cx - 0.5) * self._fov_h, 2)
        logger.debug("Bearing [%s]: %.2f°  (cx=%.4f)", camera_id, bearing, cx)
        return bearing

    def estimate_elevation(self, detections: List) -> float:
        """
        Vertical elevation angle to the primary drone.
        Positive = above horizon (sky), Negative = below horizon.
        """
        if not detections:
            return 0.0
        primary   = max(detections, key=lambda d: d.area)
        _, cy     = primary.center
        elevation = round((0.5 - cy) * self._fov_v, 2)
        return elevation

    def full_estimate(self, detections: List, camera_id: str = "primary") -> dict:
        """Return relative bearing, elevation, and true compass heading."""
        rel_bearing = self.estimate(detections, camera_id)
        elevation   = self.estimate_elevation(detections)
        true_bearing = round((self._camera_heading + rel_bearing) % 360.0, 2)
        cardinal    = self.degrees_to_cardinal(true_bearing)
        return {
            "bearing_deg":       rel_bearing,
            "elevation_deg":     elevation,
            "camera_heading_deg": self._camera_heading,
            "true_bearing_deg":  true_bearing,
            "cardinal":          cardinal,
            "camera_id":         camera_id,
        }

    def calculate_for_detection(
        self,
        det: Any,
        camera_heading: Optional[float] = None
    ) -> dict:
        """
        Calculate Step 1 True Compass Angle & Telemetry for a target detection:
          1. Relative Azimuth: (-fov/2 to +fov/2) relative to camera lens center
          2. Vertical Elevation: (+sky / -ground)
          3. Total LOS Angle: direct angular divergence from boresight
          4. True Compass Bearing: (Camera Heading + Relative Azimuth) mod 360°
          5. Cardinal Direction: 16-point compass sector (N, NE, E, ESE, S, etc.)
        """
        cx, cy = getattr(det, "center", (0.5, 0.5))

        # 1. Relative horizontal angle from camera optical center
        rel_azimuth = round((cx - 0.5) * self._fov_h, 1)

        # 2. Vertical elevation angle
        elevation = round((0.5 - cy) * self._fov_v, 1)

        # 3. Compound Line-of-sight angular divergence
        total_angle = round((rel_azimuth ** 2 + elevation ** 2) ** 0.5, 1)

        # 4. True Compass Bearing (0° - 360°)
        cam_hdg = self._camera_heading if camera_heading is None else float(camera_heading)
        true_bearing = round((cam_hdg + rel_azimuth) % 360.0, 1)

        # 5. Cardinal Direction
        cardinal = self.degrees_to_cardinal(true_bearing)
        cardinal_full = self.get_cardinal_full_name(cardinal)

        horiz_txt = f"+{rel_azimuth} deg" if rel_azimuth > 0 else f"{rel_azimuth} deg"
        vert_txt  = f"+{elevation} deg" if elevation > 0 else f"{elevation} deg"

        return {
            "azimuth_deg": rel_azimuth,                  # Relative offset
            "rel_azimuth_deg": rel_azimuth,              # Explicit alias
            "elevation_deg": elevation,
            "total_angle_deg": total_angle,
            "camera_heading_deg": cam_hdg,              # Camera direction
            "true_bearing_deg": true_bearing,            # Real-world 0°-360° Compass Bearing
            "cardinal_direction": cardinal,             # e.g. 'ESE'
            "cardinal_full_name": cardinal_full,        # e.g. 'East-South-East'
            "azimuth_str": f"Az: {horiz_txt}",
            "elevation_str": f"El: {vert_txt}",
            "true_bearing_str": f"BRG: {true_bearing:.1f} deg ({cardinal})",
            "angle_tag": f"BRG {true_bearing:.1f} deg {cardinal} | Az {horiz_txt} El {vert_txt} (LOS {total_angle} deg)",
        }

    def calculate_target_angles(
        self,
        detections: List,
        camera_heading: Optional[float] = None
    ) -> List[dict]:
        """Compute True Compass metrics for all detections in the frame."""
        return [self.calculate_for_detection(d, camera_heading) for d in detections]

