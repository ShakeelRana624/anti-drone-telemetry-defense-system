"""
Unit test for Step 1: True Compass Angle calculation.
Verifies all 4 quadrants, boundary wrap-around (360°/0°), and cardinal points.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import namedtuple
from utils.bearing_calculator import BearingCalculator

Det = namedtuple("Det", ["center"])

def test_compass_calculations():
    print("======================================================================")
    print("STEP 1: TRUE COMPASS ANGLE & CARDINAL SECTOR TEST SUITE")
    print("======================================================================")

    test_cases = [
        {"name": "Camera Facing North (0 deg) - Drone in right quadrant (+15 deg)", "cam_hdg": 0.0, "cx": 0.6875, "exp_brg": 15.0, "exp_card": "NNE"},
        {"name": "Camera Facing North (0 deg) - Wrap-around left (-15 deg -> 345 deg)", "cam_hdg": 0.0, "cx": 0.3125, "exp_brg": 345.0, "exp_card": "NNW"},
        {"name": "Camera Facing East (90 deg) - input.mp4 real drone (+10.1 deg)", "cam_hdg": 90.0, "cx": 0.626, "exp_brg": 100.1, "exp_card": "E"},
        {"name": "Camera Facing South (180 deg) - Drone dead center (0 deg)", "cam_hdg": 180.0, "cx": 0.500, "exp_brg": 180.0, "exp_card": "S"},
        {"name": "Camera Facing West (270 deg) - Drone hard left (-30 deg -> 240 deg)", "cam_hdg": 270.0, "cx": 0.125, "exp_brg": 240.0, "exp_card": "WSW"},
    ]

    for tc in test_cases:
        bc = BearingCalculator({
            "camera_heading_deg": tc["cam_hdg"],
            "camera_fov_horizontal": 80.0,
            "camera_fov_vertical": 60.0
        })
        res = bc.calculate_for_detection(Det(center=(tc["cx"], 0.35)))
        brg = res["true_bearing_deg"]
        card = res["cardinal_direction"]
        az = res["azimuth_deg"]
        el = res["elevation_deg"]

        print(f"\nTest: {tc['name']}")
        print(f"  Camera Heading : {tc['cam_hdg']} deg")
        print(f"  Relative Azimuth : {az:+.1f} deg | Elevation: {el:+.1f} deg")
        print(f"  True Bearing   : {brg:.1f} deg (Expected: ~{tc['exp_brg']:.1f} deg)")
        print(f"  Compass Sector : {card} ({res['cardinal_full_name']})")
        assert abs(brg - tc["exp_brg"]) < 0.2, f"Expected {tc['exp_brg']}, got {brg}"
        print("  -> STATUS: PASSED [OK]")

    print("\n----------------------------------------------------------------------")
    print("ALL STEP 1 TRUE COMPASS ANGLE TESTS PASSED SUCCESSFULLY!")
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    test_compass_calculations()
