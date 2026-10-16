"""
test_step2_complete.py — Comprehensive Test Suite for Distance Estimation & GPS Projection
==========================================================================================
Verifies:
  1. Optical focal length calculation from frame dimensions and horizontal FOV.
  2. Monocular pinhole distance estimation at known drone size.
  3. Elevation decomposition into horizontal ground distance and vertical altitude.
  4. Geodetic WGS-84 Vincenty direct calculation (North and East bearings).
  5. Exponential Moving Average (EMA) distance temporal smoothing.
  6. Confidence band classification based on pixel bounding box resolution.
  7. Tactical HUD string generation format.
"""

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.distance_gps_estimator import DistanceGPSEstimator, _vincenty_direct

PASS = "[PASSED]"
FAIL = "[FAILED]"


def approx(a: float, b: float, tol: float = 2.0) -> bool:
    """Check if two floats are approximately equal within absolute tolerance."""
    return abs(a - b) <= tol


def approxg(a: float, b: float, tol: float = 0.0002) -> bool:
    """Check if two geographic coordinates are approximately equal within coordinate tolerance."""
    return abs(a - b) <= tol


def test_step2_complete() -> None:
    """
    Execute comprehensive verification of all Step 2 distance and GPS projection components.
    """
    print("="*68)
    print("STEP 2: DISTANCE ESTIMATION + GPS PROJECTION TEST SUITE")
    print("="*68)

    cfg = {
        "camera_fov_horizontal": 80.0,
        "camera_fov_vertical":   60.0,
        "frame_width":  1280,
        "frame_height":  720,
        "drone_type": "dji_phantom",
        "camera_latitude":  33.684400,
        "camera_longitude": 73.047900,
    }
    est = DistanceGPSEstimator(cfg)

    # T1: Focal length
    exp_f = (1280/2.0) / math.tan(math.radians(40.0))
    diff  = abs(est._focal_px - exp_f)
    s = PASS if diff < 0.5 else FAIL
    print(f"\nTest 1 - Focal Length: Expected {exp_f:.2f} px | Got {est._focal_px:.2f} px")
    print(f"  -> {s}")
    assert s == PASS

    # T2: 100 m straight ahead elevation=0
    bbox_w = (0.590 * est._focal_px) / 100.0
    r = est.estimate(bbox_w, bbox_w*0.8, 0.0, 90.0, "t2")
    s = PASS if approx(r["slant_distance_m"], 100.0, 2.0) else FAIL
    print(f"\nTest 2 - Pinhole 100m (el=0): slant={r['slant_distance_m']}m | ground={r['ground_distance_m']}m | alt={r['drone_altitude_m']}m")
    print(f"  -> {s}")
    assert s == PASS

    # T3: 200 m slant, elevation +30 deg
    bbox_w2 = (0.590 * est._focal_px) / 200.0
    est.reset_ema("t3")
    r2 = est.estimate(bbox_w2, bbox_w2, 30.0, 45.0, "t3")
    eg = 200.0 * math.cos(math.radians(30.0))
    ea = 200.0 * math.sin(math.radians(30.0))
    s = PASS if approx(r2["ground_distance_m"], eg, 5.0) and approx(r2["drone_altitude_m"], ea, 5.0) else FAIL
    print(f"\nTest 3 - Elevation Decomp (200m, el=30): ground={r2['ground_distance_m']}m (exp~{eg:.1f}) | alt={r2['drone_altitude_m']}m (exp~{ea:.1f})")
    print(f"  -> {s}")
    assert s == PASS

    # T4: Vincenty North 1000 m
    lt, ln = _vincenty_direct(33.684400, 73.047900, 0.0, 1000.0)
    el = 33.684400 + 1000.0/111320.0
    s = PASS if approxg(lt, el, 0.0003) and approxg(ln, 73.047900, 0.0001) else FAIL
    print(f"\nTest 4 - Vincenty North 1000m: ({lt:.6f},{ln:.6f}) expected (~{el:.6f},~73.047900)")
    print(f"  -> {s}")
    assert s == PASS

    # T5: Vincenty East 500 m
    lt2, ln2 = _vincenty_direct(33.684400, 73.047900, 90.0, 500.0)
    el2 = 73.047900 + 500.0/(111320.0*math.cos(math.radians(33.6844)))
    s = PASS if approxg(lt2, 33.684400, 0.0001) and approxg(ln2, el2, 0.0003) else FAIL
    print(f"\nTest 5 - Vincenty East 500m:  ({lt2:.6f},{ln2:.6f}) expected (~33.684400,~{el2:.6f})")
    print(f"  -> {s}")
    assert s == PASS

    # T6: EMA smoothing - raw jump should be dampened
    est6 = DistanceGPSEstimator({**cfg, "distance_ema_alpha": 0.5})
    r_close = est6.estimate(50.0, 50.0, 0.0, 90.0, "ema")
    raw_far = (0.590 * est6._focal_px) / 10.0
    r_jump  = est6.estimate(10.0, 10.0, 0.0, 90.0, "ema")
    ema_damped = r_jump["slant_distance_m"] < raw_far
    s = PASS if ema_damped else FAIL
    print(f"\nTest 6 - EMA Smoothing: raw_far={raw_far:.1f}m | EMA smoothed={r_jump['slant_distance_m']:.1f}m (should be < raw)")
    print(f"  -> {s}")
    assert s == PASS

    # T7: Confidence banding
    est7 = DistanceGPSEstimator(cfg)
    rH = est7.estimate(80.0, 60.0, 0.0, 90.0, "cH")
    rM = est7.estimate(20.0, 15.0, 0.0, 90.0, "cM")
    rL = est7.estimate(5.0,  4.0,  0.0, 90.0, "cL")
    s = PASS if rH["confidence"]=="HIGH" and rM["confidence"]=="MEDIUM" and rL["confidence"]=="LOW" else FAIL
    print(f"\nTest 7 - Confidence: 80px->{rH['confidence']} | 20px->{rM['confidence']} | 5px->{rL['confidence']}")
    print(f"  -> {s}")
    assert s == PASS

    # T8: Full pipeline HUD strings
    print(f"\nTest 8 - HUD Strings:")
    print(f"  dist_str : {rH['dist_str']}")
    print(f"  grd_str  : {rH['grd_str']}")
    print(f"  gps_str  : {rH['gps_str']}")
    s = PASS if "DST:" in rH["dist_str"] and "GPS" in rH["gps_str"] else FAIL
    print(f"  -> {s}")
    assert s == PASS

    print("\n" + "="*68)
    print("ALL STEP 2 TESTS PASSED SUCCESSFULLY!")
    print("="*68)


if __name__ == "__main__":
    test_step2_complete()
