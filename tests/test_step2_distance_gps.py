"""
test_step2_distance_gps.py -- Step 2: Distance Estimation + GPS Unit Tests
===========================================================================
Tests:
  1. Focal length computation
  2. Pinhole distance at known bounding box sizes
  3. Ground vs slant distance decomposition via elevation angle
  4. Vincenty GPS accuracy (compare to known reference coordinates)
  5. EMA smoothing convergence
  6. Confidence banding
"""
import sys, math
sys.path.insert(0, r"d:\drone-detection")

from utils.distance_gps_estimator import DistanceGPSEstimator, _vincenty_direct

PASS = "[PASSED]"
FAIL = "[FAILED]"

def approx(a, b, tol=0.5):
    return abs(a - b) <= tol

def approx_gps(a, b, tol=0.00005):
    return abs(a - b) <= tol


print("="*68)
print("STEP 2: DISTANCE ESTIMATION + GPS PROJECTION TEST SUITE")
print("="*68)

cfg = {
    "camera_fov_horizontal": 80.0,
    "camera_fov_vertical":   60.0,
    "frame_width":  1280,
    "frame_height":  720,
    "drone_type": "dji_phantom",      # 0.590 m diagonal
    "camera_latitude":  33.684400,
    "camera_longitude": 73.047900,
}
est = DistanceGPSEstimator(cfg)

# ── Test 1: Focal Length ──────────────────────────────────────────
# f_px = (1280/2) / tan(40 deg) = 640 / 0.8391 = 762.7
expected_focal = (1280 / 2.0) / math.tan(math.radians(40.0))
diff = abs(est._focal_px - expected_focal)
status = PASS if diff < 0.5 else FAIL
print(f"\nTest 1 - Focal Length Computation")
print(f"  Expected: {expected_focal:.2f} px | Got: {est._focal_px:.2f} px | Diff: {diff:.3f}")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 2: Pinhole Distance (Drone 100 m away, elevation=0) ─────
# If drone is 100m away, bbox_width = (0.590 * f_px) / 100 = ~4.5 px
bbox_w = (0.590 * est._focal_px) / 100.0
result = est.estimate(bbox_w, bbox_w * 0.8, 0.0, 90.0, track_id="t1")
status = PASS if approx(result["slant_distance_m"], 100.0, tol=2.0) else FAIL
print(f"\nTest 2 - Pinhole Distance: Drone at 100 m (elevation=0)")
print(f"  BBox width: {bbox_w:.2f} px | Slant: {result['slant_distance_m']} m | Expected: ~100 m")
print(f"  Ground: {result['ground_distance_m']} m | Alt: {result['drone_altitude_m']} m")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 3: Elevation Decomposition ──────────────────────────────
# Drone 200 m slant, elevation +30 deg
# ground = 200 * cos(30) = 173.2 m, alt = 200 * sin(30) = 100.0 m
bbox_w2 = (0.590 * est._focal_px) / 200.0
est.reset_ema("t2")
result2 = est.estimate(bbox_w2, bbox_w2, 30.0, 45.0, track_id="t2")
exp_gnd = 200.0 * math.cos(math.radians(30.0))
exp_alt = 200.0 * math.sin(math.radians(30.0))
ok_gnd  = approx(result2["ground_distance_m"], exp_gnd, tol=5.0)
ok_alt  = approx(result2["drone_altitude_m"],  exp_alt, tol=5.0)
status  = PASS if (ok_gnd and ok_alt) else FAIL
print(f"\nTest 3 - Elevation Decomposition (slant~200m, el=+30 deg)")
print(f"  Ground: {result2['ground_distance_m']} m (exp ~{exp_gnd:.1f} m) | Alt: {result2['drone_altitude_m']} m (exp ~{exp_alt:.1f} m)")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 4: Vincenty GPS Accuracy ────────────────────────────────
# Known reference: Islamabad (33.6844, 73.0479)
# Move 1000 m due North (bearing=0 deg) => lat increases ~0.009 deg
# Verified using: https://www.movable-type.co.uk/scripts/latlong-vincenty.html
lat2, lon2 = _vincenty_direct(33.684400, 73.047900, 0.0, 1000.0)
exp_lat = 33.684400 + (1000.0 / 111320.0)   # ~33.6934
ok_lat  = approx_gps(lat2, exp_lat, tol=0.0002)
ok_lon  = approx_gps(lon2, 73.047900, tol=0.0001)   # longitude shouldn't change for North bearing
status  = PASS if (ok_lat and ok_lon) else FAIL
print(f"\nTest 4 - Vincenty GPS: Move 1000 m North from Islamabad")
print(f"  Got:      ({lat2:.6f}, {lon2:.6f})")
print(f"  Expected: (~{exp_lat:.6f}, ~73.047900)")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 5: Vincenty GPS - East bearing ──────────────────────────
# Move 500 m due East (bearing=90 deg) => lon increases
lat3, lon3 = _vincenty_direct(33.684400, 73.047900, 90.0, 500.0)
exp_lon3 = 73.047900 + (500.0 / (111320.0 * math.cos(math.radians(33.684400))))
ok_lat3  = approx_gps(lat3, 33.684400, tol=0.0001)
ok_lon3  = approx_gps(lon3, exp_lon3, tol=0.0002)
status   = PASS if (ok_lat3 and ok_lon3) else FAIL
print(f"\nTest 5 - Vincenty GPS: Move 500 m East from Islamabad")
print(f"  Got:      ({lat3:.6f}, {lon3:.6f})")
print(f"  Expected: (~{33.684400:.6f}, ~{exp_lon3:.6f})")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 6: EMA Smoothing ────────────────────────────────────────
est2 = DistanceGPSEstimator({**cfg, "distance_ema_alpha": 0.5})
est2.reset_ema("ema_test")
r1 = est2.estimate(30.0, 30.0, 0.0, 90.0, track_id="ema_test")
r2 = est2.estimate(10.0, 10.0, 0.0, 90.0, track_id="ema_test")
r3 = est2.estimate(10.0, 10.0, 0.0, 90.0, track_id="ema_test")
# After 3 frames converging toward ~shorter distance, EMA should be smoothing
smoothed_is_between = r1["slant_distance_m"] > r2["slant_distance_m"] or r2["slant_distance_m"] > r3["slant_distance_m"]
status = PASS if smoothed_is_between else FAIL
print(f"\nTest 6 - EMA Temporal Smoothing")
print(f"  Frame 1 dist: {r1['slant_distance_m']} m | Frame 2: {r2['slant_distance_m']} m | Frame 3: {r3['slant_distance_m']} m")
print(f"  -> STATUS: {status}")
assert status == PASS

# ── Test 7: Confidence Banding ───────────────────────────────────
est3 = DistanceGPSEstimator(cfg)
r_high = est3.estimate(80.0,  60.0,  0.0, 90.0, "ch")
r_med  = est3.estimate(20.0,  15.0,  0.0, 90.0, "cm")
r_low  = est3.estimate(5.0,   4.0,   0.0, 90.0, "cl")
ok_conf = r_high["confidence"]=="HIGH" and r_med["confidence"]=="MEDIUM" and r_low["confidence"]=="LOW"
status  = PASS if ok_conf else FAIL
print(f"\nTest 7 - Confidence Banding")
print(f"  80px: {r_high['confidence']} | 20px: {r_med['confidence']} | 5px: {r_low['confidence']}")
print(f"  -> STATUS: {status}")
assert status == PASS

print("\n" + "="*68)
print("ALL STEP 2 TESTS PASSED SUCCESSFULLY!")
print("="*68)
