"""
Detection & Tracking Stack Package
==================================
Houses the computer vision and multi-object tracking pipeline:
  • DroneDetector: YOLO11 + SAHI + TTA inference pipeline with tracking coordination.
  • DeepSORTTracker: Kalman filter and appearance metric tracking with cross-camera ID handoff.
  • DecisionEngine: Multi-frame temporal confirmation filter (4-6 frames verifying, 8 frames locked).
  • SahiTTAEngine: Slicing Aided Hyper Inference engine for distant micro-drone detection.
  • CameraManager: Multi-threaded frame grabber coordinating multiple camera inputs.
"""

from detection.detector import DroneDetector, Detection
from detection.tracker import DeepSORTTracker
from detection.decision_engine import DecisionEngine, DecisionResult, ThreatLevel, TargetThreatState
from detection.sahi_engine import SahiTTAEngine
from detection.camera_manager import CameraManager

__all__ = [
    "DroneDetector",
    "Detection",
    "DeepSORTTracker",
    "DecisionEngine",
    "DecisionResult",
    "ThreatLevel",
    "TargetThreatState",
    "SahiTTAEngine",
    "CameraManager",
]
