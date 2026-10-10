"""
detector.py — YOLO11m Inference Engine with SAHI + TTA
========================================================
Production-grade inference pipeline for military drone detection.

Pipeline per frame:
  1.  SAHI sliced inference  → detects small/distant drones in tiles
  2.  TTA (test-time augment)→ multi-augment merge for accuracy boost
  3.  NMS post-processing    → merge tile results, remove duplicates
  4.  DeepSORT tracker       → temporal confirmation & track IDs
  5.  Downstream dispatch    → alerts, logging, snapshots

Model loading strategy:
  • If  models/drone_yolo11m.pt  exists → use your custom trained model
  • Otherwise → fall back to  yolo11m.pt  (Ultralytics pretrained, auto-downloaded)
  
To swap in your trained model when training completes:
    1. Copy best.pt → models/drone_yolo11m.pt
    2. Call detector.reload_model()  (no restart required)
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import List, Optional

import numpy as np

from logging_system.evidence_recorder import EvidenceRecorder
from detection.decision_engine import DecisionEngine, DecisionResult
from detection.sahi_engine import SahiTTAEngine

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Detection data class
# ══════════════════════════════════════════════════════════════════

class Detection:
    """Single object detection result."""

    __slots__ = ("bbox", "confidence", "class_id", "class_name", "track_id")

    def __init__(
        self,
        bbox: List[float],      # [x1, y1, x2, y2] — normalised 0-1
        confidence: float,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None,
    ) -> None:
        self.bbox       = bbox
        self.confidence = confidence
        self.class_id   = class_id
        self.class_name = class_name
        self.track_id   = track_id

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def to_ltwh_pixels(self, w: int, h: int) -> List[float]:
        """Convert normalised bbox to [left, top, width, height] in pixels (DeepSORT format)."""
        x1, y1, x2, y2 = self.bbox
        return [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]

    def __repr__(self) -> str:
        cx, cy = self.center
        tid_str = f" id={self.track_id}" if self.track_id is not None else ""
        return (
            f"Detection({self.class_name!r}{tid_str} conf={self.confidence:.2f} "
            f"cx={cx:.3f} cy={cy:.3f})"
        )


# ══════════════════════════════════════════════════════════════════
# Detector
# ══════════════════════════════════════════════════════════════════

class DroneDetector:
    """
    YOLO11m-based drone detector with SAHI sliced inference and TTA.

    Parameters
    ----------
    config          : full system config dict
    tracker         : DeepSORTTracker instance
    alert_manager   : AlertManager instance
    event_logger    : EventLogger instance
    snapshot_saver  : SnapshotSaver instance
    """

    def __init__(
        self,
        config: dict,
        tracker,
        alert_manager,
        event_logger,
        snapshot_saver,
        evidence_recorder: Optional[EvidenceRecorder] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ) -> None:
        self._config         = config
        self._cfg            = config["detection"]
        self._sahi_cfg       = self._cfg.get("sahi", {})
        self._tta_cfg        = self._cfg.get("tta",  {})
        self._tracker        = tracker
        self._alert_manager  = alert_manager
        self._event_logger   = event_logger
        self._snapshot_saver = snapshot_saver
        self._evidence_recorder = evidence_recorder or EvidenceRecorder()
        self._decision_engine   = decision_engine or DecisionEngine()

        self._drone_classes: Optional[List[int]] = self._cfg.get("drone_classes")
        self._conf: float  = self._cfg.get("confidence_threshold", 0.40)
        self._iou:  float  = self._cfg.get("iou_threshold",        0.45)
        self._imgsz: int   = self._cfg.get("imgsz",                1280)
        self._device: str  = self._cfg.get("device",               "cpu")

        # Model objects (set by _load_model)
        self._yolo_model      = None          # raw Ultralytics YOLO
        self._sahi_model      = None          # SAHI AutoDetectionModel wrapper
        self._model_path: str = ""
        self._using_sahi: bool = self._sahi_cfg.get("enabled", True)
        self._using_tta:  bool = self._tta_cfg.get("enabled",  True)
        self._sahi_engine = SahiTTAEngine(
            slice_size=640,
            overlap_ratio=self._sahi_cfg.get("overlap_ratio", 0.25),
            iou_threshold=self._iou,
            use_tta=self._using_tta,
        )

        self._load_model()

    # ──────────────────────────────────────────────────────────────
    # Model loading & hot-reload
    # ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load YOLO11m model.
        Priority: custom trained (models/drone_yolo11m.pt) → pretrained fallback.
        """
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

        preferred = Path(self._cfg.get("model_path", "model/best.onnx"))
        fallback  = self._cfg.get("fallback_model", "yolo11m.pt")

        candidates = [
            preferred,
            Path("model/best.onnx"),
            Path("models/best.onnx"),
            Path("models/drone_yolo11m.pt"),
            Path("model/best.pt"),
            Path("models/best.pt"),
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break

        if found:
            path = str(found)
            logger.info("Loading CUSTOM trained model: %s", found)
        else:
            path = fallback
            logger.warning(
                "Custom model not found. Falling back to pretrained '%s'.",
                fallback,
            )

        self._yolo_model = YOLO(path)
        try:
            self._yolo_model.to(self._device)
        except Exception:
            pass
        self._model_path = path

        # SAHI wrapper
        if self._using_sahi and not path.endswith(".onnx"):
            self._load_sahi_wrapper(path)
        else:
            self._sahi_model = None

        logger.info(
            "Model ready | path=%s | device=%s | SAHI=%s | TTA=%s",
            path, self._device,
            self._using_sahi, self._using_tta,
        )

    def _load_sahi_wrapper(self, model_path: str) -> None:
        """Wrap YOLO11m with SAHI's AutoDetectionModel."""
        try:
            from sahi import AutoDetectionModel  # type: ignore
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=self._conf,
                device=self._device,
            )
            logger.info("SAHI AutoDetectionModel ready.")
        except ImportError:
            logger.warning(
                "SAHI not installed (pip install sahi). Falling back to standard inference."
            )
            self._using_sahi = False
        except Exception as exc:
            logger.warning("SAHI init failed: %s. Using standard inference.", exc)
            self._using_sahi = False

    def reload_model(self) -> None:
        """
        Hot-reload model without restarting the system.
        Call this once your training completes and you copy best.pt into place.
        """
        logger.info("Hot-reloading model …")
        self._load_model()
        logger.info("Model hot-reload complete.")

    # ──────────────────────────────────────────────────────────────
    # Public: per-frame processing
    # ──────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "primary",
    ) -> List[Detection]:
        """
        Full inference pipeline for one frame.

        Returns raw (pre-tracker) detections for this frame.
        Confirmed tracks are forwarded to alert/logging/snapshot.
        """
        h, w = frame.shape[:2]
        raw = self._infer(frame, w, h)
        confirmed = self._tracker.update(raw, camera_id, frame_shape=(h, w), frame=frame)

        # Multi-frame temporal confirmation via Decision Engine (prevents single-frame false alarms)
        decision = self._decision_engine.evaluate(confirmed, camera_id)
        if decision.should_trigger_alert and decision.confirmed_threats:
            self._handle_confirmed(decision.confirmed_threats, frame, camera_id)

        return raw

    # ──────────────────────────────────────────────────────────────
    # Inference (SAHI + TTA)
    # ──────────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray, w: int, h: int) -> List[Detection]:
        """Run SAHI sliced inference (+ TTA) and return Detection list."""
        if self._using_sahi and self._sahi_engine is not None:
            return self._sahi_engine.detect(
                self._yolo_model,
                frame,
                conf_threshold=self._conf,
                drone_classes=self._drone_classes,
            )
        return self._infer_standard(frame, w, h)

    def _infer_sahi(self, frame: np.ndarray, w: int, h: int) -> List[Detection]:
        """SAHI sliced inference pipeline."""
        try:
            from sahi.predict import get_sliced_prediction  # type: ignore

            result = get_sliced_prediction(
                image=frame,
                detection_model=self._sahi_model,
                slice_height=self._sahi_cfg.get("slice_height", 512),
                slice_width=self._sahi_cfg.get("slice_width",  512),
                overlap_height_ratio=self._sahi_cfg.get("overlap_ratio", 0.20),
                overlap_width_ratio=self._sahi_cfg.get("overlap_ratio",  0.20),
                postprocess_type=self._sahi_cfg.get("postprocess_type", "NMS"),
                postprocess_match_threshold=self._sahi_cfg.get(
                    "postprocess_match_threshold", 0.50
                ),
                verbose=0,
            )

            detections: List[Detection] = []
            for pred in result.object_prediction_list:
                cls_id   = pred.category.id
                cls_name = pred.category.name
                if self._drone_classes and cls_id not in self._drone_classes:
                    continue
                bb = pred.bbox
                # SAHI bbox: [minx, miny, maxx, maxy] in pixels → normalise
                bbox = [
                    bb.minx / w, bb.miny / h,
                    bb.maxx / w, bb.maxy / h,
                ]
                detections.append(
                    Detection(bbox, pred.score.value, cls_id, cls_name)
                )
            return detections

        except Exception as exc:
            logger.warning("SAHI inference error: %s — falling back to standard.", exc)
            return self._infer_standard(frame, w, h)

    def _infer_standard(self, frame: np.ndarray, w: int, h: int) -> List[Detection]:
        """Standard YOLO11m inference with optional TTA."""
        results = self._yolo_model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            augment=self._using_tta,    # TTA: internal Ultralytics flag
            verbose=False,
            stream=False,
        )

        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self._drone_classes and cls_id not in self._drone_classes:
                    continue
                bbox     = box.xyxyn[0].tolist()     # normalised [x1,y1,x2,y2]
                conf_val = float(box.conf[0])
                cls_name = result.names.get(cls_id, str(cls_id))
                detections.append(Detection(bbox, conf_val, cls_id, cls_name))

        return detections

    # ──────────────────────────────────────────────────────────────
    # Post-confirmation pipeline
    # ──────────────────────────────────────────────────────────────

    def _handle_confirmed(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        camera_id: str,
    ) -> None:
        """Dispatch confirmed detections to all downstream systems."""
        snapshot_path = self._snapshot_saver.save(frame, detections, camera_id)
        self._event_logger.log_detection(detections, camera_id, snapshot_path)
        self._alert_manager.trigger(detections, camera_id)

        # Trigger 5-sec forensic evidence video generation
        if self._evidence_recorder:
            try:
                cam_cfg = self._config.get("cameras", {}).get(camera_id, {})
                gps = (cam_cfg.get("latitude", 33.6844), cam_cfg.get("longitude", 73.0479))
                city = cam_cfg.get("city", "Tactical Sector Alpha")
                t = threading.Thread(
                    target=self._evidence_recorder.record_evidence_clip,
                    args=(frame.copy(), detections, camera_id),
                    kwargs={"gps_coords": gps, "city": city, "duration_sec": 5.0},
                    daemon=True,
                )
                t.start()
            except Exception as exc:
                logger.error("Evidence video generation dispatch failed: %s", exc)

        logger.warning(
            "[%s] ✅ CONFIRMED: %d drone(s) — snapshot: %s",
            camera_id.upper(), len(detections), snapshot_path,
        )

    # ──────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def using_sahi(self) -> bool:
        return self._using_sahi and self._sahi_model is not None

    @property
    def using_tta(self) -> bool:
        return self._using_tta
