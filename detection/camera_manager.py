"""
camera_manager.py — Dual Camera Feed Handler
==============================================
Manages primary and secondary IP/USB camera streams.
Each stream runs its own capture thread.  A separate processing
thread per camera pulls the latest frame and sends it to the
detector at the configured FPS rate.

Sources can be:
  - Integer device index    (0, 1, 2 …)
  - RTSP URL string         ("rtsp://user:pass@192.168.1.10/stream")
  - File path for testing   ("test_video.mp4")
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Single camera stream (background capture thread)
# ══════════════════════════════════════════════════════════════════

class CameraStream:
    """
    Opens a video source and continuously reads frames into a
    double-buffer so the consumer always gets the freshest frame
    without blocking on I/O.
    """

    def __init__(
        self,
        camera_id: str,
        source,
        fps: int,
        resolution: tuple,
        label: str = "",
    ) -> None:
        """
        Initialize an individual CameraStream double-buffered capture thread.

        Args:
            camera_id: Unique string identifier for the camera.
            source: VideoCapture source descriptor (USB integer index, RTSP URL, or file path).
            fps: Desired frame ingestion rate.
            resolution: (width, height) pixel resolution tuple.
            label: Optional human-readable camera designation.
        """
        self.camera_id  = camera_id
        self.source     = source
        self.target_fps = fps
        self.resolution = tuple(resolution)
        self.label      = label or camera_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray]     = None
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.healthy  = False          # True once first frame arrives

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open video capture device, configure capture properties, and start background frame reader thread.

        Raises:
            RuntimeError: If video source fails to open.
        """
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera '{self.camera_id}' source: {self.source!r}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)      # minimise latency

        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"cam-{self.camera_id}",
        )
        self._thread.start()
        logger.info("Camera '%s' started  (source=%s).", self.camera_id, self.source)

    def stop(self) -> None:
        """Terminate background capture thread and release OpenCV VideoCapture hardware handle."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("Camera '%s' stopped.", self.camera_id)

    # ── Read latest frame ─────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Fetch thread-safe copy of the freshest available video frame without blocking.

        Returns:
            Optional[np.ndarray]: Latest BGR video frame, or None if no frame has arrived yet.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── Internal capture loop ─────────────────────────────────────

    def _capture_loop(self) -> None:
        """Continuous frame polling loop reading from hardware source with automatic reconnect triggers."""
        consecutive_fails = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                consecutive_fails += 1
                if consecutive_fails % 30 == 1:
                    logger.warning(
                        "Camera '%s': read failure #%d.", self.camera_id, consecutive_fails
                    )
                time.sleep(0.05)
                # Try to reconnect after 5 consecutive failures
                if consecutive_fails == 5:
                    self._reconnect()
                continue
            consecutive_fails = 0
            self.healthy = True
            with self._lock:
                self._frame = frame

    def _reconnect(self) -> None:
        """Attempt to reopen the capture device."""
        logger.info("Camera '%s': attempting reconnect …", self.camera_id)
        try:
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
        except Exception as exc:
            logger.error("Camera '%s' reconnect failed: %s", self.camera_id, exc)


# ══════════════════════════════════════════════════════════════════
# Camera Manager
# ══════════════════════════════════════════════════════════════════

class CameraManager:
    """
    Manages all configured camera streams and dispatches frames
    to the detector on independent processing threads.
    """

    def __init__(self, config: dict, detector) -> None:
        """
        Initialize the CameraManager with system cameras configuration and detector pipeline reference.

        Args:
            config: System configuration dictionary containing 'cameras' definitions.
            detector: DroneDetector instance that processes incoming camera frames.
        """
        self._cam_cfg  = config.get("cameras", {})
        self._detector = detector

        self._streams:   Dict[str, CameraStream] = {}
        self._proc_threads: Dict[str, threading.Thread] = {}
        self._running = False

        self._init_streams()

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """
        Start all configured CameraStream capture workers and spawn dedicated processing threads
        pulling frames for the detector pipeline.
        """
        self._running = True
        for cam_id, stream in self._streams.items():
            try:
                stream.start()
            except RuntimeError as exc:
                logger.error("Failed to start camera '%s': %s — skipping.", cam_id, exc)
                continue
            t = threading.Thread(
                target=self._process_loop,
                args=(cam_id, stream),
                daemon=True,
                name=f"proc-{cam_id}",
            )
            t.start()
            self._proc_threads[cam_id] = t
        logger.info("CameraManager: %d stream(s) started.", len(self._proc_threads))

    def stop(self) -> None:
        """Stop all camera streams and join worker processing threads."""
        self._running = False
        for stream in self._streams.values():
            stream.stop()
        for t in self._proc_threads.values():
            t.join(timeout=5)
        logger.info("CameraManager stopped.")

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> Dict[str, bool]:
        """
        Query health status for all camera streams.

        Returns:
            Dict[str, bool]: Mapping of {camera_id: is_healthy} for web dashboard telemetry.
        """
        return {cid: s.healthy for cid, s in self._streams.items()}

    # ── Internal ─────────────────────────────────────────────────

    def _init_streams(self) -> None:
        """Initialize CameraStream instances for every configured camera entry in configuration."""
        for cam_id, cam_cfg in self._cam_cfg.items():
            self._streams[cam_id] = CameraStream(
                camera_id=cam_id,
                source=cam_cfg["source"],
                fps=cam_cfg.get("fps", 25),
                resolution=cam_cfg.get("resolution", [1280, 720]),
                label=cam_cfg.get("label", cam_id),
            )

    def _process_loop(self, camera_id: str, stream: CameraStream) -> None:
        """
        Processing worker loop: pulls latest frame from stream at target FPS and submits to detector.

        Args:
            camera_id: Stream identifier string.
            stream: CameraStream handle.
        """
        interval = 1.0 / stream.target_fps
        while self._running:
            start = time.monotonic()
            frame = stream.read()
            if frame is not None:
                try:
                    self._detector.process_frame(frame, camera_id)
                except Exception as exc:
                    logger.error(
                        "Detector error on camera '%s': %s", camera_id, exc
                    )
            elapsed = time.monotonic() - start
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)
