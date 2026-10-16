"""
siren.py — On-Site Siren Controller
=====================================
Activates a physical siren relay via GPIO (Raspberry Pi / Jetson).
Simulates gracefully on non-GPIO hardware (development / Windows).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class SirenController:
    """
    Controls a siren connected to a GPIO relay.

    Config keys (under 'alerting.siren'):
        enabled          (bool) : channel active flag
        gpio_pin         (int)  : BCM pin number
        duration_seconds (int)  : siren blast duration per alert
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the siren controller.

        Args:
            config: Alerting configuration dictionary with 'enabled', 'gpio_pin', and 'duration_seconds'.
        """
        self._enabled:  bool = config.get("enabled",          True)
        self._pin:      int  = config.get("gpio_pin",         17)
        self._duration: int  = config.get("duration_seconds", 30)
        self._lock = threading.Lock()
        self._gpio = None
        self._is_active = False
        self._active_start = 0.0
        self._cancel_event = threading.Event()
        if self._enabled:
            self._setup_gpio()

    @property
    def is_active(self) -> bool:
        """Boolean indicating whether the siren alarm is currently sounding."""
        return self._is_active

    @property
    def remaining_seconds(self) -> int:
        """Number of seconds remaining on the current active siren alarm cycle."""
        if not self._is_active:
            return 0
        elapsed = time.time() - self._active_start
        return max(0, int(self._duration - elapsed))

    def activate(self) -> None:
        """
        Sound the acoustic siren relay for the configured duration (default 30 seconds).
        Blocks current worker thread until duration completes or early deactivation is requested.
        """
        if not self._enabled:
            logger.debug("Siren disabled — skipped.")
            return
        with self._lock:
            self._cancel_event.clear()
            self._is_active = True
            self._active_start = time.time()
            logger.warning("🔊 SIREN ON — pin=%d for %ds (30-Second Audible Alarm)", self._pin, self._duration)
            self._set_pin(True)

            # Wait for duration or early cancel
            interrupted = self._cancel_event.wait(timeout=self._duration)
            self._set_pin(False)
            self._is_active = False
            if interrupted:
                logger.info("🔇 SIREN MUTED (User Interrupt)")
            else:
                logger.info("🔇 SIREN COMPLETED (30s elapsed)")

    def deactivate(self) -> None:
        """Immediately silence the siren and cancel any ongoing alarm cycle."""
        self._cancel_event.set()
        self._set_pin(False)
        self._is_active = False

    def get_status(self) -> dict:
        """
        Query current siren operational status and remaining blast time.

        Returns:
            dict: Dictionary containing enabled flag, pin, duration, is_active, and remaining_seconds.
        """
        return {
            "enabled": self._enabled,
            "pin": self._pin,
            "duration_seconds": self._duration,
            "is_active": self._is_active,
            "remaining_seconds": self.remaining_seconds
        }

    # ── GPIO ──────────────────────────────────────────────────────

    def _setup_gpio(self) -> None:
        """Initialize physical GPIO relay pin on Linux SBC (Raspberry Pi/Jetson); falls back to simulation."""
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            logger.info("GPIO ready — siren on pin %d.", self._pin)
        except ImportError:
            logger.info("RPi.GPIO unavailable — siren will simulate (dev mode).")
        except Exception as exc:
            logger.error("GPIO setup failed: %s", exc)

    def _set_pin(self, state: bool) -> None:
        """
        Set GPIO output pin voltage level.

        Args:
            state: True for HIGH (relay energized / siren active), False for LOW (silent).
        """
        if self._gpio:
            self._gpio.output(self._pin, self._gpio.HIGH if state else self._gpio.LOW)
        else:
            logger.debug("[SIM] Siren %s", "ON" if state else "OFF")

    def __del__(self) -> None:
        """Cleanup GPIO pins on object destruction if physical GPIO was acquired."""
        if self._gpio:
            try:
                self._gpio.cleanup(self._pin)
            except Exception:
                pass


_siren_instance: Optional[SirenController] = None

def get_siren_controller(config: Optional[dict] = None) -> SirenController:
    """
    Retrieve or initialize the global SirenController singleton instance.

    Args:
        config: Optional configuration dictionary.

    Returns:
        SirenController: Shared singleton instance.
    """
    global _siren_instance
    if _siren_instance is None:
        if config is None:
            config = {"enabled": True, "gpio_pin": 17, "duration_seconds": 30}
        _siren_instance = SirenController(config)
    return _siren_instance
