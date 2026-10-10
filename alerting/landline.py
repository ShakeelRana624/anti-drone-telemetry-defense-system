"""
landline.py — Landline Auto-Dialer
=====================================
Initiates an automated outbound voice call via a PSTN modem
using Hayes AT commands. Hangs up after playing the alert message.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class LandlineDialer:
    """
    PSTN modem auto-dialer.

    Config keys (under 'alerting.landline'):
        enabled      (bool) : channel active flag
        port         (str)  : serial port of PSTN modem
        dial_number  (str)  : PSTN number to call
        message      (str)  : alert message text (logged / TTS)
    """

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled",      False)
        self._port:    str  = config.get("port",         "/dev/ttyS0")
        self._number:  str  = config.get("dial_number",  "911")
        self._message: str  = config.get("message",      "Drone detected. Respond immediately.")
        self._serial = None

    def dial(self) -> None:
        """Place an outbound call and play the alert message."""
        if not self._enabled:
            logger.debug("Landline disabled — skipped.")
            return
        logger.warning("☎️  Dialling %s …", self._number)
        try:
            self._open()
            self._at("AT")
            self._at("ATM1L3")                    # speaker on, loud
            self._at(f"ATDT{self._number}")       # dial
            time.sleep(15)                         # wait for answer
            self._play_message()
            self._at("ATH")                        # hang up
            logger.info("Landline call completed.")
        except Exception as exc:
            logger.error("Landline dialler error: %s", exc)
        finally:
            self._close()

    # ── Internal ─────────────────────────────────────────────────

    def _open(self) -> None:
        import serial  # type: ignore
        self._serial = serial.Serial(self._port, 9600, timeout=5)
        time.sleep(1)

    def _close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _at(self, cmd: str) -> str:
        if not self._serial:
            return ""
        self._serial.write((cmd + "\r\n").encode())
        time.sleep(0.5)
        resp = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
        logger.debug("AT >> %s | << %s", cmd, resp.strip())
        return resp

    def _play_message(self) -> None:
        """
        Transmit the alert message.
        Extend this to integrate a TTS engine (e.g. pyttsx3) or
        pre-encoded audio file via the modem's voice playback mode.
        """
        logger.info("Landline message: %s", self._message)
        # TODO: integrate TTS or audio file playback
