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
        """
        Initialize the LandlineDialer subsystem.

        Args:
            config: Alerting configuration dictionary containing 'enabled', 'port', 'dial_number',
                    and 'message'.
        """
        self._enabled: bool = config.get("enabled",      False)
        self._port:    str  = config.get("port",         "/dev/ttyS0")
        self._number:  str  = config.get("dial_number",  "911")
        self._message: str  = config.get("message",      "Drone detected. Respond immediately.")
        self._serial = None

    def dial(self) -> None:
        """
        Place an automated outbound voice call over PSTN modem using Hayes AT commands,
        wait for connection, broadcast the voice alert message, and hang up.
        """
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
        """Establish serial connection to PSTN modem at 9600 baud."""
        import serial  # type: ignore
        self._serial = serial.Serial(self._port, 9600, timeout=5)
        time.sleep(1)

    def _close(self) -> None:
        """Close serial port connection to PSTN modem."""
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _at(self, cmd: str) -> str:
        """
        Send Hayes AT command to PSTN modem and return response.

        Args:
            cmd: Command string to transmit.

        Returns:
            str: Modem response string.
        """
        if not self._serial:
            return ""
        self._serial.write((cmd + "\r\n").encode())
        time.sleep(0.5)
        resp = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
        logger.debug("AT >> %s | << %s", cmd, resp.strip())
        return resp

    def _play_message(self) -> None:
        """
        Transmit the audio or synthesized voice alert message through the active modem call session.
        Logs message text and can be extended with external TTS engines.
        """
        logger.info("Landline message: %s", self._message)
        # TODO: integrate TTS or audio file playback
