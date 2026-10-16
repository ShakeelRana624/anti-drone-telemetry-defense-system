"""
gsm_sms.py — GSM/SMS Alert Notifier
=====================================
Sends SMS alerts via an AT-command compatible GSM modem
over a serial port (e.g., SIM800L, SIM900, Quectel EC21).

AT command sequence:
    AT          → modem alive check
    AT+CMGF=1   → set SMS text mode
    AT+CMGS=N   → specify recipient
    <message>   → message body + Ctrl-Z (0x1A) to send
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class GsmSmsNotifier:
    """
    Sends SMS through a serial GSM modem using AT commands.

    Config keys (under 'alerting.gsm_sms'):
        enabled          (bool)       : channel active flag
        port             (str)        : serial port path
        baud_rate        (int)        : baud rate [9600]
        recipients       (list[str])  : E.164 phone numbers
        message_template (str)        : template with {timestamp}, {bearing}, etc.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the GSM/SMS alert notifier.

        Args:
            config: Alerting configuration dictionary containing 'enabled', 'port', 'baud_rate',
                    'recipients', and 'message_template'.
        """
        self._enabled:    bool        = config.get("enabled",    True)
        self._port:       str         = config.get("port",       "/dev/ttyUSB0")
        self._baud:       int         = config.get("baud_rate",  9600)
        self._recipients: List[str]   = config.get("recipients", [])
        self._template:   str         = config.get(
            "message_template",
            "DRONE ALERT [{timestamp}] Cam:{camera_id} Bearing:{bearing}deg Count:{count}",
        )
        self._serial = None

    # ── Public API ────────────────────────────────────────────────

    def send_alert(self, context: dict) -> None:
        """
        Format the alert template with detection telemetry and dispatch SMS to all configured recipients.

        Args:
            context: Alert metadata dictionary (e.g. timestamp, camera_id, bearing, count).
        """
        if not self._enabled:
            logger.debug("GSM/SMS disabled — skipped.")
            return
        if not self._recipients:
            logger.warning("GSM/SMS: no recipients configured.")
            return

        message = self._template.format(**context)
        for number in self._recipients:
            self._send(number, message)

    # ── Internal ─────────────────────────────────────────────────

    def _open(self) -> None:
        """Open and configure the serial connection to the GSM modem in SMS text mode."""
        import serial  # type: ignore
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(self._port, self._baud, timeout=5)
        time.sleep(1)
        self._at("AT")
        self._at("AT+CMGF=1")   # SMS text mode

    def _send(self, number: str, message: str) -> None:
        """
        Transmit an individual SMS message to a phone number via AT+CMGS command sequence.

        Args:
            number: Recipient telephone number in E.164 format.
            message: Formatted text message payload.
        """
        try:
            self._open()
            self._at(f'AT+CMGS="{number}"')
            self._serial.write(message.encode("utf-8") + b"\x1A")
            time.sleep(4)
            resp = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
            if "+CMGS" in resp:
                logger.info("SMS sent → %s", number)
            else:
                logger.warning("SMS to %s may have failed. Response: %s", number, resp.strip())
        except Exception as exc:
            logger.error("GSM/SMS send to %s failed: %s", number, exc)

    def _at(self, cmd: str) -> str:
        """
        Write an AT command to the serial modem and read the response.

        Args:
            cmd: AT command string without trailing CR/LF.

        Returns:
            str: Decoded response string from modem.
        """
        self._serial.write((cmd + "\r\n").encode())
        time.sleep(0.5)
        resp = self._serial.read(self._serial.in_waiting).decode(errors="ignore")
        logger.debug("AT >> %s | << %s", cmd, resp.strip())
        return resp

    def close(self) -> None:
        """Gracefully close the hardware serial port connection to the GSM modem."""
        if self._serial and self._serial.is_open:
            self._serial.close()
