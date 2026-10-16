"""
Alerting Subsystem Package
==========================
Coordinates multi-channel tactical alerts when drone incursions are confirmed:
  • AlertManager: Orchestrates notifications, cooldowns, and channel dispatch.
  • SirenController: Hardware/GPIO siren control with automated sound timeouts.
  • GsmSmsNotifier: Cellular modem SMS alerts with recipient failover.
  • LandlineDialer: Automated voice call synthesizer for fixed line alarms.
  • RfComsetInterface: Tactical military-band RF transceiver packet broadcast.
"""

from alerting.alert_manager import AlertManager
from alerting.gsm_sms import GsmSmsNotifier
from alerting.landline import LandlineDialer
from alerting.rf_comset import RfComsetInterface, get_rf_comset
from alerting.siren import SirenController, get_siren_controller

# Compatibility aliases
GsmSmsAlert = GsmSmsNotifier
LandlineAlert = LandlineDialer
RfComset = RfComsetInterface

__all__ = [
    "AlertManager",
    "GsmSmsNotifier",
    "GsmSmsAlert",
    "LandlineDialer",
    "LandlineAlert",
    "RfComsetInterface",
    "RfComset",
    "get_rf_comset",
    "SirenController",
    "get_siren_controller",
]
