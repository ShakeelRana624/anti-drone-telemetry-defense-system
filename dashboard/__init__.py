"""
dashboard — Real-Time Web Monitoring & Tactical Airspace Radar Package
======================================================================
Provides a Flask-powered web interface delivering:
  • Low-latency Server-Sent Events (SSE) telemetry push
  • Live tactical boresight camera stream overlays
  • Real-time compass bearing, distance estimation, and GPS coordinates
  • Siren and RF Comset operational hardware controls
  • Forensic evidence clip library and analytics reports
"""

from dashboard.app import DashboardApp, push_sse_event

__all__ = [
    "DashboardApp",
    "push_sse_event",
]
