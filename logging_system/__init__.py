"""
logging_system — Centralized Event Logging & Evidence Recording Package
========================================================================
Provides structured JSON logging, rotating daily logs, forensic MP4 video
evidence generation, and annotated detection snapshot persistence.
"""

from logging_system.event_logger import EventLogger
from logging_system.evidence_recorder import EvidenceRecorder, get_evidence_recorder
from logging_system.snapshot_saver import SnapshotSaver

__all__ = [
    "EventLogger",
    "EvidenceRecorder",
    "get_evidence_recorder",
    "SnapshotSaver",
]
