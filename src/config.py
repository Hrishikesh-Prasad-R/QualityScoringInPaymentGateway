"""
Configuration constants for the Data Quality Scoring Engine.
All thresholds and settings are defined here for transparency and auditability.
"""
from enum import Enum
from typing import Dict


class Action(Enum):
    """Final pipeline actions - the 4 possible outcomes."""
    SAFE_TO_USE = "SAFE_TO_USE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ESCALATE = "ESCALATE"
    NO_ACTION = "NO_ACTION"


class ConfidenceBand(Enum):
    """Confidence classification for decisions."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LayerStatus(Enum):
    """Status codes for layer execution."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"
    SKIPPED = "SKIPPED"


# ============================================================================
# INPUT CONTRACT DEFAULTS (Layer 1)
# ============================================================================
ACCEPTED_FORMATS = ["csv", "json"]
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_ROW_COUNT = 100
MAX_ROW_COUNT = 50000
REQUIRED_ENCODING = "utf-8"

# ============================================================================
# QUALITY DIMENSION THRESHOLDS (Layer 4.2)
# ============================================================================
DEFAULT_QUALITY_THRESHOLDS: Dict[str, float] = {
    "completeness": 95.0,
    "accuracy": 90.0,
    "validity": 99.0,
    "uniqueness": 99.9,
    "consistency": 95.0,
    "timeliness": 90.0,
    "integrity": 95.0,
}

# Dimension weights for composite DQS calculation
DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness": 0.20,
    "accuracy": 0.20,
    "validity": 0.15,
    "uniqueness": 0.10,
    "consistency": 0.15,
    "timeliness": 0.10,
    "integrity": 0.10,
}

# ============================================================================
# ANOMALY DETECTION (Layer 4.4)
# ============================================================================
ANOMALY_HIGH_THRESHOLD = 0.75
ANOMALY_MEDIUM_THRESHOLD = 0.50
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_RANDOM_STATE = 42  # FROZEN - reproducibility

# ============================================================================
# DECISION GATE (Layer 9)
# ============================================================================
DQS_CRITICAL_THRESHOLD = 40.0  # Below this -> ESCALATE
DQS_BORDERLINE_THRESHOLD = 75.0  # Below this -> REVIEW_REQUIRED
ANOMALY_FLAG_CRITICAL_PERCENT = 5.0  # Above this -> REVIEW_REQUIRED

# ============================================================================
# CONFIDENCE BAND (Layer 8)
# ============================================================================
CONFIDENCE_HIGH_THRESHOLD = 70
CONFIDENCE_MEDIUM_THRESHOLD = 30
