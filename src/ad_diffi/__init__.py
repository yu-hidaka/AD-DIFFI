"""
AD-DIFFI: Adjusted Depth-based Isolation Forest Feature Importance
------------------------------------------------------------------
A robust feature importance method for Isolation Forest, specifically 
designed to handle mixed-type data (continuous and binary) in clinical 
informatics and anomaly detection.

Main features:
- Root-Split-Only (RSO) constraint for binary features.
- Noise-based Z-score normalization for cross-type fairness.
"""

__version__ = "0.1.0"
__author__ = "Yu Hidaka"

# Import core functions to the top-level namespace
from .core import (
    diffi_ib_binary_rso,
    calculate_ad_diffi_zscore
)

# Define the public API of the package
__all__ = [
    "diffi_ib_binary_rso",
    "calculate_ad_diffi_zscore",
]
