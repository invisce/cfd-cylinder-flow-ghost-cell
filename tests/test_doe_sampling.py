"""
Unit tests for DoE parameter sampling generator.
"""

import numpy as np
import pytest
from cases.doe_sampling import generate_paper1_doe


def test_doe_counts_and_splits():
    """Verify total counts, bounds, and strict 80/20 train/test splits."""
    cases = generate_paper1_doe(n_total=180, seed=123)
    assert len(cases) == 180
    
    n_train = sum(1 for c in cases if c["split"] == "train")
    n_test = sum(1 for c in cases if c["split"] == "holdout_test")
    assert n_train == 144
    assert n_test == 36
    assert (n_test / 180.0) == 0.20


def test_doe_parameter_ranges():
    """Verify all parameters strictly stay within specified physical bounds."""
    cases = generate_paper1_doe(n_total=180, seed=42)
    for c in cases:
        assert 80.0 <= c["Re"] <= 180.0
        assert 0.0 <= c["depth_ratio"] <= 0.1001
        assert 0.0 <= c["width_deg"] <= 60.01
        assert 0.0 <= c["theta_d_deg"] <= 180.01
