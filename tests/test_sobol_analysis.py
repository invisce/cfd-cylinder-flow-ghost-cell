"""
Unit tests for Sobol Global Sensitivity Analysis module.
"""

import numpy as np
import pytest
from surrogate.sobol_analysis import SobolSensitivityAnalyzer


def test_sobol_additive_linear_model():
    """Verify S_1 and S_T on a purely additive model where analytical variance is known."""
    # Y = 3*X0 + 0*X1 + 1*X2
    # Var(Y) = 9*Var(X0) + 1*Var(X2)
    # If X0, X1, X2 ~ U(0, 1):
    # Var(Xi) = 1/12
    # Var(Y) = (9 + 1)/12 = 10/12
    # S0 = 9/10 = 0.90, S1 = 0.0, S2 = 1/10 = 0.10
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    names = ["Re", "d", "phi"]
    analyzer = SobolSensitivityAnalyzer(bounds=bounds, names=names, seed=123)
    
    samples, N = analyzer.generate_saltelli_samples(n_samples=4096)
    
    # Model evaluation
    Y = 3.0 * samples[:, 0] + 0.0 * samples[:, 1] + 1.0 * samples[:, 2]
    
    indices = analyzer.compute_indices(Y, N)
    
    # Check S1
    assert np.isclose(indices["Re"]["S1"], 0.90, atol=0.04)
    assert np.isclose(indices["d"]["S1"], 0.00, atol=0.03)
    assert np.isclose(indices["phi"]["S1"], 0.10, atol=0.04)
    
    # Check ST
    assert np.isclose(indices["Re"]["ST"], 0.90, atol=0.04)
    assert np.isclose(indices["d"]["ST"], 0.00, atol=0.03)
    assert np.isclose(indices["phi"]["ST"], 0.10, atol=0.04)
