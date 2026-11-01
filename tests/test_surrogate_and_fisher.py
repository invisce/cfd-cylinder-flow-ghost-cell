"""
Unit tests for GP Surrogate and Fisher Identifiability modules.
"""

import numpy as np
import pytest
from surrogate.gp_model import ObservableGPSurrogate
from surrogate.fisher_identifiability import FisherIdentifiabilityAnalyzer


def test_gp_surrogate_fit_and_predict():
    """Verify GP fits synthetic response surface and predicts accurate mean/variance."""
    rng = np.random.default_rng(42)
    # 2D input: X = [Re, d]
    X_train = rng.uniform([80.0, 0.0], [180.0, 0.10], size=(30, 2))
    # Synthetic observable: y = 1.5 - 2.0*d + 0.005*Re
    y_train = 1.5 - 2.0 * X_train[:, 1] + 0.005 * X_train[:, 0]
    
    gp = ObservableGPSurrogate(n_restarts_optimizer=2)
    gp.fit(X_train, y_train)
    
    # Test on independent points
    X_test = rng.uniform([80.0, 0.0], [180.0, 0.10], size=(15, 2))
    y_test = 1.5 - 2.0 * X_test[:, 1] + 0.005 * X_test[:, 0]
    
    metrics = gp.evaluate(X_test, y_test)
    assert metrics["R2"] > 0.95
    assert metrics["NRMSE"] < 0.05
    assert metrics["coverage_95"] >= 0.80


def test_fisher_matrix_and_schur_complement():
    """Verify Fisher Information and Schur complement calculation for nuisance parameter removal."""
    analyzer = FisherIdentifiabilityAnalyzer()
    
    # 5 observables, 4 parameters: 2 target defect parameters (d, theta), 2 nuisance (Re, bias)
    # J shape: (5, 4)
    J = np.array([
        [1.0, 0.2, 0.5, 1.0],
        [0.8, 0.1, 0.4, 1.0],
        [0.0, 2.0, 0.1, 1.0],
        [0.2, 1.5, 0.2, 1.0],
        [0.5, 0.5, 0.3, 1.0]
    ])
    
    F_full = analyzer.compute_fisher_matrix(J)
    assert F_full.shape == (4, 4)
    
    # Effective Fisher matrix for target parameters (first 2)
    F_eff = analyzer.effective_fisher_matrix(F_full, n_target=2)
    assert F_eff.shape == (2, 2)
    
    metrics = analyzer.evaluate_metrics(F_eff)
    assert "d_optimality" in metrics
    assert "e_optimality" in metrics
    assert metrics["e_optimality"] > 0.0


def test_greedy_sensor_selection():
    """Verify greedy selection picks the most informative sensors."""
    analyzer = FisherIdentifiabilityAnalyzer()
    
    # 4 candidate sensors, each measuring 1 observable with respect to 2 target parameters
    # Sensor 0: high sensitivity [5.0, 0.0]
    # Sensor 1: high sensitivity [0.0, 5.0]
    # Sensor 2: noisy/weak [0.1, 0.1]
    # Sensor 3: moderate [1.0, 1.0]
    sensor_jacobians = [
        np.array([[5.0, 0.0]]),
        np.array([[0.0, 5.0]]),
        np.array([[0.1, 0.1]]),
        np.array([[1.0, 1.0]])
    ]
    
    selected, history = analyzer.select_optimal_sensor_subset(sensor_jacobians, k_sensors=2, n_target=2)
    assert len(selected) == 2
    # The two orthogonal high-sensitivity sensors (0 and 1) must be chosen
    assert 0 in selected
    assert 1 in selected
