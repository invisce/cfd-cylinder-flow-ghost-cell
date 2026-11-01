"""
Unit tests for Snapshot POD and Modal Analysis.
"""

import numpy as np
import pytest
from postprocessing.modal_analysis import SnapshotPOD, compute_phase_portrait_metrics


def test_snapshot_pod_traveling_wave():
    """Verify POD captures exact 2-mode quadrature for a harmonic traveling wave."""
    # Synthetic field: u(x, t) = sin(x - 2*pi*t) = sin(x)*cos(2*pi*t) - cos(x)*sin(2*pi*t)
    # Mode 1 and Mode 2 must share identical 50% energy, total cumulative energy = 1.0
    n_snaps = 40
    n_pts = 100
    x = np.linspace(0, 2.0 * np.pi, n_pts, endpoint=False)
    t = np.linspace(0, 1.0, n_snaps, endpoint=False)
    
    X, T = np.meshgrid(x, t)
    snapshots = np.sin(X - 2.0 * np.pi * T)
    
    pod = SnapshotPOD(n_modes=4)
    pod.fit(snapshots)
    
    # Cumulative energy of top 2 modes must be ~1.0
    assert np.isclose(pod.cumulative_energy[1], 1.0, atol=1e-3)
    # Top 2 modes must have equal energy (~0.50 each)
    assert np.isclose(pod.energy_ratios[0], 0.50, atol=1e-3)
    assert np.isclose(pod.energy_ratios[1], 0.50, atol=1e-3)
    
    # Phase portrait circularity
    metrics = compute_phase_portrait_metrics(pod.temporal_coeffs[0], pod.temporal_coeffs[1])
    assert metrics["circularity_error"] < 0.05
