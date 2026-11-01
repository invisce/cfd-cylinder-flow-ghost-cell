"""
Unit tests for Feature Extraction and Observables module.
"""

import numpy as np
import pytest
from postprocessing.extract_observables import (
    extract_time_series_features,
    extract_wake_recirculation_length,
    extract_surface_pressure_sensors
)


def test_synthetic_time_series_features():
    """Verify feature extractor correctly identifies frequency, mean, RMS and harmonics."""
    fs = 100.0  # 100 Hz sampling
    t = np.linspace(0.0, 50.0, int(50.0 * fs))
    
    # Ground truth: shedding frequency f0 = 0.20 Hz (St = 0.20 for U=1, D=1)
    f0 = 0.20
    # Cl = 0.5 * sin(2*pi*f0*t) + 0.1 * sin(2*pi*(2*f0)*t)
    cl = 0.5 * np.sin(2.0 * np.pi * f0 * t) + 0.1 * np.sin(2.0 * np.pi * 2.0 * f0 * t)
    # Cd = 1.4 + 0.05 * cos(2*pi*(2*f0)*t) (Cd oscillates at 2*f0)
    cd = 1.4 + 0.05 * np.cos(2.0 * np.pi * 2.0 * f0 * t)
    
    feats = extract_time_series_features(t, cd, cl, D=1.0, U_inf=1.0, start_fraction=0.1)
    
    # Mean Cd must be ~1.4
    assert np.isclose(feats["mean_Cd"], 1.4, atol=0.01)
    # Mean Cl must be ~0.0
    assert np.isclose(feats["mean_Cl"], 0.0, atol=0.01)
    # Strouhal number must be close to 0.20
    assert np.isclose(feats["St"], 0.20, atol=0.02)
    # Harmonic ratio 2f should be detected
    assert feats["harmonic_ratio_2f"] > 0.15


def test_recirculation_length_synthetic():
    """Verify centerline zero-crossing detection."""
    x = np.linspace(1.5, 6.0, 100)
    y = np.linspace(-1.0, 1.0, 50)
    
    # Synthetic velocity field where u(x, y=0) goes from negative to positive at x = 3.5
    # xc = 2.0, D = 1.0 -> x_base = 2.5 -> Lr = 3.5 - 2.5 = 1.0 D
    u_mean = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        u_mean[i, :] = (xi - 3.5)
        
    lr = extract_wake_recirculation_length(x, y, u_mean, xc=2.0, yc=0.0, D=1.0)
    assert np.isclose(lr, 1.0, atol=0.05)


def test_surface_pressure_sensors_synthetic():
    """Verify sensor feature structure and values."""
    angles = np.array([0.0, 90.0, 180.0])
    cp_series = np.array([
        [1.0, -0.5, -0.8],
        [1.0, -0.7, -0.8],
        [1.0, -0.3, -0.8]
    ])
    
    res = extract_surface_pressure_sensors(angles, cp_series)
    assert np.isclose(res["sensor_000_mean"], 1.0)
    assert np.isclose(res["sensor_090_mean"], -0.5)
    assert np.isclose(res["sensor_180_mean"], -0.8)
    assert np.isclose(res["sensor_000_rms"], 0.0)
    assert res["sensor_090_rms"] > 0.1
