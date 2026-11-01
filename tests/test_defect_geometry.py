"""
Unit tests for DefectCylinderGeometry module.
"""

import numpy as np
import pytest
from solver.defect_geometry import DefectCylinderGeometry, wrap_angle


def test_zero_defect_matches_circular():
    """Verify d=0 recovers textbook circular geometry."""
    geom = DefectCylinderGeometry(xc=2.0, yc=2.0, D=1.0, depth_ratio=0.0)
    thetas = np.linspace(-np.pi, np.pi, 100)
    
    # Radius must be exactly 0.5
    assert np.allclose(geom.radius(thetas), 0.5)
    # Derivative dr/dtheta must be 0
    assert np.allclose(geom.dr_dtheta(thetas), 0.0)
    # Normals must point purely radially: (cos(theta), sin(theta))
    nx, ny = geom.unit_normal(thetas)
    assert np.allclose(nx, np.cos(thetas))
    assert np.allclose(ny, np.sin(thetas))


def test_defect_center_depth():
    """Verify radius at center of defect is exactly R - delta."""
    D = 1.0
    R = 0.5
    d = 0.08  # 8% depth
    w = np.deg2rad(40.0)
    theta_d_rel = np.deg2rad(90.0)
    
    geom = DefectCylinderGeometry(xc=0.0, yc=0.0, D=D, depth_ratio=d, width_rad=w, theta_d_rad=theta_d_rel)
    
    # Center of defect
    r_center = geom.radius(geom.theta_d)
    expected = R - (d * D)
    assert np.isclose(r_center, expected, atol=1e-12)
    
    # First derivative at the center must be zero (local extremum)
    assert np.isclose(geom.dr_dtheta(geom.theta_d), 0.0, atol=1e-12)


def test_analytical_vs_finite_difference_derivative():
    """Verify analytical dr/dtheta matches numerical central differences."""
    geom = DefectCylinderGeometry(xc=1.5, yc=1.5, D=1.0, depth_ratio=0.075, width_rad=np.deg2rad(30.0), theta_d_rad=np.deg2rad(60.0))
    
    thetas = np.linspace(-np.pi, np.pi, 200)
    eps = 1e-6
    r_plus = geom.radius(thetas + eps)
    r_minus = geom.radius(thetas - eps)
    num_deriv = (r_plus - r_minus) / (2.0 * eps)
    ana_deriv = geom.dr_dtheta(thetas)
    
    # Maximum difference should be small (< 1e-6)
    assert np.allclose(ana_deriv, num_deriv, atol=1e-6)


def test_normal_orthogonality_and_magnitude():
    """Verify unit normal has length 1 and is strictly orthogonal to surface tangent."""
    geom = DefectCylinderGeometry(xc=0.0, yc=0.0, D=1.0, depth_ratio=0.10, width_rad=np.deg2rad(20.0), theta_d_rad=np.deg2rad(45.0))
    
    thetas = np.linspace(-np.pi, np.pi, 150)
    nx, ny = geom.unit_normal(thetas)
    
    # Check unit norm
    norms = np.hypot(nx, ny)
    assert np.allclose(norms, 1.0, atol=1e-12)
    
    # Compute tangent vector: t = (dx/dtheta, dy/dtheta)
    r = geom.radius(thetas)
    rp = geom.dr_dtheta(thetas)
    tx = rp * np.cos(thetas) - r * np.sin(thetas)
    ty = rp * np.sin(thetas) + r * np.cos(thetas)
    
    # Dot product n . t must be exactly 0
    dot = nx * tx + ny * ty
    assert np.allclose(dot, 0.0, atol=1e-12)


def test_signed_distance_signs():
    """Verify signed distance function signs: negative inside, positive outside."""
    geom = DefectCylinderGeometry(xc=2.0, yc=2.0, D=1.0, depth_ratio=0.05, width_rad=np.deg2rad(30.0), theta_d_rad=np.deg2rad(90.0))
    
    # Point at center is definitely inside
    assert geom.signed_distance(2.0, 2.0) < -0.4
    
    # Points far outside are positive
    assert geom.signed_distance(2.0, 4.0) > 1.0
    
    # Points on the boundary must evaluate to ~0
    thetas = np.linspace(-np.pi, np.pi, 50)
    bx, by = geom.surface_point(thetas)
    sdf_boundary = geom.signed_distance(bx, by)
    assert np.allclose(sdf_boundary, 0.0, atol=1e-5)
