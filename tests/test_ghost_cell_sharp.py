"""
Test suite for Stage 0.3: Ghost-Cell Sharp-Interface Boundary Formulation.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from ghost_cell import CircleSDF, GhostCellBoundary

def test_circle_sdf():
    print("Testing CircleSDF...")
    sdf = CircleSDF(xc=2.5, yc=1.5, radius=0.5)
    
    # Points on the circle
    assert abs(sdf.distance(3.0, 1.5)) < 1e-15
    assert abs(sdf.distance(2.5, 2.0)) < 1e-15
    assert abs(sdf.distance(2.0, 1.5)) < 1e-15
    
    # Inside and outside
    assert sdf.distance(2.5, 1.5) == -0.5
    assert sdf.distance(3.5, 1.5) == 0.5
    
    # Normal vectors
    nx, ny = sdf.normal(3.0, 1.5)
    assert abs(nx - 1.0) < 1e-15 and abs(ny) < 1e-15
    nx, ny = sdf.normal(2.5, 2.0)
    assert abs(nx) < 1e-15 and abs(ny - 1.0) < 1e-15
    print("  [OK] CircleSDF distance and normal vectors verified")

def test_flat_wall_shear_flow():
    print("Testing Ghost-Cell on simple planar boundary (Couette shear)...")
    # Simulation domain [0, 1] x [0, 1], wall at y = 0.5
    # Couette shear flow: u(x, y) = y - 0.5 for y >= 0.5, u_wall = 0
    # True velocity at wall y=0.5 is exactly 0.0.
    # Ghost points at y < 0.5 should have u_ghost = -(y_IP - 0.5) = -(0.5 - y_ghost) = y_ghost - 0.5
    # Meaning linear velocity profile continues smoothly across the boundary!
    
    class FlatWallSDF:
        def distance(self, x, y):
            return y - 0.5
        def normal(self, x, y):
            return np.zeros_like(x), np.ones_like(y)
        @property
        def radius(self):
            return 0.5
        @property
        def xc(self):
            return 0.5
        @property
        def yc(self):
            return 0.5

    nx, ny = 20, 20
    dx, dy = 1.0 / nx, 1.0 / ny
    gc = GhostCellBoundary(nx, ny, dx, dy, FlatWallSDF(), n_theta=180)

    # Initialize u with shear flow
    u = np.zeros((ny, nx + 1))
    v = np.zeros((ny + 1, nx))
    for j in range(ny):
        y = (j + 0.5) * dy
        if y >= 0.5:
            u[j, :] = y - 0.5

    gc.apply_velocity_ghost_cells(u, v, u_wall=0.0, v_wall=0.0)

    # Verify that ghost cells satisfy u = y - 0.5
    for idx, (j, i) in enumerate(gc.ghost_u_indices):
        y_g = (j + 0.5) * dy
        expected = y_g - 0.5
        actual = u[j, i]
        assert abs(actual - expected) < 1e-14, f"Mismatch at ghost cell {j},{i}: {actual} vs {expected}"

    print("  [OK] Planar wall no-slip enforcement and linear extrapolation verified")

def test_cylinder_ghost_cells():
    print("Testing Pristine Cylinder Ghost-Cell mapping and image points...")
    nx, ny = 100, 50
    dx, dy = 0.05, 0.05
    xc, yc, R = 2.5, 1.25, 0.5
    sdf = CircleSDF(xc, yc, R)
    gc = GhostCellBoundary(nx, ny, dx, dy, sdf, n_theta=360)

    # Verify ghost cells were found
    assert len(gc.ghost_u_indices) > 0
    assert len(gc.ghost_v_indices) > 0
    print(f"  Identified {len(gc.ghost_u_indices)} u-ghost faces and {len(gc.ghost_v_indices)} v-ghost faces")

    # Verify all image points are strictly in fluid (distance >= 0)
    for (j, i) in gc.ghost_u_indices:
        xg = gc.xu[i]
        yg = gc.yu[j]
        d = abs(gc.dist_u[j, i])
        nx_val, ny_val = sdf.normal(xg, yg)
        x_ip = xg + 2.0 * d * nx_val
        y_ip = yg + 2.0 * d * ny_val
        dist_ip = sdf.distance(x_ip, y_ip)
        assert dist_ip >= -1e-14, f"Image point inside solid! dist = {dist_ip}"

    # Test physical viscous boundary-layer flow around stationary cylinder:
    # V_theta(r) = A * (r - R^2 / r), with exact no-slip V_theta(R) = 0 at the physical cylinder wall.
    A = 2.0
    u = np.zeros((ny, nx + 1))
    v = np.zeros((ny + 1, nx))
    p = np.zeros((ny, nx))

    # Fill fluid velocities with physical shear profile
    for j in range(ny):
        for i in range(nx + 1):
            if gc.fluid_u[j, i]:
                x = gc.xu[i] - xc
                y = gc.yu[j] - yc
                r = np.sqrt(x**2 + y**2)
                v_theta = A * (r - R**2 / r)
                u[j, i] = -v_theta * (y / r)
    for j in range(ny + 1):
        for i in range(nx):
            if gc.fluid_v[j, i]:
                x = gc.xv[i] - xc
                y = gc.yv[j] - yc
                r = np.sqrt(x**2 + y**2)
                v_theta = A * (r - R**2 / r)
                v[j, i] = v_theta * (x / r)

    # Apply ghost cell no-slip conditions
    gc.apply_velocity_ghost_cells(u, v, u_wall=0.0, v_wall=0.0)

    # Check interpolated boundary velocity on true circle r=R
    from scipy.interpolate import RegularGridInterpolator
    u_interp = RegularGridInterpolator((gc.yu, gc.xu), u, bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((gc.yv, gc.xv), v, bounds_error=False, fill_value=0.0)

    pts = np.column_stack((gc.surf_y, gc.surf_x))
    u_boundary = u_interp(pts)
    v_boundary = v_interp(pts)
    
    max_boundary_slip = np.max(np.sqrt(u_boundary**2 + v_boundary**2))
    assert max_boundary_slip < 0.03, f"Boundary slip too large: {max_boundary_slip}"
    print(f"  [OK] Max boundary slip on true physical circle: {max_boundary_slip:.4f}")

    # Test force extraction: Uniform pressure p = 1.0 must give ZERO net force by Gauss theorem
    p.fill(1.0)
    fx_tot, fy_tot, fx_p, fy_p, fx_v, fy_v = gc.extract_true_surface_forces(u, v, p, mu=0.01)
    assert abs(fx_p) < 1e-12, f"Spurious pressure drag in uniform pressure: {fx_p}"
    assert abs(fy_p) < 1e-12, f"Spurious pressure lift in uniform pressure: {fy_p}"
    print(f"  [OK] True surface force integration: net force in uniform pressure < 1e-12")

    # Test physical viscous drag sign: a stream flowing left-to-right (u > 0) MUST produce positive viscous drag
    u_stream = np.zeros_like(u)
    v_stream = np.zeros_like(v)
    u_stream[gc.fluid_u] = 1.0
    gc.apply_velocity_ghost_cells(u_stream, v_stream, u_wall=0.0, v_wall=0.0)
    p_zero = np.zeros_like(p)
    _, _, _, _, fx_v_stream, fy_v_stream = gc.extract_true_surface_forces(u_stream, v_stream, p_zero, mu=0.01)
    assert fx_v_stream > 0.5, f"Viscous drag sign inverted or magnitude too small: {fx_v_stream}"
    assert abs(fy_v_stream) < 1e-12, f"Asymmetric viscous lift in symmetric stream: {fy_v_stream}"
    print(f"  [OK] Viscous drag sign verified positive: fx_v = {fx_v_stream:.4f} > 0, fy_v = {fy_v_stream:.2e}")

if __name__ == '__main__':
    test_circle_sdf()
    test_flat_wall_shear_flow()
    test_cylinder_ghost_cells()
    print("\nALL STAGE 0.3 SHARP-INTERFACE TESTS PASSED!\n")
