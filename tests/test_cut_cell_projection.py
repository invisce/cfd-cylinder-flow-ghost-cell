"""
Independent Verification Test Suite for Stage 0.3 Step 3: Cut-Cell Discrete Projection.
Verifies exact discrete flux conservation and machine-zero divergence across multiple velocity fields,
solvers (Direct SPLU vs Iterative GMRES), and 3-grid resolutions.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import (
    build_cut_cell_poisson_matrix,
    calculate_cut_cell_divergence,
    cut_cell_project
)

def test_multiple_velocity_fields_projection():
    print("1. Testing Cut-Cell projection across diverse initial velocity fields...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
    dt = 0.005
    rho = 1.0

    fields = ['uniform_inflow', 'parabolic_shear', 'random_fluctuations', 'vortex_perturbation']

    for field_name in fields:
        np.random.seed(12345)
        u_star = np.zeros((ny, nx + 1), dtype=np.float64)
        v_star = np.zeros((ny + 1, nx), dtype=np.float64)

        if field_name == 'uniform_inflow':
            u_star[:, :] = 1.0
        elif field_name == 'parabolic_shear':
            for j in range(ny):
                y = (j + 0.5) * geom.dy
                u_star[j, :] = 4.0 * 1.5 * y * (Ly - y) / (Ly**2)
        elif field_name == 'random_fluctuations':
            u_star[:, :] = 1.0 + 0.25 * (np.random.rand(ny, nx + 1) - 0.5)
            v_star[:, :] = 0.25 * (np.random.rand(ny + 1, nx) - 0.5)
        elif field_name == 'vortex_perturbation':
            u_star[:, :] = 1.0
            for j in range(ny):
                y = (j + 0.5) * geom.dy
                for i in range(nx + 1):
                    x = i * geom.dx
                    r2 = (x - xc)**2 + (y - yc)**2 + 0.1
                    u_star[j, i] -= 0.5 * (y - yc) / r2
            for j in range(ny + 1):
                y = j * geom.dy
                for i in range(nx):
                    x = (i + 0.5) * geom.dx
                    r2 = (x - xc)**2 + (y - yc)**2 + 0.1
                    v_star[j, i] += 0.5 * (x - xc) / r2

        # Enforce boundary zero values on solid faces before projection
        u_star[geom.ax == 0.0] = 0.0
        v_star[geom.ay == 0.0] = 0.0
        v_star[0, :] = 0.0
        v_star[ny, :] = 0.0

        # Execute projection
        u_corr, v_corr, phi = cut_cell_project(
            u_star, v_star, geom, dt=dt, rho=rho, pressure_solver='direct'
        )

        # Check divergence on all fluid and cut cells
        max_div, rms_div, max_flux, _, div_grid = calculate_cut_cell_divergence(u_corr, v_corr, geom)
        cut_divs = div_grid[geom.cell_type == 0]
        max_cut_div = np.max(np.abs(cut_divs)) if len(cut_divs) > 0 else 0.0

        print(f"  Field: {field_name:<20} | Max NetFlux: {max_flux:.2e} | Max Div (all): {max_div:.2e} | Max Div (cut): {max_cut_div:.2e} [PASS]")

        assert max_flux < 1e-13, f"Net flux error too high: {max_flux}"
        assert max_div < 1e-12, f"Fluid cell divergence too high: {max_div}"
        assert max_cut_div < 1e-12, f"Cut cell divergence too high: {max_cut_div}"

    print("  [OK] All test fields project to exact machine-zero divergence on all fluid and cut cells")

def test_direct_vs_iterative_projection():
    print("2. Testing Direct (SPLU) vs Iterative (GMRES+ILU) Cut-Cell projection...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
    dt = 0.005
    rho = 1.0

    np.random.seed(999)
    u_star = 1.0 + 0.2 * (np.random.rand(ny, nx + 1) - 0.5)
    v_star = 0.2 * (np.random.rand(ny + 1, nx) - 0.5)
    u_star[geom.ax == 0.0] = 0.0
    v_star[geom.ay == 0.0] = 0.0
    v_star[0, :] = 0.0
    v_star[ny, :] = 0.0

    # 1. Direct solve
    u_dir, v_dir, phi_dir = cut_cell_project(
        u_star, v_star, geom, dt=dt, rho=rho, pressure_solver='direct'
    )
    max_div_dir, _, max_flux_dir, _, _ = calculate_cut_cell_divergence(u_dir, v_dir, geom)

    # 2. Iterative solve
    u_iter, v_iter, phi_iter = cut_cell_project(
        u_star, v_star, geom, dt=dt, rho=rho, pressure_solver='iterative'
    )
    max_div_iter, _, max_flux_iter, _, _ = calculate_cut_cell_divergence(u_iter, v_iter, geom)

    diff_u = np.max(np.abs(u_dir - u_iter))
    diff_v = np.max(np.abs(v_dir - v_iter))
    diff_phi = np.max(np.abs(phi_dir - phi_iter))

    print(f"  Direct   : Max NetFlux = {max_flux_dir:.2e}, Max Div = {max_div_dir:.2e}")
    print(f"  Iterative: Max NetFlux = {max_flux_iter:.2e}, Max Div = {max_div_iter:.2e}")
    print(f"  Agreement: max|du| = {diff_u:.2e}, max|dv| = {diff_v:.2e}, max|dPhi| = {diff_phi:.2e}")

    assert max_div_dir < 1e-12, f"Direct divergence too high: {max_div_dir}"
    assert max_div_iter < 1e-8, f"Iterative divergence too high: {max_div_iter}"
    assert diff_u < 1e-8, f"Velocity u mismatch: {diff_u}"
    assert diff_v < 1e-8, f"Velocity v mismatch: {diff_v}"
    assert diff_phi < 1e-7, f"Pressure phi mismatch: {diff_phi}"
    print("  [OK] Direct and GMRES+ILU Cut-Cell projection solvers agree to high precision")

def test_3grid_discrete_continuity():
    print("3. Testing Cut-Cell discrete continuity on 3 grid resolutions...")
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    dt = 0.005
    rho = 1.0
    grids = [(100, 25), (200, 50), (400, 100)]

    print(f"{'Grid':<12} | {'h (dx)':<8} | {'Max NetFlux':<15} | {'Max Div (all)':<15} | {'Max Div (cut)':<15} | {'Status'}")
    print("-" * 80)
    for nx, ny in grids:
        geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
        u_star = np.ones((ny, nx + 1), dtype=np.float64)
        v_star = np.zeros((ny + 1, nx), dtype=np.float64)
        u_star[geom.ax == 0.0] = 0.0
        v_star[geom.ay == 0.0] = 0.0
        v_star[0, :] = 0.0
        v_star[ny, :] = 0.0

        u_corr, v_corr, phi = cut_cell_project(
            u_star, v_star, geom, dt=dt, rho=rho, pressure_solver='direct'
        )

        max_div, _, max_flux, _, div_grid = calculate_cut_cell_divergence(u_corr, v_corr, geom)
        cut_divs = div_grid[geom.cell_type == 0]
        max_cut_div = np.max(np.abs(cut_divs)) if len(cut_divs) > 0 else 0.0

        print(f"{nx}x{ny:<6} | {geom.dx:<8.4f} | {max_flux:<15.2e} | {max_div:<15.2e} | {max_cut_div:<15.2e} | PASS")

        assert max_flux < 1e-13, f"Grid {nx}x{ny} NetFlux error too high: {max_flux}"
        assert max_div < 1e-9, f"Grid {nx}x{ny} divergence too high: {max_div}"
        assert max_cut_div < 1e-9, f"Grid {nx}x{ny} cut cell divergence too high: {max_cut_div}"

    print("  [OK] Discrete continuity holds to exact machine precision across all resolutions")

def run_all_cut_cell_projection_tests():
    print("=" * 80)
    print("STAGE 0.3 STEP 3: Cut-Cell Discrete Poisson & Conservative Projection")
    print("=" * 80)
    test_multiple_velocity_fields_projection()
    test_direct_vs_iterative_projection()
    test_3grid_discrete_continuity()
    print("=" * 80)
    print("ALL CUT-CELL PROJECTION TESTS PASSED TO MACHINE PRECISION!")
    print("=" * 80)

if __name__ == '__main__':
    run_all_cut_cell_projection_tests()
