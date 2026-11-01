"""
Integration test for Stage 0.3: Ghost-Cell boundary connected directly to the main MAC solver.
Tests fluid mass conservation, velocity boundary conditions, and direct/iterative solvers.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from main import MAC_CylinderFlow
from mac_solver import calculate_mac_divergence

def test_ghost_cell_solver_integration():
    print("Testing Ghost-Cell integration with main solver...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    D = 1.0
    xc, yc = 5.0, 2.5
    dt = 0.005
    n_steps = 15

    # 1. Run with Direct pressure solver
    print("  Running Ghost-Cell with direct pressure solver...")
    solver_gc_dir = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, D=D, xc=xc, yc=yc, nu=0.01,
        boundary='ghost_cell', pressure_solver='direct', verbose=False
    )
    t, cd_dir, cl_dir = solver_gc_dir.solve(
        n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False
    )

    # Check discrete divergence on fluid cells
    max_div_dir, div2, count, _, _ = calculate_mac_divergence(
        solver_gc_dir.u, solver_gc_dir.v, solver_gc_dir.dx, solver_gc_dir.dy, solver_gc_dir.mask
    )
    div_rms_dir = np.sqrt(div2 / count)
    assert max_div_dir < 1e-10, f"Ghost-cell direct max divergence exceeded tolerance: {max_div_dir}"
    print(f"  [OK] Ghost-Cell Direct: Div Max = {max_div_dir:.2e}, RMS = {div_rms_dir:.2e} (Machine Zero)")

    # 2. Run with Iterative GMRES+ILU pressure solver
    print("  Running Ghost-Cell with iterative (GMRES+ILU) pressure solver...")
    solver_gc_iter = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=Lx, Ly=Ly, D=D, xc=xc, yc=yc, nu=0.01,
        boundary='ghost_cell', pressure_solver='iterative', verbose=False
    )
    t, cd_iter, cl_iter = solver_gc_iter.solve(
        n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False
    )

    max_div_iter, div2_iter, count_iter, _, _ = calculate_mac_divergence(
        solver_gc_iter.u, solver_gc_iter.v, solver_gc_iter.dx, solver_gc_iter.dy, solver_gc_iter.mask
    )
    div_rms_iter = np.sqrt(div2_iter / count_iter)
    assert max_div_iter < 1e-10, f"Ghost-cell iterative max divergence exceeded tolerance: {max_div_iter}"
    print(f"  [OK] Ghost-Cell Iterative: Div Max = {max_div_iter:.2e}, RMS = {div_rms_iter:.2e} (Machine Zero)")

    # 3. Solvers must agree to high precision
    diff_u = np.max(np.abs(solver_gc_dir.u - solver_gc_iter.u))
    diff_v = np.max(np.abs(solver_gc_dir.v - solver_gc_iter.v))
    diff_p = np.max(np.abs(solver_gc_dir.p - solver_gc_iter.p))
    assert diff_u < 1e-6, f"Direct vs Iterative u mismatch: {diff_u}"
    assert diff_v < 1e-6, f"Direct vs Iterative v mismatch: {diff_v}"
    assert diff_p < 1e-5, f"Direct vs Iterative p mismatch: {diff_p}"
    print(f"  [OK] Direct vs Iterative solver agreement: max|du| = {diff_u:.2e}, max|dp| = {diff_p:.2e}")

    # 4. Check that drag coefficient is positive and finite
    assert all(np.isfinite(cd_dir)) and all(np.isfinite(cl_dir))
    assert cd_dir[-1] > 0.0, f"Unphysical drag coefficient: {cd_dir[-1]}"
    print(f"  [OK] True surface drag coefficient at step {n_steps}: Cd = {cd_dir[-1]:.4f} > 0")

    # 5. Measure boundary slip on physical circle r = R
    from scipy.interpolate import RegularGridInterpolator
    gc = solver_gc_dir.gc
    u_interp = RegularGridInterpolator((gc.yu, gc.xu), solver_gc_dir.u, bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((gc.yv, gc.xv), solver_gc_dir.v, bounds_error=False, fill_value=0.0)
    pts = np.column_stack((gc.surf_y, gc.surf_x))
    u_b = u_interp(pts)
    v_b = v_interp(pts)
    speed_b = np.sqrt(u_b**2 + v_b**2)
    max_slip = np.max(speed_b)
    mean_slip = np.mean(speed_b)
    print(f"  [OK] Physical boundary slip on true cylinder: max = {max_slip:.4f}, mean = {mean_slip:.4f}")
    assert max_slip < 0.40, f"Boundary slip on physical circle too large: {max_slip}"
    assert mean_slip < 0.15, f"Mean boundary slip on physical circle too large: {mean_slip}"

    print("\nSUCCESS: Ghost-Cell boundary is fully integrated and validated in main solver!\n")

if __name__ == '__main__':
    test_ghost_cell_solver_integration()
