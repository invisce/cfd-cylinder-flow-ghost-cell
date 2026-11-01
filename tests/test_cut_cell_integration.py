"""
Independent Verification Test Suite for Stage 0.3: Cut-Cell Full Solver Integration.
Verifies:
  1. MAC_CylinderFlow integration with boundary='cut_cell'.
  2. Direct SPLU vs Iterative GMRES+ILU solver consistency.
  3. Machine-zero divergence and exact global flux conservation.
  4. Centered transport + AB2 time-stepping with Cut-Cell boundary.
  5. Constant-pressure zero spurious force invariance.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from main import MAC_CylinderFlow
from cut_cell_solver import calculate_cut_cell_divergence

def test_cut_cell_direct_vs_iterative():
    print("1. Testing Cut-Cell direct (SPLU) vs iterative (GMRES+ILU) solver consistency...")
    nx, ny = 100, 25
    solver_dir = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5,
        boundary='cut_cell', pressure_solver='direct', verbose=False
    )
    t_dir, cd_dir, cl_dir = solver_dir.solve(n_steps=10, dt=0.005, save_artifacts=False, verbose=False)

    solver_iter = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5,
        boundary='cut_cell', pressure_solver='iterative', verbose=False
    )
    t_iter, cd_iter, cl_iter = solver_iter.solve(n_steps=10, dt=0.005, save_artifacts=False, verbose=False)

    # 1. Divergence checks
    max_div_dir, _, max_flux_dir, _, _ = calculate_cut_cell_divergence(solver_dir.u, solver_dir.v, solver_dir.cut_geom)
    max_div_iter, _, max_flux_iter, _, _ = calculate_cut_cell_divergence(solver_iter.u, solver_iter.v, solver_iter.cut_geom)

    assert max_div_dir < 1e-13, f"Direct solver divergence too high: {max_div_dir}"
    assert max_div_iter < 1e-8, f"Iterative solver divergence too high: {max_div_iter}"

    # 2. Field consistency
    diff_u = np.max(np.abs(solver_dir.u - solver_iter.u))
    diff_v = np.max(np.abs(solver_dir.v - solver_iter.v))
    diff_p = np.max(np.abs(solver_dir.p - solver_iter.p))

    assert diff_u < 1e-6, f"u discrepancy too high: {diff_u}"
    assert diff_v < 1e-6, f"v discrepancy too high: {diff_v}"
    assert diff_p < 1e-5, f"p discrepancy too high: {diff_p}"

    print(f"  Direct Max Div   : {max_div_dir:.2e}")
    print(f"  Iterative Max Div: {max_div_iter:.2e}")
    print(f"  Field Differences: max|du|={diff_u:.2e}, max|dv|={diff_v:.2e}, max|dp|={diff_p:.2e}")
    print("  [OK] Direct and iterative Cut-Cell solvers match closely and satisfy continuity")

def test_cut_cell_mass_conservation():
    print("2. Testing Cut-Cell natural global mass conservation...")
    nx, ny = 100, 25
    solver = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5,
        boundary='cut_cell', pressure_solver='direct', verbose=False
    )
    solver.solve(n_steps=20, dt=0.005, save_artifacts=False, verbose=False)

    in_flux = np.sum(solver.u[:, 0]) * solver.dy
    out_flux = np.sum(solver.u[:, solver.nx]) * solver.dy
    flux_err = abs(out_flux - in_flux) / in_flux

    max_div, rms_div, max_flux, _, _ = calculate_cut_cell_divergence(solver.u, solver.v, solver.cut_geom)

    assert flux_err < 1e-14, f"Global flux error too high: {flux_err}"
    assert max_div < 1e-13, f"Local divergence too high: {max_div}"
    assert max_flux < 1e-14, f"Net cell flux too high: {max_flux}"

    print(f"  Inlet flux       : {in_flux:.10f}")
    print(f"  Outlet flux      : {out_flux:.10f}")
    print(f"  Flux Error       : {flux_err:.2e} (Machine Zero)")
    print(f"  Max Local Div    : {max_div:.2e}")
    print("  [OK] Natural mass conservation satisfied to machine precision without scaling")

def test_cut_cell_centered_transport_ab2():
    print("3. Testing Cut-Cell with centered transport and AB2 time integration...")
    nx, ny = 100, 25
    solver = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5,
        boundary='cut_cell', transport='centered', pressure_solver='direct', verbose=False
    )
    t, cd, cl = solver.solve(n_steps=20, dt=0.002, save_artifacts=False, verbose=False)

    max_div, _, _, _, _ = calculate_cut_cell_divergence(solver.u, solver.v, solver.cut_geom)
    assert max_div < 1e-13, f"AB2 centered Cut-Cell divergence too high: {max_div}"
    assert np.all(np.isfinite(solver.u))
    assert np.all(np.isfinite(solver.v))
    assert np.all(np.isfinite(solver.p))
    print(f"  Centered AB2 Max Div: {max_div:.2e}, Final Cd: {cd[-1]:.4f}")
    print("  [OK] Centered transport and AB2 integration operate stably with Cut-Cell boundary")

def test_cut_cell_constant_pressure_force():
    print("4. Testing Cut-Cell zero spurious force under uniform pressure...")
    nx, ny = 100, 25
    solver = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.035, yc=2.455,
        boundary='cut_cell', verbose=False
    )
    # Impose uniform pressure field P = 101325.0
    p_const = np.full((ny, nx), 101325.0)
    u_zero = np.zeros_like(solver.u)
    v_zero = np.zeros_like(solver.v)

    fx, fy, fx_p, fy_p, fx_v, fy_v = solver.cc.extract_true_surface_forces(
        u_zero, v_zero, p_const, 0.01
    )

    assert abs(fx_p) < 1e-10, f"Spurious fx_p under uniform pressure: {fx_p}"
    assert abs(fy_p) < 1e-10, f"Spurious fy_p under uniform pressure: {fy_p}"
    assert abs(fx_v) < 1e-14, f"Spurious fx_v under zero velocity: {fx_v}"
    assert abs(fy_v) < 1e-14, f"Spurious fy_v under zero velocity: {fy_v}"
    print(f"  Uniform pressure force on body: Fx = {fx_p:.2e}, Fy = {fy_p:.2e}")
    print("  [OK] Invariant zero force verified under uniform pressure")

def run_all_cut_cell_integration_tests():
    print("=" * 75)
    print("STAGE 0.3 VERIFICATION: Cut-Cell Full Solver Integration")
    print("=" * 75)
    test_cut_cell_direct_vs_iterative()
    test_cut_cell_mass_conservation()
    test_cut_cell_centered_transport_ab2()
    test_cut_cell_constant_pressure_force()
    print("=" * 75)
    print("ALL CUT-CELL SOLVER INTEGRATION TESTS PASSED CLEANLY!")
    print("=" * 75)

if __name__ == '__main__':
    run_all_cut_cell_integration_tests()
