"""
Focused regression checks for the MAC solver; exit code 0 on success.
Use --artifacts to also recompute the saved benchmark/study evidence.

Asserts:
  1. Full-domain fluid divergence RMS and Max within machine tolerance (< 1e-12)
  2. Cylinder-edge fluid divergence RMS within machine tolerance (< 1e-12)
  3. Natural global mass conservation (inlet/outlet flux mismatch < 1e-12) without scaling
  4. Positive viscous drag force (fx_v > 0)
  5. Parabolic inflow profile (Schäfer & Turek benchmark configuration) natural flux balance (< 1e-12)
  6. Mathematical equivalence between C++ OpenMP and Numba fallback force calculations (< 1e-14)
  7. Discrete divergence kernel equivalence between C++ and pure NumPy (< 1e-14)
  8. Exact order-2 recovery of the ASME GCI procedure on synthetic benchmark data
  9. Wall location, analytical diffusion order, discrete projection identity, input errors,
     cycle analysis and split-run equivalence. Saved artifacts are optional.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))

import sys
import numpy as np
from main import MAC_CylinderFlow
from mac_solver import calculate_mac_divergence
from forces import calculate_mac_forces, _calculate_mac_forces_numba
import mac_engine
from validation.convergence_study import calculate_gci

def test_uniform_mac_solver():
    solver = MAC_CylinderFlow(nx=400, ny=100, verbose=False, boundary='staircase', transport='upwind')
    solver.solve(n_steps=50, dt=0.002, save_artifacts=False, verbose=False)
    
    u, v, p, dx, dy, mask = solver.u, solver.v, solver.p, solver.dx, solver.dy, solver.mask
    ny, nx = mask.shape
    
    # 1. Divergence across all fluid cells
    max_div, sum_div2, count, core_div2, core_count = calculate_mac_divergence(u, v, dx, dy, mask)
    div_rms = np.sqrt(sum_div2 / count)
    assert div_rms < 1e-12, f"Fluid RMS divergence exceeded tolerance: {div_rms}"
    assert max_div < 1e-11, f"Fluid Max divergence exceeded tolerance: {max_div}"
    
    # 2. Cylinder-edge divergence
    sum_div2_edges = 0.0
    count_edges = 0
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                is_edge = False
                if i > 0 and mask[j, i - 1]: is_edge = True
                if i < nx - 1 and mask[j, i + 1]: is_edge = True
                if j > 0 and mask[j - 1, i]: is_edge = True
                if j < ny - 1 and mask[j + 1, i]: is_edge = True
                if is_edge:
                    div = (u[j, i + 1] - u[j, i]) / dx + (v[j + 1, i] - v[j, i]) / dy
                    sum_div2_edges += div**2
                    count_edges += 1
    
    assert count_edges > 0, "No cylinder edge cells found"
    div_edge_rms = np.sqrt(sum_div2_edges / count_edges)
    assert div_edge_rms < 1e-12, f"Cylinder edge RMS divergence exceeded tolerance: {div_edge_rms}"
    
    # 3. Natural global mass conservation
    inlet_flux = np.sum(u[:, 0]) * dy
    outlet_flux = np.sum(u[:, nx]) * dy
    rel_flux_mismatch = abs(inlet_flux - outlet_flux) / inlet_flux
    assert rel_flux_mismatch < 1e-12, f"Inlet/outlet flux mismatch exceeded tolerance: {rel_flux_mismatch}"
    
    # 4. Positive viscous drag
    fx_p, fy_p, fx_v, fy_v = calculate_mac_forces(u, v, p, mask, dx, dy, solver.nu * solver.rho)
    assert fx_v > 0.0, f"Viscous drag force must be strictly positive, got {fx_v}"
    assert np.isfinite(fx_p) and np.isfinite(fy_p), "Forces must be finite"
    
    # 5. C++ vs Numba force calculation equivalence
    mu = solver.nu * solver.rho
    f_cpp = mac_engine.calculate_mac_forces_cpp(u, v, p, mask, dx, dy, mu)
    f_nb = _calculate_mac_forces_numba(u, v, p, mask, dx, dy, mu)
    for a, b in zip(f_cpp, f_nb):
        assert abs(a - b) < 1e-14, f"C++ vs Numba force mismatch: {a} vs {b}"

def test_parabolic_inflow_conservation():
    solver = MAC_CylinderFlow(nx=200, ny=50, inflow='parabolic', verbose=False)
    solver.solve(n_steps=30, dt=0.002, save_artifacts=False, verbose=False)
    
    in_flux = np.sum(solver.u[:, 0]) * solver.dy
    out_flux = np.sum(solver.u[:, solver.nx]) * solver.dy
    mismatch = abs(out_flux - in_flux) / in_flux
    assert mismatch < 1e-12, f"Parabolic inflow flux mismatch exceeded tolerance: {mismatch}"
    
    max_div, sum_div2, count, _, _ = calculate_mac_divergence(
        solver.u, solver.v, solver.dx, solver.dy, solver.mask
    )
    div_rms = np.sqrt(sum_div2 / count)
    assert div_rms < 1e-12, f"Parabolic inflow RMS divergence exceeded tolerance: {div_rms}"

def test_divergence_kernel_equivalence():
    solver = MAC_CylinderFlow(nx=200, ny=50, verbose=False)
    solver.solve(n_steps=10, dt=0.002, save_artifacts=False, verbose=False)
    
    max_div, sum_div2, count, _, _ = calculate_mac_divergence(
        solver.u, solver.v, solver.dx, solver.dy, solver.mask
    )
    
    div_np = np.zeros_like(solver.p)
    for j in range(solver.ny):
        for i in range(solver.nx):
            if not solver.mask[j, i]:
                div_np[j, i] = (solver.u[j, i + 1] - solver.u[j, i]) / solver.dx + \
                               (solver.v[j + 1, i] - solver.v[j, i]) / solver.dy
                               
    assert abs(max_div - np.max(np.abs(div_np))) < 1e-14
    assert abs(sum_div2 - np.sum(div_np**2)) < 1e-14
    assert count == int(np.sum(~solver.mask))

def test_gci_known_order():
    f_exact = 2.5000
    f1 = f_exact + 3.0 * 0.01**2
    f2 = f_exact + 3.0 * 0.02**2
    f3 = f_exact + 3.0 * 0.04**2
    res = calculate_gci(f1, f2, f3, r=2.0)
    assert abs(res['p'] - 2.0) < 1e-10, f"GCI failed to recover order 2: {res['p']}"
    assert abs(res['f_ext'] - f_exact) < 1e-10, f"GCI failed to recover exact continuum value: {res['f_ext']}"

def test_solid_tangential_no_slip():
    # Linear tangential shear has zero Laplacian, including next to a flat
    # staircase wall. The no-slip plane is halfway to the solid-cell centre.
    n = 8
    for vertical in (False, True):
        mask = np.zeros((n, n), dtype=bool)
        u, v, p = np.zeros((n, n + 1)), np.zeros((n + 1, n)), np.zeros((n, n))
        if vertical:
            mask[:, :3] = True
            v[:, 3:] = np.arange(3, n) - 2.5
        else:
            mask[:3, :] = True
            u[3:, :] = (np.arange(3, n) - 2.5)[:, None]
        us, vs, rhs = np.zeros_like(u), np.zeros_like(v), np.zeros_like(p)
        mac_engine.predictor_rhs_cpp(u, v, us, vs, rhs, p, mask,
                                    u[:, 0].copy(), 1., 1., .01, .1, 1., False)
        actual = vs[4, 3] if vertical else us[3, 4]
        assert abs(actual - .5) < 1e-14, f"Wall location inconsistent: {actual} != 0.5"

def main():
    test_uniform_mac_solver()
    test_parabolic_inflow_conservation()
    test_divergence_kernel_equivalence()
    test_gci_known_order()
    test_solid_tangential_no_slip()
    test_analysis_edge_cases()
    test_projection_identity()
    test_diffusion_order()
    test_input_and_continuation()
    test_circular_projection()
    test_centered_transport()
    test_pressure_solver_modes()
    from tests.test_nondimensional_equivalence import verify_nondimensional_equivalence
    verify_nondimensional_equivalence()
    from tests.test_ghost_cell_sharp import test_circle_sdf, test_flat_wall_shear_flow, test_cylinder_ghost_cells
    test_circle_sdf()
    test_flat_wall_shear_flow()
    test_cylinder_ghost_cells()
    from tests.test_ghost_cell_integration import test_ghost_cell_solver_integration
    test_ghost_cell_solver_integration()
    from tests.test_cut_cell_geom import run_all_cut_cell_geometry_tests
    run_all_cut_cell_geometry_tests()
    from tests.test_cut_cell_projection import run_all_cut_cell_projection_tests
    run_all_cut_cell_projection_tests()
    from tests.test_cut_cell_integration import run_all_cut_cell_integration_tests
    run_all_cut_cell_integration_tests()
    from tests.test_cut_cell_momentum_and_forces import run_all_momentum_and_force_tests
    run_all_momentum_and_force_tests()
    from tests.test_cut_cell_wall_gradient import test_wall_gradient, test_variable_step_ab2, test_cut_cell_pressure_and_provenance
    test_wall_gradient()
    test_variable_step_ab2()
    test_cut_cell_pressure_and_provenance()
    from tests.test_wall_face_average_reconstruction import test_wall_averages, test_couette_averages
    test_wall_averages()
    test_couette_averages()
    from tests.test_cut_cell_pressure import test_surface_pressure, test_manufactured_pressure_solve
    test_surface_pressure()
    test_manufactured_pressure_solve()
    from tests.test_cut_cell_pressure_backend import test_pressure_backends
    from tests.test_cut_cell_time_restart import (test_time_and_restart,
        test_noncommuting_pressure_recovery, test_fine_grid_pressure_stability)
    test_pressure_backends()
    test_time_and_restart()
    test_noncommuting_pressure_recovery()
    test_fine_grid_pressure_stability()
    from tests.test_pressure_schur import test_pressure_schur
    test_pressure_schur()
    from tests.test_staggered_wall_flux import (test_harmonic_viscous_consistency,
        test_projected_viscous_spectrum, test_covered_faces_are_not_velocity_dofs)
    test_harmonic_viscous_consistency()
    test_projected_viscous_spectrum()
    test_covered_faces_are_not_velocity_dofs()
    from tests.test_viscous_interface import test_viscous_interface
    test_viscous_interface()
    from tests.test_cut_cell_outlet_balance import test_natural_outlet_balance
    test_natural_outlet_balance()
    from tests.test_pressure_sensor_audit import run as test_pressure_sensor_statistics
    test_pressure_sensor_statistics()
    from tests.test_unsteady_momentum_budget import run as test_unsteady_control_volume
    test_unsteady_control_volume()
    from tests.test_periodic_window_audit import run as test_common_cycle_statistics
    test_common_cycle_statistics()
    from tests.test_stage0_independence import test_independence, test_periodic_common_window
    test_independence()
    test_periodic_common_window()
    from tests.test_pressure_graph_audit import run as test_pressure_graph
    test_pressure_graph()
    from tests.test_cycle_momentum_audit import run as test_phase_momentum
    test_phase_momentum()
    if '--artifacts' in sys.argv:
        from tests.test_ghost_cell_convergence import verify_ghost_cell_boundary_convergence
        verify_ghost_cell_boundary_convergence()
        from tests.test_cut_cell_convergence import verify_cut_cell_3grid_convergence, run_cut_cell_vs_ghost_cell_comparison
        verify_cut_cell_3grid_convergence()
        run_cut_cell_vs_ghost_cell_comparison(n_steps=1000)
        from validation.benchmark_schafer_turek import verify_saved_schafer_turek
        from validation.convergence_study import verify_saved_convergence
        from validation.sensitivity_study import verify_saved_sensitivity
        verify_saved_schafer_turek()
        verify_saved_convergence()
        verify_saved_sensitivity()
        test_archive_rejects_changed_history()
    print('PASS: conservation, wall stencil, diffusion order, projection, analysis, inputs and continuation')


def test_analysis_edge_cases():
    from postprocessing.analysis import periodic_metrics
    t = np.arange(0., 60., .01)
    cl = .2 + np.sin(2*np.pi*.3*t) + .1*np.sin(4*np.pi*.3*t)
    m = periodic_metrics(t, 2 + .1*np.cos(4*np.pi*.3*t), cl, 10.)
    assert abs(m['strouhal']-.12) < 1e-6 and m['stationary'] and m['cycles'] >= 10
    m = periodic_metrics(t, np.ones_like(t), (1+t/60)*cl, 10.)
    assert not m['stationary']
    m = periodic_metrics(t[:30], t[:30]+2, cl[:30], 0.)
    assert np.isnan(m['strouhal']) and not m['stationary']
    for values in ((1, 2, 1), (0, 1, 4), (1, 1, 1), (3, 2, 1.5), (1, 2, 3.05)):
        assert calculate_gci(*values)['f_ext'] is None
    for bad in (t[::-1], t*0, t+np.nan):
        try:
            periodic_metrics(bad, cl, cl, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid history accepted')


def test_projection_identity():
    from mac_solver import build_poisson_matrix
    n = 12
    rng = np.random.default_rng(2026)
    mask = np.zeros((n, n), dtype=bool)
    mask[4:7, 4:7] = True
    phi = rng.normal(size=(n, n))
    phi[mask] = 0
    u, v = np.zeros((n, n+1)), np.zeros((n+1, n))
    mac_engine.corrector_cpp(u, v, u.copy(), v.copy(), phi, mask, np.zeros(n), .2, .1, .01, 1., False)
    div = (u[:, 1:]-u[:, :-1])/.2+(v[1:]-v[:-1])/.1
    expected = -.01*build_poisson_matrix(n, n, .2, .1, mask).dot(phi.ravel()).reshape(n, n)
    np.testing.assert_allclose(div, expected, atol=1e-12)


def test_diffusion_order():
    # u=sin(pi*y), v=0 is a no-slip shear diffusion eigenmode.
    # Test the actual C++ predictor against its analytic time derivative.
    errors = []
    for n in (16, 32):
        y = (np.arange(n)+.5)/n
        u = np.broadcast_to(np.sin(np.pi*y)[:, None], (n, n+1)).copy()
        v, p, mask = np.zeros((n+1, n)), np.zeros((n, n)), np.zeros((n, n), dtype=bool)
        us, vs, rhs = np.zeros_like(u), np.zeros_like(v), np.zeros_like(p)
        mac_engine.predictor_rhs_cpp(u, v, us, vs, rhs, p, mask, u[:, 0].copy(),
                                    1/n, 1/n, .001, .1, 1., False)
        actual = (us[:, n//2]-u[:, n//2])/.001
        errors.append(np.linalg.norm(actual + .1*np.pi**2*np.sin(np.pi*y))/np.sqrt(n))
    order = np.log2(errors[0]/errors[1])
    assert 1.95 < order < 2.05, f'Diffusion observed order: {order}'


def test_input_and_continuation():
    a = MAC_CylinderFlow(nx=80, ny=20, verbose=False)
    b = MAC_CylinderFlow(nx=80, ny=20, verbose=False)
    a.solve(n_steps=20, save_artifacts=False)
    b.solve(n_steps=10, save_artifacts=False)
    t, _, _ = b.solve(n_steps=10, save_artifacts=False)
    assert t[0] > .02 and abs(t[-1]-.04) < 1e-12
    np.testing.assert_allclose(a.u, b.u, atol=1e-12)
    for u in (a.u[:, ::-1], a.u[:-1], a.u.astype(np.float32)):
        try:
            mac_engine.calculate_mac_divergence_cpp(u, a.v, a.mask, a.dx, a.dy)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe C++ buffer accepted')
    for kwargs in ({'dt': 0}, {'n_steps': 0}, {'dt': np.nan}, {'dt': 10}):
        try:
            a.solve(save_artifacts=False, **kwargs)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError(f'Invalid solve accepted: {kwargs}')


def test_archive_rejects_changed_history():
    import tempfile
    from pathlib import Path
    from validation.benchmark_schafer_turek import verify_saved_schafer_turek
    with np.load(os.path.join(os.path.dirname(__file__), '../data/schafer_turek_results.npz'), allow_pickle=False) as saved:
        altered = {k: saved[k] for k in saved.files}
    altered['cd'][altered['t'] > 28] += .05
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'altered.npz'
        np.savez(path, **altered)
        try:
            verify_saved_schafer_turek(path)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed raw history passed archive verification')


def test_circular_projection():
    from immersed_boundary import CircularProjection
    from mac_solver import build_poisson_matrix
    ib = CircularProjection(80, 24, .025, .025, .45, .30, .15)
    x, y = ib.markers.T
    np.testing.assert_allclose((x-.45)**2+(y-.30)**2, (.15/2)**2, atol=1e-15)
    # Partition of unity and first moments protect force and torque transfer.
    xu, yu = np.meshgrid((np.arange(80)+1)*.025, (np.arange(24)+.5)*.025)
    xv, yv = np.meshgrid((np.arange(80)+.5)*.025, (np.arange(23)+1)*.025)
    np.testing.assert_allclose(ib.E @ np.ones(ib.E.shape[1]), 1., atol=1e-14)
    np.testing.assert_allclose(ib.E @ np.r_[xu.ravel(), xv.ravel()], np.tile(x, 2), atol=1e-14)
    np.testing.assert_allclose(ib.E @ np.r_[yu.ravel(), yv.ravel()], np.tile(y, 2), atol=1e-14)
    # Independently assembled legacy pressure operator must match this full-domain block.
    A = build_poisson_matrix(80, 24, .025, .025, np.zeros((24, 80), bool))
    np.testing.assert_allclose((ib.D @ ib.correction[:, :ib.np] + A).data, 0., atol=1e-10)
    rng = np.random.default_rng(7)
    u, v, p = np.zeros((24, 81)), np.zeros((25, 80)), np.zeros((24, 80))
    us, vs = rng.normal(size=u.shape), rng.normal(size=v.shape)
    us[:, 0] = 1.
    vs[[0, -1]] = 0.
    force = ib.project(us, vs, u, v, p, .001, 1.)
    # Independent full coupled sparse solve checks the fast FFT/Schur factorization.
    from scipy.sparse.linalg import spsolve
    rhs = ib.C @ ib.pack(us, vs)
    rhs[:ib.np].reshape(24, 80)[:, 0] -= us[:, 0]/.025
    z = spsolve((ib.C @ ib.correction).tocsc(), rhs)
    expected = ib.pack(us, vs)-ib.correction @ z
    np.testing.assert_allclose(ib.pack(u, v), expected, atol=1e-10)
    div = (u[:, 1:]-u[:, :-1])/.025+(v[1:]-v[:-1])/.025
    assert np.max(abs(div)) < 1e-9 and ib.last_slip < 1e-10
    np.testing.assert_allclose(u[:, -1].sum(), u[:, 0].sum(), atol=1e-10)
    assert np.isfinite(force).all()
    # Reprojection of an admissible velocity must change neither velocity nor force.
    old = ib.pack(u, v)
    ib.project(u.copy(), v.copy(), u, v, p, .001, 1.)
    np.testing.assert_allclose(ib.pack(u, v), old, atol=1e-10)
    np.testing.assert_allclose(ib.last_force, 0., atol=1e-8)
    # A manufactured constraint impulse has a known reaction and pressure.
    # Starting/ending at rest makes the interior momentum correction exactly zero.
    imposed = rng.normal(size=ib.C.shape[0])
    qstar = ib.correction @ imposed
    u.fill(0.)
    v.fill(0.)
    us.fill(0.)
    vs.fill(0.)
    us[:, 1:] = qstar[:ib.nu].reshape(24, 80)
    vs[1:-1] = qstar[ib.nu:].reshape(23, 80)
    ib.project(us, vs, u, v, p, .02, 1.7)
    expected_force = 1.7*.025**2/.02*imposed[ib.np:].reshape(2, ib.n_markers).sum(axis=1)
    np.testing.assert_allclose(ib.pack(u, v), 0., atol=1e-9)
    np.testing.assert_allclose(ib.last_force, expected_force, atol=1e-8)
    np.testing.assert_allclose(p, -1.7/.02*imposed[:ib.np].reshape(24, 80), atol=1e-8)


def test_centered_transport():
    # Transverse sine wave advected by constant u: exact derivative -2*pi*cos(2*pi*x).
    errors = {False: [], True: []}
    for n in (32, 64):
        x = (np.arange(n)+.5)/n
        u = np.ones((8, n+1))
        v = np.broadcast_to(np.sin(2*np.pi*x), (9, n)).copy()
        p, mask = np.zeros((8, n)), np.zeros((8, n), dtype=bool)
        for centered in errors:
            us, vs, rhs = np.zeros_like(u), np.zeros_like(v), np.zeros_like(p)
            mac_engine.predictor_rhs_cpp(u, v, us, vs, rhs, p, mask, np.ones(8),
                                        1/n, .125, .001, 0., 1., False, centered)
            actual = (vs[4, 2:-2]-v[4, 2:-2])/.001
            errors[centered].append(np.sqrt(np.mean((actual+2*np.pi*np.cos(2*np.pi*x[2:-2]))**2)))
            # Away from domain boundaries, linear centred transport introduces
            # no semidiscrete damping; upwind removes energy from this wave.
            energy_rate = np.sum(v[4, 2:-2]*actual)
            if centered:
                assert abs(energy_rate) < 1e-10
            else:
                assert energy_rate < 0
    assert errors[True][-1] < .1*errors[False][-1]
    assert 1.8 < np.log2(errors[True][0]/errors[True][1]) < 2.2


def test_pressure_solver_modes():
    # 1. Immersed + iterative must raise NotImplementedError
    try:
        MAC_CylinderFlow(nx=80, ny=20, boundary='immersed', pressure_solver='iterative')
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError for immersed + iterative")
    
    # 2. Staircase + direct solver
    solver_dir = MAC_CylinderFlow(nx=80, ny=20, boundary='staircase', pressure_solver='direct', verbose=False)
    solver_dir.solve(n_steps=5, dt=0.005, save_artifacts=False, verbose=False)
    max_div_dir, _, _, _, _ = calculate_mac_divergence(solver_dir.u, solver_dir.v, solver_dir.dx, solver_dir.dy, solver_dir.mask)
    assert max_div_dir < 1e-10, f"Direct solver divergence exceeded tolerance: {max_div_dir}"
    
    # 3. Staircase + iterative (GMRES + ILU)
    solver_iter = MAC_CylinderFlow(nx=80, ny=20, boundary='staircase', pressure_solver='iterative', verbose=False)
    solver_iter.solve(n_steps=5, dt=0.005, save_artifacts=False, verbose=False)
    max_div_iter, _, _, _, _ = calculate_mac_divergence(solver_iter.u, solver_iter.v, solver_iter.dx, solver_iter.dy, solver_iter.mask)
    assert max_div_iter < 1e-10, f"Iterative solver divergence exceeded tolerance: {max_div_iter}"
    
    # Solutions must match closely
    np.testing.assert_allclose(solver_dir.u, solver_iter.u, atol=1e-6)
    np.testing.assert_allclose(solver_dir.v, solver_iter.v, atol=1e-6)

if __name__ == '__main__':
    main()
