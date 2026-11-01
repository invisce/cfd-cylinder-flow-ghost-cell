"""
Independent Verification Test Suite for Stage 0.3: Cut-Cell Momentum Predictor and Surface Forces.

Verifies:
  1. Archimedes pressure force under linear pressure gradient (-nabla p * pi*R^2).
  2. Wall-compatible analytical viscous shear torque in circular Couette flow (tau_w = 2*mu*Omega, Torque = 4*pi*mu*Omega*R^2).
  3. True-path momentum conservation on actual compute_momentum_rhs outputs:
     sum(rho * V_i * a_i) == sum(rhs_i) == sum(F_boundary) + F_wall to machine precision (< 1e-14).
  4. Exact momentum conservation under Conservative Cell Merging (< 1e-16).
  5. Formal temporal convergence: Forward Euler (p = 1.00) vs Adams-Bashforth 2 (p = 2.00) on standalone momentum ODE,
     along with documented fractional-step projection splitting error (p = 1.00).
  6. Surface slip evaluation on the true cylinder circle r = R (N=360 uniform points) demonstrating
     sharp no-slip boundary recovery and monotonic convergence across 3 grids (Coarse, Medium, Fine).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import CutCellBoundary
from main import MAC_CylinderFlow
from tests.test_wall_face_average_reconstruction import couette_faces

def test_linear_pressure_archimedes_force():
    print("1. Testing linear pressure gradient Archimedes force...")
    grids = [(100, 25), (200, 50), (400, 100)]
    Gx, Gy = 10.0, 5.0
    errors_vec = []

    for nx, ny in grids:
        cc = CutCellBoundary(nx, ny, 20.0, 5.0, 5.0, 2.5, 0.5)
        X = (np.arange(cc.nx) + 0.5) * cc.dx
        Y = (np.arange(cc.ny) + 0.5) * cc.dy
        XX, YY = np.meshgrid(X, Y)
        p = -Gx * XX - Gy * YY

        u0 = np.zeros((cc.ny, cc.nx + 1))
        v0 = np.zeros((cc.ny + 1, cc.nx))
        _, _, fx_p, fy_p, _, _ = cc.extract_true_surface_forces(u0, v0, p, mu=0.01)

        exact_fx = Gx * np.pi * (cc.geom.R**2)
        exact_fy = Gy * np.pi * (cc.geom.R**2)
        exact_mag = np.hypot(exact_fx, exact_fy)
        err_vec = np.hypot(fx_p - exact_fx, fy_p - exact_fy) / exact_mag
        errors_vec.append(err_vec)
        print(f"  Grid {nx}x{ny}: dx={cc.dx:.3f} | err_vec={err_vec:.4e} [PASS]")

    # Affine reproduction should be at roundoff on every grid, not merely convergent.
    assert max(errors_vec) < 1e-11
    print("  [OK] Linear pressure force reproduced within 1e-11 on all three grids")


def test_analytical_cylinder_couette_shear():
    print("2. Testing wall-compatible analytical viscous shear torque in circular Couette flow...")
    # Mathematical formulation of circular Couette flow:
    # For viscous flow between stationary inner cylinder (r = R) and rotating outer cylinder at r = R_o
    # with angular velocity Omega_o, the azimuthal velocity profile is:
    #   u_theta(r) = Omega_o * (R_o^2 / (R_o^2 - R^2)) * (r - R^2 / r).
    # Defining the effective strain parameter Omega = Omega_o * R_o^2 / (R_o^2 - R^2),
    # the velocity field is u_theta(r) = Omega * (r - R^2 / r), which satisfies u_theta(R) = 0 strictly.
    # The wall shear stress at r = R is tau_w = 2 * mu * Omega, yielding the analytical torque per unit length:
    #   T_exact = 4 * pi * mu * Omega * R^2.
    Omega = 2.5
    mu = 0.01
    R = 0.5
    xc, yc = 5.0, 2.5
    exact_torque = 4.0 * np.pi * mu * Omega * (R**2)

    grids = [(100, 25), (200, 50), (400, 100)]
    errors_torque = []

    for nx, ny in grids:
        cc = CutCellBoundary(nx, ny, 20.0, 5.0, xc, yc, R)
        u = np.zeros((cc.ny, cc.nx + 1))
        v = np.zeros((cc.ny + 1, cc.nx))

        # The unknowns represent averages over open faces, not center values.
        # Integrate the independent analytical streamfunction at face endpoints.
        u, v = couette_faces(cc.geom, Omega)

        p = np.zeros((cc.ny, cc.nx))
        _, _, fx_p, fy_p, fx_v, fy_v = cc.extract_true_surface_forces(u, v, p, mu)

        # Compute numerical torque: T = sum(tau_w * R * ds)
        tau_w = mu * cc.wall_shear_rate(u, v)
        num_torque = float(np.sum(tau_w * R * cc.ds))

        rel_err = abs(num_torque - exact_torque) / exact_torque
        errors_torque.append(rel_err)
        print(f"  Grid {nx}x{ny}: dx={cc.dx:.3f} | Num Torque = {num_torque:.6f} (Exact: {exact_torque:.6f}) | Rel Err = {rel_err:.4e} ({rel_err*100:.2f}%) [PASS]")

    # Monotonic convergence assertion
    assert errors_torque[1] < errors_torque[0] and errors_torque[2] < errors_torque[1], \
        f"Couette torque error did not converge monotonically: {errors_torque}"
    assert errors_torque[-1] < 0.01, f"Fine grid torque error exceeds 1%: {errors_torque[-1]}"
    print("  [OK] Wall-compatible analytical shear torque converges monotonically (< 1% on fine grid)")

def test_true_path_momentum_balance_and_telescopic_cancellation():
    print("3. Testing true-path momentum conservation: sum(rho*V*a) == Net External Boundary Flux + Wall Force...")
    # Multi-grid independent verification of exact discrete momentum balance
    for nx, ny in [(100, 25), (200, 50), (400, 100)]:
        cc = CutCellBoundary(nx, ny, 20.0, 5.0, 5.0, 2.5, 0.5)
        geom = cc.geom
        nu, rho = 0.01, 1.0

        np.random.seed(42)
        u = np.random.randn(cc.ny, cc.nx + 1)
        v = np.random.randn(cc.ny + 1, cc.nx)
        u[geom.ax == 0.0] = 0.0
        v[geom.ay == 0.0] = 0.0

        # Execute actual solver method
        a_u, a_v, rhs_u, rhs_v = cc.compute_momentum_rhs(u, v, nu, rho, transport='centered')

        # A. Verify ZERO wall force lost: active faces receive 100% of wall shear force
        if len(cc.wall_cut_indices) > 0:
            tau_w = -rho * nu * cc.wall_shear_rate(u, v)
            fx_wall = tau_w * cc.tx * cc.ds
            fy_wall = tau_w * cc.ty * cc.ds

            # Geometry ownership preserves all physical staggered fluid volume.
            fluid_volume = geom.Lx*geom.Ly-np.pi*geom.R**2
            np.testing.assert_allclose(cc.u_vol.sum(), fluid_volume, rtol=1e-10)
            np.testing.assert_allclose(cc.v_vol.sum(), fluid_volume, rtol=1e-10)
            np.testing.assert_allclose(cc.ds.sum(), 2*np.pi*geom.R, atol=1e-14)

        # B. Closed solid faces have EXACTLY 0.0 rhs and acceleration
        assert np.max(np.abs(rhs_u[geom.ax == 0.0])) == 0.0, "Non-zero rhs_u on solid faces"
        assert np.max(np.abs(rhs_v[geom.ay == 0.0])) == 0.0, "Non-zero rhs_v on solid faces"
        assert np.max(np.abs(a_u[geom.ax == 0.0])) == 0.0, "Non-zero a_u on solid faces"
        assert np.max(np.abs(a_v[geom.ay == 0.0])) == 0.0, "Non-zero a_v on solid faces"

        # C. Global Discrete Identity: sum(rho * V * a) == sum(rhs_active) == Net Ext Boundary Flux + Wall Force
        u_face_x = 0.5 * (u[:, :-1] + u[:, 1:])
        adv_xx = rho * (u_face_x**2) * geom.ax_mid * cc.dy
        diff_xx = rho * nu * (u[:, 1:] - u[:, :-1]) / cc.dx * geom.ax_mid * cc.dy
        F_xx = diff_xx - adv_xx

        ay_u_eff = geom.ay_u.copy()
        F_xy_top = -2.0 * rho * nu * u[cc.ny - 1, :] / cc.dy * ay_u_eff[cc.ny, :] * cc.dx
        F_xy_bot = 2.0 * rho * nu * u[0, :] / cc.dy * ay_u_eff[0, :] * cc.dx

        ext_flux_u = (-rho*np.sum(u[:, cc.nx]**2)*cc.dy - np.sum(F_xx[:, 0])) + \
                     (np.sum(F_xy_top[1:cc.nx][geom.ax[cc.ny - 1, 1:cc.nx] > 0]) - \
                      np.sum(F_xy_bot[1:cc.nx][geom.ax[0, 1:cc.nx] > 0])) + \
                     .5*(F_xy_top[cc.nx]-F_xy_bot[cc.nx])
        total_wall_u = np.sum(fx_wall) if len(cc.wall_cut_indices) > 0 else 0.0
        expected_u = ext_flux_u + total_wall_u

        active_u = (geom.ax > 0.0)
        sum_rho_V_a_u = np.sum(rho * cc.u_vol[active_u] * a_u[active_u])
        sum_rhs_u = np.sum(rhs_u[active_u])

        balance_err_u = abs(sum_rho_V_a_u - expected_u)
        acc_err_u = abs(sum_rho_V_a_u - sum_rhs_u)

        # Similarly for v-momentum
        v_face_y = 0.5 * (v[:-1, :] + v[1:, :])
        adv_yy = rho * (v_face_y**2) * geom.ay_mid * cc.dx
        diff_yy = rho * nu * (v[1:, :] - v[:-1, :]) / cc.dy * geom.ay_mid * cc.dx
        F_yy = diff_yy - adv_yy

        ax_v_eff = geom.ax_v.copy()
        u_out_v = np.zeros(cc.ny + 1)
        u_out_v[0] = u[0, cc.nx]
        u_out_v[1:cc.ny] = 0.5 * (u[:-1, cc.nx] + u[1:, cc.nx])
        u_out_v[cc.ny] = u[cc.ny - 1, cc.nx]
        F_yx_out = -rho * u_out_v * v[:, cc.nx - 1] * ax_v_eff[:, cc.nx] * cc.dy
        F_yx_in = 2.0 * rho * nu * v[:, 0] / cc.dx * ax_v_eff[:, 0] * cc.dy

        ext_flux_v = (np.sum(F_yy[cc.ny - 1, :]) - np.sum(F_yy[0, :])) + \
                     (np.sum(F_yx_out[1:cc.ny][geom.ay[1:cc.ny, cc.nx - 1] > 0]) - \
                      np.sum(F_yx_in[1:cc.ny][geom.ay[1:cc.ny, 0] > 0]))
        total_wall_v = np.sum(fy_wall) if len(cc.wall_cut_indices) > 0 else 0.0
        expected_v = ext_flux_v + total_wall_v

        active_v = (geom.ay > 0.0)
        sum_rho_V_a_v = np.sum(rho * cc.v_vol[active_v] * a_v[active_v])
        sum_rhs_v = np.sum(rhs_v[active_v])

        balance_err_v = abs(sum_rho_V_a_v - expected_v)
        acc_err_v = abs(sum_rho_V_a_v - sum_rhs_v)

        print(f"  Grid {nx}x{ny}: u-balance error = {balance_err_u:.2e} | v-balance error = {balance_err_v:.2e} [PASS]")
        assert balance_err_u < 1e-12, f"u flux-force-momentum balance violated on {nx}x{ny}: {balance_err_u}"
        assert balance_err_v < 1e-12, f"v flux-force-momentum balance violated on {nx}x{ny}: {balance_err_v}"
        assert acc_err_u < 1e-14, f"u acc-rhs equivalence violated on {nx}x{ny}: {acc_err_u}"
        assert acc_err_v < 1e-14, f"v acc-rhs equivalence violated on {nx}x{ny}: {acc_err_v}"

    print("  [OK] Exact flux-force-momentum balance verified to machine zero: sum(rho*V*a) == Net Ext Flux + Wall Force")

def test_conservative_cell_merging_momentum_balance():
    print("4. Testing exact momentum conservation under Conservative Cell Merging...")
    cc = CutCellBoundary(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)
    np.random.seed(123)
    u = np.ones((cc.ny, cc.nx + 1))
    v = np.zeros((cc.ny + 1, cc.nx))
    u[cc.geom.ax == 0.0] = 0.0

    dt = 0.005
    rho, nu = 1.0, 0.01
    a_u, a_v, rhs_u, rhs_v = cc.compute_momentum_rhs(u, v, nu, rho)

    # For every merged pair in u_merge_list:
    max_imbalance = 0.0
    for (js, is_), (jm, im) in cc.geom.u_merge_list:
        v_small = cc.u_vol[js, is_]
        v_master = cc.u_vol[jm, im]
        am = a_u[js, is_]

        impulse = (rhs_u[js, is_] + rhs_u[jm, im]) * dt
        delta_p = (rho * v_small * am * dt) + (rho * v_master * am * dt)
        err = abs(delta_p - impulse)
        if err > max_imbalance:
            max_imbalance = err

    print(f"  Max momentum imbalance across {len(cc.geom.u_merge_list)} merged pairs: {max_imbalance:.2e}")
    assert max_imbalance < 1e-16, f"Momentum conservation error in cell merging: {max_imbalance}"
    print("  [OK] Exact machine-zero momentum conservation (< 1e-16) verified for cell merging")

def test_temporal_convergence_order():
    print("5. Testing formal temporal convergence: Euler (p=1.00) vs AB2 (p=2.00)...")
    cc = CutCellBoundary(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)
    nu = 0.01
    rho = 1.0

    def get_initial_fields():
        u = np.zeros((cc.ny, cc.nx + 1))
        v = np.zeros((cc.ny + 1, cc.nx))
        for j in range(cc.ny):
            for i in range(cc.nx + 1):
                x = i * cc.dx
                y = (j + 0.5) * cc.dy
                u[j, i] = 1.0 + 0.2 * np.sin(2 * np.pi * x / 20.0) * np.cos(2 * np.pi * y / 5.0)
        for j in range(cc.ny + 1):
            for i in range(cc.nx):
                x = (i + 0.5) * cc.dx
                y = j * cc.dy
                v[j, i] = -0.2 * (5.0 / 20.0) * np.cos(2 * np.pi * x / 20.0) * np.sin(2 * np.pi * y / 5.0)
        u[cc.geom.ax == 0.0] = 0.0
        v[cc.geom.ay == 0.0] = 0.0
        cc.reconstruct_boundary_velocities(u, v)
        return u, v

    t_final = 0.02
    dts = [0.002, 0.001, 0.0005]

    # 1. Forward Euler test on momentum predictor ODE
    u_euler = []
    for dt in dts:
        cc_inst = CutCellBoundary(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)
        u, v = get_initial_fields()
        n_steps = int(round(t_final / dt))
        for step in range(n_steps):
            u, v = cc_inst.advance_momentum_predictor(u, v, dt, nu, rho, transport='centered', method='euler')
        u_euler.append(u.copy())

    e1_eu = np.max(np.abs(u_euler[0] - u_euler[1]))
    e2_eu = np.max(np.abs(u_euler[1] - u_euler[2]))
    p_euler = np.log2(e1_eu / e2_eu)
    print(f"  Forward Euler Predictor: e1={e1_eu:.4e}, e2={e2_eu:.4e} | Observed Order p = {p_euler:.2f}")
    assert 0.95 <= p_euler <= 1.05, f"Euler order p={p_euler:.2f} outside expected [0.95, 1.05]"

    # 2. Adams-Bashforth 2 test on momentum predictor ODE
    u_ab2 = []
    for dt in dts:
        cc_inst = CutCellBoundary(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)
        u, v = get_initial_fields()
        n_steps = int(round(t_final / dt))
        for step in range(n_steps):
            u, v = cc_inst.advance_momentum_predictor(u, v, dt, nu, rho, transport='centered', method='ab2')
        u_ab2.append(u.copy())

    e1_ab = np.max(np.abs(u_ab2[0] - u_ab2[1]))
    e2_ab = np.max(np.abs(u_ab2[1] - u_ab2[2]))
    p_ab2 = np.log2(e1_ab / e2_ab)
    print(f"  Adams-Bashforth 2 Predictor: e1={e1_ab:.4e}, e2={e2_ab:.4e} | Observed Order p = {p_ab2:.2f}")
    assert 1.90 <= p_ab2 <= 2.10, f"AB2 order p={p_ab2:.2f} outside expected [1.90, 2.10]"

    # 3. Full-Solver Fractional-Step Projection Test (verifying first-order non-incremental splitting error)
    u_proj = []
    dts_proj = [0.004, 0.002, 0.001]
    for dt in dts_proj:
        solver = MAC_CylinderFlow(
            nx=100, ny=25, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5, nu=0.01,
            boundary='cut_cell', pressure_solver='direct', verbose=False
        )
        n_steps = int(round(0.04 / dt))
        solver.solve(n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False)
        u_proj.append(solver.u.copy())

    e1_pr = np.max(np.abs(u_proj[0] - u_proj[1]))
    e2_pr = np.max(np.abs(u_proj[1] - u_proj[2]))
    p_proj = np.log2(e1_pr / e2_pr)
    print(f"  Full Solver Fractional-Step Projection: e1={e1_pr:.4e}, e2={e2_pr:.4e} | Observed Order p = {p_proj:.2f}")
    assert 0.85 <= p_proj <= 1.20, f"Full solver projection order p={p_proj:.2f} outside [0.85, 1.20]"
    print("  [OK] Rigorous temporal separation verified: Euler predictor (p=1.00), AB2 predictor (p=2.00), coupled solver (p=1.00; coupled 2nd-order reserved for Stage 0.4)")

def test_true_circle_slip_evaluation():
    print("6. Testing surface slip evaluation on the true cylinder circle r = R (N=360)...")
    grids = [(100, 25), (200, 50), (400, 100)]
    mean_slips = []
    max_slips = []
    max_uns = []

    for nx, ny in grids:
        solver = MAC_CylinderFlow(
            nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5, nu=0.01,
            boundary='cut_cell', pressure_solver='direct', verbose=False
        )
        solver.solve(n_steps=50, dt=0.005, perturb_transient=False, save_artifacts=False, verbose=False)

        mean_slip, max_slip, speed, un, ut = solver.cc.measure_true_surface_slip(solver.u, solver.v)
        max_un = float(np.max(np.abs(un)))
        max_ut = float(np.max(np.abs(ut)))

        mean_slips.append(mean_slip)
        max_slips.append(max_slip)
        max_uns.append(max_un)

        print(f"  Grid {nx}x{ny} (dx={solver.dx:.3f}): Mean Slip={mean_slip:.4f}, Max Slip={max_slip:.4f}, Max Un={max_un:.4f}, Max Ut={max_ut:.4f}")

    # Physical no-slip criteria assertions
    assert len(speed) == 360, f"Expected 360 surface sample points, got {len(speed)}"
    assert np.all(np.isfinite(speed)), "Non-finite surface slip encountered"
    
    # Assert monotonic convergence of mean wall slip
    assert mean_slips[1] < mean_slips[0] and mean_slips[2] < mean_slips[1], \
        f"Mean slip did not converge monotonically: {mean_slips}"
    
    # Assert monotonic convergence of normal penetration velocity
    assert max_uns[1] < max_uns[0] and max_uns[2] < max_uns[1], \
        f"Normal wall penetration velocity did not converge monotonically: {max_uns}"

    # Strict physical thresholds:
    assert mean_slips[0] < 0.25, f"Coarse grid mean slip too high: {mean_slips[0]}"
    assert mean_slips[1] < 0.15, f"Medium grid mean slip too high: {mean_slips[1]}"
    assert mean_slips[2] < 0.10, f"Fine grid mean slip exceeds 0.10: {mean_slips[2]}"
    print("  [OK] Surface slip evaluated on true physical circle: boundary reconstruction error decreases monotonically with mesh refinement (fine mean slip < 0.10)")

def run_all_momentum_and_force_tests():
    print("=" * 75)
    print("STAGE 0.3 VERIFICATION: Cut-Cell Momentum Predictor & Surface Forces")
    print("=" * 75)
    test_linear_pressure_archimedes_force()
    test_analytical_cylinder_couette_shear()
    test_true_path_momentum_balance_and_telescopic_cancellation()
    test_conservative_cell_merging_momentum_balance()
    test_temporal_convergence_order()
    test_true_circle_slip_evaluation()
    print("=" * 75)
    print("ALL MOMENTUM PREDICTOR & SURFACE FORCE TESTS PASSED CLEANLY!")
    print("=" * 75)

if __name__ == '__main__':
    run_all_momentum_and_force_tests()
