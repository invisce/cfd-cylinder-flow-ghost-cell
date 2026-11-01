"""
Verification of rigorous mathematical equivalence between dimensional and non-dimensional MAC cylinder flow.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from main import MAC_CylinderFlow
from mac_solver import calculate_mac_divergence

def verify_nondimensional_equivalence():
    print("Testing rigorous non-dimensional scaling equivalence...")
    n_steps = 100
    nx, ny = 100, 25

    # Case A: Dimensional setup (D=0.4, nu=0.004, dt=0.002)
    dt_dim = 0.002
    solver_dim = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=8.0, Ly=2.0, D=0.4, xc=2.0, yc=1.0,
        nu=0.004, rho=1.0, U_inf=1.0, inflow='uniform', verbose=False,
        boundary='staircase', transport='centered'
    )
    t_dim, cd_dim, cl_dim = solver_dim.solve(
        n_steps=n_steps, dt=dt_dim, perturb_transient=True, perturb_duration=1.0, save_artifacts=False, verbose=False
    )

    # Case B: Non-dimensional setup (D=1.0, nu=0.01, dt=0.005 = 0.002 * 2.5)
    scale = 1.0 / 0.4  # 2.5
    dt_nondim = dt_dim * scale  # 0.005
    solver_nondim = MAC_CylinderFlow(
        nx=nx, ny=ny, Lx=8.0 * scale, Ly=2.0 * scale, D=1.0, xc=2.0 * scale, yc=1.0 * scale,
        nu=0.004 / 0.4, rho=1.0, U_inf=1.0, inflow='uniform', verbose=False,
        boundary='staircase', transport='centered'
    )
    t_nondim, cd_nondim, cl_nondim = solver_nondim.solve(
        n_steps=n_steps, dt=dt_nondim, perturb_transient=True, perturb_duration=2.5, save_artifacts=False, verbose=False
    )

    # 1. Non-dimensional time equivalence: t_nondim == t_dim * scale
    np.testing.assert_allclose(t_nondim, t_dim * scale, rtol=1e-12, atol=1e-14)
    print("  [OK] Non-dimensional time array matches exact scaling")

    # 2. Velocity field equivalence (dimensionless velocity u/U_inf should be IDENTICAL at every grid face)
    diff_u = np.max(np.abs(solver_dim.u - solver_nondim.u))
    diff_v = np.max(np.abs(solver_dim.v - solver_nondim.v))
    assert diff_u < 1e-12, f"u velocity mismatch: {diff_u}"
    assert diff_v < 1e-12, f"v velocity mismatch: {diff_v}"
    print(f"  [OK] Velocity field match: max|du| = {diff_u:.2e}, max|dv| = {diff_v:.2e}")

    # 3. Pressure field scaling: p_dim == p_nondim (since rho=1, U=1)
    diff_p = np.max(np.abs(solver_dim.p - solver_nondim.p))
    assert diff_p < 1e-10, f"Pressure field mismatch: {diff_p}"
    print(f"  [OK] Pressure field match: max|dp| = {diff_p:.2e}")

    # 4. Aerodynamic force coefficients: Cd and Cl must be IDENTICAL at all steps
    diff_cd = np.max(np.abs(cd_dim - cd_nondim))
    diff_cl = np.max(np.abs(cl_dim - cl_nondim))
    assert diff_cd < 1e-11, f"Cd history mismatch: {diff_cd}"
    assert diff_cl < 1e-11, f"Cl history mismatch: {diff_cl}"
    print(f"  [OK] Force coefficients match: max|dCd| = {diff_cd:.2e}, max|dCl| = {diff_cl:.2e}")

    # 5. Continuity: divergence scales with 1/scale (div_nondim = div_dim / scale)
    div_rms_dim, _, _, _, _ = calculate_mac_divergence(solver_dim.u, solver_dim.v, solver_dim.dx, solver_dim.dy, solver_dim.mask)
    div_rms_nondim, _, _, _, _ = calculate_mac_divergence(solver_nondim.u, solver_nondim.v, solver_nondim.dx, solver_nondim.dy, solver_nondim.mask)
    assert abs(div_rms_nondim - div_rms_dim / scale) < 1e-12
    print(f"  [OK] Discrete divergence scales exactly as 1/L: {div_rms_nondim:.2e} vs {div_rms_dim/scale:.2e}")

    print("\nSUCCESS: Dimensionless equivalence is mathematically exact to roundoff precision!\n")

if __name__ == '__main__':
    verify_nondimensional_equivalence()
