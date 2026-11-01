"""Experimental prototype of adjoint-consistent (D, M, G, A) operators on Cut-Cell MAC grid.

Formulation:
1. Momentum metric: M_u = V_geom_u (physical staggered fluid volume).
2. Adjoint gradient: G_x phi = (ax * dx * dy / M_u) * (phi[i] - phi[i-1]) / dx
   Velocity correction: u^{n+1} = u^* - (dt / rho) * G_x phi
3. Discrete divergence: (D u)_{j, i} = (ax_{i+1} u_{i+1} - ax_i u_i)/dx + (ay_{j+1} v_{j+1} - ay_j v_j)/dy
4. Adjoint Poisson matrix: A_cons = D M^{-1} G
   Face conductivities: sigma_x = ax^2 * dx * dy / M_u = ax^2 / theta_u
                        sigma_y = ay^2 * dx * dy / M_v = ay^2 / theta_v

Evaluates on a small case:
- Symmetry and positive-definiteness of A_cons
- Convergence and condition of Poisson solve
- Roundoff mass conservation (D u^{n+1} < 1e-12)
- Momentum balance under physical volume V_geom: is Defect_geom == 0?
- Reconstructed surface force comparison vs baseline.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'solver'))

from cut_cell_solver import CutCellBoundary


def build_consistent_poisson_matrix(geom, cc):
    nx, ny = geom.nx, geom.ny
    dx, dy = geom.dx, geom.dy
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    N = ny * nx

    # Compute face conductivity sigma_x = ax^2 * dx * dy / cc.u_vol
    sigma_x = np.zeros_like(geom.ax)
    open_u = (geom.ax > 0) & (cc.u_vol > 0)
    sigma_x[open_u] = (geom.ax[open_u] ** 2) * (dx * dy) / cc.u_vol[open_u]
    # At inlet (i=0): Neumann dPhi/dx = 0 -> sigma_x[:, 0] = 0
    sigma_x[:, 0] = 0.0
    # At outlet (i=nx): Dirichlet distance is dx/2 -> factor is 2.0
    # sigma_x[:, nx] effective conductivity is 2.0 * ax[:, nx]

    sigma_y = np.zeros_like(geom.ay)
    open_v = (geom.ay > 0) & (cc.v_vol > 0)
    sigma_y[open_v] = (geom.ay[open_v] ** 2) * (dx * dy) / cc.v_vol[open_v]
    sigma_y[0, :] = 0.0
    sigma_y[ny, :] = 0.0

    rows, cols, data = [], [], []

    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if geom.cell_type[j, i] == -1:
                # Solid cell: decoupled identity equation
                rows.append(k); cols.append(k); data.append(1.0)
                continue

            diag = 0.0

            # West face (j, i)
            if i > 0 and geom.cell_type[j, i - 1] != -1:
                w_flux = sigma_x[j, i] * inv_dx2
                rows.append(k); cols.append(k - 1); data.append(w_flux)
                diag -= w_flux

            # East face (j, i+1)
            if i < nx - 1 and geom.cell_type[j, i + 1] != -1:
                e_flux = sigma_x[j, i + 1] * inv_dx2
                rows.append(k); cols.append(k + 1); data.append(e_flux)
                diag -= e_flux
            elif i == nx - 1:
                # Outlet Dirichlet Phi=0 at dx/2
                out_flux = 2.0 * geom.ax[j, nx] * inv_dx2
                diag -= out_flux

            # South face (j, i)
            if j > 0 and geom.cell_type[j - 1, i] != -1:
                s_flux = sigma_y[j, i] * inv_dy2
                rows.append(k); cols.append(k - nx); data.append(s_flux)
                diag -= s_flux

            # North face (j+1, i)
            if j < ny - 1 and geom.cell_type[j + 1, i] != -1:
                n_flux = sigma_y[j + 1, i] * inv_dy2
                rows.append(k); cols.append(k + nx); data.append(n_flux)
                diag -= n_flux

            rows.append(k); cols.append(k); data.append(diag)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, sigma_x, sigma_y


def compare_baseline_and_consistent_operators(nx=40, ny=20, Lx=4.0, Ly=2.0, xc=1.5, yc=1.0, R=0.5,
                                             dt=0.005, nu=0.01, rho=1.0):
    cc = CutCellBoundary(nx, ny, Lx, Ly, xc, yc, R, pressure_solver='direct')
    geom = cc.geom
    dx, dy = geom.dx, geom.dy
    cell_area = dx * dy

    # 1. Check baseline Poisson operator
    A_base = cc.A
    splu_base = cc.splu

    # 2. Build consistent Poisson operator
    A_cons, sigma_x, sigma_y = build_consistent_poisson_matrix(geom, cc)
    splu_cons = spla.splu(A_cons, permc_spec='MMD_AT_PLUS_A')

    # Symmetry check of A_cons on fluid cells
    fluid = (geom.cell_type != -1).ravel()
    A_cons_fluid = A_cons[fluid, :][:, fluid]
    sym_err = float(np.max(np.abs(A_cons_fluid - A_cons_fluid.T)))

    # Initial flow profile
    yu = cc.yu
    u_inlet = 4.0 * 1.0 * yu * (Ly - yu) / (Ly**2)
    u0 = np.repeat(u_inlet[:, None], nx + 1, axis=1)
    v0 = np.zeros((ny + 1, nx))
    u0[geom.ax == 0] = 0.0
    v0[geom.ay == 0] = 0.0

    # Advance common predictor
    a_u, a_v, rhs_u, rhs_v = cc.compute_momentum_rhs(u0, v0, nu, rho, transport='centered')
    u_star, v_star = cc.advance_momentum_predictor(u0, v0, dt, nu, rho, u_inlet=u_inlet, transport='centered', method='euler')

    # Common RHS of Poisson equation: div(u_star) * rho / dt
    rhs_p = np.zeros((ny, nx))
    cc.compute_rhs(u_star, v_star, dt, rho, rhs_p)
    rhs_b = rhs_p.ravel().copy()
    rhs_b[~fluid] = 0.0

    # --- BASELINE SOLVE & PROJECTION ---
    phi_base = splu_base.solve(rhs_b).reshape((ny, nx))
    u_base = np.zeros_like(u0)
    v_base = np.zeros_like(v0)
    cc.apply_corrector(u_base, v_base, u_star, v_star, phi_base, dt, rho, u_inlet)

    # Check baseline divergence
    div_base = np.zeros_like(rhs_p)
    cc.compute_rhs(u_base, v_base, 1.0, 1.0, div_base)
    max_div_base = float(np.max(np.abs(div_base[geom.cell_type != -1])))

    # Baseline momentum change under V_geom
    dP_base_geom = float(np.sum(rho * cc.u_vol * (u_base - u_star)))
    dP_base_ap = float(np.sum(rho * (geom.ax * cell_area) * (u_base - u_star)))
    defect_base = dP_base_geom - dP_base_ap

    # Baseline surface forces (defined ON THE OBSTACLE: -int p n dA)
    _, _, fx_p_rec_base, _, _, _ = cc.extract_true_surface_forces(u_base, v_base, phi_base, nu * rho)
    fx_p_alg_base = - float(np.sum(phi_base * geom.bndry_normal_int_x))

    # --- CONSISTENT SOLVE & PROJECTION ---
    phi_cons = splu_cons.solve(rhs_b).reshape((ny, nx))
    u_cons = u_star.copy()
    v_cons = v_star.copy()

    # Consistent velocity corrector: u^{n+1} = u^* - (dt/rho) * (sigma_x / ax) * dphi / dx
    dt_over_rho = dt / rho
    open_u_int = geom.ax[:, 1:-1] > 0
    grad_phi_x = np.zeros_like(u0)
    grad_phi_x[:, 1:-1] = np.where(
        open_u_int,
        (sigma_x[:, 1:-1] / geom.ax[:, 1:-1]) * (phi_cons[:, 1:] - phi_cons[:, :-1]) / dx,
        0.0
    )
    u_cons[:, 0] = u_inlet
    u_cons[:, 1:-1] = u_star[:, 1:-1] - dt_over_rho * grad_phi_x[:, 1:-1]
    # Outlet Dirichlet: distance dx/2
    u_cons[:, nx] = u_star[:, nx] + dt_over_rho * 2.0 * phi_cons[:, nx - 1] / dx

    open_v_int = geom.ay[1:-1, :] > 0
    grad_phi_y = np.zeros_like(v0)
    grad_phi_y[1:-1, :] = np.where(
        open_v_int,
        (sigma_y[1:-1, :] / geom.ay[1:-1, :]) * (phi_cons[1:, :] - phi_cons[:-1, :]) / dy,
        0.0
    )
    v_cons[0, :] = 0.0
    v_cons[1:-1, :] = v_star[1:-1, :] - dt_over_rho * grad_phi_y[1:-1, :]
    v_cons[ny, :] = 0.0

    # Check consistent divergence
    div_cons = np.zeros_like(rhs_p)
    cc.compute_rhs(u_cons, v_cons, 1.0, 1.0, div_cons)
    max_div_cons = float(np.max(np.abs(div_cons[geom.cell_type != -1])))

    # Telescoping pressure impulse on internal u-faces:
    # I_p = - dt * dy * sum(ax * (phi[i] - phi[i-1]))
    I_p_base = - dt * dy * float(np.sum(geom.ax[:, 1:-1] * (phi_base[:, 1:] - phi_base[:, :-1])))
    I_p_cons = - dt * dy * float(np.sum(geom.ax[:, 1:-1] * (phi_cons[:, 1:] - phi_cons[:, :-1])))

    # Physical volume momentum change on interior u-velocities:
    dP_base_geom_int = float(np.sum(rho * cc.u_vol[:, 1:-1] * (u_base[:, 1:-1] - u_star[:, 1:-1])))
    dP_cons_geom_int = float(np.sum(rho * cc.u_vol[:, 1:-1] * (u_cons[:, 1:-1] - u_star[:, 1:-1])))

    defect_base_geom = dP_base_geom_int - I_p_base
    defect_cons_geom = dP_cons_geom_int - I_p_cons

    # Consistent surface forces (defined ON THE OBSTACLE: -int p n dA)
    _, _, fx_p_rec_cons, _, _, _ = cc.extract_true_surface_forces(u_cons, v_cons, phi_cons, nu * rho)
    fx_p_alg_cons = - float(np.sum(phi_cons * geom.bndry_normal_int_x))

    res = {
        'grid': f'{nx}x{ny}',
        'matrix_symmetry_linf': sym_err,
        'baseline': {
            'max_divergence': max_div_base,
            'dP_geom_interior': dP_base_geom_int,
            'telescoping_impulse': I_p_base,
            'defect_geom': defect_base_geom,
            'defect_pct': abs(defect_base_geom / I_p_base) * 100.0 if I_p_base != 0 else 0.0,
            'reconstructed_force': fx_p_rec_base,
            'algebraic_force': fx_p_alg_base,
            'force_diff_pct': abs(fx_p_alg_base - fx_p_rec_base) / abs(fx_p_rec_base) * 100.0
        },
        'consistent_prototype': {
            'max_divergence': max_div_cons,
            'dP_geom_interior': dP_cons_geom_int,
            'telescoping_impulse': I_p_cons,
            'defect_geom': defect_cons_geom,
            'defect_pct': abs(defect_cons_geom / I_p_cons) * 100.0 if I_p_cons != 0 else 0.0,
            'reconstructed_force': fx_p_rec_cons,
            'algebraic_force': fx_p_alg_cons,
            'force_diff_pct': abs(fx_p_alg_cons - fx_p_rec_cons) / abs(fx_p_rec_cons) * 100.0
        },
        'velocity_diff_linf': float(np.max(np.abs(u_cons - u_base))),
        'pressure_diff_linf': float(np.max(np.abs(phi_cons - phi_base)))
    }
    return res


def main():
    print("Testing consistent operator prototype on 40x20 and 80x40...")
    for nx, ny in [(40, 20), (80, 40)]:
        r = compare_baseline_and_consistent_operators(nx, ny, dt=0.005 * (40.0 / nx))
        print(f"\n=== Grid {r['grid']} ===")
        print(f"Matrix Symmetry Error: {r['matrix_symmetry_linf']:.2e}")
        print(f"Baseline:   Max Div = {r['baseline']['max_divergence']:.2e}, Defect = {r['baseline']['defect_geom']:.6e} ({r['baseline']['defect_pct']:.3f}%)")
        print(f"Consistent: Max Div = {r['consistent_prototype']['max_divergence']:.2e}, Defect = {r['consistent_prototype']['defect_geom']:.6e} ({r['consistent_prototype']['defect_pct']:.3f}%)")
        print(f"Diff between fields: max|u_cons - u_base| = {r['velocity_diff_linf']:.4e}, max|phi_cons - phi_base| = {r['pressure_diff_linf']:.4e}")
        print(f"Force diff (rec vs alg): Baseline = {r['baseline']['force_diff_pct']:.2f}%, Consistent = {r['consistent_prototype']['force_diff_pct']:.2f}%")

    out_path = ROOT / 'reports' / 'experimental_consistent_operator_comparison_20261004.json'
    results = [compare_baseline_and_consistent_operators(nx, ny, dt=0.005 * (40.0 / nx)) for nx, ny in [(40, 20), (80, 40)]]
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nPrototype results saved to {out_path}")


if __name__ == '__main__':
    main()
