"""Lightweight analytical and benchmark comparison: Baseline vs Consistent Operators.

Suite:
1. Manufactured Harmonic Pressure Solve (potential flow around cylinder p = x*(1 + R^2/r^2))
   - Analyzes L2 field error, surface Linf error, force error, and convergence order across 40x30, 80x60, 160x120.
2. Linear Pressure Gradient / Archimedes Test (p = x)
   - Checks gradient consistency G_x p on cut faces vs interior, buoyant force on circle.
3. Unsteady Navier-Stokes Advance (Re=100, AB2/CN, 20 steps)
   - Checks stability, divergence, drag/lift force, field differences on 40x20 and 80x40.
4. Volume Metric and Merging Audit
   - Compares raw volume, u_vol (with covered transfers and boundary factor), and u_vol_eff (merging).
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

from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import (
    CutCellBoundary,
    build_cut_cell_poisson_matrix,
    build_surface_pressure_quadrature,
    surface_pressure_force,
    reconstruct_pressure,
    ab2_cn_step
)
from scratch.test_experimental_consistent_operators import build_consistent_poisson_matrix


def run_manufactured_pressure_suite():
    results = []
    resolutions = [40, 80, 160]
    for n in resolutions:
        ny = 3 * n // 4
        g = CutCellGeometry2D(n, ny, 4.0, 3.0, 1.4, 1.5, 0.5)
        cc = CutCellBoundary(n, ny, 4.0, 3.0, 1.4, 1.5, 0.5, pressure_solver='direct')
        h = g.dx
        xp = (np.arange(g.nx) + 0.5) * h
        yp = (np.arange(g.ny) + 0.5) * h
        x, y = np.meshgrid(xp - g.xc, yp - g.yc)

        def exact(xx, yy):
            r2 = xx * xx + yy * yy
            return (xx * (1 + g.R**2 / r2),
                    1 + g.R**2 * (yy * yy - xx * xx) / r2**2,
                    -2 * g.R**2 * xx * yy / r2**2)

        rhs = np.zeros_like(x)
        rhs[:, 0] += exact(-g.xc, yp - g.yc)[1] / h
        rhs[0, :] += exact(xp - g.xc, -g.yc)[2] / h
        rhs[-1, :] -= exact(xp - g.xc, g.Ly - g.yc)[2] / h
        rhs[:, -1] -= 2 * exact(g.Lx - g.xc, yp - g.yc)[0] / h**2

        A_base = build_cut_cell_poisson_matrix(g)
        A_cons, sx, sy = build_consistent_poisson_matrix(g, cc)

        p_base = spla.spsolve(A_base, rhs.ravel()).reshape(x.shape)
        p_cons = spla.spsolve(A_cons, rhs.ravel()).reshape(x.shape)

        points, op, nw = build_surface_pressure_quadrature(g)
        reference = exact(points[:, 1] - g.xc, points[:, 0] - g.yc)[0]
        exact_force = np.array([-2 * np.pi * g.R**2, 0.0])

        mask = (g.cell_type != -1) & (x * x + y * y >= g.R**2)
        base_L2 = float(np.sqrt(np.average((p_base[mask] - exact(x[mask], y[mask])[0])**2, weights=g.volume_fraction[mask])))
        cons_L2 = float(np.sqrt(np.average((p_cons[mask] - exact(x[mask], y[mask])[0])**2, weights=g.volume_fraction[mask])))

        base_surf_linf = float(np.max(np.abs(reconstruct_pressure(p_base, op) - reference)))
        cons_surf_linf = float(np.max(np.abs(reconstruct_pressure(p_cons, op) - reference)))

        f_base = surface_pressure_force(p_base, op, nw)
        f_cons = surface_pressure_force(p_cons, op, nw)

        base_f_err = float(np.linalg.norm(f_base - exact_force))
        cons_f_err = float(np.linalg.norm(f_cons - exact_force))

        res_entry = {
            'grid': f'{n}x{ny}',
            'h': float(h),
            'baseline': {
                'field_L2_error': base_L2,
                'surface_Linf_error': base_surf_linf,
                'force_error': base_f_err
            },
            'consistent': {
                'field_L2_error': cons_L2,
                'surface_Linf_error': cons_surf_linf,
                'force_error': cons_f_err
            },
            'cons_to_base_ratio': {
                'field_L2': cons_L2 / base_L2 if base_L2 != 0 else 0.0,
                'surface_Linf': cons_surf_linf / base_surf_linf if base_surf_linf != 0 else 0.0,
                'force_error': cons_f_err / base_f_err if base_f_err != 0 else 0.0
            }
        }
        results.append(res_entry)

    # Compute convergence rates
    for i in range(1, len(results)):
        r0, r1 = results[i - 1], results[i]
        ratio_h = r0['h'] / r1['h']
        r1['baseline']['order_field_L2'] = float(np.log(r0['baseline']['field_L2_error'] / r1['baseline']['field_L2_error']) / np.log(ratio_h))
        r1['baseline']['order_force'] = float(np.log(r0['baseline']['force_error'] / r1['baseline']['force_error']) / np.log(ratio_h))
        r1['consistent']['order_field_L2'] = float(np.log(r0['consistent']['field_L2_error'] / r1['consistent']['field_L2_error']) / np.log(ratio_h))
        r1['consistent']['order_force'] = float(np.log(r0['consistent']['force_error'] / r1['consistent']['force_error']) / np.log(ratio_h))

    return results


def run_linear_pressure_gradient_suite():
    results = []
    for n in (40, 80):
        ny = 3 * n // 4
        g = CutCellGeometry2D(n, ny, 4.0, 3.0, 1.4, 1.5, 0.5)
        cc = CutCellBoundary(n, ny, 4.0, 3.0, 1.4, 1.5, 0.5, pressure_solver='direct')
        h = g.dx
        xp = (np.arange(g.nx) + 0.5) * h
        yp = (np.arange(g.ny) + 0.5) * h
        x, y = np.meshgrid(xp - g.xc, yp - g.yc)

        p_exact = x.copy()
        points, op, nw = build_surface_pressure_quadrature(g)
        f_rec = surface_pressure_force(p_exact, op, nw)
        exact_archimedes = -np.pi * (g.R**2)

        # Baseline gradient on open faces
        G_base = np.where(g.ax > 0, 1.0, 0.0)

        # Consistent gradient on open faces
        open_u = (g.ax > 0) & (cc.u_vol > 0)
        G_cons = np.zeros_like(g.ax)
        G_cons[open_u] = g.ax[open_u] * (g.dx * g.dy) / cc.u_vol[open_u]

        cut_faces = (g.ax > 0) & (g.ax < 1.0)
        res_entry = {
            'grid': f'{n}x{ny}',
            'exact_archimedes_force': float(exact_archimedes),
            'reconstructed_force': float(f_rec[0]),
            'reconstruction_error': float(abs(f_rec[0] - exact_archimedes)),
            'gradient_baseline_cut_faces': {
                'min': float(G_base[cut_faces].min()),
                'max': float(G_base[cut_faces].max()),
                'mean': float(G_base[cut_faces].mean())
            },
            'gradient_consistent_cut_faces': {
                'min': float(G_cons[cut_faces].min()),
                'max': float(G_cons[cut_faces].max()),
                'mean': float(G_cons[cut_faces].mean()),
                'max_relative_perturbation': float(np.max(np.abs(G_cons[cut_faces] - 1.0)))
            }
        }
        results.append(res_entry)
    return results


def run_unsteady_flow_comparison(n_steps=20):
    results = []
    for nx, ny in [(40, 20), (80, 40)]:
        cc = CutCellBoundary(nx, ny, 4.0, 2.0, 1.5, 1.0, 0.5, pressure_solver='direct')
        geom = cc.geom
        dx, dy = geom.dx, geom.dy
        dt = 0.005 * (40.0 / nx)
        nu = 0.01
        rho = 1.0
        yu = cc.yu
        Ly = geom.Ly
        u_inlet = 4.0 * 1.5 * yu * (Ly - yu) / (Ly**2)

        A_cons, sx, sy = build_consistent_poisson_matrix(geom, cc)
        splu_cons = spla.splu(A_cons.tocsc(), permc_spec='MMD_AT_PLUS_A')
        fluid = (geom.cell_type != -1).ravel()

        active_u = geom.ax[:, 1:] > 0
        active_v = geom.ay[1:-1, :] > 0
        nu_active = int(active_u.sum())

        def pack(uu, vv):
            return np.r_[uu[:, 1:][active_u], vv[1:-1][active_v]]

        def unpack(q):
            uu, vv = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
            uu[:, 0] = u_inlet
            uu[:, 1:][active_u] = q[:nu_active]
            vv[1:-1][active_v] = q[nu_active:]
            return uu, vv

        def advection(q):
            au, av, _, _ = cc.compute_momentum_rhs(*unpack(q), 0., rho, transport='centered')
            return pack(au, av)

        def diffusion(q):
            au, av, _, _ = cc.compute_momentum_rhs(*unpack(q), nu, rho, transport='centered', include_advection=False)
            return pack(au, av)

        # Baseline Run
        u_b = np.repeat(u_inlet[:, None], nx + 1, axis=1)
        v_b = np.zeros((ny + 1, nx))
        p_b = np.zeros((ny, nx))
        u_b[geom.ax == 0] = 0.0
        v_b[geom.ay == 0] = 0.0

        for _ in range(n_steps):
            u_b, v_b, p_b = cc.advance_ab2_cn(u_b, v_b, p_b, dt, nu, rho, u_inlet)

        fx_b, fy_b, fx_p_b, fy_p_b, _, _ = cc.extract_true_surface_forces(u_b, v_b, p_b, nu * rho)
        div_b = np.zeros_like(p_b)
        cc.compute_rhs(u_b, v_b, 1.0, 1.0, div_b)
        max_div_b = float(np.max(np.abs(div_b[geom.cell_type != -1])))

        # Consistent Run
        open_u_mid = geom.ax[:, 1:-1] > 0
        open_v_mid = geom.ay[1:-1, :] > 0
        ratio_x = np.zeros_like(geom.ax[:, 1:-1])
        ratio_x[open_u_mid] = sx[:, 1:-1][open_u_mid] / geom.ax[:, 1:-1][open_u_mid]
        ratio_y = np.zeros_like(geom.ay[1:-1, :])
        ratio_y[open_v_mid] = sy[1:-1, :][open_v_mid] / geom.ay[1:-1, :][open_v_mid]

        def gradient_cons(pr):
            gu, gv = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
            gu[:, 1:-1] = np.where(open_u_mid, ratio_x * np.diff(pr, axis=1) / (rho * dx), 0.0)
            gu[:, -1] = -2 * pr[:, -1] / (rho * dx)
            gv[1:-1] = np.where(open_v_mid, ratio_y * np.diff(pr, axis=0) / (rho * dy), 0.0)
            return pack(gu, gv)

        def project_cons(q, step):
            us, vs = unpack(q)
            rhs_p = np.zeros((ny, nx))
            cc.compute_rhs(us, vs, step, rho, rhs_p)
            rhs_b = rhs_p.ravel().copy()
            rhs_b[~fluid] = 0.0
            phi = splu_cons.solve(rhs_b).reshape((ny, nx))

            dt_rho = step / rho
            uu, vv = us.copy(), vs.copy()
            uu[:, 0] = u_inlet
            uu[:, 1:-1] = np.where(open_u_mid, us[:, 1:-1] - dt_rho * ratio_x * np.diff(phi, axis=1) / dx, 0.0)
            uu[:, nx] = us[:, nx] + dt_rho * 2.0 * phi[:, nx - 1] / dx

            vv[0, :] = 0.0
            vv[1:-1, :] = np.where(open_v_mid, vs[1:-1, :] - dt_rho * ratio_y * np.diff(phi, axis=0) / dy, 0.0)
            vv[ny, :] = 0.0
            return pack(uu, vv), phi

        def divergence(q):
            uu, vv = unpack(q)
            out = np.zeros((ny, nx))
            cc.compute_rhs(uu, vv, 1., 1., out)
            fl = geom.cell_type != -1
            out[fl] /= geom.volume_fraction[fl]
            return out

        u_c = np.repeat(u_inlet[:, None], nx + 1, axis=1)
        v_c = np.zeros((ny + 1, nx))
        p_c = np.zeros((ny, nx))
        u_c[geom.ax == 0] = 0.0
        v_c[geom.ay == 0] = 0.0
        cn_p, cn_adv, cn_d = None, None, None

        for _ in range(n_steps):
            p_curr = p_c if cn_p is None else cn_p
            q, half, output, history, _ = ab2_cn_step(
                pack(u_c, v_c), p_curr, dt,
                advection=advection, diffusion=diffusion, gradient=gradient_cons, project=project_cons,
                divergence=divergence, nu=nu, rho=rho,
                previous_advection=cn_adv, previous_dt=cn_d
            )
            cn_p, cn_adv, cn_d = half, history, dt
            u_c, v_c = unpack(q)
            p_c = output

        fx_c, fy_c, fx_p_c, fy_p_c, _, _ = cc.extract_true_surface_forces(u_c, v_c, p_c, nu * rho)
        div_c = np.zeros_like(p_c)
        cc.compute_rhs(u_c, v_c, 1.0, 1.0, div_c)
        max_div_c = float(np.max(np.abs(div_c[geom.cell_type != -1])))

        # Hydrodynamic non-dimensional coefficients (U_mean=1.0, D=1.0, rho=1.0)
        cd_b = fx_b / 0.5
        cl_b = fy_b / 0.5
        cd_c = fx_c / 0.5
        cl_c = fy_c / 0.5

        u_diff = float(np.max(np.abs(u_c - u_b)))
        p_diff = float(np.max(np.abs(p_c - p_b)))

        res_entry = {
            'grid': f'{nx}x{ny}',
            'n_steps': n_steps,
            'dt': float(dt),
            'baseline': {
                'cd': float(cd_b),
                'cl': float(cl_b),
                'max_divergence': max_div_b
            },
            'consistent': {
                'cd': float(cd_c),
                'cl': float(cl_c),
                'max_divergence': max_div_c
            },
            'cd_diff_pct': float(abs(cd_c - cd_b) / cd_b * 100.0),
            'field_differences': {
                'u_linf': u_diff,
                'p_linf': p_diff
            }
        }
        results.append(res_entry)
    return results


def run_volume_audit_suite():
    results = []
    cases = [
        ('40x20', 40, 20, 4.0, 2.0, 1.5, 1.0, 0.5),
        ('80x40', 80, 40, 4.0, 2.0, 1.5, 1.0, 0.5),
        ('660x123', 660, 123, 22.0, 4.1, 2.0, 2.0, 0.5)
    ]
    for name, nx, ny, Lx, Ly, xc, yc, R in cases:
        cc = CutCellBoundary(nx, ny, Lx, Ly, xc, yc, R, pressure_solver='direct' if nx < 100 else 'iterative')
        geom = cc.geom
        dx, dy = geom.dx, geom.dy
        cell_area = dx * dy

        exact_continuous_fluid_vol = Lx * Ly - np.pi * R**2

        raw_V_u = geom.u_volume_fraction * cell_area
        bndry_V_u = raw_V_u.copy()
        bndry_V_u[:, [0, -1]] *= 0.5

        sum_raw = float(raw_V_u.sum())
        sum_bndry = float(bndry_V_u.sum())
        sum_u_vol = float(cc.u_vol.sum())
        sum_u_vol_eff = float(cc.u_vol_eff.sum())

        covered_vol = float(bndry_V_u[geom.ax == 0.0].sum())
        n_covered = len(cc.u_covered)
        n_merged = len(geom.u_merge_list)

        res_entry = {
            'grid': name,
            'exact_continuous_fluid_volume': float(exact_continuous_fluid_vol),
            'sum_raw_box_volumes': sum_raw,
            'sum_boundary_halved_volumes': sum_bndry,
            'sum_production_u_vol': sum_u_vol,
            'sum_production_u_vol_eff': sum_u_vol_eff,
            'covered_face_volume_transferred': covered_vol,
            'num_covered_faces': n_covered,
            'num_merged_faces': n_merged,
            'volume_conservation_error': float(abs(sum_u_vol - exact_continuous_fluid_vol))
        }
        results.append(res_entry)
    return results


def main():
    print("Running Suite 1: Manufactured Harmonic Pressure Solve...")
    p_res = run_manufactured_pressure_suite()
    print(json.dumps(p_res, indent=2))

    print("\nRunning Suite 2: Linear Pressure Gradient & Archimedes Buoyancy...")
    lin_res = run_linear_pressure_gradient_suite()
    print(json.dumps(lin_res, indent=2))

    print("\nRunning Suite 3: Unsteady Flow Advance (Re=100, 20 steps)...")
    flow_res = run_unsteady_flow_comparison(n_steps=20)
    print(json.dumps(flow_res, indent=2))

    print("\nRunning Suite 4: Volume Metric & Merging Audit...")
    vol_res = run_volume_audit_suite()
    print(json.dumps(vol_res, indent=2))

    full_report_data = {
        'suite_1_manufactured_pressure': p_res,
        'suite_2_linear_pressure_archimedes': lin_res,
        'suite_3_unsteady_flow_comparison': flow_res,
        'suite_4_volume_audit': vol_res
    }

    out_file = ROOT / 'reports' / 'lightweight_analytical_operator_comparison_20261004.json'
    with out_file.open('w', encoding='utf-8') as f:
        json.dump(full_report_data, f, indent=2)
    print(f"\nAll analytical suites finished. Output saved to {out_file}")


if __name__ == '__main__':
    main()
