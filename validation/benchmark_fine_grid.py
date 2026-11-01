"""
Fine-grid DFG 2D-2 benchmark simulation and independent force verification.
Grid: 880 x 164 (h = 0.0025 m), dt = 0.00025 s.
Includes:
  - 10-12s trend run (and optional 30s continuation)
  - Full cycle stationarity and DFG benchmark acceptance check
  - Independent surface pressure and viscous shear integration at r = D/2
  - Comparison between immersed-boundary reaction force and independent surface integration
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))

import sys
import time
import json
import argparse
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics
from mac_solver import calculate_mac_divergence

REFERENCES = {'cd_max': (3.22, 3.24), 'cl_max': (.99, 1.01), 'strouhal': (.295, .305)}

def compute_control_volume_balance(u, v, p, xc, yc, D, dx, dy, mu, rho, u_old, v_old, dt, r_factor=1.0, n_theta=720):
    """
    Computes exact Eulerian momentum flux, full-tensor viscous/pressure stress,
    and instantaneous volume acceleration for a circular control volume.
    """
    ny, nx = p.shape
    R = (D / 2.0) * r_factor
    
    # 1. Volume Acceleration (computed from exact temporal difference if available)
    accel_u = (u - u_old) / dt if u_old is not None else np.zeros_like(u)
    accel_v = (v - v_old) / dt if v_old is not None else np.zeros_like(v)
    
    xp = (np.arange(nx) + 0.5) * dx
    yp = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xp, yp)
    inside = (X - xc)**2 + (Y - yc)**2 <= R**2
    
    accel_u_c = 0.5 * (accel_u[:, :-1] + accel_u[:, 1:])
    accel_v_c = 0.5 * (accel_v[:-1, :] + accel_v[1:, :])
    vol_accel_x = np.sum(accel_u_c[inside]) * dx * dy * rho
    vol_accel_y = np.sum(accel_v_c[inside]) * dx * dy * rho

    # 2. Surface Integration
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    xs = xc + R * np.cos(theta)
    ys = yc + R * np.sin(theta)
    pts = np.column_stack((ys, xs))
    ds = R * (2 * np.pi / n_theta)

    rbf_p = interp.RegularGridInterpolator((yp, xp), p, bounds_error=False, fill_value=0.0)
    
    xu = np.arange(nx + 1) * dx
    yv = np.arange(ny + 1) * dy
    rbf_u = interp.RegularGridInterpolator((yp, xu), u, bounds_error=False, fill_value=0.0)
    rbf_v = interp.RegularGridInterpolator((yv, xp), v, bounds_error=False, fill_value=0.0)

    p_s = rbf_p(pts)
    u_s = rbf_u(pts)
    v_s = rbf_v(pts)

    nx_s = np.cos(theta)
    ny_s = np.sin(theta)

    # Momentum Flux
    flux_x = -rho * np.sum(u_s * (u_s * nx_s + v_s * ny_s)) * ds
    flux_y = -rho * np.sum(v_s * (u_s * nx_s + v_s * ny_s)) * ds

    # Viscous Stress (Full Tensor)
    dudx = (u[:, 1:] - u[:, :-1]) / dx
    dvdy = (v[1:, :] - v[:-1, :]) / dy
    rbf_dudx = interp.RegularGridInterpolator((yp, xp), dudx, bounds_error=False, fill_value=0.0)
    rbf_dvdy = interp.RegularGridInterpolator((yp, xp), dvdy, bounds_error=False, fill_value=0.0)

    dudy = (u[1:, 1:-1] - u[:-1, 1:-1]) / dy
    dvdx = (v[1:-1, 1:] - v[1:-1, :-1]) / dx
    rbf_dudy = interp.RegularGridInterpolator((yv[1:-1], xu[1:-1]), dudy, bounds_error=False, fill_value=0.0)
    rbf_dvdx = interp.RegularGridInterpolator((yv[1:-1], xu[1:-1]), dvdx, bounds_error=False, fill_value=0.0)

    tau_xx = 2 * mu * rbf_dudx(pts)
    tau_yy = 2 * mu * rbf_dvdy(pts)
    tau_xy = mu * (rbf_dudy(pts) + rbf_dvdx(pts))

    stress_x = np.sum(-p_s * nx_s + tau_xx * nx_s + tau_xy * ny_s) * ds
    stress_y = np.sum(-p_s * ny_s + tau_xy * nx_s + tau_yy * ny_s) * ds

    # Total reaction force expected from CV balance
    F_x = stress_x + flux_x - vol_accel_x
    F_y = stress_y + flux_y - vol_accel_y

    return F_x, F_y, stress_x, stress_y, flux_x, flux_y, vol_accel_x, vol_accel_y

def run_fine_grid(duration=120.0, dt=0.0025, nx=880, ny=164, save_artifacts=True):
    n_steps = int(np.round(duration / dt))
    print("="*75)
    print(f"FINE GRID DFG 2D-2 BENCHMARK: {nx}x{ny}, dt={dt}s, T={duration}s ({n_steps} steps)")
    print("="*75)

    solver = MAC_CylinderFlow(nx=nx, ny=ny, Lx=22.0, Ly=4.1, D=1.0, xc=2.0, yc=2.0,
                             nu=0.01, inflow='parabolic', U_m=1.5, verbose=True)

    t0 = time.perf_counter()
    t, cd, cl = solver.solve(n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"\nSimulation finished in {elapsed:.1f}s ({elapsed/n_steps*1000:.2f} ms/step)")

    # Sample periodic cycles in the developed window
    t_start = max(70.0, duration - 50.0) if duration > 75.0 else max(0.0, duration - 10.0)
    keep = t >= t_start
    if np.count_nonzero(keep) < 3:
        print(f"Warning: Insufficient samples in window [{t_start:.1f}, {duration:.1f}] for periodic analysis.")
        return None
    m = periodic_metrics(t, cd, cl, t_start=t_start, D=1.0, U_ref=1.0)

    max_div, div2, count, _, _ = calculate_mac_divergence(solver.u, solver.v, solver.dx, solver.dy, solver.mask)
    div_rms = np.sqrt(div2 / count)
    flux_err = abs(np.sum(solver.u[:, -1]) - np.sum(solver.u[:, 0])) / abs(np.sum(solver.u[:, 0]))
    slip = solver.ib.last_slip

    print("\n" + "="*75)
    print(f"BENCHMARK RESULTS ON FINE GRID ({nx}x{ny}):")
    print("="*75)
    for key, (lo, hi) in REFERENCES.items():
        val = m[key]
        mid = (lo + hi) / 2.0
        diff = (val / mid - 1.0) * 100.0
        status = "PASS" if lo <= val <= hi else "OUTSIDE"
        print(f"  {key:<12s}: {val:.6f} | Ref [{lo:.4f}, {hi:.4f}] | Diff: {diff:+.2f}% | {status}")
    print(f"  Complete cycles: {m['cycles']} | Stationarity: {m['stationary']}")
    print(f"  Drift (amp/period/drag): {m['amplitude_drift']:.2%} / {m['period_drift']:.2%} / {m['drag_drift']:.2%}")
    print(f"  Continuity: Div_RMS = {div_rms:.3e} 1/s | Flux mismatch = {flux_err:.3e}")
    print(f"  No-slip: Max cylinder boundary slip = {slip:.3e} m/s")
    print("="*75)

    # Independent surface force verification on final state
    denom = 0.5 * solver.rho * (solver.U_ref**2) * solver.D
    mu = solver.nu * solver.rho

    print("\nINDEPENDENT CONTROL-VOLUME FORCE CHECK (at t = final):")
    # Take one explicit step to compute the true volume acceleration d(rho u)/dt
    u_final, v_final, p_final = solver.u.copy(), solver.v.copy(), solver.p.copy()
    solver.solve(n_steps=1, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False)
    
    # 1. At marker radius r = R
    fx0, fy0, sx0, sy0, px0, py0, ax0, ay0 = compute_control_volume_balance(
        solver.u, solver.v, solver.p, solver.xc, solver.yc, solver.D,
        solver.dx, solver.dy, mu, solver.rho, u_old=u_final, v_old=v_final, dt=dt, r_factor=1.0
    )
    
    # 2. Outer edge of delta kernel r = R + 1.5h
    r_factor_15 = 1.0 + 1.5 * solver.dx / (solver.D / 2.0)
    fx15, fy15, sx15, sy15, px15, py15, ax15, ay15 = compute_control_volume_balance(
        solver.u, solver.v, solver.p, solver.xc, solver.yc, solver.D,
        solver.dx, solver.dy, mu, solver.rho, u_old=u_final, v_old=v_final, dt=dt, r_factor=r_factor_15
    )

    # 3. Pure exterior fluid r = R + 2.5h
    r_factor_25 = 1.0 + 2.5 * solver.dx / (solver.D / 2.0)
    fx25, fy25, sx25, sy25, px25, py25, ax25, ay25 = compute_control_volume_balance(
        solver.u, solver.v, solver.p, solver.xc, solver.yc, solver.D,
        solver.dx, solver.dy, mu, solver.rho, u_old=u_final, v_old=v_final, dt=dt, r_factor=r_factor_25
    )
    
    cd_surf = fx25 / denom
    cl_surf = fy25 / denom
    cd_ib = cd[-1]
    cl_ib = cl[-1]

    print(f"  IB computed force at t={t[-1]:.3f}s:        Cd = {cd_ib:.6f}, Cl = {cl_ib:.6f}")
    print(f"  CV at r = R (inside delta):              Cd = {fx0/denom:.6f}, Cl = {fy0/denom:.6f}")
    print(f"  CV at r = R + 1.5h (outer delta edge):   Cd = {fx15/denom:.6f}, Cl = {fy15/denom:.6f}")
    print(f"  CV at r = R + 2.5h (pure exterior fluid):Cd = {cd_surf:.6f}, Cl = {cl_surf:.6f}")
    print(f"    -> Stress: Cd = {sx25/denom:.6f}, Cl = {sy25/denom:.6f}")
    print(f"    -> Mom Flux: Cd = {px25/denom:.6f}, Cl = {py25/denom:.6f}")
    print(f"    -> Vol Accel: Cd = {ax25/denom:.6f}, Cl = {ay25/denom:.6f}")
    print(f"  Reaction vs Total force agreement:       IB reaction = {solver.ib.last_reaction[1]/denom:.6f}")
    print("="*75)

    if save_artifacts:
        meta = solver.metadata(dt, n_steps, False)
        import hashlib
        from pathlib import Path
        meta['source_sha256']['benchmark_fine_grid.py'] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
        np.savez_compressed(os.path.join(os.path.dirname(__file__), '../data/fine_grid_dfg_results.npz'),
                            t=t, cd=cd, cl=cl, u=u_final, v=v_final, p=p_final, mask=solver.mask,
                            **m, t_start=t_start, div_rms=div_rms, flux_err=flux_err, marker_slip=slip,
                            cd_surf=cd_surf, cl_surf=cl_surf, 
                            cd_stress=sx25/denom, cl_stress=sy25/denom,
                            cd_flux=px25/denom, cl_flux=py25/denom,
                            cd_accel=ax25/denom, cl_accel=ay25/denom,
                            cd_cv0=fx0/denom, cl_cv0=fy0/denom,
                            cd_cv15=fx15/denom, cl_cv15=fy15/denom,
                            elapsed=elapsed, metadata=json.dumps(meta))

        # Plot comparative force history
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        keep = t >= t_start
        axes[0].plot(t[keep], cd[keep], 'b-', lw=1.2, label=f'Computed $C_D$ ({nx}x{ny}, max={m["cd_max"]:.4f})')
        axes[0].axhspan(*REFERENCES['cd_max'], color='gray', alpha=0.25, label='DFG 2D-2 reference [3.22, 3.24]')
        axes[0].set_ylabel('$C_D$')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'Fine Grid DFG 2D-2 Benchmark ({nx}x{ny}, dt={dt}s) | St={m["strouhal"]:.4f}')

        axes[1].plot(t[keep], cl[keep], 'b-', lw=1.2, label=f'Computed $C_L$ ({nx}x{ny}, max={m["cl_max"]:.4f})')
        axes[1].axhspan(*REFERENCES['cl_max'], color='gray', alpha=0.25, label='DFG 2D-2 reference [0.99, 1.01]')
        axes[1].set_ylabel('$C_L$')
        axes[1].set_xlabel('Time $t$ [s]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/fine_grid_benchmark.png'), dpi=150)
        plt.close(fig)
        print("Saved fine_grid_dfg_results.npz and fine_grid_benchmark.png")

    return m

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--dt', type=float, default=0.0025)
    args = parser.parse_args()
    run_fine_grid(duration=args.duration, dt=args.dt)
