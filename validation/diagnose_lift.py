import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""
Diagnostic script for force components in the immersed boundary solver.
Decomposes aerodynamic lift and drag into:
  - Lagrangian marker reaction force
  - Fictitious-fluid internal momentum correction
  - Independent surface pressure integration on the cylinder
  - Independent surface viscous shear integration on the cylinder
  - Control-volume momentum balance
"""
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics

def compute_surface_forces(u, v, p, xc, yc, D, dx, dy, mu, rho, r_factor=1.0, n_theta=360):
    ny, nx = p.shape
    R = (D / 2.0) * r_factor
    xp = (np.arange(nx) + 0.5) * dx
    yp = (np.arange(ny) + 0.5) * dy
    rbf_p = interp.RegularGridInterpolator((yp, xp), p, bounds_error=False, fill_value=0.0)

    # Vorticity at vertices: j in 1..ny-1, i in 1..nx-1
    dvdx = (v[1:-1, 1:] - v[1:-1, :-1]) / dx
    dudy = (u[1:, 1:-1] - u[:-1, 1:-1]) / dy
    omega = dvdx - dudy
    x_vert = np.arange(1, nx) * dx
    y_vert = np.arange(1, ny) * dy
    rbf_w = interp.RegularGridInterpolator((y_vert, x_vert), omega, bounds_error=False, fill_value=0.0)

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    xs = xc + R * np.cos(theta)
    ys = yc + R * np.sin(theta)
    pts = np.column_stack((ys, xs))
    p_s = rbf_p(pts)
    w_s = rbf_w(pts)
    ds = R * (2 * np.pi / n_theta)

    # Fluid on body:
    # Pressure: -p * n where n = (cos, sin) pointing into fluid
    fx_p = -np.sum(p_s * np.cos(theta)) * ds
    fy_p = -np.sum(p_s * np.sin(theta)) * ds
    # Viscous: tau_w = -mu * omega * t, t = (-sin, cos)
    fx_v = mu * np.sum(w_s * np.sin(theta)) * ds
    fy_v = -mu * np.sum(w_s * np.cos(theta)) * ds

    return fx_p, fy_p, fx_v, fy_v

def run_lift_diagnostics(n_steps=2000, dt=0.005):
    # Initialize solver with DFG configuration
    solver = MAC_CylinderFlow(nx=440, ny=82, Lx=22.0, Ly=4.1, D=1.0, xc=2.0, yc=2.0,
                             nu=0.01, inflow='parabolic', U_m=1.5, verbose=False)
    # Seed with the saved periodic state to diagnose in developed limit cycle
    with np.load(os.path.join(os.path.dirname(__file__), '../data/schafer_turek_results.npz'), allow_pickle=False) as d:
        solver.u[:] = d['u']
        solver.v[:] = d['v']
        solver.p[:] = d['p']
        solver.time = float(d['t'][-1])

    denom = 0.5 * solver.rho * (solver.U_ref**2) * solver.D
    mu = solver.nu * solver.rho

    t_hist = []
    cl_tot_hist = []
    cl_react_hist = []
    cl_mom_hist = []
    cd_tot_hist = []
    cd_react_hist = []
    cd_mom_hist = []

    # Surface sampling lists
    t_surf = []
    cl_p_hist = []
    cl_v_hist = []
    cd_p_hist = []
    cd_v_hist = []

    print(f"Running lift diagnostics for {n_steps} steps (dt={dt}s)...", flush=True)

    for s in range(n_steps):
        # Step forward 1 step
        solver.solve(n_steps=1, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False)
        curr_t = solver.time

        react = solver.ib.last_reaction
        mom = solver.ib.last_momentum_corr
        tot = solver.ib.last_force

        cl_react = react[1] / denom
        cl_mom = mom[1] / denom
        cl_tot = tot[1] / denom

        cd_react = react[0] / denom
        cd_mom = mom[0] / denom
        cd_tot = tot[0] / denom

        t_hist.append(curr_t)
        cl_react_hist.append(cl_react)
        cl_mom_hist.append(cl_mom)
        cl_tot_hist.append(cl_tot)
        cd_react_hist.append(cd_react)
        cd_mom_hist.append(cd_mom)
        cd_tot_hist.append(cd_tot)

        # Sample surface forces every 10 steps (every 0.005s)
        if s % 10 == 0:
            fx_p, fy_p, fx_v, fy_v = compute_surface_forces(
                solver.u, solver.v, solver.p, solver.xc, solver.yc, solver.D,
                solver.dx, solver.dy, mu, solver.rho, r_factor=1.0
            )
            t_surf.append(curr_t)
            cd_p_hist.append(fx_p / denom)
            cd_v_hist.append(fx_v / denom)
            cl_p_hist.append(fy_p / denom)
            cl_v_hist.append(fy_v / denom)

    t_arr = np.array(t_hist)
    cl_tot_arr = np.array(cl_tot_hist)
    cl_react_arr = np.array(cl_react_hist)
    cl_mom_arr = np.array(cl_mom_hist)
    cd_tot_arr = np.array(cd_tot_hist)
    cd_react_arr = np.array(cd_react_hist)
    cd_mom_arr = np.array(cd_mom_hist)

    t_surf = np.array(t_surf)
    cl_p_arr = np.array(cl_p_hist)
    cl_v_arr = np.array(cl_v_hist)
    cd_p_arr = np.array(cd_p_hist)
    cd_v_arr = np.array(cd_v_hist)

    print("\n" + "="*70)
    print("FORCE COMPONENT DECOMPOSITION (DFG 2D-2 Benchmark, Re=100)")
    print("="*70)
    print(f"Total Cl min/max/half-range: {cl_tot_arr.min():.6f} / {cl_tot_arr.max():.6f} / {np.ptp(cl_tot_arr)/2:.6f}")
    print(f"Marker reaction Cl min/max: {cl_react_arr.min():.6f} / {cl_react_arr.max():.6f} / {np.ptp(cl_react_arr)/2:.6f}")
    print(f"Fictitious mom Cl min/max:  {cl_mom_arr.min():.6f} / {cl_mom_arr.max():.6f} / {np.ptp(cl_mom_arr)/2:.6f}")
    print(f"Ratio of momentum correction to reaction amplitude: {np.ptp(cl_mom_arr) / np.ptp(cl_react_arr):.2%}")
    print("-"*70)
    print(f"Total Cd min/max/mean:       {cd_tot_arr.min():.6f} / {cd_tot_arr.max():.6f} / {cd_tot_arr.mean():.6f}")
    print(f"Marker reaction Cd mean:     {cd_react_arr.mean():.6f}")
    print(f"Fictitious mom Cd mean:      {cd_mom_arr.mean():.6f}")
    print("-"*70)
    print("INDEPENDENT SURFACE INTEGRALS (at r = D/2):")
    print(f"Surface pressure Cl:        {cl_p_arr.min():.6f} / {cl_p_arr.max():.6f} / {np.ptp(cl_p_arr)/2:.6f}")
    print(f"Surface viscous Cl:         {cl_v_arr.min():.6f} / {cl_v_arr.max():.6f} / {np.ptp(cl_v_arr)/2:.6f}")
    print(f"Surface total Cl:           {(cl_p_arr+cl_v_arr).min():.6f} / {(cl_p_arr+cl_v_arr).max():.6f} / {np.ptp(cl_p_arr+cl_v_arr)/2:.6f}")
    print("="*70)

    # Save diagnostic results
    np.savez('force_diagnostics_results.npz',
             t=t_arr, cl_tot=cl_tot_arr, cl_react=cl_react_arr, cl_mom=cl_mom_arr,
             cd_tot=cd_tot_arr, cd_react=cd_react_arr, cd_mom=cd_mom_arr,
             t_surf=t_surf, cl_p=cl_p_arr, cl_v=cl_v_arr, cd_p=cd_p_arr, cd_v=cd_v_arr)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t_arr, cl_tot_arr, 'k-', lw=1.5, label='Total $C_L$')
    axes[0].plot(t_arr, cl_react_arr, 'b--', lw=1.2, label='Marker reaction $C_{L,react}$')
    axes[0].plot(t_arr, cl_mom_arr, 'r:', lw=1.2, label='Internal momentum corr $C_{L,mom}$')
    axes[0].set_ylabel('$C_L$')
    axes[0].set_title('Lift Force Component Decomposition (DFG 2D-2)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, cd_tot_arr, 'k-', lw=1.5, label='Total $C_D$')
    axes[1].plot(t_arr, cd_react_arr, 'b--', lw=1.2, label='Marker reaction $C_{D,react}$')
    axes[1].plot(t_arr, cd_mom_arr, 'r:', lw=1.2, label='Internal momentum corr $C_{D,mom}$')
    axes[1].set_ylabel('$C_D$')
    axes[1].set_xlabel('Time $t$ [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('force_decomposition.png', dpi=150)
    plt.close(fig)
    print("Saved force_diagnostics_results.npz and force_decomposition.png")

if __name__ == '__main__':
    run_lift_diagnostics()
