import numpy as np
import matplotlib.pyplot as plt

def plot_mac_diagnostics(solver, step=0, time_val=0.0, save_path=None):
    """
    Diagnostic multi-panel visualization of MAC flow field:
      1. U-velocity (cell-centered interpolation)
      2. V-velocity (cell-centered interpolation)
      3. Pressure field
      4. Vorticity field (derivatives of cell-centred interpolated velocities)
    """
    ny, nx = solver.ny, solver.nx
    
    # Interpolate face velocities to cell centers
    u_c = 0.5 * (solver.u[:, :-1] + solver.u[:, 1:])
    v_c = 0.5 * (solver.v[:-1, :] + solver.v[1:, :])
    
    # Mask out the solid cylinder
    u_c_plot = u_c.copy()
    v_c_plot = v_c.copy()
    p_plot = solver.p.copy()
    
    u_c_plot[solver.mask] = np.nan
    v_c_plot[solver.mask] = np.nan
    p_plot[solver.mask] = np.nan
    
    # Vorticity at cell centers via 2nd-order central differences with physical boundary ghost values
    slip = getattr(solver, 'slip_walls', False)
    u_pad = np.empty((ny + 2, nx), dtype=np.float64)
    u_pad[0, :] = u_c[0, :] if slip else -u_c[0, :]
    u_pad[1:-1, :] = u_c
    u_pad[-1, :] = u_c[-1, :] if slip else -u_c[-1, :]
    
    v_pad = np.empty((ny, nx + 2), dtype=np.float64)
    v_pad[:, 0] = -v_c[:, 0]  # Inflow v=0
    v_pad[:, 1:-1] = v_c
    v_pad[:, -1] = v_c[:, -1]  # Outflow dv/dx = 0
    
    dudy = (u_pad[2:, :] - u_pad[:-2, :]) / (2.0 * solver.dy)
    dvdx = (v_pad[:, 2:] - v_pad[:, :-2]) / (2.0 * solver.dx)
    vorticity = dvdx - dudy
    vorticity[solver.mask] = np.nan

    fig, axs = plt.subplots(2, 2, figsize=(15, 7.5))
    fig.suptitle(f'Textbook MAC Cylinder Flow | Re={solver.Re:.0f} | Time: {time_val:.2f}s (Step {step})', fontsize=15)

    extent = [0, solver.Lx, 0, solver.Ly]

    # 1. U-Velocity
    im0 = axs[0, 0].imshow(u_c_plot, origin='lower', cmap='viridis', extent=extent)
    axs[0, 0].set_title('Horizontal Velocity $u$ [m/s]', fontsize=12)
    axs[0, 0].set_xlabel('$x$ [m]')
    axs[0, 0].set_ylabel('$y$ [m]')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.025, pad=0.04)

    # 2. V-Velocity
    im1 = axs[0, 1].imshow(v_c_plot, origin='lower', cmap='coolwarm', extent=extent)
    axs[0, 1].set_title('Vertical Velocity $v$ [m/s]', fontsize=12)
    axs[0, 1].set_xlabel('$x$ [m]')
    axs[0, 1].set_ylabel('$y$ [m]')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.025, pad=0.04)

    # 3. Pressure
    im2 = axs[1, 0].imshow(p_plot, origin='lower', cmap='plasma', extent=extent)
    axs[1, 0].set_title('Pressure $p$ [Pa]', fontsize=12)
    axs[1, 0].set_xlabel('$x$ [m]')
    axs[1, 0].set_ylabel('$y$ [m]')
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.025, pad=0.04)

    # 4. Vorticity
    vort_clean = vorticity[~np.isnan(vorticity)]
    vort_max = np.percentile(np.abs(vort_clean), 98) if len(vort_clean) > 0 else 5.0
    if np.isnan(vort_max) or vort_max < 1e-3: vort_max = 5.0
    
    im3 = axs[1, 1].imshow(vorticity, origin='lower', cmap='RdBu_r', vmin=-vort_max, vmax=vort_max, extent=extent)
    axs[1, 1].set_title('Vorticity $\\omega_z$ [s$^{-1}$]', fontsize=12)
    axs[1, 1].set_xlabel('$x$ [m]')
    axs[1, 1].set_ylabel('$y$ [m]')
    plt.colorbar(im3, ax=axs[1, 1], fraction=0.025, pad=0.04)

    for ax in axs.flat:
        ax.set_aspect('equal')
        circle = plt.Circle((solver.xc, solver.yc), solver.D / 2.0, color='black', fill=False, linewidth=1.5)
        ax.add_patch(circle)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
