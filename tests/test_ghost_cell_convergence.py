"""
Verification of Stage 0.3: Formal Boundary Error Convergence for Sharp-Interface Ghost-Cell Formulation.
Runs a 3-grid convergence study (coarse: 100x25, medium: 200x50, fine: 400x100)
and verifies monotonic reduction of physical boundary slip error and machine-zero divergence.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from main import MAC_CylinderFlow
from mac_solver import calculate_mac_divergence

def verify_ghost_cell_boundary_convergence():
    print("=" * 70)
    print("STAGE 0.3 VERIFICATION: Ghost-Cell Boundary Slip Error Convergence")
    print("=" * 70)

    grids = [
        (100, 25, "Coarse (100x25)"),
        (200, 50, "Medium (200x50)"),
        (400, 100, "Fine   (400x100)")
    ]

    h_list = []
    mean_slip_list = []
    max_slip_list = []
    max_div_list = []
    cd_list = []

    for nx, ny, label in grids:
        dx = 20.0 / nx
        dy = 5.0 / ny
        print(f"\nRunning {label}: dx={dx:.4f}, dy={dy:.4f}...")
        solver = MAC_CylinderFlow(
            nx=nx, ny=ny, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5, nu=0.01,
            boundary='ghost_cell', pressure_solver='direct', verbose=False
        )
        t, cd, cl = solver.solve(
            n_steps=100, dt=0.005, perturb_transient=False, save_artifacts=False, verbose=False
        )

        # 1. Divergence on fluid cells
        max_div, div2, count, _, _ = calculate_mac_divergence(
            solver.u, solver.v, solver.dx, solver.dy, solver.mask
        )

        # 2. Surface slip on true physical cylinder circle r = R
        gc = solver.gc
        u_interp = RegularGridInterpolator((gc.yu, gc.xu), solver.u, bounds_error=False, fill_value=0.0)
        v_interp = RegularGridInterpolator((gc.yv, gc.xv), solver.v, bounds_error=False, fill_value=0.0)
        pts = np.column_stack((gc.surf_y, gc.surf_x))
        u_b = u_interp(pts)
        v_b = v_interp(pts)
        speed_b = np.sqrt(u_b**2 + v_b**2)
        mean_slip = np.mean(speed_b)
        max_slip = np.max(speed_b)

        h_list.append(dx)
        mean_slip_list.append(mean_slip)
        max_slip_list.append(max_slip)
        max_div_list.append(max_div)
        cd_list.append(cd[-1])

        print(f"  Max Divergence : {max_div:.2e} (Machine Zero)")
        print(f"  Mean Slip (L1) : {mean_slip:.4f}")
        print(f"  Max Slip (Linf): {max_slip:.4f}")
        print(f"  Drag Coeff Cd  : {cd[-1]:.4f}")

    # Compute observed order of convergence p = log2(e_coarse / e_fine)
    p_mean_1 = np.log2(mean_slip_list[0] / mean_slip_list[1])
    p_mean_2 = np.log2(mean_slip_list[1] / mean_slip_list[2])
    p_max_2 = np.log2(max_slip_list[1] / max_slip_list[2])

    print("\n" + "-" * 70)
    print("CONVERGENCE SUMMARY TABLE:")
    print(f"{'Mesh':<18} | {'h (dx)':<8} | {'Mean Slip (L1)':<15} | {'Max Slip (Linf)':<15} | {'Max Div':<10}")
    print("-" * 70)
    for k, (nx, ny, label) in enumerate(grids):
        print(f"{label:<18} | {h_list[k]:<8.4f} | {mean_slip_list[k]:<15.4f} | {max_slip_list[k]:<15.4f} | {max_div_list[k]:<10.2e}")
    print("-" * 70)
    print(f"Observed Order of Convergence (Mean Slip, Coarse->Medium): p = {p_mean_1:.2f}")
    print(f"Observed Order of Convergence (Mean Slip, Medium->Fine)  : p = {p_mean_2:.2f}")
    print(f"Observed Order of Convergence (Max Slip,  Medium->Fine)  : p = {p_max_2:.2f}")
    print("-" * 70)

    # Formal Assertions
    # 1. Divergence must remain machine zero on all meshes
    for k, div in enumerate(max_div_list):
        assert div < 1e-13, f"Mesh {grids[k][0]}x{grids[k][1]} divergence too high: {div}"

    # 2. Mean slip and max slip must decrease strictly monotonically
    assert mean_slip_list[1] < mean_slip_list[0], "Mean slip did not decrease from coarse to medium"
    assert mean_slip_list[2] < mean_slip_list[1], "Mean slip did not decrease from medium to fine"
    assert max_slip_list[1] < max_slip_list[0], "Max slip did not decrease from coarse to medium"
    assert max_slip_list[2] < max_slip_list[1], "Max slip did not decrease from medium to fine"

    # 3. Asymptotic order of convergence p >= 0.75
    assert p_mean_2 >= 0.75, f"Asymptotic order p_mean={p_mean_2:.2f} below 0.75"
    assert p_max_2 >= 0.75, f"Asymptotic order p_max={p_max_2:.2f} below 0.75"

    print("\n[SUCCESS] Ghost-Cell sharp boundary achieves formal monotonic error convergence and machine-zero divergence!\n")

if __name__ == '__main__':
    verify_ghost_cell_boundary_convergence()
