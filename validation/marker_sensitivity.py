import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""
Marker-resolution sensitivity study for DFG 2D-2 benchmark at Re=100.
Compares marker arc spacing:
  - 1.0 h (fine marker resolution)
  - 1.5 h (baseline resolution)
  - 2.0 h (coarse marker resolution)
Fixed grid 440 x 82, dt = 0.0005 s, T = 15.0 s (30,000 steps).
Samples periodic shedding across t in [10.0, 15.0] s.
"""
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics
from mac_solver import calculate_mac_divergence

FACTORS = [1.0, 1.5, 2.0]
LABELS = ['1.0h', '1.5h', '2.0h']

def run_marker_study(n_steps=30000, dt=0.005, save_artifacts=True):
    results = {}
    print("="*75)
    print("MARKER RESOLUTION SENSITIVITY STUDY (DFG 2D-2, Re=100)")
    print("="*75)
    
    for factor, label in zip(FACTORS, LABELS):
        print(f"\nRunning marker spacing {label} (factor={factor})...", flush=True)
        solver = MAC_CylinderFlow(nx=440, ny=82, Lx=22.0, Ly=4.1, D=1.0, xc=2.0, yc=2.0,
                                 nu=0.01, inflow='parabolic', U_m=1.5, verbose=False,
                                 marker_spacing_factor=factor)
        n_markers = solver.ib.n_markers
        print(f"Grid: 440x82, h={solver.dx:.5f}m, Markers: {n_markers}, Spacing: {np.pi*solver.D/n_markers:.5f}m ({factor:.1f}h)")
        
        t0 = time.perf_counter()
        t, cd, cl = solver.solve(n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False, verbose=False)
        elapsed = time.perf_counter() - t0
        print(f"Completed in {elapsed:.1f}s ({elapsed/n_steps*1000:.2f} ms/step)")
        
        # Periodic metrics over last 50 seconds (t >= 100.0)
        m = periodic_metrics(t, cd, cl, t_start=100.0, D=1.0, U_ref=1.0)
        max_div, div2, count, _, _ = calculate_mac_divergence(solver.u, solver.v, solver.dx, solver.dy, solver.mask)
        div_rms = np.sqrt(div2 / count)
        flux_err = abs(np.sum(solver.u[:, -1]) - np.sum(solver.u[:, 0])) / abs(np.sum(solver.u[:, 0]))
        slip = solver.ib.last_slip
        
        results[label] = {
            'factor': factor,
            'n_markers': n_markers,
            'cd_max': m['cd_max'],
            'cd_min': m['cd_min'],
            'cd_mean': m['cd_mean'],
            'cl_max': m['cl_max'],
            'cl_min': m['cl_min'],
            'cl_amp': m['cl_amp'],
            'strouhal': m['strouhal'],
            'period': m['period'],
            'cycles': m['cycles'],
            'stationary': m['stationary'],
            'div_rms': div_rms,
            'flux_err': flux_err,
            'slip': slip,
            'elapsed': elapsed,
            't': t,
            'cd': cd,
            'cl': cl
        }
        
        print(f"Results for {label}:")
        print(f"  Cd_max = {m['cd_max']:.6f} | Cl_max = {m['cl_max']:.6f} | Cl_amp = {m['cl_amp']:.6f} | St = {m['strouhal']:.6f}")
        print(f"  Cycles = {m['cycles']} | Stationary = {m['stationary']} | Slip = {slip:.2e} m/s | Div_RMS = {div_rms:.2e}")

    print("\n" + "="*75)
    print("SUMMARY COMPARISON ACROSS MARKER RESOLUTIONS")
    print("="*75)
    print(f"{'Spacing':<10} {'Markers':<10} {'Cd_max':<12} {'Cl_max':<12} {'Cl_amp':<12} {'Strouhal':<10} {'Slip (m/s)':<12}")
    print("-"*75)
    for label in LABELS:
        r = results[label]
        print(f"{label:<10} {r['n_markers']:<10d} {r['cd_max']:<12.6f} {r['cl_max']:<12.6f} {r['cl_amp']:<12.6f} {r['strouhal']:<10.6f} {r['slip']:<12.2e}")
    print("="*75)

    if save_artifacts:
        # Save npz
        save_dict = {}
        for label in LABELS:
            for k in ('factor', 'n_markers', 'cd_max', 'cd_min', 'cd_mean', 'cl_max', 'cl_min', 'cl_amp',
                      'strouhal', 'period', 'cycles', 'stationary', 'div_rms', 'flux_err', 'slip', 'elapsed'):
                save_dict[f'{label}_{k}'] = results[label][k]
            save_dict[f'{label}_t'] = results[label]['t']
            save_dict[f'{label}_cd'] = results[label]['cd']
            save_dict[f'{label}_cl'] = results[label]['cl']
        np.savez_compressed(os.path.join(os.path.dirname(__file__), '../data/marker_sensitivity_results.npz'), **save_dict)
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for label, style in zip(LABELS, ['b-', 'k--', 'r:']):
            t = results[label]['t']
            keep = t >= 10.0
            axes[0].plot(t[keep], results[label]['cd'][keep], style, lw=1.2,
                         label=f'{label} ({results[label]["n_markers"]} markers, max={results[label]["cd_max"]:.3f})')
            axes[1].plot(t[keep], results[label]['cl'][keep], style, lw=1.2,
                         label=f'{label} ({results[label]["n_markers"]} markers, max={results[label]["cl_max"]:.3f})')
        axes[0].set_ylabel('$C_D$')
        axes[0].set_title('Marker Spacing Sensitivity (DFG 2D-2, 440x82 grid)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('$C_L$')
        axes[1].set_xlabel('Time $t$ [s]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/marker_sensitivity.png'), dpi=150)
        plt.close(fig)
        print("Saved marker_sensitivity_results.npz and marker_sensitivity.png")
        
    return results

if __name__ == '__main__':
    run_marker_study()
