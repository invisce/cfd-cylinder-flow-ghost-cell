import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""Time-step sensitivity at fixed grid and physical duration; spatial sensitivity is in convergence_study.py."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics


def compute_vortex_metrics(t_hist, cd_hist, cl_hist, t_start=87.5, D=1.0, U_inf=1.):
    return periodic_metrics(t_hist, cd_hist, cl_hist, t_start, D, U_inf)


def summarize(data):
    D_val, U_val = 1.0, 1.0
    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            if isinstance(meta, list) and len(meta) > 0:
                D_val = float(meta[0].get('D', 1.0))
                U_val = float(meta[0].get('U_ref', meta[0].get('U_inf', 1.0)))
        except Exception:
            pass
    return [compute_vortex_metrics(data[f't_{case}'], data[f'cd_{case}'], data[f'cl_{case}'],
                                  float(data['t_start']), D=D_val, U_inf=U_val) for case in ('base', 'a')]


def report(metrics):
    for label, m in zip(('dt=0.002', 'dt=0.001'), metrics):
        print(f'{label} Cd={m["cd_mean"]:.6f}, Cl_half_range={m["cl_amp"]:.6f}, '
              f'St={m["strouhal"]:.6f}, cycles={m["cycles"]}, stationary={m["stationary"]}')
    for key in ('cd_mean', 'cl_amp', 'strouhal'):
        a, b = (m[key] for m in metrics)
        print(f'{key} change relative to smaller dt: {abs(a-b)/max(abs(b), 1e-12):.3%}')
    print('Two timesteps measure sensitivity; they do not establish a temporal order or error bound.')


def plot_results(data):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for case, dt in (('base', .002), ('a', .001)):
        t = data[f't_{case}']
        keep = t >= float(data['t_start'])
        for ax, key in zip(axes, ('cd', 'cl')):
            ax.plot(t[keep], data[f'{key}_{case}'][keep], label=f'dt={dt}', lw=1.1)
            ax.legend()
            ax.grid(alpha=.25)
    axes[0].set(ylabel='$C_D$', title='Time-step sensitivity | fixed 400 x 100 grid')
    axes[1].set(ylabel='$C_L$', xlabel='Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/sensitivity_comparison.png'), dpi=150)
    plt.close(fig)


def run_sensitivity_study(n_steps_base=15000, save_artifacts=True):
    data, meta = {}, []
    for case, dt, steps in (('base', .005, n_steps_base), ('a', .0025, 2*n_steps_base)):
        print(f'Running {case}: dt={dt}, T={dt*steps:g}s', flush=True)
        solver = MAC_CylinderFlow(nx=400, ny=100, verbose=False)
        config = solver.metadata(dt, steps)
        config['end_time'] = dt*steps
        reused = False
        if case == 'a' and Path('grid_med.npz').exists():
            with np.load('grid_med.npz', allow_pickle=False) as saved:
                if json.loads(str(saved['metadata'])) == config:
                    t, cd, cl = (saved[k].copy() for k in ('t', 'cd', 'cl'))
                    reused = True
                    print('Reusing completed medium-grid case: all inputs and source hashes match.', flush=True)
        if not reused:
            t, cd, cl = solver.solve(n_steps=steps, dt=dt, save_artifacts=save_artifacts and case == 'base', verbose=False)
            config = solver.metadata(dt, steps)
        data.update({f't_{case}': t, f'cd_{case}': cd, f'cl_{case}': cl})
        meta.append(config)
    data['metadata'] = json.dumps(meta)
    data['t_start'] = max(0., n_steps_base*.005-37.5)
    metrics = summarize(data)
    report(metrics)
    if save_artifacts:
        data['metrics'] = json.dumps(metrics)
        np.savez_compressed(os.path.join(os.path.dirname(__file__), '../data/sensitivity_results.npz'), **data)
        plot_results(data)
    return metrics


def verify_saved_sensitivity(npz_path=os.path.join(os.path.dirname(__file__), '../data/sensitivity_results.npz'), plot=False):
    with np.load(npz_path, allow_pickle=False) as data:
        metrics = summarize(data)
        stored = json.loads(str(data['metrics']))
        assert len(stored) == len(metrics)
        for actual, expected in zip(metrics, stored):
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        a, b = json.loads(str(data['metadata']))
        for key in ('nx', 'ny', 'Lx', 'Ly', 'D', 'xc', 'yc', 'nu', 'rho', 'U_ref', 'inflow', 'slip_walls', 'source_sha256'):
            assert a[key] == b[key], f'Inconsistent study input: {key}'
        assert a['dt'] == 2*b['dt'] and a['dt']*a['n_steps'] == b['dt']*b['n_steps']
        report(metrics)
        if plot:
            plot_results(data)
    print('Archive recomputation: PASS')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Short smoke runs; no saved evidence')
    parser.add_argument('--verify', '--check', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.verify or args.plot:
        verify_saved_sensitivity(plot=args.plot)
    else:
        run_sensitivity_study(n_steps_base=25 if args.quick else 15000, save_artifacts=not args.quick)
