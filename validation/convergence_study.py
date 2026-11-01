import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""Three-grid sensitivity with conditional GCI estimates, not certified error bounds."""
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics, calculate_gci

CASES = ('coarse', 'med', 'fine')
QUANTITIES = ('cd_mean', 'cl_amp', 'strouhal')


def compute_shedding_metrics(t_hist, cd_hist, cl_hist, t_start=87.5, D=1.0, U_ref=1.):
    return periodic_metrics(t_hist, cd_hist, cl_hist, t_start, D, U_ref)


def summarize(data):
    D_val, U_val = 1.0, 1.0
    if 'metadata' in data:
        try:
            meta = json.loads(str(data['metadata']))
            if isinstance(meta, list) and len(meta) > 0:
                D_val = float(meta[0].get('D', 1.0))
                U_val = float(meta[0].get('U_ref', 1.0))
        except Exception:
            pass
    metrics = [compute_shedding_metrics(data[f'{case}_t'], data[f'{case}_cd'], data[f'{case}_cl'],
                                       float(data['t_start']), D=D_val, U_ref=U_val) for case in CASES]
    gcis = {}
    for key in QUANTITIES:
        values = [m[key] for m in metrics]
        gcis[key] = (calculate_gci(*values[::-1]) if np.isfinite(values).all()
                     and all(m['cycles'] >= 2 for m in metrics) else {'status': 'insufficient cycles'})
    return metrics, gcis


def report(metrics, gcis):
    for case, m in zip(CASES, metrics):
        print(f'{case:8s} Cd={m["cd_mean"]:.6f} Cl_half_range={m["cl_amp"]:.6f} '
              f'St={m["strouhal"]:.6f} cycles={m["cycles"]} stationary={m["stationary"]}')
    for key, gci in gcis.items():
        print(f'{key}: {gci}')
    if not all(m['stationary'] for m in metrics):
        print('WARNING: Not all histories pass the 2% late-cycle stationarity screen.')
    print('GCI is conditional: no independent asymptotic-range or fine-grid time-error certification.')


def plot_results(data, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    meta = json.loads(str(data['metadata']))
    for case, m, config in zip(CASES, metrics, meta):
        t, cd, cl = (data[f'{case}_{k}'] for k in ('t', 'cd', 'cl'))
        keep = t >= float(data['t_start'])
        label = f'{config["nx"]} x {config["ny"]}'
        axes[0, 0].plot(t[keep], cd[keep], label=label, lw=1)
        axes[0, 1].plot(t[keep], cl[keep], label=label, lw=1)
    h = [x['Lx']/x['nx'] for x in meta]
    for ax, key, label in zip(axes[1], ('cd_mean', 'strouhal'), ('Mean drag', 'Strouhal')):
        ax.plot(h, [m[key] for m in metrics], 'o-')
        ax.set(xlabel='Grid spacing h [m]', ylabel=label)
    axes[0, 0].set(ylabel='$C_D$', xlabel='Time [s]')
    axes[0, 1].set(ylabel='$C_L$', xlabel='Time [s]')
    for ax in axes[0]:
        ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.suptitle('Grid sensitivity | raw results; no unsupported extrapolation curve')
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/convergence_study.png'), dpi=150)
    plt.close(fig)


def run_convergence_study(n_steps=30000, dt=.0025, quick=False, save_artifacts=True):
    grids = [(200, 50), (400, 100), (800, 200)]
    if quick:
        grids = [(100, 25), (200, 50), (400, 100)]
        n_steps = 50
    data, metadata = {}, []
    for case, (nx, ny) in zip(CASES, grids):
        print(f'Running {case}: {nx} x {ny}, T={n_steps*dt:g}s', flush=True)
        solver = MAC_CylinderFlow(nx=nx, ny=ny, verbose=False)
        start = time.perf_counter()
        t, cd, cl = solver.solve(n_steps=n_steps, dt=dt, save_artifacts=False, verbose=False)
        print(f'Completed {case} in {time.perf_counter()-start:.1f}s', flush=True)
        data.update({f'{case}_t': t, f'{case}_cd': cd, f'{case}_cl': cl})
        config = solver.metadata(dt, n_steps)
        metadata.append(config)
        if save_artifacts and not quick:
            np.savez_compressed(f'grid_{case}.npz', t=t, cd=cd, cl=cl,
                                metadata=json.dumps(config), marker_slip=solver.ib.last_slip)
    data['t_start'] = max(0., n_steps*dt-37.5)
    data['metadata'] = json.dumps(metadata)
    metrics, gcis = summarize(data)
    report(metrics, gcis)
    if save_artifacts and not quick:
        data.update(metrics=json.dumps(metrics), gci=json.dumps(gcis))
        np.savez_compressed(os.path.join(os.path.dirname(__file__), '../data/convergence_results.npz'), **data)
        plot_results(data, metrics)
    return metrics, gcis


def verify_saved_convergence(npz_path=os.path.join(os.path.dirname(__file__), '../data/convergence_results.npz'), plot=False):
    with np.load(npz_path, allow_pickle=False) as data:
        metrics, gcis = summarize(data)
        stored = json.loads(str(data['metrics']))
        for actual, expected in zip(metrics, stored):
            assert actual.keys() == expected.keys()
            np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                                       rtol=1e-10, atol=1e-12, equal_nan=True)
        assert gcis == json.loads(str(data['gci'])), 'Stored GCI does not match raw histories'
        meta = json.loads(str(data['metadata']))
        assert len(meta) == 3
        for coarse, fine in zip(meta[:-1], meta[1:]):
            assert coarse['nx']*2 == fine['nx'] and coarse['ny']*2 == fine['ny']
            for key in ('Lx', 'Ly', 'D', 'xc', 'yc', 'nu', 'rho', 'U_ref', 'inflow', 'slip_walls', 'dt', 'n_steps', 'boundary', 'transport', 'source_sha256'):
                assert coarse[key] == fine[key], f'Inconsistent study input: {key}'
        report(metrics, gcis)
        if plot:
            plot_results(data, metrics)
    print('Archive recomputation: PASS')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='50-step smoke runs; does not overwrite evidence')
    parser.add_argument('--verify', '--check', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.verify or args.plot:
        verify_saved_convergence(plot=args.plot)
    else:
        run_convergence_study(quick=args.quick, save_artifacts=not args.quick)
