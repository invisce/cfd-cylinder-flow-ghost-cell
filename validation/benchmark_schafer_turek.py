import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""DFG 2D-2 comparison: archive integrity, periodicity, and reference agreement are separate."""
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics
from mac_solver import calculate_mac_divergence

REFERENCES = {'cd_max': (3.22, 3.24), 'cl_max': (.99, 1.01), 'strouhal': (.295, .305)}


def analyze_periodic_forces(t_hist, cd_hist, cl_hist, t_start=250., D=1.0, U_mean=1.):
    return periodic_metrics(t_hist, cd_hist, cl_hist, t_start, D, U_mean)


def report(metrics):
    print('DFG 2D-2 numerical comparison (original Schafer & Turek 1996 ranges)')
    for key, (lo, hi) in REFERENCES.items():
        value = metrics[key]
        print(f'{key:12s}: {value:.6f}; reference [{lo}, {hi}]; {"PASS" if lo <= value <= hi else "OUTSIDE"}')
    print(f'Complete cycles: {metrics["cycles"]}; stationarity screen: {metrics["stationary"]}')
    print(f'Amplitude / period / drag drift: {metrics["amplitude_drift"]:.2%} / '
          f'{metrics["period_drift"]:.2%} / {metrics["drag_drift"]:.2%}')
    return metrics['stationary'] and all(lo <= metrics[k] <= hi for k, (lo, hi) in REFERENCES.items())


def plot_result(t, cd, cl, metrics, nx, ny):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    keep = t >= metrics['sample_start']
    for ax, values, key, symbol in zip(axes, (cd, cl), ('cd_max', 'cl_max'), ('D', 'L')):
        ax.plot(t[keep], values[keep], lw=1.2, label=f'Computed $C_{symbol}$')
        ax.axhspan(*REFERENCES[key], color='gray', alpha=.25, label='Reference maximum range')
        ax.set_ylabel(f'$C_{symbol}$')
        ax.grid(alpha=.25)
        ax.legend(loc='upper right')
    axes[0].set_title(f'DFG 2D-2 comparison | {nx} x {ny} | St={metrics["strouhal"]:.4f}')
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__), '../figures/benchmark_schafer_turek.png'), dpi=150)
    plt.close(fig)


def run_schafer_turek_benchmark(n_steps=60000, dt=.005, nx=440, ny=82, save_artifacts=True):
    solver = MAC_CylinderFlow(nx=nx, ny=ny, Lx=22.0, Ly=4.1, D=1.0, xc=2.0, yc=2.0,
                             nu=.01, inflow='parabolic', U_m=1.5)
    start = time.perf_counter()
    t, cd, cl = solver.solve(n_steps=n_steps, dt=dt, perturb_transient=False, save_artifacts=False)
    elapsed = time.perf_counter()-start
    t_start = max(0., t[-1]-50.)
    metrics = analyze_periodic_forces(t, cd, cl, t_start)
    max_div, div2, count, _, _ = calculate_mac_divergence(solver.u, solver.v, solver.dx, solver.dy, solver.mask)
    div_rms = np.sqrt(div2/count)
    flux_err = abs(np.sum(solver.u[:, -1])-np.sum(solver.u[:, 0]))/abs(np.sum(solver.u[:, 0]))
    report(metrics)
    print(f'Divergence RMS={div_rms:.3e}, max={max_div:.3e}, relative flux mismatch={flux_err:.3e}')
    if save_artifacts:
        np.savez_compressed(os.path.join(os.path.dirname(__file__), '../data/schafer_turek_results.npz'), t=t, cd=cd, cl=cl,
                            u=solver.u, v=solver.v, p=solver.p, mask=solver.mask,
                            **metrics, t_start=t_start, div_rms=div_rms, flux_err=flux_err,
                            marker_slip=solver.ib.last_slip,
                            elapsed=elapsed, metadata=json.dumps(solver.metadata(dt, n_steps, False)))
        plot_result(t, cd, cl, metrics, nx, ny)
    return metrics


def verify_saved_schafer_turek(npz_path=os.path.join(os.path.dirname(__file__), '../data/schafer_turek_results.npz'), plot=False, require_reference=False):
    with np.load(npz_path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        D_val = float(meta.get('D', 1.0))
        U_val = float(meta.get('U_ref', meta.get('U_mean', 1.0)))
        metrics = analyze_periodic_forces(data['t'], data['cd'], data['cl'], float(data['t_start']), D=D_val, U_mean=U_val)
        for key, value in metrics.items():
            np.testing.assert_allclose(data[key], value, rtol=1e-10, atol=1e-12, equal_nan=True)
        u, v, p, mask = (data[k] for k in ('u', 'v', 'p', 'mask'))
        assert u.shape == (meta['ny'], meta['nx']+1) and v.shape == (meta['ny']+1, meta['nx'])
        assert p.shape == mask.shape == (meta['ny'], meta['nx'])
        assert all(np.isfinite(a).all() for a in (u, v, p))
        dx, dy = meta['Lx']/meta['nx'], meta['Ly']/meta['ny']
        div = ((u[:, 1:]-u[:, :-1])/dx+(v[1:]-v[:-1])/dy)[~mask]
        rms = np.sqrt(np.mean(div**2))
        flux = abs(u[:, -1].sum()-u[:, 0].sum())/abs(u[:, 0].sum())
        np.testing.assert_allclose([data['div_rms'], data['flux_err']], [rms, flux], atol=1e-12)
        assert rms < 1e-10 and flux < 1e-10, 'Conservation check failed'
        from immersed_boundary import CircularProjection
        ib = CircularProjection(meta['nx'], meta['ny'], dx, dy, meta['xc'], meta['yc'], meta['D'])
        slip = np.max(np.abs(ib.E @ ib.pack(u, v)))
        assert slip < 1e-10, 'Cylinder no-slip check failed'
        np.testing.assert_allclose(data['marker_slip'], slip, atol=1e-12)
        passed = report(metrics)
        if plot:
            plot_result(data['t'], data['cd'], data['cl'], metrics, meta['nx'], meta['ny'])
    print('Archive recomputation and conservation: PASS (not the same as reference acceptance)')
    return passed if require_reference else True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='1-second smoke run; no benchmark claim or saved artifacts')
    parser.add_argument('--verify', '--check', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Regenerate plot from verified archive')
    parser.add_argument('--require-reference', action='store_true', help='Exit nonzero unless periodic and all reference ranges pass')
    args = parser.parse_args()
    if args.verify or args.plot:
        raise SystemExit(0 if verify_saved_schafer_turek(plot=args.plot, require_reference=args.require_reference) else 1)
    metrics = run_schafer_turek_benchmark(n_steps=2000 if args.quick else 60000, save_artifacts=not args.quick)
    if args.require_reference:
        raise SystemExit(0 if report(metrics) else 1)
