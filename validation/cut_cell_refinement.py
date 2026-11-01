"""Reproducible, fixed-time Cut-Cell refinement study; no reference-acceptance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'solver'))
import numpy as np
from main import MAC_CylinderFlow
from cut_cell_solver import calculate_cut_cell_divergence


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / 'solver/main.py', ROOT / 'solver/cut_cell_geom.py',
                      ROOT / 'solver/cut_cell_solver.py', Path(__file__)]}


def run_case(nx, dt, end_time):
    if not np.isfinite([dt, end_time]).all() or dt <= 0 or end_time <= 0:
        raise ValueError("dt and end_time must be finite and positive")
    steps = round(end_time / dt)
    if nx % 4 or nx < 16 or dt <= 0 or steps < 1 or not np.isclose(steps * dt, end_time):
        raise ValueError('Require nx divisible by four and end_time an integer multiple of positive dt')
    start = time.perf_counter()
    s = MAC_CylinderFlow(nx=nx, ny=nx // 4, boundary='cut_cell', verbose=False)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    times, cd, cl = s.solve(n_steps=steps, dt=dt, perturb_transient=False,
                          save_artifacts=False, verbose=False)
    loop_seconds = time.perf_counter() - start
    front, back = s.cc.pressure_at(s.p, [[s.yc, s.xc - s.D / 2],
                                        [s.yc, s.xc + s.D / 2]])
    force = s.cc.extract_true_surface_forces(s.u, s.v, s.p, s.nu * s.rho)
    scale = .5 * s.rho * s.U_ref**2 * s.D
    mean, maximum, _, un, ut = s.cc.measure_true_surface_slip(s.u, s.v)
    div, _, flux, _, _ = calculate_cut_cell_divergence(s.u, s.v, s.cut_geom)
    result = dict(nx=nx, ny=nx // 4, dt=dt, end_time=s.time,
                  p_front=float(front), p_back=float(back), delta_p=float(front - back),
                  cd=float(force[0] / scale), cd_pressure=float(force[2] / scale),
                  cd_viscous=float(force[4] / scale), cl=float(force[1] / scale),
                  slip_mean=mean, slip_max=maximum,
                  normal_max=float(np.max(abs(un))), tangential_max=float(np.max(abs(ut))),
                  max_divergence=div, max_cell_flux=flux,
                  setup_seconds=setup_seconds, loop_seconds=loop_seconds,
                  metadata=s.metadata(dt, steps, perturb_transient=False))
    numeric = [v for v in result.values() if isinstance(v, (int, float))]
    if not np.isfinite(numeric).all():
        raise FloatingPointError('Non-finite diagnostic output')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, nargs='+', default=[400, 800])
    parser.add_argument('--dt', type=float, nargs='+', default=[.00125, .000625])
    parser.add_argument('--end-time', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new filename to preserve prior evidence')
    hashes = source_hashes()
    report = dict(source_sha256=hashes,
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  purpose='Fixed-time sensitivity study, not developed-flow validation', cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for nx in args.nx:
        for dt in args.dt:
            result = run_case(nx, dt, args.end_time)
            if source_hashes() != hashes:
                raise RuntimeError('Sources changed during study; evidence would mix revisions')
            report['cases'].append(result)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k != 'metadata'}), flush=True)


if __name__ == '__main__':
    main()
