"""Run one explicit Cut-Cell case with resource approval, immutable histories and restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'solver'))
import numpy as np
import psutil
from main import MAC_CylinderFlow
from cut_cell_solver import build_pressure_reconstruction, reconstruct_pressure
from postprocessing.analysis import periodic_metrics, window_statistics


class ResourceLimit(RuntimeError):
    pass


def analysis_hash():
    return hashlib.sha256((ROOT/'postprocessing/analysis.py').read_bytes()).hexdigest()


def snapshot_sources(output, manifest):
    files = []
    for name, expected in manifest['metadata']['source_sha256'].items():
        path = (Path(sys.modules['mac_engine'].__file__) if name == 'compiled_engine'
                else ROOT/'solver'/name)
        files.append((path, expected))
    files += [(Path(__file__), manifest['runner_sha256']),
              (ROOT/'postprocessing/analysis.py', manifest['analysis_sha256'])]
    for path, expected in files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed before snapshot creation')
    with zipfile.ZipFile(output/'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, _ in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('case.json', json.dumps(manifest['configuration'], indent=2))


def preflight(case):
    config, dt, end = case['solver'], case['dt'], case['end_time']
    nx, ny = config['nx'], config['ny']
    if (any(isinstance(n, bool) or not isinstance(n, int) or n < 4 for n in (nx, ny))
            or not np.isfinite([dt, end]).all() or min(dt, end) <= 0
            or not np.isclose(round(end/dt)*dt, end)):
        raise ValueError('Invalid grid, timestep or final time')
    if config.get('boundary') != 'cut_cell' or config.get('time_scheme') != 'ab2_cn':
        raise ValueError('Stage-zero production cases must explicitly select Cut-Cell and AB2/CN')
    if case['kind'] not in ('steady', 'periodic', 'diagnostic'):
        raise ValueError('Unknown case kind')
    if not 0 <= case['sample_start'] < end:
        raise ValueError('Sampling must start before the end of the run')
    cells = nx*ny
    mib = 192+2800*cells/2**20
    seconds = round(end/dt)*(.02+1e-5*cells)
    return dict(estimated_peak_mib=mib, estimated_seconds=seconds,
                heavy=nx >= 800 or ny >= 200 or mib > 512 or seconds > 300,
                available_mib=psutil.virtual_memory().available/2**20,
                note='Conservative estimate from small-grid timings; not a performance guarantee')


def summarize(folder, case):
    # Later restart segments supersede earlier records with the same integer step.
    records = {}
    for file in sorted(folder.glob('history-*.jsonl')):
        with file.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # A process/power interruption may tear the final write.
                    # Only an incomplete final record is recoverable; a malformed
                    # newline-terminated record remains an evidence error.
                    if not line.endswith('\n'):
                        continue
                    raise
                records[item['step']] = tuple(item[k] for k in ('time', 'cd', 'cl', 'delta_p', 'cfl', 'max_divergence'))
    if not records:
        return dict(accepted=False, ready_for_paper1=False, reason='No complete samples')
    data = np.array([records[k] for k in sorted(records)])
    t, cd, cl, dp, cfl, div = data.T
    max_divergence = float(div.max())
    max_cfl = float(cfl.max())

    if case['kind'] == 'periodic':
        periodic = periodic_metrics(t, cd, cl, case['sample_start'],
                                    D=case.get('solver', {}).get('D', 1.),
                                    U_ref=case.get('solver', {}).get('U_ref', 1.))
        if periodic['cycles'] >= 2:
            left, right = periodic['sample_start'], periodic['sample_end']
        else:
            select = t >= case['sample_start']
            if np.sum(select) < 4:
                return dict(accepted=False, ready_for_paper1=False, reason='Insufficient samples')
            left, right = float(t[select][0]), float(t[select][-1])

        stats = window_statistics(t, np.column_stack((cd, cl, dp)), left, right)
        stable = (periodic['cycles'] >= 20 and
                  max(periodic[k] for k in ('amplitude_drift', 'period_drift', 'drag_drift')) < .005)

        result = dict(
            cd_mean=float(stats['mean'][0]),
            cl_mean=float(stats['mean'][1]),
            delta_p=float(stats['mean'][2]),
            cl_rms=float(stats['rms'][1]),
            cd_rms=float(stats['rms'][0]),
            delta_p_rms=float(stats['rms'][2]),
            cd_max=float(stats['maximum'][0]),
            cd_min=float(stats['minimum'][0]),
            cl_max=float(stats['maximum'][1]),
            cl_min=float(stats['minimum'][1]),
            delta_p_max=float(stats['maximum'][2]),
            delta_p_min=float(stats['minimum'][2]),
            cd_mid_extrema=float(stats['mid_extrema'][0]),
            cl_mid_extrema=float(stats['mid_extrema'][1]),
            delta_p_mid_extrema=float(stats['mid_extrema'][2]),
            cd_half_amplitude=float(stats['half_amplitude'][0]),
            cl_half_amplitude=float(stats['half_amplitude'][1]),
            delta_p_half_amplitude=float(stats['half_amplitude'][2]),
            cl_amp=float(stats['half_amplitude'][1]),
            cd_amplitude=float(stats['amplitude'][0]),
            cl_amplitude=float(stats['amplitude'][1]),
            delta_p_amplitude=float(stats['amplitude'][2]),
            period=periodic['period'],
            strouhal=periodic['strouhal'],
            cycles=periodic['cycles'],
            sample_start=float(left),
            sample_end=float(right),
            amplitude_drift=periodic['amplitude_drift'],
            period_drift=periodic['period_drift'],
            drag_drift=periodic['drag_drift'],
            max_divergence=max_divergence,
            max_cfl=max_cfl
        )
    else:
        select = t >= case['sample_start']
        ts = t[select]
        if len(ts) < 4:
            return dict(accepted=False, ready_for_paper1=False, reason='Insufficient samples')
        left, right = float(ts[0]), float(ts[-1])
        stats = window_statistics(t, np.column_stack((cd, cl, dp)), left, right)

        ds, ls, ps = cd[select], cl[select], dp[select]
        half = len(ts) // 2
        drift = max(abs(ds[:half].mean() - ds[half:].mean()) / max(abs(ds[half:].mean()), 1e-12),
                    abs(ps[:half].mean() - ps[half:].mean()) / max(abs(ps[half:].mean()), 1e-12))
        stable = drift < .001 and np.ptp(ls) < 1e-4

        result = dict(
            cd_mean=float(stats['mean'][0]),
            cl_mean=float(stats['mean'][1]),
            delta_p=float(stats['mean'][2]),
            cl_rms=float(stats['rms'][1]),
            cd_rms=float(stats['rms'][0]),
            delta_p_rms=float(stats['rms'][2]),
            cd_max=float(stats['maximum'][0]),
            cd_min=float(stats['minimum'][0]),
            cl_max=float(stats['maximum'][1]),
            cl_min=float(stats['minimum'][1]),
            delta_p_max=float(stats['maximum'][2]),
            delta_p_min=float(stats['minimum'][2]),
            cd_mid_extrema=float(stats['mid_extrema'][0]),
            cl_mid_extrema=float(stats['mid_extrema'][1]),
            delta_p_mid_extrema=float(stats['mid_extrema'][2]),
            cd_half_amplitude=float(stats['half_amplitude'][0]),
            cl_half_amplitude=float(stats['half_amplitude'][1]),
            delta_p_half_amplitude=float(stats['half_amplitude'][2]),
            cl_amp=float(stats['half_amplitude'][1]),
            cd_amplitude=float(stats['amplitude'][0]),
            cl_amplitude=float(stats['amplitude'][1]),
            delta_p_amplitude=float(stats['amplitude'][2]),
            sample_start=float(left),
            sample_end=float(right),
            steady_drift=float(drift),
            max_divergence=max_divergence,
            max_cfl=max_cfl
        )
    reference_checks = {}
    for name, rule in case.get('reference', {}).items():
        value = result.get(name)
        reference_checks[name] = bool(value is not None and np.isfinite(value) and rule[0] <= value <= rule[1])
    result.update(stationary=bool(stable), reference_checks=reference_checks,
                  accepted=bool(stable and reference_checks and all(reference_checks.values())
                                and div.max() < 1e-10 and cfl.max() < .5),
                  acceptance_scope='Numerical reference/stationarity checks only; boundary-condition equivalence and independence require separate evidence',
                  ready_for_paper1=False)
    return {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in result.items()}


def run_case(case, output, *, approved_heavy=False, resume=False, budget_mib=None, budget_seconds=None):
    estimate = preflight(case)
    if estimate['heavy'] and not approved_heavy:
        raise PermissionError('Heavy case: announce resources and wait for the user to say ready')
    budget_mib = estimate['estimated_peak_mib'] if budget_mib is None else budget_mib
    budget_seconds = max(60., estimate['estimated_seconds']) if budget_seconds is None else budget_seconds
    if not np.isfinite([budget_mib, budget_seconds]).all() or min(budget_mib, budget_seconds) <= 0:
        raise ValueError('Resource budgets must be finite and positive')
    # A user ceiling is not an allocation request (e.g. a 4 GiB ceiling for a
    # 216 MiB case). Reserve the estimated working set; enforce the ceiling
    # and the live free-memory reserve separately while stepping.
    if estimate['available_mib'] < estimate['estimated_peak_mib']+256:
        raise ResourceLimit('Insufficient free RAM for the estimated working set plus 256 MiB reserve')
    output = Path(output)
    config_hash = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    if resume:
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['configuration_sha256'] != config_hash:
            raise ValueError('Resume configuration differs from the saved run')
        solver = MAC_CylinderFlow.from_checkpoint(output/'checkpoint.npz')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Choose an empty run directory; previous evidence is preserved')
        output.mkdir(parents=True, exist_ok=True)
        solver = MAC_CylinderFlow(**case['solver'])
        manifest = dict(configuration=case, configuration_sha256=config_hash,
                        metadata=solver.metadata(case['dt'], round(case['end_time']/case['dt'])),
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        analysis_sha256=analysis_hash(),
                        estimate=estimate)
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        snapshot_sources(output, manifest)
    if manifest['runner_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Runner changed since the saved run')
    if manifest.get('analysis_sha256') != analysis_hash():
        raise ValueError('Analysis code changed since the saved run')
    def verify_sources():
        if (manifest['metadata']['source_sha256'] != solver.metadata(case['dt'], 0)['source_sha256']
                or manifest['analysis_sha256'] != analysis_hash()
                or manifest['runner_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise RuntimeError('Sources changed during the run; mixed-revision evidence is rejected')
    angles = np.arange(24)*np.pi/12
    # Zero degrees is upstream, 90 degrees is the upper side.
    sensor_points = np.column_stack((solver.yc+solver.D/2*np.sin(angles), solver.xc-solver.D/2*np.cos(angles)))
    reference_point = case.get('pressure_reference_point', [solver.yc, max(solver.dx/2, solver.xc-5*solver.D)])
    sensor_operator = build_pressure_reconstruction(solver.cut_geom, np.vstack((sensor_points, reference_point)))
    dt = case['dt']
    remaining = round((case['end_time']-solver.time)/dt)
    if remaining <= 0:
        return summarize(output, case)
    segment = len(list(output.glob('history-*.jsonl')))
    start = time.perf_counter()
    peak_mib = psutil.Process().memory_info().rss/2**20
    history_path = output/f'history-{segment:04d}.jsonl'
    with history_path.open('x', encoding='utf-8') as history:
        def record(s, values):
            nonlocal peak_mib
            pressures = reconstruct_pressure(s.p, sensor_operator)
            values.update(step=round(s.time/dt), delta_p=float(pressures[0]-pressures[12]),
                          cp=((pressures[:24]-pressures[-1])/(.5*s.rho*s.U_ref**2)).tolist(),
                          pressure_reference=float(pressures[-1]))
            rss = psutil.Process().memory_info().rss/2**20
            peak_mib = max(peak_mib, rss)
            values.update(rss_mib=rss, elapsed_seconds=time.perf_counter()-start)
            history.write(json.dumps(values, allow_nan=False)+'\n')
            if values['step'] % 100 == 0:
                history.flush()
                verify_sources()
                s.save_checkpoint(output/'checkpoint.npz', overwrite=True)
            if rss > budget_mib or values['elapsed_seconds'] > budget_seconds or psutil.virtual_memory().available < 256*2**20:
                s.save_checkpoint(output/'checkpoint.npz', overwrite=True)
                raise ResourceLimit('Run checkpointed: resource budget exceeded; renewed user readiness is required')
        try:
            solver.solve(n_steps=remaining, dt=dt, save_artifacts=False, verbose=False,
                         perturb_transient=case.get('perturb_transient', False), step_callback=record)
        except Exception as error:
            (output/f'stopped-{segment:04d}.json').write_text(json.dumps(dict(
                reason=str(error), last_time=solver.time, peak_mib=peak_mib,
                ready_for_paper1=False), indent=2), encoding='utf-8')
            raise
    verify_sources()
    solver.save_checkpoint(output/'checkpoint.npz', overwrite=True)
    result = summarize(output, case)
    result.update(peak_mib=peak_mib, elapsed_seconds=time.perf_counter()-start,
                  source_sha256=solver.metadata(dt, remaining)['source_sha256'])
    (output/f'summary-{segment:04d}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true', help='Without this switch only preflight runs')
    parser.add_argument('--approved-heavy', action='store_true', help='Use only after explicit user readiness')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--budget-mib', type=float)
    parser.add_argument('--budget-seconds', type=float)
    args = parser.parse_args()
    case = json.loads(args.config.read_text(encoding='utf-8'))
    print(json.dumps(preflight(case), indent=2))
    if args.run:
        if args.output is None:
            parser.error('--run requires --output')
        print(json.dumps(run_case(case, args.output, approved_heavy=args.approved_heavy,
            resume=args.resume, budget_mib=args.budget_mib, budget_seconds=args.budget_seconds), indent=2))
