"""Audit all 24 stored pressure sensors; paired sensitivity is not independence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.periodic_window_audit import complete_cycles, window_statistics, file_hash


def statistics(t, cp, window=None):
    t, cp = np.asarray(t, float), np.asarray(cp, float)
    if (t.ndim != 1 or len(t) < 2 or cp.shape != (len(t), 24)
            or not np.isfinite(t).all() or not np.isfinite(cp).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Require increasing times and finite (samples, 24) sensor data')
    if window is not None:
        left,right=window
        values=window_statistics(t,cp,left,right)
        interior=(t>left)&(t<right)
        cp=np.vstack(([np.interp(left,t,cp[:,i]) for i in range(24)],cp[interior],
                      [np.interp(right,t,cp[:,i]) for i in range(24)]))
        return dict(cp_mean=values['mean'],cp_rms=values['rms'],cp_peak_to_peak=np.ptp(cp,axis=0).tolist(),
                    sample_count=len(cp),window=[float(left),float(right)],
                    integration='Piecewise-linear integration over the same complete cycles as force statistics')
    weights = np.diff(t)[:, None]
    duration = t[-1]-t[0]
    mean = np.sum(weights*(cp[:-1]+cp[1:])/2, axis=0)/duration
    centered = cp-mean
    variance = np.sum(weights*(centered[:-1]**2+centered[:-1]*centered[1:]
                                +centered[1:]**2)/3, axis=0)/duration
    return dict(cp_mean=mean.tolist(), cp_rms=np.sqrt(variance).tolist(),
                cp_peak_to_peak=np.ptp(cp, axis=0).tolist(), sample_count=len(t),
                window=[float(t[0]), float(t[-1])],
                integration='Piecewise-linear time interpolation; centered variance avoids cancellation')


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    summaries = sorted(folder.glob('summary-*.json'))
    if not summaries:
        raise ValueError('Require a completed run')
    summary = json.loads(summaries[-1].read_text())
    if (not summary['stationary'] or not np.isfinite([summary['max_cfl'],summary['max_divergence']]).all()
            or summary['max_cfl'] >= .5 or summary['max_divergence'] >= 1e-10):
        raise ValueError('Stationarity/numerical gate is unresolved')
    if summary['source_sha256'] != manifest['metadata']['source_sha256']:
        raise ValueError('Summary and manifest revisions differ')
    rows,recent,previous = [],{},0
    case = manifest['configuration']
    histories = sorted(folder.glob('history-*.jsonl'))
    for history in histories:
        with history.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n'):
                        continue
                    raise
                step=record['step']
                if isinstance(step,bool) or not isinstance(step,int):
                    raise ValueError('Invalid history step')
                canonical_time = step*case['dt']
                if not np.isfinite(record['time']) or abs(record['time']-canonical_time) > 1e-10:
                    raise ValueError('History time disagrees with the configured step')
                cp=np.asarray(record['cp'],float)
                periodic=case['kind']=='periodic'
                force=(record['cd'],record['cl']) if periodic else ()
                if cp.shape!=(24,) or not np.isfinite([*cp,record['pressure_reference'],*force]).all():
                    raise ValueError('Invalid pressure/force observations')
                signature=(tuple(cp),record['pressure_reference'],*force)
                if step<=previous:
                    if recent.get(step)!=signature:
                        raise ValueError('Conflicting or excessive pressure checkpoint replay')
                    continue
                if step!=previous+1:
                    raise ValueError('Incomplete history or missing steps')
                previous=step
                recent[step]=signature
                recent.pop(step-100,None)
                if canonical_time>=case['sample_start']-1e-12:
                    rows.append((canonical_time,cp,record['pressure_reference'],force))
    count = round(case['end_time']/case['dt'])
    if previous!=count:
        raise ValueError('Incomplete history or missing steps')
    data = rows
    if len(data)<2:
        raise ValueError('Insufficient sensor sampling observations')
    t = np.array([row[0] for row in data])
    cp = np.array([row[1] for row in data])
    pressure_reference = np.array([row[2] for row in data])
    if not np.isfinite(pressure_reference).all():
        raise ValueError('Nonfinite pressure reference')
    select = t >= case['sample_start']-1e-12
    if not np.isclose(t[select][0], case['sample_start'], atol=1e-10, rtol=0):
        raise ValueError('Sampling boundary has no recorded observation')
    window=None
    if case['kind']=='periodic':
        forces=np.asarray([row[3] for row in data])
        cycle,crossings=complete_cycles(t,forces[:,0],forces[:,1],case['sample_start'],
            D=case['solver']['D'],U_ref=manifest['metadata']['U_ref'])
        window=(crossings[0],crossings[-1])
    result = statistics(t[select], cp[select]-cp[select, 0:1],window)
    if window is not None:
        result['periodic']=cycle
    dynamic_pressure = .5*case['solver']['rho']*manifest['metadata']['U_ref']**2
    if not np.isfinite(dynamic_pressure) or dynamic_pressure<=0:
        raise ValueError('Require positive finite dynamic pressure')
    result['stored_cp_statistics'] = statistics(t[select], cp[select],window)
    result['absolute_pressure_normalized_statistics'] = statistics(t[select],
        cp[select]+pressure_reference[select, None]/dynamic_pressure,window)
    s = case['solver']
    result['stored_reference_point_yx'] = case.get('pressure_reference_point',
        [s['yc'], max(s['Lx']/s['nx']/2, s['xc']-5*s['D'])])
    result['comparison_reference'] = dict(sensor_index=0, angle_degrees=0,
        point_yx=[s['yc'], s['xc']-s['D']/2],
        definition='2*(p_sensor-p_front)/(rho*U_ref^2); removes the grid-dependent stored reference')
    result.update(run=str(folder), angles_degrees=(np.arange(24)*15).tolist(),
                  angle_convention='Zero upstream; positive toward the upper surface',
                  source_sha256=manifest['metadata']['source_sha256'],
                  history_sha256={p.name:file_hash(p) for p in histories},
                  summary_sha256=hashlib.sha256(summaries[-1].read_bytes()).hexdigest())
    return manifest, result


def compare(folders):
    records = [audit(folder) for folder in folders]
    if len(records) < 2:
        raise ValueError('Require at least two completed grids')
    baseline = records[0][0]
    def fixed(case):
        return ({k:v for k,v in case['solver'].items() if k not in {'nx','ny'}},
                {k:v for k,v in case.items() if k not in
                 {'solver','name','reference','reference_note','reference_source'}})
    for manifest, _ in records[1:]:
        if (manifest['metadata']['source_sha256'] != baseline['metadata']['source_sha256']
                or any(manifest[k] != baseline[k] for k in ('runner_sha256','analysis_sha256'))
                or fixed(manifest['configuration']) != fixed(baseline['configuration'])):
            raise ValueError('Require one revision and inputs differing only in grid resolution')
    spacing = lambda row: max(row[0]['configuration']['solver'][length]/row[0]['configuration']['solver'][n]
                              for length,n in (('Lx','nx'),('Ly','ny')))
    records.sort(key=spacing, reverse=True)
    if len(set(spacing(row) for row in records)) != len(records):
        raise ValueError('Duplicate resolutions do not establish sensitivity')
    differences = []
    for (_, coarse), (_, fine) in zip(records[:-1], records[1:]):
        mean = np.array(fine['cp_mean'])-np.array(coarse['cp_mean'])
        rms = np.array(fine['cp_rms'])-np.array(coarse['cp_rms'])
        index = int(np.argmax(abs(mean)))
        differences.append(dict(coarse_run=coarse['run'], fine_run=fine['run'],
            cp_mean_change=mean.tolist(), cp_rms_change=rms.tolist(),
            maximum_absolute_cp_mean_change=float(abs(mean[index])),
            maximum_change_angle_degrees=index*15,
            maximum_absolute_normalized_pressure_mean_change=float(np.max(np.abs(
                np.array(fine['absolute_pressure_normalized_statistics']['cp_mean'])
                -np.array(coarse['absolute_pressure_normalized_statistics']['cp_mean']))))))
    return dict(runs=[row[1] for row in records], adjacent_grid_differences=differences,
                audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Common front-pressure reference and reconstructed absolute pressure; descriptive sensitivity, not a confidence interval, asymptotic-order proof or sensor acceptance gate.',
                ready_for_paper1=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.runs)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['adjacent_grid_differences']))
