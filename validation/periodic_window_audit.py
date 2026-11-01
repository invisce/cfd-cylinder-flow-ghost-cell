"""Compare force and pressure statistics over the SAME complete-cycle window.

Cycle dispersion describes the recorded deterministic signal, not a confidence
interval for independent random observations. Stored production data stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from postprocessing.analysis import periodic_metrics, window_statistics


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def complete_cycles(t,cd,cl,t_start,D=1.,U_ref=1.):
    result = periodic_metrics(t,cd,cl,t_start,D=D,U_ref=U_ref)
    if not result['stationary'] or result['cycles'] < 4:
        raise ValueError('Require at least four resolved, stationary cycles')
    select = np.asarray(t) >= t_start
    ts,ls = np.asarray(t)[select],np.asarray(cl)[select]
    centered = ls-np.trapezoid(ls,ts)/(ts[-1]-ts[0])
    idx = np.flatnonzero((centered[:-1] < 0)&(centered[1:] >= 0))
    crossings = ts[idx]-centered[idx]*np.diff(ts)[idx]/np.diff(centered)[idx]
    if not np.allclose([crossings[0],crossings[-1]],
                       [result['sample_start'],result['sample_end']],rtol=0,atol=1e-12):
        raise ValueError('Cycle-window definition disagrees with production analysis')
    return result,crossings


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    case = manifest['configuration']
    summaries = sorted(folder.glob('summary-*.json'))
    if case['kind'] != 'periodic' or not summaries:
        raise ValueError('Require a completed periodic run')
    summary = json.loads(summaries[-1].read_text())
    if (not summary.get('stationary')
            or not np.isfinite([summary['max_cfl'],summary['max_divergence']]).all()
            or summary['max_cfl'] >= .5
            or summary['max_divergence'] >= 1e-10
            or summary['source_sha256'] != manifest['metadata']['source_sha256']):
        raise ValueError('Source, stationarity or numerical gate unresolved')
    # Keep only sampling data in RAM, but verify continuity across ALL segments.
    observations,previous,recent = [],0,{}
    count = round(case['end_time']/case['dt'])
    histories = sorted(folder.glob('history-*.jsonl'))
    for history in histories:
        with history.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n'):
                        continue  # Only a torn final write is recoverable by replay.
                    raise
                step = record['step']
                cp = np.asarray(record['cp'],float)
                if (isinstance(step,bool) or not isinstance(step,int) or cp.shape != (24,)
                        or not np.isfinite([record['time'],record['cd'],record['cl'],record['delta_p'],*cp]).all()
                        or abs(record['time']-step*case['dt']) > 1e-10):
                    raise ValueError('Invalid history step, time or sensor data')
                signature = (record['cd'],record['cl'],record['delta_p'],tuple(cp))
                if step <= previous:
                    if recent.get(step) != signature:
                        raise ValueError('Conflicting or excessive checkpoint replay')
                    continue
                if step != previous+1:
                    raise ValueError('Conflicting, missing or out-of-order history observations')
                previous = step
                recent[step] = signature
                recent.pop(step-100,None)
                if record['time'] >= case['sample_start']-1e-12:
                    observations.append([record['time'],record['cd'],record['cl'],record['delta_p'],*(cp-cp[0])])
    if previous != count:
        raise ValueError('Incomplete simulation')
    data = np.asarray(observations,float)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('Invalid sampling data')
    t,values = data[:,0],data[:,1:]
    periodic,crossings = complete_cycles(t,values[:,0],values[:,1],case['sample_start'],
                                       D=case['solver']['D'],U_ref=manifest['metadata']['U_ref'])
    full = window_statistics(t,values,crossings[0],crossings[-1])
    cycles = [window_statistics(t,values,a,b) for a,b in zip(crossings[:-1],crossings[1:])]
    result = dict(run=str(folder),periodic=periodic,common_window_statistics=full,
        columns=['cd','cl','delta_p']+[f'cp_relative_front_{i*15:03d}' for i in range(24)],
        cycle_statistics=cycles,cycle_periods=np.diff(crossings).tolist(),
        cycle_mean_range=np.ptp([row['mean'] for row in cycles],axis=0).tolist(),
        cycle_rms_range=np.ptp([row['rms'] for row in cycles],axis=0).tolist(),
        source_sha256=manifest['metadata']['source_sha256'],
        runner_sha256=manifest['runner_sha256'],analysis_sha256=manifest['analysis_sha256'],
        configuration=case,history_sha256={p.name:file_hash(p) for p in histories},
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All statistics use identical full cycles. Cycle ranges are deterministic dispersion, not confidence intervals. Pressure uses the same physical front gauge across grids.',
        ready_for_paper1=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(cycles=result['periodic']['cycles'],
                         statistics=result['common_window_statistics'])))
