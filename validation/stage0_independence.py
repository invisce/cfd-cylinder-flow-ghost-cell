"""Compare fresh run folders without rerunning CFD or certifying paper readiness."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from postprocessing.analysis import calculate_gci
from validation.periodic_window_audit import audit as periodic_audit


def compare(folders, kind):
    required = {'grid': 4, 'time': 3, 'domain': 2}[kind]
    if len(folders) < required:
        raise ValueError(f'{kind} study requires at least {required} runs')
    rows = []
    for folder in map(Path, folders):
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        summaries = sorted(folder.glob('summary-*.json'))
        if not summaries:
            raise ValueError(f'No completed run: {folder}')
        summary = json.loads(summaries[-1].read_text(encoding='utf-8'))
        if (not summary.get('stationary')
                or not np.isfinite([summary['max_divergence'], summary['max_cfl']]).all()
                or summary['max_divergence'] >= 1e-10 or summary['max_cfl'] >= .5
                or summary['source_sha256'] != manifest['metadata']['source_sha256']):
            raise ValueError(f'Unresolved stationarity/divergence/CFL: {folder}')
        rows.append((folder, manifest, summary))
    baseline = rows[0][1]
    varying = {'grid': {'nx', 'ny'}, 'time': set(),
               'domain': {'nx', 'ny', 'Lx', 'Ly', 'xc', 'yc'}}[kind]
    def fixed(case):
        return ({k: v for k, v in case['solver'].items() if k not in varying},
                {k: v for k, v in case.items() if k not in
                 {'solver', 'name', 'reference', 'reference_note', 'reference_source'}
                 and not (kind == 'time' and k == 'dt')})
    for _, manifest, _ in rows[1:]:
        for key in ('source_sha256',):
            if manifest['metadata'][key] != baseline['metadata'][key]:
                raise ValueError('Mixed solver revisions cannot establish independence')
        for key in ('runner_sha256', 'analysis_sha256'):
            if manifest[key] != baseline[key]:
                raise ValueError('Mixed run/analysis revisions cannot establish independence')
        if fixed(manifest['configuration']) != fixed(baseline['configuration']):
            raise ValueError('Inputs outside the study axis differ')
    def resolution(row):
        case = row[1]['configuration']
        s = case['solver']
        return case['dt'] if kind == 'time' else max(s['Lx']/s['nx'], s['Ly']/s['ny'])
    spacings = np.array([(row[1]['configuration']['solver']['Lx']/row[1]['configuration']['solver']['nx'],
                          row[1]['configuration']['solver']['Ly']/row[1]['configuration']['solver']['ny'])
                         for row in rows])
    if kind == 'grid' and not np.allclose(spacings[:, 0]/spacings[:, 1], spacings[0, 0]/spacings[0, 1], rtol=1e-12, atol=0):
        raise ValueError('Grid studies must refine both axes with the same ratio')
    if kind == 'domain':
        if not np.allclose(spacings, spacings[0], rtol=1e-12, atol=0):
            raise ValueError('Domain studies must preserve spatial resolution')
        rows.sort(key=lambda row: row[1]['configuration']['solver']['Lx']*row[1]['configuration']['solver']['Ly'], reverse=True)
    else:
        rows.sort(key=resolution)
        if len(set(resolution(row) for row in rows)) != len(rows):
            raise ValueError('Duplicate resolutions do not count as an independence study')
    metrics = {'cd_mean': .02}
    if baseline['configuration']['kind'] == 'periodic':
        metrics.update(cl_rms=.02, strouhal=.01)
    result = dict(kind=kind, runs=[str(row[0]) for row in rows], metrics={}, ready_for_paper1=False)
    if baseline['configuration']['kind'] == 'periodic':
        result['common_cycle_evidence'] = []
        revised = []
        for folder, manifest, summary in rows:
            evidence = periodic_audit(folder)
            stats = evidence['common_window_statistics']
            summary = dict(summary, cd_mean=stats['mean'][0], cl_rms=stats['rms'][1],
                           strouhal=evidence['periodic']['strouhal'])
            revised.append((folder, manifest, summary))
            result['common_cycle_evidence'].append(dict(
                run=str(folder), periodic=evidence['periodic'],
                history_sha256=evidence['history_sha256'], audit_sha256=evidence['audit_sha256']))
        rows = revised
    for name, tolerance in metrics.items():
        values = np.array([row[2][name] for row in rows], dtype=float)
        if not np.isfinite(values).all() or abs(values[0]) < 1e-12:
            raise ValueError(f'Unresolved or near-zero metric: {name}')
        differences = abs(np.diff(values)/values[:-1])
        entry = dict(values=values.tolist(), adjacent_relative_changes=differences.tolist(),
                     threshold=tolerance, finest_pair_pass=bool(differences[0] < tolerance))
        if kind == 'grid':
            h = [resolution(row) for row in rows]
            entry['gci'] = calculate_gci(*values[:3], r=h[1]/h[0], r32=h[2]/h[1])
            entry['coarser_gci'] = calculate_gci(*values[1:4], r=h[2]/h[1], r32=h[3]/h[2])
        result['metrics'][name] = entry
    result['thresholds_met'] = all(v['finest_pair_pass'] for v in result['metrics'].values())
    result['scope'] = 'Sensitivity thresholds only. Review both grid triplets, sampling uncertainty and reference boundary conditions before accepting independence.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('grid', 'time', 'domain'))
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(args.runs, args.kind), indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(result)
    print(result)
