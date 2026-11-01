"""Rebuild baseline grid/GCI/timestep tables without modifying simulation inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from postprocessing.analysis import calculate_gci


def consolidate(results, baseline_csv, gates, output):
    paths = [Path(p) for p in (results,baseline_csv,gates)]
    report = json.loads(paths[0].read_text(encoding='utf-8'))
    screening = json.loads(paths[2].read_text(encoding='utf-8'))
    with paths[1].open(encoding='utf-8-sig',newline='') as stream:
        old = {int(row['ny']):row for row in csv.DictReader(stream)}
    if report['status'] != 'completed':
        raise ValueError('Baseline batch is incomplete')
    grids = [164,205,246,328]
    values = {}
    for ny in grids:
        case = report['cases'][f'four_grid/{ny}']
        if (case['dt'] != .0005 or not case['summary']['completed'] or not case['summary']['source_unchanged']
                or not case['statistics']['settled'] or case['summary']['max_divergence'] >= 1e-10
                or case['summary']['max_cfl'] >= .5):
            raise ValueError(f'Incompatible or failed baseline case {ny}')
        values[ny] = case['statistics']
        if not np.isfinite([values[ny][k] for k in ('cl_rms','cd_mean','strouhal')]).all():
            raise ValueError('Nonfinite baseline observable')
    metrics = ('cl_rms','cd_mean','strouhal')
    limits = dict(cl_rms=2.,cd_mean=2.,strouhal=1.)
    pairs = {}
    for coarse,fine in zip(grids[:-1],grids[1:]):
        changes = {k:100*abs(values[fine][k]-values[coarse][k])/abs(values[coarse][k]) for k in metrics}
        pairs[f'{coarse}->{fine}'] = dict(change_percent=changes,
            pass_all=all(changes[k] < limits[k] for k in metrics),denominator='coarser grid')
    fits = {}
    for label,triple in (('consecutive',[328,246,205]),('alternative_sensitivity_only',[328,246,164])):
        fine,medium,coarse = triple
        fits[label] = dict(grids_fine_to_coarse=triple,
            estimates={k:calculate_gci(*(values[n][k] for n in triple),r=fine/medium,r32=medium/coarse) for k in metrics},
            caveat='Algebraic three-point fits. Neither the triple nor a small GCI certifies an asymptotic regime.')
    timestep = {}
    for ny in (205,246):
        current = report['cases'][f'four_grid/{ny}']['statistics']
        half = report['cases'][f'half_dt/{ny}']['statistics']
        quarter = report['cases'][f'quarter_dt/{ny}']['statistics']
        for stats in (current,half,quarter):
            if not stats['settled'] or stats['cycles'] < 5:
                raise ValueError('Unsettled timestep comparison')
        timestep[str(ny)] = dict(
            first_halving_percent=100*abs(current['cl_rms']-float(old[ny]['cl_rms']))/abs(float(old[ny]['cl_rms'])),
            second_halving_percent=100*abs(quarter['cl_rms']-half['cl_rms'])/abs(half['cl_rms']),
            fixed_dt_window_shift_percent=100*abs(half['cl_rms']-current['cl_rms'])/abs(current['cl_rms']),
            windows=dict(dt_001=[float(old[ny]['sample_start']),float(old[ny]['sample_end'])],
                         dt_0005_early=current['window'],dt_0005_late=half['window'],dt_00025=quarter['window']),
            note='First and second halvings use two different late windows; no common-window three-dt temporal order is claimed.')
    errors = [c['variant_gradient_linf'] for c in screening['cases']]
    sizes = [c['nx'] for c in screening['cases']]
    orders = [float(np.log(a/b)/np.log(n2/n1)) for a,b,n1,n2 in zip(errors[:-1],errors[1:],sizes[:-1],sizes[1:])]
    result = dict(ready_for_paper1=False, professor_limits_percent=limits, selected_grid=246,
        finer_check_grid=328, grid_pair_limits_pass=pairs['246->328']['pass_all'],
        gci_acceptance='Unresolved: asymptotic behaviour not established; report conditional fits and their sensitivity.',
        pairs=pairs,gci=fits,timestep=timestep, pressure_variant_spatial_orders=orders,
        pressure_variant_original_passed=screening['passed'],
        pressure_variant_experiment_eligible=screening['experiment_eligible'],
        rms_cause='Not isolated by the existing dynamic runs; no parity or pressure-closure causal attribution.',
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'consolidated.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'four_grid.csv').open('x',encoding='utf-8',newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['ny','cl_rms','cd_mean','strouhal'])
        writer.writerows([ny,*(values[ny][k] for k in metrics)] for ny in grids)
    lines = ['# Consolidated Ghost baseline evidence','',
        'Grid246 passes the three internal change limits against328. Stage0 is not complete.',
        'No numerical solver run or input modification is performed by this script.','',
        '## Grid changes (coarser denominator)','',
        '| Pair | RMS % | Mean drag % | St % | All limits |','|---|---:|---:|---:|---|']
    for label,entry in pairs.items():
        c=entry['change_percent'];lines.append(f'| {label} | {c["cl_rms"]:.6f} | {c["cd_mean"]:.6f} | {c["strouhal"]:.6f} | {entry["pass_all"]} |')
    lines += ['', '## Conditional GCI fits','', '| Triple | Metric | p | Fine GCI % |','|---|---|---:|---:|']
    for label,entry in fits.items():
        for key,fit in entry['estimates'].items():
            lines.append(f'| {entry["grids_fine_to_coarse"]} ({label}) | {key} | {fit["p"]} | {fit["gci_fine_pct"]} |')
    lines += ['', 'These fits do not establish asymptotic convergence. The alternative triple is disclosed sensitivity analysis.',
        '', '## Timestep sensitivity','', '| ny | First halving % | Second halving % | Fixed-dt window shift % |','|---|---:|---:|---:|']
    for ny,entry in timestep.items():
        lines.append(f'| {ny} | {entry["first_halving_percent"]:.9f} | {entry["second_halving_percent"]:.9f} | {entry["fixed_dt_window_shift_percent"]:.9f} |')
    lines += ['', 'The two halving comparisons use different late sampling windows; see JSON for exact endpoints.',
        '',f'Pressure-variant spatial orders: {orders}. Original accuracy-cap failure remains recorded.',
        '', 'Production parity-probe numbers require independent reproduction. The long pressure-variant RMS comparison has not been established by these inputs.',
        '', 'Remaining Stage0 work includes validated iterative pressure/monitoring, same-version validation and mean-flow Lr, formal timestep acceptance, and domain enlargement sensitivity.']
    (output/'GHOST_CELL_CONSOLIDATED_EVIDENCE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--baseline-csv',required=True)
    parser.add_argument('--gates',required=True)
    parser.add_argument('--output',required=True)
    print(json.dumps(consolidate(**vars(parser.parse_args())),indent=2))
