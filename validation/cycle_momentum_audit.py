"""Phase-wide unsteady force balance from a private completed-state clone.

Original CFD histories/checkpoints remain immutable. Restartable probe histories
record scalar momentum/outer-force observations, not full field snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validation.run_stage0_case import MAC_CylinderFlow,ResourceLimit,preflight
from validation.unsteady_momentum_budget import fluid_momentum,boundary_momentum
from validation.steady_momentum_budget import outer_forces
from validation.periodic_window_audit import file_hash


def balance_statistics(observations,normalization):
    data = np.asarray(observations,float)
    if (data.ndim != 2 or data.shape[1] != 9 or len(data) < 4
            or not np.isfinite(data).all() or np.any(np.diff(data[:,0]) <= 0)
            or not np.isfinite(normalization) or normalization <= 0):
        raise ValueError('Invalid momentum history or coefficient normalization')
    t = data[:,0]
    # For equal timesteps this is the centered second-order storage derivative.
    if not np.allclose(np.diff(t),np.diff(t)[0],rtol=1e-7,atol=1e-12):
        raise ValueError('Probe timesteps must be uniform')
    ts = t[1:-1]
    result = dict(window=[float(ts[0]),float(ts[-1])],sample_count=len(ts))
    average = lambda x: np.trapezoid(x,ts,axis=0)/(ts[-1]-ts[0])
    for name,start in (('shared_volume',1),('outer_mass_moment',3)):
        storage = (data[2:,start:start+2]-data[:-2,start:start+2])/(t[2:]-t[:-2])[:,None]
        surface = data[1:-1,7:9]
        inferred = data[1:-1,5:7]-storage
        error = inferred-surface
        surface_fluctuation = surface-average(surface)
        surface_rms = np.sqrt(average(surface_fluctuation**2))
        result[name] = dict(mean_force=average(inferred).tolist(),surface_mean_force=average(surface).tolist(),
            mean_coefficient_difference=(average(error)/normalization).tolist(),
            rms_coefficient_difference=(np.sqrt(average(error**2))/normalization).tolist(),
            maximum_absolute_coefficient_difference=(np.max(abs(error),axis=0)/normalization).tolist(),
            surface_rms_coefficients=(surface_rms/normalization).tolist(),
            # Normalize drag by its mean and lift by its resolved oscillation RMS.
            drag_rms_error_relative_to_mean=float(np.sqrt(average(error**2))[0]/max(abs(average(surface)[0]),1e-12)),
            lift_rms_error_relative_to_fluctuation=float(np.sqrt(average(error**2))[1]/max(surface_rms[1],1e-12)))
    return result


def audit(folder,output,*,cycles=2,budget_seconds=900,rss_ceiling_mib=4096,resume=False):
    folder,output = Path(folder),Path(output)
    manifest = json.loads((folder/'manifest.json').read_text())
    summary = json.loads(sorted(folder.glob('summary-*.json'))[-1].read_text())
    case = manifest['configuration']
    if (case['kind'] != 'periodic' or not summary.get('stationary')
            or summary['max_cfl'] >= .5 or summary['max_divergence'] >= 1e-10
            or case['solver'].get('slip_walls',False) or cycles < 2):
        raise ValueError('Require resolved completed periodic no-slip channel data and >=2 cycles')
    count = int(np.ceil(cycles*summary['period']/case['dt']))+2
    estimate = preflight(dict(case,kind='diagnostic',end_time=count*case['dt'],sample_start=0))
    if estimate['available_mib'] < estimate['estimated_peak_mib']+256:
        raise ResourceLimit('Insufficient free RAM for this private probe')
    checkpoint = folder/'checkpoint.npz'
    original_hash = file_hash(checkpoint)
    identity = dict(input_checkpoint_sha256=original_hash,cycles=cycles,steps=count,
        diagnostic_sha256=file_hash(Path(__file__)),outer_quadrature_sha256=file_hash(ROOT/'validation/steady_momentum_budget.py'))
    if resume:
        record = json.loads((output/'manifest.json').read_text())
        if record['identity'] != identity:
            raise ValueError('Probe input/source differs from its checkpoint')
        solver = MAC_CylinderFlow.from_checkpoint(output/'checkpoint.npz')
    else:
        output.mkdir(parents=True,exist_ok=False)
        solver = MAC_CylinderFlow.from_checkpoint(checkpoint)
        if not np.isclose(solver.time,case['end_time'],rtol=0,atol=1e-10) or solver.metadata(case['dt'],0)['source_sha256'] != manifest['metadata']['source_sha256']:
            raise ValueError('Probe requires matching completed source and time')
        record = dict(identity=identity,start_time=solver.time,configuration=case,compute_seconds=0.,
                      source_sha256=manifest['metadata']['source_sha256'],segments=[],ready_for_paper1=False)
    save = lambda: (output/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    def observation(s):
        vol = fluid_momentum(s.u,s.v,s.cc.u_vol,s.cc.v_vol,s.rho)
        ext = boundary_momentum(s.u,s.dx,s.dy,s.rho)
        force = outer_forces(s.u,s.v,s.p,s.dx,s.dy,s.nu*s.rho,s.rho)['force']
        denominator = .5*s.rho*s.U_ref**2*s.D
        surface = denominator*np.array([s.last_step_diagnostics['cd'],s.last_step_diagnostics['cl']])
        return [s.time,*vol,*ext,*force,*surface]
    remaining = count-round((solver.time-record['start_time'])/case['dt'])
    allowance = budget_seconds-record['compute_seconds']
    if allowance <= 0:
        raise ResourceLimit('Original cumulative probe time budget exhausted')
    segment = len(record['segments'])
    started = time.perf_counter()
    peak = psutil.Process().memory_info().rss/2**20
    record.update(status='running',budget_seconds=budget_seconds,rss_ceiling_mib=rss_ceiling_mib)
    save()
    with (output/f'history-{segment:04d}.jsonl').open('x',encoding='utf-8') as stream:
        if not resume:
            stream.write(json.dumps(observation(solver))+'\n')
        def callback(s,values):
            nonlocal peak
            stream.write(json.dumps(observation(s))+'\n')
            peak = max(peak,psutil.Process().memory_info().rss/2**20)
            if round((s.time-record['start_time'])/case['dt'])%100 == 0:
                stream.flush()
                if (s.metadata(case['dt'],0)['source_sha256'] != record['source_sha256']
                        or file_hash(Path(__file__)) != identity['diagnostic_sha256']
                        or file_hash(ROOT/'validation/steady_momentum_budget.py') != identity['outer_quadrature_sha256']):
                    raise RuntimeError('Sources changed during private probe')
                s.save_checkpoint(output/'checkpoint.npz',overwrite=True)
            if peak > rss_ceiling_mib or time.perf_counter()-started > allowance or psutil.virtual_memory().available < 256*2**20:
                s.save_checkpoint(output/'checkpoint.npz',overwrite=True)
                raise ResourceLimit('Private probe checkpointed at resource limit')
        try:
            solver.solve(n_steps=remaining,dt=case['dt'],save_artifacts=False,verbose=False,
                         step_callback=callback,**solver.run_options)
            record['status'] = 'integrated'
        except Exception as error:
            record.update(status='stopped',reason=str(error))
            raise
        finally:
            elapsed = time.perf_counter()-started
            record['compute_seconds'] += elapsed
            record['segments'].append(dict(elapsed_seconds=elapsed,peak_mib=peak,last_time=solver.time))
            save()
    solver.save_checkpoint(output/'checkpoint.npz',overwrite=True)
    if solver.metadata(case['dt'],0)['source_sha256'] != record['source_sha256']:
        raise RuntimeError('Solver changed during private probe')
    if original_hash != file_hash(checkpoint):
        raise RuntimeError('Original production checkpoint changed')
    observations = []
    for history in sorted(output.glob('history-*.jsonl')):
        observations += [json.loads(line) for line in history.read_text().splitlines()]
    if len(observations) != count+1:
        raise ValueError('Missing private probe observations')
    result = balance_statistics(observations,.5*solver.rho*solver.U_ref**2*solver.D)
    result.update(input_checkpoint_sha256=original_hash,source_sha256=record['source_sha256'],
        diagnostic_sha256=identity['diagnostic_sha256'],outer_quadrature_sha256=identity['outer_quadrature_sha256'],
        scope='Phase-wide independent outer quadrature with two storage routes. Grid/time convergence remain required; no automatic scientific acceptance.',ready_for_paper1=False)
    with (output/'summary.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    record['status'] = 'completed'
    save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--budget-seconds',type=float,default=900)
    args = parser.parse_args()
    print(json.dumps(audit(args.run,args.output,cycles=args.cycles,
        budget_seconds=args.budget_seconds,resume=args.resume)))
