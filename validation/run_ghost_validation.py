"""One explicit Ghost validation case with pressure telemetry and mean-wake sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','FFT_WORKERS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'solver')]
import numpy as np
from main import MAC_CylinderFlow
from postprocessing.analysis import periodic_metrics, window_statistics
from postprocessing.cylinder_observables import CylinderSamples, mean_observables


def run(config_path, output, restart=None):
    case=json.loads(Path(config_path).read_text(encoding='utf-8'))
    dt,end,left=case['dt'],case['end_time'],case['sample_start']
    if (not np.isfinite([dt,end,left]).all() or dt<=0 or not 0<=left<end
            or case['kind'] not in ('steady','periodic','diagnostic')
            or case['solver'].get('boundary')!='ghost_cell' or case['solver'].get('time_scheme')!='ab2_cn'):
        raise ValueError('Invalid explicit Ghost validation case')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    s=MAC_CylinderFlow.from_checkpoint(restart) if restart else MAC_CylinderFlow(**case['solver'])
    if s.configuration != MAC_configuration(case['solver']):
        raise ValueError('Restart/configuration mismatch')
    steps=round((end-s.time)/dt)
    if steps<=0 or abs(s.time+steps*dt-end)>1e-9 or left<s.time:
        raise ValueError('Window/end must be covered by this run at its declared timestep')
    initial_time=s.time
    samples=CylinderSamples(s,output/'centerline.bin',stride=case.get('profile_stride',10))
    samples.record(force=True)
    started=time.perf_counter();checkpoint_clock=started;step=0
    maxima=dict(max_divergence=0.,cfl=0.,pressure_residual_linf=0.)
    metadata=s.metadata(dt,steps,perturb_transient=False)
    manifest=dict(case=case,initial_time=initial_time,restart=str(restart) if restart else None,
        restart_sha256=hashlib.sha256(Path(restart).read_bytes()).hexdigest() if restart else None,
        metadata=metadata,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observables_sha256=hashlib.sha256((ROOT/'postprocessing/cylinder_observables.py').read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256((ROOT/'postprocessing/analysis.py').read_bytes()).hexdigest(),
        ram_stop_guard=False,ready_for_paper1=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def save_status(value):
        temporary=output/'summary.tmp'
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temporary,output/'summary.json')
    with (output/'history.jsonl').open('x',encoding='utf-8') as history:
        def observe(state,diagnostics):
            nonlocal checkpoint_clock,step
            step+=1
            p=samples.pressures()
            row=dict(diagnostics,step=step,delta_p=float(p[0]-p[1]),
                     wall_seconds=time.perf_counter()-started)
            history.write(json.dumps(row,allow_nan=False)+'\n')
            for key in maxima:maxima[key]=max(maxima[key],float(row[key]))
            samples.record(pressures=p)
            if time.perf_counter()-checkpoint_clock>45:
                samples.flush();history.flush()
                state.save_checkpoint(output/'checkpoint.npz',overwrite=True)
                checkpoint_clock=time.perf_counter()
                save_status(dict(completed=False,steps=step,time=state.time,seconds=time.perf_counter()-started,**maxima))
        try:
            s.solve(n_steps=steps,dt=dt,perturb_transient=False,save_artifacts=False,
                    verbose=False,step_callback=observe)
            samples.record(force=True);samples.close();history.flush()
            s.save_checkpoint(output/'checkpoint.npz',overwrite=True)
        except BaseException as error:
            samples.close();history.flush()
            s.save_checkpoint(output/'checkpoint.npz',overwrite=True)
            failed_step=dict(s.last_step_diagnostics)
            if failed_step.get('time') == s.time:
                for key in maxima:
                    if key in failed_step:maxima[key]=max(maxima[key],float(failed_step[key]))
            save_status(dict(completed=False,steps=step,time=s.time,error=f'{type(error).__name__}: {error}',
                             last_step_diagnostics=failed_step,
                             seconds=time.perf_counter()-started,**maxima))
            raise
    with (output/'history.jsonl').open(encoding='utf-8') as history:
        rows=[json.loads(line) for line in history]
    t=np.array([r['time'] for r in rows]);cd=np.array([r['cd'] for r in rows]);cl=np.array([r['cl'] for r in rows])
    dp=np.array([r['delta_p'] for r in rows]);window=[max(left,float(t[0])),float(t[-1])]
    periodic=None
    if case['kind']=='periodic':
        periodic=periodic_metrics(t,cd,cl,left,D=s.D,U_ref=s.U_ref)
        if periodic['cycles']<5:
            save_status(dict(completed=True,accepted=False,ready_for_paper1=False,steps=step,
                             reason='Insufficient complete cycles; no periodic mean-wake claim',**maxima))
            return
        window=[periodic['sample_start'],periodic['sample_end']]
    stats=window_statistics(t,np.column_stack((cd,cl,dp)),*window)
    mean=mean_observables(output/'centerline.bin',*window)
    (output/'mean_observables.json').write_text(json.dumps(mean,indent=2,allow_nan=False),encoding='utf-8')
    if periodic:
        periodic={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in periodic.items()}
    save_status(dict(completed=True,ready_for_paper1=False,steps=step,time=s.time,
        seconds=time.perf_counter()-started,statistics=stats,periodic=periodic,
        mean_flow_available=True,benchmark_accepted=False,
        acceptance_note='Reference-specific force, pressure, Lr, stationarity and domain checks are separate.',
        sample_window=window,**maxima))


def MAC_configuration(options):
    import inspect
    signature=inspect.signature(MAC_CylinderFlow)
    bound=signature.bind(**options);bound.apply_defaults()
    result=dict(bound.arguments);result['verbose']=False
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config-path',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--restart')
    run(**vars(parser.parse_args()))
