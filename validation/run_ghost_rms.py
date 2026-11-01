"""Bounded, restartable, same-configuration Ghost-Cell RMS grid comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('FFT_WORKERS','1')
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('GHOST_RMS_BASE', ROOT if (ROOT/'manifest.json').exists()
    else ROOT/'reports/ghost_rms_numba_fft_20261009')).resolve()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def source_path(manifest):
    source=Path(manifest['source'])
    return source if source.is_absolute() else BASE/source


def restart_ghost(cls,path,nx,ny):
    import numpy as np
    with np.load(path,allow_pickle=False) as data:
        config=json.loads(str(data['checkpoint_metadata']))['configuration']
    if (config.get('boundary')!='ghost_cell' or config.get('time_scheme')!='ab2_cn'
            or (config['nx'],config['ny'])!=(nx,ny)):
        raise ValueError('Ghost restart requires the same grid and AB2/CN configuration')
    return cls.from_checkpoint(path)


def prepare():
    import shutil
    BASE.mkdir(exist_ok=True)
    snapshot=BASE/'solver'
    snapshot.mkdir(exist_ok=True)
    for path in (ROOT/'solver').iterdir():
        if path.suffix in ('.py','.pyd'):
            shutil.copy2(path,snapshot/path.name)
    (snapshot/'src').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'solver/src/bindings.cpp',snapshot/'src/bindings.cpp')
    shutil.copy2(ROOT/'postprocessing/analysis.py',BASE/'shared_analysis.py')
    old=json.loads((ROOT/'reports/rms_three_leads_20261008/batch_manifest.json').read_text())
    manifest=dict(source=old['source'],source_sha256=old['source_sha256'],
        grids=[[660,123],[880,164]],dt=.001,duration=50.,warmup=30.,
        limits=dict(seconds=7200,disk_bytes=1024**3),
        solver_sha256={str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file() and '__pycache__' not in str(path)},
        runner_sha256=sha(Path(__file__)),analysis_sha256=sha(BASE/'shared_analysis.py'),
        rms_criterion=dict(percent=2.,denominator='coarser-grid RMS'),
        reference_scope='Reference trace comparisons are separate; no 2% reference requirement.',
        warmstart='Both grids start from the same developed Cut-Cell checkpoint, reprojected onto the new Ghost constraints; discard30 time units.',
        literature='https://engineering.jhu.edu/fsag/wp-content/uploads/2014/06/JCP_sharp_interface.pdf',
        caveat='This staggered quadratic-normal implementation is not a reproduction of the paper collocated solver; validate through the recorded tests and runs.')
    assert sha(manifest['source'])==manifest['source_sha256']
    write_json(BASE/'manifest.json',manifest)
    return manifest


def statistics(directory,manifest):
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location('ghost_shared_analysis',BASE/'shared_analysis.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows=[]
    with (directory/'history.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            rows.append([row[k] for k in ('time','cd','cl','cl_pressure','cl_viscous')])
    data=np.asarray(rows)
    start=data[0,0]-manifest['dt']+manifest['warmup']
    t,cd,cl=data[:,:3].T
    metrics=module.periodic_metrics(t,cd,cl,start,D=1.,U_ref=1.)
    if metrics['cycles']<3:
        return dict(cycles=metrics['cycles'],settled=False)
    left,right=metrics['sample_start'],metrics['sample_end']
    stats=module.window_statistics(t,data[:,1:],left,right)
    keep=t>=start; tt=t[keep]; cc=cl[keep]
    centered=cc-np.trapezoid(cc,tt)/(tt[-1]-tt[0])
    ids=np.flatnonzero((centered[:-1]<0)&(centered[1:]>=0))
    zeros=tt[ids]-centered[ids]*np.diff(tt)[ids]/np.diff(centered)[ids]
    cycles=[module.window_statistics(t,cl[:,None],a,b)['rms'][0] for a,b in zip(zeros[:-1],zeros[1:])]
    drift=100*abs(cycles[-1]-cycles[-2])/cycles[-1]
    return dict(cycles=metrics['cycles'],window=[left,right],cl_rms=stats['rms'][1],
        cd_mean=stats['mean'][0],cl_mean=stats['mean'][1],cl_pressure_rms=stats['rms'][2],
        cl_viscous_rms=stats['rms'][3],strouhal=metrics['strouhal'],cycle_cl_rms=cycles,
        last_cycle_change_percent=drift,settled=bool(metrics['stationary'] and drift<.2))


def run(ny,pilot=False,resume=False):
    import numpy as np
    import psutil
    from scipy.interpolate import RectBivariateSpline
    manifest=json.loads((BASE/'manifest.json').read_text())
    grids={height:width for width,height in manifest['grids']}
    if ny not in grids:
        raise ValueError(f'Grid ny={ny} is not in this package manifest')
    if manifest.get('portable_package'):
        sys.path.insert(0,str(BASE))
        from bootstrap import verify
        verify(BASE)
    snapshot=BASE/'solver'
    assert sha(Path(__file__))==manifest['runner_sha256']
    assert sha(BASE/'shared_analysis.py')==manifest['analysis_sha256']
    for name,digest in manifest['solver_sha256'].items():
        assert sha(snapshot/name)==digest,name
    source_checkpoint=source_path(manifest)
    assert sha(source_checkpoint)==manifest['source_sha256']
    sys.path.insert(0,str(snapshot))
    from main import MAC_CylinderFlow
    directory=BASE/('pilot_' if pilot else 'grid_')/str(ny)
    directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    summary_path=directory/'summary.json'
    previous=json.loads(summary_path.read_text()) if resume else None
    if resume:
        assert not previous['completed'] and previous['status'].startswith('Resource stop:')
        assert previous['manifest_sha256']==sha(BASE/'manifest.json')
        solver=MAC_CylinderFlow.from_checkpoint(directory/'checkpoint.npz')
        count,last=0,None
        with (directory/'history.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                last=json.loads(line); count+=1; assert last['step']==count
        assert count==previous['steps'] and abs(last['time']-solver.time)<1e-10
    else:
        if manifest.get('source_mode')=='ghost_restart':
            solver=restart_ghost(MAC_CylinderFlow,source_checkpoint,grids[ny],ny)
            q=solver.gg.pack(solver.u,solver.v)
        else:
            with np.load(source_checkpoint,allow_pickle=False) as source:
                metadata=json.loads(str(source['checkpoint_metadata']))
                config=metadata['configuration'].copy()
                config.update(nx=grids[ny],ny=ny,boundary='ghost_cell',time_scheme='ab2_cn',pressure_solver='direct')
                solver=MAC_CylinderFlow(**config)
                old=metadata['configuration']
                dx,dy=old['Lx']/old['nx'],old['Ly']/old['ny']
                for key,x,y,xx,yy in (
                    ('u',np.arange(old['nx']+1)*dx,(np.arange(old['ny'])+.5)*dy,solver.gc.xu,solver.gc.yu),
                    ('v',(np.arange(old['nx'])+.5)*dx,np.arange(old['ny']+1)*dy,solver.gc.xv,solver.gc.yv),
                    ('p',(np.arange(old['nx'])+.5)*dx,(np.arange(old['ny'])+.5)*dy,solver.gc.xp,solver.gc.yp)):
                    getattr(solver,key)[:]=RectBivariateSpline(y,x,source[key])(yy,xx)
                q,_=solver.gg.project(solver.gg.pack(solver.u,solver.v),1.)
                solver.u[:],solver.v[:]=solver.gg.unpack(q)
                half=RectBivariateSpline((np.arange(old['ny'])+.5)*dy,(np.arange(old['nx'])+.5)*dx,source['cn_pressure'])(solver.gc.yp,solver.gc.xp)
                solver.gg.half_pressure=half.ravel()[solver.gg.pi]
                solver.time=float(metadata['time'])
        write_json(directory/'preflight.json',dict(configuration=solver.configuration,source_sha256=manifest['source_sha256'],
            thread_settings={key:os.environ.get(key) for key in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','FFT_WORKERS')},
            initial_time=solver.time,initial_divergence=float(np.max(abs(solver.gg.divergence(q)))),
            setup_seconds=time.perf_counter()-started,backend='Numba MAC fluxes and time updates; FFT with cached boundary correction',
            pressure_inverse=solver.gg.pressure_inverse_info))
    target=100 if pilot else round(manifest['duration']/manifest['dt'])
    record=dict(completed=False,steps=0,max_divergence=0.,max_cfl=0.,peak_mib=0.,
        max_flux_mismatch=0.,max_momentum_residual=0.,min_free_mib=psutil.virtual_memory().available/2**20)
    if previous:
        record.update({key:previous[key] for key in record})
    elapsed_before=0. if previous is None else previous['seconds']
    solver.run_options=dict(perturb_transient=False)
    process=psutil.Process()
    last_checkpoint=time.perf_counter()
    loop_started=last_checkpoint
    status='completed'
    solver.save_checkpoint(directory/'checkpoint.npz',overwrite=resume)
    try:
        with (directory/'history.jsonl').open('a' if resume else 'x',encoding='utf-8') as stream:
            def callback(s,diag):
                nonlocal last_checkpoint
                record['steps']+=1
                rss=process.memory_info().rss/2**20
                free=psutil.virtual_memory().available/2**20
                influx=float(np.sum(s.u[:,0])*s.dy); outflux=float(np.sum(s.u[:,-1])*s.dy)
                mismatch=abs(outflux-influx)/abs(influx)
                row=dict(diag,step=record['steps'],flux_mismatch=mismatch)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                record['peak_mib']=max(record['peak_mib'],rss)
                record['min_free_mib']=min(record['min_free_mib'],free)
                record['max_divergence']=max(record['max_divergence'],diag['max_divergence'])
                record['max_cfl']=max(record['max_cfl'],diag['cfl'])
                record['max_flux_mismatch']=max(record['max_flux_mismatch'],mismatch)
                record['max_momentum_residual']=max(record['max_momentum_residual'],diag['momentum_residual_linf'])
                if elapsed_before+time.perf_counter()-started>manifest['limits']['seconds']:
                    raise RuntimeError('Resource stop: wall-time budget reached')
                if (directory/'history.jsonl').stat().st_size>manifest['limits']['disk_bytes']:
                    raise RuntimeError('Resource stop: disk budget reached')
                if time.perf_counter()-last_checkpoint>45 or record['steps']==target:
                    stream.flush(); os.fsync(stream.fileno())
                    s.save_checkpoint(directory/'checkpoint.npz',overwrite=True)
                    print(json.dumps(dict(ny=ny,steps=record['steps'],time=s.time,rss_mib=rss,free_mib=free)),flush=True)
                    last_checkpoint=time.perf_counter()
            solver.solve(target-record['steps'],dt=manifest['dt'],perturb_transient=False,
                save_artifacts=False,verbose=False,step_callback=callback)
        record['completed']=True
    except Exception as error:
        status=str(error) if str(error).startswith('Resource stop:') else f'{type(error).__name__}: {error}'
    finally:
        solver.save_checkpoint(directory/'checkpoint.npz',overwrite=True)
        record.update(status=status,seconds=elapsed_before+time.perf_counter()-started,
            loop_seconds=time.perf_counter()-loop_started,final_time=solver.time,
            manifest_sha256=sha(BASE/'manifest.json'),history_sha256=sha(directory/'history.jsonl'),
            checkpoint_sha256=sha(directory/'checkpoint.npz'),source_unchanged=sha(source_checkpoint)==manifest['source_sha256'])
        write_json(summary_path,record)
    if record['completed'] and not pilot:
        write_json(directory/'statistics.json',statistics(directory,manifest))
    print(json.dumps(record,allow_nan=False),flush=True)
    return record['completed']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--ny',type=int)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        if (BASE/'manifest.json').exists() and json.loads((BASE/'manifest.json').read_text()).get('portable_package'):
            parser.error('Portable packages are prepared by build_ghost_linux_package.py; use bootstrap.py')
        print(json.dumps(prepare(),allow_nan=False))
    else:
        if args.ny is None:
            parser.error('--ny is required')
        sys.exit(0 if run(args.ny,args.pilot,args.resume) else 1)
