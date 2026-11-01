"""Recompute reference statistics and force diagnostics without advancing CFD."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from postprocessing.analysis import window_statistics,periodic_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def components(t,total,pressure,viscous,left,right):
    closure=float(np.max(abs(total-pressure-viscous)))
    if closure>1e-10:
        raise ValueError(f'Force components do not sum to total: {closure}')
    stats=window_statistics(t,np.column_stack((total,pressure,viscous)),left,right)
    vt,vp,vv=np.square(stats['rms'])
    if vt<=0:
        raise ValueError('Variance decomposition requires a nonconstant total force')
    covariance=float((vt-vp-vv)/2)
    correlation=covariance/np.sqrt(vp*vv) if vp*vv>0 else None
    return dict(rms=stats['rms'],variance=[float(vt),float(vp),float(vv)],
        covariance=covariance,correlation=correlation,closure_linf=closure,
        variance_shares_percent=[float(100*x/vt) for x in (vp,vv,2*covariance)])


def reference_metrics(archive):
    member='logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4'
    with zipfile.ZipFile(archive) as file:
        data=np.loadtxt(io.BytesIO(file.read(member)))
    t,cd,cl=10*data[:,1],data[:,3],data[:,4]
    periodic=periodic_metrics(t,cd,cl,t[0],D=1.,U_ref=1.)
    stats=window_statistics(t,np.column_stack((cd,cl)),periodic['sample_start'],periodic['sample_end'])
    return dict(archive_sha256=sha(archive),member=member,
        source='https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html',
        cd_mean=stats['mean'][0],cl_rms=stats['rms'][1],cl_mean=stats['mean'][1],
        cl_half_amplitude=stats['half_amplitude'][1],strouhal=periodic['strouhal'],
        cycles=periodic['cycles'],window=stats['window'],
        scope='Finite-resolution official trace, two resolved complete cycles; no continuum uncertainty certificate.'), (t,cl)


def checkpoint_audit(folder,config):
    from optimized_ghost_cell import GhostCellIntegrator
    from cut_cell_solver import build_surface_pressure_quadrature,build_pressure_reconstruction
    nx,ny=config['nx'],config['ny']
    dx,dy=config['Lx']/nx,config['Ly']/ny
    xc,yc,R=config['xc'],config['yc'],config['D']/2
    xu,yu=np.arange(nx+1)*dx,(np.arange(ny)+.5)*dy
    xv,yv=(np.arange(nx)+.5)*dx,np.arange(ny+1)*dy
    xp,yp=xv,yu
    x,y=np.meshgrid(xp,yp)
    geom=SimpleNamespace(nx=nx,ny=ny,dx=dx,dy=dy,Lx=config['Lx'],Ly=config['Ly'],
        xc=xc,yc=yc,R=R,cell_type=np.where(np.hypot(x-xc,y-yc)>R,1,-1))
    obj=GhostCellIntegrator.__new__(GhostCellIntegrator)
    obj.geom=geom; obj.solver=SimpleNamespace(dx=dx,dy=dy)
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as file:
        u,v,p=(file[key].copy() for key in ('u','v','p'))
    points,operator,normal_weights=build_surface_pressure_quadrature(geom)
    ds=2*np.pi*R/len(points)
    normals=normal_weights/ds
    x,y=np.meshgrid(xu,yu); active_u=np.hypot(x-xc,y-yc)>R+1e-13; active_u[:,0]=False
    x,y=np.meshgrid(xv,yv); active_v=np.hypot(x-xc,y-yc)>R+1e-13; active_v[[0,-1]]=False
    derivatives=np.column_stack((obj._wall_gradient(xu,yu,active_u,points) @ u.ravel(),
        obj._wall_gradient(xv,yv,active_v,points) @ v.ravel()))
    dn_normal=np.sum(derivatives*normals,axis=1)
    pressure_force=-np.sum((operator @ p.ravel())[:,None]*normal_weights,axis=0)
    mu=config['nu']*config['rho']
    viscous_force=mu*np.sum(derivatives+normals*dn_normal[:,None],axis=0)*ds
    coefficient=.5*config['rho']*config['U_inf']**2*config['D']
    forces=np.r_[pressure_force+viscous_force,pressure_force,viscous_force]/coefficient
    last=json.loads((folder/'history.jsonl').read_text().splitlines()[-1])
    recorded=np.array([last[key] for key in ('cd','cl','cd_pressure','cl_pressure','cd_viscous','cl_viscous')])
    mismatch=float(np.max(abs(forces-recorded)))
    if mismatch>1e-10:
        raise ValueError(f'Checkpoint force replay mismatch: {mismatch}')
    ub=RegularGridInterpolator((yu,xu),u)(points)
    vb=RegularGridInterpolator((yv,xv),v)(points)
    un=ub*normals[:,0]+vb*normals[:,1]
    ut=-ub*normals[:,1]+vb*normals[:,0]
    probes=np.array([[yc,xc-R],[yc,xc+R]])
    front,back=build_pressure_reconstruction(geom,probes) @ p.ravel()
    return dict(force_replay_linf=mismatch,quadrature_points=len(points),
        final_instant_delta_p=float(front-back),
        reconstructed_wall_normal_linf=float(np.max(abs(un))),
        reconstructed_wall_tangent_linf=float(np.max(abs(ut))),
        normal_derivative_rms=float(np.sqrt(np.mean(dn_normal**2))),
        normal_derivative_viscous_lift_term=float(mu*np.sum(normals[:,1]*dn_normal)*ds/coefficient),
        scope='One final phase; boundary interpolation diagnostics are not a time-averaged pressure/force validation.')


def audit(results,package,archive,output):
    results,package,archive,output=map(Path,(results,package,archive,output))
    if output.exists():
        raise FileExistsError('Preserve prior evidence: choose a new output directory')
    manifest=json.loads((results/'manifest.json').read_text())
    batch=json.loads((results/'aws_batch_report.json').read_text())
    for name,digest in manifest['solver_sha256'].items():
        if sha(package/'solver'/name)!=digest:
            raise ValueError(f'Frozen solver mismatch: {name}')
    sys.path.insert(0,str(package/'solver'))
    ref,trace=reference_metrics(archive)
    report=dict(reference=ref,grids={},ready_for_paper1=False)
    histories={}
    for ny in (123,164):
        folder=results/f'grid_/{ny}'
        summary=json.loads((folder/'summary.json').read_text())
        if sha(folder/'history.jsonl')!=summary['history_sha256'] or sha(folder/'checkpoint.npz')!=summary['checkpoint_sha256']:
            raise ValueError('Run evidence hash mismatch')
        rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        if len(rows)!=50000 or [r['step'] for r in rows]!=list(range(1,50001)):
            raise ValueError('Missing/replayed history records')
        t=np.array([r['time'] for r in rows])
        values=np.array([[r[k] for k in ('cd','cl','cl_pressure','cl_viscous')] for r in rows])
        left,right=batch['statistics'][str(ny)]['window']
        stats=window_statistics(t,values,left,right)
        metrics=dict(cd_mean=stats['mean'][0],cl_rms=stats['rms'][1],strouhal=batch['statistics'][str(ny)]['strouhal'])
        report['grids'][str(ny)]=dict(metrics=metrics,
            reference_error_percent={key:100*abs(value-ref[key])/abs(ref[key]) for key,value in metrics.items()},
            force_components=components(t,*values[:,1:].T,left,right),
            checkpoint=checkpoint_audit(folder,json.loads((folder/'preflight.json').read_text())['configuration']),
            window=[left,right],cycles=batch['statistics'][str(ny)]['cycles'])
        histories[str(ny)]=(t,values[:,1])
    coarse,fine=(report['grids'][str(ny)] for ny in (123,164))
    report['between_grid_change_percent']={key:100*abs(fine['metrics'][key]-coarse['metrics'][key])/abs(coarse['metrics'][key])
        for key in ('cd_mean','cl_rms','strouhal')}
    c,f=coarse['force_components'],fine['force_components']
    total_gap=c['variance'][0]-f['variance'][0]
    report['variance_gap_shares_percent']=dict(pressure=100*(c['variance'][1]-f['variance'][1])/total_gap,
        viscous=100*(c['variance'][2]-f['variance'][2])/total_gap,
        covariance=200*(c['covariance']-f['covariance'])/total_gap)
    report['inference']='Pressure variance dominates this two-grid variance difference; this is a decomposition, not proof of a pressure-operator defect.'
    output.mkdir(parents=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for label,(t,cl),window in [('Official reference',trace,ref['window'])]+[
            (f'Ghost ny={ny}',histories[str(ny)],report['grids'][str(ny)]['window']) for ny in (123,164)]:
        a,b=window; cycles=ref['cycles'] if label=='Official reference' else 5
        period=(b-a)/cycles
        phase=np.linspace(0,1,401)
        axes[0].plot(phase,np.interp(a+phase*period,t,cl),label=label)
    axes[0].set(xlabel='Cycle phase',ylabel='Lift coefficient'); axes[0].legend(fontsize=8)
    axes[1].bar(['Pressure','Viscous','Covariance'],list(report['variance_gap_shares_percent'].values()))
    axes[1].set(ylabel='Share of two-grid variance gap (%)')
    fig.savefig(output/'reference_and_variance.png',dpi=160); plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=ROOT/'reports/reference_data/20141124_dfg_bench2_official.zip')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    t=np.linspace(0,2*np.pi,1001); p=np.sin(t); v=.5*p
    check=components(t,p+v,p,v,t[0],t[-1])
    assert abs(check['correlation']-1)<1e-12 and abs(sum(check['variance_shares_percent'])-100)<1e-12
    result=audit(args.results,args.package,args.reference,args.output)
    print(json.dumps(dict(reference_rms=result['reference']['cl_rms'],
        reference_errors={key:value['reference_error_percent'] for key,value in result['grids'].items()},
        variance_gap_shares=result['variance_gap_shares_percent']),indent=2))
