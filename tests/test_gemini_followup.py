"""Small AWS-only checks for the Gemini follow-up changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','FFT_WORKERS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'solver')]
import numpy as np
from main import MAC_CylinderFlow
from optimized_ghost_cell import extension
from postprocessing.analysis import calculate_gci
from postprocessing.cylinder_observables import CylinderSamples, mean_observables, recirculation_length
from validation.consolidate_stage0_evidence import consolidate
from validation.migrate_ghost_backend_checkpoint import migrate
from pressure_schur import monitored_pressure_solver
import scipy.sparse as sp


def solver(**options):
    return MAC_CylinderFlow(nx=40,ny=30,Lx=4.,Ly=3.,xc=1.5,yc=1.5,
        boundary='ghost_cell',time_scheme='ab2_cn',verbose=False,**options)


def test_flat_quadratic():
    h=.1
    x,y=np.arange(-2,3)*h,np.arange(-2.5,5)*h
    class Patch:
        def distance(self,xx,yy):return np.where(abs(xx)<h/4,yy,1.)
        def normal(self,xx,yy):return 0.,1.
    xx,yy=np.meshgrid(x,y);sdf=Patch();active=sdf.distance(xx,yy)>0
    errors=[]
    for mode in ('bilinear','biquadratic'):
        matrix,ids=extension(x,y,sdf,active,interpolation=mode)
        field=(matrix @ (yy*yy).ravel()[ids]).reshape(yy.shape)
        j=int(np.argmin(abs(y+.5*h)));i=2
        lap=(field[j,i]-2*field[j+1,i]+field[j+2,i])/h**2
        errors.append(abs(lap-2))
    assert errors[0]>.05 and errors[1]<1e-12,errors
    print('Flat quadratic Laplacian errors:',errors)


def test_pressure_and_backends():
    reference=solver(); iterative=solver(pressure_solver='iterative')
    outputs=[]
    for s in (reference,iterative):
        saved=[]
        s.solve(n_steps=12,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False,
                step_callback=lambda _,d:saved.append(d))
        assert all(len(d['pressure_solves'])==d['pressure_solve_count'] for d in saved)
        assert len({id(d['pressure_solves']) for d in saved})==len(saved)
        d=s.last_step_diagnostics
        assert d['pressure_residual_linf']<1e-9 and d['pressure_solve_count']>=2,d
        assert d['pressure_iterations']>=0 and d['pressure_seconds']>=0,d
        outputs.append((s.u.copy(),s.v.copy(),s.p.copy()))
    errors=[float(np.max(abs(a-b))) for a,b in zip(*outputs)]
    assert max(errors[:2])<1e-9 and errors[2]<1e-8,errors
    assert iterative.last_step_diagnostics['pressure_iterations']>0
    print('Direct/GMRES field differences:',errors)
    for options in (dict(ghost_pressure_extension='quadratic'),
                    dict(ghost_velocity_interpolation='biquadratic'),
                    dict(ghost_pressure_extension='quadratic',ghost_velocity_interpolation='biquadratic')):
        s=solver(**options)
        s.solve(n_steps=20,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
        assert s.last_step_diagnostics['max_divergence']<1e-10
        assert s.last_step_diagnostics['cfl']<.5
        if options.get('ghost_pressure_extension')=='quadratic':
            xx,yy=np.meshgrid(s.gc.xp,s.gc.yp)
            field=1+xx+yy+xx*yy+xx*xx+yy*yy
            valid=np.diff(s.gg.P.indptr)>0
            assert np.max(abs((s.gg.P @ field.ravel()[s.gg.pi])[valid]-field.ravel()[valid]))<1e-10
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.npz';s.save_checkpoint(checkpoint)
            restored=MAC_CylinderFlow.from_checkpoint(checkpoint)
            for key in ('u','v','p'):assert np.array_equal(getattr(s,key),getattr(restored,key))
            s.solve(n_steps=3,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
            restored.solve(n_steps=3,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
            for key in ('u','v','p'):assert np.max(abs(getattr(s,key)-getattr(restored,key)))<1e-10
    try:solver(ghost_pressure_extension='unknown')
    except ValueError:pass
    else:raise AssertionError('Invalid reconstruction accepted')


def test_observable_window():
    with tempfile.TemporaryDirectory() as directory:
        s=solver();path=Path(directory)/'samples.bin';samples=CylinderSamples(s,path,stride=1)
        for t in (0.,.2,.7,1.1,1.5):
            s.time=t;s.u[:]=np.arange(s.nx+1)*s.dx-(s.xc+s.D/2)-.4+t
            s.p[:]=np.broadcast_to(s.gc.xp,s.p.shape)+2*t
            samples.record()
        samples.close()
        result=mean_observables(path,.1,.5)
        assert abs(result['delta_p_mean']+s.D)<1e-10,result
        assert abs(result['recirculation']['length_D']-.1)<1e-10,result
        try:mean_observables(path,-.1,1.)
        except ValueError:pass
        else:raise AssertionError('Uncovered window accepted')
        with path.open('ab') as stream:stream.write(b'x')
        try:mean_observables(path,.1,.5)
        except ValueError:pass
        else:raise AssertionError('Torn binary sample accepted')
    assert recirculation_length([1.,2.,3.],[-1.,-.5,-.1],1.,1.)['length_D'] is None
    assert recirculation_length([1.,2.,3.],[1.,1.,1.],1.,1.)['length_D'] is None


def test_large_gci_root():
    result=calculate_gci(.7239812526495173,.7260259580264329,.7429881604448899,r=328/246,r32=246/205)
    assert abs(result['p']-12.0747747284)<1e-8,result
    assert result['status']=='conditional estimate' and result['order_search_upper']>10
    exact=calculate_gci(1+.01**2,1+.02**2,1+.03**2,r=2.,r32=1.5)
    assert abs(exact['p']-2)<1e-9,exact
    assert calculate_gci(1.,2.,1.5,r=2.)['p'] is None


def test_pressure_failure_reporting():
    matrix=2*sp.eye(2,format='csr')
    solve,info=monitored_pressure_solver(matrix,np.arange(2),matrix,lambda a:a/2,(1,2),iterative=True)
    assert np.max(abs(solve(np.array([2.,4.]))-[1.,2.]))<1e-12
    assert info['last_solve']['iterations']>0 and info['last_solve']['status']==0
    assert info['last_solve']['residual_target_linf']>=1e-10
    assert info['last_solve']['backward_error_linf']<1e-12
    assert np.array_equal(solve(np.zeros(2)),np.zeros(2))
    with patch('pressure_schur.gmres',side_effect=[(np.array([1.+1e-7,2.]),0),
                                                   (np.array([-1e-7,0.]),0)]) as calls:
        assert np.max(abs(solve(np.array([2.,4.]))-[1.,2.]))<1e-12
        assert calls.call_count==2 and info['last_solve']['refinement_iterations']==1
    for status in (0,1):
        with patch('pressure_schur.gmres',return_value=(np.zeros(2),status)):
            try:solve(np.ones(2))
            except RuntimeError:pass
            else:raise AssertionError('False or failed GMRES convergence accepted')
    try:solve(np.array([np.nan,1.]))
    except ValueError:pass
    else:raise AssertionError('Nonfinite RHS accepted')


def test_candidate_time_order():
    results=[]
    for choices in (dict(ghost_pressure_extension='quadratic'),
                    dict(ghost_velocity_interpolation='biquadratic'),
                    dict(ghost_pressure_extension='quadratic',ghost_velocity_interpolation='biquadratic')):
        states=[]
        timesteps=(.004,.002,.001,.0005,.000125)
        for dt in timesteps:
            s=solver(**choices)
            s.solve(n_steps=round(.064/dt),dt=dt,perturb_transient=False,save_artifacts=False,verbose=False)
            states.append((s.gg.pack(s.u,s.v).copy(),s.p.ravel()[s.gg.pi].copy()))
        errors=np.array([[np.sqrt(np.mean((a[k]-states[-1][k])**2)) for k in (0,1)] for a in states[:-1]])
        orders=np.log2(errors[:-1]/errors[1:])
        results.append(dict(options=choices,timesteps=list(timesteps),errors=errors.tolist(),orders=orders.tolist(),
                            coarse_pair_passed=bool(np.all((orders[0]>1.8)&(orders[0]<2.2))),
                            finest_pairs_passed=bool(np.all((orders[-2:]>1.8)&(orders[-2:]<2.2)))))
        print('Candidate temporal orders:',choices,orders.tolist(),flush=True)
    (ROOT/'candidate_temporal_results.json').write_text(json.dumps(results,indent=2))
    assert all(r['finest_pairs_passed'] for r in results),results


def test_consolidation_and_runner():
    evidence=ROOT/'evidence'
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory)
        paths=[evidence/'stage0_grid_batch.json',evidence/'baseline_four_grid.csv',evidence/'pressure_variant_small_gates.json']
        if not all(p.exists() for p in paths):
            raise RuntimeError('This suite requires the frozen evidence files in the AWS package')
        before=[p.read_bytes() for p in paths]
        result=consolidate(*paths,directory/'consolidated')
        assert result['grid_pair_limits_pass'] and not result['ready_for_paper1']
        assert result['gci']['consecutive']['estimates']['cl_rms']['p']>10
        assert [p.read_bytes() for p in paths]==before
        case=dict(kind='diagnostic',solver=dict(nx=40,ny=30,Lx=4.,Ly=3.,xc=1.5,yc=1.5,
            boundary='ghost_cell',time_scheme='ab2_cn',verbose=False,pressure_solver='iterative'),
            dt=.001,end_time=.02,sample_start=.005,profile_stride=1)
        config=directory/'case.json';config.write_text(json.dumps(case))
        command=[sys.executable,str(ROOT/'validation/run_ghost_validation.py'),'--config-path',str(config)]
        subprocess.run(command+['--output',str(directory/'first')],check=True)
        first=json.loads((directory/'first/summary.json').read_text())
        assert first['completed'] and first['mean_flow_available'] and not first['benchmark_accepted']
        case.update(end_time=.04,sample_start=.025);config.write_text(json.dumps(case))
        subprocess.run(command+['--output',str(directory/'restart'),'--restart',str(directory/'first/checkpoint.npz')],check=True)
        second=json.loads((directory/'restart/summary.json').read_text())
        assert second['completed'] and abs(second['time']-.04)<1e-12
        assert first['pressure_residual_linf']<1e-9 and second['pressure_residual_linf']<1e-9
        from validation.run_ghost_validation import run
        def rejected_step(state,**kwargs):
            state.time=.001
            state.last_step_diagnostics=dict(time=.001,cfl=.6,max_divergence=2e-14,
                                             pressure_residual_linf=1e-12)
            raise RuntimeError('Controlled failed-step diagnostics')
        case.update(end_time=.02,sample_start=.005);config.write_text(json.dumps(case))
        with patch.object(MAC_CylinderFlow,'solve',rejected_step):
            try:run(config,directory/'failed')
            except RuntimeError as error:assert str(error)=='Controlled failed-step diagnostics'
            else:raise AssertionError('Failed integration was accepted')
        failed=json.loads((directory/'failed/summary.json').read_text())
        assert not failed['completed'] and failed['steps']==0 and failed['cfl']==.6
        assert failed['last_step_diagnostics']['time']==.001


def test_frozen_baseline_parity():
    baseline=os.environ.get('GHOST_BASELINE_ROOT')
    if not baseline:
        raise RuntimeError('Set GHOST_BASELINE_ROOT to the original AWS package for the baseline check')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'baseline.npz'
        code="""import sys,numpy as np
sys.path.insert(0,sys.argv[1]+'/solver')
from main import MAC_CylinderFlow
s=MAC_CylinderFlow(nx=40,ny=30,Lx=4.,Ly=3.,xc=1.5,yc=1.5,boundary='ghost_cell',time_scheme='ab2_cn',verbose=False)
s.solve(n_steps=12,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
np.savez(sys.argv[2],u=s.u,v=s.v,p=s.p)
s.save_checkpoint(sys.argv[2]+'.checkpoint.npz')
s.solve(n_steps=3,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
np.savez(sys.argv[2]+'.next.npz',u=s.u,v=s.v,p=s.p)
"""
        subprocess.run([sys.executable,'-c',code,baseline,str(path)],check=True)
        s=solver();s.solve(n_steps=12,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
        with np.load(path,allow_pickle=False) as frozen:
            errors={key:float(np.max(abs(getattr(s,key)-frozen[key]))) for key in ('u','v','p')}
        assert max(errors.values())<1e-12,errors
        print('Frozen baseline parity:',json.dumps(errors))
        converted=Path(directory)/'converted.npz'
        provenance=migrate(str(path)+'.checkpoint.npz',baseline,converted)
        assert not provenance['history_reset']
        restored=MAC_CylinderFlow.from_checkpoint(converted)
        restored.solve(n_steps=3,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
        with np.load(str(path)+'.next.npz',allow_pickle=False) as frozen:
            differences={key:float(np.max(abs(getattr(restored,key)-frozen[key]))) for key in ('u','v','p')}
        assert max(differences.values())<1e-8,differences
        print('Verified backend-only checkpoint migration:',json.dumps(differences))


if __name__=='__main__':
    for test in (test_flat_quadratic,test_pressure_and_backends,test_observable_window,test_large_gci_root,test_pressure_failure_reporting,
                 test_candidate_time_order,test_consolidation_and_runner,test_frozen_baseline_parity):
        test();print(test.__name__+': PASS',flush=True)
