import sys
from pathlib import Path
import numpy as np
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
from main import MAC_CylinderFlow
import numba_kernels as nk
import optimized_ghost_cell as ghost


def solver(nx=88,ny=16,**kwargs):
    return MAC_CylinderFlow(nx=nx,ny=ny,Lx=22.,Ly=4.1,xc=2.,yc=2.,
        boundary='ghost_cell',time_scheme='ab2_cn',verbose=False,**kwargs)


def test_operator_and_backend_parity():
    s=solver(); g=s.gg
    q=np.random.default_rng(6).normal(size=g.qsize)
    compiled=g.advection(q).copy()
    assert np.max(abs(compiled-g.advection(q,compiled=False)))<1e-11
    u,v=g.unpack(q)
    nk.compute_flux_u_numba(u,v,g.ones,g.corners,s.dx,s.dy,1.,s.nu,False,True,False,g.fxx,g.fxy)
    nk.accumulate_rhs_u_numba(g.fxx,g.fxy,u,g.apertures,s.dy,1.,False,g.ru)
    nk.compute_flux_v_numba(u,v,g.ones,g.corners,s.dx,s.dy,1.,s.nu,True,False,g.fyy,g.fyx)
    nk.accumulate_rhs_v_numba(g.fyy,g.fyx,g.rv)
    diffusion=g.pack(g.ru,g.rv)/(s.dx*s.dy)
    diffusion[:len(g.ui)][np.unravel_index(g.ui,s.u.shape)[1]==s.nx]*=2
    assert np.max(abs(diffusion-g.diffusion(q)))<1e-11
    projected,phi=g.project(q,.003)
    assert np.max(abs(g.divergence(projected)))<1e-10
    assert np.max(abs(g.A @ phi-g.divergence(q)/.003))<1e-9
    u,v=g.unpack(projected)
    assert np.max(abs(g.pack(u,v)-projected))<1e-12


def test_time_order():
    states=[]
    for dt in (.004,.002,.001,.00025):
        s=solver()
        s.solve(n_steps=round(.064/dt),dt=dt,perturb_transient=False,save_artifacts=False,verbose=False)
        states.append((s.gg.pack(s.u,s.v),s.p.ravel()[s.gg.pi]))
    errors=np.array([[np.sqrt(np.mean((state[k]-states[-1][k])**2)) for k in (0,1)] for state in states[:-1]])
    orders=np.log2(errors[:-1]/errors[1:])
    print('Ghost coupled velocity/pressure time orders',orders.tolist())
    assert np.all((orders>1.8)&(orders<2.2)),orders


def test_physical_surface_traction():
    errors=[]
    for nx,ny in ((40,30),(80,60),(120,90)):
        s=MAC_CylinderFlow(nx=nx,ny=ny,Lx=4.,Ly=3.,xc=1.5,yc=1.5,
            boundary='ghost_cell',time_scheme='ab2_cn',verbose=False)
        gc,g=s.gc,s.gg
        xx,yy=np.meshgrid(gc.xu,gc.yu); rr=np.hypot(xx-s.xc,yy-s.yc)
        u=-(rr-s.D/2)*(yy-s.yc)/rr
        xx,yy=np.meshgrid(gc.xv,gc.yv); rr=np.hypot(xx-s.xc,yy-s.yc)
        v=(rr-s.D/2)*(xx-s.xc)/rr
        p=np.broadcast_to(gc.xp,s.p.shape)
        forces=g.surface_forces(u,v,p,s.nu*s.rho)
        assert abs(forces[2]+np.pi*(s.D/2)**2)<1e-10
        assert abs(forces[3])<1e-10
        dn=np.column_stack((g.du_dn @ u.ravel(),g.dv_dn @ v.ravel()))
        tangential=-g.normals[:,1]*dn[:,0]+g.normals[:,0]*dn[:,1]
        torque=s.nu*(s.D/2)*np.sum(tangential)*(np.pi*s.D/len(tangential))
        exact=2*np.pi*s.nu*(s.D/2)**2
        errors.append(abs(torque-exact)/exact)
    print('Ghost analytical torque relative errors',errors)
    assert errors[2]<errors[1]<errors[0] and errors[-1]<.01


def test_compiled_vector_parity():
    compiled=solver(); reference=solver()
    compiled.solve(n_steps=12,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
    original=ghost.ab2_cn_step
    def reference_step(*args,**kwargs):
        kwargs['compiled_vectors']=False
        return original(*args,**kwargs)
    with patch.object(ghost,'ab2_cn_step',reference_step):
        reference.solve(n_steps=12,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
    for key in ('u','v','p'):
        assert np.max(abs(getattr(compiled,key)-getattr(reference,key)))<1e-10,key
    for kernel in (nk.pack_numba,nk.explicit_ab2_numba,nk.assemble_cn_rhs_numba,nk.compute_coupled_residual_numba):
        assert kernel.nopython_signatures,kernel


def test_complete_restart():
    continuous=solver(); divided=solver()
    continuous.solve(n_steps=20,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
    divided.solve(n_steps=7,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'checkpoint.npz'
        divided.save_checkpoint(path)
        resumed=MAC_CylinderFlow.from_checkpoint(path)
        resumed.solve(n_steps=13,dt=.001,perturb_transient=False,save_artifacts=False,verbose=False)
        for key in ('u','v','p'):
            assert np.max(abs(getattr(continuous,key)-getattr(resumed,key)))<1e-11,key
        with np.load(path,allow_pickle=False) as data:
            arrays={key:data[key].copy() for key in data.files if key!='half_pressure'}
        np.savez(path,**arrays)
        try:
            MAC_CylinderFlow.from_checkpoint(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing Ghost pressure history accepted')


if __name__=='__main__':
    test_operator_and_backend_parity()
    test_time_order()
    test_physical_surface_traction()
    test_compiled_vector_parity()
    test_complete_restart()
    print('Ghost Numba parity, coupled projection, temporal order and traction: PASS')
