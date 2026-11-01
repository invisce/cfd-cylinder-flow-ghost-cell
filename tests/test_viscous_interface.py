import sys
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solver'))
from main import MAC_CylinderFlow
from optimized_cut_cell import apply_cut_cell_optimizations
from wall_reconstruction import cartesian_circle_basis, _basis


def make():
    assert psutil.virtual_memory().available > 256*2**20
    return MAC_CylinderFlow(nx=60,ny=30,Lx=6.,Ly=3.,xc=1.5,yc=1.5,
        boundary='cut_cell',time_scheme='ab2_cn',pressure_solver='cg_dct',verbose=False)


def test_viscous_interface():
    s = make()
    cc,g = s.cc,s.cc.geom
    angle = .73
    x,y = g.xc+g.R*np.cos(angle),g.yc+g.R*np.sin(angle)
    boundary = cartesian_circle_basis(x,y,g,angle)
    assert np.max(abs(boundary['u'])) < 1e-12
    assert np.max(abs(boundary['v'])) < 1e-12
    x,y = x+.13,y+.07
    np.testing.assert_allclose(_basis(x,y,g,angle,max(g.dx,g.dy),True),
        cartesian_circle_basis(x,y,g,angle)['psi'],atol=1e-14,rtol=1e-14)
    delta = 1e-6
    center = cartesian_circle_basis(x,y,g,angle)
    for axis,key in ((0,'x'),(1,'y')):
        dx,dy = (delta,0.) if axis == 0 else (0.,delta)
        low = cartesian_circle_basis(x-dx,y-dy,g,angle)
        high = cartesian_circle_basis(x+dx,y+dy,g,angle)
        for velocity in ('u','v'):
            np.testing.assert_allclose((high[velocity]-low[velocity])/(2*delta),
                center[velocity+key],atol=1e-7,rtol=1e-8)
    rng = np.random.default_rng(74)
    u,v = rng.normal(size=s.u.shape),rng.normal(size=s.v.shape)
    u[g.ax==0],v[g.ay==0] = 0.,0.
    cases = ((0.,True),(.01,False),(.01,True))
    expected = [tuple(a.copy() for a in cc.compute_momentum_rhs(u,v,nu,1.,'centered',adv))
                for nu,adv in cases]
    for au,av,ru,rv in expected:
        for acceleration,rhs,volume in ((au,ru,cc.u_vol),(av,rv,cc.v_vol)):
            np.testing.assert_allclose(np.sum(acceleration*volume),rhs.sum(),
                atol=1e-12,rtol=1e-12)
    field = np.r_[u.ravel(),v.ravel()]
    for component in ('u','v'):
        fluxes = []
        for axis in ('x','y'):
            stencil = cc.diffusion_faces[component,axis]
            shape = ({('u','x'):g.ax_mid,('u','y'):g.ay_u,
                ('v','x'):g.ax_v,('v','y'):g.ay_mid}[component,axis]).shape
            flux = np.zeros(shape)
            flux.ravel()[stencil.indices] = stencil.correction @ field
            if (component,axis) in (('u','y'),('v','x')):
                outer = flux[[0,-1],:] if axis == 'y' else flux[:,[0,-1]]
                assert np.max(abs(outer)) == 0.
            fluxes.append(flux)
        fx,fy = fluxes
        rhs = np.zeros_like(u if component == 'u' else v)
        if component == 'u':
            rhs[:,:-1] += fx
            rhs[:,1:] -= fx
            rhs += np.diff(fy,axis=0)
        else:
            rhs[:-1,:] += fy
            rhs[1:,:] -= fy
            rhs += np.diff(fx,axis=1)
        assert abs(rhs.sum()) < 1e-12*max(1.,np.abs(rhs).sum())
    apply_cut_cell_optimizations(s)
    maximum = 0.
    for (nu,adv),reference in zip(cases,expected):
        actual = cc.compute_momentum_rhs(u,v,nu,1.,'centered',adv)
        for a,b in zip(actual,reference):
            maximum = max(maximum,float(np.max(abs(a-b))))
            np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-11)
    baseline = make()
    for candidate in (baseline,s):
        candidate.solve(n_steps=10,dt=.001,perturb_transient=False,save_artifacts=False)
        assert candidate.last_step_diagnostics['max_divergence'] < 1e-10
        assert candidate.last_step_diagnostics['cfl'] < .5
    for key in ('u','v','p'):
        np.testing.assert_allclose(getattr(s,key),getattr(baseline,key),atol=2e-9,rtol=1e-9)
    assert psutil.Process().memory_info().rss < 4*2**30
    assert psutil.virtual_memory().available > 256*2**20
    print('PASS: no-slip basis, independent derivatives, all momentum branches and 10 AB2/CN steps; max RHS difference',maximum)


if __name__ == '__main__':
    test_viscous_interface()
