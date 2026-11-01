"""Production check: harmonic Couette velocity has zero viscous acceleration."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))
import numpy as np
from cut_cell_solver import CutCellBoundary
from tests.test_wall_face_average_reconstruction import couette_faces


def test_harmonic_viscous_consistency():
    errors=[]
    for nx in (100,200,400):
        cc=CutCellBoundary(nx,nx//4,20.,5.,5.,2.5,.5,pressure_solver='cg_dct')
        u,v=couette_faces(cc.geom)
        au,av,ru,rv=cc.compute_momentum_rhs(u,v,.01,1.,include_advection=False)
        error=0.
        for acceleration,ap,x,y in ((au,cc.geom.ax,cc.xu,cc.yu),(av,cc.geom.ay,cc.xv,cc.yv)):
            xx,yy=np.meshgrid(x-5.,y-2.5)
            mask=(ap>0)&(np.hypot(xx,yy)<.5+2*cc.dx)
            error=max(error,float(np.max(abs(acceleration[mask]))))
        errors.append(error)
        print(nx,'harmonic viscous acceleration error',error,flush=True)
    assert errors[2]<errors[1]<errors[0],errors


def test_projected_viscous_spectrum():
    # Assemble the actual projected linear operator independently, column by
    # column. Shifted circles previously exposed positive growth eigenvalues.
    # This is a modal check on two geometries, not a general stability proof.
    for args in ((24,12,4.,2.,1.03,1.01,.35), (20,20,2.,2.,1.03,1.01,.4)):
        cc = CutCellBoundary(*args, pressure_solver='cg_dct')
        umask, vmask = cc.geom.ax[:,1:]>0, cc.geom.ay[1:-1]>0
        nu = int(umask.sum())
        size = nu+int(vmask.sum())
        matrix = np.empty((size,size))
        for k in range(size):
            q = np.zeros(size)
            q[k] = 1.
            u, v = np.zeros_like(cc.geom.ax), np.zeros_like(cc.geom.ay)
            u[:,1:][umask], v[1:-1][vmask] = q[:nu], q[nu:]
            au, av, _, _ = cc.compute_momentum_rhs(u,v,1.,1.,include_advection=False)
            rhs = np.zeros((cc.ny,cc.nx))
            cc.compute_rhs(au,av,1.,1.,rhs)
            pressure = cc.solve_pressure(rhs,1.,1.)
            pu, pv = np.zeros_like(u), np.zeros_like(v)
            cc.apply_corrector(pu,pv,au,av,pressure,1.,1.,np.zeros(cc.ny))
            matrix[:,k] = np.r_[pu[:,1:][umask],pv[1:-1][vmask]]
        growth = float(np.linalg.eigvals(matrix).real.max())
        print(args[:2], 'projected viscous spectral abscissa', growth, flush=True)
        assert growth < 1e-8, growth
        for volume, fraction, aperture in ((cc.u_vol,cc.geom.u_volume_fraction,cc.geom.ax),
                                            (cc.v_vol,cc.geom.v_volume_fraction,cc.geom.ay)):
            np.testing.assert_allclose(volume.sum(),cc.geom.Lx*cc.geom.Ly-np.pi*cc.geom.R**2,rtol=1e-10)
            assert np.all(volume[aperture==0] == 0)


def test_covered_faces_are_not_velocity_dofs():
    cc = CutCellBoundary(100,25,20.,5.,5.,2.5,.5,pressure_solver='cg_dct')
    rng = np.random.default_rng(83)
    u, v = rng.normal(size=cc.geom.ax.shape), rng.normal(size=cc.geom.ay.shape)
    u[cc.geom.ax==0], v[cc.geom.ay==0] = 0., 0.
    changed_u, changed_v = u.copy(), v.copy()
    changed_u[cc.geom.ax==0], changed_v[cc.geom.ay==0] = 173., -129.
    for advection in (True,False):
        actual = cc.compute_momentum_rhs(u,v,.01,1.,include_advection=advection)
        changed = cc.compute_momentum_rhs(changed_u,changed_v,.01,1.,include_advection=advection)
        for a, b in zip(actual,changed):
            np.testing.assert_array_equal(a,b)


if __name__=='__main__':
    test_harmonic_viscous_consistency()
    test_projected_viscous_spectrum()
    test_covered_faces_are_not_velocity_dofs()
    print('Production staggered viscous consistency PASS')
