"""Independent manufactured outlet balance and physical staggered volume checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import cut_cell_solver as cut
from main import MAC_CylinderFlow


def test_natural_outlet_balance():
    solver = MAC_CylinderFlow(nx=120, ny=40, Lx=12., Ly=4., D=1., xc=2., yc=2.,
        nu=.01, rho=1., U_inf=1., boundary='cut_cell', time_scheme='ab2_cn',
        pressure_solver='cg_dct', slip_walls=True, verbose=False)
    cc = solver.cc
    expected_volume = solver.Lx*solver.Ly-np.pi*(solver.D/2)**2
    np.testing.assert_allclose(cc.u_vol.sum(), expected_volume, rtol=1e-10)
    np.testing.assert_allclose(cc.v_vol.sum(), expected_volume, rtol=1e-10)
    np.testing.assert_allclose(cc.u_vol[:, -1], .5*solver.dx*solver.dy)
    np.testing.assert_allclose(cc.v_vol[[0, -1]], .5*solver.dx*solver.dy)

    # This local affine patch is divergence-free, has zero Laplacian, and
    # satisfies both components of natural traction at the straight outlet.
    slope = .2
    u = np.broadcast_to(slope*np.arange(solver.nx+1)*solver.dx, solver.u.shape).copy()
    v = np.broadcast_to(-slope*np.arange(solver.ny+1)[:, None]*solver.dy, solver.v.shape).copy()
    p = np.full_like(solver.p, solver.rho*solver.nu*slope)
    operators = {}
    class Captured(Exception):
        pass
    def capture(q, pressure, dt, **callbacks):
        operators.update(callbacks)
        raise Captured
    original = cut.ab2_cn_step
    cut.ab2_cn_step = capture
    try:
        cc.advance_ab2_cn(u, v, p, .002, solver.nu, solver.rho, u[:, 0])
    except Captured:
        pass
    finally:
        cut.ab2_cn_step = original
    mask = solver.cut_geom.ax[:, 1:] > 0
    pressure_gradient = np.zeros_like(u)
    pressure_gradient[:, 1:][mask] = operators['gradient'](p)[:int(mask.sum())]
    au, _, _, _ = cc.compute_momentum_rhs(u, v, solver.nu, solver.rho,
        transport='centered', include_advection=False)
    # Exclude wall stencils: this is an outlet patch, not a solved channel.
    np.testing.assert_allclose((au-pressure_gradient)[2:-2, -1], 0., atol=1e-12)

    # Uniform transport cannot create an artificial normal-outlet momentum.
    au, _, _, _ = cc.compute_momentum_rhs(np.ones_like(u), np.zeros_like(v),
        solver.nu, solver.rho, transport='centered')
    np.testing.assert_allclose(au[:, -1], 0., atol=1e-12)


if __name__ == '__main__':
    test_natural_outlet_balance()
    print('Natural outlet balance and physical staggered volumes PASS')
