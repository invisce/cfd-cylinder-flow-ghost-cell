"""Exact fluid-block inverse and unchanged pressure solves on displaced circles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import numpy as np
from scipy.fft import dct, idct
from scipy.sparse.linalg import LinearOperator
from cut_cell_solver import CutCellBoundary, build_cut_cell_poisson_matrix
from pressure_schur import boundary_schur_preconditioner


def test_pressure_schur():
    for nx, ny, xc, yc in ((40,16,2.37,1.53),(80,32,2.331,1.511)):
        cc=CutCellBoundary(nx,ny,8.,3.2,xc,yc,.5,pressure_solver='cg_dct')
        def inverse(rhs):
            spectrum=dct(dct(rhs,type=4,axis=1,norm='ortho'),type=2,axis=0,norm='ortho')
            return idct(idct(spectrum/cc.pressure_eigenvalues,type=2,axis=0,norm='ortho'),type=4,axis=1,norm='ortho')
        preconditioner, info=boundary_schur_preconditioner(cc.geom,build_cut_cell_poisson_matrix,inverse)
        cc.pressure_preconditioner=LinearOperator(cc.pressure_operator.shape, dtype=float,
                                                matvec=cc._pressure_precondition)
        actual=np.random.default_rng(11).normal(size=int(cc.fluid_pressure.sum()))
        rhs=cc.pressure_operator @ actual
        np.testing.assert_allclose(preconditioner @ rhs,actual,atol=2e-9,rtol=0)
        grid=cc._pressure_grid(-rhs)
        plain=cc.solve_pressure(grid,.002,1.)
        before=cc.last_pressure_diagnostics['iterations']
        cc.pressure_preconditioner=preconditioner
        accelerated=cc.solve_pressure(grid,.002,1.)
        after=cc.last_pressure_diagnostics['iterations']
        np.testing.assert_allclose(accelerated,plain,atol=1e-9,rtol=0)
        assert after <= 3 and after < before
        assert cc.last_pressure_diagnostics['predicted_divergence'] < 1e-10
        print(nx,ny,info,'CG iterations',before,'->',after)
        fallback, note=boundary_schur_preconditioner(cc.geom,build_cut_cell_poisson_matrix,inverse,max_rank=1)
        assert fallback is None and note['reason']=='interface_rank_limit'


if __name__=='__main__':
    test_pressure_schur()
    print('Boundary Schur pressure checks PASS')
