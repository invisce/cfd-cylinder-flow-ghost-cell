"""Small, runnable pressure-backend equivalence and true-residual checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import numpy as np
from cut_cell_solver import CutCellBoundary, apply_cut_cell_poisson, build_cut_cell_poisson_matrix
from main import MAC_CylinderFlow


def test_pressure_backends():
    results = []
    for mode in ('direct', 'iterative', 'cg_dct'):
        cc = CutCellBoundary(40, 16, 8., 3.2, 2.37, 1.53, .5, pressure_solver=mode)
        p = np.random.default_rng(12).normal(size=(16, 40))
        A = build_cut_cell_poisson_matrix(cc.geom)
        np.testing.assert_allclose(apply_cut_cell_poisson(p, cc.geom).ravel(), A @ p.ravel(), atol=1e-12)
        p[cc.geom.cell_type == -1] = 0.
        rhs = apply_cut_cell_poisson(p, cc.geom)
        actual = cc.solve_pressure(rhs, .002, 1.)
        np.testing.assert_allclose(actual, p, atol=1e-9, rtol=0)
        assert cc.last_pressure_diagnostics['predicted_divergence'] < 1e-10
        s = MAC_CylinderFlow(nx=40, ny=10, boundary='cut_cell', pressure_solver=mode, verbose=False)
        s.solve(n_steps=5, dt=.001, perturb_transient=False, save_artifacts=False)
        assert s.last_step_diagnostics['max_divergence'] < 1e-10
        results.append(s.u.copy())
        print(mode, cc.last_pressure_diagnostics)
    for u in results[1:]:
        np.testing.assert_allclose(u, results[0], atol=1e-10, rtol=0)
    # Odd grid: several centers land exactly on the circle (3-4-5 triangle).
    # Roundoff in the exterior filter must not create a spurious pressure lift.
    s = MAC_CylinderFlow(nx=100, ny=25, boundary='cut_cell', time_scheme='ab2_cn',
                        pressure_solver='cg_dct', verbose=False)
    s.solve(n_steps=20, dt=.002, perturb_transient=False, save_artifacts=False)
    assert abs(s.last_step_diagnostics['cl']) < 1e-10


if __name__ == '__main__':
    test_pressure_backends()
    print('Pressure backend checks PASS')
