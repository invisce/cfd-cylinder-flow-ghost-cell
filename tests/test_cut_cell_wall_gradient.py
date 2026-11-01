"""Independent polynomial wall derivative, force reaction, and variable-step checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import numpy as np
from cut_cell_solver import CutCellBoundary, extract_cut_cell_surface_forces
from tests.test_wall_face_average_reconstruction import couette_faces


def test_wall_gradient():
    for nx in [100, 200, 400]:
        cc = CutCellBoundary(nx, nx//4, 20., 5., 5., 2.5, .5)
        u, v = couette_faces(cc.geom)
        rate = cc.wall_shear_rate(u, v)
        p = np.zeros((cc.ny, cc.nx))
        np.testing.assert_array_equal(cc.extract_true_surface_forces(u, v, p, .01),
                                      extract_cut_cell_surface_forces(u, v, p, cc.geom, .01))
        torque = float(np.sum(.01*rate*.5*cc.ds))
        error = abs(torque/(4*np.pi*.01*2.5*.5**2)-1)
        print(f'{nx}: analytical torque relative error={error:.6g}')
        if nx == 400:
            assert error < .01, 'Fine-grid analytical torque must be within 1%'


def test_variable_step_ab2():
    cc = CutCellBoundary(20, 8, 10., 4., 3., 2., .5)
    u = np.zeros((8, 21)); v = np.zeros((9, 20))
    cc.prev_au = u.copy(); cc.prev_av = v.copy(); cc.prev_dt = .1
    cc.compute_momentum_rhs = lambda *args, **kwargs: (np.ones_like(u), np.ones_like(v), None, None)
    us, vs = cc.advance_momentum_predictor(u, v, .2, .01, 1., method='ab2')
    # h/(2*h_previous)=1; extrapolated acceleration = 2*1 - 1*0.
    np.testing.assert_allclose([us[1, 1], vs[1, 1]], .4, atol=1e-14)


def test_cut_cell_pressure_and_provenance():
    from main import MAC_CylinderFlow
    s = MAC_CylinderFlow(nx=40, ny=10, boundary='cut_cell', transport='upwind', verbose=False)
    s.solve(n_steps=3, dt=.001, save_artifacts=False, verbose=False)
    # Predictor does not include the old pressure; phi is the new pressure, not an increment.
    np.testing.assert_allclose(s.p, s.phi, atol=0, rtol=0)
    metadata = s.metadata(.001, 3)
    assert {'cut_cell_solver.py', 'cut_cell_geom.py'} <= metadata['source_sha256'].keys()
    assert metadata['perturb_duration'] == 2.5*s.D/s.U_ref


if __name__ == '__main__':
    test_wall_gradient()
    test_variable_step_ab2()
    test_cut_cell_pressure_and_provenance()
    print('PASS: wall derivative, analytical torque and variable-step AB2')
