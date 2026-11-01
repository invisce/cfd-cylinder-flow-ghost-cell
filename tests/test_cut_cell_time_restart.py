"""Production curved-wall time convergence, operator split and complete restart checks."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import numpy as np
from main import MAC_CylinderFlow
from time_integration import ab2_cn_step


def make():
    return MAC_CylinderFlow(nx=40, ny=10, boundary='cut_cell', time_scheme='ab2_cn',
                            pressure_solver='cg_dct', verbose=False)


def test_time_and_restart():
    fields = []
    for dt in (.004, .002, .001, .0005):
        s = make()
        s.solve(n_steps=round(.08/dt), dt=dt, perturb_transient=False, save_artifacts=False)
        fields.append((s.u.copy(), s.p.copy()))
        assert s.last_step_diagnostics['max_divergence'] < 1e-10
    for k in (0, 1):
        errors = [np.linalg.norm(fields[i][k]-fields[i+1][k]) for i in range(3)]
        orders = np.log2(np.array(errors[:-1])/errors[1:])
        print('Curved-wall velocity/pressure', k, 'time orders:', orders)
        assert np.all((orders > 1.8) & (orders < 2.2))
    s = make()
    all_terms = s.cc.compute_momentum_rhs(s.u, s.v, s.nu, s.rho, transport='centered')
    adv = s.cc.compute_momentum_rhs(s.u, s.v, 0., s.rho, transport='centered')
    visc = s.cc.compute_momentum_rhs(s.u, s.v, s.nu, s.rho, transport='centered', include_advection=False)
    for total, a, d in zip(all_terms, adv, visc):
        np.testing.assert_allclose(total, a+d, atol=1e-13, rtol=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'restart.npz'
        s.solve(n_steps=7, dt=.001, save_artifacts=False)
        s.save_checkpoint(path)
        restored = MAC_CylinderFlow.from_checkpoint(path)
        for candidate in (s, restored):
            candidate.solve(n_steps=9, dt=.0005, save_artifacts=False, **candidate.run_options)
        for field in ('u', 'v', 'p'):
            np.testing.assert_array_equal(getattr(s, field), getattr(restored, field))
        assert s.time == restored.time
        try:
            s.save_checkpoint(path)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Checkpoint was silently overwritten')
        with np.load(path, allow_pickle=False) as data:
            content = {k: data[k].copy() for k in data.files}
        metadata = json.loads(str(content['checkpoint_metadata']))
        metadata['source_sha256']['time_integration.py'] = 'changed'
        content['checkpoint_metadata'] = json.dumps(metadata)
        np.savez(Path(directory)/'bad.npz', **content)
        try:
            MAC_CylinderFlow.from_checkpoint(Path(directory)/'bad.npz')
        except ValueError:
            pass
        else:
            raise AssertionError('Changed-code checkpoint accepted')


def test_noncommuting_pressure_recovery():
    # Constrained analytical ODE: q_x=0, q_y=exp(-3t), p=-exp(-3t).
    # Its viscous operator couples gradient and solenoidal components,
    # as cut-cell/merged operators do; Cartesian commutation is unavailable.
    L = np.array([[-2., -1.], [-1., -3.]])
    def project(q, dt):
        return np.array([0., q[1]]), np.array([q[0]/dt])
    errors = []
    for dt in (.02, .01, .005):
        q, p = np.array([0., 1.]), np.array([-1.])
        history = previous_dt = None
        for _ in range(round(.2/dt)):
            q, p, output, history, _ = ab2_cn_step(q, p, dt,
                advection=lambda q: np.zeros_like(q), diffusion=lambda q: L @ q,
                gradient=lambda p: np.array([p[0], 0.]), project=project,
                divergence=lambda q: q[:1], nu=1., previous_advection=history,
                previous_dt=previous_dt)
            previous_dt = dt
        errors.append(abs(output[0]+np.exp(-.6)))
        assert q[0] == 0.
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all((orders > 1.8) & (orders < 2.2)), orders


def test_fine_grid_pressure_stability():
    # This actual DFG cut geometry failed at step 7 under the continuum
    # rotational formula. Keep the original timestep and CFL gate.
    s = MAC_CylinderFlow(nx=440, ny=82, Lx=22., Ly=4.1, xc=2., yc=2., nu=.05,
                        inflow='parabolic', U_m=1.5, boundary='cut_cell',
                        transport='centered', time_scheme='ab2_cn',
                        pressure_solver='cg_dct', verbose=False)
    s.solve(n_steps=40, dt=.005, perturb_transient=False, save_artifacts=False)
    assert s.time == .2 and s.last_step_diagnostics['cfl'] < .5
    assert s.last_step_diagnostics['max_divergence'] < 1e-10


if __name__ == '__main__':
    test_time_and_restart()
    test_noncommuting_pressure_recovery()
    test_fine_grid_pressure_stability()
    print('Time integration and restart checks PASS')
