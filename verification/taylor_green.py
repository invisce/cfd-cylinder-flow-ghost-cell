"""Periodic staggered Taylor-Green verification using the production AB2/CN step."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import numpy as np
from scipy.fft import fft2, ifft2
from time_integration import ab2_cn_step
from cut_cell_solver import periodic_momentum_operator


def run(n, dt, end_time=.4, nu=.1):
    if n < 8 or not np.isfinite([dt, end_time, nu]).all() or min(dt, end_time, nu) <= 0:
        raise ValueError('Require n>=8 and finite positive dt, end_time and nu')
    steps = round(end_time/dt)
    if steps < 1 or not np.isclose(steps*dt, end_time):
        raise ValueError('end_time must be an integer multiple of dt')
    h = 2*np.pi/n
    yu, xu = np.meshgrid((np.arange(n)+.5)*h, np.arange(n)*h, indexing='ij')
    yv, xv = np.meshgrid(np.arange(n)*h, (np.arange(n)+.5)*h, indexing='ij')
    yp, xp = np.meshgrid((np.arange(n)+.5)*h, (np.arange(n)+.5)*h, indexing='ij')
    pack = lambda u, v: np.r_[u.ravel(), v.ravel()]
    unpack = lambda q: q.reshape(2, n, n)
    def exact(t):
        a = np.exp(-2*nu*t)
        return (pack(a*np.sin(xu)*np.cos(yu), -a*np.cos(xv)*np.sin(yv)),
                .25*a*a*(np.cos(2*xp)+np.cos(2*yp)))
    flux = periodic_momentum_operator(n, h)
    advection = lambda q: flux(q, 0., True)
    diffusion = lambda q: flux(q, nu, False)
    def gradient(p):
        return pack(p-np.roll(p, 1, 1), p-np.roll(p, 1, 0))/h
    def divergence(q):
        u, v = unpack(q)
        return (np.roll(u, -1, 1)-u+np.roll(v, -1, 0)-v)/h
    eigenvalues = -4*(np.sin(np.pi*np.arange(n)/n)[:, None]**2
                       + np.sin(np.pi*np.arange(n)/n)[None, :]**2)/h**2
    eigenvalues[0, 0] = 1.
    def project(q, step):
        rhs = fft2(divergence(q)/step)
        rhs[0, 0] = 0.
        phi = ifft2(rhs/eigenvalues).real
        return q-step*gradient(phi), phi
    q, p = exact(0.)
    history = previous_dt = None
    max_divergence = 0.
    for _ in range(steps):
        q, p, output, history, _ = ab2_cn_step(q, p, dt, advection=advection, diffusion=diffusion,
            gradient=gradient, project=project, divergence=divergence, nu=nu,
            previous_advection=history, previous_dt=previous_dt)
        previous_dt = dt
        max_divergence = max(max_divergence, float(np.max(abs(divergence(q)))))
    q_exact, p_exact = exact(end_time)
    p_error = output-output.mean()-p_exact
    norms = lambda a: dict(l1=float(np.mean(abs(a))), l2=float(np.sqrt(np.mean(a*a))), linf=float(np.max(abs(a))))
    return dict(n=n, dt=dt, end_time=end_time, velocity=norms(q-q_exact), pressure=norms(p_error),
                max_divergence=max_divergence), q, output-output.mean()


def verify():
    spatial = [run(n, .0025)[0] for n in (32, 64, 128, 256)]
    for field in ('velocity', 'pressure'):
        for norm in ('l1', 'l2', 'linf'):
            orders = np.log2([spatial[i][field][norm]/spatial[i+1][field][norm] for i in range(3)])
            print(field, 'spatial', norm, 'orders', orders)
            assert np.all((orders > 1.8) & (orders < 2.2))
    _, qr, pr = run(32, .0003125)
    temporal = []
    for dt in (.04, .02, .01, .005):
        info, q, p = run(32, dt)
        temporal.append(dict(dt=dt, velocity_l2=float(np.linalg.norm(q-qr)/np.sqrt(q.size)),
                             pressure_l2=float(np.linalg.norm(p-pr)/np.sqrt(p.size))))
    for field in ('velocity_l2', 'pressure_l2'):
        orders = np.log2([temporal[i][field]/temporal[i+1][field] for i in range(3)])
        print(field, 'temporal orders', orders)
        assert np.all((orders > 1.8) & (orders < 2.2))
    assert max(c['max_divergence'] for c in spatial) < 1e-10
    return dict(spatial=spatial, temporal=temporal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Refusing to overwrite prior evidence')
    result = verify()
    root = Path(__file__).resolve().parents[1]
    result['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('solver/time_integration.py', 'solver/cut_cell_solver.py',
                     'solver/wall_reconstruction.py', 'solver/pressure_schur.py', 'solver/staggered_wall.py',
                     'verification/taylor_green.py')}
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False))
    print('Taylor-Green checks PASS')
