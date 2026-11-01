"""Independent diagnostic of the DFG do-nothing outlet traction.

Uses one-sided boundary extrapolation, not the pressure-correction boundary value.
Small residuals do not prove discrete boundary-condition equivalence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def outlet_traction(u, v, p, dx, mu):
    if (p.ndim != 2 or min(p.shape) < 3 or
            u.shape != (p.shape[0], p.shape[1]+1) or
            v.shape != (p.shape[0]+1, p.shape[1]) or
            not np.isfinite([dx, mu]).all() or dx <= 0 or mu < 0 or
            not all(np.isfinite(a).all() for a in (u, v, p))):
        raise ValueError('Invalid outlet fields, spacing or viscosity')
    # The reference uses mu*partial_x(velocity)-p*(1,0), not symmetric stress.
    ux = (3*u[:, -1]-4*u[:, -2]+u[:, -3])/(2*dx)
    vx = (2*v[:, -1]-3*v[:, -2]+v[:, -3])/dx
    pressure = 1.5*p[:, -1]-.5*p[:, -2]
    normal, tangent = mu*ux-pressure, mu*vx
    norm = lambda a: dict(linf=float(np.max(abs(a))), l2=float(np.sqrt(np.mean(a*a))))
    return dict(normal=norm(normal), tangent=norm(tangent),
                boundary_pressure=norm(pressure), velocity_normal_derivative=norm(ux))


def audit(checkpoint):
    with np.load(checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data['checkpoint_metadata']))
        config = metadata['configuration']
        result = outlet_traction(data['u'], data['v'], data['p'],
                                 config['Lx']/config['nx'], config['rho']*config['nu'])
    return dict(checkpoint=str(checkpoint.resolve()), time=metadata['time'],
                checkpoint_source_sha256=metadata['source_sha256'],
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                traction=result, boundary_equivalence_proven=False,
                caveat='Continuum extrapolation diagnostic; grid convergence and discrete boundary audit remain required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['traction']))
