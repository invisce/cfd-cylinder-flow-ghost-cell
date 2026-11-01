"""Independent outer-control-volume force estimate for stationary channel cases.

This continuum quadrature check is not a machine-zero discrete conservation test.
Its boundary interpolation error and omitted transient momentum must be assessed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def outer_forces(u, v, p, dx, dy, mu, rho):
    # Pressure is extrapolated to the rectangle, independently of body sampling.
    pin, pout = 1.5*p[:, 0]-.5*p[:, 1], 1.5*p[:, -1]-.5*p[:, -2]
    pbottom, ptop = 1.5*p[0]-.5*p[1], 1.5*p[-1]-.5*p[-2]
    vin, vout = np.zeros(v.shape[0]), 1.5*v[:, -1]-.5*v[:, -2]
    uy_in = np.gradient(u[:, 0], dy, edge_order=2)
    uy_out = np.gradient(u[:, -1], dy, edge_order=2)
    ux_in = (-3*u[:, 0]+4*u[:, 1]-u[:, 2])/(2*dx)
    ux_out = (3*u[:, -1]-4*u[:, -2]+u[:, -3])/(2*dx)
    vx_in = (9*v[:, 0]-v[:, 1])/(3*dx)
    vx_out = (2*v[:, -1]-3*v[:, -2]+v[:, -3])/dx
    average_y = lambda a: .5*(a[:-1]+a[1:])
    # Quadratic one-sided derivative with the prescribed no-slip wall value.
    uy_bottom = (9*u[0]-u[1])/(3*dy)
    uy_top = -(9*u[-1]-u[-2])/(3*dy)
    vy_bottom = (-3*v[0]+4*v[1]-v[2])/(2*dy)
    vy_top = (3*v[-1]-4*v[-2]+v[-3])/(2*dy)
    convection = rho*np.array([
        np.sum(u[:, 0]**2-u[:, -1]**2)*dy,
        np.sum(u[:, 0]*average_y(vin)-u[:, -1]*average_y(vout))*dy])
    pressure = np.array([np.sum(pin-pout)*dy, np.sum(pbottom-ptop)*dx])
    viscous = mu*np.array([
        2*np.sum(ux_out-ux_in)*dy + np.trapezoid(uy_top-uy_bottom, dx=dx),
        np.sum(uy_out-uy_in+average_y(vx_out-vx_in))*dy + 2*np.sum(vy_top-vy_bottom)*dx])
    return dict(convection=convection, pressure=pressure, viscous=viscous,
                force=convection+pressure+viscous)


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    case = manifest['configuration']
    summary = json.loads(sorted(folder.glob('summary-*.json'))[-1].read_text())
    if case['kind'] == 'periodic' or not summary['stationary'] or case['solver'].get('slip_walls', False):
        raise ValueError('Requires a stationary, no-slip channel case')
    path = folder/'checkpoint.npz'
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['checkpoint_metadata']))
        if (meta['source_sha256'] != manifest['metadata']['source_sha256']
                or not np.isclose(meta['time'], case['end_time'], rtol=0, atol=1e-10)):
            raise ValueError('Checkpoint is not the matching completed state')
        s = case['solver']
        dx, dy = s['Lx']/s['nx'], s['Ly']/s['ny']
        terms = outer_forces(data['u'], data['v'], data['p'], dx, dy, s['rho']*s['nu'], s['rho'])
    norm = .5*s['rho']*manifest['metadata']['U_ref']**2*s['D']
    surface = norm*np.array([meta['diagnostics']['cd'], meta['diagnostics']['cl']])
    return dict(terms={k: value.tolist() for k, value in terms.items()},
                surface_force=surface.tolist(), force_difference=(terms['force']-surface).tolist(),
                coefficient_difference=((terms['force']-surface)/norm).tolist(),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Steady assumption; independent continuum boundary quadrature; assess spatial convergence, not machine-zero equality',
                ready_for_paper1=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.run), indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(result)
    print(result)
