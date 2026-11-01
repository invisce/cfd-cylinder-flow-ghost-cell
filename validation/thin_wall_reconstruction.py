"""Isolate wall-gradient accuracy on an analytical thin annular shear layer.

psi(s)=-s+delta*(1-exp(-s/delta)), s=max(r-R,0), gives zero wall
velocity, exact open-face averages and du_theta/dr=1/delta at the wall.
This is a reconstruction test, not a Navier-Stokes solution without forcing.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solver'))
from cut_cell_geom import CutCellGeometry2D
from wall_reconstruction import build_wall_gradient


def face_averages(geom,delta):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Require a positive layer thickness')
    x,y = np.meshgrid(np.arange(geom.nx+1)*geom.dx-geom.xc,
                       np.arange(geom.ny+1)*geom.dy-geom.yc)
    distance = np.maximum(np.hypot(x,y)-geom.R,0.)
    psi = -distance-delta*np.expm1(-distance/delta)
    u,v = np.zeros_like(geom.ax),np.zeros_like(geom.ay)
    np.divide(np.diff(psi,axis=0),geom.ax*geom.dy,out=u,where=geom.ax > 0)
    np.divide(-np.diff(psi,axis=1),geom.ay*geom.dx,out=v,where=geom.ay > 0)
    return u,v


def study():
    started = time.perf_counter()
    angles = .17+np.arange(64)*2*np.pi/64
    rows = []
    for n in (40,80,120,160):
        geom = CutCellGeometry2D(n,3*n//4,4.,3.,2.037,1.527,.5)
        operator = build_wall_gradient(geom,angles)
        entry = dict(nx=n,ny=3*n//4,h=geom.dx,diameter_cells=1/geom.dx,layers=[])
        for delta in (.07,.16):
            u,v = face_averages(geom,delta)
            rate = operator@np.r_[u.ravel(),v.ravel()]
            error = abs(delta*rate-1)
            entry['layers'].append(dict(delta=delta,mean_relative_error=float(error.mean()),
                maximum_relative_error=float(error.max()),mean_reconstructed_rate=float(rate.mean()),
                exact_rate=1/delta))
        rows.append(entry)
    return dict(rows=rows,elapsed_seconds=time.perf_counter()-started,
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reconstruction_sha256=hashlib.sha256((ROOT/'solver/wall_reconstruction.py').read_bytes()).hexdigest(),
        scope='Analytical face-average wall-gradient reconstruction only; not a CFD force/reference acceptance gate.',
        ready_for_paper1=False)


if __name__ == '__main__':
    output = ROOT/'reports/thin_wall_reconstruction_current_20261004.json'
    if output.exists():
        raise FileExistsError(output)
    result = study()
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))
