"""Static manufactured-field probes at production spacings; no CFD advancement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','FFT_WORKERS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'solver'),str(ROOT/'scratch')]
import numpy as np
from main import MAC_CylinderFlow
from ghost_wall_probes_20261015 import exact, analytic_selfcheck, norms


def probe(resolution, variant):
    choices=dict(baseline={},pressure=dict(ghost_pressure_extension='quadratic'),
        velocity=dict(ghost_velocity_interpolation='biquadratic'),
        both=dict(ghost_pressure_extension='quadratic',ghost_velocity_interpolation='biquadratic'))
    started=time.perf_counter()
    s=MAC_CylinderFlow(nx=4*resolution,ny=3*resolution,Lx=4.,Ly=3.,xc=1.5,yc=1.5,
        boundary='ghost_cell',time_scheme='ab2_cn',verbose=False,**choices[variant])
    g,gc=s.gg,s.gc
    xu,yu=np.meshgrid(gc.xu-s.xc,gc.yu-s.yc)
    xv,yv=np.meshgrid(gc.xv-s.xc,gc.yv-s.yc)
    xp,yp=np.meshgrid(gc.xp-s.xc,gc.yp-s.yc)
    eu,ev,ep=exact(xu,yu),exact(xv,yv),exact(xp,yp)
    q=np.r_[eu[0].ravel()[g.ui],ev[1].ravel()[g.vi]]
    target=np.r_[eu[3].ravel()[g.ui],ev[4].ravel()[g.vi]]
    x=np.r_[xu.ravel()[g.ui],xv.ravel()[g.vi]]
    y=np.r_[yu.ravel()[g.ui],yv.ravel()[g.vi]]
    distance=np.hypot(x,y)-s.D/2
    interior=np.minimum.reduce((x+s.xc,s.Lx-s.xc-x,y+s.yc,s.Ly-s.yc-y))>3*s.dx
    near=interior & (distance<2*s.dx)
    gradient=g.G @ ep[2].ravel()[g.pi]
    diffusion=g.L @ q+g.diff_offset
    return dict(D_over_h=resolution,R_over_h=resolution/2,variant=variant,
        grid=[s.nx,s.ny],center_over_h=[s.xc/s.dx,s.yc/s.dy],
        gradient=norms(gradient-target,near),diffusion=norms(diffusion-target,near),
        stokes_balance=norms(diffusion-gradient,near),seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolutions',type=int,nargs='+',default=[30,40,50,60,80])
    parser.add_argument('--variants',nargs='+',choices=['baseline','pressure','velocity','both'],default=['baseline','pressure','velocity','both'])
    args=parser.parse_args()
    if args.output.exists() or any(n<10 for n in args.resolutions):
        parser.error('Output must be new and resolutions must be >=10')
    report=dict(completed=False,analytic_selfcheck=analytic_selfcheck(),cases=[],
        scope='Same D/h and integer-center phase as production, on a local 4Dx3D manufactured domain. Resolution and boundary geometry remain confounded; not an isolated parity experiment or a dynamic RMS test.',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'solver').glob('*.py')})
    def save():
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temporary,args.output)
    save()
    try:
        for n in args.resolutions:
            for variant in args.variants:
                result=probe(n,variant);report['cases'].append(result);save()
                print(json.dumps(result,allow_nan=False),flush=True)
        report['completed']=True;save()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';save();raise
