"""Separable MAC pressure and Helmholtz inverses with fixed channel boundaries."""
import numpy as np
import scipy.sparse as sp
from scipy.fft import dct,idct,dst,idst
import os


def rectangle(nx,ny,dx,dy,kind,nu=0.,dt=0.,rho=1.):
    if kind=='pressure':
        xx,yy=nx,ny
        left,right,lower,upper=1.,3.,1.,1.
        sx=(np.arange(nx)+.5)*np.pi/(2*nx)
        sy=np.arange(ny)*np.pi/(2*ny)
        tx,ty=4,2
        forward,backward=dct,idct
    elif kind=='u':
        xx,yy=nx,ny
        left,right,lower,upper=2.,2.,3.,3.
        sx=(np.arange(nx)+.5)*np.pi/(2*nx)
        sy=(np.arange(ny)+1)*np.pi/(2*ny)
        tx,ty=3,2
        forward,backward=dst,idst
    elif kind=='v':
        xx,yy=nx,ny-1
        left,right,lower,upper=3.,1.,2.,2.
        sx=(np.arange(nx)+.5)*np.pi/(2*nx)
        sy=(np.arange(ny-1)+1)*np.pi/(2*ny)
        tx,ty=4,1
        forward,backward=dst,idst
    else:
        raise ValueError('Unknown rectangular MAC field')
    diagx=np.full(xx,-2.); diagx[0]=-left; diagx[-1]=-right
    diagy=np.full(yy,-2.); diagy[0]=-lower; diagy[-1]=-upper
    lowerx=np.ones(xx-1)
    if kind=='u':
        lowerx[-1]=2.
    lx=sp.diags((lowerx,diagx,np.ones(xx-1)),(-1,0,1),format='csr')/dx**2
    ly=sp.diags((np.ones(yy-1),diagy,np.ones(yy-1)),(-1,0,1),format='csr')/dy**2
    lap=sp.kron(sp.eye(yy),lx)+sp.kron(ly,sp.eye(xx))
    rates=4*np.sin(sy[:,None])**2/dy**2+4*np.sin(sx[None,:])**2/dx**2
    if kind=='pressure':
        matrix=(lap/rho).tocsc()
        divisor=-rates/rho
    else:
        matrix=(sp.eye(xx*yy)-.5*dt*nu*lap).tocsc()
        divisor=1+.5*dt*nu*rates
    workers = int(os.environ.get('FFT_WORKERS', 4))
    def solve(rhs):
        from scipy.fft import set_workers
        with set_workers(workers):
            transformed=forward(forward(rhs,type=tx,axis=1),type=ty,axis=0,overwrite_x=True)
            transformed/=divisor
            return backward(backward(transformed,type=ty,axis=0,overwrite_x=True),type=tx,axis=1,overwrite_x=True)
    probe=np.random.default_rng(17).normal(size=(yy,xx))
    error=np.max(abs(matrix @ solve(probe).ravel()-probe.ravel()))
    if error>1e-10:
        raise RuntimeError(f'Rectangular transform mismatch: {kind}, {error:g}')
    return matrix,solve,(yy,xx)
