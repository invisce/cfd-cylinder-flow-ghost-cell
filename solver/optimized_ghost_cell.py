"""Stationary-circle MAC ghost extension, coupled projection and cached AB2/CN."""
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from cut_cell_solver import build_surface_pressure_quadrature
from time_integration import ab2_cn_step
from rectangular_inverse import rectangle
from pressure_schur import boundary_row_inverse, monitored_pressure_solver
from ghost_reconstruction import exterior_pressure_extension
import numba_kernels as nk


def extension(x, y, sdf, active, *, neumann=False, interpolation='bilinear'):
    if interpolation not in ('bilinear', 'biquadratic'):
        raise ValueError('Unknown ghost probe interpolation')
    xx, yy = np.meshgrid(x, y)
    distance = sdf.distance(xx, yy)
    ids = np.flatnonzero(active)
    mapping = np.full(active.shape, -1, dtype=int)
    mapping.ravel()[ids] = np.arange(len(ids))
    rows, cols, values = [ids], [np.arange(len(ids))], [np.ones(len(ids))]
    h = max(x[1]-x[0], y[1]-y[0])
    jj, ii = np.nonzero((distance <= 1e-13) & (distance >= -2*h))
    for j, i in zip(jj, ii):
        d = max(0., -distance[j, i])
        nx, ny = sdf.normal(x[i], y[j])
        s = max(d, 1.5*h)
        for _ in range(8):
            stencils = []
            for ell in (s, 2*s):
                px, py = x[i]+(d+ell)*nx, y[j]+(d+ell)*ny
                fi, fj = (px-x[0])/(x[1]-x[0]), (py-y[0])/(y[1]-y[0])
                quadratic = interpolation == 'biquadratic'
                i0, j0 = (int(np.floor(fi+.5))-1, int(np.floor(fj+.5))-1) if quadratic else (int(np.floor(fi)), int(np.floor(fj)))
                size = 3 if quadratic else 2
                if not (0 <= i0 <= len(x)-size and 0 <= j0 <= len(y)-size):
                    raise ValueError('Ghost probe reaches an outer boundary')
                indices = mapping[j0:j0+size, i0:i0+size].ravel()
                a, b = fi-i0, fj-j0
                if quadratic:
                    wx = np.array([.5*(a-1)*(a-2), -a*(a-2), .5*a*(a-1)])
                    wy = np.array([.5*(b-1)*(b-2), -b*(b-2), .5*b*(b-1)])
                    coeff = np.outer(wy, wx).ravel()
                else:
                    coeff = np.array([(1-a)*(1-b),a*(1-b),(1-a)*b,a*b])
                stencils.append((indices, coeff))
            if all((indices >= 0).all() for indices, _ in stencils):
                break
            s *= 1.25
        else:
            raise ValueError('No exterior ghost stencil')
        weights = ((4*s*s-d*d)/(3*s*s),(d*d-s*s)/(3*s*s)) if neumann else (
            -d*(d+2*s)/(s*s),d*(d+s)/(2*s*s))
        for factor, (indices, coeff) in zip(weights, stencils):
            rows.append(np.full(len(indices),j*len(x)+i)); cols.append(indices); values.append(factor*coeff)
    return sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                         shape=(active.size,len(ids))), ids


class GhostCellIntegrator:
    def __init__(self, solver):
        self.solver = solver
        gc = solver.gc
        nx, ny, dx, dy = solver.nx, solver.ny, solver.dx, solver.dy
        active_u = gc.dist_u > 1e-13
        active_v = gc.dist_v > 1e-13
        active_u[:,0] = False
        active_v[[0,-1]] = False
        active_p = gc.dist_p > 1e-13
        interpolation = solver.ghost_velocity_interpolation
        eu, self.ui = extension(gc.xu,gc.yu,gc.sdf,active_u,interpolation=interpolation)
        ev, self.vi = extension(gc.xv,gc.yv,gc.sdf,active_v,interpolation=interpolation)
        self.reconstruction_info = dict(velocity=interpolation, pressure=solver.ghost_pressure_extension)
        if solver.ghost_pressure_extension == 'quadratic':
            self.P, self.pi, fit_info = exterior_pressure_extension(gc.xp,gc.yp,gc.sdf,active_p)
            self.reconstruction_info.update(fit_info)
        else:
            self.P, self.pi = extension(gc.xp,gc.yp,gc.sdf,active_p,neumann=True)
        self.E = sp.block_diag((eu,ev),format='csr')
        self.usize = solver.u.size
        self.qsize = len(self.ui)+len(self.vi)
        self.offset = np.zeros(self.E.shape[0])
        self.offset.reshape(-1)[:self.usize].reshape(solver.u.shape)[:,0] = solver.u_inlet
        j,i = np.unravel_index(self.pi,solver.p.shape)
        r = np.arange(len(self.pi))
        uf = j*(nx+1)+i
        vf = self.usize+j*nx+i
        self.Dfull = sp.csr_matrix((np.concatenate((-np.ones(len(r))/dx,np.ones(len(r))/dx,
            -np.ones(len(r))/dy,np.ones(len(r))/dy)),(np.tile(r,4),np.concatenate((uf,uf+1,vf,vf+nx)))),
            shape=(len(r),self.E.shape[0]))
        self.D = (self.Dfull @ self.E).tocsr()
        self.div_offset = self.Dfull @ self.offset
        rows,cols,values = [],[],[]
        ju,iu = np.unravel_index(self.ui,solver.u.shape)
        self.outlet_u = np.flatnonzero(iu==nx)
        ru = np.arange(len(self.ui)); interior = iu < nx
        rows.extend((ru[interior],ru[interior],ru[~interior]))
        cols.extend((ju[interior]*nx+iu[interior],ju[interior]*nx+iu[interior]-1,ju[~interior]*nx+nx-1))
        values.extend((np.full(interior.sum(),1/dx),np.full(interior.sum(),-1/dx),np.full((~interior).sum(),-2/dx)))
        jv,iv = np.unravel_index(self.vi,solver.v.shape)
        rv = len(ru)+np.arange(len(self.vi))
        rows.extend((rv,rv)); cols.extend((jv*nx+iv,(jv-1)*nx+iv))
        values.extend((np.full(len(rv),1/dy),np.full(len(rv),-1/dy)))
        raw_g = sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(self.qsize,solver.p.size))
        self.G = (raw_g @ self.P / solver.rho).tocsr()
        self.A = (self.D @ self.G).tocsc()
        rectangular,inverse,shape = rectangle(nx,ny,dx,dy,'pressure',rho=solver.rho)
        self.pressure_solve,self.pressure_inverse_info = monitored_pressure_solver(
            self.A,self.pi,rectangular,inverse,shape,iterative=solver.pressure_solver == 'iterative')
        self._pressure_calls = []
        rows,cols,values = [],[],[]
        def add(row,col,value):
            rows.append(row); cols.append(col); values.append(np.broadcast_to(value,row.shape))
        add(ru,self.ui,-2/dx**2-np.where((ju==0)|(ju==ny-1),3.,2.)/dy**2)
        add(ru,self.ui-1,np.where(iu==nx,2.,1.)/dx**2)
        add(ru[interior],self.ui[interior]+1,1/dx**2)
        for shift,keep in ((nx+1,ju<ny-1),(-nx-1,ju>0)):
            add(ru[keep],self.ui[keep]+shift,1/dy**2)
        add(rv,self.usize+self.vi,-np.where(iv==0,3.,np.where(iv==nx-1,1.,2.))/dx**2-2/dy**2)
        for shift,keep in ((1,iv<nx-1),(-1,iv>0),(nx,np.ones(len(rv),bool)),(-nx,np.ones(len(rv),bool))):
            add(rv[keep],self.usize+self.vi[keep]+shift,1/(dx**2 if abs(shift)==1 else dy**2))
        raw_l = sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                             shape=(self.qsize,self.E.shape[0]))
        self.L = (solver.nu*raw_l @ self.E).tocsr()
        self._raw_l = raw_l
        self.diff_offset = solver.nu*(raw_l @ self.offset)
        self.previous_advection = self.previous_dt = self.half_pressure = None
        self._dt = self._helmholtz = None
        self.u = np.zeros_like(solver.u); self.v = np.zeros_like(solver.v)
        self.field = np.zeros(self.E.shape[0])
        self.buf_div = np.zeros(len(self.pi))
        self.buf_G_phi = np.zeros(self.qsize)
        self.buf_L_q = np.zeros(self.qsize)
        self.buf_q_pack = np.zeros(self.qsize)
        self.fxx = np.zeros((ny,nx)); self.fxy = np.zeros((ny+1,nx+1))
        self.fyy = np.zeros((ny,nx)); self.fyx = np.zeros((ny+1,nx+1))
        self.ru = np.zeros_like(solver.u); self.rv = np.zeros_like(solver.v)
        self.ones = np.ones((ny,nx)); self.corners = np.ones((ny+1,nx+1))
        self.apertures = np.ones_like(solver.u)
        self.geom = SimpleNamespace(nx=nx,ny=ny,dx=dx,dy=dy,Lx=solver.Lx,Ly=solver.Ly,
            xc=solver.xc,yc=solver.yc,R=solver.D/2,cell_type=np.where(active_p,1,-1))
        points,self.pressure_surface,self.normal_weights = build_surface_pressure_quadrature(self.geom)
        self.surface_points = points
        self.normals = self.normal_weights/(np.pi*solver.D/len(points))
        self.du_dn = self._wall_gradient(gc.xu,gc.yu,active_u,points)
        self.dv_dn = self._wall_gradient(gc.xv,gc.yv,active_v,points)
        self.last_diagnostics = {}
        self.step_buffers = {}

    def _wall_gradient(self,x,y,active,points):
        xx,yy = np.meshgrid(x,y)
        h = max(self.solver.dx,self.solver.dy)
        radius = np.hypot(xx-self.geom.xc,yy-self.geom.yc)
        ids = np.flatnonzero(active & (radius<self.geom.R+6*h))
        coords = np.column_stack((yy.ravel()[ids],xx.ravel()[ids]))
        tree = cKDTree(coords)
        distances,_ = tree.query(points,k=14)
        neighbors = tree.query_ball_point(points,distances[:,-1]*(1+1e-12))
        rows,cols,values = [],[],[]
        for k,near in enumerate(neighbors):
            ys,xs = coords[near].T
            theta = np.arctan2(points[k,0]-self.geom.yc,points[k,1]-self.geom.xc)
            angle = np.arctan2(ys-self.geom.yc,xs-self.geom.xc)-theta
            n = (np.hypot(xs-self.geom.xc,ys-self.geom.yc)-self.geom.R)/h
            t = self.geom.R*np.arctan2(np.sin(angle),np.cos(angle))/h
            weights,_,rank,_ = np.linalg.lstsq(np.array([n,n*n,n*t]),[1/h,0.,0.],rcond=1e-12)
            if rank!=3:
                raise ValueError('Unresolved wall-gradient stencil')
            rows.extend([k]*len(near)); cols.extend(ids[near]); values.extend(weights)
        return sp.csr_matrix((values,(rows,cols)),shape=(len(points),active.size))

    def pack(self,u,v,out=None):
        if out is None: out = np.empty(self.qsize)
        nk.pack_numba(u,v,self.ui,self.vi,out)
        return out

    def unpack(self,q):
        nk.apply_csr_spmv_numba(self.E.data, self.E.indices, self.E.indptr, q, self.field)
        self.field += self.offset
        self.u[:] = self.field[:self.usize].reshape(self.u.shape)
        self.v[:] = self.field[self.usize:].reshape(self.v.shape)
        return self.u,self.v

    def advection(self,q,compiled=True,out=None):
        u,v = self.unpack(q)
        def call(function,*args):
            (function if compiled else function.py_func)(*args)
        s = self.solver
        call(nk.compute_flux_u_numba,u,v,self.ones,self.corners,s.dx,s.dy,1.,0.,False,True,True,self.fxx,self.fxy)
        call(nk.accumulate_rhs_u_numba,self.fxx,self.fxy,u,self.apertures,s.dy,1.,True,self.ru)
        call(nk.compute_flux_v_numba,u,v,self.ones,self.corners,s.dx,s.dy,1.,0.,True,True,self.fyy,self.fyx)
        call(nk.accumulate_rhs_v_numba,self.fyy,self.fyx,self.rv)
        if out is None: out = np.empty(self.qsize)
        result = self.pack(self.ru,self.rv,out=out)
        result /= (s.dx*s.dy)
        result[self.outlet_u] *= 2
        return result

    def diffusion(self,q,out=None):
        if out is None: out = np.empty(self.qsize)
        nk.apply_csr_spmv_numba(self.L.data, self.L.indices, self.L.indptr, q, out)
        out += self.diff_offset
        return out

    def gradient(self,p,out=None):
        if out is None: out = np.empty(self.qsize)
        nk.apply_csr_spmv_numba(self.G.data, self.G.indices, self.G.indptr, p.ravel() if p.ndim > 1 else p, out)
        return out

    def divergence(self,q,out=None):
        if out is None: out = np.empty(len(self.pi))
        nk.apply_csr_spmv_numba(self.D.data, self.D.indices, self.D.indptr, q, out)
        out += self.div_offset
        return out

    def project(self,q,dt,out=None):
        rhs = self.divergence(q, out=self.buf_div)
        rhs /= dt
        phi = self.pressure_solve(rhs)
        self._pressure_calls.append(dict(self.pressure_inverse_info['last_solve']))
        nk.apply_csr_spmv_numba(self.G.data, self.G.indices, self.G.indptr, phi, self.buf_G_phi)
        if out is None: out = np.empty_like(q)
        nk.project_update_numba(q, dt, self.buf_G_phi, out)
        error = float(np.max(np.abs(self.divergence(out, out=self.buf_div))))
        if error>1e-10 or not np.isfinite(out).all():
            raise RuntimeError(f'Ghost projection failed: divergence={error:g}')
        return out,phi

    def H_matvec(self, q, dt, out=None):
        if out is None: out = np.empty_like(q)
        nk.evaluate_H_matvec_fused_numba(self.L.data, self.L.indices, self.L.indptr, q, dt, out)
        return out

    def advance(self,u,v,p,dt):
        self._pressure_calls.clear()
        inlet = self.offset[:self.usize].reshape(self.u.shape)[:,0]
        if not np.array_equal(inlet,self.solver.u_inlet):
            inlet[:] = self.solver.u_inlet
            self.div_offset = self.Dfull @ self.offset
            self.diff_offset = self.solver.nu*(self._raw_l @ self.offset)
        if self._dt!=dt:
            s=self.solver
            nu=len(self.ui)
            solves=[]; self.helmholtz_inverse_info=[]
            for kind,selected,block in (
                ('u',np.unravel_index(self.ui,self.u.shape)[0]*s.nx+np.unravel_index(self.ui,self.u.shape)[1]-1,self.L[:nu,:nu]),
                ('v',(np.unravel_index(self.vi,self.v.shape)[0]-1)*s.nx+np.unravel_index(self.vi,self.v.shape)[1],self.L[nu:,nu:])):
                rectangular,inverse,shape=rectangle(s.nx,s.ny,s.dx,s.dy,kind,s.nu,dt,s.rho)
                actual=(sp.eye(block.shape[0],format='csc')-.5*dt*block).tocsc()
                solve,info=boundary_row_inverse(actual,selected,rectangular,inverse,shape)
                solves.append(solve); self.helmholtz_inverse_info.append(info)
            self._helmholtz=lambda rhs:np.concatenate((solves[0](rhs[:nu]),solves[1](rhs[nu:])))
            self._dt = dt
        pressure = p.ravel()[self.pi] if self.half_pressure is None else self.half_pressure
        if 'pack' not in self.step_buffers:
            self.step_buffers['pack'] = np.empty(self.qsize)
        q_pack = self.pack(u,v, out=self.step_buffers['pack'])
        q,half,output,history,diagnostics = ab2_cn_step(q_pack,pressure,dt,
            advection=self.advection,diffusion=self.diffusion,gradient=self.gradient,project=self.project,
            divergence=self.divergence,nu=self.solver.nu,rho=self.solver.rho,
            previous_advection=self.previous_advection,previous_dt=self.previous_dt,
            helmholtz_solve=self._helmholtz,compiled_vectors=True,
            H_matvec=self.H_matvec, buffers=self.step_buffers)
        self.half_pressure,self.previous_advection,self.previous_dt = half,history,dt
        self.last_diagnostics = diagnostics
        calls = self._pressure_calls
        self.last_diagnostics.update(
            pressure_backend=self.pressure_inverse_info['backend'],
            pressure_solve_count=len(calls),
            pressure_iterations=sum(c['iterations'] for c in calls),
            pressure_refinement_iterations=sum(c['refinement_iterations'] for c in calls),
            pressure_residual_linf=max(c['residual_linf'] for c in calls),
            pressure_residual_relative_l2=max(c['residual_relative_l2'] for c in calls),
            pressure_seconds=sum(c['seconds'] for c in calls), pressure_solves=list(calls))
        uu,vv = self.unpack(q)
        return uu,vv,(self.P @ output).reshape(p.shape)

    def surface_forces(self,u,v,p,mu):
        pressure = self.pressure_surface @ p.ravel()
        dn = np.column_stack((self.du_dn @ u.ravel(),self.dv_dn @ v.ravel()))
        traction = mu*(dn+self.normals*np.sum(dn*self.normals,axis=1)[:,None])
        fp = -np.sum(pressure[:,None]*self.normal_weights,axis=0)
        fv = np.sum(traction,axis=0)*(np.pi*self.solver.D/len(pressure))
        return *(fp+fv),*fp,*fv
