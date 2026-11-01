"""Coupled MAC pressure/body projection for one fixed circular immersed boundary.

Constraints C=[D; E] impose cell divergence and interpolated marker velocity.
Solve (C W C.T) z = C q* + prescribed-boundary terms, then q=q*-W C.T z.
W=1 except outlet faces (half control volume, W=2). No penalty parameter.
See Taira & Colonius (2007), doi:10.1016/j.jcp.2007.03.005.
"""
import numpy as np
import scipy.sparse as sp
from scipy.fft import dct, idct
from scipy.linalg import cho_factor, cho_solve


def delta3(r):
    """Roma's three-point regularized delta, in grid units (Uhlmann 2005)."""
    r = np.abs(np.asarray(r))
    return np.where(r <= .5, (1 + np.sqrt(np.maximum(0., 1-3*r*r)))/3,
                    np.where(r < 1.5, (5-3*r-np.sqrt(np.maximum(0., 1-3*(1-r)**2)))/6, 0.))


class CircularProjection:
    def __init__(self, nx, ny, dx, dy, xc, yc, diameter, marker_spacing_factor=1.5):
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        self.marker_spacing_factor = marker_spacing_factor
        self.np = nx*ny
        self.nu = nx*ny  # unknown u[:, 1:], including outlet; v[1:-1, :]
        nq = self.nu + nx*(ny-1)
        h = max(dx, dy)
        radius = diameter/2
        if min(xc-radius, yc-radius, nx*dx-xc-radius, ny*dy-yc-radius) < 2*h:
            raise ValueError('Immersed cylinder needs two cells of clearance at outer boundaries')
        # ponytail: one fixed circle only; moving bodies would require refactorization.
        self.n_markers = max(12, int(np.ceil(np.pi*diameter/(marker_spacing_factor*min(dx, dy)))))
        theta = 2*np.pi*np.arange(self.n_markers)/self.n_markers
        self.markers = np.column_stack((xc+radius*np.cos(theta), yc+radius*np.sin(theta)))
        cells = np.arange(nx*ny).reshape(ny, nx)
        rows = [cells.ravel(), cells[:, 1:].ravel(), cells[:-1].ravel(), cells[1:].ravel()]
        cols = [cells.ravel(), cells[:, :-1].ravel(), self.nu+cells[:-1].ravel(), self.nu+cells[:-1].ravel()]
        vals = [np.full(nx*ny, 1/dx), np.full(ny*(nx-1), -1/dx),
                np.full(nx*(ny-1), 1/dy), np.full(nx*(ny-1), -1/dy)]
        self.D = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.np, nq)).tocsr()
        rows, cols, vals = [], [], []
        for component in range(2):
            for k, (x, y) in enumerate(self.markers):
                # Index coordinates for u[:,1:] and v[1:-1,:].
                a, b = (x/dx-1, y/dy-.5) if component == 0 else (x/dx-.5, y/dy-1)
                for j in range(int(np.floor(b))-1, int(np.floor(b))+3):
                    for i in range(int(np.floor(a))-1, int(np.floor(a))+3):
                        weight = float(delta3(a-i)*delta3(b-j))
                        if weight > 0:
                            if not (0 <= i < nx and 0 <= j < ny-component):
                                raise ValueError('Truncated marker support')
                            rows.append(component*self.n_markers+k)
                            cols.append(component*self.nu+j*nx+i)
                            vals.append(weight)
        self.E = sp.coo_matrix((vals, (rows, cols)), shape=(2*self.n_markers, nq)).tocsr()
        self.C = sp.vstack((self.D, self.E), format='csr')
        weights = np.ones(nq)
        weights[np.arange(nx-1, self.nu, nx)] = 2.
        self.correction = (sp.diags(weights) @ self.C.T).tocsr()
        # The full rectangular pressure block is separable: Neumann in y,
        # Neumann/Dirichlet in x. Exact discrete cosine inversion avoids LU fill.
        self.eigenvalues = (4*np.sin(np.pi*np.arange(ny)/(2*ny))[:, None]**2/dy**2
                            + 4*np.sin(np.pi*(np.arange(nx)+.5)/(2*nx))[None, :]**2/dx**2)
        self.B = (self.D @ self.correction[:, self.np:]).tocsc()
        schur = (self.E @ self.correction[:, self.np:]).toarray()
        for k in range(2*self.n_markers):
            response = self.pressure_solve(self.B[:, k].toarray().reshape(ny, nx)).ravel()
            schur[:, k] -= self.B.T @ response
        self.schur = cho_factor(.5*(schur+schur.T))
        self.last_force = np.zeros(2)
        self.last_reaction = np.zeros(2)
        self.last_momentum_corr = np.zeros(2)
        self.last_slip = 0.
        # Area weights only enter the fictitious-interior momentum correction.
        yy, xx = np.mgrid[:ny, :nx]
        fraction = np.zeros((ny, nx))
        for sy in (np.arange(8)+.5)/8:
            for sx in (np.arange(8)+.5)/8:
                fraction += ((xx+sx)*dx-xc)**2+((yy+sy)*dy-yc)**2 < radius**2
        jj, ii = np.nonzero(fraction)
        self.body_j = slice(jj.min(), jj.max()+1)
        self.body_i = slice(ii.min(), ii.max()+1)
        self.inside = fraction[self.body_j, self.body_i]/64

    def pressure_solve(self, rhs):
        transformed = dct(dct(rhs, type=4, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
        return idct(idct(transformed/self.eigenvalues, type=2, axis=0, norm='ortho'),
                    type=4, axis=1, norm='ortho')

    def pack(self, u, v):
        return np.concatenate((u[:, 1:].ravel(), v[1:-1].ravel()))

    def momentum_inside(self, u, v):
        j, i = self.body_j, self.body_i
        return self.dx*self.dy*np.array([np.sum(self.inside*(u[j, i]+u[j, slice(i.start+1, i.stop+1)])*.5),
                                        np.sum(self.inside*(v[j, i]+v[slice(j.start+1, j.stop+1), i])*.5)])

    def project(self, u_star, v_star, u, v, p, dt, rho):
        old_momentum = self.momentum_inside(u, v)
        q = self.pack(u_star, v_star)
        rhs = self.C @ q
        rhs[:self.np].reshape(self.ny, self.nx)[:, 0] -= u_star[:, 0]/self.dx
        pressure_free = self.pressure_solve(rhs[:self.np].reshape(self.ny, self.nx)).ravel()
        body = cho_solve(self.schur, rhs[self.np:]-self.B.T @ pressure_free)
        pressure = pressure_free-self.pressure_solve((self.B @ body).reshape(self.ny, self.nx)).ravel()
        z = np.r_[pressure, body]
        q -= self.correction @ z
        u[:, 0] = u_star[:, 0]
        u[:, 1:] = q[:self.nu].reshape(self.ny, self.nx)
        v[0], v[-1] = 0., 0.
        v[1:-1] = q[self.nu:].reshape(self.ny-1, self.nx)
        # G=-W D.T, so physical pressure has the opposite sign to z_pressure.
        p[:] = -rho/dt*z[:self.np].reshape(self.ny, self.nx)
        self.last_reaction = rho*self.dx*self.dy/dt*z[self.np:].reshape(2, self.n_markers).sum(axis=1)
        self.last_momentum_corr = rho/dt*(self.momentum_inside(u, v)-old_momentum)
        self.last_force = self.last_reaction + self.last_momentum_corr
        self.last_slip = float(np.max(np.abs(self.E @ q)))
        return self.last_force
