"""Experimental, conservative high-order correction of compatible transport.

Each component uses the production ownership/merge partition. The compact
correction is K = (C-C.T)/2, C = -B diag(Q) (S-S0). Here B is a shared-face
incidence and S/S0 reproduce constants. Consequently K is skew, K*1=0 and
1.T*K=0 for the local correction, including in an open channel.

This proves the correction's discrete energy/momentum properties, not spatial
accuracy or DFG acceptance. The baseline diffusion and projection are untouched.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from compatible_transport import CompatibleAdvection
from wall_reconstruction import _segments


def _face_moments(geom, component, j, i, x0=0., y0=0., scale=1.):
    if component == 'u':
        fixed = (i*geom.dx-x0)/scale
        spans = _segments(j*geom.dy, (j+1)*geom.dy, i*geom.dx, geom.xc, geom.yc, geom.R)
        other = y0
    else:
        fixed = (j*geom.dy-y0)/scale
        spans = _segments(i*geom.dx, (i+1)*geom.dx, j*geom.dy, geom.yc, geom.xc, geom.R)
        other = x0
    length = sum(b-a for a, b in spans)
    if length <= 0:
        raise ValueError('Moment requested for a covered velocity face')
    first = sum((b-a)*((a+b)/2-other)/scale for a, b in spans)/length
    second = sum((b-a)*(((a+b)/2-other)**2+(b-a)**2/12)/scale**2 for a, b in spans)/length
    x, y, x2, y2 = (fixed, first, fixed**2, second) if component == 'u' else (first, fixed, second, fixed**2)
    return np.array([1., x, y, x2, x*y, y2])


class _ComponentCorrection:
    def __init__(self, boundary, component):
        self.boundary, self.component = boundary, component
        geom = boundary.geom
        aperture = geom.ax if component == 'u' else geom.ay
        volume = getattr(boundary, component+'_vol')
        self.active = np.flatnonzero(aperture > 0)
        root = np.arange(aperture.size)
        for source, master in getattr(geom, component+'_merge_list'):
            root[np.ravel_multi_index(source, aperture.shape)] = np.ravel_multi_index(master, aperture.shape)
        labels, group = np.unique(root[self.active], return_inverse=True)
        self.group = group
        self.masses = np.bincount(group, weights=volume.ravel()[self.active], minlength=len(labels))
        raw_labels = np.full(aperture.size, -1)
        raw_labels[self.active] = group
        raw_labels[getattr(boundary, component+'_covered')] = raw_labels[getattr(boundary, component+'_owner')]
        self.labels = raw_labels.reshape(aperture.shape)
        moments = np.array([_face_moments(geom, component, *np.unravel_index(int(k), aperture.shape)) for k in self.active])
        weighted = moments*volume.ravel()[self.active, None]
        group_moments = np.stack([np.bincount(group, weights=weighted[:, k], minlength=len(labels)) for k in range(6)], axis=1)/self.masses[:, None]
        tree = cKDTree(group_moments[:, 1:3])
        face_info, rows, cols, values, b_rows, b_cols, b_vals = [], [], [], [], [], [], []
        conditions, reproduction = [], []
        h = max(geom.dx, geom.dy)
        for axis in ('x', 'y'):
            face_aperture = {('u', 'x'): geom.ax_mid, ('u', 'y'): geom.ay_u,
                             ('v', 'x'): geom.ax_v, ('v', 'y'): geom.ay_mid}[component, axis]
            centered = (component, axis) in (('u', 'x'), ('v', 'y'))
            for j, i in np.argwhere(face_aperture > 0):
                x, y = (i+.5*centered)*geom.dx, (j+.5*centered)*geom.dy
                if np.hypot(x-geom.xc, y-geom.yc) >= geom.R+4*h:
                    continue
                low = (j, i) if centered else (j-1, i) if axis == 'y' else (j, i-1)
                high = (j, i+1) if centered and axis == 'x' else (j+1, i) if centered else (j, i)
                left, right = int(self.labels[low]), int(self.labels[high])
                if min(left, right) < 0:
                    raise ValueError('An open momentum face has no fluid ownership')
                if left == right:
                    continue
                if axis == 'x':
                    spans = _segments(j*geom.dy if centered else (j-.5)*geom.dy,
                                      (j+1)*geom.dy if centered else (j+.5)*geom.dy,
                                      x, geom.xc, geom.yc, geom.R)
                else:
                    spans = _segments(i*geom.dx if centered else (i-.5)*geom.dx,
                                      (i+1)*geom.dx if centered else (i+.5)*geom.dx,
                                      y, geom.yc, geom.xc, geom.R)
                length = sum(b-a for a, b in spans)
                first = sum((b-a)*((a+b)/2-(y if axis == 'x' else x))/h for a, b in spans)/length
                second = sum((b-a)*(((a+b)/2-(y if axis == 'x' else x))**2+(b-a)**2/12)/h**2 for a, b in spans)/length
                target = np.array([1., 0., first, 0., 0., second]) if axis == 'x' else np.array([1., first, 0., second, 0., 0.])
                for width in (3., 4., 5.):
                    neighbours = np.array(sorted(tree.query_ball_point([x, y], width*h)), dtype=int)
                    m = group_moments[neighbours]
                    design = np.column_stack([np.ones(len(m)), (m[:, 1]-x)/h, (m[:, 2]-y)/h,
                        (m[:, 3]-2*x*m[:, 1]+x*x)/h**2,
                        (m[:, 4]-x*m[:, 2]-y*m[:, 1]+x*y)/h**2,
                        (m[:, 5]-2*y*m[:, 2]+y*y)/h**2])
                    weights = 1/(1+np.sum(design[:, 1:3]**2, axis=1))
                    a = weights[:, None]*design
                    if len(m) >= 6 and np.linalg.matrix_rank(a, tol=1e-11) == 6:
                        coefficients = (target @ np.linalg.pinv(a, rcond=1e-12))*weights
                        coefficients[np.argmax(abs(coefficients))] += 1-coefficients.sum()
                        conditions.append(float(np.linalg.cond(a)))
                        reproduction.append(float(np.max(abs(coefficients @ design-target))))
                        break
                else:
                    raise ValueError('Insufficient scalar moments for mimetic reconstruction')
                k = len(face_info)
                face_info.append(((component, axis), np.ravel_multi_index((j, i), face_aperture.shape)))
                rows.extend([k]*len(neighbours)); cols.extend(neighbours); values.extend(coefficients)
                rows.extend([k, k]); cols.extend([left, right]); values.extend([-.5, -.5])
                b_rows.extend([left, right]); b_cols.extend([k, k]); b_vals.extend([1., -1.])
        self.faces = face_info
        self.delta_S = coo_matrix((values, (rows, cols)), shape=(len(face_info), len(labels))).tocsr()
        self.B = coo_matrix((b_vals, (b_rows, b_cols)), shape=(len(labels), len(face_info))).tocsr()
        self.fit_diagnostics = dict(faces=len(face_info), condition_max=max(conditions),
                                   polynomial_reproduction_linf=max(reproduction),
                                   constant_error=float(np.max(abs(self.delta_S @ np.ones(len(labels))))))

    def apply(self, velocity, mass_flux, rho):
        volume = getattr(self.boundary, self.component+'_vol').ravel()[self.active]
        representative = np.bincount(self.group, weights=volume*velocity.ravel()[self.active], minlength=len(self.masses))/self.masses
        Q = np.array([mass_flux[key].ravel()[index] for key, index in self.faces])
        force = -.5*rho*(self.B @ (Q*(self.delta_S @ representative))-
                         self.delta_S.T @ (Q*(self.B.T @ representative)))
        increment = np.zeros_like(velocity)
        increment.ravel()[self.active] = force[self.group]/(rho*self.masses[self.group])
        return increment, dict(total_force=float(force.sum()),
                               energy_work=float(representative @ force),
                               **self.fit_diagnostics)


class MimeticAdvection:
    """High-order skew correction of the shared-potential cluster scheme."""
    def __init__(self, boundary):
        self.boundary = boundary
        self.central = CompatibleAdvection(boundary, method='shared_potential', quantity='cluster_centered')
        self.corrections = {c: _ComponentCorrection(boundary, c) for c in ('u', 'v')}
        self.last_diagnostics = None

    def rhs(self, u, v, rho=1.):
        values = list(self.central.rhs(u, v, rho))
        flux = self.central.potential.apply(u, v)
        diagnostics = {}
        for k, (c, velocity) in enumerate((('u', u), ('v', v))):
            increment, diagnostics[c] = self.corrections[c].apply(velocity, flux, rho)
            values[k] += increment
            values[k+2] += rho*getattr(self.boundary, c+'_vol')*increment
        self.last_diagnostics = dict(transport=self.central.last_diagnostics, correction=diagnostics)
        return tuple(values)
