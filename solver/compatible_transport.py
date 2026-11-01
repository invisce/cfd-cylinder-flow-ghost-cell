"""Experimental local compatibility repair, not enabled by the production solver.

The primary field must already satisfy continuity. A cached, aperture-weighted
graph solve changes only shared, open dual faces near the cylinder. Its outer
fluxes are fixed. The component compatibility mean is retained and reported,
never silently discarded. Pressure, viscosity and small-cell maps are untouched.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from staggered_wall import transfer_covered_rhs
from wall_reconstruction import _basis, build_streamfunction_fit
from staggered_wall import point_operator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def centered_mass_fluxes(geom, u, v):
    """Production interior advecting mass fluxes (without density)."""
    ux = .5*(u[:, :-1]+u[:, 1:])*geom.ax_mid*geom.dy
    uy = np.zeros_like(geom.ay_u)
    uy[1:-1, 1:-1] = .5*(v[1:-1, :-1]+v[1:-1, 1:])*geom.ay_u[1:-1, 1:-1]*geom.dx
    vy = .5*(v[:-1]+v[1:])*geom.ay_mid*geom.dx
    vx = np.zeros_like(geom.ax_v)
    vx[1:-1, 1:-1] = .5*(u[:-1, 1:-1]+u[1:, 1:-1])*geom.ax_v[1:-1, 1:-1]*geom.dy
    return {('u', 'x'): ux, ('u', 'y'): uy, ('v', 'x'): vx, ('v', 'y'): vy}


def net_flux(geom, flux, component):
    out = np.zeros_like(geom.ax if component == 'u' else geom.ay)
    if component == 'u':
        out[:, 1:-1] = np.diff(flux[component, 'x'], axis=1)+np.diff(flux[component, 'y'], axis=0)[:, 1:-1]
    else:
        out[1:-1] = np.diff(flux[component, 'x'], axis=1)[1:-1]+np.diff(flux[component, 'y'], axis=0)
    return out


class _LocalProjection:
    def __init__(self, geom, component):
        self.geom, self.component = geom, component
        fraction = geom.u_volume_fraction if component == 'u' else geom.v_volume_fraction
        x, y = np.meshgrid((np.arange(fraction.shape[1])+(component == 'v')*.5)*geom.dx,
                           (np.arange(fraction.shape[0])+(component == 'u')*.5)*geom.dy)
        mask = (fraction > 0) & (np.hypot(x-geom.xc, y-geom.yc) < geom.R+4*max(geom.dx, geom.dy))
        mask[[0, -1]] = False
        mask[:, [0, -1]] = False
        self.ids = np.flatnonzero(mask)
        self.volume = fraction.ravel()[self.ids]*geom.cell_area
        labels = np.full(mask.shape, -1, dtype=int)
        labels.ravel()[self.ids] = np.arange(len(self.ids))
        self.faces, rows, columns, values, weights = [], [], [], [], []
        for axis in ('x', 'y'):
            aperture = {('u', 'x'): geom.ax_mid, ('u', 'y'): geom.ay_u,
                        ('v', 'x'): geom.ax_v, ('v', 'y'): geom.ay_mid}[component, axis]
            low, high = np.full(aperture.shape, -1), np.full(aperture.shape, -1)
            if component == 'u' and axis == 'x':
                low[:], high[:] = labels[:, :-1], labels[:, 1:]
            elif component == 'v' and axis == 'y':
                low[:], high[:] = labels[:-1], labels[1:]
            elif axis == 'x':
                low[:, 1:], high[:, :-1] = labels, labels
            else:
                low[1:], high[:-1] = labels, labels
            selected = np.flatnonzero((aperture > 0) & (low >= 0) & (high >= 0))
            start = len(weights)
            edge_ids = np.arange(start, start+len(selected))
            rows.extend(low.ravel()[selected]); rows.extend(high.ravel()[selected])
            columns.extend(edge_ids); columns.extend(edge_ids)
            values.extend(np.ones(len(selected))); values.extend(-np.ones(len(selected)))
            scale = geom.dy/geom.dx if axis == 'x' else geom.dx/geom.dy
            weights.extend(aperture.ravel()[selected]*scale)
            self.faces.append(((component, axis), selected, edge_ids))
        self.weights = np.asarray(weights)
        self.B = coo_matrix((values, (rows, columns)), shape=(len(self.ids), len(weights))).tocsr()
        L = (self.B @ diags(self.weights) @ self.B.T).tocsc()
        count, groups = connected_components(L, directed=False)
        self.blocks = []
        for group in range(count):
            ids = np.flatnonzero(groups == group)
            anchor = ids[np.argmax(self.volume[ids])]
            free = ids[ids != anchor]
            factor = splu(L[free][:, free]) if len(free) else None
            self.blocks.append((ids, free, factor))

    def apply(self, flux):
        before = net_flux(self.geom, flux, self.component).ravel()[self.ids]
        potential = np.zeros(len(self.ids))
        means = []
        for ids, free, factor in self.blocks:
            mean = float(before[ids].sum()/self.volume[ids].sum())
            means.append(mean)
            if abs(mean) > 1e-10:
                raise ValueError(f'Local {self.component} patch boundary is incompatible: mean={mean:g}')
            residual = before[ids]-mean*self.volume[ids]
            if factor is not None:
                potential[free] = factor.solve(-residual[np.searchsorted(ids, free)])
        correction = self.weights*(self.B.T @ potential)
        for key, selected, edge_ids in self.faces:
            flux[key].ravel()[selected] += correction[edge_ids]
        after = net_flux(self.geom, flux, self.component).ravel()[self.ids]
        return dict(nodes=len(self.ids), edges=len(self.weights),
                    net_linf_before=float(np.max(abs(before))), net_linf_after=float(np.max(abs(after))),
                    raw_divergence_linf_before=float(np.max(abs(before)/self.volume)),
                    raw_divergence_linf_after=float(np.max(abs(after)/self.volume)),
                    compatibility_means=means, correction_linf=float(np.max(abs(correction))))


class _SharedPotential:
    """One potential per dual vertex, never a different fit per bounding face.

    Cubic midpoint interpolation is global on the interior. Physical-boundary
    midpoint values retain linear interpolation. This is an experimental spatial
    scheme, not an assertion of equivalence to the legacy central transport.
    """
    def __init__(self, geom):
        self.geom, self.patches = geom, []
        h = max(geom.dx, geom.dy)
        for c in ('u', 'v'):
            shape = (geom.ny+1, geom.nx) if c == 'u' else (geom.ny, geom.nx+1)
            x, y = np.meshgrid((np.arange(shape[1])+(c == 'u')*.5)*geom.dx,
                               (np.arange(shape[0])+(c == 'v')*.5)*geom.dy)
            radius = np.hypot(x-geom.xc, y-geom.yc)
            solid = radius <= geom.R
            ids = np.flatnonzero((radius > geom.R) & (radius < geom.R+4*h))
            xx, yy = x.ravel()[ids], y.ravel()[ids]
            angles = np.arctan2(yy-geom.yc, xx-geom.xc)
            fit = build_streamfunction_fit(geom, angles)
            basis = np.array([_basis(a, b, geom, angle, h) for a, b, angle in zip(xx, yy, angles)])
            operator = point_operator(fit, np.arange(len(ids)), basis)
            weight = np.clip((4*h-(radius.ravel()[ids]-geom.R))/(2*h), 0., 1.)
            weight = weight**2*(3-2*weight)
            self.patches.append((c, solid, ids, operator, weight))
        x, y = np.meshgrid(np.arange(geom.nx+1)*geom.dx, np.arange(geom.ny+1)*geom.dy)
        self.solid_nodes = np.hypot(x-geom.xc, y-geom.yc) < geom.R
        if not np.any(self.solid_nodes):
            raise ValueError('Shared-potential experiment needs at least one solid grid vertex')

    def apply(self, u, v):
        geom = self.geom
        bottom = -np.r_[0., np.cumsum(geom.ay[0]*v[0]*geom.dx)]
        psi = bottom[None, :]+np.vstack([np.zeros((1, geom.nx+1)), np.cumsum(geom.ax*u*geom.dy, axis=0)])
        wall = float(np.mean(psi[self.solid_nodes]))
        if np.ptp(psi[self.solid_nodes]) > 1e-9:
            raise ValueError('The primary potential is not constant in the solid')
        pu, pv = .5*(psi[:, :-1]+psi[:, 1:]), .5*(psi[:-1]+psi[1:])
        pu[:, 1:-1] = (-psi[:, :-3]+9*psi[:, 1:-2]+9*psi[:, 2:-1]-psi[:, 3:])/16
        pv[1:-1] = (-psi[:-3]+9*psi[1:-2]+9*psi[2:-1]-psi[3:])/16
        field = np.r_[u.ravel(), v.ravel()]
        potentials = {'u': pu, 'v': pv}
        for c, solid, ids, operator, weight in self.patches:
            value = potentials[c]
            value.ravel()[ids] = (1-weight)*value.ravel()[ids]+weight*(wall+operator @ field)
            value[solid] = wall
        flux = centered_mass_fluxes(geom, u, v)
        flux['u', 'x'] = np.diff(pu, axis=0)
        flux['u', 'y'][1:-1, 1:-1] = -np.diff(pu, axis=1)[1:-1]
        flux['v', 'x'][1:-1, 1:-1] = np.diff(pv, axis=0)[:, 1:-1]
        flux['v', 'y'] = -np.diff(pv, axis=1)
        return flux


class CompatibleAdvection:
    """Standalone experiment using an existing boundary manager's stencils.

    Capture the legacy callable before wrapping a manager. A new experiment must
    discard legacy AB2 history and record this module's hash explicitly. The
    object does not project or modify its input velocity fields.
    """
    def __init__(self, boundary, method='local_projection', quantity='reconstructed'):
        self.boundary, self.geom = boundary, boundary.geom
        self.baseline_rhs = boundary.compute_momentum_rhs
        geom = self.geom
        margin = min(geom.xc, geom.Lx-geom.xc, geom.yc, geom.Ly-geom.yc)-geom.R
        if margin <= 5*max(geom.dx, geom.dy):
            raise ValueError('This local experiment requires a cylinder clear of external boundaries')
        if method not in ('local_projection', 'shared_potential'):
            raise ValueError('Unknown compatible transport experiment')
        if quantity not in ('reconstructed', 'cluster_centered'):
            raise ValueError('Unknown advected-quantity experiment')
        self.quantity = quantity
        self.method = method
        self.projectors = {c: _LocalProjection(geom, c) for c in ('u', 'v')}
        self.potential = _SharedPotential(geom) if method == 'shared_potential' else None
        self.last_diagnostics = None

    def fluxes(self, u, v):
        geom, faces = self.geom, self.boundary.momentum_faces
        if u.shape != geom.ax.shape or v.shape != geom.ay.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('Finite face velocity arrays matching geometry are required')
        divergence = np.diff(geom.ax*u, axis=1)/geom.dx+np.diff(geom.ay*v, axis=0)/geom.dy
        if np.max(abs(divergence)) > 1e-10:
            raise ValueError('Compatible transport requires a primary divergence-free field')
        field = np.r_[u.ravel(), v.ravel()]
        flux, samples = centered_mass_fluxes(geom, u, v), {}
        for key, stencil in faces.items():
            normal = (stencil.u if key[1] == 'x' else stencil.v) @ field
            advected = (stencil.u if key[0] == 'u' else stencil.v) @ field
            integrate = lambda x: np.bincount(stencil.patch, weights=stencil.lengths*x,
                                            minlength=len(stencil.indices))
            normal_integral, advected_integral = integrate(normal), integrate(advected)
            mean = advected_integral/integrate(np.ones(len(advected)))
            covariance = integrate(normal*advected)-normal_integral*mean
            flux[key].ravel()[stencil.indices] = normal_integral
            samples[key] = (mean, covariance, covariance+normal_integral*mean)
        if self.potential is None:
            diagnostics = {c: projector.apply(flux) for c, projector in self.projectors.items()}
        else:
            old_flux = flux
            flux = self.potential.apply(u, v)
            diagnostics = {}
            for c, projector in self.projectors.items():
                before = net_flux(geom, old_flux, c).ravel()[projector.ids]
                after = net_flux(geom, flux, c).ravel()[projector.ids]
                diagnostics[c] = dict(net_linf_before=float(np.max(abs(before))), net_linf_after=float(np.max(abs(after))),
                                       raw_divergence_linf_before=float(np.max(abs(before)/projector.volume)),
                                       raw_divergence_linf_after=float(np.max(abs(after)/projector.volume)))
        return flux, samples, diagnostics

    def rhs(self, u, v, rho=1.):
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError('A finite positive density is required')
        flux, samples, diagnostics = self.fluxes(u, v)
        b, geom = self.boundary, self.geom
        means = {('u', 'x'): .5*(u[:, :-1]+u[:, 1:]), ('v', 'y'): .5*(v[:-1]+v[1:])}
        means['u', 'y'] = np.zeros_like(geom.ay_u)
        means['u', 'y'][1:-1] = .5*(u[:-1]+u[1:])
        means['v', 'x'] = np.zeros_like(geom.ax_v)
        means['v', 'x'][:, 1:-1] = .5*(v[:, :-1]+v[:, 1:])
        centered = None
        if self.quantity == 'cluster_centered':
            representatives = {}
            for c, q in (('u', u), ('v', v)):
                q = q.copy()
                volume = getattr(b, c+'_vol')
                for source, master in getattr(geom, c+'_merge_list'):
                    source, master = tuple(source), tuple(master)
                    common = (volume[source]*q[source]+volume[master]*q[master])/(volume[source]+volume[master])
                    q[source] = q[master] = common
                q.ravel()[getattr(b, c+'_covered')] = q.ravel()[getattr(b, c+'_owner')]
                representatives[c] = q
            qu, qv = representatives['u'], representatives['v']
            centered = {('u', 'x'): .5*(qu[:, :-1]+qu[:, 1:]),
                        ('v', 'y'): .5*(qv[:-1]+qv[1:]),
                        ('u', 'y'): np.zeros_like(geom.ay_u),
                        ('v', 'x'): np.zeros_like(geom.ax_v)}
            centered['u', 'y'][1:-1] = .5*(qu[:-1]+qu[1:])
            centered['v', 'x'][:, 1:-1] = .5*(qv[:, :-1]+qv[:, 1:])
        legacy = centered_mass_fluxes(geom, u, v)
        difference = {}
        for key, mass in flux.items():
            # Keep legacy boundary terms, high-order face covariance and viscosity.
            repaired = mass*(means[key] if centered is None else centered[key])
            old = legacy[key]*means[key]
            stencil = b.momentum_faces[key]
            mean, covariance, legacy_product = samples[key]
            if centered is None:
                repaired.ravel()[stencil.indices] = mass.ravel()[stencil.indices]*mean+covariance
            old.ravel()[stencil.indices] = legacy_product
            difference[key] = -rho*(repaired-old)
        _, _, ru, rv = self.baseline_rhs(u, v, 0., rho, transport='centered')
        out = []
        for component, rhs in (('u', ru), ('v', rv)):
            correction = net_flux(geom, difference, component)
            transfer_covered_rhs(correction, getattr(b, component+'_covered'), getattr(b, component+'_owner'))
            rhs += correction
            volume = getattr(b, component+'_vol')
            fraction = geom.u_volume_fraction if component == 'u' else geom.v_volume_fraction
            aperture = geom.ax if component == 'u' else geom.ay
            rate = np.zeros_like(rhs)
            active = (fraction >= .3) & (aperture > 0)
            rate[active] = rhs[active]/(rho*volume[active])
            for source, master in getattr(geom, component+'_merge_list'):
                source, master = tuple(source), tuple(master)
                rate[source] = rate[master] = (rhs[source]+rhs[master])/(rho*(volume[source]+volume[master]))
            rhs[aperture == 0] = 0.
            out.append((rate, rhs))
        self.last_diagnostics = diagnostics
        return out[0][0], out[1][0], out[0][1], out[1][1]
