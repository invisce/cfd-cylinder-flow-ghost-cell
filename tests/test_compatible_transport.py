"""Physical face integrals and shared transport, without pressure factorization."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import numpy as np
from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import CutCellBoundary
from compatible_transport import CompatibleAdvection, net_flux
from staggered_wall import face_stencil, wall_stencil, owned_volumes, transfer_covered_rhs
from wall_reconstruction import _segments
from tests.test_wall_face_average_reconstruction import couette_faces


def manager(n, shift=0.):
    geom = CutCellGeometry2D(n, n, 3., 3., 1.5+shift, 1.5+shift, .5)
    return manager_from_geometry(geom)


def manager_from_geometry(geom):
    boundary = SimpleNamespace(geom=geom, nx=geom.nx, ny=geom.ny, dx=geom.dx, dy=geom.dy,
                               slip_walls=False, wall_cut_indices=[1])
    boundary.wall = wall_stencil(geom)
    boundary.momentum_faces = {(c, a): face_stencil(geom, c, a)
                               for c in ('u', 'v') for a in ('x', 'y')}
    boundary.tx, boundary.ty, boundary.ds = boundary.wall.tx, boundary.wall.ty, boundary.wall.ds
    boundary.wall_shear_rate = lambda u, v: boundary.wall.rate @ np.r_[u.ravel(), v.ravel()]
    for c in ('u', 'v'):
        volume, donors, owners = owned_volumes(geom, c)
        setattr(boundary, c+'_vol', volume)
        setattr(boundary, c+'_covered', donors)
        setattr(boundary, c+'_owner', owners)
        effective = volume.copy()
        for source, master in getattr(geom, c+'_merge_list'):
            source, master = tuple(source), tuple(master)
            effective[source] = effective[master] = volume[source]+volume[master]
        setattr(boundary, c+'_vol_eff', effective)
    boundary.compute_momentum_rhs = lambda *a, **kw: CutCellBoundary.compute_momentum_rhs(boundary, *a, **kw)
    return boundary


def exact_advection(boundary, omega=2.5):
    """Independent 16-point physical flux integration of u theta=omega(r-R^2/r)."""
    geom = boundary.geom
    nodes, weights = np.polynomial.legendre.leggauss(16)
    flux = {}
    for c, axis in boundary.momentum_faces:
        aperture = {('u', 'x'): geom.ax_mid, ('u', 'y'): geom.ay_u,
                    ('v', 'x'): geom.ax_v, ('v', 'y'): geom.ay_mid}[c, axis]
        centered = (c, axis) in (('u', 'x'), ('v', 'y'))
        value = np.zeros_like(aperture)
        for j, i in np.argwhere(aperture > 0):
            x, y = (i+.5*centered)*geom.dx, (j+.5*centered)*geom.dy
            if np.hypot(x-geom.xc, y-geom.yc) > geom.R+7*geom.dx:
                continue
            fixed = x if axis == 'x' else y
            index, spacing = (j, geom.dy) if axis == 'x' else (i, geom.dx)
            low, high = (index*spacing, (index+1)*spacing) if centered else ((index-.5)*spacing, (index+.5)*spacing)
            centers = (geom.xc, geom.yc) if axis == 'x' else (geom.yc, geom.xc)
            spans = _segments(low, high, fixed, *centers, geom.R)
            for a, b in spans:
                coordinates = (a+b)/2+(b-a)/2*nodes
                xx, yy = (np.full_like(nodes, fixed), coordinates) if axis == 'x' else (coordinates, np.full_like(nodes, fixed))
                xx, yy = xx-geom.xc, yy-geom.yc
                coefficient = omega*(1-geom.R**2/(xx*xx+yy*yy))
                u, v = -coefficient*yy, coefficient*xx
                normal, quantity = (u if axis == 'x' else v), (u if c == 'u' else v)
                value[j, i] -= (b-a)/2*np.dot(weights, normal*quantity)
        flux[c, axis] = value
    accelerations = {}
    for c in ('u', 'v'):
        rhs = net_flux(geom, flux, c)
        transfer_covered_rhs(rhs, getattr(boundary, c+'_covered'), getattr(boundary, c+'_owner'))
        volume = getattr(boundary, c+'_vol')
        acceleration = np.divide(rhs, volume, out=np.zeros_like(rhs), where=volume > 0)
        for source, master in getattr(geom, c+'_merge_list'):
            source, master = tuple(source), tuple(master)
            acceleration[source] = acceleration[master] = (rhs[source]+rhs[master])/(volume[source]+volume[master])
        accelerations[c] = acceleration
    return accelerations


def test_shared_mass_flux_and_unchanged_velocity():
    for n, shift in ((30, 0.), (60, .007), (90, 0.)):
        boundary = manager(n, shift)
        u, v = couette_faces(boundary.geom)
        saved_u, saved_v = u.copy(), v.copy()
        for method in ('local_projection', 'shared_potential'):
            op = CompatibleAdvection(boundary, method=method)
            flux, _, diagnostics = op.fluxes(u, v)
            for c in ('u', 'v'):
                assert diagnostics[c]['raw_divergence_linf_after'] < 1e-9, diagnostics
                assert diagnostics[c]['net_linf_after'] < 1e-12
                # A constant scalar advected by this flux has no local source.
                residual = net_flux(boundary.geom, flux, c).ravel()[op.projectors[c].ids]
                assert np.max(abs(3.7*residual)) < 1e-12
            values = op.rhs(u, v)
            legacy = boundary.compute_momentum_rhs(u, v, 0., 1., transport='centered')
            for k, c in enumerate(('u', 'v')):
                volume = getattr(boundary, c+'_vol')
                assert abs((volume*values[k]).sum()-values[k+2].sum()) < 1e-11
                assert abs(values[k+2].sum()-legacy[k+2].sum()) < 1e-11
        np.testing.assert_array_equal(u, saved_u)
        np.testing.assert_array_equal(v, saved_v)


def test_rejects_non_solenoidal_or_invalid_input():
    boundary = manager(30)
    op = CompatibleAdvection(boundary)
    u, v = couette_faces(boundary.geom)
    u[2, 2] += .1
    from unittest import TestCase
    with TestCase().assertRaisesRegex(ValueError, 'divergence-free'):
        op.rhs(u, v)
    with TestCase().assertRaisesRegex(ValueError, 'density'):
        op.rhs(u, v, rho=-1.)


def test_discrete_energy_conservation_and_kinematics():
    from scratch.validate_closed_energy_transport_20261005 import closed_faces, project_kinematic, compute_energy_and_rate
    for n in (40, 60):
        b = manager(n, 0.007)
        u0, v0, psi0 = closed_faces(b, return_psi=True)
        u, v = project_kinematic(b, u0, v0, psi0)

        # 1. Kinematic constraints & exact machine-zero boundaries:
        for s, m in b.geom.u_merge_list:
            np.testing.assert_allclose(u[tuple(s)], u[tuple(m)], atol=1e-13)
        for s, m in b.geom.v_merge_list:
            np.testing.assert_allclose(v[tuple(s)], v[tuple(m)], atol=1e-13)
        div = np.diff(b.geom.ax * u, axis=1) / b.geom.dx + np.diff(b.geom.ay * v, axis=0) / b.geom.dy
        assert np.max(np.abs(div[b.geom.cell_type != -1])) < 1e-14
        assert np.max(np.abs(u[:, [0, -1]])) == 0.0
        assert np.max(np.abs(v[[0, -1], :])) == 0.0
        assert np.max(np.abs(u[b.geom.ax == 0.0])) == 0.0
        assert np.max(np.abs(v[b.geom.ay == 0.0])) == 0.0

        # 2. Gate 1 & Inner Product Equivalence:
        for method in ('shared_potential', 'local_projection'):
            op = CompatibleAdvection(b, method=method)
            vals = op.rhs(u, v)
            metrics = compute_energy_and_rate(b, u, v, vals[0], vals[1], vals[2], vals[3])
            assert metrics['rate_discrepancy'] < 1e-14
            assert abs(metrics['true_rate']) < 5e-4
            # Gate 3 domain momentum balance
            assert max(abs(vals[2].sum()), abs(vals[3].sum())) < 1e-15
            assert abs(np.sum(b.u_vol * vals[0]) - vals[2].sum()) < 1e-15

        # 3. Cluster centered exact energy preservation & momentum:
        op_cc = CompatibleAdvection(b, method='shared_potential', quantity='cluster_centered')
        vals_cc = op_cc.rhs(u, v)
        metrics_cc = compute_energy_and_rate(b, u, v, vals_cc[0], vals_cc[1], vals_cc[2], vals_cc[3])
        assert abs(metrics_cc['true_rate']) < 1e-15
        assert metrics_cc['rate_discrepancy'] < 1e-15
        assert max(abs(vals_cc[2].sum()), abs(vals_cc[3].sum())) < 1e-15


if __name__ == '__main__':
    test_shared_mass_flux_and_unchanged_velocity()
    test_rejects_non_solenoidal_or_invalid_input()
    test_discrete_energy_conservation_and_kinematics()
    print('Compatible transport tests PASS')

