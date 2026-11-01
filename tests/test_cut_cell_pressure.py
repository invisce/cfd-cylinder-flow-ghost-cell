"""Independent surface-pressure sampling and force checks, without a flow solve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'solver'))
import numpy as np
from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import (build_surface_pressure_quadrature, reconstruct_pressure,
                             surface_pressure_force, extract_cut_cell_surface_forces,
                             build_cut_cell_poisson_matrix)


def test_surface_pressure():
    for shift in (0., .37):
        errors = []
        for n in (40, 80, 160):
            h = 4/n
            g = CutCellGeometry2D(n, 3*n//4, 4., 3., 1.4+shift*h, 1.5+.23*h, .5)
            points, op, nw = build_surface_pressure_quadrature(g)
            x, y = np.meshgrid((np.arange(g.nx)+.5)*h-g.xc,
                               (np.arange(g.ny)+.5)*h-g.yc)
            xb, yb = points[:, 1]-g.xc, points[:, 0]-g.yc
            polynomial = lambda x, y: 1+2*x-3*y+.4*x*x+.2*x*y-.7*y*y
            p = polynomial(x, y)
            p[g.cell_type == -1] = np.nan  # Solid placeholder values must never leak.
            np.testing.assert_allclose(reconstruct_pressure(p, op), polynomial(xb, yb), atol=2e-12, rtol=0)
            exact = np.pi*g.R**2*np.array([-2., 3.])
            np.testing.assert_allclose(surface_pressure_force(p, op, nw), exact, atol=2e-12, rtol=0)
            np.testing.assert_allclose(surface_pressure_force(p+10000, op, nw), exact, atol=2e-9, rtol=0)
            if n == 40:
                u, v = np.zeros((g.ny, g.nx+1)), np.zeros((g.ny+1, g.nx))
                np.testing.assert_allclose(extract_cut_cell_surface_forces(u, v, p, g, .01)[2:4],
                                           exact, atol=2e-12, rtol=0)
                broken = p.copy().ravel()
                broken[op.indices[0]] = np.nan
                try:
                    reconstruct_pressure(broken, op)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Non-finite fluid pressure silently accepted')
            # A cubic field is not in the fitted basis; require measured convergence.
            p += .3*x**3-.2*y**3
            reference = polynomial(xb, yb)+.3*xb**3-.2*yb**3
            error = np.max(abs(reconstruct_pressure(p, op)-reference))
            force_exact = exact + 3*np.pi*g.R**4/4*np.array([-.3, .2])
            force_error = np.linalg.norm(surface_pressure_force(p, op, nw)-force_exact)
            errors.append(error)
            print(f'shift={shift} n={n}: cubic pressure Linf={error:.6g}, force error={force_error:.6g}')
        assert errors[1] < errors[0]/3 and errors[2] < errors[1]/3


def test_manufactured_pressure_solve():
    """Harmonic p=x*(1+R^2/r^2) has zero normal derivative on the circular wall.

    Supply its analytical outer boundary values/fluxes to the production matrix;
    this tests the pressure solve as well as surface extraction, without NS dynamics.
    """
    from scipy.sparse.linalg import spsolve
    errors = []
    for n in (40, 80, 160):
        g = CutCellGeometry2D(n, 3*n//4, 4., 3., 1.4, 1.5, .5)
        h = g.dx
        xp, yp = (np.arange(g.nx)+.5)*h, (np.arange(g.ny)+.5)*h
        x, y = np.meshgrid(xp-g.xc, yp-g.yc)
        def exact(x, y):
            r2 = x*x+y*y
            return (x*(1+g.R**2/r2), 1+g.R**2*(y*y-x*x)/r2**2,
                    -2*g.R**2*x*y/r2**2)
        rhs = np.zeros_like(x)
        rhs[:, 0] += exact(-g.xc, yp-g.yc)[1]/h
        rhs[0, :] += exact(xp-g.xc, -g.yc)[2]/h
        rhs[-1, :] -= exact(xp-g.xc, g.Ly-g.yc)[2]/h
        rhs[:, -1] -= 2*exact(g.Lx-g.xc, yp-g.yc)[0]/h**2
        A = build_cut_cell_poisson_matrix(g)
        p = spsolve(A, rhs.ravel()).reshape(x.shape)
        points, op, nw = build_surface_pressure_quadrature(g)
        reference = exact(points[:, 1]-g.xc, points[:, 0]-g.yc)[0]
        surface_error = np.max(abs(reconstruct_pressure(p, op)-reference))
        mask = (g.cell_type != -1) & (x*x+y*y >= g.R**2)
        field_error = np.sqrt(np.average((p[mask]-exact(x[mask], y[mask])[0])**2,
                                        weights=g.volume_fraction[mask]))
        force_error = np.linalg.norm(surface_pressure_force(p, op, nw)-[-2*np.pi*g.R**2, 0.])
        residual = np.max(abs(A @ p.ravel()-rhs.ravel()))
        assert residual < 1e-8
        errors.append((field_error, surface_error, force_error))
        print(f'Poisson n={n}: field L2={field_error:.6g}, surface Linf={surface_error:.6g}, force error={force_error:.6g}')
    errors = np.array(errors)
    assert np.all(errors[1:] < errors[:-1]), 'Manufactured pressure errors must decrease'
    assert np.all(np.log2(errors[:-1, 0]/errors[1:, 0]) > 1.5)
    assert errors[-1, 2]/(2*np.pi*g.R**2) < .001


if __name__ == '__main__':
    test_surface_pressure()
    test_manufactured_pressure_solve()
    print('Surface pressure checks PASS')
