"""Independent polynomial streamfunction: exact open-face integrals and wall derivative."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'solver'))
import numpy as np
from cut_cell_geom import CutCellGeometry2D
from wall_reconstruction import build_wall_gradient


def manufactured_faces(geom):
    x, y = np.meshgrid(np.arange(geom.nx+1)*geom.dx-geom.xc,
                       np.arange(geom.ny+1)*geom.dy-geom.yc)
    s = x*x+y*y-geom.R**2
    # psi=s**2*y in fluid; psi=0 in the stationary solid. It gives u=v=0
    # on the circle and tangential wall derivative -8*R**3*sin(theta).
    psi = np.where(s >= 0, s*s*y, 0.)
    u, v = np.zeros_like(geom.ax), np.zeros_like(geom.ay)
    np.divide(np.diff(psi, axis=0), geom.ax*geom.dy, out=u, where=geom.ax > 0)
    np.divide(-np.diff(psi, axis=1), geom.ay*geom.dx, out=v, where=geom.ay > 0)
    return u, v


def couette_faces(geom, omega=2.5):
    """Exact integrals, including partially open faces, from Couette streamfunction."""
    x, y = np.meshgrid(np.arange(geom.nx+1)*geom.dx-geom.xc,
                       np.arange(geom.ny+1)*geom.dy-geom.yc)
    radius = np.maximum(np.hypot(x, y), geom.R)
    psi = -omega*(.5*(radius**2-geom.R**2)-geom.R**2*np.log(radius/geom.R))
    u, v = np.zeros_like(geom.ax), np.zeros_like(geom.ay)
    np.divide(np.diff(psi, axis=0), geom.ax*geom.dy, out=u, where=geom.ax > 0)
    np.divide(-np.diff(psi, axis=1), geom.ay*geom.dx, out=v, where=geom.ay > 0)
    return u, v


def test_wall_averages():
    angles = .17+np.arange(8)*np.pi/4
    errors = []
    for n in (40, 80, 160):
        geom = CutCellGeometry2D(n, 3*n//4, 4., 3., 2.037, 1.527, .5)
        operator = build_wall_gradient(geom, angles)
        u, v = manufactured_faces(geom)
        actual = operator @ np.r_[u.ravel(), v.ravel()]
        error = max(abs(actual+8*geom.R**3*np.sin(angles)))
        errors.append(error)
        print(n, 'manufactured wall derivative error', error)
    assert errors[2] < errors[1] < errors[0] and errors[2] < .01


def test_couette_averages():
    angles = .17+np.arange(32)*np.pi/16
    errors = []
    for n in (40, 80, 160):
        geom = CutCellGeometry2D(n, 3*n//4, 4., 3., 2.037, 1.527, .5)
        operator = build_wall_gradient(geom, angles)
        u, v = couette_faces(geom)
        error = max(abs(operator @ np.r_[u.ravel(), v.ravel()]-5.))/5.
        errors.append(error)
        print(n, 'exact face-average Couette derivative relative error', error)
    assert errors[2] < errors[1] < errors[0] and errors[2] < .01


if __name__ == '__main__':
    test_wall_averages()
    test_couette_averages()
    print('Open-face wall reconstruction check PASS')
