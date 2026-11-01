"""Analytical channel with no body: outer stresses must close momentum in both axes."""
import numpy as np
from validation.steady_momentum_budget import outer_forces


def test_poiseuille_budget():
    for nx, ny in ((30, 12), (60, 24)):
        L, H, mu, umax = 3., 1.2, .05, 1.5
        dx, dy = L/nx, H/ny
        y = (np.arange(ny)+.5)*dy
        x = (np.arange(nx)+.5)*dx
        u = np.broadcast_to((4*umax*y*(H-y)/H**2)[:, None], (ny, nx+1))
        v = np.zeros((ny+1, nx))
        p = np.broadcast_to(8*mu*umax/H**2*(L-x), (ny, nx))
        result = outer_forces(u, v, p, dx, dy, mu, 1.)
        np.testing.assert_allclose(result['force'], 0., atol=1e-12)
        np.testing.assert_allclose(result['pressure'][0], 8*mu*umax*L/H, atol=1e-12)


if __name__ == '__main__':
    test_poiseuille_budget()
    print('Independent control-volume analytical check PASS')
