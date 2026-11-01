"""Manufactured outlet traction with known polynomial derivatives and pressure gauge."""
import numpy as np
from validation.outlet_traction import outlet_traction


def test_outlet_traction():
    nx, ny, dx, mu = 12, 5, .1, .03
    x = np.arange(nx+1)*dx
    xc = (np.arange(nx)+.5)*dx
    u = np.broadcast_to(2*x+x*x, (ny, nx+1)).copy()
    v = np.broadcast_to(3*xc+xc*xc, (ny+1, nx)).copy()
    # Boundary pressure exactly equals mu*du/dx, while tangential traction is nonzero.
    p = np.broadcast_to(mu*(2+2*xc), (ny, nx)).copy()
    result = outlet_traction(u, v, p, dx, mu)
    assert result['normal']['linf'] < 1e-14
    np.testing.assert_allclose(result['tangent']['linf'], mu*(3+2*nx*dx), atol=1e-14)
    p += .4
    shifted = outlet_traction(u, v, p, dx, mu)
    np.testing.assert_allclose(shifted['normal']['linf'], .4, atol=1e-14)


if __name__ == '__main__':
    test_outlet_traction()
    print('Outlet traction analytical check PASS')
