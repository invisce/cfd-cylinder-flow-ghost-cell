import numpy as np

try:
    import mac_engine
    _HAS_CPP = hasattr(mac_engine, 'calculate_mac_forces_cpp')
except ImportError:
    _HAS_CPP = False

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

@njit(fastmath=True)
def _calculate_mac_forces_numba(u, v, p, mask, dx, dy, mu):
    ny, nx = mask.shape
    fx_p = 0.0
    fy_p = 0.0
    fx_v = 0.0
    fy_v = 0.0
    
    for j in range(ny):
        for i in range(nx):
            if mask[j, i]:
                # 1. West face of solid: adjacent fluid is at (j, i-1)
                if i > 0 and not mask[j, i - 1]:
                    fx_p += p[j, i - 1] * dy
                    v_fluid = 0.5 * (v[j, i - 1] + v[j + 1, i - 1])
                    fy_v += mu * (v_fluid / (0.5 * dx)) * dy
                    
                # 2. East face of solid: adjacent fluid is at (j, i+1)
                if i < nx - 1 and not mask[j, i + 1]:
                    fx_p -= p[j, i + 1] * dy
                    v_fluid = 0.5 * (v[j, i + 1] + v[j + 1, i + 1])
                    fy_v += mu * (v_fluid / (0.5 * dx)) * dy
                    
                # 3. South face of solid: adjacent fluid is at (j-1, i)
                if j > 0 and not mask[j - 1, i]:
                    fy_p += p[j - 1, i] * dx
                    u_fluid = 0.5 * (u[j - 1, i] + u[j - 1, i + 1])
                    fx_v += mu * (u_fluid / (0.5 * dy)) * dx
                    
                # 4. North face of solid: adjacent fluid is at (j+1, i)
                if j < ny - 1 and not mask[j + 1, i]:
                    fy_p -= p[j + 1, i] * dx
                    u_fluid = 0.5 * (u[j + 1, i] + u[j + 1, i + 1])
                    fx_v += mu * (u_fluid / (0.5 * dy)) * dx
                    
    return fx_p, fy_p, fx_v, fy_v

def calculate_mac_forces(u, v, p, mask, dx, dy, mu):
    """
    Compute pressure and viscous forces on the staircase cylinder.

    Use the C++ kernel when available, otherwise use the Numba version above.
    """
    if _HAS_CPP:
        return mac_engine.calculate_mac_forces_cpp(u, v, p, mask, dx, dy, mu)
    return _calculate_mac_forces_numba(u, v, p, mask, dx, dy, mu)
