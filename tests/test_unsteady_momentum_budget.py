"""Analytical accelerating-flow CV identity and temporal-error checks."""
import numpy as np
from validation.unsteady_momentum_budget import fluid_momentum, storage_corrected_force, boundary_momentum
from validation.steady_momentum_budget import outer_forces


def run():
    nx,ny,dx,dy,rho = 12,8,.1,.2,1.5
    vu = np.full((ny,nx+1),dx*dy); vu[:,[0,-1]] *= .5
    vv = np.full((ny+1,nx),dx*dy); vv[[0,-1],:] *= .5
    area = nx*dx*ny*dy
    center,h = .7,.1
    errors = []
    for half_width in (h,h/2):
        u_before = np.full(vu.shape,np.sin(center-half_width))
        u_after = np.full(vu.shape,np.sin(center+half_width))
        before = fluid_momentum(u_before,np.zeros(vv.shape),vu,vv,rho)
        after = fluid_momentum(u_after,np.zeros(vv.shape),vu,vv,rho)
        u = np.full(vu.shape,np.sin(center))
        p = np.tile(-rho*np.cos(center)*(np.arange(nx)+.5)*dx,(ny,1))
        outer = outer_forces(u,np.zeros(vv.shape),p,dx,dy,0.,rho)['force']
        assert np.allclose(outer,[rho*area*np.cos(center),0],atol=1e-12)
        force = storage_corrected_force(outer,before,after,half_width)
        errors.append(abs(force[0]))
        assert abs(force[1]) < 1e-12
    assert 1.99 < np.log2(errors[0]/errors[1]) < 2.01
    assert np.allclose(fluid_momentum(np.ones(vu.shape),np.zeros(vv.shape),vu,vv,rho),[rho*area,0])
    assert np.allclose(boundary_momentum(np.ones(vu.shape),dx,dy,rho),[rho*area,0])
    # psi=x*y*(Ly-y): v=0 at horizontal walls and exact My=-rho*Lx*Ly^3/6.
    boundary_errors = []
    for count in (8,16):
        step = ny*dy/count
        x = np.arange(nx+1)*dx
        y = (np.arange(count)+.5)*step
        u = x[None,:]*(ny*dy-2*y[:,None])
        value = boundary_momentum(u,dx,step,rho)
        assert abs(value[0]) < 1e-12
        boundary_errors.append(abs(value[1]+rho*nx*dx*(ny*dy)**3/6))
    assert np.isclose(boundary_errors[0]/boundary_errors[1],4,rtol=1e-10)
    try:
        storage_corrected_force([0,0],[0,0],[0,0],0)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero derivative interval accepted')
    print('PASS: accelerating rectangle, force sign, half-cell volume, temporal order, independent boundary-moment identity/order')


if __name__ == '__main__':
    run()
