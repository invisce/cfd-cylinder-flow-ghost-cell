"""AB2/CN with discrete incremental pressure recovery, shared by verification."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import numba_kernels as nk


def ab2_cn_step(q, pressure, dt, *, advection, diffusion, gradient, project,
                divergence, nu, rho=1., previous_advection=None, previous_dt=None,
                helmholtz_solve=None, compiled_vectors=False, H_matvec=None, buffers=None):
    """Pressure state is at the previous half time; return half- and integer-time pressure.

    advection returns -N, diffusion includes nu and any affine boundary contribution.
    gradient includes 1/rho. project(q, dt) returns (divergence-free q, pressure increment).
    """
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(q).all()
            or not np.isfinite(pressure).all()):
        raise ValueError('Finite fields and a positive timestep are required')
    if compiled_vectors:

        if buffers is None:
            buffers = {}
        def get_buf(name):
            if name not in buffers:
                buffers[name] = np.empty_like(q)
            return buffers[name]
        if 'now_a' not in buffers:
            buffers['now_a'] = np.empty_like(q)
            buffers['now_b'] = np.empty_like(q)
        buffers['now_a'], buffers['now_b'] = buffers['now_b'], buffers['now_a']
        now = advection(q, out=buffers['now_a'])
    else:
        now = advection(q)
    if previous_advection is None:
        # Projected midpoint startup avoids a first-order initial advection estimate.
        if compiled_vectors:
            diff1 = diffusion(q, out=get_buf('diff1'))
            grad1 = gradient(pressure, out=get_buf('grad1'))
            midpoint = get_buf('midpoint')
            np.add(now, diff1, out=midpoint)
            np.subtract(midpoint, grad1, out=midpoint)
            np.multiply(midpoint, 0.5 * dt, out=midpoint)
            np.add(midpoint, q, out=midpoint)
            half, _ = project(midpoint, .5*dt, out=get_buf('half_q'))
            explicit = advection(half, out=get_buf('explicit'))
        else:
            half, _ = project(q + .5*dt*(now + diffusion(q)-gradient(pressure)), .5*dt)
            explicit = advection(half)
    else:
        if previous_dt is None or not np.isfinite(previous_dt) or previous_dt <= 0:
            raise ValueError('AB2 history requires a finite positive previous timestep')
        alpha = dt/(2*previous_dt)
        if compiled_vectors:
            explicit = get_buf('explicit')
            nk.explicit_ab2_numba(now, previous_advection, alpha, explicit)
        else:
            explicit = (1+alpha)*now-alpha*previous_advection
    if compiled_vectors:
        if hasattr(getattr(diffusion, '__self__', None), 'diff_offset'):
            offset = getattr(diffusion, '__self__', None).diff_offset
        else:
            zeros = get_buf('zeros')
            zeros.fill(0.0)
            offset = diffusion(zeros, out=get_buf('offset'))
        buf_diffusion = get_buf('buf_diffusion')
        buf_gradient = get_buf('buf_gradient')
        buf_diffusion = diffusion(q, out=buf_diffusion)
        buf_gradient = gradient(pressure, out=buf_gradient)
    else:
        offset = diffusion(np.zeros_like(q))
    
    if compiled_vectors and H_matvec is not None:
        buf_H_matvec = get_buf('buf_H_matvec')
        H = LinearOperator((q.size, q.size), dtype=float,
            matvec=lambda x: H_matvec(x, dt, out=buf_H_matvec))
    elif compiled_vectors:
        buf_H_diffusion = get_buf('buf_H_diffusion')
        buf_H_out = get_buf('buf_H_out')
        def compiled_matvec(x):
            diffusion(x, out=buf_H_diffusion)
            np.subtract(buf_H_diffusion, offset, out=buf_H_out)
            np.multiply(buf_H_out, -0.5 * dt, out=buf_H_out)
            np.add(x, buf_H_out, out=buf_H_out)
            return buf_H_out
        H = LinearOperator((q.size, q.size), dtype=float,
            matvec=compiled_matvec)
    else:
        H = LinearOperator((q.size, q.size), dtype=float,
            matvec=lambda x: x-.5*dt*(diffusion(x)-offset))
    if compiled_vectors:
        rhs = get_buf('rhs')
        nk.assemble_cn_rhs_numba(q, explicit, buf_diffusion, buf_gradient, offset, dt, rhs)
    else:
        rhs = q+dt*(explicit+.5*diffusion(q)-gradient(pressure))+.5*dt*offset
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    if helmholtz_solve is None:
        star, info = gmres(H, rhs, x0=q, rtol=1e-12, atol=1e-13,
                           restart=30, maxiter=100, callback=callback, callback_type='pr_norm')
    else:
        star, info = helmholtz_solve(rhs), 0
    residual = float(np.max(abs(H @ star-rhs)))
    if info or not np.isfinite(star).all() or residual > 1e-10*max(1., float(np.max(abs(rhs)))):
        raise RuntimeError(f'CN momentum solve failed: info={info}, residual={residual:g}')
    
    if compiled_vectors and H_matvec is not None:
        result, _ = project(star, dt, out=get_buf('result'))
        projected_action = buf_diffusion
        H_matvec(result, dt, out=projected_action)
        momentum_defect = explicit
        np.subtract(rhs, projected_action, out=momentum_defect)
        buf_proj = star
        np.add(result, momentum_defect, out=buf_proj)
        _, increment = project(buf_proj, dt, out=buf_proj)
    else:
        result, _ = project(star, dt)
        projected_action = H @ result
        momentum_defect = rhs-projected_action
        _, increment = project(result+momentum_defect, dt)
    
    if compiled_vectors:
        if 'half_pressure' not in buffers:
            buffers['half_pressure'] = np.empty_like(pressure)
        half_pressure = buffers['half_pressure']
        np.add(pressure, increment, out=half_pressure)
    else:
        half_pressure = pressure+increment
    factor = 1. if previous_advection is None else dt/(dt+previous_dt)
    if compiled_vectors:
        if 'output_pressure' not in buffers:
            buffers['output_pressure'] = np.empty_like(pressure)
        output_pressure = buffers['output_pressure']
        np.multiply(factor, increment, out=output_pressure)
        np.add(half_pressure, output_pressure, out=output_pressure)
    else:
        output_pressure = half_pressure+factor*increment
    if compiled_vectors:
        coupled_residual = explicit
        grad_delta_p = gradient(increment, out=buf_gradient)
        nk.compute_coupled_residual_numba(projected_action, rhs, grad_delta_p, dt, coupled_residual)
    else:
        coupled_residual = projected_action-rhs+dt*gradient(increment)
    return result, half_pressure, output_pressure, now, dict(
        momentum_iterations=iterations[0], momentum_residual_linf=residual,
        coupled_momentum_residual_linf=float(np.max(abs(coupled_residual))))
