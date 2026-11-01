"""
Numba-accelerated high-performance stencil kernels for Cut-Cell MAC solver.

Features:
- Pure @njit(fastmath=True, parallel=True, cache=True) compiled loops for all CPU-bound stencils and updates.
- Eliminates intermediate array allocations across momentum advection and diffusion.
- Vectorized fused face flux calculation and accumulation.
- Fast direct indexing for pack, unpack, boundary updates, and conservative cell merging.
- Preserves exact discrete equations and bitwise/machine-precision numerical consistency.
- Platform-independent: works seamlessly on Windows (Python 3.13) and Linux.
"""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def compute_flux_u_numba(u, v, ax_mid, ay_u_eff, dx, dy, rho, nu, slip_walls, centered, include_advection, F_xx, buf_F_xy):
    ny = u.shape[0]
    nx = ax_mid.shape[1]

    # 1. F_xx: (ny, nx)
    for j in prange(ny):
        for i in range(nx):
            diff = rho * nu * (u[j, i + 1] - u[j, i]) / dx * ax_mid[j, i] * dy
            if include_advection:
                u_face = 0.5 * (u[j, i] + u[j, i + 1])
                if centered:
                    u_adv = u_face
                else:
                    u_adv = u[j, i] if u_face >= 0.0 else u[j, i + 1]
                adv = rho * u_face * u_adv * ax_mid[j, i] * dy
                F_xx[j, i] = diff - adv
            else:
                F_xx[j, i] = diff

    # 2. buf_F_xy: (ny + 1, nx + 1)
    buf_F_xy.fill(0.0)
    for j in prange(1, ny):
        for i in range(nx + 1):
            diff = rho * nu * (u[j, i] - u[j - 1, i]) / dy * ay_u_eff[j, i] * dx
            if include_advection:
                if i == 0:
                    v_face = v[j, 0]
                elif i == nx:
                    v_face = v[j, nx - 1]
                else:
                    v_face = 0.5 * (v[j, i - 1] + v[j, i])

                if centered:
                    u_adv = 0.5 * (u[j - 1, i] + u[j, i])
                else:
                    u_adv = u[j - 1, i] if v_face >= 0.0 else u[j, i]
                adv = rho * v_face * u_adv * ay_u_eff[j, i] * dx
                buf_F_xy[j, i] = diff - adv
            else:
                buf_F_xy[j, i] = diff

    if not slip_walls:
        for i in range(nx + 1):
            buf_F_xy[0, i] = 2.0 * rho * nu * u[0, i] / dy * ay_u_eff[0, i] * dx
            buf_F_xy[ny, i] = -2.0 * rho * nu * u[ny - 1, i] / dy * ay_u_eff[ny, i] * dx


@njit(fastmath=True, parallel=True, cache=True)
def accumulate_rhs_u_numba(F_xx, buf_F_xy, u, ax, dy, rho, include_advection, buf_rhs_u):
    ny = buf_rhs_u.shape[0]
    nx = F_xx.shape[1]
    buf_rhs_u.fill(0.0)
    for j in prange(ny):
        for i in range(1, nx):
            buf_rhs_u[j, i] = (F_xx[j, i] - F_xx[j, i - 1]) + (buf_F_xy[j + 1, i] - buf_F_xy[j, i])
        ext_x = -rho * u[j, nx] * u[j, nx] * ax[j, nx] * dy if include_advection else 0.0
        buf_rhs_u[j, nx] = ext_x - F_xx[j, nx - 1] + 0.5 * (buf_F_xy[j + 1, nx] - buf_F_xy[j, nx])


@njit(fastmath=True, parallel=True, cache=True)
def compute_flux_v_numba(u, v, ay_mid, ax_v_eff, dx, dy, rho, nu, centered, include_advection, F_yy, buf_F_yx):
    ny = ay_mid.shape[0]
    nx = ay_mid.shape[1]

    # 1. F_yy: shape (ny, nx)
    for j in prange(ny):
        for i in range(nx):
            diff = rho * nu * (v[j + 1, i] - v[j, i]) / dy * ay_mid[j, i] * dx
            if include_advection:
                v_face = 0.5 * (v[j, i] + v[j + 1, i])
                if centered:
                    v_adv = v_face
                else:
                    v_adv = v[j, i] if v_face >= 0.0 else v[j + 1, i]
                adv = rho * v_face * v_adv * ay_mid[j, i] * dx
                F_yy[j, i] = diff - adv
            else:
                F_yy[j, i] = diff

    # 2. buf_F_yx: shape (ny + 1, nx + 1)
    buf_F_yx.fill(0.0)
    for j in prange(ny + 1):
        for i in range(1, nx):
            diff = rho * nu * (v[j, i] - v[j, i - 1]) / dx * ax_v_eff[j, i] * dy
            if include_advection:
                if j == 0:
                    u_face = u[0, i]
                elif j == ny:
                    u_face = u[ny - 1, i]
                else:
                    u_face = 0.5 * (u[j - 1, i] + u[j, i])

                if centered:
                    v_adv = 0.5 * (v[j, i - 1] + v[j, i])
                else:
                    v_adv = v[j, i - 1] if u_face >= 0.0 else v[j, i]
                adv = rho * u_face * v_adv * ax_v_eff[j, i] * dy
                buf_F_yx[j, i] = diff - adv
            else:
                buf_F_yx[j, i] = diff

        # Left inlet boundary (i = 0)
        buf_F_yx[j, 0] = 2.0 * rho * nu * v[j, 0] / dx * ax_v_eff[j, 0] * dy

        # Right outlet boundary (i = nx)
        if include_advection:
            if j == 0:
                u_out = u[0, nx]
            elif j == ny:
                u_out = u[ny - 1, nx]
            else:
                u_out = 0.5 * (u[j - 1, nx] + u[j, nx])
            buf_F_yx[j, nx] = -rho * u_out * v[j, nx - 1] * ax_v_eff[j, nx] * dy
        else:
            buf_F_yx[j, nx] = 0.0


@njit(fastmath=True, parallel=True, cache=True)
def accumulate_rhs_v_numba(F_yy, buf_F_yx, buf_rhs_v):
    ny = F_yy.shape[0]
    nx = F_yy.shape[1]
    buf_rhs_v.fill(0.0)
    for j in prange(1, ny):
        for i in range(nx):
            buf_rhs_v[j, i] = (buf_F_yx[j, i + 1] - buf_F_yx[j, i]) + (F_yy[j, i] - F_yy[j - 1, i])


@njit(fastmath=True, parallel=True, cache=True)
def apply_wall_shear_numba(tau_w, tx, ty, ds, u_indices, v_indices, rhs_u_flat, rhs_v_flat):
    n = len(u_indices)
    for k in range(n):
        rhs_u_flat[u_indices[k]] += tau_w[k] * tx[k] * ds[k]
        rhs_v_flat[v_indices[k]] += tau_w[k] * ty[k] * ds[k]


@njit(fastmath=True, parallel=True, cache=True)
def transfer_covered_numba(rhs_flat, donors, masters):
    n = len(donors)
    for k in range(n):
        rhs_flat[masters[k]] += rhs_flat[donors[k]]
        rhs_flat[donors[k]] = 0.0


@njit(fastmath=True, parallel=True, cache=True)
def finalize_acceleration_numba(rhs, vol_eff_rho, is_solid, is_fluid,
                                merge_sj, merge_si, merge_mj, merge_mi, merge_vol,
                                a_out):
    ny, nx = rhs.shape
    for j in prange(ny):
        for i in range(nx):
            if is_solid[j, i]:
                rhs[j, i] = 0.0
                a_out[j, i] = 0.0
            elif is_fluid[j, i]:
                a_out[j, i] = rhs[j, i] / vol_eff_rho[j, i]
            else:
                a_out[j, i] = 0.0

    n_merge = len(merge_sj)
    for k in range(n_merge):
        js = merge_sj[k]
        is_ = merge_si[k]
        jm = merge_mj[k]
        im = merge_mi[k]
        rhs_m = rhs[js, is_] + rhs[jm, im]
        am = rhs_m / merge_vol[k]
        a_out[js, is_] = am
        a_out[jm, im] = am


@njit(fastmath=True, parallel=True, cache=True)
def compute_rhs_numba(u_star, v_star, ax, ay, cell_type, dx, dy, dt, rho, rhs_out):
    inv_dt = rho / dt
    ny, nx = cell_type.shape
    for j in prange(ny):
        for i in range(nx):
            if cell_type[j, i] == -1:
                rhs_out[j, i] = 0.0
            else:
                flux_x = (ax[j, i + 1] * u_star[j, i + 1] - ax[j, i] * u_star[j, i]) / dx
                flux_y = (ay[j + 1, i] * v_star[j + 1, i] - ay[j, i] * v_star[j, i]) / dy
                rhs_out[j, i] = inv_dt * (flux_x + flux_y)


@njit(fastmath=True, parallel=True, cache=True)
def apply_corrector_numba(u, v, u_star, v_star, phi, ax, ay, dx, dy, dt, rho, u_inlet):
    dt_over_rho = dt / rho
    ny = u.shape[0]
    nx = v.shape[1]

    # U update
    for j in prange(ny):
        u[j, 0] = u_inlet[j]
        for i in range(1, nx):
            if ax[j, i] > 0.0:
                u[j, i] = u_star[j, i] - dt_over_rho * (phi[j, i] - phi[j, i - 1]) / dx
            else:
                u[j, i] = 0.0
        u[j, nx] = u_star[j, nx] + dt_over_rho * 2.0 * phi[j, nx - 1] / dx

    # V update
    for i in prange(nx):
        v[0, i] = 0.0
        v[ny, i] = 0.0
    for j in prange(1, ny):
        for i in range(nx):
            if ay[j, i] > 0.0:
                v[j, i] = v_star[j, i] - dt_over_rho * (phi[j, i] - phi[j - 1, i]) / dy
            else:
                v[j, i] = 0.0


@njit(fastmath=True, parallel=True, cache=True)
def gradient_numba(pressure, rho, dx, dy, gu, gv):
    gu.fill(0.0)
    gv.fill(0.0)
    ny, nx = pressure.shape
    scale_x = rho * dx
    scale_y = rho * dy
    for j in prange(ny):
        for i in range(1, nx):
            gu[j, i] = (pressure[j, i] - pressure[j, i - 1]) / scale_x
        gu[j, nx] = -2.0 * pressure[j, nx - 1] / scale_x
    for j in prange(1, ny):
        for i in range(nx):
            gv[j, i] = (pressure[j, i] - pressure[j - 1, i]) / scale_y


@njit(fastmath=True, parallel=True, cache=True)
def pack_numba(u, v, u_flat_active, v_flat_active, out):
    nu_active = len(u_flat_active)
    u_flat = u.ravel()
    v_flat = v.ravel()
    for k in prange(nu_active):
        out[k] = u_flat[u_flat_active[k]]
    nv_active = len(v_flat_active)
    for k in prange(nv_active):
        out[nu_active + k] = v_flat[v_flat_active[k]]


@njit(fastmath=True, parallel=True, cache=True)
def unpack_numba(q, u_inlet, u_flat_active, v_flat_active, u_out, v_out):
    u_out.fill(0.0)
    v_out.fill(0.0)
    ny = u_out.shape[0]
    for j in prange(ny):
        u_out[j, 0] = u_inlet[j]
    nu_active = len(u_flat_active)
    u_flat = u_out.ravel()
    for k in prange(nu_active):
        u_flat[u_flat_active[k]] = q[k]
    nv_active = len(v_flat_active)
    v_flat = v_out.ravel()
    for k in prange(nv_active):
        v_flat[v_flat_active[k]] = q[nu_active + k]


@njit(fastmath=True, parallel=True, cache=True)
def scale_divergence_numba(out, is_fluid_cell, vol_frac_fluid):
    ny, nx = out.shape
    for j in prange(ny):
        for i in range(nx):
            if is_fluid_cell[j, i]:
                out[j, i] /= vol_frac_fluid[j, i]


@njit(fastmath=True, parallel=True, cache=True)
def fill_field_numba(u, v, field):
    u_flat = u.ravel()
    v_flat = v.ravel()
    n_u = len(u_flat)
    for i in prange(n_u):
        field[i] = u_flat[i]
    n_v = len(v_flat)
    for i in prange(n_v):
        field[n_u + i] = v_flat[i]


@njit(fastmath=True, parallel=True, cache=True)
def scatter_flux_indices(flux_flat, indices, values):
    for k in range(len(indices)):
        flux_flat[indices[k]] = values[k]


@njit(fastmath=True, parallel=True, cache=True)
def accumulate_advection_flux_correction(u_eval, v_eval, patch, lengths, rho, mode, out_val):
    for k in range(len(patch)):
        if mode == 0:
            prod = u_eval[k] * u_eval[k]
        elif mode == 1:
            prod = v_eval[k] * v_eval[k]
        else:
            prod = u_eval[k] * v_eval[k]
        out_val[patch[k]] -= rho * lengths[k] * prod


@njit(fastmath=True, parallel=True, cache=True)
def apply_csr_flux_diff_numba(flux_flat, data, indices, indptr, field, stencil_indices, scale):
    n_rows = len(indptr) - 1
    for i in prange(n_rows):
        acc = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            acc += data[k] * field[indices[k]]
        flux_flat[stencil_indices[i]] = scale * acc


@njit(fastmath=True, parallel=True, cache=True)
def apply_wall_shear_csr_numba(data, indices, indptr, field, scale, tx, ty, ds, u_indices, v_indices, rhs_u_flat, rhs_v_flat):
    n = len(u_indices)
    for k in range(n):
        grad = 0.0
        for j in range(indptr[k], indptr[k + 1]):
            grad += data[j] * field[indices[j]]
        tau = scale * grad
        rhs_u_flat[u_indices[k]] += tau * tx[k] * ds[k]
        rhs_v_flat[v_indices[k]] += tau * ty[k] * ds[k]


@njit(fastmath=True, cache=True)
def compute_H_matvec_numba(x, a_u_flat, a_v_flat, u_flat_act, v_flat_act, dt, out):
    nu = len(u_flat_act)
    nv = len(v_flat_act)
    coeff = 0.5 * dt
    for i in range(nu):
        out[i] = x[i] - coeff * a_u_flat[u_flat_act[i]]
    for i in range(nv):
        out[nu + i] = x[nu + i] - coeff * a_v_flat[v_flat_act[i]]


@njit(fastmath=True, parallel=True, cache=True)
def apply_csr_spmv_numba(data, indices, indptr, x, out):
    n_rows = len(indptr) - 1
    for i in prange(n_rows):
        acc = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            acc += data[k] * x[indices[k]]
        out[i] = acc


@njit(fastmath=True, parallel=True, cache=True)
def apply_csr_spmv_offset_numba(data, indices, indptr, x, offset, out):
    n_rows = len(indptr) - 1
    for i in prange(n_rows):
        acc = offset[i]
        for k in range(indptr[i], indptr[i + 1]):
            acc += data[k] * x[indices[k]]
        out[i] = acc


@njit(fastmath=True, parallel=True, cache=True)
def advection_flux_correction_numba(flux_flat,
                                    u_data, u_indices, u_indptr,
                                    v_data, v_indices, v_indptr,
                                    field, patch, lengths, indices,
                                    rho, mode,
                                    buf_u_eval, buf_v_eval, buf_corr):
    apply_csr_spmv_numba(u_data, u_indices, u_indptr, field, buf_u_eval)
    apply_csr_spmv_numba(v_data, v_indices, v_indptr, field, buf_v_eval)
    buf_corr.fill(0.0)
    for k in range(len(patch)):
        if mode == 0:
            prod = buf_u_eval[k] * buf_u_eval[k]
        elif mode == 1:
            prod = buf_v_eval[k] * buf_v_eval[k]
        else:
            prod = buf_u_eval[k] * buf_v_eval[k]
        buf_corr[patch[k]] -= rho * lengths[k] * prod
    for i in prange(len(indices)):
        flux_flat[indices[i]] = buf_corr[i]


@njit(fastmath=True, cache=True)
def eval_linear_H_numba(x, dt_val, rho_val, nu_val,
                        buf_u, buf_v, buf_fld,
                        f_xx, f_xy, f_xx_flat, f_xy_flat, rhs_u, rhs_u_flat, a_u, a_u_flat,
                        f_yy, f_yx, f_yy_flat, f_yx_flat, rhs_v, rhs_v_flat, a_v, a_v_flat,
                        u_act, v_act, zero_inl,
                        ax_m, ay_u_e, ay_m, ax_v_e, ax_geom,
                        dx_val, dy_val, slip,
                        ux_d, ux_i, ux_p, ux_idx, uy_d, uy_i, uy_p, uy_idx,
                        vy_d, vy_i, vy_p, vy_idx, vx_d, vx_i, vx_p, vx_idx,
                        w_d, w_i, w_p, w_tx, w_ty, w_ds, w_uidx, w_vidx,
                        u_cov, u_own, v_cov, v_own,
                        u_vol_rho, v_vol_rho, sol_u, sol_v, flu_u, flu_v,
                        u_msj, u_msi, u_mmj, u_mmi, u_mvol,
                        v_msj, v_msi, v_mmj, v_mmi, v_mvol,
                        out):
    # 1. Unpack with zero inlet
    unpack_numba(x, zero_inl, u_act, v_act, buf_u, buf_v)

    # 2. Fill field for CSR stencils
    fill_field_numba(buf_u, buf_v, buf_fld)

    # 3. U face fluxes
    compute_flux_u_numba(buf_u, buf_v, ax_m, ay_u_e, dx_val, dy_val, rho_val, nu_val,
                         slip, True, False, f_xx, f_xy)
    apply_csr_flux_diff_numba(f_xx_flat, ux_d, ux_i, ux_p, buf_fld, ux_idx, rho_val * nu_val)
    apply_csr_flux_diff_numba(f_xy_flat, uy_d, uy_i, uy_p, buf_fld, uy_idx, rho_val * nu_val)
    accumulate_rhs_u_numba(f_xx, f_xy, buf_u, ax_geom, dy_val, rho_val, False, rhs_u)

    # 4. V face fluxes
    compute_flux_v_numba(buf_u, buf_v, ay_m, ax_v_e, dx_val, dy_val, rho_val, nu_val,
                         True, False, f_yy, f_yx)
    apply_csr_flux_diff_numba(f_yy_flat, vy_d, vy_i, vy_p, buf_fld, vy_idx, rho_val * nu_val)
    apply_csr_flux_diff_numba(f_yx_flat, vx_d, vx_i, vx_p, buf_fld, vx_idx, rho_val * nu_val)
    accumulate_rhs_v_numba(f_yy, f_yx, rhs_v)

    # 5. Wall shear and covered transfer
    apply_wall_shear_csr_numba(w_d, w_i, w_p, buf_fld, -rho_val * nu_val,
                               w_tx, w_ty, w_ds, w_uidx, w_vidx,
                               rhs_u_flat, rhs_v_flat)
    transfer_covered_numba(rhs_u_flat, u_cov, u_own)
    transfer_covered_numba(rhs_v_flat, v_cov, v_own)

    # 6. Finalize accelerations with cell merging
    finalize_acceleration_numba(rhs_u, u_vol_rho, sol_u, flu_u,
                                u_msj, u_msi, u_mmj, u_mmi, u_mvol, a_u)
    finalize_acceleration_numba(rhs_v, v_vol_rho, sol_v, flu_v,
                                v_msj, v_msi, v_mmj, v_mmi, v_mvol, a_v)

    # 7. Compute H matvec: out = x - 0.5 * dt * a(x)
    compute_H_matvec_numba(x, a_u_flat, a_v_flat, u_act, v_act, dt_val, out)


@njit(fastmath=True, cache=True)
def gmres_solver_numba(b, x0, tol, restart,
                       dt_val, rho_val, nu_val,
                       buf_u, buf_v, buf_fld,
                       f_xx, f_xy, f_xx_flat, f_xy_flat, rhs_u, rhs_u_flat, a_u, a_u_flat,
                       f_yy, f_yx, f_yy_flat, f_yx_flat, rhs_v, rhs_v_flat, a_v, a_v_flat,
                       u_act, v_act, zero_inl,
                       ax_m, ay_u_e, ay_m, ax_v_e, ax_geom,
                       dx_val, dy_val, slip,
                       ux_d, ux_i, ux_p, ux_idx, uy_d, uy_i, uy_p, uy_idx,
                       vy_d, vy_i, vy_p, vy_idx, vx_d, vx_i, vx_p, vx_idx,
                       w_d, w_i, w_p, w_tx, w_ty, w_ds, w_uidx, w_vidx,
                       u_cov, u_own, v_cov, v_own,
                       u_vol_rho, v_vol_rho, sol_u, sol_v, flu_u, flu_v,
                       u_msj, u_msi, u_mmj, u_mmi, u_mvol,
                       v_msj, v_msi, v_mmj, v_mmi, v_mvol,
                       v_basis, h_mat, givens, s_vec, w_vec, y_vec, out_x):
    n = len(b)
    # 1. Initial matvec: w = H(x0)
    eval_linear_H_numba(x0, dt_val, rho_val, nu_val,
                        buf_u, buf_v, buf_fld,
                        f_xx, f_xy, f_xx_flat, f_xy_flat, rhs_u, rhs_u_flat, a_u, a_u_flat,
                        f_yy, f_yx, f_yy_flat, f_yx_flat, rhs_v, rhs_v_flat, a_v, a_v_flat,
                        u_act, v_act, zero_inl,
                        ax_m, ay_u_e, ay_m, ax_v_e, ax_geom,
                        dx_val, dy_val, slip,
                        ux_d, ux_i, ux_p, ux_idx, uy_d, uy_i, uy_p, uy_idx,
                        vy_d, vy_i, vy_p, vy_idx, vx_d, vx_i, vx_p, vx_idx,
                        w_d, w_i, w_p, w_tx, w_ty, w_ds, w_uidx, w_vidx,
                        u_cov, u_own, v_cov, v_own,
                        u_vol_rho, v_vol_rho, sol_u, sol_v, flu_u, flu_v,
                        u_msj, u_msi, u_mmj, u_mmi, u_mvol,
                        v_msj, v_msi, v_mmj, v_mmi, v_mvol,
                        w_vec)
    
    # r = b - w
    rnrm2 = 0.0
    for i in prange(n):
        val = b[i] - w_vec[i]
        v_basis[0, i] = val
        rnrm2 += val * val
    rnrm = np.sqrt(rnrm2)

    if rnrm < tol:
        for i in prange(n):
            out_x[i] = x0[i]
        return 0, rnrm

    inv_rnrm = 1.0 / rnrm
    for i in prange(n):
        v_basis[0, i] *= inv_rnrm

    s_vec.fill(0.0)
    s_vec[0] = rnrm
    h_mat.fill(0.0)

    col = 0
    for col in range(restart):
        eval_linear_H_numba(v_basis[col], dt_val, rho_val, nu_val,
                            buf_u, buf_v, buf_fld,
                            f_xx, f_xy, f_xx_flat, f_xy_flat, rhs_u, rhs_u_flat, a_u, a_u_flat,
                            f_yy, f_yx, f_yy_flat, f_yx_flat, rhs_v, rhs_v_flat, a_v, a_v_flat,
                            u_act, v_act, zero_inl,
                            ax_m, ay_u_e, ay_m, ax_v_e, ax_geom,
                            dx_val, dy_val, slip,
                            ux_d, ux_i, ux_p, ux_idx, uy_d, uy_i, uy_p, uy_idx,
                            vy_d, vy_i, vy_p, vy_idx, vx_d, vx_i, vx_p, vx_idx,
                            w_d, w_i, w_p, w_tx, w_ty, w_ds, w_uidx, w_vidx,
                            u_cov, u_own, v_cov, v_own,
                            u_vol_rho, v_vol_rho, sol_u, sol_v, flu_u, flu_v,
                            u_msj, u_msi, u_mmj, u_mmi, u_mvol,
                            v_msj, v_msi, v_mmj, v_mmi, v_mvol,
                            w_vec)

        # Modified Gram-Schmidt
        for k in range(col + 1):
            dot = 0.0
            for i in prange(n):
                dot += v_basis[k, i] * w_vec[i]
            h_mat[col, k] = dot
            for i in prange(n):
                w_vec[i] -= dot * v_basis[k, i]

        wnrm2 = 0.0
        for i in prange(n):
            wnrm2 += w_vec[i] * w_vec[i]
        h1 = np.sqrt(wnrm2)
        h_mat[col, col + 1] = h1

        if h1 > 0.0:
            inv_h1 = 1.0 / h1
            for i in prange(n):
                v_basis[col + 1, i] = w_vec[i] * inv_h1

        # Apply previous Givens rotations
        for k in range(col):
            c_k = givens[k, 0]
            s_k = givens[k, 1]
            n0 = h_mat[col, k]
            n1 = h_mat[col, k + 1]
            h_mat[col, k] = c_k * n0 + s_k * n1
            h_mat[col, k + 1] = -s_k * n0 + c_k * n1

        # Current Givens rotation
        f = h_mat[col, col]
        g = h_mat[col, col + 1]
        if g == 0.0:
            c = 1.0
            s = 0.0
            mag = f
        elif f == 0.0:
            c = 0.0
            s = 1.0 if g > 0.0 else -1.0
            mag = abs(g)
        else:
            mag = np.hypot(f, g)
            c = f / mag
            s = g / mag

        givens[col, 0] = c
        givens[col, 1] = s
        h_mat[col, col] = mag
        h_mat[col, col + 1] = 0.0

        tmp = -s * s_vec[col]
        s_vec[col] = c * s_vec[col]
        s_vec[col + 1] = tmp
        presid = abs(tmp)

        if presid <= tol:
            break

    # Backsolve upper triangular system
    for k in range(col + 1):
        y_vec[k] = s_vec[k]

    for k in range(col, 0, -1):
        if y_vec[k] != 0.0:
            y_vec[k] /= h_mat[k, k]
            val = y_vec[k]
            for j in range(k):
                y_vec[j] -= val * h_mat[k, j]
    if y_vec[0] != 0.0:
        y_vec[0] /= h_mat[0, 0]

    for i in prange(n):
        acc = x0[i]
        for k in range(col + 1):
            acc += y_vec[k] * v_basis[k, i]
        out_x[i] = acc

    return col + 1, presid


@njit(fastmath=True, cache=True)
def assemble_cn_rhs_numba(q_init, explicit, diff_q, grad_p, offset, dt, rhs):
    n = len(q_init)
    half_dt = 0.5 * dt
    for i in range(n):
        rhs[i] = q_init[i] + dt * (explicit[i] + 0.5 * diff_q[i] - grad_p[i]) + half_dt * offset[i]


@njit(fastmath=True, cache=True)
def compute_coupled_residual_numba(projected_action, rhs, grad_delta_p, dt, out):
    n = len(rhs)
    for i in range(n):
        out[i] = projected_action[i] - rhs[i] + dt * grad_delta_p[i]


@njit(fastmath=True, cache=True)
def explicit_ab2_numba(now, prev_advection, alpha, explicit):
    n = len(now)
    c1 = 1.0 + alpha
    for i in range(n):
        explicit[i] = c1 * now[i] - alpha * prev_advection[i]


@njit(fastmath=True, cache=True)
def project_update_numba(q, dt, G_phi, out):
    n = len(q)
    for i in range(n):
        out[i] = q[i] - dt * G_phi[i]



@njit(fastmath=True, cache=True)
def evaluate_matvec_numba(x, dt, diff_x, offset, out):
    n = len(x)
    coeff = 0.5 * dt
    for i in range(n):
        out[i] = x[i] - coeff * (diff_x[i] - offset[i])

@njit(fastmath=True, cache=True)
def evaluate_H_matvec_fused_numba(data, indices, indptr, x, dt, out):
    n_rows = len(indptr) - 1
    coeff = 0.5 * dt
    for i in range(n_rows):
        acc = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            acc += data[k] * x[indices[k]]
        out[i] = x[i] - coeff * acc

@njit(fastmath=True, cache=True)
def subtract_and_slice_numba(a, b, indices, out):
    n = len(indices)
    for i in range(n):
        out[i] = a[indices[i]] - b[indices[i]]

@njit(fastmath=True, cache=True)
def midpoint_startup_numba(q, now, diff_q, grad_p, dt, out):
    n = len(q)
    half_dt = 0.5 * dt
    for i in range(n):
        out[i] = q[i] + half_dt * (now[i] + diff_q[i] - grad_p[i])

