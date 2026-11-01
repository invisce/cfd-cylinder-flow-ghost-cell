"""Boundary-row correction of a rectangular pressure inverse.

Covered cells retain a disconnected virtual Laplacian with a pinned gauge.
Only interface rows and the pin differ from the rectangular stencil, so no
domain-sized inverse or dense matrix is stored. The fluid block is unchanged.
"""
from types import SimpleNamespace
import warnings
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning
from scipy.sparse.linalg import LinearOperator, gmres
import scipy.sparse as sp


def monitored_pressure_solver(actual, indices, rectangular, inverse, shape, *, iterative=False):
    indices = np.asarray(indices)
    absolute = abs(actual).tocsr()
    roundoff_weights = (2*np.diff(absolute.indptr)+2)*np.finfo(float).eps
    if iterative:
        def precondition(values):
            rhs = np.zeros(shape)
            rhs.ravel()[indices] = values
            return inverse(rhs).ravel()[indices]
        preconditioner = LinearOperator(actual.shape, matvec=precondition, dtype=float)
        info = dict(backend='gmres_rectangle', preconditioner='separable_rectangle',
                    krylov_rtol=1e-12, krylov_atol=1e-12)
    else:
        direct, info = boundary_row_inverse(actual,indices,rectangular,inverse,shape)
        info['backend'] = 'fft_boundary_schur'
    def solve(values):
        started = time.perf_counter()
        if not np.isfinite(values).all():
            raise ValueError('Pressure RHS must be finite')
        count = 0
        def callback(_):
            nonlocal count
            count += 1
        def krylov(rhs):
            return gmres(actual, rhs, M=preconditioner, rtol=info['krylov_rtol'], atol=info['krylov_atol'],
                         restart=50, maxiter=100, callback=callback, callback_type='pr_norm')
        if iterative:
            # The independent true-residual gate below remains the acceptance criterion.
            result, status = krylov(values)
        else:
            result, status = direct(values), 0
        residual = values-actual @ result
        target = max(1e-10,64*np.finfo(float).eps*max(1.,float(np.max(abs(values)))))
        refinements = info.get('last_refinement_iterations',0)
        if iterative:
            refinements = 0
            for _ in range(2):
                if status or np.max(abs(residual)) <= target:break
                correction, status = krylov(residual)
                result += correction
                residual = values-actual @ result
                refinements += 1
        linf = float(np.max(abs(residual)))
        relative = float(np.linalg.norm(residual)/max(np.linalg.norm(values), np.finfo(float).tiny))
        action_scale = absolute @ abs(result)+abs(values)
        roundoff_bound = float(np.max(roundoff_weights*action_scale))
        # Cancellation in A@phi can exceed an RHS-only bound even after refinement.
        target = max(target,roundoff_bound)
        info['last_solve'] = dict(iterations=count, residual_linf=linf, residual_relative_l2=relative,
                                  refinement_iterations=refinements,
                                  residual_target_linf=target, roundoff_bound_linf=roundoff_bound,
                                  backward_error_linf=linf/max(float(np.max(action_scale)),np.finfo(float).tiny),
                                  status=int(status), seconds=time.perf_counter()-started)
        if status or not np.isfinite(result).all() or linf > target:
            raise RuntimeError(f'Pressure solve failed: {info["last_solve"]}')
        return result
    return solve, info


def boundary_row_inverse(actual, indices, rectangle, inverse, shape):
    """Exact boundary-row correction of a separable inverse, with true-action checks."""
    indices = np.asarray(indices)
    covered = np.ones(rectangle.shape[0],bool)
    covered[indices] = False
    coo = actual.tocoo()
    extended = sp.csr_matrix((coo.data,(indices[coo.row],indices[coo.col])),shape=rectangle.shape)
    virtual = rectangle.tocoo()
    keep = covered[virtual.row] & covered[virtual.col]
    extended += sp.csr_matrix((virtual.data[keep],(virtual.row[keep],virtual.col[keep])),shape=rectangle.shape)
    delta = (extended-rectangle).tocsr()
    tolerance = 8*np.finfo(float).eps*max(1.,float(np.max(abs(rectangle.diagonal()))))
    delta.data[abs(delta.data)<tolerance] = 0.
    delta.eliminate_zeros()
    rows = np.flatnonzero(np.diff(delta.indptr))
    if len(rows)>2048:
        raise ValueError('Ghost inverse exceeds the bounded boundary-row rank')
    correction = delta[rows]
    schur = np.eye(len(rows))
    impulse = np.zeros(shape)
    for column,row in enumerate(rows):
        impulse.ravel()[row] = 1.
        schur[:,column] += correction @ inverse(impulse).ravel()
        impulse.ravel()[row] = 0.
    with warnings.catch_warnings():
        warnings.simplefilter('error',LinAlgWarning)
        factor = lu_factor(schur)
    rhs_buffer = np.zeros(shape)
    def apply(values):
        rhs = rhs_buffer
        rhs.fill(0.)
        rhs.ravel()[indices] = values
        first = inverse(rhs)
        rhs.fill(0.)
        rhs.ravel()[rows] = lu_solve(factor,correction @ first.ravel())
        result = (first-inverse(rhs)).ravel()[indices]
        return result
    details=dict(rank=len(rows),dense_bytes=int(schur.nbytes),roundoff_drop_tolerance=tolerance)
    def solve(values):
        result = apply(values)
        target = max(1e-10,32*np.finfo(float).eps*max(1.,float(np.max(abs(values)))))
        count = 0
        for _ in range(2):
            residual = values-actual @ result
            if np.max(abs(residual))<=target:
                break
            result += apply(residual)
            count += 1
        details['last_refinement_iterations'] = count
        return result
    return solve,details


def boundary_schur_preconditioner(geom, assemble, rectangle_inverse, max_rank=2048):
    covered = geom.cell_type == -1
    ax, ay = geom.ax.copy(), geom.ay.copy()
    ax[:, 1:-1][covered[:, :-1] & covered[:, 1:]] = 1.
    ay[1:-1][covered[:-1] & covered[1:]] = 1.
    fluid_types = np.ones_like(geom.cell_type)
    def matrix(xfaces, yfaces):
        virtual = SimpleNamespace(nx=geom.nx, ny=geom.ny, dx=geom.dx, dy=geom.dy,
                                  ax=xfaces, ay=yfaces, cell_type=fluid_types)
        return -assemble(virtual)
    extended = matrix(ax, ay)
    if covered.any():
        pin = int(np.flatnonzero(covered)[0])
        extended = extended.tolil()
        extended[pin, :] = 0.
        extended[:, pin] = 0.
        extended[pin, pin] = 1.
        extended = extended.tocsc()
    difference = (extended-matrix(np.ones_like(ax), np.ones_like(ay))).tocsr()
    difference.eliminate_zeros()
    rows = np.flatnonzero(np.diff(difference.indptr))
    if len(rows) > max_rank:
        return None, dict(kind='rectangular_dct', reason='interface_rank_limit', rank=len(rows))
    delta = difference[rows]
    schur = np.eye(len(rows))
    impulse = np.zeros(covered.shape)
    for k, index in enumerate(rows):
        impulse.ravel()[index] = 1.
        schur[:, k] += delta @ rectangle_inverse(impulse).ravel()
        impulse.ravel()[index] = 0.
    if len(rows):
        with warnings.catch_warnings():
            warnings.simplefilter('error', LinAlgWarning)
            try:
                factor = lu_factor(schur)
            except LinAlgWarning:
                return None, dict(kind='rectangular_dct', reason='singular_interface_factor', rank=len(rows))
        if not np.isfinite(factor[0]).all():
            return None, dict(kind='rectangular_dct', reason='nonfinite_interface_factor', rank=len(rows))
    fluid = (~covered).ravel()
    def apply(values):
        rhs = np.zeros(covered.shape)
        rhs.ravel()[fluid] = values
        first = rectangle_inverse(rhs)
        if len(rows):
            correction = lu_solve(factor, delta @ first.ravel())
            rhs.fill(0.)
            rhs.ravel()[rows] = correction
            first -= rectangle_inverse(rhs)
        return first.ravel()[fluid]
    size = int(fluid.sum())
    return LinearOperator((size, size), dtype=float, matvec=apply), dict(
        kind='dct_boundary_schur', rank=len(rows), dense_bytes=int(schur.nbytes))
