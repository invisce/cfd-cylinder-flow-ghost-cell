"""
Cut-Cell / Embedded Boundary Discrete Poisson Solver and Conservative Projection Engine.

Provides:
  1. build_cut_cell_poisson_matrix: exact symmetric discrete Laplacian matrix for cut-cell geometry.
  2. calculate_cut_cell_divergence: conservative cell-by-cell net volumetric flux and divergence.
  3. cut_cell_project: exact discrete projection operator satisfying discrete continuity to machine precision.

Conforms to conservative Finite Volume Embedded Boundary formulations (AMReX EB / Basilisk embed).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import dct, idct
import time
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from cut_cell_geom import CutCellGeometry2D
from time_integration import ab2_cn_step
from wall_reconstruction import reconstruct_wall_gradient
from pressure_schur import boundary_schur_preconditioner
from staggered_wall import face_stencil, wall_stencil, correct_flux, owned_volumes, transfer_covered_rhs, diffusion_stencil
from types import SimpleNamespace

def build_pressure_reconstruction(geom, points):
    """Quadratic point-pressure reconstruction from nearby exterior cell centers.

    Points are (y, x). Pressure DOFs in this solver live at Cartesian centers,
    not fluid centroids. Covered/interior centers never enter the stencil.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Pressure sample points must be finite (y, x) pairs')
    if np.any((points[:, 0] < 0) | (points[:, 0] > geom.Ly) |
              (points[:, 1] < 0) | (points[:, 1] > geom.Lx)):
        raise ValueError('Pressure sample points must lie inside the domain')
    if np.any(np.hypot(points[:, 1]-geom.xc, points[:, 0]-geom.yc) < geom.R*(1-1e-12)):
        raise ValueError('Physical pressure cannot be sampled inside the solid')
    if not len(points):
        raise ValueError('At least one pressure sample is required')
    # Only nearby centers can enter these stencils; avoid a domain-sized KD tree.
    padding = 4*max(geom.dx, geom.dy)
    lo = np.maximum(0, np.floor((points.min(axis=0)-padding)/[geom.dy, geom.dx]).astype(int))
    hi = np.minimum([geom.ny, geom.nx], np.ceil((points.max(axis=0)+padding)/[geom.dy, geom.dx]).astype(int)+1)
    jj, ii = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing='ij')
    y, x = (jj+.5)*geom.dy, (ii+.5)*geom.dx
    exterior = (geom.cell_type[jj, ii] != -1) & (np.hypot(x-geom.xc, y-geom.yc) >= geom.R*(1-1e-12))
    indices = (jj*geom.nx+ii)[exterior]
    if len(indices) < 12:
        raise ValueError('Pressure reconstruction needs at least 12 exterior centers')
    centers = np.column_stack((y[exterior], x[exterior]))
    tree = cKDTree(centers)
    distances, _ = tree.query(points, k=12)
    # Include ties at the cutoff so reflection symmetry does not depend on tree ordering.
    neighbors = tree.query_ball_point(points, distances[:, -1]*(1+1e-12))
    rows, columns, weights = [], [], []
    for k, nearby in enumerate(neighbors):
        dy, dx = ((centers[nearby] - points[k]) / [geom.dy, geom.dx]).T
        basis = np.array([np.ones(len(nearby)), dx, dy, dx*dx, dx*dy, dy*dy])
        w, _, rank, _ = np.linalg.lstsq(basis, [1., 0., 0., 0., 0., 0.], rcond=1e-12)
        if rank != 6:
            raise ValueError('Pressure stencil cannot resolve a quadratic polynomial')
        rows.extend([k]*len(nearby))
        columns.extend(indices[nearby])
        weights.extend(w)
    return sp.csr_matrix((weights, (rows, columns)),
                         shape=(len(points), geom.nx * geom.ny))


def build_surface_pressure_quadrature(geom):
    # At least four surface samples per smallest grid spacing; reuse 360-point minimum.
    count = max(360, 4 * int(np.ceil(2 * np.pi * geom.R / min(geom.dx, geom.dy))))
    theta = np.arange(count) * (2 * np.pi / count)
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    points = np.column_stack((geom.yc + geom.R * normals[:, 1],
                              geom.xc + geom.R * normals[:, 0]))
    return points, build_pressure_reconstruction(geom, points), normals * (2*np.pi*geom.R/count)


def reconstruct_pressure(p, operator):
    """Apply a precomputed stencil; fail on invalid fluid data, ignore solid values."""
    values = np.asarray(p).ravel()
    if values.size != operator.shape[1] or not np.isfinite(values[operator.indices]).all():
        raise ValueError('Pressure field has invalid size or non-finite stencil values')
    return operator @ values


def surface_pressure_force(p, operator, normal_weights):
    pb = reconstruct_pressure(p, operator)
    # Remove the gauge before integration to avoid cancellation of large offsets.
    return -(pb - pb.mean()) @ normal_weights

def apply_cut_cell_poisson(phi, geom):
    """Production stencil, identical to the assembled matrix without CSR/LU storage."""
    out = np.zeros_like(phi)
    fx = geom.ax[:, 1:-1] * np.diff(phi, axis=1) / geom.dx**2
    fy = geom.ay[1:-1, :] * np.diff(phi, axis=0) / geom.dy**2
    out[:, :-1] += fx
    out[:, 1:] -= fx
    out[:-1, :] += fy
    out[1:, :] -= fy
    out[:, -1] -= 2*geom.ax[:, -1]*phi[:, -1]/geom.dx**2
    solid = geom.cell_type == -1
    out[solid] = phi[solid]
    return out


def periodic_momentum_operator(n, h):
    """Periodic full-fluid specialization of the production flux operator.

    A one-cell periodic halo keeps outer channel boundary fluxes outside the
    returned region. No cylinder geometry or pressure factorization is needed.
    """
    size = n+2
    cells = np.ones((size, size))
    vertices = np.ones((size+1, size+1))
    ax, ay = np.ones((size, size+1)), np.ones((size+1, size))
    geom = SimpleNamespace(ax=ax, ay=ay, ax_mid=cells, ay_mid=cells,
        ay_u=vertices, ax_v=vertices, u_volume_fraction=ax, v_volume_fraction=ay,
        u_merge_list=[], v_merge_list=[])
    manager = SimpleNamespace(nx=size, ny=size, dx=h, dy=h, geom=geom, slip_walls=False,
        u_vol=ax*h*h, v_vol=ay*h*h, wall_cut_indices=[])
    def evaluate(q, nu, include_advection):
        u, v = np.asarray(q).reshape(2, n, n)
        u = np.pad(u, ((1, 1), (1, 2)), mode='wrap')
        v = np.pad(v, ((1, 2), (1, 1)), mode='wrap')
        au, av, _, _ = CutCellBoundary.compute_momentum_rhs(manager, u, v, nu, 1.,
            transport='centered', include_advection=include_advection)
        return np.r_[au[1:-1, 1:-2].ravel(), av[1:-2, 1:-1].ravel()]
    return evaluate

def build_cut_cell_poisson_matrix(geom):
    """
    Constructs the exact discrete Finite Volume Laplacian matrix A for the 2D Cut-Cell MAC grid.
    
    For every fluid or cut cell (j, i), the discrete flux across each face is weighted
    by the exact face open length fraction (ax, ay):
      - West face  (j, i)  : ax[j, i] * (Phi[j, i] - Phi[j, i-1]) / dx^2
      - East face  (j, i+1): ax[j, i+1] * (Phi[j, i+1] - Phi[j, i]) / dx^2
      - South face (j, i)  : ay[j, i] * (Phi[j, i] - Phi[j-1, i]) / dy^2
      - North face (j+1, i): ay[j+1, i] * (Phi[j+1, i] - Phi[j, i]) / dy^2
      
    Boundary conditions:
      - Inlet (x=0, face i=0): Homogeneous Neumann (dPhi/dx = 0)
      - Channel walls (y=0, y=Ly): Homogeneous Neumann (dPhi/dy = 0)
      - Cylinder wall: Homogeneous Neumann (dPhi/dn = 0, no penetration Q_wall = 0)
      - Outlet (x=Lx, face i=nx): Dirichlet Phi = 0 at physical boundary distance dx/2.
      - Covered solid cells: Decoupled identity equation Phi = 0.
      
    Properties:
      - Symmetrically coupled on fluid/cut cells.
      - Weakly diagonally dominant everywhere; strictly dominant at outlet.
      - Compatible with Direct LU (SPLU) and iterative GMRES/CG solvers.
    """
    nx, ny = geom.nx, geom.ny
    dx, dy = geom.dx, geom.dy
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    N = ny * nx
    max_nnz = N * 5

    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    data = np.empty(max_nnz, dtype=np.float64)
    ptr = 0

    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            if geom.cell_type[j, i] == -1:
                # Covered solid cell: decoupled identity equation Phi = 0
                rows[ptr] = c; cols[ptr] = c; data[ptr] = 1.0; ptr += 1
                continue

            diag = 0.0

            # 1. West face: vertical face (j, i)
            ax_w = geom.ax[j, i]
            if i == 0:
                pass  # Channel inlet Neumann (dPhi/dx = 0)
            elif ax_w > 0.0:
                coeff = ax_w * inv_dx2
                rows[ptr] = c; cols[ptr] = c - 1; data[ptr] = coeff; ptr += 1
                diag -= coeff

            # 2. East face: vertical face (j, i+1)
            ax_e = geom.ax[j, i + 1]
            if i == nx - 1:
                # Outlet Dirichlet Phi = 0 at distance dx/2
                diag -= 2.0 * ax_e * inv_dx2
            elif ax_e > 0.0:
                coeff = ax_e * inv_dx2
                rows[ptr] = c; cols[ptr] = c + 1; data[ptr] = coeff; ptr += 1
                diag -= coeff

            # 3. South face: horizontal face (j, i)
            ay_s = geom.ay[j, i]
            if j == 0:
                pass  # Bottom wall Neumann (dPhi/dy = 0)
            elif ay_s > 0.0:
                coeff = ay_s * inv_dy2
                rows[ptr] = c; cols[ptr] = c - nx; data[ptr] = coeff; ptr += 1
                diag -= coeff

            # 4. North face: horizontal face (j+1, i)
            ay_n = geom.ay[j + 1, i]
            if j == ny - 1:
                pass  # Top wall Neumann (dPhi/dy = 0)
            elif ay_n > 0.0:
                coeff = ay_n * inv_dy2
                rows[ptr] = c; cols[ptr] = c + nx; data[ptr] = coeff; ptr += 1
                diag -= coeff

            rows[ptr] = c; cols[ptr] = c; data[ptr] = diag; ptr += 1

    A = sp.csc_matrix((data[:ptr], (rows[:ptr], cols[:ptr])), shape=(N, N))
    return A

def calculate_cut_cell_divergence(u, v, geom):
    """
    Computes cell-by-cell net volumetric flux and discrete divergence
    accounting for face open fractions (ax, ay) and volume fractions (kappa).
    
    NetFlux[j, i] = ax[j, i+1]*dy*u[j, i+1] - ax[j, i]*dy*u[j, i]
                  + ay[j+1, i]*dx*v[j+1, i] - ay[j, i]*dx*v[j, i]
                  
    div[j, i] = NetFlux[j, i] / (kappa[j, i] * dx * dy)
    
    Returns:
      max_div_fluid: maximum |div| across all fluid and cut cells
      rms_div_fluid: root-mean-square |div| across all fluid and cut cells
      max_net_flux : maximum |NetFlux| across all fluid and cut cells
      net_flux_grid: full 2D array of NetFlux
      div_grid     : full 2D array of divergence
    """
    nx, ny = geom.nx, geom.ny
    dx, dy = geom.dx, geom.dy
    net_flux_grid = np.zeros((ny, nx), dtype=np.float64)
    div_grid = np.zeros((ny, nx), dtype=np.float64)

    # Compute net flux on all cells
    for j in range(ny):
        for i in range(nx):
            if geom.cell_type[j, i] != -1:
                fx = geom.ax[j, i + 1] * dy * u[j, i + 1] - geom.ax[j, i] * dy * u[j, i]
                fy = geom.ay[j + 1, i] * dx * v[j + 1, i] - geom.ay[j, i] * dx * v[j, i]
                net_flux_grid[j, i] = fx + fy
                vol = geom.volume_fraction[j, i] * dx * dy
                if vol > 0.0:
                    div_grid[j, i] = (fx + fy) / vol

    fluid_mask = geom.cell_type != -1
    fluid_divs = div_grid[fluid_mask]
    fluid_fluxes = net_flux_grid[fluid_mask]

    max_div_fluid = float(np.max(np.abs(fluid_divs))) if len(fluid_divs) > 0 else 0.0
    rms_div_fluid = float(np.sqrt(np.mean(fluid_divs**2))) if len(fluid_divs) > 0 else 0.0
    max_net_flux = float(np.max(np.abs(fluid_fluxes))) if len(fluid_fluxes) > 0 else 0.0

    return max_div_fluid, rms_div_fluid, max_net_flux, net_flux_grid, div_grid

def cut_cell_project(u_star, v_star, geom, dt, rho, pressure_solver='direct', splu=None, A=None, M=None):
    """
    Executes conservative projection on provisional velocity fields:
      1. Assemble RHS = (rho / dt) * (div_EB u_star)
      2. Solve discrete Poisson equation A * Phi = RHS
      3. Correct velocity on open faces:
         u[j, i] = u_star[j, i] - (dt / rho) * (Phi[j, i] - Phi[j, i-1]) / dx
         v[j, i] = v_star[j, i] - (dt / rho) * (Phi[j, i] - Phi[j-1, i]) / dy
         
    Returns:
      u_new, v_new, phi
    """
    nx, ny = geom.nx, geom.ny
    dx, dy = geom.dx, geom.dy
    inv_dt = rho / dt
    dt_over_rho = dt / rho

    # Build operator if not provided
    if A is None:
        A = build_cut_cell_poisson_matrix(geom)
    if pressure_solver == 'direct' and splu is None:
        splu = spla.splu(A, permc_spec='MMD_AT_PLUS_A')
    elif pressure_solver == 'iterative' and M is None:
        ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=10)
        M = spla.LinearOperator(A.shape, ilu.solve)

    # 1. Compute RHS
    rhs = np.zeros((ny, nx), dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            if geom.cell_type[j, i] != -1:
                flux_x = (geom.ax[j, i + 1] * u_star[j, i + 1] - geom.ax[j, i] * u_star[j, i]) / dx
                flux_y = (geom.ay[j + 1, i] * v_star[j + 1, i] - geom.ay[j, i] * v_star[j, i]) / dy
                rhs[j, i] = inv_dt * (flux_x + flux_y)

    rhs_flat = rhs.ravel()

    # 2. Solve Poisson equation
    if pressure_solver == 'direct':
        phi_flat = splu.solve(rhs_flat)
    else:
        phi_flat, info = spla.gmres(A, rhs_flat, M=M, rtol=1e-10, atol=0.0)
        if info != 0:
            raise RuntimeError(f"Cut-Cell GMRES pressure solver failed to converge: info={info}")

    phi = phi_flat.reshape((ny, nx))

    # 3. Correct velocity fields
    u_new = u_star.copy()
    v_new = v_star.copy()

    # u: vertical faces i in [1, nx-1]
    for j in range(ny):
        for i in range(1, nx):
            if geom.ax[j, i] > 0.0:
                u_new[j, i] -= dt_over_rho * (phi[j, i] - phi[j, i - 1]) / dx
            else:
                u_new[j, i] = 0.0
        # Outlet face i = nx
        u_new[j, nx] += dt_over_rho * 2.0 * phi[j, nx - 1] / dx

    # v: horizontal faces j in [1, ny-1]
    for j in range(1, ny):
        for i in range(nx):
            if geom.ay[j, i] > 0.0:
                v_new[j, i] -= dt_over_rho * (phi[j, i] - phi[j - 1, i]) / dy
            else:
                v_new[j, i] = 0.0

    return u_new, v_new, phi

def extract_cut_cell_surface_forces(u, v, p, geom, mu):
    """
    Computes reconstructed pressure and viscous forces on the physical cylinder surface
    by integrating along boundary segments:
      - Pressure force: F_p = -oint p n ds = -sum (p_b * A_b)
      - Viscous force : F_v =  oint tau . n ds = sum (tau_w * t * ds)
    """
    _, pressure_operator, normal_weights = build_surface_pressure_quadrature(geom)
    fx_p, fy_p = surface_pressure_force(p, pressure_operator, normal_weights)
    wall = wall_stencil(geom)
    tau_w = mu * reconstruct_wall_gradient(u, v, geom, wall.rate)
    fx_v = float(np.sum(tau_w * wall.tx * wall.ds))
    fy_v = float(np.sum(tau_w * wall.ty * wall.ds))

    fx_tot = fx_p + fx_v
    fy_tot = fy_p + fy_v
    return fx_tot, fy_tot, fx_p, fy_p, fx_v, fy_v

def measure_cut_cell_surface_slip(u, v, geom):
    """Measures physical boundary slip velocity at the true cylinder surface."""
    dx, dy = geom.dx, geom.dy
    nx, ny = geom.nx, geom.ny
    xu = np.arange(nx + 1) * dx
    yu = (np.arange(ny) + 0.5) * dy
    xv = (np.arange(nx) + 0.5) * dx
    yv = np.arange(ny + 1) * dy

    u_interp = RegularGridInterpolator((yu, xu), u, bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((yv, xv), v, bounds_error=False, fill_value=0.0)

    cut_mask = geom.cell_type == 0
    pts_b = np.column_stack((geom.bndry_cy[cut_mask], geom.bndry_cx[cut_mask]))
    ub = u_interp(pts_b)
    vb = v_interp(pts_b)
    slip = np.hypot(ub, vb)
    return float(np.mean(slip)), float(np.max(slip)), slip

class CutCellBoundary:
    """
    High-performance manager for Cut-Cell / Embedded Boundary Cartesian flow solver.
    Encapsulates geometry, discrete Poisson matrix, solver factorization,
    projection, predictor boundary treatment, and surface traction extraction.
    """
    def __init__(self, nx, ny, Lx, Ly, xc, yc, R, pressure_solver='direct', slip_walls=False):
        if pressure_solver not in ('direct', 'iterative', 'cg_dct'):
            raise ValueError('Unknown Cut-Cell pressure solver')
        self.geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
        self.nx = nx
        self.ny = ny
        self.dx = self.geom.dx
        self.dy = self.geom.dy
        self.pressure_solver = pressure_solver
        self.slip_walls = slip_walls
        self.fluid_pressure = (self.geom.cell_type != -1).ravel()
        self.last_pressure_diagnostics = {}
        self.pressure_preconditioner_info = None
        self.cn_advection = self.cn_dt = self.cn_pressure = None
        if pressure_solver == 'cg_dct':
            self.A = spla.LinearOperator((nx*ny, nx*ny), dtype=float,
                matvec=lambda x: apply_cut_cell_poisson(x.reshape(ny, nx), self.geom).ravel())
        else:
            self.A = build_cut_cell_poisson_matrix(self.geom)

        if pressure_solver == 'direct':
            self.splu = spla.splu(self.A, permc_spec='MMD_AT_PLUS_A')
            self.M = None
        elif pressure_solver == 'iterative':
            self.splu = None
            ilu = spla.spilu(self.A, drop_tol=1e-4, fill_factor=10)
            self.M = spla.LinearOperator(self.A.shape, ilu.solve)
        else:
            self.splu = self.M = None
            self.pressure_eigenvalues = (
                4*np.sin(np.pi*np.arange(ny)/(2*ny))[:, None]**2/self.dy**2
                + 4*np.sin(np.pi*(np.arange(nx)+.5)/(2*nx))[None, :]**2/self.dx**2)
            nf = int(self.fluid_pressure.sum())
            self.pressure_operator = spla.LinearOperator((nf, nf), dtype=float,
                matvec=self._positive_pressure_action)
            self.pressure_preconditioner = spla.LinearOperator((nf, nf), dtype=float,
                matvec=self._pressure_precondition)
            corrected, self.pressure_preconditioner_info = boundary_schur_preconditioner(
                self.geom, build_cut_cell_poisson_matrix, self._rectangle_pressure_inverse)
            if corrected is not None:
                self.pressure_preconditioner = corrected

        # Precompute coordinate grids for fast interpolations
        self.xp = (np.arange(nx) + 0.5) * self.dx
        self.yp = (np.arange(ny) + 0.5) * self.dy
        self.xu = np.arange(nx + 1) * self.dx
        self.yu = (np.arange(ny) + 0.5) * self.dy
        self.xv = (np.arange(nx) + 0.5) * self.dx
        self.yv = np.arange(ny + 1) * self.dy

        # Precompute cut segment properties for fast surface force extraction
        self.cut_mask = self.geom.cell_type == 0
        self.b_cx = self.geom.bndry_cx[self.cut_mask]
        self.b_cy = self.geom.bndry_cy[self.cut_mask]
        self.b_nx = self.geom.bndry_normal_unit_x[self.cut_mask]
        self.b_ny = self.geom.bndry_normal_unit_y[self.cut_mask]
        self.A_bx = self.geom.bndry_normal_int_x[self.cut_mask]
        self.A_by = self.geom.bndry_normal_int_y[self.cut_mask]
        self.ds = self.geom.bndry_area[self.cut_mask]

        self.pts_b = np.column_stack((self.b_cy, self.b_cx))
        self.dn = self.dx
        # Arc centroids lie inside the circle; sample from its true surface.
        self.pts_p = np.column_stack((yc + (R + self.dn) * self.b_ny,
                                      xc + (R + self.dn) * self.b_nx))
        self.wall = wall_stencil(self.geom)
        self.tx, self.ty, self.ds = self.wall.tx, self.wall.ty, self.wall.ds
        self.wall_gradient_operator = self.wall.rate
        self.momentum_faces = {(component, axis): face_stencil(self.geom, component, axis)
                               for component in ('u', 'v') for axis in ('x', 'y')}
        diffusion_reach = max(np.hypot(self.dx,2*self.dy),np.hypot(2*self.dx,self.dy))/max(self.dx,self.dy)
        self.diffusion_reconstruction = {key: face_stencil(self.geom,*key,reach=diffusion_reach,cartesian=True)
                                        for key in self.momentum_faces}
        self.diffusion_faces = {key: diffusion_stencil(self.geom,*key,stencil,self.diffusion_reconstruction[key])
                               for key,stencil in self.momentum_faces.items()}

        self.pressure_points, self.surface_pressure_operator, self.pressure_normal_weights = (
            build_surface_pressure_quadrature(self.geom))

        # Staggered momentum control volumes and merging precomputations
        self.u_vol, self.u_covered, self.u_owner = owned_volumes(self.geom, 'u')
        self.v_vol, self.v_covered, self.v_owner = owned_volumes(self.geom, 'v')
        self.u_vol_eff = self.u_vol.copy()
        for (js, is_), (jm, im) in self.geom.u_merge_list:
            vm = self.u_vol[js, is_] + self.u_vol[jm, im]
            self.u_vol_eff[js, is_] = vm
            self.u_vol_eff[jm, im] = vm

        self.v_vol_eff = self.v_vol.copy()
        for (js, is_), (jm, im) in self.geom.v_merge_list:
            vm = self.v_vol[js, is_] + self.v_vol[jm, im]
            self.v_vol_eff[js, is_] = vm
            self.v_vol_eff[jm, im] = vm

        self.wall_cut_indices = np.argwhere(self.cut_mask)

        # History buffers for AB2 time-stepping
        self.prev_au = None
        self.prev_av = None
        self.prev_dt = None

        # Precompute boundary velocity reconstruction stencils
        self._precompute_boundary_reconstruction()

    def _precompute_boundary_reconstruction(self):
        """
        Precomputes image-point stencils for near-boundary velocity reconstruction.
        Identifies solid grid nodes (distance < 0 from cylinder surface) bordering
        fluid cells, and sets up normal image-point probes into the fluid.
        Enforces Dirichlet no-slip wall condition u(r=R) = 0 to second-order accuracy.
        """
        XU, YU = np.meshgrid(self.xu, self.yu)
        self.dist_u = np.hypot(XU - self.geom.xc, YU - self.geom.yc) - self.geom.R

        XV, YV = np.meshgrid(self.xv, self.yv)
        self.dist_v = np.hypot(XV - self.geom.xc, YV - self.geom.yc) - self.geom.R

        # 1. u ghost faces: solid nodes (dist_u < 0) bordering fluid (dist_u >= 0)
        self.u_ghost_info = []
        for j in range(self.ny):
            for i in range(self.nx + 1):
                if self.dist_u[j, i] < 0:
                    has_fluid = False
                    if i > 0 and self.dist_u[j, i - 1] >= 0: has_fluid = True
                    if i < self.nx and self.dist_u[j, i + 1] >= 0: has_fluid = True
                    if j > 0 and self.dist_u[j - 1, i] >= 0: has_fluid = True
                    if j < self.ny - 1 and self.dist_u[j + 1, i] >= 0: has_fluid = True
                    if has_fluid:
                        x = i * self.dx
                        y = (j + 0.5) * self.dy
                        r = np.hypot(x - self.geom.xc, y - self.geom.yc)
                        ds = self.geom.R - r
                        nx_val = (x - self.geom.xc) / r if r > 0 else 1.0
                        ny_val = (y - self.geom.yc) / r if r > 0 else 0.0
                        delta = max(ds, 0.5 * min(self.dx, self.dy))
                        x_ip = x + (ds + delta) * nx_val
                        y_ip = y + (ds + delta) * ny_val
                        gamma = ds / delta
                        self.u_ghost_info.append(((j, i), x_ip, y_ip, gamma))

        # 2. v ghost faces: solid nodes (dist_v < 0) bordering fluid (dist_v >= 0)
        self.v_ghost_info = []
        for j in range(self.ny + 1):
            for i in range(self.nx):
                if self.dist_v[j, i] < 0:
                    has_fluid = False
                    if i > 0 and self.dist_v[j, i - 1] >= 0: has_fluid = True
                    if i < self.nx - 1 and self.dist_v[j, i + 1] >= 0: has_fluid = True
                    if j > 0 and self.dist_v[j - 1, i] >= 0: has_fluid = True
                    if j < self.ny and self.dist_v[j + 1, i] >= 0: has_fluid = True
                    if has_fluid:
                        x = (i + 0.5) * self.dx
                        y = j * self.dy
                        r = np.hypot(x - self.geom.xc, y - self.geom.yc)
                        ds = self.geom.R - r
                        nx_val = (x - self.geom.xc) / r if r > 0 else 1.0
                        ny_val = (y - self.geom.yc) / r if r > 0 else 0.0
                        delta = max(ds, 0.5 * min(self.dx, self.dy))
                        x_ip = x + (ds + delta) * nx_val
                        y_ip = y + (ds + delta) * ny_val
                        gamma = ds / delta
                        self.v_ghost_info.append(((j, i), x_ip, y_ip, gamma))

        # Vectorized lookup tables for u reconstruction (only solid faces ax == 0)
        u_rec_j, u_rec_i, u_rec_j0, u_rec_i0, u_rec_w, u_rec_gam = [], [], [], [], [], []
        u_eval_j, u_eval_i, u_eval_j0, u_eval_i0, u_eval_w, u_eval_gam = [], [], [], [], [], []
        for (j, i), x_ip, y_ip, gamma in self.u_ghost_info:
            i0 = int(np.floor(x_ip / self.dx))
            j0 = int(np.floor((y_ip - 0.5 * self.dy) / self.dy))
            x0 = i0 * self.dx
            y0 = (j0 + 0.5) * self.dy
            i0 = max(0, min(self.nx - 1, i0))
            j0 = max(0, min(self.ny - 2, j0))
            rx = max(0.0, min(1.0, (x_ip - x0) / self.dx))
            ry = max(0.0, min(1.0, (y_ip - y0) / self.dy))
            w = [(1.0 - rx) * (1.0 - ry), rx * (1.0 - ry), (1.0 - rx) * ry, rx * ry]

            u_eval_j.append(j); u_eval_i.append(i); u_eval_j0.append(j0); u_eval_i0.append(i0)
            u_eval_w.append(w); u_eval_gam.append(gamma)

            if self.geom.ax[j, i] == 0.0:
                u_rec_j.append(j); u_rec_i.append(i); u_rec_j0.append(j0); u_rec_i0.append(i0)
                u_rec_w.append(w); u_rec_gam.append(gamma)

        self.u_rec_j = np.array(u_rec_j, dtype=np.int32)
        self.u_rec_i = np.array(u_rec_i, dtype=np.int32)
        self.u_rec_j0 = np.array(u_rec_j0, dtype=np.int32)
        self.u_rec_i0 = np.array(u_rec_i0, dtype=np.int32)
        self.u_rec_w = np.array(u_rec_w, dtype=np.float64) if len(u_rec_w) > 0 else np.empty((0, 4))
        self.u_rec_gamma = np.array(u_rec_gam, dtype=np.float64)

        self.u_eval_j = np.array(u_eval_j, dtype=np.int32)
        self.u_eval_i = np.array(u_eval_i, dtype=np.int32)
        self.u_eval_j0 = np.array(u_eval_j0, dtype=np.int32)
        self.u_eval_i0 = np.array(u_eval_i0, dtype=np.int32)
        self.u_eval_w = np.array(u_eval_w, dtype=np.float64) if len(u_eval_w) > 0 else np.empty((0, 4))
        self.u_eval_gamma = np.array(u_eval_gam, dtype=np.float64)

        # Vectorized lookup tables for v reconstruction (only solid faces ay == 0)
        v_rec_j, v_rec_i, v_rec_j0, v_rec_i0, v_rec_w, v_rec_gam = [], [], [], [], [], []
        v_eval_j, v_eval_i, v_eval_j0, v_eval_i0, v_eval_w, v_eval_gam = [], [], [], [], [], []
        for (j, i), x_ip, y_ip, gamma in self.v_ghost_info:
            i0 = int(np.floor((x_ip - 0.5 * self.dx) / self.dx))
            j0 = int(np.floor(y_ip / self.dy))
            x0 = (i0 + 0.5) * self.dx
            y0 = j0 * self.dy
            i0 = max(0, min(self.nx - 2, i0))
            j0 = max(0, min(self.ny - 1, j0))
            rx = max(0.0, min(1.0, (x_ip - x0) / self.dx))
            ry = max(0.0, min(1.0, (y_ip - y0) / self.dy))
            w = [(1.0 - rx) * (1.0 - ry), rx * (1.0 - ry), (1.0 - rx) * ry, rx * ry]

            v_eval_j.append(j); v_eval_i.append(i); v_eval_j0.append(j0); v_eval_i0.append(i0)
            v_eval_w.append(w); v_eval_gam.append(gamma)

            if self.geom.ay[j, i] == 0.0:
                v_rec_j.append(j); v_rec_i.append(i); v_rec_j0.append(j0); v_rec_i0.append(i0)
                v_rec_w.append(w); v_rec_gam.append(gamma)

        self.v_rec_j = np.array(v_rec_j, dtype=np.int32)
        self.v_rec_i = np.array(v_rec_i, dtype=np.int32)
        self.v_rec_j0 = np.array(v_rec_j0, dtype=np.int32)
        self.v_rec_i0 = np.array(v_rec_i0, dtype=np.int32)
        self.v_rec_w = np.array(v_rec_w, dtype=np.float64) if len(v_rec_w) > 0 else np.empty((0, 4))
        self.v_rec_gamma = np.array(v_rec_gam, dtype=np.float64)

        self.v_eval_j = np.array(v_eval_j, dtype=np.int32)
        self.v_eval_i = np.array(v_eval_i, dtype=np.int32)
        self.v_eval_j0 = np.array(v_eval_j0, dtype=np.int32)
        self.v_eval_i0 = np.array(v_eval_i0, dtype=np.int32)
        self.v_eval_w = np.array(v_eval_w, dtype=np.float64) if len(v_eval_w) > 0 else np.empty((0, 4))
        self.v_eval_gamma = np.array(v_eval_gam, dtype=np.float64)

        # Deep interior masks
        self.u_deep_interior_mask = (self.dist_u < 0)
        if len(self.u_eval_j) > 0:
            self.u_deep_interior_mask[self.u_eval_j, self.u_eval_i] = False

        self.v_deep_interior_mask = (self.dist_v < 0)
        if len(self.v_eval_j) > 0:
            self.v_deep_interior_mask[self.v_eval_j, self.v_eval_i] = False

    def reconstruct_boundary_velocities(self, u, v, u_wall=0.0, v_wall=0.0):
        """
        Fast vectorized reconstruction of solid boundary faces (ax == 0, ay == 0)
        via normal image-point reflection satisfying u(r=R) = u_wall to second order.
        """
        if len(self.u_rec_j) > 0:
            u_ip = (self.u_rec_w[:, 0] * u[self.u_rec_j0, self.u_rec_i0] +
                    self.u_rec_w[:, 1] * u[self.u_rec_j0, self.u_rec_i0 + 1] +
                    self.u_rec_w[:, 2] * u[self.u_rec_j0 + 1, self.u_rec_i0] +
                    self.u_rec_w[:, 3] * u[self.u_rec_j0 + 1, self.u_rec_i0 + 1])
            u[self.u_rec_j, self.u_rec_i] = u_wall - self.u_rec_gamma * (u_ip - u_wall)

        if len(self.v_rec_j) > 0:
            v_ip = (self.v_rec_w[:, 0] * v[self.v_rec_j0, self.v_rec_i0] +
                    self.v_rec_w[:, 1] * v[self.v_rec_j0, self.v_rec_i0 + 1] +
                    self.v_rec_w[:, 2] * v[self.v_rec_j0 + 1, self.v_rec_i0] +
                    self.v_rec_w[:, 3] * v[self.v_rec_j0 + 1, self.v_rec_i0 + 1])
            v[self.v_rec_j, self.v_rec_i] = v_wall - self.v_rec_gamma * (v_ip - v_wall)

    def build_boundary_evaluation_fields(self, u, v, u_wall=0.0, v_wall=0.0):
        """
        Fast vectorized construction of continuous velocity fields u_eval and v_eval
        for surface force extraction and wall slip evaluation.
        """
        u_eval = u.copy()
        v_eval = v.copy()

        if len(self.u_eval_j) > 0:
            u_ip = (self.u_eval_w[:, 0] * u[self.u_eval_j0, self.u_eval_i0] +
                    self.u_eval_w[:, 1] * u[self.u_eval_j0, self.u_eval_i0 + 1] +
                    self.u_eval_w[:, 2] * u[self.u_eval_j0 + 1, self.u_eval_i0] +
                    self.u_eval_w[:, 3] * u[self.u_eval_j0 + 1, self.u_eval_i0 + 1])
            u_eval[self.u_eval_j, self.u_eval_i] = u_wall - self.u_eval_gamma * (u_ip - u_wall)

        if len(self.v_eval_j) > 0:
            v_ip = (self.v_eval_w[:, 0] * v[self.v_eval_j0, self.v_eval_i0] +
                    self.v_eval_w[:, 1] * v[self.v_eval_j0, self.v_eval_i0 + 1] +
                    self.v_eval_w[:, 2] * v[self.v_eval_j0 + 1, self.v_eval_i0] +
                    self.v_eval_w[:, 3] * v[self.v_eval_j0 + 1, self.v_eval_i0 + 1])
            v_eval[self.v_eval_j, self.v_eval_i] = v_wall - self.v_eval_gamma * (v_ip - v_wall)

        u_eval[self.u_deep_interior_mask] = 0.0
        v_eval[self.v_deep_interior_mask] = 0.0
        return u_eval, v_eval

    def apply_predictor_boundary(self, u_star, v_star, u_inlet=None):
        """Enforces physical boundary conditions without unphysical open-fraction damping."""
        if u_inlet is not None:
            u_star[:, 0] = u_inlet
        u_star[self.geom.ax == 0.0] = 0.0
        v_star[self.geom.ay == 0.0] = 0.0
        v_star[0, :] = 0.0
        v_star[self.ny, :] = 0.0

    def _pressure_grid(self, values):
        full = np.zeros(self.nx*self.ny)
        full[self.fluid_pressure] = values
        return full.reshape(self.ny, self.nx)

    def _positive_pressure_action(self, values):
        return -(self.A @ self._pressure_grid(values).ravel())[self.fluid_pressure]

    def _pressure_precondition(self, values):
        return self._rectangle_pressure_inverse(self._pressure_grid(values)).ravel()[self.fluid_pressure]

    def _rectangle_pressure_inverse(self, full):
        transformed = dct(dct(full, type=4, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
        return idct(idct(transformed/self.pressure_eigenvalues, type=2, axis=0, norm='ortho'),
                    type=4, axis=1, norm='ortho')

    def solve_pressure(self, rhs, dt, rho, divergence_tolerance=1e-10):
        """Solve and verify the *unpreconditioned* residual in physical divergence units."""
        if (rhs.shape != (self.ny, self.nx) or not np.isfinite(rhs).all()
                or not np.isfinite([dt, rho, divergence_tolerance]).all()
                or min(dt, rho, divergence_tolerance) <= 0):
            raise ValueError('Invalid pressure solve inputs')
        start = time.perf_counter()
        b = rhs.ravel().copy()
        b[~self.fluid_pressure] = 0.
        count = [0]
        def callback(_):
            count[0] += 1
        kappa = self.geom.volume_fraction.ravel()[self.fluid_pressure]
        solution = np.zeros_like(b)
        residual = b.copy()
        for _ in range(3):
            if self.pressure_solver == 'direct':
                solution += self.splu.solve(residual)
            elif self.pressure_solver == 'cg_dct':
                delta, info = spla.cg(self.pressure_operator, -residual[self.fluid_pressure],
                    M=self.pressure_preconditioner, rtol=1e-12, atol=0., maxiter=1000, callback=callback)
                if info:
                    raise RuntimeError(f'CG-DCT failed: info={info}')
                solution[self.fluid_pressure] += delta
            else:
                delta, info = spla.gmres(self.A, residual, M=self.M, rtol=1e-12, atol=0.,
                                        callback=callback, callback_type='pr_norm')
                if info:
                    raise RuntimeError(f'Pressure GMRES failed: info={info}')
                solution += delta
            residual = b - self.A @ solution
            projected_divergence = dt/rho*np.max(abs(residual[self.fluid_pressure])/kappa)
            if projected_divergence < divergence_tolerance*.5:
                break
        self.last_pressure_diagnostics = dict(
            residual_linf=float(np.max(abs(residual))),
            residual_l2=float(np.linalg.norm(residual)), iterations=count[0],
            predicted_divergence=float(projected_divergence), seconds=time.perf_counter()-start)
        if self.pressure_preconditioner_info is not None:
            self.last_pressure_diagnostics['preconditioner'] = self.pressure_preconditioner_info
        if projected_divergence >= divergence_tolerance:
            raise RuntimeError(f'Pressure residual exceeds divergence budget: {projected_divergence:g}')
        return solution.reshape(self.ny, self.nx)

    def compute_rhs(self, u_star, v_star, dt, rho, rhs_out):
        """Vectorized computation of discrete divergence RHS for Cut-Cell Poisson equation."""
        inv_dt = rho / dt
        flux_x = (self.geom.ax[:, 1:] * u_star[:, 1:] - self.geom.ax[:, :-1] * u_star[:, :-1]) / self.dx
        flux_y = (self.geom.ay[1:, :] * v_star[1:, :] - self.geom.ay[:-1, :] * v_star[:-1, :]) / self.dy
        rhs_out[:, :] = inv_dt * (flux_x + flux_y)
        rhs_out[self.geom.cell_type == -1] = 0.0

    def apply_corrector(self, u, v, u_star, v_star, phi, dt, rho, u_inlet):
        """Vectorized Cut-Cell velocity corrector ensuring exact face flux consistency."""
        dt_over_rho = dt / rho
        u[:, 0] = u_inlet
        u[:, 1:-1] = np.where(self.geom.ax[:, 1:-1] > 0.0,
                              u_star[:, 1:-1] - dt_over_rho * (phi[:, 1:] - phi[:, :-1]) / self.dx,
                              0.0)
        u[:, self.nx] = u_star[:, self.nx] + dt_over_rho * 2.0 * phi[:, self.nx - 1] / self.dx

        v[0, :] = 0.0
        v[1:-1, :] = np.where(self.geom.ay[1:-1, :] > 0.0,
                              v_star[1:-1, :] - dt_over_rho * (phi[1:, :] - phi[:-1, :]) / self.dy,
                              0.0)
        v[self.ny, :] = 0.0

    def extract_true_surface_forces(self, u, v, p, mu):
        """Integrate reconstructed pressure and wall shear on the physical circle."""
        fx_p, fy_p = surface_pressure_force(p, self.surface_pressure_operator, self.pressure_normal_weights)

        tau_w = mu * self.wall_shear_rate(u, v)
        fx_v = float(np.sum(tau_w * self.tx * self.ds))
        fy_v = float(np.sum(tau_w * self.ty * self.ds))

        fx_tot = fx_p + fx_v
        fy_tot = fy_p + fy_v
        return fx_tot, fy_tot, fx_p, fy_p, fx_v, fy_v

    def pressure_at(self, p, points):
        """Sample physical pressure using the same reconstruction as surface forces."""
        return reconstruct_pressure(p, build_pressure_reconstruction(self.geom, points))

    def wall_shear_rate(self, u, v):
        """Shared traction from physical open-face averages, with stationary-wall no-slip.

        The local streamfunction fit enforces no-slip and incompressibility and
        integrates each basis along the actual open face. Cartesian face centers
        may lie inside the solid and are not point-velocity samples.
        """
        return reconstruct_wall_gradient(u, v, self.geom, self.wall_gradient_operator)

    def measure_surface_slip(self, u, v):
        """Measures physical boundary slip velocity at the true cylinder surface."""
        u_eval, v_eval = self.build_boundary_evaluation_fields(u, v)
        u_interp = RegularGridInterpolator((self.yu, self.xu), u_eval, bounds_error=False, fill_value=0.0)
        v_interp = RegularGridInterpolator((self.yv, self.xv), v_eval, bounds_error=False, fill_value=0.0)
        ub = u_interp(self.pts_b)
        vb = v_interp(self.pts_b)
        slip = np.hypot(ub, vb)
        return float(np.mean(slip)), float(np.max(slip)), slip

    def compute_momentum_rhs(self, u, v, nu, rho, transport='upwind', include_advection=True):
        """
        Computes exact conservative momentum RHS on staggered Cut-Cell control volumes.
        Includes telescopic internal advective & diffusive fluxes on open faces,
        physical wall shear traction, and conservative cell merging for small cut cells.

        Returns:
            a_u (ny, nx+1): acceleration field for u
            a_v (ny+1, nx): acceleration field for v
            rhs_u (ny, nx+1): net momentum rate of change for u
            rhs_v (ny+1, nx): net momentum rate of change for v
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        geom = self.geom
        curved = len(self.wall_cut_indices) > 0
        field = np.r_[u.ravel(), v.ravel()] if curved else None

        # 1. u face fluxes
        # East vertical face F_xx on ax_mid: shape (ny, nx)
        u_face_x = 0.5 * (u[:, :-1] + u[:, 1:])
        if transport == 'centered':
            u_adv_x = u_face_x
        else:
            u_adv_x = np.where(u_face_x >= 0, u[:, :-1], u[:, 1:])
        ax_mid_eff = geom.ax_mid
        adv_xx = rho * u_face_x * u_adv_x * ax_mid_eff * dy
        if not include_advection:
            adv_xx.fill(0.)
        diff_xx = rho * nu * (u[:, 1:] - u[:, :-1]) / dx * ax_mid_eff * dy
        F_xx = diff_xx - adv_xx
        if curved:
            correct_flux(F_xx, self.momentum_faces['u', 'x'], field, rho, nu, 'u', 'x', include_advection)
            if nu != 0:
                correction = self.diffusion_faces['u','x']
                F_xx.ravel()[correction.indices] += rho*nu*(correction.correction @ field)

        # North horizontal face F_xy on ay_u: shape (ny + 1, nx + 1)
        F_xy = np.zeros((ny + 1, nx + 1), dtype=np.float64)
        v_face_full = np.zeros((ny - 1, nx + 1), dtype=np.float64)
        v_face_full[:, 0] = v[1:ny, 0]
        v_face_full[:, 1:nx] = 0.5 * (v[1:ny, :-1] + v[1:ny, 1:])
        v_face_full[:, nx] = v[1:ny, nx - 1]

        if transport == 'centered':
            u_adv_y = 0.5 * (u[0:ny-1, :] + u[1:ny, :])
        else:
            u_adv_y = np.where(v_face_full >= 0, u[0:ny-1, :], u[1:ny, :])
        ay_u_eff = geom.ay_u.copy()
        adv_xy = rho * v_face_full * u_adv_y * ay_u_eff[1:ny, :] * dx
        if not include_advection:
            adv_xy.fill(0.)
        diff_xy = rho * nu * (u[1:ny, :] - u[0:ny-1, :]) / dy * ay_u_eff[1:ny, :] * dx
        F_xy[1:ny, :] = diff_xy - adv_xy
        # Channel walls (no-slip u = 0 at distance 0.5*dy)
        if not self.slip_walls:
            F_xy[0, :] = 2.0 * rho * nu * u[0, :] / dy * ay_u_eff[0, :] * dx
            F_xy[ny, :] = -2.0 * rho * nu * u[ny - 1, :] / dy * ay_u_eff[ny, :] * dx

        if curved:
            correct_flux(F_xy, self.momentum_faces['u', 'y'], field, rho, nu, 'u', 'y', include_advection)
            if nu != 0:
                correction = self.diffusion_faces['u','y']
                F_xy.ravel()[correction.indices] += rho*nu*(correction.correction @ field)

        rhs_u = np.zeros((ny, nx + 1), dtype=np.float64)
        rhs_u[:, 1:nx] = (F_xx[:, 1:nx] - F_xx[:, :nx-1]) + (F_xy[1:, 1:nx] - F_xy[:-1, 1:nx])
        # Natural outlet: exterior pressure and viscous traction cancel in
        # the coupled momentum equation. Retain convection, west-face flux,
        # and transverse flux over the half-width boundary volume. The
        # existing -2*p/(rho*dx) pressure action supplies its west pressure.
        exterior_x = -rho*u[:, nx]**2*geom.ax[:, nx]*dy if include_advection else 0.
        rhs_u[:, nx] = (exterior_x-F_xx[:, -1]
                       + .5*(F_xy[1:, nx]-F_xy[:-1, nx]))

        # 2. v face fluxes
        # North horizontal face F_yy on ay_mid: shape (ny, nx)
        v_face_y = 0.5 * (v[:-1, :] + v[1:, :])
        if transport == 'centered':
            v_adv_y = v_face_y
        else:
            v_adv_y = np.where(v_face_y >= 0, v[:-1, :], v[1:, :])
        ay_mid_eff = geom.ay_mid
        adv_yy = rho * v_face_y * v_adv_y * ay_mid_eff * dx
        if not include_advection:
            adv_yy.fill(0.)
        diff_yy = rho * nu * (v[1:, :] - v[:-1, :]) / dy * ay_mid_eff * dx
        F_yy = diff_yy - adv_yy
        if curved:
            correct_flux(F_yy, self.momentum_faces['v', 'y'], field, rho, nu, 'v', 'y', include_advection)
            if nu != 0:
                correction = self.diffusion_faces['v','y']
                F_yy.ravel()[correction.indices] += rho*nu*(correction.correction @ field)

        # East vertical face F_yx on ax_v: shape (ny + 1, nx + 1)
        F_yx = np.zeros((ny + 1, nx + 1), dtype=np.float64)
        u_face_v_full = np.zeros((ny + 1, nx - 1), dtype=np.float64)
        u_face_v_full[0, :] = u[0, 1:nx]
        u_face_v_full[1:ny, :] = 0.5 * (u[:-1, 1:nx] + u[1:, 1:nx])
        u_face_v_full[ny, :] = u[ny - 1, 1:nx]

        if transport == 'centered':
            v_adv_x = 0.5 * (v[:, 0:nx-1] + v[:, 1:nx])
        else:
            v_adv_x = np.where(u_face_v_full >= 0, v[:, 0:nx-1], v[:, 1:nx])
        ax_v_eff = geom.ax_v.copy()
        adv_yx = rho * u_face_v_full * v_adv_x * ax_v_eff[:, 1:nx] * dy
        if not include_advection:
            adv_yx.fill(0.)
        diff_yx = rho * nu * (v[:, 1:nx] - v[:, 0:nx-1]) / dx * ax_v_eff[:, 1:nx] * dy
        F_yx[:, 1:nx] = diff_yx - adv_yx
        # Inlet v = 0
        F_yx[:, 0] = 2.0 * rho * nu * v[:, 0] / dx * ax_v_eff[:, 0] * dy
        # Outlet dv/dx = 0
        u_out_v = np.zeros(ny + 1, dtype=np.float64)
        u_out_v[0] = u[0, nx]
        u_out_v[1:ny] = 0.5 * (u[:-1, nx] + u[1:, nx])
        u_out_v[ny] = u[ny - 1, nx]
        if include_advection:
            F_yx[:, nx] = -rho * u_out_v * v[:, nx - 1] * ax_v_eff[:, nx] * dy

        if curved:
            correct_flux(F_yx, self.momentum_faces['v', 'x'], field, rho, nu, 'v', 'x', include_advection)
            if nu != 0:
                correction = self.diffusion_faces['v','x']
                F_yx.ravel()[correction.indices] += rho*nu*(correction.correction @ field)

        rhs_v = np.zeros((ny + 1, nx), dtype=np.float64)
        rhs_v[1:ny, :] = (F_yx[1:ny, 1:] - F_yx[1:ny, :-1]) + (F_yy[1:ny, :] - F_yy[:ny-1, :])

        # Wall arcs belong to the actual staggered volumes, including covered
        # central-face wedges. Relocate their entire RHS and volume together.
        if curved:
            if nu != 0:
                tau_w = -rho * nu * self.wall_shear_rate(u, v)
                np.add.at(rhs_u.ravel(), self.wall.u_indices, tau_w*self.tx*self.ds)
                np.add.at(rhs_v.ravel(), self.wall.v_indices, tau_w*self.ty*self.ds)
            transfer_covered_rhs(rhs_u, self.u_covered, self.u_owner)
            transfer_covered_rhs(rhs_v, self.v_covered, self.v_owner)

        # 4. Conservative Cell Merging:
        a_u = np.zeros_like(u)
        fluid_u = (geom.u_volume_fraction >= 0.3) & (geom.ax > 0.0)
        a_u[fluid_u] = rhs_u[fluid_u] / (rho * self.u_vol[fluid_u])
        for (js, is_), (jm, im) in geom.u_merge_list:
            rhs_m = rhs_u[js, is_] + rhs_u[jm, im]
            vol_m = self.u_vol_eff[js, is_]
            am = rhs_m / (rho * vol_m)
            a_u[js, is_] = am
            a_u[jm, im] = am

        a_v = np.zeros_like(v)
        fluid_v = (geom.v_volume_fraction >= 0.3) & (geom.ay > 0.0)
        a_v[fluid_v] = rhs_v[fluid_v] / (rho * self.v_vol[fluid_v])
        for (js, is_), (jm, im) in geom.v_merge_list:
            rhs_m = rhs_v[js, is_] + rhs_v[jm, im]
            vol_m = self.v_vol_eff[js, is_]
            am = rhs_m / (rho * vol_m)
            a_v[js, is_] = am
            a_v[jm, im] = am

        # Zero out solid faces to guarantee exact discrete domain momentum balance
        rhs_u[geom.ax == 0.0] = 0.0
        rhs_v[geom.ay == 0.0] = 0.0
        a_u[geom.ax == 0.0] = 0.0
        a_v[geom.ay == 0.0] = 0.0

        return a_u, a_v, rhs_u, rhs_v

    def advance_ab2_cn(self, u, v, p, dt, nu, rho, u_inlet):
        """Coupled production step, retaining pressure and advection restart history."""
        active_u = self.geom.ax[:, 1:] > 0
        active_v = self.geom.ay[1:-1, :] > 0
        nu_active = int(active_u.sum())
        def pack(uu, vv):
            return np.r_[uu[:, 1:][active_u], vv[1:-1][active_v]]
        def unpack(q):
            uu, vv = np.zeros_like(u), np.zeros_like(v)
            uu[:, 0] = u_inlet
            uu[:, 1:][active_u] = q[:nu_active]
            vv[1:-1][active_v] = q[nu_active:]
            return uu, vv
        def advection(q):
            au, av, _, _ = self.compute_momentum_rhs(*unpack(q), 0., rho, transport='centered')
            return pack(au, av)
        def diffusion(q):
            au, av, _, _ = self.compute_momentum_rhs(*unpack(q), nu, rho,
                                                     transport='centered', include_advection=False)
            return pack(au, av)
        def gradient(pressure):
            gu, gv = np.zeros_like(u), np.zeros_like(v)
            gu[:, 1:-1] = np.diff(pressure, axis=1)/(rho*self.dx)
            gu[:, -1] = -2*pressure[:, -1]/(rho*self.dx)
            gv[1:-1] = np.diff(pressure, axis=0)/(rho*self.dy)
            return pack(gu, gv)
        def divergence(q):
            uu, vv = unpack(q)
            out = np.zeros_like(p)
            self.compute_rhs(uu, vv, 1., 1., out)
            fluid = self.geom.cell_type != -1
            out[fluid] /= self.geom.volume_fraction[fluid]
            return out
        pressure_solves = []
        def project(q, step):
            us, vs = unpack(q)
            rhs = np.zeros_like(p)
            self.compute_rhs(us, vs, step, rho, rhs)
            phi = self.solve_pressure(rhs, step, rho)
            pressure_solves.append(self.last_pressure_diagnostics.copy())
            uu, vv = np.zeros_like(u), np.zeros_like(v)
            self.apply_corrector(uu, vv, us, vs, phi, step, rho, u_inlet)
            return pack(uu, vv), phi
        pressure = p if self.cn_pressure is None else self.cn_pressure
        q, half, output, history, diagnostics = ab2_cn_step(pack(u, v), pressure, dt,
            advection=advection, diffusion=diffusion, gradient=gradient, project=project,
            divergence=divergence, nu=nu, rho=rho,
            previous_advection=self.cn_advection, previous_dt=self.cn_dt)
        self.cn_pressure, self.cn_advection, self.cn_dt = half, history, dt
        # Includes midpoint startup and the discrete pressure recovery solve.
        self.last_pressure_diagnostics.update(
            iterations=sum(item['iterations'] for item in pressure_solves),
            seconds=sum(item['seconds'] for item in pressure_solves),
            pressure_solves=len(pressure_solves))
        self.last_pressure_diagnostics.update(diagnostics)
        uu, vv = unpack(q)
        return uu, vv, output

    def advance_momentum_predictor(self, u, v, dt, nu, rho, u_inlet=None, transport='upwind', method='euler'):
        """Advances provisional velocity fields using Forward Euler or AB2."""
        a_u, a_v, _, _ = self.compute_momentum_rhs(u, v, nu, rho, transport=transport)

        if method == 'ab2' and self.prev_au is not None and self.prev_av is not None:
            alpha = dt / (2 * self.prev_dt)
            u_star = u + dt * ((1 + alpha) * a_u - alpha * self.prev_au)
            v_star = v + dt * ((1 + alpha) * a_v - alpha * self.prev_av)
        else:
            u_star = u + dt * a_u
            v_star = v + dt * a_v

        self.prev_au = a_u.copy()
        self.prev_av = a_v.copy()
        self.prev_dt = dt

        self.apply_predictor_boundary(u_star, v_star, u_inlet=u_inlet)
        return u_star, v_star

    def measure_true_surface_slip(self, u, v):
        """
        Measures boundary slip velocity on the true physical cylinder circle r = R
        sampled at uniform azimuthal angles.
        Applies near-boundary velocity reconstruction to recover the physical no-slip
        profile on Cartesian stencils across the immersed boundary.
        Returns:
            mean_slip (float): area-weighted L1 mean slip velocity
            max_slip (float): Linf maximum slip velocity
            speed (np.ndarray): slip magnitude at each azimuthal point
            u_n (np.ndarray): normal velocity component
            u_t (np.ndarray): tangential velocity component
        """
        u_eval, v_eval = self.build_boundary_evaluation_fields(u, v)
        u_interp = RegularGridInterpolator((self.yu, self.xu), u_eval, bounds_error=False, fill_value=0.0)
        v_interp = RegularGridInterpolator((self.yv, self.xv), v_eval, bounds_error=False, fill_value=0.0)
        pts = np.column_stack((self.geom.surf_y, self.geom.surf_x))
        u_pts = u_interp(pts)
        v_pts = v_interp(pts)

        u_n = u_pts * self.geom.surf_nx + v_pts * self.geom.surf_ny
        u_t = -u_pts * self.geom.surf_ny + v_pts * self.geom.surf_nx
        speed = np.hypot(u_pts, v_pts)

        mean_slip = float(np.mean(speed))
        max_slip = float(np.max(speed))
        return mean_slip, max_slip, speed, u_n, u_t
