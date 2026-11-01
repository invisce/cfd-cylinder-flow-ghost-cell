"""
Ghost-Cell Immersed Boundary Method (GCIBM) for Staggered MAC Grids.

Implements sharp-interface representation for solid bodies using Signed Distance
Functions (SDF) and bilinear image-point extrapolation to enforce no-slip conditions
at the true physical boundary without regularized delta smearing.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class CircleSDF:
    """Exact Signed Distance Function for a circular cylinder."""
    def __init__(self, xc, yc, radius):
        self.xc = xc
        self.yc = yc
        self.radius = radius

    def distance(self, x, y):
        """Returns signed distance: negative inside, positive outside."""
        r = np.sqrt((x - self.xc)**2 + (y - self.yc)**2)
        return r - self.radius

    def normal(self, x, y):
        """Returns unit outward normal vector (pointing into fluid)."""
        r = np.sqrt((x - self.xc)**2 + (y - self.yc)**2)
        r_safe = np.maximum(r, 1e-14)
        nx = (x - self.xc) / r_safe
        ny = (y - self.yc) / r_safe
        return nx, ny

class GhostCellBoundary:
    """
    Sharp interface Ghost-Cell generator and operator for staggered MAC grid.
    
    Grid layout:
      - u faces: (ny, nx+1) at x = i*dx, y = (j+0.5)*dy
      - v faces: (ny+1, nx) at x = (i+0.5)*dx, y = j*dy
      - p cells: (ny, nx) at x = (i+0.5)*dx, y = (j+0.5)*dy
    """
    def __init__(self, nx, ny, dx, dy, sdf, n_theta=360):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.sdf = sdf
        self.n_theta = n_theta

        # 1. Grid coordinates
        self.xu = np.arange(nx + 1) * dx
        self.yu = (np.arange(ny) + 0.5) * dy
        self.xv = (np.arange(nx) + 0.5) * dx
        self.yv = np.arange(ny + 1) * dy
        self.xp = self.xv
        self.yp = self.yu

        # 2. Fluid vs solid classification
        XU, YU = np.meshgrid(self.xu, self.yu)
        XV, YV = np.meshgrid(self.xv, self.yv)
        XP, YP = np.meshgrid(self.xp, self.yp)

        self.dist_u = sdf.distance(XU, YU)
        self.dist_v = sdf.distance(XV, YV)
        self.dist_p = sdf.distance(XP, YP)

        self.fluid_u = self.dist_u >= 0.0
        self.fluid_v = self.dist_v >= 0.0
        self.fluid_p = self.dist_p >= 0.0

        # 3. Identify Ghost faces (solid faces with at least one fluid neighbor in 4-point stencil)
        self.ghost_u_indices = self._find_ghost_faces(self.fluid_u)
        self.ghost_v_indices = self._find_ghost_faces(self.fluid_v)

        # 4. Compute Image Points and interpolation stencils for Ghost faces
        self.u_ip_weights, self.u_ip_cells, self.u_gammas = self._build_interpolation(
            self.ghost_u_indices, self.xu, self.yu, self.dist_u, is_u=True
        )
        self.v_ip_weights, self.v_ip_cells, self.v_gammas = self._build_interpolation(
            self.ghost_v_indices, self.xv, self.yv, self.dist_v, is_u=False
        )

        # 5. Physical boundary integration points for surface force extraction
        self.theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        self.ds = sdf.radius * (2 * np.pi / n_theta)
        self.surf_x = sdf.xc + sdf.radius * np.cos(self.theta)
        self.surf_y = sdf.yc + sdf.radius * np.sin(self.theta)
        self.surf_nx = np.cos(self.theta)
        self.surf_ny = np.sin(self.theta)
        self.surf_tx = -np.sin(self.theta)
        self.surf_ty = np.cos(self.theta)

    def _find_ghost_faces(self, fluid_mask):
        """A solid face is a ghost face if it has at least one neighboring fluid face."""
        ny, nx = fluid_mask.shape
        ghost_list = []
        for j in range(ny):
            for i in range(nx):
                if not fluid_mask[j, i]:  # solid
                    has_fluid_neighbor = False
                    if i > 0 and fluid_mask[j, i - 1]: has_fluid_neighbor = True
                    if i < nx - 1 and fluid_mask[j, i + 1]: has_fluid_neighbor = True
                    if j > 0 and fluid_mask[j - 1, i]: has_fluid_neighbor = True
                    if j < ny - 1 and fluid_mask[j + 1, i]: has_fluid_neighbor = True
                    if has_fluid_neighbor:
                        ghost_list.append((j, i))
        return ghost_list

    def _build_interpolation(self, ghost_indices, x_coords, y_coords, dist_field, is_u=True):
        """Construct normal probe image points in fluid and bilinear interpolation weights."""
        weights = []
        cells = []
        gammas = []
        dx = self.dx
        dy = self.dy
        nx = len(x_coords)
        ny = len(y_coords)
        for (j, i) in ghost_indices:
            xg = x_coords[i]
            yg = y_coords[j]
            d = abs(dist_field[j, i])
            # Canonical mirror reflection: image point placed at distance d outside boundary
            # Clamped to min 0.5*h to keep probe point safely in fluid when d is small
            delta_ell = max(d, 0.5 * min(dx, dy))
            nx_val, ny_val = self.sdf.normal(xg, yg)

            # Probe into fluid from boundary intercept point:
            # x_B = xg + d*nx_val
            # x_IP = x_B + delta_ell*nx_val = xg + (d + delta_ell)*nx_val
            x_ip = xg + (d + delta_ell) * nx_val
            y_ip = yg + (d + delta_ell) * ny_val

            # Bilinear interpolation cell indices in the respective face grid
            if is_u:
                i0 = int(np.floor(x_ip / dx))
                j0 = int(np.floor((y_ip - 0.5 * dy) / dy))
                x0 = i0 * dx
                y0 = (j0 + 0.5) * dy
            else:
                i0 = int(np.floor((x_ip - 0.5 * dx) / dx))
                j0 = int(np.floor(y_ip / dy))
                x0 = (i0 + 0.5) * dx
                y0 = j0 * dy

            # Clamp within valid bounds
            i0 = max(0, min(nx - 2, i0))
            j0 = max(0, min(ny - 2, j0))

            # Bilinear weights
            rx = (x_ip - x0) / dx
            ry = (y_ip - y0) / dy
            rx = max(0.0, min(1.0, rx))
            ry = max(0.0, min(1.0, ry))

            w00 = (1.0 - rx) * (1.0 - ry)
            w10 = rx * (1.0 - ry)
            w01 = (1.0 - rx) * ry
            w11 = rx * ry

            weights.append((w00, w10, w01, w11))
            cells.append((j0, i0))
            gammas.append(d / delta_ell)

        return np.array(weights), cells, np.array(gammas)

    def apply_velocity_ghost_cells(self, u, v, u_wall=0.0, v_wall=0.0):
        """
        Enforce Dirichlet boundary conditions at true physical boundary
        by setting ghost cell velocities via image-point reflection.
        u_ghost = u_wall - gamma * (u_IP - u_wall)
        """
        # 1. Update u ghost faces
        for idx, (j, i) in enumerate(self.ghost_u_indices):
            w = self.u_ip_weights[idx]
            j0, i0 = self.u_ip_cells[idx]
            gamma = self.u_gammas[idx]
            u_ip = (w[0] * u[j0, i0] +
                    w[1] * u[j0, i0 + 1] +
                    w[2] * u[j0 + 1, i0] +
                    w[3] * u[j0 + 1, i0 + 1])
            u[j, i] = u_wall - gamma * (u_ip - u_wall)

        # 2. Update v ghost faces
        for idx, (j, i) in enumerate(self.ghost_v_indices):
            w = self.v_ip_weights[idx]
            j0, i0 = self.v_ip_cells[idx]
            gamma = self.v_gammas[idx]
            v_ip = (w[0] * v[j0, i0] +
                    w[1] * v[j0, i0 + 1] +
                    w[2] * v[j0 + 1, i0] +
                    w[3] * v[j0 + 1, i0 + 1])
            v[j, i] = v_wall - gamma * (v_ip - v_wall)

    def extract_true_surface_forces(self, u, v, p, mu):
        """
        Integrate pressure and viscous shear stress directly on the true cylinder surface
        r = R at n_theta azimuthal locations.
        """
        from scipy.interpolate import RegularGridInterpolator

        # Interpolators for pressure and velocities
        p_interp = RegularGridInterpolator((self.yp, self.xp), p, bounds_error=False, fill_value=0.0)
        u_interp = RegularGridInterpolator((self.yu, self.xu), u, bounds_error=False, fill_value=0.0)
        v_interp = RegularGridInterpolator((self.yv, self.xv), v, bounds_error=False, fill_value=0.0)

        pts = np.column_stack((self.surf_y, self.surf_x))
        p_surf = p_interp(pts)

        # Compute velocity gradient tensor at surface points using probed central differences
        eps = 1e-4 * self.dx
        pts_x_plus = np.column_stack((self.surf_y, self.surf_x + eps))
        pts_x_minus = np.column_stack((self.surf_y, self.surf_x - eps))
        pts_y_plus = np.column_stack((self.surf_y + eps, self.surf_x))
        pts_y_minus = np.column_stack((self.surf_y - eps, self.surf_x))

        du_dx = (u_interp(pts_x_plus) - u_interp(pts_x_minus)) / (2.0 * eps)
        du_dy = (u_interp(pts_y_plus) - u_interp(pts_y_minus)) / (2.0 * eps)
        dv_dx = (v_interp(pts_x_plus) - v_interp(pts_x_minus)) / (2.0 * eps)
        dv_dy = (v_interp(pts_y_plus) - v_interp(pts_y_minus)) / (2.0 * eps)

        # Full symmetric viscous stress tensor tau = mu * (grad u + grad u^T)
        tau_xx = 2.0 * mu * du_dx
        tau_yy = 2.0 * mu * dv_dy
        tau_xy = mu * (du_dy + dv_dx)

        # 1. Pressure force components on body: F_p = -oint p * n ds (n points outward into fluid)
        fx_p = -np.sum(p_surf * self.surf_nx) * self.ds
        fy_p = -np.sum(p_surf * self.surf_ny) * self.ds

        # 2. Viscous force components on body: F_v = oint tau . n ds
        fx_v = np.sum((tau_xx * self.surf_nx + tau_xy * self.surf_ny) * self.ds)
        fy_v = np.sum((tau_xy * self.surf_nx + tau_yy * self.surf_ny) * self.ds)

        fx_tot = fx_p + fx_v
        fy_tot = fy_p + fy_v

        return fx_tot, fy_tot, fx_p, fy_p, fx_v, fy_v
