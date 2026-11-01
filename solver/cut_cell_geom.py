"""
Cut-Cell / Embedded Boundary Cartesian Geometry Engine for Staggered MAC Grids.

Computes exact conservative geometric quantities for a 2D circular cylinder on a uniform MAC grid:
  1. Cell fluid volume (area) fractions: kappa[j, i] in [0, 1]
  2. Vertical face open length fractions: ax[j, i] in [0, 1] (u-faces, shape ny x (nx+1))
  3. Horizontal face open length fractions: ay[j, i] in [0, 1] (v-faces, shape (ny+1) x nx)
  4. Cell classification: 1 (regular fluid), 0 (cut cell), -1 (covered solid)
  5. Boundary segment lengths (ds[j, i]), outward unit normals, and segment centroids
  6. Fluid region centroids (fluid_cx, fluid_cy) inside each cell
  7. Staggered momentum control volume fractions for u and v

Conforms to conservative Embedded Boundary (EB) finite volume formulations (AMReX EB / Basilisk embed).
"""
import numpy as np
from scipy.integrate import quad

class CutCellGeometry2D:
    """
    2D Conservative Cut-Cell Cartesian geometry for circular obstacles on staggered MAC grids.
    """
    def __init__(self, nx, ny, Lx, Ly, xc, yc, R):
        if nx < 4 or ny < 4:
            raise ValueError("nx and ny must be >= 4")
        if Lx <= 0 or Ly <= 0 or R <= 0:
            raise ValueError("Domain dimensions and radius must be positive")
        if not (R < xc < Lx - R and R < yc < Ly - R):
            raise ValueError("Cylinder must be located strictly inside the domain boundaries")

        self.nx = int(nx)
        self.ny = int(ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.xc = float(xc)
        self.yc = float(yc)
        self.R = float(R)

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cell_area = self.dx * self.dy

        # 1. Compute face open fractions (ax for u-faces, ay for v-faces)
        self._compute_face_fractions()

        # 2. Compute cell volume fractions, fluid centroids, and cell types
        self._compute_cell_volumes_and_centroids()

        # 3. Compute boundary segments, outward normals, and boundary centroids
        self._compute_boundary_segments()

        # 4. Compute staggered momentum control volume fractions
        self._compute_staggered_volumes()

        # 5. Compute staggered face open fractions (ax_mid, ay_mid, ay_u, ax_v)
        self._compute_staggered_face_fractions()

        # 6. Compute conservative cell merging tables for small cut cells (< 0.3)
        self._compute_cell_merging_maps()

        # 7. Compute true cylinder surface sampling points (r = R)
        self._compute_true_surface_points()

    def _compute_face_fractions(self):
        """
        Computes the open fluid length fraction for every grid face.
        Vertical faces (ax): x = i*dx, y in [j*dy, (j+1)*dy]
        Horizontal faces (ay): y = j*dy, x in [i*dx, (i+1)*dx]
        """
        # ax: shape (ny, nx + 1)
        self.ax = np.ones((self.ny, self.nx + 1), dtype=np.float64)
        for i in range(self.nx + 1):
            x = i * self.dx
            dx_c = abs(x - self.xc)
            if dx_c < self.R:
                h = np.sqrt(self.R**2 - dx_c**2)
                ys1 = self.yc - h
                ys2 = self.yc + h
                for j in range(self.ny):
                    yb = j * self.dy
                    yt = (j + 1) * self.dy
                    inter_b = max(yb, ys1)
                    inter_t = min(yt, ys2)
                    if inter_t > inter_b:
                        solid_len = inter_t - inter_b
                        val = 1.0 - solid_len / self.dy
                        self.ax[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

        # ay: shape (ny + 1, nx)
        self.ay = np.ones((self.ny + 1, self.nx), dtype=np.float64)
        for j in range(self.ny + 1):
            y = j * self.dy
            dy_c = abs(y - self.yc)
            if dy_c < self.R:
                h = np.sqrt(self.R**2 - dy_c**2)
                xs1 = self.xc - h
                xs2 = self.xc + h
                for i in range(self.nx):
                    xl = i * self.dx
                    xr = (i + 1) * self.dx
                    inter_l = max(xl, xs1)
                    inter_r = min(xr, xs2)
                    if inter_r > inter_l:
                        solid_len = inter_r - inter_l
                        val = 1.0 - solid_len / self.dx
                        self.ay[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

    def _compute_cell_volumes_and_centroids(self):
        """
        Computes fluid volume fraction kappa[j, i] in [0, 1] and fluid centroid (fluid_cx, fluid_cy).
        Classifies cells into:
          cell_type =  1: regular fluid cell (kappa == 1.0)
          cell_type =  0: cut cell           (0.0 < kappa < 1.0)
          cell_type = -1: covered solid cell (kappa == 0.0)
        """
        self.volume_fraction = np.ones((self.ny, self.nx), dtype=np.float64)
        self.cell_type = np.ones((self.ny, self.nx), dtype=np.int32)
        self.fluid_cx = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.fluid_cy = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Baseline centroids at cell centers
        for j in range(self.ny):
            ym = (j + 0.5) * self.dy
            for i in range(self.nx):
                xm = (i + 0.5) * self.dx
                self.fluid_cx[j, i] = xm
                self.fluid_cy[j, i] = ym

        for j in range(self.ny):
            yb = j * self.dy
            yt = (j + 1) * self.dy
            for i in range(self.nx):
                xl = i * self.dx
                xr = (i + 1) * self.dx

                # Quick bounding box screening: closest distance from cell to cylinder center
                cx = np.clip(self.xc, xl, xr)
                cy = np.clip(self.yc, yb, yt)
                if (cx - self.xc)**2 + (cy - self.yc)**2 >= self.R**2 - 1e-13:
                    # Pure fluid cell
                    self.volume_fraction[j, i] = 1.0
                    self.cell_type[j, i] = 1
                    continue

                # Check if all four corners are strictly inside the cylinder
                corners_inside = all(
                    (x - self.xc)**2 + (y - self.yc)**2 <= self.R**2 + 1e-12
                    for x in (xl, xr) for y in (yb, yt)
                )
                if corners_inside:
                    # Pure solid cell
                    self.volume_fraction[j, i] = 0.0
                    self.cell_type[j, i] = -1
                    continue

                # Cut cell: integrate 1D solid profile analytically / quadrature
                def h_solid(x):
                    dx_c = abs(x - self.xc)
                    if dx_c >= self.R:
                        return 0.0
                    dy_c = np.sqrt(self.R**2 - dx_c**2)
                    y_bot = max(yb, self.yc - dy_c)
                    y_top = min(yt, self.yc + dy_c)
                    return max(0.0, y_top - y_bot)

                def mx_solid(x):
                    return x * h_solid(x)

                def my_solid(x):
                    dx_c = abs(x - self.xc)
                    if dx_c >= self.R:
                        return 0.0
                    dy_c = np.sqrt(self.R**2 - dx_c**2)
                    y_bot = max(yb, self.yc - dy_c)
                    y_top = min(yt, self.yc + dy_c)
                    if y_top <= y_bot:
                        return 0.0
                    return 0.5 * (y_top**2 - y_bot**2)

                pts = []
                for xb in (self.xc - self.R, self.xc + self.R):
                    if xl < xb < xr:
                        pts.append(xb)
                for yh in (yb, yt):
                    if abs(yh - self.yc) < self.R:
                        ch = np.sqrt(self.R**2 - (yh - self.yc)**2)
                        for x_cand in (self.xc - ch, self.xc + ch):
                            if xl < x_cand < xr:
                                pts.append(x_cand)

                As, _ = quad(h_solid, xl, xr, points=pts, epsabs=1e-12, epsrel=1e-10)
                Mxs, _ = quad(mx_solid, xl, xr, points=pts, epsabs=1e-12, epsrel=1e-10)
                Mys, _ = quad(my_solid, xl, xr, points=pts, epsabs=1e-12, epsrel=1e-10)

                if As >= self.cell_area - 1e-12:
                    self.volume_fraction[j, i] = 0.0
                    self.cell_type[j, i] = -1
                elif As <= 1e-12:
                    self.volume_fraction[j, i] = 1.0
                    self.cell_type[j, i] = 1
                else:
                    A_fluid = max(0.0, min(self.cell_area, self.cell_area - As))
                    self.volume_fraction[j, i] = A_fluid / self.cell_area
                    self.cell_type[j, i] = 0
                    if A_fluid > 1e-14:
                        Mx_f = 0.5 * (xr**2 - xl**2) * self.dy - Mxs
                        My_f = self.dx * 0.5 * (yt**2 - yb**2) - Mys
                        self.fluid_cx[j, i] = Mx_f / A_fluid
                        self.fluid_cy[j, i] = My_f / A_fluid

    def _compute_boundary_segments(self):
        """
        Computes the exact boundary arc length inside each cell,
        along with the outward unit normal (pointing from solid into fluid)
        and the boundary segment centroid.
        """
        self.bndry_area = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_normal_int_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_normal_int_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_normal_unit_x = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_normal_unit_y = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_cx = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.bndry_cy = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Collect all angles where the circle intersects vertical or horizontal grid lines
        angles = [0.0, 2.0 * np.pi]
        i_min = int(np.ceil((self.xc - self.R) / self.dx))
        i_max = int(np.floor((self.xc + self.R) / self.dx))
        for i in range(max(0, i_min), min(self.nx + 1, i_max + 1)):
            x = i * self.dx
            cos_val = np.clip((x - self.xc) / self.R, -1.0, 1.0)
            th1 = np.arccos(cos_val)
            angles.extend([th1, 2.0 * np.pi - th1])

        j_min = int(np.ceil((self.yc - self.R) / self.dy))
        j_max = int(np.floor((self.yc + self.R) / self.dy))
        for j in range(max(0, j_min), min(self.ny + 1, j_max + 1)):
            y = j * self.dy
            sin_val = np.clip((y - self.yc) / self.R, -1.0, 1.0)
            alpha = np.arcsin(sin_val)
            th1 = alpha if alpha >= 0 else 2.0 * np.pi + alpha
            th2 = np.pi - alpha
            if th2 < 0:
                th2 += 2.0 * np.pi
            angles.extend([th1, th2])

        angles = np.array(sorted(list(set(angles))))
        keep = [angles[0]]
        for a in angles[1:]:
            if a - keep[-1] > 1e-12:
                keep.append(a)
        if 2.0 * np.pi - keep[-1] < 1e-12:
            keep[-1] = 2.0 * np.pi
        else:
            keep.append(2.0 * np.pi)
        angles = np.array(keep)

        # Integrate arc segments on each sub-interval
        for k in range(len(angles) - 1):
            th_a = angles[k]
            th_b = angles[k + 1]
            dth = th_b - th_a
            if dth < 1e-13:
                continue
            th_mid = 0.5 * (th_a + th_b)
            x_mid = self.xc + self.R * np.cos(th_mid)
            y_mid = self.yc + self.R * np.sin(th_mid)
            i = int(np.floor(x_mid / self.dx))
            j = int(np.floor(y_mid / self.dy))
            if 0 <= i < self.nx and 0 <= j < self.ny:
                seg_len = self.R * dth
                int_nx = self.R * (np.sin(th_b) - np.sin(th_a))
                int_ny = -self.R * (np.cos(th_b) - np.cos(th_a))
                int_cx = self.R * (self.xc * dth + int_nx)
                int_cy = self.R * (self.yc * dth + int_ny)

                self.bndry_area[j, i] += seg_len
                self.bndry_normal_int_x[j, i] += int_nx
                self.bndry_normal_int_y[j, i] += int_ny
                self.bndry_cx[j, i] += int_cx
                self.bndry_cy[j, i] += int_cy

        cut_mask = self.bndry_area > 0
        self.bndry_cx[cut_mask] /= self.bndry_area[cut_mask]
        self.bndry_cy[cut_mask] /= self.bndry_area[cut_mask]
        norms = np.hypot(self.bndry_normal_int_x[cut_mask], self.bndry_normal_int_y[cut_mask])
        self.bndry_normal_unit_x[cut_mask] = self.bndry_normal_int_x[cut_mask] / norms
        self.bndry_normal_unit_y[cut_mask] = self.bndry_normal_int_y[cut_mask] / norms

        # Backward compatibility aliases
        self.bndry_normal_x = self.bndry_normal_unit_x
        self.bndry_normal_y = self.bndry_normal_unit_y

    def _compute_staggered_volumes(self):
        """
        Computes exact momentum control volume fractions for staggered u and v faces:
        u_volume_fraction[j, i]: exact volume fraction of u control volume [x_{i-0.5}, x_{i+0.5}] x [y_j, y_{j+1}]
        v_volume_fraction[j, i]: exact volume fraction of v control volume [x_i, x_{i+1}] x [y_{j-0.5}, y_{j+0.5}]
        """
        # 1. u control volumes: shape (ny, nx + 1)
        self.u_volume_fraction = np.ones((self.ny, self.nx + 1), dtype=np.float64)
        for i in range(self.nx + 1):
            xl = (i - 0.5) * self.dx
            xr = (i + 0.5) * self.dx
            xl_cl = max(0.0, xl)
            xr_cl = min(self.Lx, xr)
            vol_box = (xr_cl - xl_cl) * self.dy
            if vol_box <= 0:
                continue

            for j in range(self.ny):
                yb = j * self.dy
                yt = (j + 1) * self.dy
                cx = np.clip(self.xc, xl_cl, xr_cl)
                cy = np.clip(self.yc, yb, yt)
                if (cx - self.xc)**2 + (cy - self.yc)**2 >= self.R**2 - 1e-13:
                    self.u_volume_fraction[j, i] = 1.0
                    continue
                corners_inside = all(
                    (x - self.xc)**2 + (y - self.yc)**2 <= self.R**2 + 1e-12
                    for x in (xl_cl, xr_cl) for y in (yb, yt)
                )
                if corners_inside:
                    self.u_volume_fraction[j, i] = 0.0
                    continue

                def h_solid_u(x):
                    dx_c = abs(x - self.xc)
                    if dx_c >= self.R:
                        return 0.0
                    dy_c = np.sqrt(self.R**2 - dx_c**2)
                    return max(0.0, min(yt, self.yc + dy_c) - max(yb, self.yc - dy_c))

                pts = []
                for xb in (self.xc - self.R, self.xc + self.R):
                    if xl_cl < xb < xr_cl:
                        pts.append(xb)
                for yh in (yb, yt):
                    if abs(yh - self.yc) < self.R:
                        ch = np.sqrt(self.R**2 - (yh - self.yc)**2)
                        for x_cand in (self.xc - ch, self.xc + ch):
                            if xl_cl < x_cand < xr_cl:
                                pts.append(x_cand)

                As, _ = quad(h_solid_u, xl_cl, xr_cl, points=pts, epsabs=1e-12, epsrel=1e-10)
                self.u_volume_fraction[j, i] = max(0.0, min(1.0, 1.0 - As / vol_box))

        # 2. v control volumes: shape (ny + 1, nx)
        self.v_volume_fraction = np.ones((self.ny + 1, self.nx), dtype=np.float64)
        for j in range(self.ny + 1):
            yb = (j - 0.5) * self.dy
            yt = (j + 0.5) * self.dy
            yb_cl = max(0.0, yb)
            yt_cl = min(self.Ly, yt)
            vol_box = self.dx * (yt_cl - yb_cl)
            if vol_box <= 0:
                continue

            for i in range(self.nx):
                xl = i * self.dx
                xr = (i + 1) * self.dx
                cx = np.clip(self.xc, xl, xr)
                cy = np.clip(self.yc, yb_cl, yt_cl)
                if (cx - self.xc)**2 + (cy - self.yc)**2 >= self.R**2 - 1e-13:
                    self.v_volume_fraction[j, i] = 1.0
                    continue
                corners_inside = all(
                    (x - self.xc)**2 + (y - self.yc)**2 <= self.R**2 + 1e-12
                    for x in (xl, xr) for y in (yb_cl, yt_cl)
                )
                if corners_inside:
                    self.v_volume_fraction[j, i] = 0.0
                    continue

                def h_solid_v(x):
                    dx_c = abs(x - self.xc)
                    if dx_c >= self.R:
                        return 0.0
                    dy_c = np.sqrt(self.R**2 - dx_c**2)
                    return max(0.0, min(yt_cl, self.yc + dy_c) - max(yb_cl, self.yc - dy_c))

                pts = []
                for xb in (self.xc - self.R, self.xc + self.R):
                    if xl < xb < xr:
                        pts.append(xb)
                for yh in (yb_cl, yt_cl):
                    if abs(yh - self.yc) < self.R:
                        ch = np.sqrt(self.R**2 - (yh - self.yc)**2)
                        for x_cand in (self.xc - ch, self.xc + ch):
                            if xl < x_cand < xr:
                                pts.append(x_cand)

                As, _ = quad(h_solid_v, xl, xr, points=pts, epsabs=1e-12, epsrel=1e-10)
                self.v_volume_fraction[j, i] = max(0.0, min(1.0, 1.0 - As / vol_box))

    def _compute_staggered_face_fractions(self):
        """
        Computes open fluid length fractions for staggered momentum control volume faces:
          1. ax_mid: vertical faces at cell centerlines x = (i+0.5)*dx, y in [j*dy, (j+1)*dy]
                     Shared by u[j, i] (east) and u[j, i+1] (west). Shape: (ny, nx)
          2. ay_mid: horizontal faces at cell centerlines y = (j+0.5)*dy, x in [i*dx, (i+1)*dx]
                     Shared by v[j, i] (north) and v[j+1, i] (south). Shape: (ny, nx)
          3. ay_u  : horizontal faces for u-volumes at y = j*dy, x in [(i-0.5)*dx, (i+0.5)*dx]
                     Shared by u[j-1, i] (north) and u[j, i] (south). Shape: (ny+1, nx+1)
          4. ax_v  : vertical faces for v-volumes at x = i*dx, y in [(j-0.5)*dy, (j+0.5)*dy]
                     Shared by v[j, i-1] (east) and v[j, i] (west). Shape: (ny+1, nx+1)
        """
        # 1. ax_mid: shape (ny, nx)
        self.ax_mid = np.ones((self.ny, self.nx), dtype=np.float64)
        for i in range(self.nx):
            x = (i + 0.5) * self.dx
            dxc = abs(x - self.xc)
            if dxc < self.R:
                h = np.sqrt(self.R**2 - dxc**2)
                ys1, ys2 = self.yc - h, self.yc + h
                for j in range(self.ny):
                    yb, yt = j * self.dy, (j + 1) * self.dy
                    inter = max(0.0, min(yt, ys2) - max(yb, ys1))
                    val = 1.0 - inter / self.dy
                    self.ax_mid[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

        # 2. ay_mid: shape (ny, nx)
        self.ay_mid = np.ones((self.ny, self.nx), dtype=np.float64)
        for j in range(self.ny):
            y = (j + 0.5) * self.dy
            dyc = abs(y - self.yc)
            if dyc < self.R:
                h = np.sqrt(self.R**2 - dyc**2)
                xs1, xs2 = self.xc - h, self.xc + h
                for i in range(self.nx):
                    xl, xr = i * self.dx, (i + 1) * self.dx
                    inter = max(0.0, min(xr, xs2) - max(xl, xs1))
                    val = 1.0 - inter / self.dx
                    self.ay_mid[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

        # 3. ay_u: shape (ny + 1, nx + 1)
        self.ay_u = np.ones((self.ny + 1, self.nx + 1), dtype=np.float64)
        for j in range(self.ny + 1):
            y = j * self.dy
            dyc = abs(y - self.yc)
            if dyc < self.R:
                h = np.sqrt(self.R**2 - dyc**2)
                xs1, xs2 = self.xc - h, self.xc + h
                for i in range(self.nx + 1):
                    xl = max(0.0, (i - 0.5) * self.dx)
                    xr = min(self.Lx, (i + 0.5) * self.dx)
                    if xr > xl:
                        inter = max(0.0, min(xr, xs2) - max(xl, xs1))
                        val = 1.0 - inter / (xr - xl)
                        self.ay_u[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

        # 4. ax_v: shape (ny + 1, nx + 1)
        self.ax_v = np.ones((self.ny + 1, self.nx + 1), dtype=np.float64)
        for i in range(self.nx + 1):
            x = i * self.dx
            dxc = abs(x - self.xc)
            if dxc < self.R:
                h = np.sqrt(self.R**2 - dxc**2)
                ys1, ys2 = self.yc - h, self.yc + h
                for j in range(self.ny + 1):
                    yb = max(0.0, (j - 0.5) * self.dy)
                    yt = min(self.Ly, (j + 0.5) * self.dy)
                    if yt > yb:
                        inter = max(0.0, min(yt, ys2) - max(yb, ys1))
                        val = 1.0 - inter / (yt - yb)
                        self.ax_v[j, i] = 0.0 if val < 1e-12 else (1.0 if val > 1.0 - 1e-12 else val)

    def _compute_cell_merging_maps(self, kappa_thresh=0.3):
        """
        Precomputes Conservative Cell Merging tables for small cut momentum cells:
        Any cell with 0 < kappa < kappa_thresh is merged into its master fluid neighbor
        along the outward boundary normal into the fluid.
        """
        # u-cell merging: only active open faces with fluid flow (ax > 0)
        self.u_merge_list = []
        for j in range(self.ny):
            for i in range(1, self.nx):
                if self.ax[j, i] > 0.0:
                    k = self.u_volume_fraction[j, i]
                    if 0.0 < k < kappa_thresh:
                        x = i * self.dx
                        y = (j + 0.5) * self.dy
                        r_x = x - self.xc
                        r_y = y - self.yc
                        r = np.hypot(r_x, r_y)
                        unx = r_x / r if r > 0 else 1.0
                        uny = r_y / r if r > 0 else 0.0
                        if abs(unx) >= abs(uny):
                            di = 1 if unx > 0 else -1
                            dj = 0
                        else:
                            di = 0
                            dj = 1 if uny > 0 else -1
                        ti, tj = i + di, j + dj
                        while 0 <= tj < self.ny and 0 <= ti < self.nx and self.u_volume_fraction[tj, ti] < kappa_thresh:
                            ti += di
                            tj += dj
                        if 0 <= tj < self.ny and 0 <= ti < self.nx:
                            self.u_merge_list.append(((j, i), (tj, ti)))

        # v-cell merging: only active open faces with fluid flow (ay > 0)
        self.v_merge_list = []
        for j in range(1, self.ny):
            for i in range(self.nx):
                if self.ay[j, i] > 0.0:
                    k = self.v_volume_fraction[j, i]
                    if 0.0 < k < kappa_thresh:
                        x = (i + 0.5) * self.dx
                        y = j * self.dy
                        r_x = x - self.xc
                        r_y = y - self.yc
                        r = np.hypot(r_x, r_y)
                        unx = r_x / r if r > 0 else 1.0
                        uny = r_y / r if r > 0 else 0.0
                        if abs(unx) >= abs(uny):
                            di = 1 if unx > 0 else -1
                            dj = 0
                        else:
                            di = 0
                            dj = 1 if uny > 0 else -1
                        ti, tj = i + di, j + dj
                        while 0 <= tj <= self.ny and 0 <= ti < self.nx and self.v_volume_fraction[tj, ti] < kappa_thresh:
                            ti += di
                            tj += dj
                        if 0 <= tj <= self.ny and 0 <= ti < self.nx:
                            self.v_merge_list.append(((j, i), (tj, ti)))

    def _compute_true_surface_points(self, n_theta=360):
        """
        Precomputes uniform sampling points and arc lengths on the true physical circle r = R.
        Enables identical, unbiased boundary slip and force evaluation across all solvers.
        """
        self.n_theta = int(n_theta)
        self.surf_theta = np.linspace(0.0, 2.0 * np.pi, self.n_theta, endpoint=False)
        self.surf_x = self.xc + self.R * np.cos(self.surf_theta)
        self.surf_y = self.yc + self.R * np.sin(self.surf_theta)
        self.surf_nx = np.cos(self.surf_theta)
        self.surf_ny = np.sin(self.surf_theta)
        self.surf_ds = 2.0 * np.pi * self.R / self.n_theta

    def get_summary(self):
        """Returns a structured dictionary of geometric metrics and validation errors."""
        exact_area = np.pi * (self.R**2)
        exact_perim = 2.0 * np.pi * self.R
        total_solid_area = np.sum((1.0 - self.volume_fraction) * self.cell_area)
        total_bndry_len = np.sum(self.bndry_area)

        cut_kappas = self.volume_fraction[self.cell_type == 0]
        min_kappa = float(np.min(cut_kappas)) if len(cut_kappas) > 0 else 1.0
        max_kappa = float(np.max(cut_kappas)) if len(cut_kappas) > 0 else 1.0

        return {
            'grid': (self.nx, self.ny),
            'dx_dy': (self.dx, self.dy),
            'exact_solid_area': exact_area,
            'computed_solid_area': float(total_solid_area),
            'area_abs_error': float(abs(total_solid_area - exact_area)),
            'area_rel_error': float(abs(total_solid_area - exact_area) / exact_area),
            'exact_perimeter': exact_perim,
            'computed_perimeter': float(total_bndry_len),
            'perimeter_abs_error': float(abs(total_bndry_len - exact_perim)),
            'fluid_cells': int(np.sum(self.cell_type == 1)),
            'cut_cells': int(np.sum(self.cell_type == 0)),
            'solid_cells': int(np.sum(self.cell_type == -1)),
            'min_volume_fraction_cut': min_kappa,
            'max_volume_fraction_cut': max_kappa
        }
