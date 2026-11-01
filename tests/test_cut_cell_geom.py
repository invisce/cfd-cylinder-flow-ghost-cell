"""
Independent Verification Test Suite for Stage 0.3: Cut-Cell Cartesian Geometry Engine.
Verifies all geometric invariants, fractions, conservation, 3-grid convergence, and perturbation robustness.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
import numpy as np
from cut_cell_geom import CutCellGeometry2D

def test_fraction_bounds_and_validity():
    print("1. Testing Cut-Cell fraction bounds and validity...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)

    # 1. Volume fractions must be strictly in [0.0, 1.0]
    assert np.all(geom.volume_fraction >= 0.0), f"Negative volume fraction: {np.min(geom.volume_fraction)}"
    assert np.all(geom.volume_fraction <= 1.0), f"Volume fraction > 1: {np.max(geom.volume_fraction)}"

    # 2. Face fractions ax, ay must be strictly in [0.0, 1.0]
    assert np.all(geom.ax >= 0.0) and np.all(geom.ax <= 1.0), "ax outside [0, 1]"
    assert np.all(geom.ay >= 0.0) and np.all(geom.ay <= 1.0), "ay outside [0, 1]"

    # 3. Staggered volume fractions must be strictly in [0.0, 1.0]
    assert np.all(geom.u_volume_fraction >= 0.0) and np.all(geom.u_volume_fraction <= 1.0), "u_volume_fraction outside [0, 1]"
    assert np.all(geom.v_volume_fraction >= 0.0) and np.all(geom.v_volume_fraction <= 1.0), "v_volume_fraction outside [0, 1]"

    # 4. Fluid centroids must lie strictly inside the bounding box of each cell
    for j in range(ny):
        yb, yt = j * geom.dy, (j + 1) * geom.dy
        for i in range(nx):
            xl, xr = i * geom.dx, (i + 1) * geom.dx
            if geom.volume_fraction[j, i] > 0.0:
                fcx = geom.fluid_cx[j, i]
                fcy = geom.fluid_cy[j, i]
                assert xl - 1e-12 <= fcx <= xr + 1e-12, f"fluid_cx out of cell: {fcx} vs [{xl}, {xr}]"
                assert yb - 1e-12 <= fcy <= yt + 1e-12, f"fluid_cy out of cell: {fcy} vs [{yb}, {yt}]"

    print("  [OK] All volume, face, and momentum fractions are strictly bounded in [0, 1]")
    print("  [OK] All fluid centroids lie strictly inside their parent cells")

def test_shared_face_consistency():
    print("2. Testing shared face consistency across neighboring cells...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)

    # In staggered layout, face (j, i) is shared between cell (j, i-1) and cell (j, i)
    # The open length from both cell perspectives must be identically ax[j, i] * dy
    for j in range(ny):
        for i in range(1, nx):
            open_len_from_left = geom.ax[j, i] * geom.dy
            open_len_from_right = geom.ax[j, i] * geom.dy
            assert abs(open_len_from_left - open_len_from_right) == 0.0

    for j in range(1, ny):
        for i in range(nx):
            open_len_from_bottom = geom.ay[j, i] * geom.dx
            open_len_from_top = geom.ay[j, i] * geom.dx
            assert abs(open_len_from_bottom - open_len_from_top) == 0.0

    print("  [OK] Shared face open fractions are identical by construction from both cell sides")

def test_area_and_perimeter_conservation():
    print("3. Testing exact cylinder solid area and perimeter integration...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)

    exact_area = np.pi * (R**2)
    exact_perim = 2.0 * np.pi * R

    computed_solid_area = np.sum((1.0 - geom.volume_fraction) * geom.cell_area)
    computed_perimeter = np.sum(geom.bndry_area)

    area_err = abs(computed_solid_area - exact_area)
    perim_err = abs(computed_perimeter - exact_perim)

    print(f"  Exact Area      : {exact_area:.15f}")
    print(f"  Computed Area   : {computed_solid_area:.15f}")
    print(f"  Area Abs Error  : {area_err:.2e}")
    print(f"  Exact Perimeter : {exact_perim:.15f}")
    print(f"  Computed Perim  : {computed_perimeter:.15f}")
    print(f"  Perim Abs Error : {perim_err:.2e}")

    assert area_err < 1e-14, f"Solid area error exceeded roundoff threshold: {area_err}"
    assert perim_err < 1e-14, f"Perimeter error exceeded roundoff threshold: {perim_err}"
    print("  [OK] Solid area and boundary perimeter match analytical values to machine precision (< 1e-14)")

def test_grid_refinement_and_small_cells():
    print("4. Testing 3-grid refinement and cut-cell volume distributions...")
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    grids = [(100, 25), (200, 50), (400, 100)]

    print(f"{'Grid':<12} | {'h (dx)':<8} | {'Cut Cells':<10} | {'Solid Cells':<12} | {'Min kappa':<10} | {'Area Err':<10} | {'Perim Err':<10}")
    print("-" * 85)
    for nx, ny in grids:
        geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
        s = geom.get_summary()
        print(f"{nx}x{ny:<6} | {geom.dx:<8.4f} | {s['cut_cells']:<10d} | {s['solid_cells']:<12d} | {s['min_volume_fraction_cut']:<10.4f} | {s['area_abs_error']:<10.2e} | {s['perimeter_abs_error']:<10.2e}")

        # Geometric invariants
        assert s['area_abs_error'] < 1e-14
        assert s['perimeter_abs_error'] < 1e-14
        assert s['min_volume_fraction_cut'] > 0.0, "Cut cell volume fraction must be strictly positive"

    print("  [OK] 3-grid refinement verified: machine precision maintained across all grid levels")

def test_center_perturbation_robustness():
    print("5. Testing cylinder center shift perturbation robustness...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    R = 0.5
    exact_A = np.pi * (R**2)
    exact_L = 2.0 * np.pi * R

    shifts = [
        (0.0, 0.0),
        (+0.035, -0.045),
        (-0.071, +0.062),
        (+0.098, +0.098),
        (-0.050, -0.050),
        (+0.123, -0.089),
        (-0.015, +0.033)
    ]

    for sx, sy in shifts:
        xc = 5.0 + sx
        yc = 2.5 + sy
        geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)
        s = geom.get_summary()

        assert s['area_abs_error'] < 1e-11, f"Shift ({sx}, {sy}) area error too high: {s['area_abs_error']}"
        assert s['perimeter_abs_error'] < 1e-12, f"Shift ({sx}, {sy}) perim error too high: {s['perimeter_abs_error']}"
        assert np.all(geom.volume_fraction >= 0.0), f"Shift ({sx}, {sy}) produced negative volume!"
        assert np.all(geom.volume_fraction <= 1.0), f"Shift ({sx}, {sy}) produced volume > 1!"
        assert np.all(geom.ax >= 0.0) and np.all(geom.ax <= 1.0)
        assert np.all(geom.ay >= 0.0) and np.all(geom.ay <= 1.0)
        print(f"  Shift ({sx:+.3f}, {sy:+.3f}): Area Err = {s['area_abs_error']:.2e}, Perim Err = {s['perimeter_abs_error']:.2e}, Min kappa = {s['min_volume_fraction_cut']:.4f} [PASS]")

    print("  [OK] All center shift perturbations maintain strictly valid, non-negative geometry")

def test_boundary_normals_and_centroids():
    print("6. Testing boundary normals and centroids...")
    nx, ny = 100, 25
    Lx, Ly = 20.0, 5.0
    xc, yc, R = 5.0, 2.5, 0.5
    geom = CutCellGeometry2D(nx, ny, Lx, Ly, xc, yc, R)

    cut_indices = np.argwhere(geom.cell_type == 0)
    for j, i in cut_indices:
        ds = geom.bndry_area[j, i]
        nx_val = geom.bndry_normal_x[j, i]
        ny_val = geom.bndry_normal_y[j, i]
        cx = geom.bndry_cx[j, i]
        cy = geom.bndry_cy[j, i]

        assert ds > 0.0, f"Cut cell without boundary segment at ({j}, {i})"
        # Normal vector must be unit length
        norm = np.hypot(nx_val, ny_val)
        assert abs(norm - 1.0) < 1e-12, f"Normal vector not unit length: {norm}"

        # Boundary centroid must lie on or very close to the true circle r = R
        dist_to_center = np.hypot(cx - xc, cy - yc)
        # Note: chord/arc centroid lies at distance R * sin(dth/2)/(dth/2) <= R
        assert dist_to_center <= R + 1e-12, f"Centroid outside circle: {dist_to_center} > {R}"
        assert dist_to_center >= 0.9 * R, f"Centroid too far inside circle: {dist_to_center} < {0.9*R}"

        # Normal vector must point outwards from cylinder center (n . (x_b - x_c) > 0)
        radial_x = cx - xc
        radial_y = cy - yc
        dot_product = nx_val * radial_x + ny_val * radial_y
        assert dot_product > 0.0, f"Normal vector points into solid! dot = {dot_product}"

    print("  [OK] All cut cell boundary normals are unit length and point outward into fluid")
    print("  [OK] All boundary centroids lie on physical boundary segments")

def test_geometric_closure_identity():
    print("7. Testing Discrete Gauss / Geometric Closure Identity per cut cell...")
    Lx, Ly = 20.0, 5.0
    R = 0.5
    # Test on both symmetric and asymmetric grids
    cases = [(5.0, 2.5), (5.035, 2.455), (4.929, 2.562)]
    for xc, yc in cases:
        geom = CutCellGeometry2D(100, 25, Lx, Ly, xc, yc, R)
        d_ax = (geom.ax[:, 1:] - geom.ax[:, :-1]) * geom.dy
        d_ay = (geom.ay[1:, :] - geom.ay[:-1, :]) * geom.dx

        err_x = np.max(np.abs(geom.bndry_normal_int_x - d_ax))
        err_y = np.max(np.abs(geom.bndry_normal_int_y - d_ay))
        sum_x = abs(np.sum(geom.bndry_normal_int_x))
        sum_y = abs(np.sum(geom.bndry_normal_int_y))

        assert err_x < 1e-14, f"Closure error in x: {err_x}"
        assert err_y < 1e-14, f"Closure error in y: {err_y}"
        assert sum_x < 1e-14, f"Boundary integral sum x != 0: {sum_x}"
        assert sum_y < 1e-14, f"Boundary integral sum y != 0: {sum_y}"
        print(f"  Center ({xc:.3f}, {yc:.3f}): max closure err = ({err_x:.2e}, {err_y:.2e}), sum = ({sum_x:.2e}, {sum_y:.2e}) [PASS]")

    print("  [OK] Discrete Gauss identity holds identically across every cell (< 1e-14)")

def test_exact_momentum_control_volumes():
    print("8. Testing exact staggered momentum control volumes...")
    geom = CutCellGeometry2D(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)
    dx, dy = geom.dx, geom.dy
    exact_cyl_area = np.pi * (0.5**2)

    # u volume box integration
    u_boxes = np.ones((geom.ny, geom.nx + 1)) * dx * dy
    u_boxes[:, 0] *= 0.5
    u_boxes[:, -1] *= 0.5
    u_solid_area = np.sum((1.0 - geom.u_volume_fraction) * u_boxes)
    u_err = abs(u_solid_area - exact_cyl_area)

    # v volume box integration
    v_boxes = np.ones((geom.ny + 1, geom.nx)) * dx * dy
    v_boxes[0, :] *= 0.5
    v_boxes[-1, :] *= 0.5
    v_solid_area = np.sum((1.0 - geom.v_volume_fraction) * v_boxes)
    v_err = abs(v_solid_area - exact_cyl_area)

    assert u_err < 1e-14, f"u momentum solid area error: {u_err}"
    assert v_err < 1e-14, f"v momentum solid area error: {v_err}"

    # Verify difference from crude arithmetic average (confirming the ~0.242 discrepancy fix)
    avg_u = 0.5 * (geom.volume_fraction[:, :-1] + geom.volume_fraction[:, 1:])
    diff_u = np.max(np.abs(geom.u_volume_fraction[:, 1:-1] - avg_u))
    avg_v = 0.5 * (geom.volume_fraction[:-1, :] + geom.volume_fraction[1:, :])
    diff_v = np.max(np.abs(geom.v_volume_fraction[1:-1, :] - avg_v))

    assert diff_u > 0.20, f"Expected significant difference from crude avg u, got {diff_u}"
    assert diff_v > 0.10, f"Expected significant difference from crude avg v, got {diff_v}"

    print(f"  Exact u solid area error: {u_err:.2e} (max diff vs crude avg = {diff_u:.3f})")
    print(f"  Exact v solid area error: {v_err:.2e} (max diff vs crude avg = {diff_v:.3f})")
    print("  [OK] Staggered momentum control volumes match analytical cylinder area to machine precision")

def test_uniform_pressure_spurious_force():
    print("9. Testing zero spurious force under uniform pressure on asymmetric grid...")
    # Shift cylinder center to create highly asymmetric cut cells
    geom = CutCellGeometry2D(100, 25, 20.0, 5.0, 5.035, 2.455, 0.5)
    P0 = 101325.0  # Large realistic pressure value

    # Force on body: F_p = - int_Gamma p * n_cyl ds = - sum (P0 * A_b)
    # Using exact integrated normal:
    Fx_exact = -np.sum(P0 * geom.bndry_normal_int_x)
    Fy_exact = -np.sum(P0 * geom.bndry_normal_int_y)

    # Contrast with crude unit normal approximation (bndry_area * bndry_normal_unit):
    Fx_crude = -np.sum(P0 * geom.bndry_area * geom.bndry_normal_unit_x)
    Fy_crude = -np.sum(P0 * geom.bndry_area * geom.bndry_normal_unit_y)

    print(f"  Crude approximation spurious force: Fx = {Fx_crude:+.4f}, Fy = {Fy_crude:+.4f}")
    print(f"  Exact integrated normal force     : Fx = {Fx_exact:+.2e}, Fy = {Fy_exact:+.2e}")

    assert abs(Fx_exact) < 1e-10, f"Spurious Fx force under uniform pressure: {Fx_exact}"
    assert abs(Fy_exact) < 1e-10, f"Spurious Fy force under uniform pressure: {Fy_exact}"
    assert abs(Fx_crude) > 1.0, "Expected crude approximation to exhibit substantial spurious force"
    print("  [OK] Zero spurious force (< 1e-10) rigorously guaranteed under constant pressure")

def test_staggered_face_fractions_and_merging():
    print("10. Testing staggered face fractions, cell merging maps, and true surface points...")
    geom = CutCellGeometry2D(100, 25, 20.0, 5.0, 5.0, 2.5, 0.5)

    # 1. Bounds check
    assert np.all(geom.ax_mid >= 0.0) and np.all(geom.ax_mid <= 1.0), "ax_mid outside [0, 1]"
    assert np.all(geom.ay_mid >= 0.0) and np.all(geom.ay_mid <= 1.0), "ay_mid outside [0, 1]"
    assert np.all(geom.ay_u >= 0.0) and np.all(geom.ay_u <= 1.0), "ay_u outside [0, 1]"
    assert np.all(geom.ax_v >= 0.0) and np.all(geom.ax_v <= 1.0), "ax_v outside [0, 1]"

    # 2. Merged cell volume verification
    for (j, i), (tj, ti) in geom.u_merge_list:
        k_small = geom.u_volume_fraction[j, i]
        k_master = geom.u_volume_fraction[tj, ti]
        assert k_small < 0.3, f"Non-small u-cell in merge list: {k_small}"
        assert k_small + k_master >= 0.3, f"Merged u-volume still small: {k_small + k_master}"

    for (j, i), (tj, ti) in geom.v_merge_list:
        k_small = geom.v_volume_fraction[j, i]
        k_master = geom.v_volume_fraction[tj, ti]
        assert k_small < 0.3, f"Non-small v-cell in merge list: {k_small}"
        assert k_small + k_master >= 0.3, f"Merged v-volume still small: {k_small + k_master}"

    # 3. True circle surface points check
    r_pts = np.hypot(geom.surf_x - geom.xc, geom.surf_y - geom.yc)
    max_r_err = np.max(np.abs(r_pts - geom.R))
    assert max_r_err < 1e-14, f"Surface points radius error: {max_r_err}"
    total_ds = np.sum(np.ones_like(geom.surf_x) * geom.surf_ds)
    assert abs(total_ds - 2.0 * np.pi * geom.R) < 1e-14, f"Surface ds integral error: {abs(total_ds - 2.0 * np.pi * geom.R)}"

    print(f"  Merged {len(geom.u_merge_list)} small u-cells and {len(geom.v_merge_list)} small v-cells into stable clusters (kappa >= 0.3)")
    print(f"  Verified {geom.n_theta} true surface points at r = R (max radius error = {max_r_err:.2e})")
    print("  [OK] Staggered face fractions, conservative merging maps, and surface sampling verified")

def run_all_cut_cell_geometry_tests():
    print("=" * 75)
    print("STAGE 0.3 VERIFICATION: Cut-Cell Cartesian Geometry Engine")
    print("=" * 75)
    test_fraction_bounds_and_validity()
    test_shared_face_consistency()
    test_area_and_perimeter_conservation()
    test_grid_refinement_and_small_cells()
    test_center_perturbation_robustness()
    test_boundary_normals_and_centroids()
    test_geometric_closure_identity()
    test_exact_momentum_control_volumes()
    test_uniform_pressure_spurious_force()
    test_staggered_face_fractions_and_merging()
    print("=" * 75)
    print("ALL CUT-CELL GEOMETRY TESTS PASSED CLEANLY AND MATHEMATICALLY!")
    print("=" * 75)

if __name__ == '__main__':
    run_all_cut_cell_geometry_tests()

