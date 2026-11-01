"""Automated regression and verification test for Adjoint-Consistent Cut-Cell Operators.

Validates:
1. Matrix symmetry of A_cons on fluid cells.
2. Algebraic projection defect elimination to machine precision.
3. Manufactured harmonic pressure accuracy: baseline achieves 1.5x-2.2x lower L2 error than consistent prototype.
4. Linear pressure gradient: baseline maintains exact 1.0000 acceleration, consistent operator exhibits >80% distortion.
5. Domain fluid volume conservation: cc.u_vol conserves exact continuous fluid volume to roundoff.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'solver'))
sys.path.insert(0, str(ROOT / 'scratch'))

from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import (
    CutCellBoundary,
    build_cut_cell_poisson_matrix,
    build_surface_pressure_quadrature,
    surface_pressure_force,
    reconstruct_pressure
)
from test_experimental_consistent_operators import (
    build_consistent_poisson_matrix,
    compare_baseline_and_consistent_operators
)
from test_lightweight_analytical_operators import (
    run_manufactured_pressure_suite,
    run_linear_pressure_gradient_suite,
    run_volume_audit_suite
)


def test_consistent_matrix_symmetry_and_defect_elimination():
    """Verify that A_cons is symmetric and eliminates the projection defect on 40x20."""
    res = compare_baseline_and_consistent_operators(nx=40, ny=20, dt=0.005)

    # 1. Symmetry on fluid cells
    assert res['matrix_symmetry_linf'] < 1e-14, (
        f"A_cons is not symmetric on fluid cells: error = {res['matrix_symmetry_linf']}"
    )

    # 2. Both operators achieve machine-zero divergence
    assert res['baseline']['max_divergence'] < 1e-12
    assert res['consistent_prototype']['max_divergence'] < 1e-12

    # 3. Defect elimination
    base_defect = abs(res['baseline']['defect_geom'])
    cons_defect = abs(res['consistent_prototype']['defect_geom'])
    assert cons_defect < 1e-14, f"Consistent operator defect not eliminated: {cons_defect}"
    assert base_defect > 1e-3, f"Baseline defect expected non-zero, got {base_defect}"


def test_manufactured_pressure_accuracy_degradation():
    """Verify that consistent operator increases potential pressure error compared to baseline."""
    res = run_manufactured_pressure_suite()
    for entry in res:
        base_L2 = entry['baseline']['field_L2_error']
        cons_L2 = entry['consistent']['field_L2_error']
        ratio = cons_L2 / base_L2
        # Consistent operator must have higher L2 error (at least 1.4x higher)
        assert ratio > 1.4, f"Grid {entry['grid']}: expected consistent error ratio > 1.4, got {ratio:.2f}"


def test_linear_pressure_gradient_distortion():
    """Verify that consistent operator distorts uniform acceleration by >80% on cut faces."""
    res = run_linear_pressure_gradient_suite()
    for entry in res:
        base_grad = entry['gradient_baseline_cut_faces']
        cons_grad = entry['gradient_consistent_cut_faces']

        # Baseline gradient must be exactly 1.0
        assert abs(base_grad['min'] - 1.0) < 1e-14
        assert abs(base_grad['max'] - 1.0) < 1e-14

        # Consistent operator must distort acceleration by >80%
        assert cons_grad['max_relative_perturbation'] > 0.80, (
            f"Grid {entry['grid']}: expected perturbation > 0.80, got {cons_grad['max_relative_perturbation']:.2f}"
        )


def test_volume_metric_conservation():
    """Verify cc.u_vol conserves exact continuous fluid volume to roundoff."""
    res = run_volume_audit_suite()
    for entry in res:
        assert entry['volume_conservation_error'] < 1e-14, (
            f"Grid {entry['grid']}: volume conservation error {entry['volume_conservation_error']} exceeds 1e-14"
        )
        # u_vol_eff must exceed u_vol due to cell merging
        assert entry['sum_production_u_vol_eff'] > entry['sum_production_u_vol']


if __name__ == '__main__':
    test_consistent_matrix_symmetry_and_defect_elimination()
    print("test_consistent_matrix_symmetry_and_defect_elimination: PASS")
    test_manufactured_pressure_accuracy_degradation()
    print("test_manufactured_pressure_accuracy_degradation: PASS")
    test_linear_pressure_gradient_distortion()
    print("test_linear_pressure_gradient_distortion: PASS")
    test_volume_metric_conservation()
    print("test_volume_metric_conservation: PASS")
    print("\nALL CONSISTENT OPERATOR TESTS PASS!")
