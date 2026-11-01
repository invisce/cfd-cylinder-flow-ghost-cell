"""Verification of Discrete Pressure-Momentum Consistency across Production AB2/CN Checkpoints."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scratch'))

from diagnose_pressure_momentum_consistency import audit_production_step


def test_production_step_checkpoints():
    """Verify 3-checkpoint momentum conservation, telescoping, and disjoint partition."""
    for nx, ny in [(40, 20), (80, 40)]:
        dt = 0.005 * (40.0 / nx)
        res = audit_production_step(nx=nx, ny=ny, dt=dt)

        # 1. Disjoint partition verification
        part = res['disjoint_partition']['verification']
        assert part['sum_disjoint_categories'] == part['unique_faces_with_diff'], (
            f"Overlapping or missing categories in partition on {nx}x{ny}: "
            f"sum={part['sum_disjoint_categories']}, unique={part['unique_faces_with_diff']}"
        )

        # 2. Checkpoint 1: Predictor GMRES solve
        cp1 = res['checkpoint_1_predictor']
        assert cp1['gmres_linear_residual_linf'] < 1e-12, (
            f"GMRES predictor residual exceeds tolerance on {nx}x{ny}: {cp1['gmres_linear_residual_linf']}"
        )

        # 3. Checkpoint 2: Projection telescoping under aperture metric
        cp2 = res['checkpoint_2_projection']
        assert cp2['aperture_telescoping_defect'] < 1e-14, (
            f"Aperture telescoping defect non-zero on {nx}x{ny}: {cp2['aperture_telescoping_defect']}"
        )

        # 4. Checkpoint 3: Velocity field unaltered during pressure recovery
        cp3 = res['checkpoint_3_pressure_recovery']
        assert cp3['velocity_field_altered'] is False


def test_defect_monotonic_refinement():
    """Verify that the geometric metric defect decreases monotonically under refinement."""
    res_coarse = audit_production_step(nx=40, ny=20, dt=0.005)
    res_fine = audit_production_step(nx=80, ny=40, dt=0.0025)

    def_coarse = res_coarse['checkpoint_2_projection']['geometric_metric_defect']
    def_fine = res_fine['checkpoint_2_projection']['geometric_metric_defect']

    assert def_fine < 0.5 * def_coarse, (
        f"Geometric defect did not shrink as O(h) or faster: coarse={def_coarse}, fine={def_fine}"
    )


if __name__ == '__main__':
    test_production_step_checkpoints()
    test_defect_monotonic_refinement()
    print("test_pressure_momentum_consistency.py: ALL PRODUCTION CHECKS PASS")
