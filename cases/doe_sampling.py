"""
Design of Experiments (DoE) Generator for Paper 1.

Generates reproducible Latin Hypercube / Sobol sampling across parameter space:
    - Reynolds number: Re in [80, 180]
    - Defect depth ratio: d = delta/D in [0.0, 0.10]
    - Defect angular width: w = 2*phi in [10 deg, 60 deg]
    - Defect location: theta_d in [0 deg, 180 deg] (0 at upstream stagnation)

Includes:
    1. Baseline pristine cylinders (d = 0)
    2. Deterministic extreme and separation-region calibration cases
    3. Quasi-random space-filling Sobol/LHS sequence (150-200 total cases)
    4. Stratified 80/20 Train vs Holdout-Test split (prevents data leakage)
"""

import json
import numpy as np
from pathlib import Path


def generate_paper1_doe(n_total=180, seed=42, output_json=None, output_csv=None):
    """
    Generate the complete parameter manifest for Paper 1.
    
    Parameters:
        n_total     : Total number of target simulation cases (default 180, in [150, 200]).
        seed        : Random seed for exact reproducibility.
        output_json : Path to export JSON manifest.
        output_csv  : Path to export CSV manifest.
    
    Returns:
        list of dict: Every item is a complete case configuration.
    """
    rng = np.random.default_rng(seed)
    cases = []
    case_id = 0

    # 1. Baseline Pristine Cylinders (d = 0) across all nominal Re
    nominal_res = [80.0, 100.0, 120.0, 150.0, 180.0]
    for re in nominal_res:
        cases.append({
            "case_id": f"P1_{case_id:04d}",
            "case_type": "pristine_baseline",
            "Re": float(re),
            "depth_ratio": 0.0,
            "width_deg": 0.0,
            "theta_d_deg": 0.0,
            "split": "train" if re != 120.0 else "holdout_test"
        })
        case_id += 1

    # 2. Deterministic Extreme & Separation Cases
    # Separation zone is around theta_d ~ 80 to 110 deg
    det_configs = [
        # (Re, d, w_deg, theta_d_deg, split)
        (100.0, 0.10, 60.0, 0.0, "train"),      # Max defect at stagnation
        (100.0, 0.10, 60.0, 90.0, "train"),     # Max defect at shoulder
        (100.0, 0.10, 60.0, 180.0, "train"),    # Max defect at base
        (100.0, 0.05, 30.0, 85.0, "train"),     # Separation point
        (100.0, 0.05, 30.0, 105.0, "train"),    # Immediate post-separation
        (80.0, 0.10, 40.0, 90.0, "train"),
        (180.0, 0.10, 40.0, 90.0, "train"),
        (150.0, 0.025, 20.0, 45.0, "holdout_test"),
        (120.0, 0.075, 40.0, 90.0, "holdout_test"),
        (100.0, 0.025, 10.0, 0.0, "holdout_test"), # Tiny defect at stagnation
    ]
    for re, d, w, th, split in det_configs:
        cases.append({
            "case_id": f"P1_{case_id:04d}",
            "case_type": "deterministic_calibration",
            "Re": float(re),
            "depth_ratio": float(d),
            "width_deg": float(w),
            "theta_d_deg": float(th),
            "split": split
        })
        case_id += 1

    # 3. Space-filling Latin Hypercube Sampling (LHS) for the remaining slots
    n_lhs = n_total - len(cases)
    
    # Parameter bounds:
    # Re: [80.0, 180.0]
    # d : [0.01, 0.10]
    # w : [10.0, 60.0] deg
    # th: [0.0, 180.0] deg
    
    # Generate Latin Hypercube intervals
    lhs_norm = np.zeros((n_lhs, 4))
    for dim in range(4):
        # Permutation of intervals
        perm = rng.permutation(n_lhs)
        # Uniform jitter inside interval
        jitter = rng.uniform(0.0, 1.0, size=n_lhs)
        lhs_norm[:, dim] = (perm + jitter) / n_lhs

    re_samples = 80.0 + lhs_norm[:, 0] * (180.0 - 80.0)
    d_samples = 0.01 + lhs_norm[:, 1] * (0.10 - 0.01)
    w_samples = 10.0 + lhs_norm[:, 2] * (60.0 - 10.0)
    th_samples = 0.0 + lhs_norm[:, 3] * 180.0

    # Assign 80/20 train/test split randomly for LHS points
    split_indices = rng.permutation(n_lhs)
    n_test_target = int(round(0.20 * n_total))
    current_test_count = sum(1 for c in cases if c["split"] == "holdout_test")
    remaining_test_needed = max(0, n_test_target - current_test_count)

    test_mask = np.zeros(n_lhs, dtype=bool)
    test_mask[split_indices[:remaining_test_needed]] = True

    for i in range(n_lhs):
        cases.append({
            "case_id": f"P1_{case_id:04d}",
            "case_type": "space_filling_lhs",
            "Re": float(round(re_samples[i], 2)),
            "depth_ratio": float(round(d_samples[i], 4)),
            "width_deg": float(round(w_samples[i], 2)),
            "theta_d_deg": float(round(th_samples[i], 2)),
            "split": "holdout_test" if test_mask[i] else "train"
        })
        case_id += 1

    # Save to JSON if requested
    if output_json:
        p_json = Path(output_json)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump({
                "manifest_version": "1.0",
                "campaign": "Paper1_Hydrodynamic_Observability",
                "total_cases": len(cases),
                "n_train": sum(1 for c in cases if c["split"] == "train"),
                "n_test": sum(1 for c in cases if c["split"] == "holdout_test"),
                "cases": cases
            }, f, indent=2)

    # Save to CSV if requested
    if output_csv:
        p_csv = Path(output_csv)
        p_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "case_type", "Re", "depth_ratio", "width_deg", "theta_d_deg", "split"])
            writer.writeheader()
            writer.writerows(cases)

    return cases


if __name__ == "__main__":
    cases = generate_paper1_doe(
        n_total=180,
        output_json="configs/phase1_doe_manifest.json",
        output_csv="configs/phase1_doe_manifest.csv"
    )
    print(f"Generated {len(cases)} cases successfully.")
