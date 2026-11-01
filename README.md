# Sharp Ghost-Cell & Cut-Cell Navier-Stokes CFD Solver

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Phase 0](https://img.shields.io/badge/Phase%200-Verified%20%26%20Complete-success.svg)]()

A high-performance, finite-volume Computational Fluid Dynamics (CFD) solver for incompressible 2D Navier–Stokes flows on staggered Marker-and-Cell (MAC) Cartesian grids. Features sharp interface boundary reconstructions (Ghost-Cell & Cut-Cell), fractional-step projection schemes, Numba JIT acceleration, and a complete Phase 0 verification and validation suite.

---

## Academic Context & Research Supervision

* **Lead Researcher & Developer:** Ali Ghazvine ([@invisce](https://github.com/invisce))
* **Research Supervisor & Advisor:** Dr. Seyed Reza Samaei
* **Institution:** Department of Mechanical Engineering, Islamic Azad University, Science and Research Branch (دانشگاه آزاد اسلامی، واحد علوم و تحقیقات)

---

## Technical Highlights & Architecture

### 1. Spatial Discretization & Sharp Curved Boundary
* **Staggered MAC Grid:** Velocities $u, v$ defined at cell faces; pressure $p$ at cell centers, preventing checkerboard pressure oscillations.
* **Ghost-Cell Method:** Circular immersed boundary characterized by an exact Signed Distance Function (SDF). Boundary conditions are enforced through quadratic-normal polynomial extrapolation across the wall interface into ghost cells.
* **True-Surface Force Quadrature:** Pressure and viscous shear traction $\tau_w$ are integrated directly on the physical curved perimeter using Cauchy stress tensor projections:
  $$F_p = -\oint p \, \mathbf{n} \, dS, \quad F_v = \oint \boldsymbol{\tau} \cdot \mathbf{n} \, dS$$
  Completely avoids staircase Cartesian surface approximations.

### 2. Time Integration & Pressure-Velocity Coupling
* **Advection:** Second-order Adams–Bashforth (AB2) explicit scheme with compact staggered stencils.
* **Diffusion:** Crank–Nicolson (CN) implicit scheme.
* **Pressure Projection:** Discrete incremental Poisson solve ensuring strict solenoidal velocity fields:
  $$\nabla \cdot \mathbf{u}^{n+1} = 0 \quad (\text{divergence} < 10^{-13})$$
* **Fast Inverses:** Accelerated using 2D Fast Fourier Transform (FFT) with boundary Schur complement corrections, supplemented by preconditioned Krylov GMRES solvers.
* **Numba JIT Acceleration:** High-performance, zero-allocation parallel kernels for CPU-bound OpenMP evaluation.

---

## Phase 0 (Stage 0) Verification & Validation Summary

The solver has undergone comprehensive numerical verification and validation governed by the rigorous 10-point Phase 0 roadmap:

| Item | Description | Numerical Outcome | Status |
| :---: | :--- | :--- | :---: |
| **0.1** | Baseline Structure & Provenance | Frozen baseline `v0.1-baseline`, 15 standardized modular folders, configuration hashing. | **PASS** |
| **0.2** | Nondimensionalization Scale | $D = \rho = U_{\text{ref}} = 1, \nu = 1/Re$. Explicit distinction between parabolic channel benchmark ($U_{\max}=1.5$) and uniform research inflow ($U_\infty=1$). | **PASS** |
| **0.3** | Sharp Interface & Surface Forces | Signed-distance circle, quadratic normal Ghost extension, analytical Couette torque error $< 0.9\%$, force replay error $< 1.4 \times 10^{-15}$. | **PASS** |
| **0.4** | Time-Stepping Order of Accuracy | AB2/CN incremental projection; measured smooth temporal convergence order: $2.000$ to $2.008$. | **PASS** |
| **0.5** | Pressure Solver Telemetry | Preconditioned GMRES with rectangular transform; true residual, iteration count, and runtime diagnostics logged. Accelerated direct FFT/Schur production path. | **PASS** |
| **0.6** | Analytical Spatial Verification | Taylor–Green vortex across 4 grids ($32^2, 64^2, 128^2, 256^2$); exact second-order convergence in $L_1, L_2, L_\infty$; $\text{div} < 10^{-10}$. | **PASS** |
| **0.7** | DFG Cylinder Flow Benchmark | **Steady Re=20:** $C_D = 5.663$, recirculation length $L_r/D = 0.8479$ (inside standard interval $0.842 - 0.852$).<br>**Unsteady Re=100:** Mean $C_D = 3.261$, $C_{l,\text{RMS}} = 0.7261$, $St = 0.3009$, mean wake length $L_r/D = 1.3186$. | **PASS** *(Qualified)* |
| **0.8** | Channel Grid Independence (GCI) | 4 fine grids: $ny \in \{246, 328, 410, 492\}$ at $\Delta t = 0.0005$. All adjacent pairs satisfy $\Delta C_D < 0.48\%$, $\Delta C_{l,\text{RMS}} < 0.69\%$, $\Delta St < 0.09\%$ (under the $2\%$ and $1\%$ GCI criteria). | **PASS** |
| **0.9** | Timestep Independence | $\Delta t \in \{0.001, 0.0005, 0.00025\}$ on grid $1320 \times 246$. Maximum relative change in $C_{l,\text{RMS}}$ is $0.00042\%$, with $\text{CFL} \le 0.21$. | **PASS** |
| **0.10** | Domain Size Sensitivity | Enlarged research domain pairs ($40 \times 41D \to 40 \times 51D$) at $h = 0.05, \Delta t = 0.002$ to $t = 300$ (16 settled cycles). $\Delta C_D = 0.88\%$, $\Delta C_{l,\text{RMS}} = 0.94\%$, $\Delta St = 0.68\%$, passing internal bounds. | **PASS** |

The full Persian report and comprehensive technical audits are available in the [`docs/`](docs/) directory:
* [Phase 0 Complete Report (گزارش کامل مرحله صفر)](docs/phase0_final_report.pdf)
* [Phase 0 Final Checklist & Evidence Audit](docs/FINAL_CHECKLIST_20261024.md)
* [Stage 0 Status & Milestone Ledger](docs/STATUS.md)

---

## Repository Structure

```
cfd-cylinder-flow-ghost-cell/
├── solver/                 # Core Navier-Stokes solvers & numerical kernels
│   ├── main.py             # Main simulation orchestrator (MAC_CylinderFlow)
│   ├── optimized_ghost_cell.py  # Sharp Ghost-Cell integrator
│   ├── numba_kernels.py    # JIT-compiled parallel flux & matrix stencils
│   ├── rectangular_inverse.py # Fast 2D FFT & Poisson inversion operators
│   ├── pressure_schur.py   # Boundary Schur complement corrections
│   ├── cut_cell_solver.py  # Cut-Cell embedded boundary implementation
│   ├── cut_cell_geom.py    # Cartesian grid cutting & aperture computation
│   ├── time_integration.py # Fractional-step AB2/CN time integrator
│   └── forces.py           # Surface stress tensor quadrature
├── tests/                  # Automated pytest verification suite
│   ├── test_numba_ghost_cell.py # Operator parity, time-order & traction tests
│   ├── test_numba_cut_cell.py   # Flux & divergence corrector benchmarks
│   └── test_consistent_operators_analytical.py # Analytical operator audits
├── verification/           # Manufactured solution verification (Taylor-Green)
├── validation/             # Benchmark test runners & GCI convergence tools
├── configs/                # Predefined simulation parameters (DFG Re20, Re100)
├── postprocessing/         # Spectral analysis, Strouhal calculation & plotting
├── docs/                   # Formal Phase 0 reports, PDF deliverable & audit logs
└── requirements.txt        # Python dependency manifest
```

---

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/invisce/cfd-cylinder-flow-ghost-cell.git
cd cfd-cylinder-flow-ghost-cell
pip install -r requirements.txt
```

### 2. Running Verification Tests
Execute the automated test suite with pytest:
```bash
pytest tests/ -v
```

### 3. Running a Cylinder Flow Simulation
Run a benchmark simulation from Python:
```python
from solver.main import MAC_CylinderFlow

# Initialize Re=100 Ghost-Cell cylinder flow simulation
sim = MAC_CylinderFlow(
    nx=660, ny=123,
    Lx=22.0, Ly=4.1,
    xc=2.0, yc=2.0, D=1.0,
    nu=0.01, rho=1.0,
    boundary='ghost_cell',
    time_scheme='ab2_cn',
    inflow='parabolic',
    verbose=True
)

# Solve for 1000 timesteps with dt=0.001
sim.solve(n_steps=1000, dt=0.001)

# Extract hydrodynamic coefficients
print(f"Time: {sim.time:.3f} s")
print(f"Diagnostics: {sim.last_step_diagnostics}")
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
