# Design Document: Sharp-Interface Boundary Formulation on Staggered MAC Grid

## 1. Objective
Transition the current `v0.1-baseline` (which relies on a regularized immersed boundary and staircase geometry) to a strict sharp-interface formulation. This upgrade fulfills **Stage 0.3** of the research roadmap, eliminating boundary-layer smearing and providing the boundary resolution required for the downstream defect observability and Digital Twin studies.

## 2. Architectural Comparison

### Option A: Ghost-Cell Immersed Boundary Method (GCIBM)
**Concept:**  
Identify solid grid nodes adjacent to the fluid ("ghost nodes"). Using a Signed Distance Function (SDF), project normal vectors from the ghost nodes across the true circular interface to locate corresponding image points in the fluid. Linearly or bilinearly interpolate fluid velocities to the image points and extrapolate back to the ghost nodes to enforce the Dirichlet no-slip condition $u=0, v=0$ at the true physical boundary.

* **Characteristics:**  
  * **Stencil Simplicity:** The internal fluid discretization maintains the standard Cartesian 5-point Laplacian and MAC face arrangement.
  * **Matrix Structure:** Depending on the pressure boundary condition treatment (Neumann extrapolation vs. standard fluid-cell Poisson projection with modified RHS), the pressure linear system can retain structural symmetry, though asymmetry can arise if non-symmetric interpolation stencils are embedded directly into the operator.
  * **Conservation & Oscillations:** Mass conservation is discrete-conservative on the Cartesian cells, but fluid cells cut by the boundary do not account for fractional volume. Freshly cleared cells during moving-boundary simulations can produce non-physical pressure spikes unless specialized temporal smoothing or cut-off stencils are applied.

### Option B: Cut-Cell Finite-Volume Method
**Concept:**  
Truncate Cartesian MAC control volumes intersected by the cylinder boundary. Compute exact geometric cut face apertures ($\theta_x, \theta_y \in [0, 1]$) and cell volume fractions ($\alpha \in [0, 1]$) using analytical circle-line intersection or polygonal clipping.

* **Characteristics:**  
  * **Strict Conservation:** Discrete mass and momentum conservation are formulated directly on the true fluid control volumes $\alpha_{i,j} \Delta x \Delta y$.
  * **Small-Cell Problem:** Cells with tiny volume fractions ($\alpha \ll 1$) severely restrict explicit time stepping ($\Delta t \sim \alpha \Delta x / U$) and lead to ill-conditioned linear systems unless cell-merging, cell-linking, or flux-redistribution techniques are employed.
  * **Operator Properties:** The resulting discrete divergence and gradient operators modify the discrete Poisson Laplacian $\nabla \cdot (\frac{1}{\alpha} \nabla p)$. While symmetric formulations exist (e.g., standard FV cell-centered discretizations), row scaling and irregular stencils often degrade matrix condition numbers, requiring robust iterative Krylov solvers (such as GMRES or BiCGSTAB) paired with algebraic multigrid or ILU preconditioners.

## 3. Recommended Roadmap Alignment
Rather than treating these methods as philosophical absolutes, the implementation choice should align with the project phases:

1. **Phase 1 (Fixed Cylinder Defect Observability):**  
   A Ghost-Cell method with bilinear image-point interpolation and surface pressure/vorticity extraction on the physical circle provides an immediate, efficient upgrade over regularized delta smearing without requiring a full unstructured/cut-cell mesh geometric engine.
2. **Phase 2 & Phase 5 (Moving Cylinder & VIV Digital Twin):**  
   If moving-boundary pressure oscillations in the Ghost-Cell method interfere with sensor signal processing, transitioning to a volume-conserving cut-cell or moving-ALE formulation can be evaluated using the benchmark error budgets established in Stage 0.

## 4. Discussion Point for the Advisor
**Summary Question:**  
For the fixed-cylinder benchmark and defect sensitivity mapping in Paper 1, would you prefer starting with a verified **Ghost-Cell Method** (using SDF and image-point extrapolation to capture sharp wall gradients while preserving Cartesian solver speed), or should we immediately build the full **Cut-Cell geometric clipping and cell-merging infrastructure**?
