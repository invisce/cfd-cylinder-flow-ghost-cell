# Stage 0 final evidence review — 2026-10-24

Authority: the professor's original items 0.1–0.10, preserved in `D:/Coding/Python/staggered cylinder flow/reports/stage0_closure_20261014/authority_stage0.txt`. Optional appendix ideas are not acceptance gates.

**Decision:** the planned verification, benchmark, channel independence and research-domain campaigns are finished. All ten items have evidence, with the scope and qualifications below. Item 0.10 now passes the declared domain screen. This supports a Stage 0 handover with documented limitations; it does not certify the grid and timestep of the enlarged research case for production of the main paper data. `ready_for_paper1` therefore remains false for that production claim.

| Item | Evidence and disposition |
|---|---|
| 0.1 Baseline/configuration | Complete. `v0.1-baseline` resolves to `0eb7e69f530565540a1dc9bdc1be795c9103b451`; all 15 required directories exist. Archived per-run configurations and results are retained. |
| 0.2 Nondimensionalization | Complete in the audited cases: D=rho=Uref=1, nu=1/Re. Benchmark mean velocity and uniform research inflow are explicitly distinguished. |
| 0.3 Sharp interface/forces | Complete implementation and verification evidence: Ghost-Cell signed-distance circle, pressure and viscous traction integrated on the actual circle. No return to staircase force summation. |
| 0.4 Time integration | Complete implementation and smooth verification evidence: AB2 convection, CN diffusion and incremental pressure; measured smooth temporal orders approximately 2.000–2.008. |
| 0.5 Pressure solution/monitoring | Complete implementation requirement: preconditioned GMRES, residual/iteration/runtime monitoring and production-size pilot evidence. Long production here uses the direct FFT/Schur backend; it is not presented as GMRES production evidence. Multigrid is optional in the professor's wording. |
| 0.6 Verification | Complete. Taylor–Green on 32,64,128,256 grids with velocity/pressure L1,L2,Linf, approximately second-order spatial convergence, and recorded divergence below 1e-10. |
| 0.7 Cylinder validation | Complete with the already documented qualifications in ITEM_0_7_ACCEPTANCE.md. Steady Re20 and periodic Re100, forces, pressure, St and wake measurements are available. Original benchmark outlet-choice protocol is declared; exact later FeatFlow boundary identity is not claimed. Numerical reference discrepancies remain disclosed. |
| 0.8 Grid independence/GCI | Complete for the channel benchmark: four grids ny=246,328,410,492, common dt=0.0005; all adjacent drag/RMS changes below 2% and St below 1%. Richardson/GCI calculated with conditional asymptotic interpretation. No transfer of that certificate to a different geometry/inflow/resolution is justified. |
| 0.9 Timestep independence | Complete for selected channel grid 1320x246: dt=0.001,0.0005,0.00025 over a common available developed interval, five complete cycles each, maximum CFL=0.210746. This does not establish timestep independence for every future research case. |
| 0.10 Domain sensitivity | Completed and passing for selected 40x41D against 40x51D, h=0.05, dt=0.002, uniform inflow and no-slip lateral walls. Inlet/center/outlet clearances are 15D/25D; lateral clearances exceed 10D. Both runs reached t=300 and 16 stationary cycles. |

## Final domain evidence

| Quantity | 40x41D / 800x820 | 40x51D / 800x1020 | Change relative to 40x41D |
|---|---:|---:|---:|
| Mean Cd | 1.507932162833 | 1.494705649247 | 0.877129% |
| Centered Cl RMS | 0.285103197301 | 0.282428021803 | 0.938318% |
| St | 0.167482355521 | 0.166350134094 | 0.676024% |

The campaign's 2% drag/RMS and 1% St domain screens all pass. They are declared internal screens; item 0.10 itself does not prescribe those numeric percentages. The earlier 31D→41D comparison failed its RMS/St screens and remains retained, not relabelled.

Fresh read-only reconstruction on AWS checked 139,900 sequential production records per new case, dt/time consistency, finite values, full-history divergence/CFL, complete-cycle statistics and saved-summary agreement. Reconstructed metric differences were exactly zero. All 44 frozen source-file hashes were verified. Maximum production divergence was 2.1316282073e-14 for both cases; maximum CFL was 0.124930843 and 0.124199002. No simulation steps were advanced during this review.

## Scope still requiring a decision before main research data

The successful four-grid/three-timestep studies use the confined parabolic-inlet benchmark. The successful enlarged-domain study uses uniform inflow at D/h=20 and dt=0.002. Thus a single configuration has not yet been shown to combine the research domain with the professor's final-grid and selected-grid timestep requirements. Calling that research configuration independently validated would overstate the evidence. This is the applicability of items 0.8–0.9 to the final chosen configuration, not a new threshold, a demand for four additional domains, or a reason to repeat the completed benchmark studies.

No new runs, solver changes, optional investigations or new benchmark requirements are scheduled by this audit. The existing checklist can be handed over with the qualifications above. An unconditional claim that all future main-paper production is numerically certified is not supported.

## Evidence locations

- Baseline and original requirements: `D:/Coding/Python/staggered cylinder flow/` and its `reports/stage0_closure_20261014/authority_stage0.txt`.
- Verification/pressure implementation: `D:/Coding/Python/deliverables/ghost_gemini_followup_results_20261015/`.
- Four-grid results, retained failures and conditional GCI: `D:/Coding/Python/deliverables/ghost_fine_grids_results_20261018/`; `batch_status.json` is inside its compact results archive.
- Selected-grid timestep study: `D:/Coding/Python/deliverables/ghost_timestep_closure_results_20261021/`.
- Steady validation: `D:/Coding/Python/deliverables/ghost_re20_steady_results_20261021/`.
- Periodic reference comparison and accepted 0.7 limitations: this folder's `periodic_reference_audit.json`, `ITEM_0_7_REVIEW.md` and `ITEM_0_7_ACCEPTANCE.md`.
- Final domain status/configurations/summaries/manifests and fresh reconstruction: `D:/Coding/Python/deliverables/ghost_domain_fix_20261022/completed_evidence_20261024/`.

Downloaded compact domain evidence SHA256: `25aa26ff18d6e455ec9f0554f0fd079f6f3df27ac7b2aad61fbe7bfed38d3d87`.
Full remote archive: `/home/ubuntu/ghost_domain_fix_20261022/aws_results.zip`, 370829553 bytes, SHA256 `08471325f9da7ae0be2a9efafee97a3d5ac56aac0ebedd6eec8c73614c012bfd`. Full archive remains on AWS; compact evidence is downloaded and hash checked. The recorded shutdown action was poweroff requested; EC2 control-plane state was not queried.
