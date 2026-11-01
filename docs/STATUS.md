# Stage0 evidence and remaining work, updated 2026-10-24

Latest review: [FINAL_CHECKLIST_20261024.md](FINAL_CHECKLIST_20261024.md). The domain campaign is completed and its 41D→51D pair passes. All ten items have evidence for the configurations stated below. The final research-domain grid/timestep certificate is not established by the channel studies, so unconditional main-paper production readiness remains unconfirmed. Historical launch and earlier failed-domain notes below are retained as provenance and are superseded by this review.

Authority: the professor's original items0.1-0.10. Optional appendices are not completion gates. All numerical work stays on AWS, one job and one thread at a time. Original archives and numerical kernels remain preserved.

| Item | Current evidence | Remaining limit |
|---|---|---|
|0.1|Baseline tag0eb7e69f530565540a1dc9bdc1be795c9103b451 and all15 required folders verified. Per-case configurations retained.|No new restructuring needed.|
|0.2|D=rho=Uref1, nu1/Re. Uniform research inflow1; parabolic benchmark Umax1.5 and mean1. Pressure/time scaling recorded.|Keep benchmark and uniform-inlet reference scales distinct.|
|0.3|Frozen sharp Ghost circle, signed distance, reconstructed true-circle pressure and viscous surface quadrature; previous analytical traction tests passed.|Implementation evidence is separate from reference accuracy.|
|0.4|AB2 convection, CN diffusion, incremental pressure. Existing coupled smooth temporal tests and TGV temporal tests showed approximately second order.|No new numerical closure introduced.|
|0.5|Preconditioned GMRES implemented and tested including1320x246 pilots, parity/restart, failure rejection. Current production histories record actual pressure residuals, iterations and wall time.|Fast production uses direct FFT/Schur, not GMRES. Older histories lacking detailed pressure telemetry remain labelled historical; counts are not fabricated.|
|0.6|Four-grid TGV velocity/pressure L1,L2,Linf evidence, orders about1.976-2.001. Recorded projection divergence below1e-10.|Reusing valid AWS evidence; no redundant rerun.|
|0.7 COMPLETE WITH LIMITATIONS|Fresh steady Re20 and stationary periodic Re100 measurements, true centered RMS, pressure and mean-wake extraction. Re20 Lr/D0.847930 falls in original0.842-0.852 interval. Official periodic pressure trace now postprocessed together with forces, including phase-conditioned pressure.|Reference differences and outlet-definition limits are disclosed. No blanket2% reference-error gate exists in the professor text. The precise later-reference outlet is not declared mathematically identical merely because numerical results are close.|
|0.8|Four channel grids246/328/410/492, commondt0.0005, complete-cycle statistics and recorded hashes; all adjacent RMS/drag/St criteria passed. GCI retained with fit limitations.|These are channel results, not a grid certificate for every research domain.|
|0.9|Selected channel grid1320x246 atdt0.001,0.0005,0.00025; common developed initial checkpoint and elapsed window. Five settled cycles each; RMS largest-to-smallest change0.000425%, maxCFL0.210746.|Completed for this selected channel case. No general time-order claim inferred solely from statistics.|
|0.10 COMPLETE FOR TESTED PAIR|40x41D→40x51D at h0.05/dt0.002 completed to t300 with16 stationary cycles each. MeanCd change0.877129%, centeredClRMS0.938318%, St0.676024%; full histories and44 source hashes freshly verified on2026-10-24.|Passes the declared internal domain screen. No-slip lateral walls and D/h20 scope retained; this is not a spatial/timestep independence certificate for the final research configuration.|

## Completed AWS queue

Completed control root: /home/ubuntu/ghost_domain_isolation_20261022. Completed pair preserved at /home/ubuntu/ghost_stage0_completion_20261021. No numerical job remained active at retrieval; no new run is queued.

The missing30x31D case at600x620 completed with commonh0.05 anddt0.002: pilot tot20,100-step seeded startup tot20.2, then production tot300 sampled aftert200. Its16 stationary cycles and reconstructed history statistics were verified. The existing domains were reused without rerunning them. This isolates height at fixed streamwise clearances, then grouped inlet/outlet changes at fixed height; it does not separate inlet from outlet or quantify interactions.

The control finished in7h18m45s within its12-hour ceiling. Source/configuration, history reconstruction, divergence/CFL and stationarity checks passed. The declared domain screen reuses existing internal scales of1% St and2% mean drag/centered RMS; it is not represented as a new professor-specified numeric threshold. See ../ghost_domain_isolation_20261022/VERIFIED_RESULT.md.

Independent shutdown ownership was transferred without interrupting the existing numerical child. The batch archives results and attempts EC2 stop on completion or failure, including archive failure. No RAM stopping guard was reintroduced. No data are published.

## Evidence locations

- ghost_fine_grids_results_20261018: verified four-grid comparison and GCI.
- ghost_timestep_closure_results_20261021: completed common-window three-timestep study.
- ghost_re20_steady_results_20261021: steady validation, source/statistics reconstruction and wake-reference audit.
- ghost_gemini_followup_results_20261015: pressure implementation/telemetry, parity, regression and TGV evidence.
- This folder: current queue receipts, plan and fresh periodic reference audit after retrieval.

ready_for_paper1 remains false. Benchmark comparison, numerical independence and overall Stage0 readiness are reported separately. No unsuccessful or unexecuted item is marked complete, and no optional new development is added as a gate.

## Targeted domain fix, launched after the completed investigation

The user requested a successful fix and specified a six-day handover deadline. A separate AWS batch at /home/ubuntu/ghost_domain_fix_20261022 now tests only new lateral heights with the frozen optimized solver. The completed40x31D case is reused;40x41D /800x820 runs first, and40x51D /800x1020 runs only if needed. No third fallback is queued. Commonh0.05,dt0.002, one thread, checkpoint handling and automatic shutdown are retained. Total computation cap40h. The baseline history, source/configuration checks, adaptive pass/fail handling and archive-failure shutdown self-tests passed before launch. Controller PID2134, watcher PID2135 and the single numerical child PID2137 were confirmed live; shutdown was armed. Item0.10 remains open pending actual results. The domain1%/2% screen is internal and is not a verbatim numerical requirement of the professor. No completed benchmark, grid or timestep study is being rerun.

## Updated0.7 assessment
The saved-field outlet check supports qualified acceptance under the original benchmark's declared-outlet protocol. See ITEM_0_7_ACCEPTANCE.md. Exact later-FeatFlow replication is not claimed. Domain study and overall readiness remain pending.


- [x] **0.7: complete with documented limitations.** Original benchmark validation with the declared outlet choice; no exact later-FeatFlow replication claim. Overall Stage0 remains incomplete while the domain study is pending.


## Domain batch result, 2026-10-22
The sequential domain batch completed normally and automatically shut down. Both domains reached t300 and16 stationary cycles. Verified changes30x21D ->40x31D: meanCd2.792911%, centeredClRMS1.878410%, St2.204085%. The predeclared domain screen did not pass for drag orSt; item0.10 remains open. Full source hashes and history statistics were verified. Evidence: ../ghost_domain_results_20261022/VERIFIED_REPORT_20261022.md. No new CFD run was started during retrieval.

