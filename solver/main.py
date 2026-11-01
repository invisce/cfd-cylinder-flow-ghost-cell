import numpy as np
import time
import hashlib
import json
import platform
import os
import tempfile
from pathlib import Path
import scipy
import mac_engine
from mac_solver import build_poisson_matrix, calculate_mac_divergence
from forces import calculate_mac_forces
from immersed_boundary import CircularProjection
from ghost_cell import CircleSDF, GhostCellBoundary
from cut_cell_solver import CutCellBoundary, calculate_cut_cell_divergence
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

class MAC_CylinderFlow:
    """
    Textbook Marker-and-Cell (MAC) finite-volume projection solver for 2D cylinder flow.
    
    Grid layout:
      - Domain: [0, Lx] x [0, Ly] discretized into nx x ny cell-centered control volumes
      - dx = Lx / nx, dy = Ly / ny
      - Scalar pressure p, correction phi, and solid mask at cell centers (j in [0, ny-1], i in [0, nx-1])
      - Horizontal velocity u on vertical faces (ny x (nx + 1)):
          * u[:, 0] sits at x = 0 (inflow boundary face)
          * u[:, nx] sits at x = Lx (outflow boundary face)
      - Vertical velocity v on horizontal faces ((ny + 1) x nx):
          * v[0, :] sits at y = 0 (bottom wall boundary face)
          * v[ny, :] sits at y = Ly (top wall boundary face)
          
    Projection & Conservation:
      - Discrete Dirichlet outlet pressure correction (Phi=0 at distance dx/2) matches the
        Poisson operator boundary condition identically.
      - Global mass conservation (inlet flux == outlet flux) is naturally satisfied to machine
        precision without any post-projection artificial velocity scaling.
    """
    def __init__(self, nx=400, ny=100, Lx=20.0, Ly=5.0, D=1.0, xc=5.0, yc=2.5, 
                 nu=0.01, rho=1.0, U_inf=1.0, inflow='uniform', U_m=None,
                 slip_walls=False, verbose=True, boundary='immersed', transport='centered',
                 pressure_solver='direct', marker_spacing_factor=1.5, time_scheme='legacy',
                 ghost_pressure_extension='neumann', ghost_velocity_interpolation='bilinear'):
        if boundary not in ('immersed', 'staircase', 'ghost_cell', 'cut_cell') or transport not in ('centered', 'upwind'):
            raise ValueError('Choose boundary=immersed/staircase/ghost_cell/cut_cell and transport=centered/upwind')
        if pressure_solver not in ('direct', 'iterative', 'cg_dct'):
            raise ValueError("pressure_solver must be direct, iterative or cg_dct")
        if pressure_solver == 'cg_dct' and boundary != 'cut_cell':
            raise ValueError("cg_dct is supported only for cut_cell")
        if boundary == 'immersed' and pressure_solver == 'iterative':
            raise NotImplementedError(
                "pressure_solver='iterative' is currently supported for Cartesian/staircase/ghost-cell formulations. "
                "The immersed boundary projection uses the exact FFT/DCT Schur-complement direct solver."
            )
        if time_scheme not in ('legacy', 'ab2_cn'):
            raise ValueError('time_scheme must be legacy or ab2_cn')
        if time_scheme == 'ab2_cn' and (boundary not in ('cut_cell','ghost_cell') or transport != 'centered'):
            raise ValueError('ab2_cn requires cut_cell/ghost_cell and centered transport')
        self.time_scheme = time_scheme
        if ghost_pressure_extension not in ('neumann', 'quadratic') or ghost_velocity_interpolation not in ('bilinear', 'biquadratic'):
            raise ValueError('Unknown Ghost pressure extension or velocity interpolation')
        if (ghost_pressure_extension != 'neumann' or ghost_velocity_interpolation != 'bilinear') and (boundary != 'ghost_cell' or time_scheme != 'ab2_cn'):
            raise ValueError('Ghost reconstruction options require Ghost AB2/CN')
        self.ghost_pressure_extension = ghost_pressure_extension
        self.ghost_velocity_interpolation = ghost_velocity_interpolation
        self.boundary, self.transport, self.pressure_solver = boundary, transport, pressure_solver
        self.marker_spacing_factor = marker_spacing_factor
        self._previous_u_rhs = self._previous_v_rhs = None
        self._previous_dt = None
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 4 for n in (nx, ny)):
            raise ValueError('nx and ny must be integers >= 4')
        if not all(np.isfinite(x) and x > 0 for x in (Lx, Ly, D, nu, rho, U_inf)):
            raise ValueError('Domain lengths, D, nu, rho and U_inf must be finite and positive')
        if inflow not in ('uniform', 'parabolic'):
            raise ValueError('inflow must be uniform or parabolic')
        if U_m is not None and (not np.isfinite(U_m) or U_m <= 0):
            raise ValueError('U_m must be finite and positive')
        if not (D/2 < xc < Lx-D/2 and D/2 < yc < Ly-D/2):
            raise ValueError('Cylinder must lie strictly inside the channel')
        self.configuration = dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, D=D, xc=xc, yc=yc,
            nu=nu, rho=rho, U_inf=U_inf, inflow=inflow, U_m=U_m, slip_walls=slip_walls,
            boundary=boundary, transport=transport, pressure_solver=pressure_solver,
            marker_spacing_factor=marker_spacing_factor, time_scheme=time_scheme, verbose=False,
            ghost_pressure_extension=ghost_pressure_extension,
            ghost_velocity_interpolation=ghost_velocity_interpolation)
        self.run_options = {}
        self.last_step_diagnostics = {}
        self.time = 0.0
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.verbose = verbose
        
        self.nu = nu
        self.rho = rho
        self.U_inf = U_inf
        self.inflow = inflow
        self.slip_walls = slip_walls
        
        self.D = D
        self.xc = xc
        self.yc = yc
        
        # Characteristic reference velocity for non-dimensional coefficients
        if inflow == 'parabolic':
            # In Schäfer & Turek: parabolic profile with max U_m, mean U_ref = 2/3 * U_m
            self.U_m = 1.5 * U_inf if U_m is None else U_m
            self.U_ref = (2.0 / 3.0) * self.U_m
        else:
            self.U_m = U_inf
            self.U_ref = U_inf
            
        self.Re = self.U_ref * self.D / self.nu
        
        # Cell-centered coordinates and solid obstacle mask
        self.y_centers = (np.arange(self.ny) + 0.5) * self.dy
        self.x_centers = (np.arange(self.nx) + 0.5) * self.dx
        
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)
        # Normalized radius-squared to eliminate IEEE-754 roundoff differences across length scales:
        r_norm_sq = (((self.x_centers - self.xc) / (self.D / 2.0))**2)[None, :] + \
                    (((self.y_centers - self.yc) / (self.D / 2.0))**2)[:, None]
        self.mask[:, :] = r_norm_sq <= (1.0 + 1e-13)
        if not self.mask.any() or self.mask[[0, -1], :].any() or self.mask[:, [0, -1]].any():
            raise ValueError('Grid must resolve the cylinder and a fluid layer at every outer boundary')
                    
        # Allocate MAC staggered field arrays
        self.u = np.zeros((self.ny, self.nx + 1), dtype=np.float64)
        self.v = np.zeros((self.ny + 1, self.nx), dtype=np.float64)
        self.p = np.zeros((self.ny, self.nx), dtype=np.float64)
        
        # Work arrays for C++ zero-copy execution
        self.u_star = np.zeros_like(self.u)
        self.v_star = np.zeros_like(self.v)
        self.rhs = np.zeros_like(self.p)
        self.phi = np.zeros_like(self.p)
        self._zero_pressure = np.zeros_like(self.p)
        self.cc = (CutCellBoundary(nx, ny, Lx, Ly, xc, yc, D / 2.0,
                                  pressure_solver=pressure_solver, slip_walls=slip_walls)
                   if boundary == 'cut_cell' else None)
        self.cut_geom = self.cc.geom if self.cc else None

        if boundary == 'immersed':
            self.engine_mask = np.zeros_like(self.mask)
        elif boundary == 'cut_cell':
            self.engine_mask = (self.cut_geom.cell_type == -1)
        else:
            self.engine_mask = self.mask
        
        # Define baseline inlet velocity profile u_inlet[j]
        self.u_inlet = np.zeros(self.ny, dtype=np.float64)
        self.u_inlet_base = np.zeros(self.ny, dtype=np.float64)
        for j in range(self.ny):
            y = self.y_centers[j]
            if self.inflow == 'parabolic':
                val = 4.0 * self.U_m * y * (self.Ly - y) / (self.Ly**2)
            else:
                val = self.U_inf
            self.u_inlet[j] = val
            self.u_inlet_base[j] = val
                
        # Initialize velocity field
        for j in range(self.ny):
            self.u[j, :] = self.u_inlet[j]
            for i in range(self.nx):
                if self.engine_mask[j, i]:
                    self.u[j, i] = 0.0
                    self.u[j, i + 1] = 0.0
                    self.v[j, i] = 0.0
                    self.v[j + 1, i] = 0.0
        if self.cc:
            self.u[self.cut_geom.ax == 0] = 0.
            self.v[self.cut_geom.ay == 0] = 0.
                    
        # Assemble sparse discrete Laplacian
        if self.verbose:
            print(f"Building {boundary} projection ({self.nx}x{self.ny}, dx={self.dx:.4f}, dy={self.dy:.4f})...")
        t0 = time.time()
        self.ib = (CircularProjection(nx, ny, self.dx, self.dy, xc, yc, D, self.marker_spacing_factor)
                   if boundary == 'immersed' else None)
        self.gc = (GhostCellBoundary(nx, ny, self.dx, self.dy, CircleSDF(xc, yc, D / 2.0))
                   if boundary == 'ghost_cell' else None)
        self.gg = None
        if self.ib:
            self.A = None
        elif self.cc:
            self.A = self.cc.A
        elif self.gc and self.time_scheme == 'ab2_cn':
            self.A = None
        else:
            self.A = build_poisson_matrix(nx, ny, self.dx, self.dy, self.mask)
        
        self.splu = None
        self.M = None
        if self.cc:
            self.splu = self.cc.splu
            self.M = self.cc.M
        elif not self.ib and self.A is not None:
            if self.pressure_solver == 'direct':
                self.splu = spla.splu(self.A, permc_spec='MMD_AT_PLUS_A')
            elif self.pressure_solver == 'iterative':
                # Use ILU preconditioner for GMRES
                ilu = spla.spilu(self.A, drop_tol=1e-4, fill_factor=10)
                self.M = spla.LinearOperator(self.A.shape, ilu.solve)

        if boundary == 'ghost_cell':
            self.mask_u_left = self.mask[:, :-1]
            self.mask_u_right = self.mask[:, 1:]
            self.pure_fluid_u = (~self.mask_u_left) & (~self.mask_u_right)
            self.boundary_u = self.mask_u_left != self.mask_u_right
            self.pure_solid_u = self.mask_u_left & self.mask_u_right

            self.mask_v_bot = self.mask[:-1, :]
            self.mask_v_top = self.mask[1:, :]
            self.pure_fluid_v = (~self.mask_v_bot) & (~self.mask_v_top)
            self.boundary_v = self.mask_v_bot != self.mask_v_top
            self.pure_solid_v = self.mask_v_bot & self.mask_v_top

        if self.gc and self.time_scheme == 'ab2_cn':
            if pressure_solver not in ('direct', 'iterative') or slip_walls:
                raise ValueError('Ghost AB2/CN requires direct/iterative pressure and no-slip channel walls')
            from optimized_ghost_cell import GhostCellIntegrator
            self.gg = GhostCellIntegrator(self)
            self.A = self.gg.A
            initial, _ = self.gg.project(self.gg.pack(self.u,self.v),1.)
            self.u[:],self.v[:] = self.gg.unpack(initial)

        if self.ib:
            self.ib.project(self.u.copy(), self.v.copy(), self.u, self.v, self.p, 1., rho)
            self.p.fill(0.)
        elif self.cc and self.time_scheme == 'ab2_cn':
            # A timestep-independent solenoidal initial state is required for a time-order study.
            self.cc.compute_rhs(self.u, self.v, 1., rho, self.rhs)
            initial_phi = self.cc.solve_pressure(self.rhs, 1., rho)
            self.cc.apply_corrector(self.u, self.v, self.u.copy(), self.v.copy(), initial_phi,
                                    1., rho, self.u_inlet)
        if self.verbose:
            print(f"Projection ready in {time.time()-t0:.3f}s (Re={self.Re:.1f}).")


    def solve(self, n_steps=25000, dt=0.002, perturb_transient=True, perturb_duration=None, perturb_amp=0.15,
              save_artifacts=True, verbose=None, step_callback=None):
        if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
            raise ValueError('n_steps must be a positive integer')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if perturb_duration is None:
            perturb_duration = 2.5 * (self.D / self.U_ref)
        if not np.isfinite(perturb_duration) or perturb_duration <= 0 or not np.isfinite(perturb_amp) or abs(perturb_amp) >= 1:
            raise ValueError('Require perturb_duration > 0 and finite |perturb_amp| < 1')
        self.run_options = dict(perturb_transient=perturb_transient,
                                perturb_duration=perturb_duration, perturb_amp=perturb_amp)
        if verbose is None:
            verbose = self.verbose
            
        if verbose:
            print(f"Running MAC Solver on {self.nx}x{self.ny} grid for {n_steps} steps (dt={dt}s)...")
        t0 = time.time()
        
        cd_history = []
        cl_history = []
        time_history = []
        
        rhs_flat = self.rhs.ravel()
        
        start_time = self.time
        for s in range(n_steps):
            current_time = start_time + (s + 1) * dt
            if s % 100 == 0:
                # Conservative explicit advection/diffusion screen, not a stability proof.
                rate = (np.max(np.abs(self.u))/self.dx + np.max(np.abs(self.v))/self.dy
                        + (0. if self.time_scheme == 'ab2_cn' else 2*self.nu*(self.dx**-2 + self.dy**-2)))
                if not np.isfinite(rate) or dt*rate > (.5 if self.time_scheme == 'ab2_cn' else 1.):
                    raise RuntimeError(f'Explicit timestep unsafe at t={self.time:g}: dt*rate={dt*rate:g}; reduce dt')
            
            # Seed physical shedding with temporary asymmetric perturbation during startup
            if perturb_transient and self.inflow == 'uniform' and current_time <= perturb_duration:
                decay = 1.0 - (current_time / perturb_duration)
                pert = perturb_amp * np.sin(2.0 * np.pi * self.y_centers / self.Ly) * decay
                self.u_inlet[:] = self.u_inlet_base * (1.0 + pert)
            else:
                self.u_inlet[:] = self.u_inlet_base
                
            # 1. Predictor & Poisson RHS
            nonincremental = self.ib is not None or self.cc is not None or self.transport == 'centered'
            if self.gg:
                self.u[:],self.v[:],self.p[:] = self.gg.advance(self.u,self.v,self.p,dt)
            elif self.cc and self.time_scheme == 'ab2_cn':
                self.u[:], self.v[:], self.p[:] = self.cc.advance_ab2_cn(
                    self.u, self.v, self.p, dt, self.nu, self.rho, self.u_inlet)
            elif self.cc:
                method = 'ab2' if self.transport == 'centered' else 'euler'
                u_star, v_star = self.cc.advance_momentum_predictor(
                    self.u, self.v, dt, self.nu, self.rho, u_inlet=self.u_inlet,
                    transport=self.transport, method=method
                )
                self.u_star[:, :] = u_star
                self.v_star[:, :] = v_star
            else:
                mac_engine.predictor_rhs_cpp(
                    self.u, self.v, self.u_star, self.v_star, self.rhs,
                    self._zero_pressure if nonincremental else self.p, self.engine_mask,
                    self.u_inlet, self.dx, self.dy, dt, self.nu, self.rho, self.slip_walls,
                    self.transport == 'centered'
                )
                if self.gc:
                    self.gc.apply_velocity_ghost_cells(self.u_star, self.v_star, u_wall=0.0, v_wall=0.0)

                if self.transport == 'centered':
                    du = (self.u_star[:, 1:-1]-self.u[:, 1:-1])/dt
                    dv = (self.v_star[1:-1]-self.v[1:-1])/dt
                    if self._previous_dt is not None:
                        alpha = dt/(2*self._previous_dt)
                        self.u_star[:, 1:-1] += dt*alpha*(du-self._previous_u_rhs)
                        self.v_star[1:-1] += dt*alpha*(dv-self._previous_v_rhs)
                    self._previous_u_rhs, self._previous_v_rhs, self._previous_dt = du, dv, dt
                    if self.gc:
                        self.gc.apply_velocity_ghost_cells(self.u_star, self.v_star, u_wall=0.0, v_wall=0.0)
            
            # 2. Solve discrete Poisson equation for pressure correction Phi
            if self.gg:
                pass
            elif self.ib:
                fx, fy = self.ib.project(self.u_star, self.v_star, self.u, self.v, self.p, dt, self.rho)
            elif self.cc and self.time_scheme == 'ab2_cn':
                pass
            elif self.cc:
                self.cc.compute_rhs(self.u_star, self.v_star, dt, self.rho, self.rhs)
                self.phi[:] = self.cc.solve_pressure(self.rhs, dt, self.rho)
                self.cc.apply_corrector(self.u, self.v, self.u_star, self.v_star, self.phi, dt, self.rho, self.u_inlet)
            else:
                # AB2 modifies the provisional velocity, so recompute its divergence.
                self.rhs[:] = self.rho/dt*((self.u_star[:, 1:]-self.u_star[:, :-1])/self.dx
                                          +(self.v_star[1:]-self.v_star[:-1])/self.dy)
                self.rhs[self.mask] = 0.
                if self.pressure_solver == 'direct':
                    phi_flat = self.splu.solve(rhs_flat)
                else:
                    phi_flat, info = spla.gmres(self.A, rhs_flat, M=self.M, rtol=1e-8, atol=0.0)
                    if info != 0:
                        raise RuntimeError(f"GMRES pressure solver failed to converge at step {s}: info={info}")
                self.phi[:, :] = phi_flat.reshape((self.ny, self.nx))
            
                # Corrector and traction integration.
                if self.gc:
                    dt_over_rho = dt / self.rho

                    # Vectorized U corrector
                    self.u[:, 0] = self.u_inlet
                    self.u[:, 1:-1] = self.u_star[:, 1:-1]
                    self.u[:, 1:-1][self.pure_fluid_u] -= dt_over_rho * (self.phi[:, 1:] - self.phi[:, :-1])[self.pure_fluid_u] / self.dx
                    self.u[:, 1:-1][self.pure_solid_u] = 0.0
                    self.u[:, self.nx] = self.u_star[:, self.nx] + dt_over_rho * 2.0 * self.phi[:, self.nx - 1] / self.dx

                    # Vectorized V corrector
                    self.v[0, :] = 0.0
                    self.v[1:-1, :] = self.v_star[1:-1, :]
                    self.v[1:-1, :][self.pure_fluid_v] -= dt_over_rho * (self.phi[1:, :] - self.phi[:-1, :])[self.pure_fluid_v] / self.dy
                    self.v[1:-1, :][self.pure_solid_v] = 0.0
                    self.v[self.ny, :] = 0.0

                    # Update ghost cells on solid faces for next step's stencils
                    self.gc.apply_velocity_ghost_cells(self.u, self.v, u_wall=0.0, v_wall=0.0)
                    # Boundary faces between fluid and solid must remain projection-consistent (dphi/dn = 0)
                    self.u[:, 1:-1][self.boundary_u] = self.u_star[:, 1:-1][self.boundary_u]
                    self.v[1:-1, :][self.boundary_v] = self.v_star[1:-1, :][self.boundary_v]
                else:
                    mac_engine.corrector_cpp(
                        self.u, self.v, self.u_star, self.v_star, self.phi, self.mask,
                        self.u_inlet, self.dx, self.dy, dt, self.rho, self.slip_walls
                    )
            
            if not self.ib and not self.gg and not (self.cc and self.time_scheme == 'ab2_cn'):
                if nonincremental:
                    self.p[:] = self.phi
                else:
                    self.p += self.phi
            
            if self.gg:
                fx,fy,fx_p,fy_p,fx_v,fy_v = self.gg.surface_forces(self.u,self.v,self.p,self.nu*self.rho)
            elif self.ib:
                pass
            elif self.cc:
                fx, fy, fx_p, fy_p, fx_v, fy_v = self.cc.extract_true_surface_forces(
                    self.u, self.v, self.p, self.nu * self.rho
                )
            elif self.gc:
                fx, fy, fx_p, fy_p, fx_v, fy_v = self.gc.extract_true_surface_forces(
                    self.u, self.v, self.p, self.nu * self.rho
                )
            else:
                fx_p, fy_p, fx_v, fy_v = calculate_mac_forces(
                    self.u, self.v, self.p, self.mask, self.dx, self.dy, self.nu * self.rho
                )
                fx, fy = fx_p+fx_v, fy_p+fy_v
            
            denom = 0.5 * self.rho * (self.U_ref**2) * self.D
            Cd = fx / denom
            Cl = fy / denom
            if not np.isfinite(Cd) or not np.isfinite(Cl):
                raise FloatingPointError(f'Non-finite force at t={current_time:g}; results not saved')
            self.time = current_time
            if self.gg:
                div_max = float(np.max(abs(self.gg.divergence(self.gg.pack(self.u,self.v)))))
                self.last_step_diagnostics = dict(self.gg.last_diagnostics,
                    time=self.time,dt=dt,max_divergence=div_max,
                    cfl=float(dt*(np.max(abs(self.u))/self.dx+np.max(abs(self.v))/self.dy)),
                    cd=float(Cd),cl=float(Cl),cd_pressure=float(fx_p/denom),cd_viscous=float(fx_v/denom),
                    cl_pressure=float(fy_p/denom),cl_viscous=float(fy_v/denom))
                if div_max>=1e-10 or self.last_step_diagnostics['cfl']>=.5:
                    raise RuntimeError('Ghost step failed divergence/CFL gate')
            elif self.cc:
                flux_div = ((self.cut_geom.ax[:, 1:]*self.u[:, 1:]-self.cut_geom.ax[:, :-1]*self.u[:, :-1])/self.dx
                            + (self.cut_geom.ay[1:]*self.v[1:]-self.cut_geom.ay[:-1]*self.v[:-1])/self.dy)
                fluid = self.cut_geom.cell_type != -1
                div_max = float(np.max(abs(flux_div[fluid])/self.cut_geom.volume_fraction[fluid]))
                self.last_step_diagnostics = dict(self.cc.last_pressure_diagnostics,
                    time=self.time, dt=dt, max_divergence=div_max,
                    cfl=float(dt*(np.max(abs(self.u))/self.dx+np.max(abs(self.v))/self.dy)),
                    cd=float(Cd), cl=float(Cl), cd_pressure=float(fx_p/denom), cd_viscous=float(fx_v/denom),
                    cl_pressure=float(fy_p/denom), cl_viscous=float(fy_v/denom))
                if self.time_scheme == 'ab2_cn' and (div_max >= 1e-10 or self.last_step_diagnostics['cfl'] >= .5):
                    raise RuntimeError('Step failed the divergence/CFL acceptance gate')
            if step_callback is not None:
                step_callback(self, self.last_step_diagnostics.copy())

            
            cd_history.append(Cd)
            cl_history.append(Cl)
            time_history.append(current_time)
            
            if s % 1000 == 0 or s == n_steps - 1:
                if self.cc:
                    max_div, div_all, max_flux, _, _ = calculate_cut_cell_divergence(
                        self.u, self.v, self.cut_geom
                    )
                else:
                    max_div, sum_div2, count, core_div2, core_count = calculate_mac_divergence(
                        self.u, self.v, self.dx, self.dy, self.mask
                    )
                    div_all = np.sqrt(sum_div2 / count) if count > 0 else 0.0
                
                # Check natural flux balance
                in_flux = np.sum(self.u[:, 0]) * self.dy
                out_flux = np.sum(self.u[:, self.nx]) * self.dy
                mismatch = abs(out_flux - in_flux) / in_flux
                
                if verbose:
                    print(f"Step {s:5d} | Div_RMS: {div_all:.2e} | Div_Max: {max_div:.2e} | "
                          f"FluxErr: {mismatch:.2e} | Cd: {Cd:.4f} | Cl: {Cl:.4f}")

                
        elapsed = time.time() - t0
        if verbose:
            print(f"Simulation completed in {elapsed:.2f}s ({elapsed/n_steps*1000:.2f} ms/step)")
        
        t_arr = np.array(time_history)
        cd_arr = np.array(cd_history)
        cl_arr = np.array(cl_history)
        
        if save_artifacts:
            # Force history plot
            plt.figure(figsize=(11, 4.5))
            start_idx = min(5000, len(time_history) // 4)
            plt.plot(t_arr[start_idx:], cd_arr[start_idx:], label=f'Drag Coeff $C_D$ (mean={np.mean(cd_arr[start_idx:]):.3f})', color='#d62728', lw=1.5)
            plt.plot(t_arr[start_idx:], cl_arr[start_idx:], label=f'Lift Coeff $C_L$ (amp={0.5*(np.max(cl_arr[start_idx:])-np.min(cl_arr[start_idx:])):.3f})', color='#1f77b4', lw=1.5)
            plt.xlabel('Time $t$ (s)', fontsize=12)
            plt.ylabel('Force Coefficient', fontsize=12)
            plt.title(f'Confined Cylinder Aerodynamic Forces (Textbook MAC Grid {self.nx}x{self.ny}, Re={self.Re:.0f})', fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='upper right', framealpha=0.9)
            fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../figures'))
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
            fig_path = os.path.join(fig_dir, 'forces_mac_direct.png') if os.path.exists(fig_dir) else 'forces_mac_direct.png'
            data_path = os.path.join(data_dir, 'final_state_direct.npz') if os.path.exists(data_dir) else 'final_state_direct.npz'

            plt.savefig(fig_path, dpi=150)
            plt.close()
            if verbose:
                print(f"Saved {fig_path}")
            
            # Save the final fields, histories, and run metadata.
            np.savez(data_path, 
                     u=self.u, v=self.v, p=self.p, mask=self.mask, 
                     nx=self.nx, ny=self.ny, Lx=self.Lx, Ly=self.Ly,
                     dx=self.dx, dy=self.dy, 
                     cd_history=cd_arr, cl_history=cl_arr, time_history=t_arr,
                     nu=self.nu, rho=self.rho, U_inf=self.U_inf, U_ref=self.U_ref,
                     D=self.D, xc=self.xc, yc=self.yc, dt=dt, Re=self.Re,
                     inflow=self.inflow, slip_walls=self.slip_walls,
                     marker_slip=self.ib.last_slip if self.ib else np.nan,
                     metadata=json.dumps(self.metadata(dt, n_steps, perturb_transient,
                                                      perturb_duration, perturb_amp)))
            if verbose:
                print(f"Saved {data_path}")
            
        return t_arr, cd_arr, cl_arr

    def save_checkpoint(self, path, *, overwrite=False):
        """Atomically save complete sharp-interface history; preserve existing archives."""
        if not self.cc and not self.gg:
            raise ValueError('Checkpoints require Cut-Cell or Ghost-Cell AB2/CN')
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in ('u', 'v', 'p', 'phi', 'u_inlet')}
        state = self.cc if self.cc is not None else self.gg
        names = ('prev_au', 'prev_av', 'cn_advection', 'cn_pressure') if self.cc else ('previous_advection','half_pressure')
        arrays.update({k: getattr(state, k) for k in names if getattr(state, k) is not None})
        meta = dict(schema=1, configuration=self.configuration, time=self.time,
                    run_options=self.run_options, prev_dt=self.cc.prev_dt if self.cc else self.gg.previous_dt,
                    cn_dt=self.cc.cn_dt if self.cc else self.gg.previous_dt,
                    diagnostics=self.last_step_diagnostics,
                    source_sha256=self.metadata(1., 0)['source_sha256'])
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez(stream, checkpoint_metadata=json.dumps(meta), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def from_checkpoint(cls, path):
        """Reject incompatible provenance or malformed history instead of silently restarting AB2."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['checkpoint_metadata']))
            if meta.get('schema') != 1 or meta['configuration'].get('boundary') not in ('cut_cell','ghost_cell'):
                raise ValueError('Unsupported checkpoint')
            root = Path(__file__).resolve().parent
            for name, digest in meta['source_sha256'].items():
                source = (Path(mac_engine.__file__) if name == 'compiled_engine' else root/name).resolve()
                if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Checkpoint source mismatch: {name}')
            solver = cls(**meta['configuration'])
            if meta['source_sha256'] != solver.metadata(1., 0)['source_sha256']:
                raise ValueError('Incomplete checkpoint source manifest')
            for name in ('u', 'v', 'p', 'phi', 'u_inlet'):
                value = data[name]
                target = getattr(solver, name)
                if value.shape != target.shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid checkpoint field: {name}')
                target[:] = value
            solver.time = float(meta['time'])
            if not np.isfinite(solver.time) or solver.time < 0:
                raise ValueError('Invalid checkpoint time')
            solver.run_options = meta['run_options']
            solver.last_step_diagnostics = meta['diagnostics']
            if solver.gg:
                for name,shape in (('previous_advection',(solver.gg.qsize,)),('half_pressure',(len(solver.gg.pi),))):
                    value = data[name].copy() if name in data else None
                    if solver.time>0 and (value is None or value.shape!=shape or not np.isfinite(value).all()):
                        raise ValueError(f'Incomplete Ghost AB2/CN history: {name}')
                    setattr(solver.gg,name,value)
                solver.gg.previous_dt = meta['prev_dt']
                if solver.time>0 and (solver.gg.previous_dt is None or not np.isfinite(solver.gg.previous_dt) or solver.gg.previous_dt<=0):
                    raise ValueError('Invalid Ghost timestep history')
                return solver
            for name in ('prev_au', 'prev_av', 'cn_advection', 'cn_pressure'):
                value = data[name].copy() if name in data else None
                if value is not None and not np.isfinite(value).all():
                    raise ValueError(f'Invalid checkpoint history: {name}')
                setattr(solver.cc, name, value)
            solver.cc.prev_dt, solver.cc.cn_dt = meta['prev_dt'], meta['cn_dt']
            if solver.time_scheme == 'ab2_cn' and solver.time > 0:
                count = np.count_nonzero(solver.cut_geom.ax[:, 1:])+np.count_nonzero(solver.cut_geom.ay[1:-1])
                if (solver.cc.cn_advection is None or solver.cc.cn_advection.shape != (count,)
                        or solver.cc.cn_pressure is None or solver.cc.cn_pressure.shape != solver.p.shape
                        or solver.cc.cn_dt is None or not np.isfinite(solver.cc.cn_dt) or solver.cc.cn_dt <= 0):
                    raise ValueError('Incomplete AB2/CN checkpoint history')
            elif solver.time > 0:
                if (solver.cc.prev_au is None or solver.cc.prev_au.shape != solver.u.shape
                        or solver.cc.prev_av is None or solver.cc.prev_av.shape != solver.v.shape
                        or solver.cc.prev_dt is None or not np.isfinite(solver.cc.prev_dt) or solver.cc.prev_dt <= 0):
                    raise ValueError('Incomplete predictor checkpoint history')
            return solver

    def metadata(self, dt, n_steps, perturb_transient=True, perturb_duration=None, perturb_amp=.15):
        """Record inputs and the exact source/binary used, without a new tracking dependency."""
        root = Path(__file__).resolve().parent
        files = ('main.py', 'mac_solver.py', 'forces.py', 'immersed_boundary.py', 'ghost_cell.py',
                 'cut_cell_geom.py', 'cut_cell_solver.py', 'time_integration.py',
                 'wall_reconstruction.py', 'pressure_schur.py', 'staggered_wall.py', 'src/bindings.cpp', 'setup.py')
        if self.gg:
            files += ('optimized_ghost_cell.py','numba_kernels.py','rectangular_inverse.py','ghost_reconstruction.py')
        if perturb_duration is None:
            perturb_duration = 2.5 * self.D / self.U_ref
        hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
        hashes['compiled_engine'] = hashlib.sha256(Path(mac_engine.__file__).read_bytes()).hexdigest()
        backend = (('ghost_gmres_rectangle' if self.pressure_solver == 'iterative' else 'ghost_fft_boundary_schur') if self.gg else 'dct_schur' if self.ib else 'cg_dct' if self.pressure_solver == 'cg_dct'
                   else 'splu_direct' if self.pressure_solver == 'direct' else 'gmres_ilu')
        return dict(nx=self.nx, ny=self.ny, Lx=self.Lx, Ly=self.Ly, D=self.D,
                    xc=self.xc, yc=self.yc, nu=self.nu, rho=self.rho, U_ref=self.U_ref,
                    U_inf=self.U_inf, U_m=self.U_m, inflow=self.inflow, slip_walls=self.slip_walls,
                    dt=dt, n_steps=n_steps, end_time=self.time, boundary=self.boundary,
                    transport=self.transport, time_integrator=('AB2_CN_discrete_incremental' if self.time_scheme == 'ab2_cn'
                        else 'AB2' if self.transport == 'centered' else 'Euler'), time_scheme=self.time_scheme,
                    pressure_solver=self.pressure_solver, pressure_backend=backend,
                    ghost_pressure_extension=self.ghost_pressure_extension,
                    ghost_velocity_interpolation=self.ghost_velocity_interpolation,
                    pressure_preconditioner=(self.gg.pressure_inverse_info if self.gg else self.cc.pressure_preconditioner_info if self.cc else None),
                    marker_count=self.ib.n_markers if self.ib else 0,
                    omp_threads=os.environ.get('OMP_NUM_THREADS', 'default'),
                    blas_threads=os.environ.get('OPENBLAS_NUM_THREADS', 'default'),
                    perturb_transient=perturb_transient, perturb_duration=perturb_duration,
                    perturb_amp=perturb_amp, python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                    source_sha256=hashes)

if __name__ == '__main__':
    solver = MAC_CylinderFlow(nx=400, ny=100, verbose=True)
    solver.solve(n_steps=15000, dt=0.002, save_artifacts=True)
