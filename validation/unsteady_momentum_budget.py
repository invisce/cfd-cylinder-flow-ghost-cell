"""One-phase unsteady CV diagnostic; never use a steady balance for periodic flow.

For a fixed body: F_body = outer traction - outward momentum flux - dM_fluid/dt.
The outer quadrature is independent of the body-force extraction. Storage uses
the solver's geometric volume weights, so this is not independent geometry proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.run_stage0_case import preflight, MAC_CylinderFlow
from validation.steady_momentum_budget import outer_forces


def fluid_momentum(u, v, volume_u, volume_v, rho):
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('Require positive finite density')
    for field, volume in ((u,volume_u),(v,volume_v)):
        if (field.shape != volume.shape or not np.isfinite(field).all()
                or not np.isfinite(volume).all() or np.any(volume < 0)):
            raise ValueError('Invalid velocity or volume quadrature')
    return rho*np.array([np.sum(u*volume_u), np.sum(v*volume_v)])


def storage_corrected_force(outer_force, momentum_before, momentum_after, half_width):
    values = np.array([outer_force, momentum_before, momentum_after], float)
    if values.shape != (3,2) or not np.isfinite(values).all() or not np.isfinite(half_width) or half_width <= 0:
        raise ValueError('Invalid centered momentum observations')
    return values[0]-(values[2]-values[1])/(2*half_width)


def boundary_momentum(u, dx, dy, rho):
    """M=rho*integral x(u.n), y(u.n) for fixed impermeable body/channel walls.

    This divergence-theorem route avoids interior volume quadrature. Its
    remaining independent approximation is the midpoint outer-face moment.
    """
    if u.ndim != 2 or not np.isfinite(u).all() or not np.isfinite([dx,dy,rho]).all() or min(dx,dy,rho) <= 0:
        raise ValueError('Invalid outer mass-flux observations')
    length = (u.shape[1]-1)*dx
    y = (np.arange(u.shape[0])+.5)*dy
    return rho*np.array([length*np.sum(u[:,-1])*dy, np.sum(y*(u[:,-1]-u[:,0]))*dy])


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    summaries = sorted(folder.glob('summary-*.json'))
    if not summaries:
        raise ValueError('Require a completed run')
    summary = json.loads(summaries[-1].read_text())
    case = manifest['configuration']
    if (case['kind'] != 'periodic' or not summary['stationary']
            or summary['max_divergence'] >= 1e-10 or summary['max_cfl'] >= .5
            or case['solver'].get('slip_walls',False)):
        raise ValueError('Require a numerically resolved, stationary periodic channel case')
    estimate = preflight(dict(case, kind='diagnostic', end_time=4*case['dt'], sample_start=0))
    if estimate['heavy'] or estimate['available_mib'] < estimate['estimated_peak_mib']+256:
        raise PermissionError('This probe requires separate heavy-run readiness or additional free RAM')
    checkpoint = folder/'checkpoint.npz'
    original_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = dict(checkpoint_sha256=original_digest, probes=[], ready_for_paper1=False)
    started = time.perf_counter()
    process = psutil.Process()
    for subdivision in (1,2):
        solver = MAC_CylinderFlow.from_checkpoint(checkpoint)
        if (solver.metadata(1.,0)['source_sha256'] != manifest['metadata']['source_sha256']
                or not np.isclose(solver.time,case['end_time'],atol=1e-10,rtol=0)):
            raise ValueError('Checkpoint is not the matching completed state')
        h = case['dt']/subdivision
        moments = [fluid_momentum(solver.u,solver.v,solver.cc.u_vol,solver.cc.v_vol,solver.rho)]
        boundary_moments = [boundary_momentum(solver.u,solver.dx,solver.dy,solver.rho)]
        center = {}
        def record(s, diagnostics):
            moments.append(fluid_momentum(s.u,s.v,s.cc.u_vol,s.cc.v_vol,s.rho))
            boundary_moments.append(boundary_momentum(s.u,s.dx,s.dy,s.rho))
            if len(moments)-1 == subdivision:
                terms = outer_forces(s.u,s.v,s.p,s.dx,s.dy,s.rho*s.nu,s.rho)
                center.update(time=s.time, outer_force=terms['force'],
                    terms={k:value.tolist() for k,value in terms.items()},
                    surface_force=.5*s.rho*s.U_ref**2*s.D*np.array([diagnostics['cd'],diagnostics['cl']]))
            if (time.perf_counter()-started > 180 or process.memory_info().rss > 512*2**20
                    or psutil.virtual_memory().available < 256*2**20):
                raise RuntimeError('Bounded diagnostic resource guard stopped the probe; original checkpoint preserved')
        solver.solve(n_steps=subdivision+1,dt=h,perturb_transient=case['perturb_transient'],
                     save_artifacts=False,verbose=False,step_callback=record)
        # Both derivatives are centered at the same time T+original_dt.
        before, after = moments[subdivision-1],moments[subdivision+1]
        force = storage_corrected_force(center['outer_force'],before,after,h)
        boundary_before, boundary_after = boundary_moments[subdivision-1],boundary_moments[subdivision+1]
        boundary_force = storage_corrected_force(center['outer_force'],boundary_before,boundary_after,h)
        result['probes'].append(dict(center_time=center['time'],half_width=h,
            storage_rate=((after-before)/(2*h)).tolist(),outer_terms=center['terms'],
            cv_force=force.tolist(),surface_force=center['surface_force'].tolist(),
            force_difference=(force-center['surface_force']).tolist(),
            boundary_moment_storage_rate=((boundary_after-boundary_before)/(2*h)).tolist(),
            boundary_moment_cv_force=boundary_force.tolist(),
            boundary_moment_force_difference=(boundary_force-center['surface_force']).tolist(),
            storage_quadrature_difference=(((after-before)-(boundary_after-boundary_before))/(2*h)).tolist()))
        del solver
    assert np.isclose(result['probes'][0]['center_time'],result['probes'][1]['center_time'],rtol=0,atol=1e-10)
    assert original_digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result.update(elapsed_seconds=time.perf_counter()-started,
        source_sha256=manifest['metadata']['source_sha256'],
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outer_quadrature_sha256=hashlib.sha256((ROOT/'validation/steady_momentum_budget.py').read_bytes()).hexdigest(),
        derivative_step_sensitivity=np.abs(np.array(result['probes'][0]['storage_rate'])
                                            -np.array(result['probes'][1]['storage_rate'])).tolist(),
        scope='Single matched phase after checkpoint; independent outer force, shared-volume and external mass-moment storage routes, centered derivatives. External route assumes impermeable fixed body and zero wall-normal flow. Not cycle-wide or spatial acceptance.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.run)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(probes=result['probes'],derivative_step_sensitivity=result['derivative_step_sensitivity'])))
