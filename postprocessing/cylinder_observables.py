"""Disk-backed centerline samples; integrate the final complete-cycle window."""
import json
from pathlib import Path
import numpy as np
from cut_cell_solver import build_pressure_reconstruction


def recirculation_length(x, velocity, rear, diameter):
    x, velocity = np.asarray(x), np.asarray(velocity)
    if (x.ndim != 1 or x.shape != velocity.shape or len(x) < 2
            or not np.isfinite([rear,diameter]).all() or diameter <= 0
            or not np.isfinite(x).all() or not np.isfinite(velocity).all()
            or np.any(np.diff(x) <= 0) or x[0] < rear-1e-10):
        raise ValueError('Invalid wake centerline')
    negative = np.flatnonzero(velocity < 0)
    if not len(negative):
        return dict(status='no_resolved_reverse_flow', length_D=None)
    start = int(negative[0])
    if start > 1:
        return dict(status='detached_negative_region', length_D=None)
    crossings = np.flatnonzero((velocity[start:-1] < 0) & (velocity[start+1:] >= 0))
    if not len(crossings):
        return dict(status='reattachment_outside_sampled_domain', length_D=None)
    i = start+int(crossings[0])
    zero = x[i]-velocity[i]*(x[i+1]-x[i])/(velocity[i+1]-velocity[i])
    return dict(status='resolved', length_D=float((zero-rear)/diameter), x_reattachment=float(zero),
                definition='First reattachment of the connected mean reverse-flow region; distance from cylinder rear / D')


class CylinderSamples:
    def __init__(self, solver, path, stride=10):
        if solver.gg is None or isinstance(stride,bool) or not isinstance(stride,int) or stride < 1:
            raise ValueError('Cylinder sampling requires Ghost AB2/CN and a positive integer stride')
        self.solver, self.path, self.stride = solver, Path(path), stride
        self.calls, self.last = 0, -np.inf
        geom = solver.gg.geom
        self.pressure_operator = build_pressure_reconstruction(geom,
            np.array([[solver.yc,solver.xc-solver.D/2],[solver.yc,solver.xc+solver.D/2]]))
        self.first = int(np.ceil((solver.xc+solver.D/2)/solver.dx-1e-12))
        self.x = np.arange(self.first,solver.nx+1)*solver.dx
        fy = solver.yc/solver.dy-.5
        self.j = int(np.floor(fy)); self.weight = fy-self.j
        if not 0 <= self.j < solver.ny-1:
            raise ValueError('Wake centerline interpolation reaches an outer wall')
        metadata = dict(schema=1, columns=3+len(self.x), dtype='<f8', stride=stride,
                        x=self.x.tolist(), rear=solver.xc+solver.D/2, diameter=solver.D,
                        quantities=['time','p_front','p_back','u_centerline'],
                        sampling='Linear MAC interpolation in y; temporal trapezoids clipped to the requested window')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix('.json').open('x',encoding='utf-8') as stream:
            json.dump(metadata,stream,indent=2)
        self.stream = self.path.open('xb')

    def pressures(self):
        return np.asarray(self.pressure_operator @ self.solver.p.ravel())

    def record(self, *, force=False, pressures=None):
        self.calls += 1
        if not force and self.calls % self.stride:
            return
        s = self.solver
        if s.time == self.last:
            return
        if s.time < self.last:
            raise ValueError('Sample times must increase')
        pressure = self.pressures() if pressures is None else pressures
        line = (1-self.weight)*s.u[self.j,self.first:]+self.weight*s.u[self.j+1,self.first:]
        row = np.concatenate(([s.time],pressure,line)).astype('<f8')
        if not np.isfinite(row).all():
            raise ValueError('Nonfinite cylinder sample')
        self.stream.write(row.tobytes())
        self.last = s.time

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def mean_observables(path, left, right):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    columns = meta['columns']
    if (meta['schema'] != 1 or columns != 3+len(meta['x']) or meta['dtype'] != '<f8'
            or path.stat().st_size % (8*columns) or not np.isfinite([left,right]).all() or right <= left):
        raise ValueError('Invalid sample file or integration window')
    count = path.stat().st_size//(8*columns)
    if count < 2:
        raise ValueError('At least two samples are required')
    rows = np.memmap(path,dtype='<f8',mode='r',shape=(count,columns))
    if rows[0,0] > left or rows[-1,0] < right:
        raise ValueError('Samples do not cover the full requested window')
    integral = np.zeros(columns-1)
    duration = 0.
    previous = np.array(rows[0])
    if not np.isfinite(previous).all():
        raise ValueError('Nonfinite sample')
    max_dt = 0.
    for current in rows[1:]:
        if not np.isfinite(current).all() or current[0] <= previous[0]:
            raise ValueError('Nonfinite, duplicate or unordered sample')
        a,b = max(left,previous[0]),min(right,current[0])
        if b > a:
            dt = current[0]-previous[0]
            slope = (current[1:]-previous[1:])/dt
            va = previous[1:]+(a-previous[0])*slope
            vb = previous[1:]+(b-previous[0])*slope
            integral += .5*(b-a)*(va+vb)
            duration += b-a
            max_dt = max(max_dt,float(dt))
        previous = np.array(current)
    if not np.isclose(duration,right-left,rtol=1e-12,atol=1e-12):
        raise ValueError('Incomplete integration window')
    mean = integral/duration
    return dict(window=[float(left),float(right)], p_front_mean=float(mean[0]),
                p_back_mean=float(mean[1]), delta_p_mean=float(mean[0]-mean[1]),
                x=meta['x'], u_centerline_mean=mean[2:].tolist(), sample_max_dt=max_dt,
                recirculation=recirculation_length(meta['x'],mean[2:],meta['rear'],meta['diameter']),
                caveat='Temporal profile sampling error requires a stride-sensitivity check; no instantaneous checkpoint is substituted for the mean.')
