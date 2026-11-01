import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../solver")))
"""Shared force-history analysis. No frequency is reported without resolved cycles."""
import numpy as np


def window_statistics(t, values, left, right):
    """
    Time-weighted statistics over an arbitrary window [left, right] for vector data.
    Evaluates exact piecewise-linear integrals for:
      - mean: time-weighted average
      - rms: centered root-mean-square about the window mean
      - minimum, maximum: range extrema on the window
      - mid_extrema: (minimum + maximum) / 2
      - half_amplitude: (maximum - minimum) / 2
      - amplitude: maximum - minimum
    """
    t, values = np.asarray(t, float), np.asarray(values, float)
    if (t.ndim != 1 or len(t) < 2 or values.ndim != 2 or len(values) != len(t)
            or not np.isfinite(t).all() or not np.isfinite(values).all()
            or np.any(np.diff(t) <= 0) or not np.isfinite([left, right]).all()
            or not t[0] <= left < right <= t[-1]):
        raise ValueError('Require finite increasing times, vector data and a covered window')
    inside = (t > left) & (t < right)
    times = np.r_[left, t[inside], right]
    data = np.vstack(([np.interp(left, t, v) for v in values.T], values[inside],
                      [np.interp(right, t, v) for v in values.T]))
    weights = np.diff(times)[:, None]
    duration = right - left
    mean = np.sum(weights * (data[:-1] + data[1:]) / 2, axis=0) / duration
    centered = data - mean
    variance = np.sum(weights * (centered[:-1]**2 + centered[:-1] * centered[1:]
                                 + centered[1:]**2) / 3, axis=0) / duration
    minimum = data.min(axis=0)
    maximum = data.max(axis=0)
    mid_extrema = (minimum + maximum) / 2
    half_amplitude = (maximum - minimum) / 2
    amplitude = maximum - minimum
    return dict(mean=mean.tolist(), rms=np.sqrt(np.maximum(0.0, variance)).tolist(),
                minimum=minimum.tolist(), maximum=maximum.tolist(),
                mid_extrema=mid_extrema.tolist(), half_amplitude=half_amplitude.tolist(),
                amplitude=amplitude.tolist(),
                window=[float(left), float(right)])


def periodic_metrics(t, cd, cl, t_start, D=.4, U_ref=1.):
    t, cd, cl = (np.asarray(x, dtype=float) for x in (t, cd, cl))
    if (any(x.ndim != 1 for x in (t, cd, cl)) or not (len(t) == len(cd) == len(cl))
            or len(t) < 3 or not all(np.isfinite(x).all() for x in (t, cd, cl))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Histories must be finite, equal-length vectors with strictly increasing time')
    if not all(np.isfinite(x) for x in (t_start, D, U_ref)) or D <= 0 or U_ref <= 0:
        raise ValueError('Require finite t_start and positive D, U_ref')
    keep = t >= t_start
    t, cd, cl = t[keep], cd[keep], cl[keep]
    if len(t) < 3:
        raise ValueError('Analysis window contains fewer than three samples')
    centered = cl - np.trapezoid(cl, t)/(t[-1]-t[0])
    # Same-direction crossings measure full periods, even for asymmetric waves.
    idx = np.flatnonzero((centered[:-1] < 0) & (centered[1:] >= 0))
    zeros = t[idx] - centered[idx]*np.diff(t)[idx]/np.diff(centered)[idx]
    periods = np.diff(zeros)
    resolved = (len(periods) >= 2 and np.ptp(cl) > 1e-10
                and np.min(periods) >= 10*np.max(np.diff(t)))
    cycles = len(periods) if resolved else 0
    if resolved:
        a, b = zeros[0], zeros[-1]
    else:
        a, b = t[0], t[-1]
    stats = window_statistics(t, np.column_stack((cd, cl)), a, b)
    result = dict(cd_mean=float(stats['mean'][0]),
                  cd_max=float(stats['maximum'][0]), cd_min=float(stats['minimum'][0]),
                  cd_mid_extrema=float(stats['mid_extrema'][0]),
                  cd_half_amplitude=float(stats['half_amplitude'][0]),
                  cd_amplitude=float(stats['amplitude'][0]),
                  cd_rms=float(stats['rms'][0]),
                  cl_mean=float(stats['mean'][1]),
                  cl_max=float(stats['maximum'][1]), cl_min=float(stats['minimum'][1]),
                  cl_mid_extrema=float(stats['mid_extrema'][1]),
                  cl_half_amplitude=float(stats['half_amplitude'][1]),
                  cl_amp=float(stats['half_amplitude'][1]),
                  cl_amplitude=float(stats['amplitude'][1]),
                  cl_rms=float(stats['rms'][1]),
                  period=float(periods.mean()) if resolved else float('nan'),
                  strouhal=float(D/U_ref/periods.mean()) if resolved else float('nan'),
                  cycles=cycles, sample_start=float(a), sample_end=float(b),
                  stationary=False, amplitude_drift=float('nan'), period_drift=float('nan'),
                  drag_drift=float('nan'))
    if cycles >= 4:
        amps, means = [], []
        for left, right in zip(zeros[:-1], zeros[1:]):
            tt = np.r_[left, t[(t > left) & (t < right)], right]
            amps.append(np.ptp(np.interp(tt, t, cl))/2)
            means.append(np.trapezoid(np.interp(tt, t, cd), tt)/(right-left))
        half = cycles//2
        def drift(values):
            return float(abs(np.mean(values[-half:])-np.mean(values[:half]))
                         / max(abs(np.mean(values[-half:])), 1e-12))
        result.update(amplitude_drift=drift(amps), period_drift=drift(periods), drag_drift=drift(means))
        # A declared student-project screening threshold, not proof of a limit cycle.
        result['stationary'] = max(result[k] for k in ('amplitude_drift', 'period_drift', 'drag_drift')) < .02
    return result


def calculate_gci(f1, f2, f3, r=2., Fs=1.25, r32=None):
    """Conditional monotone three-grid estimate, NOT an error bound.

    Inputs run from fine to coarse; r=h2/h1 and optional r32=h3/h2.

    The normalized GCI ratio is |f1/f2| when p comes from these same three
    values. It cannot independently certify that they are asymptotic.
    """
    if not all(np.isfinite(x) for x in (f1, f2, f3, r, Fs)) or r <= 1 or Fs < 1:
        raise ValueError('GCI requires finite values, r > 1 and Fs >= 1')
    if r32 is not None and (not np.isfinite(r32) or r32 <= 1):
        raise ValueError('The coarser spacing ratio must exceed one')
    result = dict(p=None, f_ext=None, e_a21_pct=None, gci_fine_pct=None,
                  status='not estimable', reason='Nonmonotone, divergent, constant or nearly zero data')
    scale = max(abs(f1), abs(f2), abs(f3), np.finfo(float).tiny)
    e21, e32 = f2-f1, f3-f2
    if abs(f1) > 1e-12*scale:
        result['e_a21_pct'] = float(100*abs(e21/f1))
    if min(abs(e21), abs(e32), abs(f1)) <= 1e-12*scale or e21*e32 <= 0:
        return result
    if r32 is None or np.isclose(r32, r):
        p = float(np.log(abs(e32/e21))/np.log(r))
    else:
        from scipy.optimize import brentq
        def equation(order):
            return np.exp(order*np.log(r))*np.expm1(order*np.log(r32))/np.expm1(order*np.log(r))-abs(e32/e21)
        upper = 10.
        while equation(.01)*equation(upper) > 0 and upper < 80.:
            upper *= 2
        result['order_search_upper'] = upper
        if equation(.01)*equation(upper) >= 0:
            result['reason'] = f'No positive resolved order in [0.01, {upper:g}] for these unequal spacings'
            return result
        p = float(brentq(equation, .01, upper))
    result['p'] = p
    if p <= 0:
        return result
    # Nearly equal differences make extrapolation ill-conditioned. Show p only.
    if p < .1:
        result['reason'] = 'Near-zero observed order; extrapolation is ill-conditioned'
        return result
    result.update(f_ext=float(f1-e21/(r**p-1)),
                  gci_fine_pct=float(100*Fs*abs(e21/f1)/(r**p-1)),
                  status='conditional estimate',
                  reason='Requires asymptotic behaviour and negligible time/sampling error; not established by three points alone')
    return result
