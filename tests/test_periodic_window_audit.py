"""Analytical tests of window trimming and common-cycle statistics."""
import sys
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocessing.analysis import window_statistics, periodic_metrics
from validation.periodic_window_audit import complete_cycles, audit
from validation.run_stage0_case import summarize


def test_linear_and_invalid_windows():
    # 1. Linear test with non-uniform time grid
    t = np.array([0., .1, .4, 1.])
    values = np.column_stack((1e6 + t, 3 - 2 * t))
    result = window_statistics(t, values, .2, .8)
    assert np.allclose(result['mean'], [1e6 + .5, 2.], rtol=0, atol=2e-10)
    assert np.allclose(result['rms'], [.6 / np.sqrt(12), 1.2 / np.sqrt(12)], rtol=0, atol=1e-10)
    assert np.allclose(result['mid_extrema'], [1e6 + .5, 2.], rtol=0, atol=2e-10)
    assert np.allclose(result['half_amplitude'], [.3, .6], rtol=0, atol=2e-10)
    assert np.allclose(result['amplitude'], [.6, 1.2], rtol=0, atol=2e-10)

    for a, b in ((-.1, .8), (.2, 1.1), (.8, .2), (0., float('nan'))):
        try:
            window_statistics(t, values, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid integration window accepted')


def test_periodic_nonzero_mean_and_cycle_trimming():
    # 2. Periodic signal with non-zero mean and fractional cycle start/end
    t = np.linspace(0, 10.3, 10301)
    cd = 3.0 + 0.1 * np.cos(4 * np.pi * t)
    cl = 0.2 + 1.0 * np.sin(2 * np.pi * t)
    
    # Analysis starts at t_start=1.0 (not a zero crossing) and t ends at 10.3 (not a zero crossing)
    periodic, crossings = complete_cycles(t, cd, cl, 1.0)
    result = window_statistics(t, np.column_stack((cd, cl)), crossings[0], crossings[-1])

    assert periodic['stationary'] and periodic['cycles'] >= 8
    assert abs(periodic['strouhal'] - 1.0) < 1e-5
    # Mean of cl is exactly 0.2, mean of cd is exactly 3.0
    assert np.allclose(result['mean'], [3.0, 0.2], atol=1e-8)
    # RMS around the mean: for sin(2*pi*t) is 1/sqrt(2), for 0.1*cos(4*pi*t) is 0.1/sqrt(2)
    assert np.allclose(result['rms'], [0.1 / np.sqrt(2), 1.0 / np.sqrt(2)], atol=5e-6)
    # Extrema and amplitudes
    assert np.allclose(result['mid_extrema'], [3.0, 0.2], atol=1e-8)
    assert np.allclose(result['half_amplitude'], [0.1, 1.0], atol=1e-8)
    assert np.allclose(result['amplitude'], [0.2, 2.0], atol=1e-8)

    # Constant signal should fail periodic detection
    try:
        complete_cycles(t, cd, np.ones_like(t), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError('Constant lift certified as periodic')


def test_unequal_time_intervals():
    # 3. Unequal time intervals (non-uniform time grid)
    np.random.seed(42)
    dt_base = 0.001
    dt_unequal = dt_base * (1.0 + 0.6 * np.sin(np.linspace(0, 20 * np.pi, 8000)))
    t = np.r_[0.0, np.cumsum(dt_unequal)]
    # Trim to ~8.0
    t = t[t <= 8.0]
    
    # Quadratic signal on unequal grid: f(t) = t^2
    # Integral of t^2 from 1 to 7 is (7^3 - 1^3)/3 = 342/3 = 114
    # Mean is 114 / 6 = 19
    values = (t**2)[:, None]
    result = window_statistics(t, values, 1.0, 7.0)
    assert abs(result['mean'][0] - 19.0) < 1e-3  # Trapeze approximation on grid with h~0.001

    # Unequal sinusoidal periodic signal
    cl = 0.5 + 2.0 * np.sin(2 * np.pi * t)
    cd = 1.5 + 0.3 * np.cos(4 * np.pi * t)
    periodic, crossings = complete_cycles(t, cd, cl, 0.5)
    stats = window_statistics(t, np.column_stack((cd, cl)), crossings[0], crossings[-1])
    assert abs(stats['mean'][1] - 0.5) < 1e-4
    assert abs(stats['rms'][1] - 2.0 / np.sqrt(2)) < 1e-4
    assert abs(stats['mid_extrema'][1] - 0.5) < 1e-4
    assert abs(stats['half_amplitude'][1] - 2.0) < 1e-4


def test_restart_multisegment_and_corruptions():
    # 4 & 5. Multi-segment history, restart replay, torn files, missing steps, and summarize/audit agreement
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        case_config = dict(kind='periodic', dt=0.01, end_time=10.3, sample_start=1.0, solver=dict(D=1.0))
        manifest = dict(
            configuration=case_config,
            metadata=dict(U_ref=1.0, source_sha256={'fixture': 'same'}),
            runner_sha256='fixture',
            analysis_sha256='fixture'
        )
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        (folder / 'summary-0000.json').write_text(json.dumps(dict(
            stationary=True, max_cfl=0.1, max_divergence=0.0, source_sha256={'fixture': 'same'}
        )))
        
        lines = []
        for step in range(1, 1031):
            time = step * 0.01
            cl = 0.2 + np.sin(2 * np.pi * time)
            cd = 3.0 + 0.1 * np.cos(4 * np.pi * time)
            dp = 2.5 + 0.05 * np.cos(2 * np.pi * time)
            cp = np.arange(24) * cl + 1e5 * time
            lines.append(json.dumps(dict(
                step=step, time=time, cd=cd, cl=cl, delta_p=dp, cp=cp.tolist(),
                cfl=0.15, max_divergence=1e-14
            )) + '\n')

        # Split history into two segments with a 1-step overlap (step 500)
        (folder / 'history-0000.jsonl').write_text(''.join(lines[:500]))
        tail = folder / 'history-0001.jsonl'
        tail.write_text(''.join(lines[499:]))

        # Check that summarize and audit produce matching results
        audit_res = audit(folder)
        summ_res = summarize(folder, case_config)

        # Exact agreement between summarize and audit on the cycle window
        assert np.isclose(summ_res['cl_rms'], audit_res['common_window_statistics']['rms'][1], rtol=0, atol=1e-12)
        assert np.isclose(summ_res['cd_mean'], audit_res['common_window_statistics']['mean'][0], rtol=0, atol=1e-12)
        assert np.isclose(summ_res['cl_mean'], audit_res['common_window_statistics']['mean'][1], rtol=0, atol=1e-12)
        assert np.isclose(summ_res['delta_p'], audit_res['common_window_statistics']['mean'][2], rtol=0, atol=1e-12)
        assert np.isclose(summ_res['sample_start'], audit_res['common_window_statistics']['window'][0], rtol=0, atol=1e-12)
        assert np.isclose(summ_res['sample_end'], audit_res['common_window_statistics']['window'][1], rtol=0, atol=1e-12)

        # Replay with tiny float jitter on time should still resolve cycles
        replay = [json.loads(line) for line in lines[490:500]]
        for record in replay:
            record['time'] += 1e-14
        (folder / 'history-0000.jsonl').write_text(''.join(lines[:500]) + '{"step":501')
        tail.write_text(''.join(json.dumps(row) + '\n' for row in replay) + ''.join(lines[500:]))
        assert audit(folder)['periodic']['cycles'] == audit_res['periodic']['cycles']

        # Conflicting record: step 499 with cl modified
        conflict = json.loads(lines[499])
        conflict['cl'] += 1.0
        for corrupted in ([json.dumps(conflict) + '\n'] + lines[500:], lines[501:]):
            tail.write_text(''.join(corrupted))
            try:
                audit(folder)
            except ValueError:
                pass
            else:
                raise AssertionError('Conflicting or missing history accepted')


def run():
    test_linear_and_invalid_windows()
    test_periodic_nonzero_mean_and_cycle_trimming()
    test_unequal_time_intervals()
    test_restart_multisegment_and_corruptions()
    print('PASS: all lightweight analytical and history tests passed cleanly!')


if __name__ == '__main__':
    run()
