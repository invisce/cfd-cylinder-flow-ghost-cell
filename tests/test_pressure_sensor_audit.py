"""Independent analytical integration checks for the exported sensor audit."""
import sys
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation.pressure_sensor_audit import statistics, audit, compare


def run():
    t = np.array([0., .1, .4, 1.])
    offsets = np.arange(24)/3
    cp = 1e6+offsets[None,:]+t[:,None]
    result = statistics(t, cp)
    assert np.allclose(result['cp_mean'], 1e6+offsets+.5, rtol=0, atol=2e-10)
    assert np.allclose(result['cp_rms'], np.sqrt(1/12), rtol=0, atol=1e-10)
    trimmed=statistics(t,cp,(.2,.8))
    assert np.allclose(trimmed['cp_mean'],1e6+offsets+.5,rtol=0,atol=3e-10)
    assert np.allclose(trimmed['cp_rms'],.6/np.sqrt(12),rtol=0,atol=1e-10)
    constant = statistics(t, np.full((4,24), 3.25))
    assert np.max(constant['cp_rms']) == 0
    for times, values in (([0,0], np.ones((2,24))), ([0,1], np.ones((2,23))),
                          ([0,1], np.full((2,24), np.nan))):
        try:
            statistics(times, values)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid sensor observations accepted')
    with TemporaryDirectory() as directory:
        folders = []
        for nx, gauge in ((10,1.), (20,10.)):
            folder = Path(directory)/str(nx)
            folder.mkdir()
            folders.append(folder)
            manifest = dict(configuration=dict(name=str(nx), kind='steady', dt=.25,
                end_time=1., sample_start=.25, solver=dict(nx=nx, ny=nx, Lx=1., Ly=1.,
                xc=.5, yc=.5, D=.1, rho=2.)), metadata=dict(U_ref=1., source_sha256={'fixture':'same'}),
                runner_sha256='same', analysis_sha256='same')
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'summary-0000.json').write_text(json.dumps(dict(stationary=True,
                max_cfl=.1, max_divergence=0., source_sha256={'fixture':'same'})))
            with (folder/'history-0000.jsonl').open('w') as stream:
                for step in range(1,5):
                    time = step*.25
                    reference = gauge+gauge*time
                    stream.write(json.dumps(dict(step=step, time=time,
                        cp=(offsets+time-reference).tolist(), pressure_reference=reference))+'\n')
        result = compare(folders)
        assert result['adjacent_grid_differences'][0]['maximum_absolute_cp_mean_change'] < 1e-12
        assert result['adjacent_grid_differences'][0]['maximum_absolute_normalized_pressure_mean_change'] < 1e-12
        assert np.allclose(result['runs'][0]['absolute_pressure_normalized_statistics']['cp_mean'], offsets+.625)
        # A periodic run uses complete-cycle endpoints for all pressure representations.
        manifest['configuration'].update(kind='periodic',dt=.01,end_time=10.3,sample_start=1.)
        folder=folders[-1]
        (folder/'manifest.json').write_text(json.dumps(manifest))
        with (folder/'history-0000.jsonl').open('w') as stream:
            for step in range(1,1031):
                time=step*.01
                reference=10.+10.*time
                stream.write(json.dumps(dict(step=step,time=time,cd=3+.1*np.cos(4*np.pi*time),
                    cl=.2+np.sin(2*np.pi*time),cp=(offsets+time-reference).tolist(),
                    pressure_reference=reference))+'\n')
        _,periodic=audit(folder)
        assert periodic['periodic']['cycles']>=8
        assert periodic['window']==periodic['stored_cp_statistics']['window']
        absolute=periodic['absolute_pressure_normalized_statistics']
        assert absolute['window']==periodic['window']
        assert np.allclose(absolute['cp_mean'],offsets+sum(absolute['window'])/2)
        assert np.allclose(absolute['cp_rms'],np.diff(absolute['window'])[0]/np.sqrt(12))
        history = folders[0]/'history-0000.jsonl'
        history.write_text(''.join(history.read_text().splitlines(keepends=True)[1:]))
        try:
            audit(folders[0])
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete production history accepted')
    print('PASS: analytic mean/RMS, nonuniform times, gauge invariance, invalid/missing data')


def test_24_sensor_angular_audit():
    json_path = ROOT / 'reports' / 're100_surface_pressure_sensors_angular_audit_20261004.json'
    assert json_path.exists(), f"Missing audit JSON: {json_path}"
    data = json.loads(json_path.read_text(encoding='utf-8'))

    # 1. Structural checks
    for key in ['grids', 'sensors_comparison', 'sector_summary', 'macro_sectors', 'key_findings']:
        assert key in data, f"Missing key '{key}' in audit JSON"

    # 2. Check grids
    for gid in ['g1', 'g2', 'g3']:
        assert gid in data['grids']
        g = data['grids'][gid]
        assert g['sample_count'] > 20000
        assert len(g['sensors']) == 24

        # Check angle distribution & weights
        angles = np.array([s['angle_degrees'] for s in g['sensors']])
        assert np.allclose(angles, np.arange(24) * 15.0)

        w_lift = np.array([s['w_lift'] for s in g['sensors']])
        # Exact gauge invariance of lift: sum w_lift == 0
        assert abs(np.sum(w_lift)) < 1e-15

        # Reconstructed pressure lift ratio vs full NS
        ratio = g['forces_24_sensor_reconstructed_pressure']['ratio_cl_p_rms_to_total_cl_rms']
        assert 0.88 < ratio < 0.93, f"Unexpected ratio {ratio} for grid {gid}"

        # Lift variance budget closure
        shares = np.array([s['lift_variance_share_pct'] for s in g['sensors']])
        assert np.isclose(np.sum(shares), 100.0, atol=1e-10)

    # 3. Macro-sector physics verification on fine grid (G3)
    macro = data['macro_sectors']
    top_share = macro['top_shear_layer']['lift_share_g3']
    bot_share = macro['bottom_shear_layer']['lift_share_g3']
    stag_share = macro['stagnation_front']['lift_share_g3']
    wake_share = macro['wake_base_suction']['lift_share_g3']

    # Combined top and bottom shear layers account for ~70% of lift variance
    assert (top_share + bot_share) > 68.0, f"Expected top+bottom > 68%, got {top_share + bot_share}%"
    # Stagnation front accounts for < 1.5%
    assert stag_share < 1.5, f"Expected stagnation < 1.5%, got {stag_share}%"
    # Wake suction accounts for < 10% (around 8.4% across 150-210)
    assert wake_share < 10.0, f"Expected wake < 10%, got {wake_share}%"

    # 4. Sensitivity monotonicity: top & bottom apexes (90 deg and 270 deg) have largest RMS delta
    s_comp = data['sensors_comparison']
    delta_rms = np.array([s['delta_p_rms_g3_minus_g1'] for s in s_comp])
    idx_max = np.argmax(delta_rms)
    assert idx_max in (6, 18), f"Expected maximum delta at apex (6 or 18), got {idx_max}"

    # 5. Check markdown report exists and contains required sections
    md_path = ROOT / 'reports' / 're100_surface_pressure_sensors_angular_audit_20261004.md'
    assert md_path.exists(), f"Missing audit MD: {md_path}"
    md_text = md_path.read_text(encoding='utf-8')
    assert '۲۴ حسگر' in md_text
    assert 'جدول جامع آماری' in md_text
    assert 'سهم‌بندی واریانس برآ' in md_text
    print('PASS: 24-sensor angular audit verification (JSON structure, gauge invariance, variance budget, sector concentration)')


if __name__ == '__main__':
    run()
    test_24_sensor_angular_audit()
