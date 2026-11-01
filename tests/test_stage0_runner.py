import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
import json
import tempfile
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from validation.run_stage0_case import ResourceLimit, preflight, run_case, summarize
from postprocessing.analysis import calculate_gci


def test_runner():
    pilot = json.loads((ROOT/'configs/stage0_dfg_re20_pilot.json').read_text())
    assert preflight(pilot)['heavy']
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)/'run'
        try:
            run_case(pilot, output)
        except PermissionError:
            pass
        else:
            raise AssertionError('Heavy run started without readiness')
        assert not output.exists()
        case = copy.deepcopy(pilot)
        case['solver'].update(nx=40, ny=10, Lx=20., Ly=5., xc=5., yc=2.5,
                              nu=.01, inflow='uniform')
        case.update(dt=.001, end_time=.01, sample_start=.004)
        assert not preflight(case)['heavy']
        # A large allowed ceiling must not require that much RAM to be free;
        # the small case's estimated working set plus reserve is sufficient.
        with patch('validation.run_stage0_case.psutil.virtual_memory',
                   return_value=SimpleNamespace(available=512*2**20)):
            result = run_case(case, output, budget_mib=4096)
        assert result['max_divergence'] < 1e-10
        assert not result['accepted'] and not result['ready_for_paper1']
        row = json.loads((output/'history-0000.jsonl').read_text().splitlines()[-1])
        assert len(row['cp']) == 24 and np.isfinite(row['cp']).all()
        assert row['step'] == 10
        assert abs(row['cl']-row['cl_pressure']-row['cl_viscous']) < 1e-12
        assert (output/'checkpoint.npz').exists()
        with zipfile.ZipFile(output/'source_snapshot.zip') as archive:
            assert 'solver/time_integration.py' in archive.namelist()
            assert 'solver/wall_reconstruction.py' in archive.namelist()
            assert json.loads(archive.read('case.json')) == case
        run_case(case, output, resume=True)
        try:
            run_case(case, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Prior run was overwritten')
        interrupted = Path(directory)/'interrupted'
        try:
            run_case(case, interrupted, budget_seconds=1e-9)
        except ResourceLimit:
            pass
        else:
            raise AssertionError('Exceeded time budget was ignored')
        assert (interrupted/'checkpoint.npz').exists()
        assert (interrupted/'stopped-0000.json').exists()
        with (interrupted/'history-0000.jsonl').open('a', encoding='utf-8') as history:
            history.write('{"step":')  # Simulate a torn final JSON write after a saved checkpoint.
        resumed = run_case(case, interrupted, resume=True)
        assert resumed['max_divergence'] < 1e-10
        with np.load(output/'checkpoint.npz') as full, np.load(interrupted/'checkpoint.npz') as split:
            for field in ('u', 'v', 'p'):
                np.testing.assert_array_equal(full[field], split[field])
        with (interrupted/'history-0001.jsonl').open('a', encoding='utf-8') as history:
            history.write('{"bad":\n')
        try:
            summarize(interrupted, case)
        except json.JSONDecodeError:
            pass
        else:
            raise AssertionError('Malformed complete history record was accepted')
        # Analysis provenance is just as important as the solver provenance.
        manifest_path = interrupted/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['analysis_sha256'] = 'changed'
        manifest_path.write_text(json.dumps(manifest))
        try:
            run_case(case, interrupted, resume=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed analysis code was accepted')
    h = np.array([1/120, 1/80, 1/40])
    gci = calculate_gci(*(3+2*h*h), r=h[1]/h[0], r32=h[2]/h[1])
    assert abs(gci['p']-2) < 1e-9 and abs(gci['f_ext']-3) < 1e-12


if __name__ == '__main__':
    test_runner()
    print('Runner and unequal-grid GCI checks PASS')
