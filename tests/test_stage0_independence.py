import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
import json
import math
import tempfile
from validation.stage0_independence import compare


def test_independence():
    with tempfile.TemporaryDirectory() as directory:
        runs = []
        for n in (20, 40, 80, 120):
            folder = Path(directory)/str(n)
            folder.mkdir()
            case = dict(kind='steady', solver=dict(nx=n*4, ny=n*3, Lx=4., Ly=3., nu=.01),
                        dt=.001, sample_start=100., end_time=200.)
            manifest = dict(configuration=case, metadata={'source_sha256': {'solver': 'same'}},
                            runner_sha256='same', analysis_sha256='same')
            summary = dict(stationary=True, max_divergence=1e-14, max_cfl=.2, source_sha256={'solver':'same'},
                           cd_mean=3+2/n**2, cl_rms=.4+1/n**2, strouhal=.2+.2/n**2)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'summary-0000.json').write_text(json.dumps(summary))
            runs.append(folder)
        result = compare(runs, 'grid')
        assert result['thresholds_met'] and not result['ready_for_paper1']
        assert abs(result['metrics']['cd_mean']['gci']['p']-2) < 1e-8
        for mutation in ('count', 'revision', 'viscosity', 'aspect'):
            changed = copy.deepcopy(manifest)
            if mutation == 'revision':
                changed['metadata']['source_sha256']['solver'] = 'different'
            elif mutation == 'viscosity':
                changed['configuration']['solver']['nu'] = .02
            elif mutation == 'aspect':
                changed['configuration']['solver']['ny'] += 1
            (runs[-1]/'manifest.json').write_text(json.dumps(changed))
            try:
                compare(runs[:3] if mutation == 'count' else runs, 'grid')
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted invalid {mutation} evidence')


def test_periodic_common_window():
    with tempfile.TemporaryDirectory() as directory:
        runs = []
        for n in (20,40,80,120):
            folder = Path(directory)/str(n)
            folder.mkdir()
            case = dict(kind='periodic',solver=dict(nx=n*4,ny=n*3,Lx=4.,Ly=3.,nu=.01,D=1.),
                        dt=.01,sample_start=1.,end_time=10.3)
            manifest = dict(configuration=case,metadata=dict(source_sha256={'solver':'same'},U_ref=1.),
                            runner_sha256='same',analysis_sha256='same')
            # Deliberately inconsistent legacy aggregates must not affect the comparison.
            summary = dict(stationary=True,max_divergence=1e-14,max_cfl=.2,
                           source_sha256={'solver':'same'},cd_mean=n,cl_rms=n,strouhal=n)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'summary-0000.json').write_text(json.dumps(summary))
            with (folder/'history-0000.jsonl').open('w') as stream:
                for step in range(1,1031):
                    t=step*.01
                    stream.write(json.dumps(dict(step=step,time=t,cd=3+.1*math.cos(4*math.pi*t),
                        cl=.2+math.sin(2*math.pi*t),delta_p=2.,cp=[0.]*24))+'\n')
            runs.append(folder)
        result=compare(runs,'grid')
        assert result['thresholds_met'] and len(result['common_cycle_evidence'])==4
        assert max(result['metrics']['cl_rms']['adjacent_relative_changes'])<1e-12
        summary['max_cfl']=float('nan')
        (runs[-1]/'summary-0000.json').write_text(json.dumps(summary))
        try:
            compare(runs,'grid')
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite numerical gate accepted')


if __name__ == '__main__':
    test_independence()
    test_periodic_common_window()
    print('Independence evidence checks PASS')
