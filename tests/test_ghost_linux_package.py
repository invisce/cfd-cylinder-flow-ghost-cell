"""Packaging-only checks: never construct a solver or advance a timestep."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder=load('linux_package_builder',ROOT/'validation/build_ghost_linux_package.py')
bootstrap=load('linux_package_bootstrap',ROOT/'validation/ghost_linux_bootstrap.py')


class LinuxPackageTests(unittest.TestCase):
    def test_same_grid_restart_preserves_multistep_state(self):
        runner=load('restart_runner',ROOT/'validation/run_ghost_rms.py')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'checkpoint.npz'
            config=dict(nx=40,ny=20,boundary='ghost_cell',time_scheme='ab2_cn')
            np.savez(path,checkpoint_metadata=json.dumps(dict(configuration=config)))
            state=SimpleNamespace(previous_dt=.001,half_pressure=np.array([3.]),
                                  previous_advection=np.array([4.]))
            restored=SimpleNamespace(gg=state)
            cls=Mock(); cls.from_checkpoint.return_value=restored
            self.assertIs(runner.restart_ghost(cls,path,40,20),restored)
            self.assertEqual(restored.gg.previous_dt,.001)
            cls.from_checkpoint.assert_called_once_with(path)
            cls.reset_mock()
            with self.assertRaises(ValueError):
                runner.restart_ghost(cls,path,80,40)
            cls.from_checkpoint.assert_not_called()
            config['boundary']='cut_cell'
            np.savez(path,checkpoint_metadata=json.dumps(dict(configuration=config)))
            with self.assertRaises(ValueError):
                runner.restart_ghost(cls,path,40,20)

    def test_portability_provenance_and_archive_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'project'
            for name in ('solver/src','validation','postprocessing','reports/ghost_rms_numba_fft_20261009'):
                (root/name).mkdir(parents=True)
            (root/'solver/main.py').write_text('pass\n')
            (root/'solver/mac_engine.cp313-win_amd64.pyd').write_bytes(b'windows binary')
            (root/'solver/src/bindings.cpp').write_text('native source\n')
            for name in ('run_ghost_rms.py','ghost_linux_bootstrap.py'):
                (root/'validation'/name).write_text('pass\n')
            (root/'postprocessing/analysis.py').write_text('pass\n')
            (root/'requirements.txt').write_text('numba\npsutil\n')
            checkpoint=root/'checkpoint.npz'
            checkpoint.write_bytes(b'unchanged input')
            old=root/'reports/ghost_rms_numba_fft_20261009/manifest.json'
            old.write_text(json.dumps(dict(source=str(checkpoint),source_sha256=builder.sha(checkpoint),
                grids=[[660,123],[880,164]],limits=dict(seconds=7200,disk_bytes=1024**3))))
            old_bytes=old.read_bytes()
            output,archive=builder.package(root,Path(temporary)/'linux')
            manifest=bootstrap.verify(output,native=False)
            self.assertEqual(manifest['source'],'inputs/developed_cut_cell.npz')
            self.assertIn('src/bindings.cpp',manifest['solver_sha256'])
            self.assertFalse(list(output.rglob('*.pyd')))
            self.assertEqual(checkpoint.read_bytes(),b'unchanged input')
            self.assertEqual(old.read_bytes(),old_bytes)
            with zipfile.ZipFile(archive) as stream:
                self.assertTrue(all('\\' not in name for name in stream.namelist()))
                self.assertIn('linux/inputs/developed_cut_cell.npz',stream.namelist())
            with self.assertRaises(FileExistsError):
                builder.package(root,output)
            (output/'solver/main.py').write_text('tampered\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                bootstrap.verify(output,native=False)
            extra,_=builder.package(root,Path(temporary)/'finer',[[1100,205],[1320,246]])
            new_manifest=bootstrap.verify(extra,native=False)
            self.assertEqual(new_manifest['grids'],[[1100,205],[1320,246]])
            self.assertEqual(new_manifest['limits']['seconds'],10800)
            with self.assertRaises(ValueError):
                builder.package(root,Path(temporary)/'invalid',[[1100,205],[1320,205]])

    def test_runner_preserves_explicit_threads_and_resolves_relative_source(self):
        code='''import runpy, tempfile
from pathlib import Path
module=runpy.run_path('validation/run_ghost_rms.py',run_name='packaging_test')
assert module['os'].environ['NUMBA_NUM_THREADS']=='2'
assert module['os'].environ['FFT_WORKERS']=='2'
assert module['source_path']({'source':'inputs/source.npz'})==module['BASE']/'inputs/source.npz'
'''
        env=dict(os.environ,NUMBA_NUM_THREADS='2',FFT_WORKERS='2')
        subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,check=True)

    def test_bootstrap_refuses_rebuild_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'manifest.json').write_text(json.dumps(dict(solver_sha256={},source='input',
                source_sha256=builder.sha(ROOT/'requirements.txt'),runner_sha256='',analysis_sha256='',package_sha256={})))
            (root/'pilot_/123').mkdir(parents=True)
            (root/'pilot_/123/checkpoint.npz').write_bytes(b'preserve')
            original=bootstrap.verify
            bootstrap.verify=lambda *args,**kwargs: None
            try:
                with self.assertRaisesRegex(RuntimeError,'fresh package'):
                    bootstrap.build(root)
            finally:
                bootstrap.verify=original


if __name__=='__main__':
    unittest.main()
