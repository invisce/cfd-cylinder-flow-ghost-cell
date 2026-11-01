"""Build and verify the native extension without advancing the solver."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, native=True):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    entries={f'solver/{name}':digest for name,digest in manifest['solver_sha256'].items()}
    entries.update({manifest['source']:manifest['source_sha256'],
        'validation/run_ghost_rms.py':manifest['runner_sha256'],
        'shared_analysis.py':manifest['analysis_sha256']})
    entries.update(manifest['package_sha256'])
    if native:
        runtime=json.loads((root/'native_manifest.json').read_text())
        if runtime['package_manifest_sha256']!=sha(root/'manifest.json'):
            raise ValueError('Native build belongs to another package')
        if runtime['python']!=platform.python_version() or runtime['machine']!=platform.machine():
            raise ValueError('Native build Python/architecture mismatch')
        for name,version in runtime['dependencies'].items():
            if importlib.metadata.version(name)!=version:
                raise ValueError(f'Dependency changed after build: {name}')
        entries.update(runtime['native_sha256'])
    for name,digest in entries.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:
            raise ValueError(f'Package hash mismatch: {name}')
    return manifest


def build(root):
    root=Path(root).resolve()
    verify(root,native=False)
    if any(root.glob('grid_/*/checkpoint.npz')) or any(root.glob('pilot_/*/checkpoint.npz')):
        raise RuntimeError('Preserve native provenance: build a fresh package after runs have started')
    subprocess.run([sys.executable,'setup.py','build_ext','--inplace','--force'],
        cwd=root/'solver',check=True)
    binaries=list((root/'solver').glob('mac_engine*.so'))+list((root/'solver').glob('mac_engine*.pyd'))
    if len(binaries)!=1:
        raise RuntimeError('Expected one native extension for this interpreter')
    runtime=dict(package_manifest_sha256=sha(root/'manifest.json'),
        python=platform.python_version(),machine=platform.machine(),platform=platform.platform(),
        dependencies={name:importlib.metadata.version(name) for name in
            ('numpy','scipy','matplotlib','numba','psutil','pybind11','setuptools')},
        native_sha256={path.relative_to(root).as_posix():sha(path) for path in binaries})
    temporary=root/'native_manifest.tmp'
    temporary.write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    os.replace(temporary,root/'native_manifest.json')
    check_imports(root)


def check_imports(root):
    root=Path(root).resolve()
    verify(root)
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'solver'))
    for name in ('main','optimized_ghost_cell','time_integration','numba_kernels','shared_analysis'):
        importlib.import_module(name)
    engine=importlib.import_module('mac_engine')
    runtime=json.loads((root/'native_manifest.json').read_text())
    if Path(engine.__file__).resolve() not in {(root/name).resolve() for name in runtime['native_sha256']}:
        raise ValueError('Imported native extension is outside the sealed package')
    print('Package hashes and imports verified; no simulation executed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('MPLBACKEND','Agg')
    root=Path(__file__).resolve().parent
    if args.check:
        check_imports(root)
    else:
        build(root)
