"""Create a fresh portable Ghost RMS package; never rewrite archived runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(root,destination,grids=None):
    root=Path(root).resolve()
    destination=Path(destination).resolve()
    archive=destination.with_suffix('.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Choose a new package directory; existing outputs are preserved')
    previous=json.loads((root/'reports/ghost_rms_numba_fft_20261009/manifest.json').read_text())
    grids=previous['grids'] if grids is None else grids
    if (not grids or any(len(row)!=2 or any(type(n)!=int or n<4 for n in row) for row in grids)
            or len({row[1] for row in grids})!=len(grids)):
        raise ValueError('Grids need distinct NY values and integer NX/NY >=4')
    checkpoint=Path(previous['source'])
    if not checkpoint.is_absolute():
        checkpoint=root/'reports/ghost_rms_numba_fft_20261009'/checkpoint
    if sha(checkpoint)!=previous['source_sha256']:
        raise ValueError('Warm-start checkpoint hash mismatch')
    destination.mkdir(parents=True)
    solver=destination/'solver'
    solver.mkdir()
    for path in (root/'solver').glob('*.py'):
        shutil.copy2(path,solver/path.name)
    shutil.copytree(root/'solver/src',solver/'src')
    (destination/'validation').mkdir()
    shutil.copy2(root/'validation/run_ghost_rms.py',destination/'validation/run_ghost_rms.py')
    shutil.copy2(root/'validation/ghost_linux_bootstrap.py',destination/'bootstrap.py')
    shutil.copy2(root/'postprocessing/analysis.py',destination/'shared_analysis.py')
    shutil.copy2(root/'requirements.txt',destination/'requirements.txt')
    (destination/'inputs').mkdir()
    shutil.copy2(checkpoint,destination/'inputs/developed_cut_cell.npz')
    manifest=dict(previous,grids=grids,portable_package=True,source='inputs/developed_cut_cell.npz',
        solver_sha256={path.relative_to(solver).as_posix():sha(path)
            for path in solver.rglob('*') if path.is_file()},
        runner_sha256=sha(destination/'validation/run_ghost_rms.py'),
        analysis_sha256=sha(destination/'shared_analysis.py'),
        package_sha256={name:sha(destination/name) for name in ('bootstrap.py','requirements.txt')})
    if max(row[1] for row in grids)>164:
        manifest['limits']=dict(previous['limits'],seconds=10800)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (destination/'README.md').write_text('''# Ghost RMS Linux package

Use Ubuntu x86-64 and Python 3.13 for the initial deployment. No simulation starts during setup.

```sh
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python bootstrap.py
python bootstrap.py --check
```

Install development headers matching your Python interpreter. If Ubuntu's default Python is
not 3.13, use a matching Python 3.13 interpreter and headers, or validate the available version
before long runs. bootstrap.py records actual dependency versions and the native binary hash.
The Windows source checkpoint is a read-only warm start, not a cross-platform solver restart.
Fresh Linux checkpoints must be resumed using this same sealed package and native build.

All thread pools default to one in the runner. Set NUMBA_NUM_THREADS and FFT_WORKERS before
launching to compare one thread against the instance's available vCPUs; keep BLAS/OpenMP at one
initially. Choose measured settings rather than assuming more threads are faster.

Only when ready for a simulation:
```sh
NUMBA_NUM_THREADS=1 FFT_WORKERS=1 python validation/run_ghost_rms.py --ny 123 --pilot
NUMBA_NUM_THREADS=1 FFT_WORKERS=1 python validation/run_ghost_rms.py --ny 123
NUMBA_NUM_THREADS=1 FFT_WORKERS=1 python validation/run_ghost_rms.py --ny 164
```

The pilot performs 100 steps; each full run performs 50,000. Outputs are under pilot_/ or grid_/.
Use --resume only for a recorded resource stop. Preserve the package, native_manifest.json,
history, checkpoint and summary together. Copy results off the instance before deleting it.
Do not rebuild after runs begin. No RAM stopping limit is enabled; time and disk limits remain.
Run a short save/resume parity check before production; it has not been validated on Linux here.
Stop EC2 when finished; this runner does not stop the AWS instance automatically.
''',encoding='utf-8')
    if grids!=previous['grids']:
        readme=destination/'README.md'
        text=readme.read_text(encoding='utf-8')
        text=text.replace('--ny 123 --pilot',f'--ny {grids[0][1]} --pilot')
        text=text.replace('--ny 123\n',f'--ny {grids[0][1]}\n')
        text=text.replace('--ny 164\n',f'--ny {grids[-1][1]}\n')
        readme.write_text(text,encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as stream:
        for path in destination.rglob('*'):
            if path.is_file():
                stream.write(path,f'{destination.name}/{path.relative_to(destination).as_posix()}')
    return destination,archive


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grids',nargs='+',metavar='NXxNY')
    args=parser.parse_args()
    grids=None if args.grids is None else [list(map(int,grid.split('x'))) for grid in args.grids]
    folder,archive=package(Path(__file__).resolve().parents[1],args.output,grids)
    print(folder)
    print(archive)
