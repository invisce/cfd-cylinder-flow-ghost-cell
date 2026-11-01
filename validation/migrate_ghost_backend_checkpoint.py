"""Explicit backend-only migration: verify old provenance and identical discrete operators."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'solver')]
import numpy as np
from main import MAC_CylinderFlow


def operator_hash(matrix):
    matrix=matrix.tocsr(copy=True)
    matrix.sort_indices()
    digest=hashlib.sha256()
    digest.update(np.asarray(matrix.shape,dtype='<i8').tobytes())
    for value in (matrix.data,matrix.indices,matrix.indptr):
        digest.update(value.tobytes())
    return digest.hexdigest()


def migrate(checkpoint, snapshot, output, pressure_solver='iterative'):
    checkpoint,snapshot,output=Path(checkpoint).resolve(),Path(snapshot).resolve(),Path(output).resolve()
    if output.exists() or output.with_suffix('.migration.json').exists():
        raise FileExistsError(output)
    helper='''import json,sys,numpy as np,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'solver'))
from main import MAC_CylinderFlow
s=MAC_CylinderFlow.from_checkpoint(sys.argv[2])
if s.gg is None:raise ValueError('Only verified Ghost AB2/CN checkpoints can be migrated')
def digest(matrix):
 matrix=matrix.tocsr(copy=True);matrix.sort_indices();h=hashlib.sha256()
 h.update(np.asarray(matrix.shape,dtype='<i8').tobytes())
 for value in (matrix.data,matrix.indices,matrix.indptr):h.update(value.tobytes())
 return h.hexdigest()
with np.load(sys.argv[2],allow_pickle=False) as source:meta=json.loads(str(source['checkpoint_metadata']))
meta['operator_sha256']={name:digest(getattr(s.gg,name)) for name in ('E','P','D','G','A','L')}
arrays={name:getattr(s,name) for name in ('u','v','p','phi','u_inlet')}
arrays.update(previous_advection=s.gg.previous_advection,half_pressure=s.gg.half_pressure,
              ui=s.gg.ui,vi=s.gg.vi,pi=s.gg.pi,metadata=json.dumps(meta))
if any(arrays[k] is None for k in ('previous_advection','half_pressure')):raise ValueError('Developed history required')
np.savez(sys.argv[3],**arrays)
'''
    with tempfile.TemporaryDirectory() as directory:
        transfer=Path(directory)/'verified_state.npz'
        subprocess.run([sys.executable,'-c',helper,str(snapshot),str(checkpoint),str(transfer)],check=True)
        with np.load(transfer,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            config=dict(meta['configuration'],pressure_solver=pressure_solver)
            s=MAC_CylinderFlow(**config)
            hashes={name:operator_hash(getattr(s.gg,name)) for name in ('E','P','D','G','A','L')}
            if hashes != meta['operator_sha256']:
                raise ValueError('Operators differ: backend-only migration cannot change reconstruction or discretization')
            for name in ('ui','vi','pi'):
                if not np.array_equal(getattr(s.gg,name),data[name]):raise ValueError('Changed active unknowns')
            for name in ('u','v','p','phi','u_inlet'):
                value=data[name]
                if value.shape!=getattr(s,name).shape or not np.isfinite(value).all():raise ValueError('Invalid field')
                getattr(s,name)[:]=value
            s.gg.previous_advection=data['previous_advection'].copy()
            s.gg.half_pressure=data['half_pressure'].copy()
            s.gg.previous_dt=meta['prev_dt']
            s.time=meta['time'];s.run_options=meta['run_options']
    if np.max(abs(s.gg.divergence(s.gg.pack(s.u,s.v))))>=1e-10:
        raise ValueError('Imported state is not solenoidal')
    provenance=dict(source_checkpoint=str(checkpoint),source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        old_source_sha256=meta['source_sha256'],new_source_sha256=s.metadata(1.,0)['source_sha256'],
        operator_sha256=hashes,mode='backend_only_identical_operators',history_reset=False)
    s.save_checkpoint(output)
    with output.with_suffix('.migration.json').open('x',encoding='utf-8') as stream:
        json.dump(provenance,stream,indent=2)
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--pressure-solver',choices=['direct','iterative'],default='iterative')
    print(json.dumps(migrate(**vars(parser.parse_args())),indent=2))
