"""Check pressure uniqueness from the actual fluid graph, not a lift symmetry test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.csgraph import connected_components
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solver'))
from cut_cell_geom import CutCellGeometry2D
from cut_cell_solver import build_cut_cell_poisson_matrix


def inspect(geom):
    fluid = (geom.cell_type != -1).ravel()
    matrix = -build_cut_cell_poisson_matrix(geom)[fluid][:,fluid].tocsr()
    scale = max(float(np.max(abs(matrix.data))),1.)
    tolerance = 1e-12*scale
    asymmetric = matrix-matrix.T
    symmetry_error = float(np.max(abs(asymmetric.data))) if asymmetric.nnz else 0.
    offdiag = matrix.copy()
    offdiag.setdiag(0)
    offdiag.eliminate_zeros()
    components,labels = connected_components(offdiag,directed=False)
    grounding = np.asarray(matrix.sum(axis=1)).ravel()
    grounded = [bool(np.any(grounding[labels == k] > tolerance)) for k in range(components)]
    positive_edges = float(offdiag.data.max()) if offdiag.nnz else 0.
    valid = (symmetry_error < tolerance and positive_edges <= tolerance
             and grounding.min() > -tolerance and all(grounded))
    jj,ii = np.indices((geom.ny,geom.nx))
    checkerboard = ((-1.)**(jj+ii)).ravel()[fluid]
    quotient = float(checkerboard@(matrix@checkerboard)/(checkerboard@checkerboard))
    return dict(fluid_nodes=int(fluid.sum()),components=components,
        every_component_grounded_at_outlet=all(grounded),symmetry_error=symmetry_error,
        largest_positive_offdiagonal=positive_edges,minimum_grounding=float(grounding.min()),
        coercivity_graph_conditions_met=bool(valid),checkerboard_rayleigh_quotient=quotient,
        checkerboard_energy_times_h_squared=quotient*min(geom.dx,geom.dy)**2,
        conclusion='Graph-Laplacian energy is positive for every nonzero pressure vector when all components connect to a Dirichlet grounding. This excludes checkerboard/null-space decoupling; it does not prove noise-free transient sensor signals.',
        ready_for_paper1=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    case = json.loads(args.config.read_text())
    s = case['solver']
    geom = CutCellGeometry2D(s['nx'],s['ny'],s['Lx'],s['Ly'],s['xc'],s['yc'],s['D']/2)
    result = inspect(geom)
    result.update(configuration_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operator_sha256=hashlib.sha256((ROOT/'solver/cut_cell_solver.py').read_bytes()).hexdigest())
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(result))
