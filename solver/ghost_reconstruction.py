"""Experimental exterior pressure continuation; baseline closure remains selectable."""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def exterior_pressure_extension(x, y, sdf, active):
    xx, yy = np.meshgrid(x, y)
    distance = sdf.distance(xx, yy)
    dx, dy = x[1]-x[0], y[1]-y[0]
    h = max(dx,dy)
    ids = np.flatnonzero(active)
    labels = np.full(active.size,-1,dtype=int)
    labels[ids] = np.arange(len(ids))
    selected = np.flatnonzero(active & (distance < 6*h))
    if len(selected) < 12:
        raise ValueError('Pressure continuation needs twelve exterior centers')
    ghosts = np.flatnonzero((distance <= 1e-13) & (distance >= -2*h))
    centers = np.column_stack((yy.ravel()[selected],xx.ravel()[selected]))
    tree = cKDTree(centers)
    rows, cols, values = [ids], [np.arange(len(ids))], [np.ones(len(ids))]
    conditions, errors, counts = [], [], []
    for row in ghosts:
        point = np.array([yy.ravel()[row],xx.ravel()[row]])
        target = np.array([1.,0.,0.,0.,0.,0.])
        for count in (12,20,32):
            radius,_ = tree.query(point,k=min(count,len(centers)))
            near = tree.query_ball_point(point,radius[-1]*(1+1e-12))
            sy,sx = ((centers[near]-point)/[dy,dx]).T
            basis = np.array([np.ones(len(near)),sx,sy,sx*sx,sx*sy,sy*sy])
            weights,_,rank,_ = np.linalg.lstsq(basis,target,rcond=1e-12)
            condition = float(np.linalg.cond(basis))
            weights[np.argmax(abs(weights))] += 1-weights.sum()
            error = float(np.max(abs(basis @ weights-target)))
            if rank == 6 and np.isfinite(condition) and condition <= 1e8 and error <= 1e-10:
                break
        else:
            raise ValueError('Unresolved exterior pressure polynomial')
        rows.append(np.full(len(near),row));cols.append(labels[selected[near]]);values.append(weights)
        conditions.append(condition);errors.append(error);counts.append(len(near))
    matrix = sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),
                           shape=(active.size,len(ids)))
    return matrix, ids, dict(stencil_condition_max=max(conditions,default=0.),
                            polynomial_reproduction_linf=max(errors,default=0.),
                            neighbor_count_max=max(counts,default=0))
