"""Geometric staggered-volume wall and open-face quadrature.

No primal-cell barycentric force spreading. Shared Cartesian fluxes are sampled
once over their physical open intervals. All stencils consume face averages.
"""
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from wall_reconstruction import _segments, _basis, build_streamfunction_fit, cartesian_circle_basis


def derivative_basis(x, y, angles, geom, cartesian=False):
    if cartesian:
        return cartesian_circle_basis(x,y,geom,angles)
    h = max(geom.dx, geom.dy)
    xx, yy = np.asarray(x)-geom.xc, np.asarray(y)-geom.yc
    r = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    d = ((r-geom.R)/h)[:, None]
    s = (geom.R/h*np.arctan2(np.sin(theta-angles), np.cos(theta-angles)))[:, None]
    a, b = np.array([2,3,2,4,3,2]), np.array([0,0,1,0,1,2])
    radial = a*d**(a-1)*s**b
    angular = geom.R*b*d**a*s**np.maximum(b-1,0)
    rr = a*(a-1)/h*d**(a-2)*s**b
    rt = geom.R/h*a*b*d**(a-1)*s**np.maximum(b-1,0)
    tt = geom.R**2/h*b*(b-1)*d**a*s**np.maximum(b-2,0)
    nx, ny = (xx/r)[:, None], (yy/r)[:, None]
    tx, ty = -ny, nx
    transverse = radial/r[:, None]+tt/r[:, None]**2
    cross = rt/r[:, None]-angular/r[:, None]**2
    ux = rr*nx*ny+transverse*tx*ty+cross*(nx*ty+tx*ny)
    uy = rr*ny**2+transverse*ty**2+2*cross*ny*ty
    vx = -(rr*nx**2+transverse*tx**2+2*cross*nx*tx)
    return dict(u=radial*ny+angular*ty/r[:, None],
                v=-radial*nx-angular*tx/r[:, None], ux=ux, uy=uy, vx=vx, vy=-ux,
                wall_rate=-rr)


def point_operator(fit, patch, basis):
    count = len(patch)
    operator = csr_matrix((basis.ravel(), (np.repeat(np.arange(count),6),
                 (6*np.asarray(patch)[:,None]+np.arange(6)).ravel())), shape=(count,fit.shape[0]))
    return operator @ fit


def average_basis(geom, component, j, i, angle, cartesian=False):
    """Exact local polynomial average on a physical velocity face."""
    h = max(geom.dx, geom.dy)
    if component == 'u':
        x = i*geom.dx
        spans = _segments(j*geom.dy, (j+1)*geom.dy, x, geom.xc, geom.yc, geom.R)
        value = sum((_basis(x,b,geom,angle,h,cartesian)-_basis(x,a,geom,angle,h,cartesian)
                     for a,b in spans), np.zeros(6))
    else:
        y = j*geom.dy
        spans = _segments(i*geom.dx, (i+1)*geom.dx, y, geom.yc, geom.xc, geom.R)
        value = -sum((_basis(b,y,geom,angle,h,cartesian)-_basis(a,y,geom,angle,h,cartesian)
                      for a,b in spans), np.zeros(6))
    length = sum(b-a for a,b in spans)
    return value/length if length > 0 else np.zeros(6)


def face_stencil(geom, component, axis, reach=2., cartesian=False):
    aperture = {('u','x'):geom.ax_mid, ('u','y'):geom.ay_u,
                ('v','x'):geom.ax_v, ('v','y'):geom.ay_mid}[component,axis]
    centered = (component,axis) in (('u','x'),('v','y'))
    offset = .5 if centered else 0.
    h = max(geom.dx, geom.dy)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    faces, angles, points, lengths, patch = [], [], [], [], []
    for j in range(max(0,int((geom.yc-geom.R-reach*h)/geom.dy)-1),
                   min(aperture.shape[0],int((geom.yc+geom.R+reach*h)/geom.dy)+2)):
        for i in range(max(0,int((geom.xc-geom.R-reach*h)/geom.dx)-1),
                       min(aperture.shape[1],int((geom.xc+geom.R+reach*h)/geom.dx)+2)):
            if not centered and ((axis == 'x' and i in (0,geom.nx)) or
                                 (axis == 'y' and j in (0,geom.ny))):
                continue  # Physical outer boundary conditions have their own flux.
            x, y = (i+offset)*geom.dx, (j+offset)*geom.dy
            tolerance = 64*np.finfo(float).eps*max(geom.Lx, geom.Ly)
            if aperture[j,i] <= 0 or np.hypot(x-geom.xc,y-geom.yc) > geom.R+reach*h+tolerance:
                continue
            if axis == 'x':
                lo, hi = (j*geom.dy,(j+1)*geom.dy) if centered else (
                    max(0.,(j-.5)*geom.dy),min(geom.Ly,(j+.5)*geom.dy))
                intervals = _segments(lo,hi,x,geom.xc,geom.yc,geom.R)
            else:
                lo, hi = (i*geom.dx,(i+1)*geom.dx) if centered else (
                    max(0.,(i-.5)*geom.dx),min(geom.Lx,(i+.5)*geom.dx))
                intervals = _segments(lo,hi,y,geom.yc,geom.xc,geom.R)
            if not intervals:
                continue
            k = len(faces)
            faces.append(j*aperture.shape[1]+i)
            angles.append(np.arctan2(y-geom.yc,x-geom.xc))
            for low,high in intervals:
                coordinate = (low+high)/2+(high-low)/2*nodes
                points.extend([(x,t) if axis=='x' else (t,y) for t in coordinate])
                lengths.extend((high-low)/2*weights)
                patch.extend([k]*len(nodes))
    fit = build_streamfunction_fit(geom,angles,cartesian)
    patch = np.asarray(patch,dtype=int)
    xy = np.asarray(points).reshape(-1,2)
    basis = derivative_basis(xy[:,0],xy[:,1],np.asarray(angles)[patch],geom,cartesian)
    average = csr_matrix((lengths,(patch,np.arange(len(patch)))),shape=(len(faces),len(patch)))
    # Restore two-point diffusion of the unresolved reconstruction residual.
    # Smooth fitted fields see zero correction; unresolved face modes retain
    # the ordinary conservative dissipative flux.
    link_rows, link_cols, link_values, residual_basis = [], [], [], []
    velocity_shape = geom.ax.shape if component == 'u' else geom.ay.shape
    offset = 0 if component == 'u' else geom.ax.size
    spacing = geom.dx if axis == 'x' else geom.dy
    for k, face in enumerate(faces):
        j, i = np.unravel_index(face, aperture.shape)
        low = (j,i) if centered else (j-1,i) if axis == 'y' else (j,i-1)
        high = (j,i+1) if centered and axis == 'x' else (j+1,i) if centered else (j,i)
        scale = aperture[j,i]*(geom.dy if axis == 'x' else geom.dx)/spacing
        residual_basis.append(scale*(average_basis(geom,component,*high,angles[k],cartesian)-
                                     average_basis(geom,component,*low,angles[k],cartesian)))
        for position, sign in ((low,-1), (high,1)):
            physical_aperture = geom.ax if component == 'u' else geom.ay
            if physical_aperture[position] == 0:
                continue  # Covered-face diagnostic values are not fluid DOFs.
            link_rows.append(k)
            link_cols.append(offset+np.ravel_multi_index(position,velocity_shape))
            link_values.append(sign*scale)
    links = csr_matrix((link_values,(link_rows,link_cols)), shape=(len(faces),fit.shape[1]))
    residual = links-point_operator(fit,np.arange(len(faces)),np.asarray(residual_basis).reshape(-1,6))
    return SimpleNamespace(indices=np.asarray(faces,dtype=int),
        u=point_operator(fit,patch,basis['u']),v=point_operator(fit,patch,basis['v']),
        gradient=average @ point_operator(fit,patch,basis[component+axis])+residual,
        patch=patch,lengths=np.asarray(lengths))


def correct_flux(flux, stencil, field, rho, nu, component, axis, advection=True):
    value = rho*nu*(stencil.gradient @ field)
    if advection:
        u, v = stencil.u @ field, stencil.v @ field
        product = u*u if (component,axis)==('u','x') else v*v if (component,axis)==('v','y') else u*v
        value -= rho*np.bincount(stencil.patch,weights=stencil.lengths*product,minlength=len(stencil.indices))
    flux.ravel()[stencil.indices] = value


def diffusion_correction(geom, component, axis, reconstructed):
    aperture = {('u','x'):geom.ax_mid, ('u','y'):geom.ay_u,
                ('v','x'):geom.ax_v, ('v','y'):geom.ay_mid}[component,axis]
    velocity_aperture = geom.ax if component == 'u' else geom.ay
    centered = (component,axis) in (('u','x'),('v','y'))
    normal = geom.dx if axis == 'x' else geom.dy
    length = geom.dy if axis == 'x' else geom.dx
    offset = 0 if component == 'u' else geom.ax.size
    excluded = set(reconstructed.indices)
    h = max(geom.dx,geom.dy)
    inner, width = max(3*h,.5*geom.R), .5*geom.R
    rows, columns, values, faces = [], [], [], []
    for j in range(max(0,int((geom.yc-geom.R-inner-width)/geom.dy)-1),
                   min(aperture.shape[0],int((geom.yc+geom.R+inner+width)/geom.dy)+2)):
        for i in range(max(0,int((geom.xc-geom.R-inner-width)/geom.dx)-1),
                       min(aperture.shape[1],int((geom.xc+geom.R+inner+width)/geom.dx)+2)):
            face = j*aperture.shape[1]+i
            if face in excluded or aperture[j,i] < 1-1e-12:
                continue
            x,y = (i+(.5 if centered else 0))*geom.dx, (j+(.5 if centered else 0))*geom.dy
            t = np.clip((np.hypot(x-geom.xc,y-geom.yc)-geom.R-inner)/width,0.,1.)
            edge = np.clip((min(x,geom.Lx-x,y,geom.Ly-y)-2*h)/width,0.,1.)
            weight = (1-t*t*(3-2*t))*edge*edge*(3-2*edge)
            if weight <= 0:
                continue
            entries = []
            if centered:
                for shift,coefficient in zip((-1,0,1,2),(1,-3,3,-1)):
                    position = (j,i+shift) if axis == 'x' else (j+shift,i)
                    entries.append((position,coefficient/(24*normal)))
            else:
                for transverse,average in zip((-1,0,1),(1/24,22/24,1/24)):
                    for shift,coefficient in zip((-2,-1,0,1),(1,-15,15,-1)):
                        position = (j+shift,i+transverse) if axis == 'y' else (j+transverse,i+shift)
                        entries.append((position,average*coefficient/(12*normal)))
                low = (j-1,i) if axis == 'y' else (j,i-1)
                entries.extend([(low,1/normal),((j,i),-1/normal)])
            if any(not (0 <= jj < velocity_aperture.shape[0] and 0 <= ii < velocity_aperture.shape[1])
                   or velocity_aperture[jj,ii] < 1-1e-12 for (jj,ii),_ in entries):
                continue
            row = len(faces)
            faces.append(face)
            for position,value in entries:
                rows.append(row)
                columns.append(offset+np.ravel_multi_index(position,velocity_aperture.shape))
                values.append(weight*length*value)
    correction = csr_matrix((values,(rows,columns)),shape=(len(faces),geom.ax.size+geom.ay.size))
    base_rows, base_columns, base_values = [], [], []
    for row,face in enumerate(faces):
        j,i = np.unravel_index(face,aperture.shape)
        low = (j,i) if centered else (j-1,i) if axis == 'y' else (j,i-1)
        high = (j,i+1) if centered and axis == 'x' else (j+1,i) if centered else (j,i)
        for position,sign in ((low,-1),(high,1)):
            base_rows.append(row)
            base_columns.append(offset+np.ravel_multi_index(position,velocity_aperture.shape))
            base_values.append(sign*length/normal)
    base = csr_matrix((base_values,(base_rows,base_columns)),shape=correction.shape)
    return SimpleNamespace(indices=np.asarray(faces,dtype=int),gradient=base+correction,correction=correction)


def diffusion_stencil(geom, component, axis, original, near):
    regular = diffusion_correction(geom,component,axis,near)
    from scipy.sparse import vstack
    indices = np.r_[near.indices,regular.indices]
    gradient = vstack((near.gradient,regular.gradient),format='csr')
    old = {int(face):row for row,face in enumerate(original.indices)}
    aperture = {('u','x'):geom.ax_mid, ('u','y'):geom.ay_u,
                ('v','x'):geom.ax_v, ('v','y'):geom.ay_mid}[component,axis]
    centered = (component,axis) in (('u','x'),('v','y'))
    shape = geom.ax.shape if component == 'u' else geom.ay.shape
    offset = 0 if component == 'u' else geom.ax.size
    scale = (geom.dy/geom.dx) if axis == 'x' else (geom.dx/geom.dy)
    rows,columns,values = [],[],[]
    for row,face in enumerate(indices):
        if int(face) in old:
            k = old[int(face)]
            lo,hi = original.gradient.indptr[k:k+2]
            rows.extend([row]*(hi-lo))
            columns.extend(original.gradient.indices[lo:hi])
            values.extend(original.gradient.data[lo:hi])
        else:
            j,i = np.unravel_index(face,aperture.shape)
            low = (j,i) if centered else (j-1,i) if axis == 'y' else (j,i-1)
            high = (j,i+1) if centered and axis == 'x' else (j+1,i) if centered else (j,i)
            for position,sign in ((low,-1),(high,1)):
                rows.append(row)
                columns.append(offset+np.ravel_multi_index(position,shape))
                values.append(sign*scale*aperture[j,i])
    baseline = csr_matrix((values,(rows,columns)),shape=gradient.shape)
    return SimpleNamespace(indices=indices,gradient=gradient,correction=gradient-baseline)


def wall_stencil(geom):
    cuts = [0.,2*np.pi]
    for center,spacing,kind in ((geom.xc,geom.dx,'x'),(geom.yc,geom.dy,'y')):
        for k in range(int(np.floor(2*(center-geom.R)/spacing)),
                       int(np.ceil(2*(center+geom.R)/spacing))+1):
            value = (k*spacing/2-center)/geom.R
            if abs(value) < 1:
                theta = np.arccos(value) if kind=='x' else np.arcsin(value)
                cuts.extend([theta%(2*np.pi),(-theta if kind=='x' else np.pi-theta)%(2*np.pi)])
    cuts = np.unique(cuts)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    angles, theta, ds, patch = [], [], [], []
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        if hi-lo < 1e-12:
            continue
        patch.extend([len(angles)]*3)
        angles.append((lo+hi)/2)
        theta.extend((lo+hi)/2+(hi-lo)/2*nodes)
        ds.extend(geom.R*(hi-lo)/2*weights)
    theta, patch = np.asarray(theta),np.asarray(patch,dtype=int)
    x, y = geom.xc+geom.R*np.cos(theta),geom.yc+geom.R*np.sin(theta)
    fit = build_streamfunction_fit(geom,angles)
    basis = derivative_basis(x,y,np.asarray(angles)[patch],geom)
    return SimpleNamespace(rate=point_operator(fit,patch,basis['wall_rate']),
        tx=-np.sin(theta),ty=np.cos(theta),ds=np.asarray(ds),
        u_indices=(np.floor(y/geom.dy).astype(int)*(geom.nx+1)+np.floor(x/geom.dx+.5).astype(int)),
        v_indices=(np.floor(y/geom.dy+.5).astype(int)*geom.nx+np.floor(x/geom.dx).astype(int)))


def owned_volumes(geom, component):
    """Assign partial fluid volumes with covered central faces to exterior DOFs.

    Transfer their complete momentum RHS through the same map at each step.
    The physical fluid volume is retained; covered velocity DOFs stay zero.
    """
    aperture = geom.ax if component == 'u' else geom.ay
    fraction = geom.u_volume_fraction if component == 'u' else geom.v_volume_fraction
    volume = fraction.copy()*geom.dx*geom.dy
    # Geometry fractions are relative to boxes clipped to the physical domain.
    # Outer staggered boxes have half the nominal width/height.
    if component == 'u':
        volume[:, [0, -1]] *= .5
    else:
        volume[[0, -1], :] *= .5
    donors = np.flatnonzero((aperture == 0) & (fraction > 0))
    masters = []
    for donor in donors:
        j, i = np.unravel_index(donor, aperture.shape)
        x = (i+(0 if component == 'u' else .5))*geom.dx-geom.xc
        y = (j+(.5 if component == 'u' else 0))*geom.dy-geom.yc
        scaled_x, scaled_y = abs(x/geom.dx), abs(y/geom.dy)
        tie_tolerance = 64*np.finfo(float).eps*max(1., scaled_x, scaled_y)
        sj, si = ((0, 1 if x >= 0 else -1) if scaled_x >= scaled_y-tie_tolerance
                  else (1 if y >= 0 else -1, 0))
        while True:
            j, i = j+sj, i+si
            if not (0 <= j < aperture.shape[0] and 0 <= i < aperture.shape[1]):
                raise ValueError('Covered momentum volume has no exterior owner')
            if aperture[j, i] > 0 and fraction[j, i] >= .3:
                masters.append(np.ravel_multi_index((j, i), aperture.shape))
                break
    masters = np.asarray(masters, dtype=int)
    np.add.at(volume.ravel(), masters, volume.ravel()[donors])
    volume.ravel()[donors] = 0.
    return volume, donors, masters


def transfer_covered_rhs(rhs, donors, masters):
    np.add.at(rhs.ravel(), masters, rhs.ravel()[donors])
    rhs.ravel()[donors] = 0.
