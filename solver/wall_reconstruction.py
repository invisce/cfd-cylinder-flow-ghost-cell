"""No-slip, divergence-free wall gradient from open-face velocity averages.

Fit a local streamfunction psi=d**2 times a quadratic polynomial in normal
distance and arc length. Face averages are exact streamfunction differences,
so partially open faces do not masquerade as values at Cartesian centers.
"""
import numpy as np
from scipy.sparse import csr_matrix


def reconstruct_wall_gradient(u, v, geom, operator=None):
    """Apply the same physical reconstruction in both cached and standalone paths."""
    if u.shape != geom.ax.shape or v.shape != geom.ay.shape:
        raise ValueError('Wall velocity arrays have invalid shapes')
    if operator is None:
        cut = geom.cell_type == 0
        operator = build_wall_gradient(geom, np.arctan2(
            geom.bndry_normal_unit_y[cut], geom.bndry_normal_unit_x[cut]))
    field = np.concatenate((u.ravel(), v.ravel()))
    if not np.isfinite(field[operator.indices]).all():
        raise ValueError('Wall reconstruction has non-finite active face velocities')
    return operator @ field


def _segments(lo, hi, coordinate, normal_center, tangent_center, radius):
    offset = coordinate-normal_center
    if abs(offset) >= radius:
        return [(lo, hi)]
    half = np.sqrt(max(0., radius**2-offset**2))
    return [(a, b) for a, b in ((lo, min(hi, tangent_center-half)),
                               (max(lo, tangent_center+half), hi)) if b-a > 1e-14]


def cartesian_circle_basis(x, y, geom, angles):
    h = max(geom.dx,geom.dy)
    xx,yy = np.asarray(x)-geom.xc,np.asarray(y)-geom.yc
    q = (xx-geom.R*np.cos(angles))/h
    z = (yy-geom.R*np.sin(angles))/h
    s = (xx*xx+yy*yy-geom.R**2)/(2*geom.R*h)
    sx,sy = xx/(geom.R*h),yy/(geom.R*h)
    one,zero = np.ones_like(q),np.zeros_like(q)
    polynomial = np.stack((one,q,z,q*q,q*z,z*z),axis=-1)
    px = np.stack((zero,one,zero,2*q,z,zero),axis=-1)/h
    py = np.stack((zero,zero,one,zero,q,2*z),axis=-1)/h
    pxx = np.array([0,0,0,2,0,0])/h**2
    pxy = np.array([0,0,0,0,1,0])/h**2
    pyy = np.array([0,0,0,0,0,2])/h**2
    s,sx,sy = s[...,None],sx[...,None],sy[...,None]
    xx2 = h*(2*(sx*sx+s/(geom.R*h))*polynomial+4*s*sx*px+s*s*pxx)
    xy2 = h*(2*sx*sy*polynomial+2*s*sx*py+2*s*sy*px+s*s*pxy)
    yy2 = h*(2*(sy*sy+s/(geom.R*h))*polynomial+4*s*sy*py+s*s*pyy)
    return dict(psi=h*s*s*polynomial,u=h*(2*s*sy*polynomial+s*s*py),
                v=-h*(2*s*sx*polynomial+s*s*px),ux=xy2,uy=yy2,vx=-xx2,vy=-xy2)


def _basis(x, y, geom, angle, h, cartesian=False):
    if cartesian:
        xx, yy = x-geom.xc, y-geom.yc
        q = (xx-geom.R*np.cos(angle))/h
        z = (yy-geom.R*np.sin(angle))/h
        s = (xx*xx+yy*yy-geom.R**2)/(2*geom.R*h)
        return h*s*s*np.array([1.,q,z,q*q,q*z,z*z])
    xx, yy = x-geom.xc, y-geom.yc
    distance = (np.hypot(xx, yy)-geom.R)/h
    arc = geom.R*np.arctan2(np.sin(np.arctan2(yy, xx)-angle),
                           np.cos(np.arctan2(yy, xx)-angle))/h
    return h*distance**2*np.array([1., distance, arc, distance**2, distance*arc, arc**2])


def build_streamfunction_fit(geom, angles, cartesian=False):
    """Return six local coefficients per angle from physical open-face averages."""
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('Wall angles must be a finite one-dimensional array')
    h = max(geom.dx, geom.dy)
    u_size = geom.ny*(geom.nx+1)
    rows, columns, values = [], [], []
    for row, angle in enumerate(angles):
        xwall, ywall = geom.xc+geom.R*np.cos(angle), geom.yc+geom.R*np.sin(angle)
        for width in (3., 4., 5.):
            design, indices, weights = [], [], []
            for component in (0, 1):
                count_x, count_y = (geom.nx+1, geom.ny) if component == 0 else (geom.nx, geom.ny+1)
                for j in range(max(0, int((ywall-width*h)/geom.dy)-1),
                               min(count_y, int((ywall+width*h)/geom.dy)+2)):
                    for i in range(max(0, int((xwall-width*h)/geom.dx)-1),
                                   min(count_x, int((xwall+width*h)/geom.dx)+2)):
                        aperture = geom.ax[j, i] if component == 0 else geom.ay[j, i]
                        if aperture <= 0:
                            continue
                        if component == 0:
                            x = i*geom.dx
                            segments = _segments(j*geom.dy, (j+1)*geom.dy, x, geom.xc, geom.yc, geom.R)
                            length = sum(b-a for a, b in segments)
                            if not length:
                                continue
                            y = sum((a+b)/2*(b-a) for a, b in segments)/length
                            basis = sum((_basis(x, b, geom, angle, h,cartesian)-_basis(x, a, geom, angle, h,cartesian)
                                         for a, b in segments), np.zeros(6))/length
                            index = j*(geom.nx+1)+i
                        else:
                            y = j*geom.dy
                            segments = _segments(i*geom.dx, (i+1)*geom.dx, y, geom.yc, geom.xc, geom.R)
                            length = sum(b-a for a, b in segments)
                            if not length:
                                continue
                            x = sum((a+b)/2*(b-a) for a, b in segments)/length
                            basis = -sum((_basis(b, y, geom, angle, h,cartesian)-_basis(a, y, geom, angle, h,cartesian)
                                          for a, b in segments), np.zeros(6))/length
                            index = u_size+j*geom.nx+i
                        distance = np.hypot(x-xwall, y-ywall)/h
                        if distance > width+64*np.finfo(float).eps*max(geom.Lx,geom.Ly)/h:
                            continue
                        design.append(basis)
                        indices.append(index)
                        weights.append(1/(1+distance**2))
            if len(design) < 6:
                continue
            weighted = np.asarray(design)*np.asarray(weights)[:, None]
            if np.linalg.matrix_rank(weighted, tol=1e-11) == 6:
                coefficients = np.linalg.pinv(weighted, rcond=1e-12)*weights
                for k in range(6):
                    rows.extend([6*row+k]*len(indices))
                    columns.extend(indices)
                    values.extend(coefficients[k])
                break
        else:
            raise ValueError('Insufficient independent open faces for wall reconstruction')
    return csr_matrix((values, (rows, columns)), shape=(6*len(angles), u_size+(geom.ny+1)*geom.nx))


def build_wall_gradient(geom, angles):
    """Return CSR mapping concatenated u/v face averages to tangential du/dn."""
    fit = build_streamfunction_fit(geom, angles)
    return -2/max(geom.dx, geom.dy)*fit[np.arange(0, fit.shape[0], 6)]
