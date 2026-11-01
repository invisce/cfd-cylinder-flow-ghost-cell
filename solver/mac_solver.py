import numpy as np
import scipy.sparse as sp
import mac_engine

def build_poisson_matrix(nx, ny, dx, dy, mask):
    """
    Builds the exact discrete Finite Volume Laplacian matrix for the textbook MAC grid.
    
    Domain layout:
      - nx cells along x in [0, Lx], dx = Lx / nx
      - ny cells along y in [0, Ly], dy = Ly / ny
      - Scalar unknowns Phi[j, i] at cell centers (j in [0, ny-1], i in [0, nx-1])
      - Unknown index c = j * nx + i, total unknowns N = ny * nx
      
    Boundary conditions on pressure correction Phi:
      - Inlet (x=0, face i=0): Homogeneous Neumann (dPhi/dx = 0)
      - Walls (y=0, j=0 face and y=Ly, j=ny face): Homogeneous Neumann (dPhi/dy = 0)
      - Cylinder solid faces: Homogeneous Neumann (dPhi/dn = 0)
      - Outlet (x=Lx, face i=nx): Dirichlet Phi = 0 at physical boundary distance dx/2.
        Gradient: (0 - Phi[j, nx-1]) / (dx/2) = -2*Phi[j, nx-1]/dx
        Discrete flux contribution to row: -2 / dx^2 on the diagonal.
        
    Properties:
      - Symmetric, weakly diagonally dominant fluid block; strictly dominant at outlet.
      - Fluid block is negative definite if every fluid component reaches the outlet.
        Positive solid dummy diagonals make the full matrix indefinite.
      - Nonsingular for this connected channel (no pinning needed).
      - Divergence matches discrete Laplacian identically, guaranteeing exact global
        mass conservation without post-projection velocity scaling.
    """
    N = ny * nx
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    
    # Pre-allocate COO arrays for fast assembly
    max_nnz = N * 5
    rows = np.empty(max_nnz, dtype=np.int32)
    cols = np.empty(max_nnz, dtype=np.int32)
    data = np.empty(max_nnz, dtype=np.float64)
    ptr = 0
    
    for j in range(ny):
        for i in range(nx):
            c = j * nx + i
            if mask[j, i]:
                # Solid cell: decoupled dummy equation Phi = 0
                rows[ptr] = c; cols[ptr] = c; data[ptr] = 1.0; ptr += 1
                continue
                
            diag = 0.0
            
            # 1. West face
            if i == 0:
                pass # Inlet Neumann (dPhi/dx = 0)
            elif mask[j, i - 1]:
                pass # Solid wall Neumann (dPhi/dx = 0)
            else:
                rows[ptr] = c; cols[ptr] = c - 1; data[ptr] = inv_dx2; ptr += 1
                diag -= inv_dx2
                
            # 2. East face
            if i == nx - 1:
                # Outlet Dirichlet Phi = 0 at distance dx/2
                diag -= 2.0 * inv_dx2
            elif mask[j, i + 1]:
                pass # Solid wall Neumann (dPhi/dx = 0)
            else:
                rows[ptr] = c; cols[ptr] = c + 1; data[ptr] = inv_dx2; ptr += 1
                diag -= inv_dx2
                
            # 3. South face
            if j == 0:
                pass # Bottom wall Neumann (dPhi/dy = 0)
            elif mask[j - 1, i]:
                pass # Solid wall Neumann (dPhi/dy = 0)
            else:
                rows[ptr] = c; cols[ptr] = c - nx; data[ptr] = inv_dy2; ptr += 1
                diag -= inv_dy2
                
            # 4. North face
            if j == ny - 1:
                pass # Top wall Neumann (dPhi/dy = 0)
            elif mask[j + 1, i]:
                pass # Solid wall Neumann (dPhi/dy = 0)
            else:
                rows[ptr] = c; cols[ptr] = c + nx; data[ptr] = inv_dy2; ptr += 1
                diag -= inv_dy2
                
            rows[ptr] = c; cols[ptr] = c; data[ptr] = diag; ptr += 1
            
    A = sp.csc_matrix((data[:ptr], (rows[:ptr], cols[:ptr])), shape=(N, N))
    return A

def calculate_mac_divergence(u, v, dx, dy, mask):
    """
    Computes discrete divergence div = (u[j, i+1] - u[j, i])/dx + (v[j+1, i] - v[j, i])/dy
    for all fluid cells in the domain.
    """
    return mac_engine.calculate_mac_divergence_cpp(u, v, mask, dx, dy)
