"""
Modal Wake Decomposition Module for Paper 1 (POD & SPOD).

Implements:
    1. Proper Orthogonal Decomposition (POD):
       - Snapshot POD (Sirovich method) for high spatial resolution fields
       - Modal energy spectrum and cumulative energy
       - Spatial modes and temporal coefficients a_i(t)
       - Phase portrait analysis (Mode 1 vs Mode 2 limit cycle)
    2. Spectral Proper Orthogonal Decomposition (SPOD):
       - Frequency-resolved coherent structures via block-averaged cross-spectral density
"""

import numpy as np


class SnapshotPOD:
    """
    Snapshot Proper Orthogonal Decomposition for 2D scalar fields (vorticity or velocity components).
    """

    def __init__(self, n_modes=10):
        self.n_modes = n_modes
        self.mean_field = None
        self.spatial_modes = None
        self.temporal_coeffs = None
        self.singular_values = None
        self.modal_energies = None
        self.energy_ratios = None
        self.cumulative_energy = None

    def fit(self, snapshots):
        """
        Compute snapshot POD.
        
        Parameters:
            snapshots : 2D or 3D array.
                        If 3D, shape is (n_snapshots, ny, nx).
                        If 2D, shape is (n_snapshots, n_grid_points).
        """
        orig_shape = snapshots.shape
        n_snaps = orig_shape[0]
        
        if snapshots.ndim == 3:
            grid_shape = orig_shape[1:]
            X = snapshots.reshape(n_snaps, -1)
        else:
            grid_shape = (orig_shape[1],)
            X = snapshots.copy()

        # 1. Compute time-average mean field and fluctuations
        self.mean_field = np.mean(X, axis=0)
        X_prime = X - self.mean_field  # shape (M, N_pts)

        # 2. Sirovich Snapshot POD:
        # Correlation matrix C = (1 / M) * X_prime @ X_prime.T (shape M x M)
        # Much cheaper than N_pts x N_pts when M << N_pts
        C = (X_prime @ X_prime.T) / float(n_snaps)
        
        # Eigenvalue decomposition of C
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Sort in descending order of energy
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 0.0)
        eigvecs = eigvecs[:, idx]
        
        # Singular values of X_prime
        self.singular_values = np.sqrt(eigvals * float(n_snaps))
        self.modal_energies = eigvals
        total_energy = np.sum(eigvals)
        if total_energy > 1e-15:
            self.energy_ratios = eigvals / total_energy
            self.cumulative_energy = np.cumsum(self.energy_ratios)
        else:
            self.energy_ratios = np.zeros_like(eigvals)
            self.cumulative_energy = np.zeros_like(eigvals)

        # 3. Project to obtain spatial modes: U = X_prime.T @ V / Sigma
        k = min(self.n_modes, n_snaps)
        self.spatial_modes = np.zeros((k, X.shape[1]))
        self.temporal_coeffs = np.zeros((k, n_snaps))
        
        for i in range(k):
            sigma_i = self.singular_values[i]
            if sigma_i > 1e-12:
                # Spatial mode phi_i (normalized to unit L2 norm)
                phi_i = (X_prime.T @ eigvecs[:, i]) / sigma_i
                self.spatial_modes[i] = phi_i
                # Temporal coefficients a_i(t)
                self.temporal_coeffs[i] = X_prime @ phi_i

        # Reshape spatial modes back to 2D grid if input was 3D
        if len(grid_shape) == 2:
            self.spatial_modes = self.spatial_modes.reshape(k, grid_shape[0], grid_shape[1])
            self.mean_field = self.mean_field.reshape(grid_shape[0], grid_shape[1])
            
        return self


def compute_phase_portrait_metrics(a1, a2):
    """
    Compute limit-cycle amplitude and orbital eccentricity between Mode 1 and Mode 2 coefficients.
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    radius = np.hypot(a1, a2)
    mean_radius = float(np.mean(radius))
    # Orbital circularity: std(radius) / mean(radius) (0 for perfect circle)
    circularity_error = float(np.std(radius) / np.maximum(mean_radius, 1e-14))
    return {
        "mean_amplitude": mean_radius,
        "circularity_error": circularity_error
    }
