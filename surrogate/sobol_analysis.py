"""
Global Sensitivity Analysis (GSA) Module for Paper 1.

Implements variance-based Sobol sensitivity analysis using Saltelli's extension:
    - First-order indices: S_i = V_i / Var(Y)
    - Total-order indices: S_Ti = 1 - V_~i / Var(Y)

Maps input parameters:
    1. Re       : Reynolds number [80, 180]
    2. d        : Defect depth ratio [0.0, 0.10]
    3. phi_deg  : Defect half-width [5.0, 30.0] deg
    4. theta_deg: Defect location [0.0, 180.0] deg

Evaluates fast Gaussian Process surrogates over large sample matrices (e.g. N = 4096)
to provide converged Sobol sensitivity indices for every observable.
"""

import numpy as np


class SobolSensitivityAnalyzer:
    """
    Computes first-order and total-order Sobol sensitivity indices.
    """

    def __init__(self, bounds, names=None, seed=42):
        """
        Parameters:
            bounds : List of (min, max) tuples for each parameter, shape (D, 2).
            names  : Optional list of parameter names.
            seed   : Random seed for sampling.
        """
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(self.bounds)
        self.names = names if names is not None else [f"X{i}" for i in range(self.dim)]
        self.rng = np.random.default_rng(seed)

    def generate_saltelli_samples(self, n_samples=2048):
        """
        Generate Saltelli sample matrix of shape (n_samples * (2*D + 2), D).
        
        Uses two independent matrices A and B in [0, 1]^D, and forms cross-matrices:
            A, B, A_B^(i) for i = 1..D
        """
        N = n_samples
        D = self.dim
        
        # Generate independent quasi-random sample matrices A and B
        A_norm = self.rng.uniform(0.0, 1.0, size=(N, D))
        B_norm = self.rng.uniform(0.0, 1.0, size=(N, D))
        
        # Cross matrices A_B^(i): matrix A with i-th column replaced by B's i-th column
        A_B_norm = []
        for i in range(D):
            AB_i = A_norm.copy()
            AB_i[:, i] = B_norm[:, i]
            A_B_norm.append(AB_i)
            
        all_norm = np.vstack([A_norm, B_norm] + A_B_norm)
        
        # Scale to physical bounds
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        all_phys = low + all_norm * (high - low)
        
        return all_phys, N

    def compute_indices(self, Y, N):
        """
        Compute first-order S_i and total-order S_Ti from model evaluations.
        
        Parameters:
            Y : 1D array of length N * (D + 2) containing model evaluations.
            N : Number of base Saltelli samples.
            
        Returns:
            dict: {param_name: {"S1": float, "ST": float}}
        """
        D = self.dim
        Y = np.asarray(Y, dtype=float).ravel()
        
        y_A = Y[:N]
        y_B = Y[N:2*N]
        y_AB = [Y[(2 + i)*N : (3 + i)*N] for i in range(D)]
        
        var_y = np.var(y_A)
        if var_y < 1e-15:
            # Constant output -> zero sensitivity
            return {name: {"S1": 0.0, "ST": 0.0} for name in self.names}
            
        mean_y = np.mean(y_A)
        
        results = {}
        for i in range(D):
            # Jansen (1999) / Saltelli estimator for total-order:
            # ST_i = (1 / (2*N)) * sum( (y_A - y_AB_i)^2 ) / var(y)
            diff_T = y_A - y_AB[i]
            V_T = (1.0 / (2.0 * N)) * np.sum(diff_T**2)
            S_T = V_T / var_y
            
            # Saltelli (2002) estimator for first-order:
            # S1_i = ( (1 / N) * sum( y_B * (y_AB_i - y_A) ) ) / var(y)
            diff_1 = y_B * (y_AB[i] - y_A)
            V_1 = (1.0 / N) * np.sum(diff_1)
            S_1 = V_1 / var_y
            
            # Clean boundaries
            S_1 = float(np.clip(S_1, 0.0, 1.0))
            S_T = float(np.clip(S_T, 0.0, 1.5))
            
            results[self.names[i]] = {
                "S1": S_1,
                "ST": S_T
            }
            
        return results
