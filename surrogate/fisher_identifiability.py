"""
Fisher Information Matrix & Observability Module for Paper 1.

Implements the complementary innovation from Section 9 of the research plan:
    "Observability of defect after removing nuisance parameters"

Partitions parameters into:
    - Target defect parameters: d = [delta/D, phi, theta_d]
    - Nuisance parameters: n = [Re, sensor_bias, inflow_angle]

Effective Fisher Information Matrix via Schur Complement:
    F_eff = F_dd - F_dn * inv(F_nn) * F_nd

Provides:
    - D-optimality: log(det(F_eff))
    - E-optimality: lambda_min(F_eff)
    - Condition number: cond(F_eff)
    - Greedy optimal sensor subset selection for minimal sensor networks.
"""

import numpy as np


class FisherIdentifiabilityAnalyzer:
    """
    Fisher Information and Identifiability Analyzer for Hydrodynamic Defect Sensing.
    """

    def __init__(self, noise_cov=None):
        """
        Parameters:
            noise_cov : Diagonal or full covariance matrix R of sensor noise (shape n_obs x n_obs).
                        If None, identity matrix is assumed.
        """
        self.noise_cov = noise_cov

    def compute_fisher_matrix(self, jacobian, noise_cov=None):
        """
        Compute full Fisher Information Matrix:
            F = J^T * inv(R) * J
            
        Parameters:
            jacobian  : Sensitivity Jacobian matrix of shape (n_obs, n_params).
                        J_ij = d(y_i) / d(theta_j).
            noise_cov : Noise covariance matrix R (shape n_obs x n_obs).
            
        Returns:
            F : Fisher matrix of shape (n_params, n_params).
        """
        J = np.asarray(jacobian, dtype=float)
        R = noise_cov if noise_cov is not None else self.noise_cov
        
        if R is None:
            # Identity noise: F = J^T * J
            return J.T @ J
        else:
            R = np.asarray(R, dtype=float)
            if R.ndim == 1:
                # Diagonal noise variance
                inv_R = np.diag(1.0 / np.maximum(R, 1e-14))
                return J.T @ (inv_R @ J)
            else:
                inv_R = np.linalg.pinv(R)
                return J.T @ (inv_R @ J)

    def effective_fisher_matrix(self, F_full, n_target=3):
        """
        Compute effective Fisher Information Matrix for target parameters by eliminating
        nuisance parameters using the Schur complement:
            F_eff = F_dd - F_dn * inv(F_nn) * F_nd
            
        Parameters:
            F_full   : Full Fisher Information Matrix of shape (P, P).
            n_target : Number of target defect parameters (first n_target columns/rows).
            
        Returns:
            F_eff    : Effective Fisher matrix of shape (n_target, n_target).
        """
        F = np.asarray(F_full, dtype=float)
        P = F.shape[0]
        
        if P == n_target:
            return F  # No nuisance parameters
            
        F_dd = F[:n_target, :n_target]
        F_dn = F[:n_target, n_target:]
        F_nd = F[n_target:, :n_target]
        F_nn = F[n_target:, n_target:]
        
        # Check invertibility of nuisance block
        try:
            inv_F_nn = np.linalg.inv(F_nn)
        except np.linalg.LinAlgError:
            inv_F_nn = np.linalg.pinv(F_nn)
            
        F_eff = F_dd - (F_dn @ (inv_F_nn @ F_nd))
        # Ensure symmetry
        F_eff = 0.5 * (F_eff + F_eff.T)
        return F_eff

    def evaluate_metrics(self, F):
        """
        Evaluate D-optimality, E-optimality, and condition number for a Fisher matrix.
        
        Returns:
            dict: {
                "d_optimality": float (log det),
                "e_optimality": float (min eigenvalue),
                "condition_number": float,
                "eigenvalues": list
            }
        """
        eigvals = np.linalg.eigvalsh(F)
        # Numerical protection for non-negative eigenvalues
        eigvals_clean = np.maximum(eigvals, 1e-15)
        
        log_det = float(np.sum(np.log(eigvals_clean)))
        e_opt = float(np.min(eigvals_clean))
        max_eig = float(np.max(eigvals_clean))
        cond_num = float(max_eig / e_opt)
        
        return {
            "d_optimality": log_det,
            "e_optimality": e_opt,
            "condition_number": cond_num,
            "eigenvalues": eigvals_clean.tolist()
        }

    def select_optimal_sensor_subset(self, sensor_jacobians, k_sensors=3, n_target=3):
        """
        Greedy forward selection of optimal k-sensor subset maximizing log-determinant (D-optimality).
        
        Parameters:
            sensor_jacobians : List or 3D array of shape (N_sensors, n_sensor_obs, n_params).
            k_sensors        : Desired number of sensors to select.
            n_target         : Number of target parameters.
            
        Returns:
            selected_indices : List of integer sensor indices.
            history          : D-optimality score progression.
        """
        n_sensors = len(sensor_jacobians)
        selected = []
        score_history = []
        
        available = set(range(n_sensors))
        
        for step in range(k_sensors):
            best_score = -np.inf
            best_idx = None
            
            for idx in available:
                candidate = selected + [idx]
                # Stack Jacobians for candidate sensors
                stacked_J = np.vstack([sensor_jacobians[i] for i in candidate])
                F_full = self.compute_fisher_matrix(stacked_J)
                F_eff = self.effective_fisher_matrix(F_full, n_target=n_target)
                score = self.evaluate_metrics(F_eff)["d_optimality"]
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
            if best_idx is not None:
                selected.append(best_idx)
                available.remove(best_idx)
                score_history.append(best_score)
                
        return selected, score_history
