"""
Gaussian Process (GP) Surrogate Modeling Module for Paper 1.

Builds and validates multi-fidelity / multi-output GP models mapping input parameters:
    X = [Re, d, phi_rad, theta_d_rad]
to scalar hydrodynamic observables:
    Y in {mean_Cd, rms_Cd, mean_Cl, rms_Cl, St, Lr, sensor_Cp, ...}

Features:
    - Matérn 5/2 kernel with Automatic Relevance Determination (ARD)
    - Independent test validation metrics: R2, NRMSE, MAE
    - 95% uncertainty confidence intervals and empirical coverage calibration
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class ObservableGPSurrogate:
    """
    Gaussian Process Surrogate for a single or multiple scalar hydrodynamic observables.
    """

    def __init__(self, nu=2.5, n_restarts_optimizer=5, alpha=1e-6):
        """
        Parameters:
            nu                    : Matérn smoothness parameter (default 2.5 for C2 smoothness).
            n_restarts_optimizer  : Restarts for MLE kernel hyperparameter fitting.
            alpha                 : Numerical regularization / white noise floor.
        """
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Fit GP model on training data.
        
        Parameters:
            X_train : 2D array of shape (N, n_features).
            y_train : 1D or 2D array of shape (N,) or (N, 1).
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        X_norm = self.scaler_X.fit_transform(X_train)
        y_norm = self.scaler_y.fit_transform(y_train).ravel()

        n_features = X_train.shape[1]
        # ARD length scales: individual length scale per parameter
        length_scales = np.ones(n_features)
        
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=length_scales, length_scale_bounds=(1e-2, 1e2), nu=self.nu)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False
        )
        self.model.fit(X_norm, y_norm)
        self.is_fitted = True
        return self

    def predict(self, X_test, return_std=True):
        """
        Predict observable value and epistemic uncertainty std at new design points.
        
        Parameters:
            X_test     : 2D array of shape (M, n_features).
            return_std : If True, returns posterior standard deviation.
            
        Returns:
            y_pred, (y_std): Predicted means and standard deviations in physical units.
        """
        if not self.is_fitted:
            raise RuntimeError("Surrogate has not been fitted yet.")
            
        X_test = np.asarray(X_test)
        X_norm = self.scaler_X.transform(X_test)
        
        if return_std:
            y_norm_pred, y_norm_std = self.model.predict(X_norm, return_std=True)
            # Rescale back to physical units
            y_scale = self.scaler_y.scale_[0]
            y_mean = self.scaler_y.mean_[0]
            
            y_pred = y_norm_pred * y_scale + y_mean
            y_std = y_norm_std * y_scale
            return y_pred, y_std
        else:
            y_norm_pred = self.model.predict(X_norm, return_std=False)
            y_scale = self.scaler_y.scale_[0]
            y_mean = self.scaler_y.mean_[0]
            return y_norm_pred * y_scale + y_mean

    def evaluate(self, X_test, y_test):
        """
        Evaluate predictive quality against an independent holdout test dataset.
        
        Returns:
            dict: R2, NRMSE, MAE, coverage_95
        """
        y_test = np.asarray(y_test).ravel()
        y_pred, y_std = self.predict(X_test, return_std=True)
        
        residuals = y_test - y_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        y_range = float(np.max(y_test) - np.min(y_test))
        nrmse = float(rmse / y_range) if y_range > 1e-12 else 0.0
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0
        
        # Empirical 95% coverage: fraction of test points inside [y_pred - 1.96*y_std, y_pred + 1.96*y_std]
        ci_lower = y_pred - 1.96 * y_std
        ci_upper = y_pred + 1.96 * y_std
        coverage_95 = float(np.mean((y_test >= ci_lower) & (y_test <= ci_upper)))
        
        return {
            "R2": r2,
            "NRMSE": nrmse,
            "MAE": mae,
            "coverage_95": coverage_95
        }
