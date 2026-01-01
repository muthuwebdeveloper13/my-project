# src/modules/maximization.py
import numpy as np


class GMMMaximization:
    """
    Module 4: Maximization Step (M-Step)
    Updates GMM parameters using responsibilities
    """

    def __init__(self, reg_covar=1e-6):
        """
        reg_covar: small value added to diagonal for numerical stability
        """
        self.reg_covar = reg_covar

    def run_m_step(self, X, responsibilities):
        """
        Perform M-step of EM algorithm

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        responsibilities : ndarray of shape (n_samples, n_components)
            Posterior probabilities from E-step

        Returns
        -------
        params : dict
            Updated GMM parameters
        """

        n_samples, n_features = X.shape
        n_components = responsibilities.shape[1]

        # Effective number of points per cluster
        Nk = responsibilities.sum(axis=0)  # shape (K,)

        # ===============================
        # Update weights
        # ===============================
        weights = Nk / n_samples

        # ===============================
        # Update means
        # ===============================
        means = np.zeros((n_components, n_features))
        for k in range(n_components):
            means[k] = np.sum(
                responsibilities[:, k][:, np.newaxis] * X,
                axis=0
            ) / Nk[k]

        # ===============================
        # Update covariances
        # ===============================
        covariances = np.zeros((n_components, n_features, n_features))

        for k in range(n_components):
            diff = X - means[k]
            weighted_diff = responsibilities[:, k][:, np.newaxis] * diff

            covariances[k] = (
                weighted_diff.T @ diff
            ) / Nk[k]

            # Regularization for numerical stability
            covariances[k] += self.reg_covar * np.eye(n_features)

        return {
            "weights": weights,
            "means": means,
            "covariances": covariances
        }
