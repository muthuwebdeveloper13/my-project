import numpy as np
from typing import Tuple, Dict
from src.modules.expectation import GMMExpectation  # for log-likelihood

class GMMMaximization:
    """
    Module 4: Maximization Step (M-Step) for Gaussian Mixture Model (GMM)
    ---------------------------------------------------------------------
    Updates GMM parameters based on responsibilities γ(z_nk)
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps  # Numerical stability

    # --------------------------------------------------
    # Update Means μ_k
    # --------------------------------------------------
    def update_means(self, X: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        Args:
            X     : (N, D) data matrix
            gamma : (N, K) responsibilities
        Returns:
            means : (K, D)
        """
        N_k = gamma.sum(axis=0)  # shape (K,)
        means = (gamma.T @ X) / (N_k[:, np.newaxis] + self.eps)
        return means

    # --------------------------------------------------
    # Update Covariances Σ_k
    # --------------------------------------------------
    def update_covariances(self, X: np.ndarray, gamma: np.ndarray, means: np.ndarray) -> np.ndarray:
        """
        Args:
            X     : (N, D)
            gamma : (N, K)
            means : (K, D)
        Returns:
            covariances : (K, D, D)
        """
        N, D = X.shape
        K = gamma.shape[1]
        covariances = np.zeros((K, D, D))

        for k in range(K):
            diff = X - means[k]  # (N, D)
            weighted_diff = diff.T * gamma[:, k]  # (D, N)
            covariances[k] = (weighted_diff @ diff) / (gamma[:, k].sum() + self.eps)
            # Numerical stability
            covariances[k] += np.eye(D) * self.eps

        return covariances

    # --------------------------------------------------
    # Update Mixture Weights π_k
    # --------------------------------------------------
    def update_weights(self, gamma: np.ndarray) -> np.ndarray:
        """
        Args:
            gamma : (N, K)
        Returns:
            weights : (K,)
        """
        N = gamma.shape[0]
        weights = gamma.sum(axis=0) / N
        return weights

    # --------------------------------------------------
    # Compute Log-Likelihood (for convergence check)
    # --------------------------------------------------
    def compute_log_likelihood(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Uses EStep to compute log-likelihood
        """
        e_step = GMMExpectation(eps=self.eps)
        params = {
            "n_components": means.shape[0],
            "means": means,
            "covariances": covariances,
            "weights": weights
        }
        return e_step.compute_log_likelihood(X, params)

    # --------------------------------------------------
    # Full M-Step Wrapper
    # --------------------------------------------------
    def run_m_step(
        self,
        X: np.ndarray,
        gamma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Perform full M-Step.

        Returns:
            means       : (K, D)
            covariances : (K, D, D)
            weights     : (K,)
            log_likelihood : float
        """
        means = self.update_means(X, gamma)
        covariances = self.update_covariances(X, gamma, means)
        weights = self.update_weights(gamma)
        log_likelihood = self.compute_log_likelihood(X, means, covariances, weights)

        return means, covariances, weights, log_likelihood
