# =========================
# File: maximization.py
# =========================

import numpy as np
from modules.expectation import GMMExpectation

class GMMMaximization:
    """
    Module 4: Maximization Step (M-Step) for Gaussian Mixture Model (GMM)
    Updates means, covariances, and mixture weights based on responsibilities
    """

    def __init__(self, eps=1e-9):
        self.eps = eps  # Numerical stability

    # -----------------------------
    # Update Means μ_k
    # -----------------------------
    def update_means(self, X, gamma):
        """
        X: (N, D)
        gamma: (N, K)
        Returns: means (K, D)
        """
        N_k = gamma.sum(axis=0)  # shape (K,)
        means = (gamma.T @ X) / (N_k[:, np.newaxis] + self.eps)
        return means

    # -----------------------------
    # Update Covariances Σ_k
    # -----------------------------
    def update_covariances(self, X, gamma, means):
        """
        X: (N, D)
        gamma: (N, K)
        means: (K, D)
        Returns: covariances (K, D, D)
        """
        N, D = X.shape
        K = gamma.shape[1]
        covariances = np.zeros((K, D, D))
        for k in range(K):
            diff = X - means[k]
            weighted_diff = diff.T * gamma[:, k]
            covariances[k] = (weighted_diff @ diff) / (gamma[:, k].sum() + self.eps)
            covariances[k] += np.eye(D) * self.eps  # numerical stability
        return covariances

    # -----------------------------
    # Update Weights π_k
    # -----------------------------
    def update_weights(self, gamma):
        """
        gamma: (N, K)
        Returns: weights (K,)
        """
        N = gamma.shape[0]
        weights = gamma.sum(axis=0) / N
        return weights

    # -----------------------------
    # Compute Log-Likelihood (using E-Step)
    # -----------------------------
    def compute_log_likelihood(self, X, means, covariances, weights):
        """
        Uses E-Step module to compute log-likelihood
        """
        e_step = GMMExpectation(eps=self.eps)
        log_likelihood = e_step.compute_log_likelihood(X, means, covariances, weights)
        return log_likelihood

    # -----------------------------
    # Full M-Step wrapper
    # -----------------------------
    def run_m_step(self, X, gamma):
        """
        Runs full M-Step: update parameters and return log-likelihood
        Returns: means, covariances, weights, log_likelihood
        """
        means = self.update_means(X, gamma)
        covariances = self.update_covariances(X, gamma, means)
        weights = self.update_weights(gamma)
        log_likelihood = self.compute_log_likelihood(X, means, covariances, weights)
        return means, covariances, weights, log_likelihood
