# ========================
# File: expectation.py
# ========================

import numpy as np

class GMMExpectation:
    """
    Module 3: Expectation Step (E-Step) for Gaussian Mixture Model (GMM)
    Computes responsibilities (gamma) and log-likelihood
    """

    def __init__(self, eps=1e-9):
        self.eps = eps  # Numerical stability

    def compute_gaussian_pdf(self, X, mean, covariance):
        """
        Compute multivariate Gaussian PDF for each sample
        X: (N, D)
        mean: (D,)
        covariance: (D, D)
        Returns: (N,) probabilities
        """
        D = X.shape[1]
        cov = covariance + np.eye(D) * self.eps
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * det_cov + self.eps)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        return norm_const * np.exp(exponent)

    def estimate_responsibilities(self, X, means, covariances, weights):
        """
        Compute posterior probabilities (responsibilities) gamma(z_nk)
        X: (N, D)
        means: (K, D)
        covariances: (K, D, D)
        weights: (K,)
        Returns: gamma (N, K)
        """
        N = X.shape[0]
        K = means.shape[0]
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = weights[k] * self.compute_gaussian_pdf(X, means[k], covariances[k])
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = self.eps
        gamma /= row_sums
        return gamma

    def compute_log_likelihood(self, X, means, covariances, weights):
        """
        Compute total log-likelihood of the data under current GMM parameters
        """
        N = X.shape[0]
        K = means.shape[0]
        likelihood = np.zeros((N, K))
        for k in range(K):
            likelihood[:, k] = weights[k] * self.compute_gaussian_pdf(X, means[k], covariances[k])
        total_likelihood = likelihood.sum(axis=1)
        total_likelihood[total_likelihood == 0] = self.eps
        log_likelihood = np.sum(np.log(total_likelihood))
        return log_likelihood

    def run_e_step(self, X, means, covariances, weights):
        """
        Run the full E-Step: compute responsibilities and log-likelihood
        Returns: gamma, log_likelihood
        """
        gamma = self.estimate_responsibilities(X, means, covariances, weights)
        log_likelihood = self.compute_log_likelihood(X, means, covariances, weights)
        return gamma, log_likelihood
