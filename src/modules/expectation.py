# src/modules/expectation.py

import numpy as np
from typing import Tuple


class GMMExpectation:
    """
    Module 3: Expectation Step (E-Step)
    Computes responsibilities and log-likelihood.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps  # For numerical stability

    # --------------------------------------------------
    # Multivariate Gaussian PDF
    # --------------------------------------------------
    def compute_gaussian_pdf(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> np.ndarray:
        """
        Compute multivariate Gaussian probability density function.

        Args:
            X : Data matrix (N, D)
            mean : Mean vector (D,)
            covariance : Covariance matrix (D, D)

        Returns:
            pdf values (N,)
        """
        n_features = X.shape[1]

        cov = covariance + np.eye(n_features) * self.eps
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1.0 / np.sqrt(
            ((2 * np.pi) ** n_features) * det_cov
        )

        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

        return norm_const * np.exp(exponent)

    # --------------------------------------------------
    # E-Step: Responsibility Computation
    # --------------------------------------------------
    def estimate_responsibilities(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute responsibilities γ(z_nk).

        Returns:
            Responsibility matrix (N, K)
        """
        N = X.shape[0]
        K = means.shape[0]

        responsibilities = np.zeros((N, K))

        for k in range(K):
            pdf = self.compute_gaussian_pdf(
                X, means[k], covariances[k]
            )
            responsibilities[:, k] = weights[k] * pdf

        # Normalize responsibilities
        sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
        sum_responsibilities[sum_responsibilities == 0] = self.eps

        responsibilities /= sum_responsibilities

        return responsibilities

    # --------------------------------------------------
    # Log-Likelihood Computation
    # --------------------------------------------------
    def compute_log_likelihood(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Compute total log-likelihood of the data.
        """
        N = X.shape[0]
        K = means.shape[0]

        likelihood = np.zeros((N, K))

        for k in range(K):
            pdf = self.compute_gaussian_pdf(
                X, means[k], covariances[k]
            )
            likelihood[:, k] = weights[k] * pdf

        total_likelihood = np.sum(likelihood, axis=1)
        total_likelihood[total_likelihood == 0] = self.eps

        log_likelihood = np.sum(np.log(total_likelihood))

        return log_likelihood

    # --------------------------------------------------
    # Full E-Step Wrapper
    # --------------------------------------------------
    def run_e_step(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Perform full E-Step.

        Returns:
            responsibilities (N, K)
            log_likelihood (float)
        """
        responsibilities = self.estimate_responsibilities(
            X, means, covariances, weights
        )

        log_likelihood = self.compute_log_likelihood(
            X, means, covariances, weights
        )

        return responsibilities, log_likelihood
