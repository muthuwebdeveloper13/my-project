import numpy as np
from typing import Tuple, Dict

class GMMExpectation:
    """
    Module 3: Expectation Step (E-Step) for Gaussian Mixture Model (GMM)
    ---------------------------------------------------------------------
    Responsibilities and log-likelihood computation.
    """

    def __init__(self, eps: float = 1e-9):
        """
        Args:
            eps : float, small number for numerical stability
        """
        self.eps = eps

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
            X          : (N, D) data matrix
            mean       : (D,) mean vector
            covariance : (D, D) covariance matrix

        Returns:
            pdf : (N,) probability density for each sample
        """
        N, D = X.shape

        # Add eps for numerical stability
        cov = covariance + np.eye(D) * self.eps
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** D) * det_cov + self.eps)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)

        return norm_const * np.exp(exponent)

    # --------------------------------------------------
    # Estimate Responsibilities γ(z_nk)
    # --------------------------------------------------
    def estimate_responsibilities(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute responsibilities γ(z_nk) for all samples and components.

        γ(z_nk) = π_k * N(x_n | μ_k, Σ_k) / Σ_j π_j * N(x_n | μ_j, Σ_j)

        Args:
            X      : (N, D) data matrix
            params : dictionary containing GMM parameters
                     - 'n_components' : int K
                     - 'means'        : (K, D)
                     - 'covariances'  : (K, D, D)
                     - 'weights'      : (K,)

        Returns:
            gamma : (N, K) responsibilities matrix
        """
        N, D = X.shape
        K = params['n_components']
        means = params['means']
        covs = params['covariances']
        weights = params['weights']

        gamma = np.zeros((N, K))

        for k in range(K):
            pdf = self.compute_gaussian_pdf(X, means[k], covs[k])
            gamma[:, k] = weights[k] * pdf

        # Normalize across components for each sample
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = self.eps
        gamma /= row_sums

        return gamma

    # --------------------------------------------------
    # Compute Log-Likelihood
    # --------------------------------------------------
    def compute_log_likelihood(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute total log-likelihood of the data under current GMM parameters.

        Args:
            X      : (N, D) data matrix
            params : dictionary containing GMM parameters
                     - 'n_components' : int K
                     - 'means'        : (K, D)
                     - 'covariances'  : (K, D, D)
                     - 'weights'      : (K,)

        Returns:
            log_likelihood : float
        """
        N, D = X.shape
        K = params['n_components']
        means = params['means']
        covs = params['covariances']
        weights = params['weights']

        likelihood = np.zeros((N, K))
        for k in range(K):
            pdf = self.compute_gaussian_pdf(X, means[k], covs[k])
            likelihood[:, k] = weights[k] * pdf

        total_likelihood = likelihood.sum(axis=1)
        total_likelihood[total_likelihood == 0] = self.eps

        log_likelihood = np.sum(np.log(total_likelihood))
        return log_likelihood

    # --------------------------------------------------
    # Run Full E-Step
    # --------------------------------------------------
    def run_e_step(
        self,
        X: np.ndarray,
        params: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Perform full E-Step: compute responsibilities and log-likelihood.

        Args:
            X      : (N, D) data matrix
            params : dictionary containing GMM parameters

        Returns:
            responsibilities : (N, K)
            log_likelihood   : float
        """
        responsibilities = self.estimate_responsibilities(X, params)
        log_likelihood = self.compute_log_likelihood(X, params)
        return responsibilities, log_likelihood
