import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


class GMMEvaluation:

    def __init__(self, X, labels, responsibilities, means, covariances, weights):
        self.X = X
        self.labels = labels
        self.responsibilities = responsibilities
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.K = len(weights)

    # -----------------------------
    # Silhouette Score
    # -----------------------------
    def silhouette(self):
        if len(np.unique(self.labels)) < 2:
            return 0.0
        try:
            return silhouette_score(self.X, self.labels)
        except:
            return 0.0

    # -----------------------------
    # BIC and AIC (Optimized Version)
    # -----------------------------
    def bic_aic(self):
        """
        Optimized BIC and AIC calculation using vectorized operations
        """
        N, D = self.X.shape
        K = self.K
        
        # Vectorized log-likelihood calculation
        log_likelihood = self._compute_log_likelihood_vectorized()
        
        # If vectorized calculation fails, use approximation
        if np.isinf(log_likelihood) or np.isnan(log_likelihood):
            log_likelihood = self._compute_log_likelihood_approx()
        
        # Number of parameters
        # For each component: D means, D*(D+1)/2 covariance parameters (symmetry), 1 weight
        # Total: K*(D + D*(D+1)/2 + 1) - 1 (weights sum to 1, so one less degree of freedom)
        cov_params_per_component = D * (D + 1) // 2
        p = K * (D + cov_params_per_component + 1) - 1
        
        bic = -2 * log_likelihood + p * np.log(N)
        aic = -2 * log_likelihood + 2 * p
        
        return bic, aic
    
    def _compute_log_likelihood_vectorized(self):
        """Vectorized log-likelihood calculation"""
        try:
            N, D = self.X.shape
            K = self.K
            
            # Pre-calculate constants
            log_weights = np.log(self.weights + 1e-300)
            log_det_cov = np.zeros(K)
            inv_cov = np.zeros_like(self.covariances)
            
            for k in range(K):
                # Add small value to diagonal for numerical stability
                cov = self.covariances[k] + np.eye(D) * 1e-6
                sign, logdet = np.linalg.slogdet(cov)
                log_det_cov[k] = logdet if sign > 0 else -np.inf
                inv_cov[k] = np.linalg.inv(cov)
            
            # Calculate log probabilities for all points and all components
            log_probs = np.zeros((N, K))
            
            for k in range(K):
                diff = self.X - self.means[k]
                # Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ)
                mahalanobis = np.sum(diff @ inv_cov[k] * diff, axis=1)
                
                # Log of Gaussian PDF
                log_probs[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det_cov[k] + mahalanobis)
            
            # Add log weights
            log_probs += log_weights
            
            # Log-sum-exp trick for numerical stability
            max_log = np.max(log_probs, axis=1, keepdims=True)
            exp_log_probs = np.exp(log_probs - max_log)
            sum_exp = np.sum(exp_log_probs, axis=1)
            log_likelihood = np.sum(np.log(sum_exp + 1e-300) + max_log.flatten())
            
            return log_likelihood
            
        except Exception as e:
            print(f"⚠ Vectorized log-likelihood failed: {e}")
            return -np.inf
    
    def _compute_log_likelihood_approx(self):
        """Approximate log-likelihood using responsibilities"""
        try:
            # Use responsibilities to approximate log-likelihood
            # L = Σ_n log Σ_k π_k N(x_n|μ_k,Σ_k) ≈ Σ_n log max_responsibility
            max_responsibilities = np.max(self.responsibilities, axis=1)
            log_likelihood = np.sum(np.log(max_responsibilities + 1e-300))
            return log_likelihood
        except:
            return -np.inf
    
    def _compute_log_likelihood_fast(self):
        """Fast but approximate log-likelihood"""
        N = len(self.X)
        # Very rough approximation based on average responsibilities
        avg_resp = np.mean(self.responsibilities)
        return N * np.log(avg_resp + 1e-300)

    # -----------------------------
    # Compare with KMeans (Optimized)
    # -----------------------------
    def compare_with_kmeans(self):
        """Compare GMM with KMeans clustering"""
        try:
            # Limit data size for KMeans if dataset is large
            if len(self.X) > 10000:
                # Sample data for KMeans comparison
                indices = np.random.choice(len(self.X), 10000, replace=False)
                X_sample = self.X[indices]
                labels_sample = self.labels[indices]
            else:
                X_sample = self.X
                labels_sample = self.labels
            
            # Run KMeans with n_init=1 for speed
            kmeans = KMeans(n_clusters=self.K, n_init=1, random_state=42, max_iter=100)
            km_labels = kmeans.fit_predict(X_sample)
            
            # Calculate silhouette scores
            try:
                gmm_sil = silhouette_score(X_sample, labels_sample)
            except:
                gmm_sil = 0.0
                
            try:
                km_sil = silhouette_score(X_sample, km_labels)
            except:
                km_sil = 0.0
            
            return gmm_sil, km_sil
            
        except Exception as e:
            print(f"⚠ KMeans comparison failed: {e}")
            return 0.0, 0.0
    
    def get_cluster_statistics(self):
        """Get basic statistics about each cluster"""
        stats = []
        for k in range(self.K):
            cluster_points = self.X[self.labels == k]
            if len(cluster_points) > 0:
                stats.append({
                    'cluster': k,
                    'size': len(cluster_points),
                    'percentage': len(cluster_points) / len(self.X) * 100,
                    'mean': np.mean(cluster_points, axis=0),
                    'std': np.std(cluster_points, axis=0),
                    'weight': self.weights[k]
                })
        return stats