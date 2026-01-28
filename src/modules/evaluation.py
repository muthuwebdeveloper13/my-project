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
            return None
        return silhouette_score(self.X, self.labels)

    # -----------------------------
    # BIC and AIC
    # -----------------------------
    def bic_aic(self):
        N, D = self.X.shape
        K = self.K

        log_likelihood = 0.0

        for n in range(N):
            p_x = 0.0
            for k in range(K):
                diff = self.X[n] - self.means[k]
                inv_cov = np.linalg.inv(self.covariances[k])
                det_cov = np.linalg.det(self.covariances[k])

                norm = 1.0 / np.sqrt((2*np.pi)**D * det_cov)
                exp_val = np.exp(-0.5 * diff.T @ inv_cov @ diff)

                p_x += self.weights[k] * norm * exp_val

            log_likelihood += np.log(p_x + 1e-9)

        # number of parameters
        p = K * (D + D*D + 1)

        bic = -2 * log_likelihood + p * np.log(N)
        aic = -2 * log_likelihood + 2 * p

        return bic, aic

    # -----------------------------
    # Compare with KMeans
    # -----------------------------
    def compare_with_kmeans(self):
        kmeans = KMeans(n_clusters=self.K, random_state=0)
        km_labels = kmeans.fit_predict(self.X)

        gmm_sil = self.silhouette()
        km_sil = silhouette_score(self.X, km_labels)

        return gmm_sil, km_sil
