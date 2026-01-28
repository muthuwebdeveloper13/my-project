# ========================
# File: initialization.py
# ========================

import numpy as np
from sklearn.cluster import KMeans

class GMMInitializer:
    """
    Module 2: Initialization of GMM Parameters
    """

    def __init__(self, data):
        self.data = data.values if hasattr(data, 'values') else data
        self.N, self.D = self.data.shape
        self.K = None
        self.means = None
        self.covariances = None
        self.weights = None

    def set_num_components(self, K):
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer")
        self.K = K
        print(f"✅ Number of clusters (K) set to: {self.K}")
        return self.K

    def init_means(self, method="kmeans"):
        if self.K is None:
            raise ValueError("Set number of clusters first using set_num_components(K)")

        print("Initializing means...")
        if method == "random":
            indices = np.random.choice(self.N, self.K, replace=False)
            self.means = self.data[indices]
            print("✔ Means initialized randomly")
        elif method == "kmeans":
            kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=42)
            kmeans.fit(self.data)
            self.means = kmeans.cluster_centers_
            print("✔ Means initialized using K-Means")
        else:
            raise ValueError("Method must be 'kmeans' or 'random'")

        print("   Means shape:", self.means.shape)
        print("   First 2 mean vectors:\n", self.means[:2])
        return self.means

    def init_covariances(self, method="identity"):
        print("Initializing covariance matrices...")
        self.covariances = []
        if method == "identity":
            for _ in range(self.K):
                self.covariances.append(np.eye(self.D))
            print("✔ Covariances set as identity matrices")
        elif method == "sample":
            sample_cov = np.cov(self.data.T)
            for _ in range(self.K):
                self.covariances.append(sample_cov)
            print("✔ Covariances set as sample covariance")
        else:
            raise ValueError("Covariance method must be 'identity' or 'sample'")

        self.covariances = np.array(self.covariances)
        print("   Covariance shape:", self.covariances.shape)
        return self.covariances

    def init_weights(self):
        print("Initializing mixture weights...")
        self.weights = np.ones(self.K) / self.K
        print("✔ Weights initialized uniformly")
        print("   Weights:", self.weights)
        return self.weights

    def initialize_all(self, K, mean_method="kmeans", cov_method="identity"):
        self.set_num_components(K)
        self.init_means(mean_method)
        self.init_covariances(cov_method)
        self.init_weights()
        print("\n🎯 GMM parameters ready for EM algorithm")
        return self.means, self.covariances, self.weights
