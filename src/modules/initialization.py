import numpy as np
from sklearn.cluster import KMeans

class GMMInitializer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.K = None
        self.means = None
        self.covariances = None
        self.weights = None

    # Set Number of Components
    def set_num_components(self, K: int):
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer")
        self.K = K
        print(f"✔ Number of GMM components set to K = {self.K}")

    # Initialize Means
    def init_means(self, X: np.ndarray, method: str = "kmeans"):
        if self.K is None:
            raise ValueError("K not set. Call set_num_components(K) first.")

        if method == "kmeans":
            kmeans = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            self.means = kmeans.cluster_centers_
        elif method == "random":
            indices = np.random.choice(X.shape[0], self.K, replace=False)
            self.means = X[indices]
        else:
            raise ValueError("method must be 'kmeans' or 'random'")

        print("✔ Means initialized")
        return self.means

    # Initialize Covariances
    def init_covariances(self, X: np.ndarray, method: str = "identity"):
        if self.K is None:
            raise ValueError("K not set. Call set_num_components(K) first.")

        n_features = X.shape[1]
        self.covariances = np.zeros((self.K, n_features, n_features))

        if method == "identity":
            for k in range(self.K):
                self.covariances[k] = np.eye(n_features)
        elif method == "sample":
            kmeans = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            for k in range(self.K):
                cluster_data = X[labels == k]
                if len(cluster_data) > 1:
                    self.covariances[k] = np.cov(cluster_data.T) + np.eye(n_features) * 1e-6
                else:
                    self.covariances[k] = np.eye(n_features)
        else:
            raise ValueError("method must be 'identity' or 'sample'")

        print("✔ Covariances initialized")
        return self.covariances

    # Initialize Weights
    def init_weights(self, method: str = "uniform"):
        if self.K is None:
            raise ValueError("K not set. Call set_num_components(K) first.")
        if method != "uniform":
            raise ValueError("Only 'uniform' initialization supported")
        self.weights = np.ones(self.K) / self.K
        print("✔ Weights initialized")
        return self.weights

    # Initialize All
    def initialize_all(self, X, K, mean_method="kmeans", cov_method="identity", weight_method="uniform"):
        self.set_num_components(K)
        self.init_means(X, mean_method)
        self.init_covariances(X, cov_method)
        self.init_weights(weight_method)

        return {
            "n_components": self.K,
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights
        }
