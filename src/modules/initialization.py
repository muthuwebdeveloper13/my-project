# src/modules/initialization.py

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from typing import Dict, Tuple


class GMMInitializer:
    """
    Module 2: GMM Parameter Initialization
    Initializes means, covariances, and weights for GMM.
    """

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.means = None
        self.covariances = None
        self.weights = None

    # --------------------------------------------------
    # Initialize Means
    # --------------------------------------------------
    def init_means(self, X: np.ndarray, method: str = "kmeans") -> np.ndarray:
        if method == "kmeans":
            print("✔ Initializing means using K-Means...")
            kmeans = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X)
            self.means = kmeans.cluster_centers_

        elif method == "random":
            print("✔ Initializing means randomly...")
            indices = np.random.choice(X.shape[0], self.n_components, replace=False)
            self.means = X[indices]

        else:
            raise ValueError("Invalid mean initialization method")

        print(f"   Means shape: {self.means.shape}")
        return self.means

    # --------------------------------------------------
    # Initialize Covariances
    # --------------------------------------------------
    def init_covariances(self, X: np.ndarray, method: str = "per_cluster") -> np.ndarray:
        print(f"✔ Initializing covariances using '{method}' method...")

        n_features = X.shape[1]
        self.covariances = np.zeros(
            (self.n_components, n_features, n_features)
        )

        if method == "identity":
            for k in range(self.n_components):
                self.covariances[k] = np.eye(n_features)

        elif method == "per_cluster":
            kmeans = KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(X)

            for k in range(self.n_components):
                cluster_data = X[labels == k]
                if cluster_data.shape[0] > 1:
                    self.covariances[k] = np.cov(cluster_data.T)
                else:
                    self.covariances[k] = np.eye(n_features)

                # Numerical stability
                self.covariances[k] += np.eye(n_features) * 1e-6

        else:
            raise ValueError("Invalid covariance initialization method")

        print("✔ Covariances initialized successfully!")
        return self.covariances

    # --------------------------------------------------
    # Initialize Weights
    # --------------------------------------------------
    def init_weights(self, method: str = "uniform") -> np.ndarray:
        print(f"✔ Initializing weights using '{method}' method...")

        if method == "uniform":
            self.weights = np.ones(self.n_components) / self.n_components
        else:
            raise ValueError("Invalid weight initialization method")

        print(f"   Weights: {self.weights}")
        return self.weights

    # --------------------------------------------------
    # Initialize All Parameters
    # --------------------------------------------------
    def initialize_all(
        self,
        X: np.ndarray,
        mean_method: str = "kmeans",
        cov_method: str = "per_cluster",
        weight_method: str = "uniform"
    ) -> Dict:

        self.init_means(X, mean_method)
        self.init_covariances(X, cov_method)
        self.init_weights(weight_method)

        return {
            "n_components": self.n_components,
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights,
            "initialization_method": mean_method
        }

    # --------------------------------------------------
    # Visualization of Initialization
    # --------------------------------------------------
    def visualize_initialization(
        self,
        X: np.ndarray,
        feature_indices=(0, 1),
        title: str = "GMM Initialization",
        save_path: str = None
    ) -> None:

        if X.shape[1] < 2:
            print("⚠️  Visualization skipped (less than 2 features).")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X[:, feature_indices[0]],
            X[:, feature_indices[1]],
            alpha=0.5,
            label="Data Points"
        )

        plt.scatter(
            self.means[:, feature_indices[0]],
            self.means[:, feature_indices[1]],
            color="red",
            marker="x",
            s=200,
            label="Initial Means"
        )

        plt.title(title)
        plt.xlabel(f"Feature {feature_indices[0]}")
        plt.ylabel(f"Feature {feature_indices[1]}")
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"✔ Initialization plot saved to {save_path}")

        plt.show()

    # --------------------------------------------------
    # Save Initialized Parameters
    # --------------------------------------------------
    def save_parameters(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, {
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights
        })
        print(f"✔ Initialized parameters saved to {path}")


# ======================================================
# Helper function called from main.py
# ======================================================
def initialize_gmm_from_data(
    X: np.ndarray,
    config: Dict
) -> Tuple[GMMInitializer, Dict]:

    initializer = GMMInitializer(
        n_components=config["clustering"]["n_components"],
        random_state=config["clustering"]["random_state"]
    )

    params = initializer.initialize_all(
        X=X,
        mean_method="kmeans",
        cov_method="per_cluster",
        weight_method="uniform"
    )

    return initializer, params
