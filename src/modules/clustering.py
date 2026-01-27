# =========================
# File: modules/clustering.py
# =========================

import numpy as np

class GMMClustering:
    """
    Module 6: Clustering & Prediction for Gaussian Mixture Model
    """

    def __init__(self, means, covariances, weights):
        """
        Initialize with learned GMM parameters
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_components = means.shape[0]
        self.dim = means.shape[1]

    # -----------------------------
    # Assign Cluster
    # -----------------------------
    def assign_cluster(self, responsibilities):
        """
        Assign each data point to the Gaussian with highest responsibility
        """
        cluster_labels = np.argmax(responsibilities, axis=1)
        print(f"✅ Cluster assignment done (first 10 samples): {cluster_labels[:10]}")
        return cluster_labels

    # -----------------------------
    # Predict Probability
    # -----------------------------
    def predict_probability(self, responsibilities, n_samples=5):
        """
        Return probability of first n_samples belonging to each cluster
        """
        probs = responsibilities[:n_samples]
        print(f"✅ Probability for first {n_samples} sample(s):\n{probs}")
        return probs

    # -----------------------------
    # Generate Samples from GMM
    # -----------------------------
    def generate_samples(self, n_samples=5):
        """
        Generate synthetic data from the learned GMM
        """
        samples = []
        weights = self.weights / np.sum(self.weights)  # normalize weights
        print(f"Normalized weights used for sampling: {weights}")

        for _ in range(n_samples):
            # Choose component based on normalized weights
            k = np.random.choice(self.n_components, p=weights)
            sample = np.random.multivariate_normal(self.means[k], self.covariances[k])
            samples.append(sample)

        samples = np.array(samples)
        print(f"✅ Generated {n_samples} synthetic sample(s):\n{samples}")
        return samples
