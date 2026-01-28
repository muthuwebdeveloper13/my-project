# =========================
# File: modules/visualization.py
# =========================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class GMMVisualization:
    """
    Module 7: Visualization for Gaussian Mixture Model
    """

    def __init__(self, X, means, covariances, labels):
        self.X = X
        self.means = means
        self.covariances = covariances
        self.labels = labels
        self.K = means.shape[0]

    # ------------------------------------------------
    # Plot Clusters
    # ------------------------------------------------
    def plot_clusters(self):
        """
        Shows final cluster assignment
        """
        plt.figure(figsize=(8, 6))

        for k in range(self.K):
            cluster_points = self.X[self.labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f"Cluster {k}")

        plt.scatter(self.means[:, 0], self.means[:, 1],
                    c='black', s=120, marker='X', label="Cluster Centers")

        plt.title("GMM Cluster Assignment")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("✅ plot_clusters(): Cluster visualization displayed")

    # ------------------------------------------------
    # Plot Gaussian Ellipses
    # ------------------------------------------------
    def plot_gaussians(self):
        """
        Visualizes ellipses representing Gaussian components
        """
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plot data points
        plt.scatter(self.X[:, 0], self.X[:, 1], s=8, alpha=0.4)

        # Plot Gaussian ellipses
        for k in range(self.K):
            mean = self.means[k]
            cov = self.covariances[k]

            eigenvalues, eigenvectors = np.linalg.eigh(cov[:2, :2])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            width, height = 2 * np.sqrt(eigenvalues)

            ellipse = Ellipse(
                xy=mean[:2],
                width=width,
                height=height,
                angle=angle,
                edgecolor='red',
                facecolor='none',
                linewidth=2
            )

            ax.add_patch(ellipse)
            ax.scatter(mean[0], mean[1], c='red', s=80, marker='X')

        plt.title("Gaussian Components (Ellipses)")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

        print("✅ plot_gaussians(): Gaussian ellipses displayed")

    # ------------------------------------------------
    # Plot Distributions (1D / 2D density)
    # ------------------------------------------------
    def plot_distributions(self):
        """
        Plots 2D GMM density distribution
        """
        x = self.X[:, 0]
        y = self.X[:, 1]

        xmin, xmax = x.min()-1, x.max()+1
        ymin, ymax = y.min()-1, y.max()+1

        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                             np.linspace(ymin, ymax, 100))

        pos = np.dstack((xx, yy))
        Z = np.zeros(xx.shape)

        for k in range(self.K):
            mean = self.means[k][:2]
            cov = self.covariances[k][:2, :2]

            inv_cov = np.linalg.inv(cov)
            diff = pos - mean

            expo = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
            norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
            Z += norm_const * np.exp(-0.5 * expo)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, levels=30, cmap='viridis')
        plt.scatter(x, y, s=5, c='white')
        plt.title("GMM Probability Density Distribution")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Density")
        plt.show()

        print("✅ plot_distributions(): GMM density plot displayed")
