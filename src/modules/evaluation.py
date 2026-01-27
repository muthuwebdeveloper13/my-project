# ============================================
# Module 8: Evaluation Module for GMM
# File: modules/evaluation.py
# ============================================

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

class GMMEvaluation:
    """
    Module 8: Evaluation Module
    ---------------------------
    1. Silhouette Score
    2. BIC / AIC Scores
    3. Comparison with K-Means
    """

    def __init__(self, X):
        self.X = X

    # ---------------------------------
    # Silhouette Score
    # ---------------------------------
    def silhouette_score_gmm(self, cluster_labels):
        """
        Measures clustering quality (-1 to +1)
        """
        score = silhouette_score(self.X, cluster_labels)
        print("\n============================================================")
        print("🔍 SILHOUETTE SCORE (Clustering Quality)")
        print("============================================================")
        print(f"Silhouette Score: {score:.4f}")

        if score > 0.5:
            print("🟢 Interpretation: Excellent cluster separation")
        elif score > 0.25:
            print("🟡 Interpretation: Moderate cluster separation")
        else:
            print("🔴 Interpretation: Poor cluster separation")

        return score

    # ---------------------------------
    # BIC / AIC Scores
    # ---------------------------------
    def bic_aic_scores(self, k_range=range(1, 8)):
        """
        Helps choose optimal number of clusters (K)
        """
        bic_scores = []
        aic_scores = []

        print("\n============================================================")
        print("📊 BIC / AIC SCORES (Model Selection)")
        print("============================================================")

        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(self.X)

            bic = gmm.bic(self.X)
            aic = gmm.aic(self.X)

            bic_scores.append(bic)
            aic_scores.append(aic)

            print(f"K = {k} | BIC = {bic:.2f} | AIC = {aic:.2f}")

        best_k_bic = k_range[np.argmin(bic_scores)]
        best_k_aic = k_range[np.argmin(aic_scores)]

        print("\n✅ Optimal K based on BIC:", best_k_bic)
        print("✅ Optimal K based on AIC:", best_k_aic)

        return bic_scores, aic_scores

    # ---------------------------------
    # Compare with K-Means
    # ---------------------------------
    def compare_with_kmeans(self, gmm_labels, K):
        """
        Compare GMM clustering with K-Means clustering
        """
        kmeans = KMeans(n_clusters=K, random_state=42)
        kmeans_labels = kmeans.fit_predict(self.X)

        gmm_sil = silhouette_score(self.X, gmm_labels)
        kmeans_sil = silhouette_score(self.X, kmeans_labels)

        print("\n============================================================")
        print("⚖️ GMM vs K-MEANS COMPARISON")
        print("============================================================")
        print(f"GMM Silhouette Score    : {gmm_sil:.4f}")
        print(f"K-Means Silhouette Score: {kmeans_sil:.4f}")

        if gmm_sil > kmeans_sil:
            print("🏆 GMM performs better than K-Means")
        else:
            print("🏆 K-Means performs better than GMM")

        return gmm_sil, kmeans_sil
