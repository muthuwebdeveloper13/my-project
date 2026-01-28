import os as os_module
import sys
import pandas as pd
import numpy as np


# =========================================================
# PATH CONFIGURATION
# =========================================================
# This allows Python to find project modules correctly
sys.path.append(os_module.path.dirname(os_module.path.abspath(__file__)))
sys.path.append(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))


# =========================================================
# DIRECTORY STRUCTURE
# =========================================================
# Ensure required folders exist
os_module.makedirs("outputs/final", exist_ok=True)
os_module.makedirs("data/processed", exist_ok=True)


# =========================================================
# MODULE IMPORTS
# =========================================================
from config import Config
from modules.data_processing import DataPreprocessor
from modules.initialization import GMMInitializer
from modules.expectation import GMMExpectation
from modules.maximization import GMMMaximization
from modules.convergence import ConvergenceChecker
from modules.clustering import GMMClustering
from modules.visualization import GMMVisualization
from modules.evaluation import GMMEvaluation
from modules.documentation import GMMDocumentation




# =========================================================
# MAIN PIPELINE FUNCTION
# =========================================================
def main():


    # =====================================================
    # STEP 0: LOAD CONFIGURATION
    # =====================================================
    # Reads config.yaml and validates parameters
    config = Config.load("config.yaml")
    config.validate()


    # =====================================================
    # MODULE 1: DATA COLLECTION & PREPROCESSING
    # =====================================================
    # Loads dataset, cleans data, handles missing values,
    # selects features, normalizes, and prepares final dataset
    preprocessor = DataPreprocessor(config)
    data, num_cols, cat_cols = preprocessor.get_preprocessed_data()


    # Save processed dataset
    data.to_csv("data/processed/preprocessed_data.csv", index=False)


    # Convert to numerical matrix for ML
    X = data.values   # Shape: (N samples, D features)


    # =====================================================
    # MODULE 2: GMM PARAMETER INITIALIZATION
    # =====================================================
    # Initialize means, covariances, and weights
    K = 3  # Number of Gaussian components (clusters)


    initializer = GMMInitializer(data)
    means, covariances, weights = initializer.initialize_all(
        K,
        mean_method="kmeans",     # Initialize means using k-means
        cov_method="identity"     # Initialize covariances as identity
    )


    # =====================================================
    # MODULE 3+4+5: FULL EM ALGORITHM
    # (Expectation + Maximization + Convergence)
    # =====================================================
    e_step = GMMExpectation()
    m_step = GMMMaximization()
    convergence = ConvergenceChecker(tol=1e-3, max_iter=50)


    log_likelihoods = []           # Stores log-likelihood history
    prev_log_likelihood = None     # For convergence check


    # ---------------- EM LOOP ----------------
    for iteration in range(convergence.max_iter):


        # -------- E-STEP --------
        # Computes responsibilities (posterior probabilities γ)
        responsibilities, log_likelihood = e_step.run_e_step(
            X, means, covariances, weights
        )


        log_likelihoods.append(log_likelihood)


        # -------- M-STEP --------
        # Updates means, covariances, and weights using γ
        means, covariances, weights, log_likelihood = m_step.run_m_step(
            X, responsibilities
        )


        # -------- CONVERGENCE CHECK --------
        if convergence.check_convergence(
            log_likelihood, prev_log_likelihood, iteration + 1
        ):
            break


        prev_log_likelihood = log_likelihood


    # Save EM outputs
    np.save("outputs/final/responsibilities.npy", responsibilities)
    np.save("outputs/final/cluster_labels.npy", np.argmax(responsibilities, axis=1))
    np.save("outputs/final/final_means.npy", means)
    np.save("outputs/final/final_covariances.npy", covariances)
    np.save("outputs/final/final_weights.npy", weights)


    # =====================================================
    # MODULE 6: CLUSTERING & PREDICTION
    # =====================================================
    clustering = GMMClustering(means, covariances, weights)
    cluster_labels = clustering.assign_cluster(responsibilities)


    # =====================================================
    # MODULE 7: VISUALIZATION
    # =====================================================
    viz = GMMVisualization(X, means, covariances, cluster_labels)
    viz.plot_clusters()        # Data points + clusters
    viz.plot_gaussians()       # Gaussian ellipses
    viz.plot_distributions()   # Density plots


    # =====================================================
    # MODULE 8: EVALUATION
    # =====================================================
    evaluator = GMMEvaluation(
        X,
        cluster_labels,
        responsibilities,
        means,
        covariances,
        weights
    )


    silhouette = evaluator.silhouette()
    bic, aic = evaluator.bic_aic()
    gmm_sil, kmeans_sil = evaluator.compare_with_kmeans()


    # =====================================================
    # MODULE 9: DOCUMENTATION & REPORT
    # =====================================================
    documentation = GMMDocumentation(output_dir="outputs/final")


    documentation.generate_summary(
        means=means,
        covariances=covariances,
        weights=weights,
        log_likelihoods=log_likelihoods,
        cluster_labels=cluster_labels
    )


    documentation.export_model(
        means=means,
        covariances=covariances,
        weights=weights
    )

# =========================================================
# PROGRAM ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()