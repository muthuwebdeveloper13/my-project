import os as os_module
import sys
import pandas as pd
import numpy as np

sys.path.append(os_module.path.dirname(os_module.path.abspath(__file__)))
sys.path.append(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))

# Create necessary directories
os_module.makedirs("outputs/final", exist_ok=True)
os_module.makedirs("data/processed", exist_ok=True)

# Import modules
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


def main():
    print("\n📋 SYSTEM INITIALIZATION")
    print("="*70)

    config = Config.load("config.yaml")
    config.validate()

    print("✅ Configuration loaded and validated successfully")
    print("System parameters initialized")
    print("Research pipeline started\n")

    # ==================================================
    # MODULE 1
    # ==================================================
    preprocessor = DataPreprocessor(config)
    data, numerical_cols, categorical_cols = preprocessor.get_preprocessed_data()

    os_module.makedirs("data/processed", exist_ok=True)
    data.to_csv("data/processed/preprocessed_data.csv", index=False)

    print("\n📁 Data Storage")
    print("------------------------------------------------------")
    print("✔ Preprocessed dataset saved")
    print("Path: data/processed/preprocessed_data.csv")


    print("=== MODULE 2: GMM INITIALIZATION ===")


    # Load preprocessed numerical data (Module 1 output assumed)
    data = pd.read_csv("data/processed/preprocessed_data.csv")


    # Initialize GMM parameters
    initializer = GMMInitializer(data)
    K = 3 # Number of clusters
    means, covariances, weights = initializer.initialize_all(K, mean_method="kmeans", cov_method="identity")


    # Summary output for easy understanding
    print("\n--- SUMMARY OF INITIALIZATION ---")
    print("Number of clusters (K):", K)
    print("Means (first 2 rows):\n", means[:2])
    print("Covariance matrices shape:", covariances.shape)
    print("Mixture weights:", weights)
    print("\n✅ Module 2 completed successfully")

    '''
    print("============================================================")
    print("MODULE 3: EXPECTATION STEP (E-STEP) - GAUSSIAN MIXTURE MODEL")
    print("============================================================\n")

    # -------------------------------
    # Step 1: Load preprocessed data
    # -------------------------------
    preprocessed_file = "data/processed/preprocessed_data.csv"

    try:
        data = pd.read_csv(preprocessed_file)
        print(f"✅ Preprocessed data loaded successfully from '{preprocessed_file}'")
        print(f"   → Shape: {data.shape}")
        print(f"   → Columns: {list(data.columns)}\n")
    except FileNotFoundError:
        print(f"❌ File not found: {preprocessed_file}")
        exit()

    # -------------------------------
    # Step 2: Initialize GMM Parameters (Module 2)
    # -------------------------------
    K = 3  # Number of clusters
    initializer = GMMInitializer(data)
    means, covariances, weights = initializer.initialize_all(
        K,
        mean_method="kmeans",
        cov_method="identity"
    )

    # -------------------------------
    # Step 3: Run E-Step
    # -------------------------------
    e_step = GMMExpectation()
    gamma, log_likelihood = e_step.run_e_step(data.values, means, covariances, weights)

    # -------------------------------
    # Step 4: Display Results Clearly
    # -------------------------------
    print("\n--- E-STEP RESULTS ---")
    print(f"Shape of responsibilities (gamma): {gamma.shape}")
    print("First 5 rows of responsibilities (gamma):\n", gamma[:5])

    print("\nCluster assignment probabilities for first 5 samples:")
    for i in range(5):
        probs = gamma[i]
        print(f"Sample {i+1}: ", end="")
        for k in range(K):
            print(f"Cluster {k+1}: {probs[k]:.3f}  ", end="")
        print()

    print(f"\nLog-likelihood of current parameters: {log_likelihood:.6f}")

    print("\n✅ Module 3 (E-Step) completed successfully")
    print("============================================================")

    print("============================================================")


    print("MODULE 4: MAXIMIZATION STEP (M-STEP) - GAUSSIAN MIXTURE MODEL")
    print("============================================================\n")

    # -------------------------------
    # Step 1: Load preprocessed data
    # -------------------------------
    preprocessed_file = "data/processed/preprocessed_data.csv"

    try:
        data = pd.read_csv(preprocessed_file)
        print(f"✅ Preprocessed data loaded from '{preprocessed_file}'")
        print(f"   → Shape: {data.shape}")
    except FileNotFoundError:
        print(f"❌ File not found: {preprocessed_file}")
        exit()

    X = data.values

    # -------------------------------
    # Step 2: Initialize GMM Parameters (Module 2)
    # -------------------------------
    K = 3
    initializer = GMMInitializer(data)
    means, covariances, weights = initializer.initialize_all(K, mean_method="kmeans", cov_method="identity")

    # -------------------------------
    # Step 3: Compute Responsibilities (Module 3)
    # -------------------------------
    e_step = GMMExpectation()
    gamma, _ = e_step.run_e_step(X, means, covariances, weights)

    # -------------------------------
    # Step 4: Run M-Step
    # -------------------------------
    m_step = GMMMaximization()
    new_means, new_covariances, new_weights, log_likelihood = m_step.run_m_step(X, gamma)

    # -------------------------------
    # Step 5: Display Results Clearly
    # -------------------------------
    print("\n--- M-STEP RESULTS ---")
    print(f"Updated Means (first 5 samples of each cluster):\n{new_means}\n")

    print(f"Updated Weights (π_k):\n{new_weights}\n")

    print("Updated Covariances (first cluster):\n", new_covariances[0])

    print(f"\nLog-likelihood after M-Step: {log_likelihood:.6f}")

    print("\n✅ Module 4 (M-Step) completed successfully")
    print("============================================================")


    print("============================================================")
    print("MODULE 5: CONVERGENCE CHECKING - EM ALGORITHM")
    print("============================================================\n")

    # -------------------------------
    # Sample log-likelihood values (from EM iterations)
    # In real EM: these come from E-Step + M-Step loop
    # -------------------------------
    sample_log_likelihoods = [-1200.0, -1100.5, -1050.2, -1020.1, -1010.0, -1008.5, -1008.0]

    tol = 1e-2
    max_iter = 10

    checker = ConvergenceChecker(tol=tol, max_iter=max_iter)
    prev_ll = None

    # -------------------------------
    # Run convergence check iteration
    # -------------------------------
    for iteration, ll in enumerate(sample_log_likelihoods, 1):
        print(f"Iteration {iteration}: Log-Likelihood = {ll:.6f}")
        converged = checker.check_convergence(ll, prev_ll, iteration)
        prev_ll = ll
        if converged:
            break

    # -------------------------------
    # Plot convergence
    # -------------------------------
    checker.plot_log_likelihood()

    print("\n✅ Module 5 (Convergence Checking) completed successfully")
    print("============================================================")   '''


    print("============================================================")
    print("FULL EM ALGORITHM: GAUSSIAN MIXTURE MODEL (Modules 3+4+5)")
    print("============================================================\n")

    # -------------------------------
    # Step 1: Load preprocessed data
    # -------------------------------
    preprocessed_file = "data/processed/preprocessed_data.csv"

    try:
        data = pd.read_csv(preprocessed_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing file: {preprocessed_file}")

    X = data.values  # shape (N, D)

    # -------------------------------
    # Step 2: Initialize GMM Parameters
    # -------------------------------
    K = 3  # number of clusters

    initializer = GMMInitializer(data)
    means, covariances, weights = initializer.initialize_all(
        K,
        mean_method="kmeans",
        cov_method="identity"
    )

    # -------------------------------
    # Step 3: Setup EM modules
    # -------------------------------
    e_step = GMMExpectation()
    m_step = GMMMaximization()
    convergence_checker = ConvergenceChecker(tol=1e-3, max_iter=20)

    prev_log_likelihood = None

    # IMPORTANT: define variables before loop
    responsibilities = None
    log_likelihood = None

    # -------------------------------
    # Step 4: Before EM Loop
    # ------------------------------
    # Before the EM loop, initialize the list
    log_likelihoods = []

    prev_log_likelihood = None

    for iteration in range(1, convergence_checker.max_iter + 1):

        # -------- E-STEP --------
        responsibilities, log_likelihood = e_step.run_e_step(X, means, covariances, weights)

        # Save log-likelihood for later plotting/report
        log_likelihoods.append(log_likelihood)

        print(f"Iteration {iteration}: Log-Likelihood (after E-Step) = {log_likelihood:.6f}")

        # -------- M-STEP --------
        means, covariances, weights, log_likelihood = m_step.run_m_step(X, responsibilities)
        print(f"Iteration {iteration}: Log-Likelihood (after M-Step) = {log_likelihood:.6f}")

        # Convergence check
        if convergence_checker.check_convergence(log_likelihood, prev_log_likelihood, iteration):
            break

        prev_log_likelihood = log_likelihood

    # -------------------------------
    # Step 5: EM Algorithm Loop
    # -------------------------------
    for iteration in range(1, convergence_checker.max_iter + 1):

        # -------- E-STEP --------
        responsibilities, log_likelihood = e_step.run_e_step(
            X, means, covariances, weights
        )

        # -------- M-STEP --------
        means, covariances, weights, log_likelihood = m_step.run_m_step(
            X, responsibilities
        )

        # -------- Convergence Check --------
        if convergence_checker.check_convergence(
            log_likelihood, prev_log_likelihood, iteration
        ):
            break

        prev_log_likelihood = log_likelihood

    # -------------------------------
    # Step 6: Save final EM outputs
    # -------------------------------
    import os
    os.makedirs("outputs/final", exist_ok=True)

    np.save("outputs/final/responsibilities.npy", responsibilities)
    np.save("outputs/final/cluster_labels.npy", np.argmax(responsibilities, axis=1))
    np.save("outputs/final/final_means.npy", means)
    np.save("outputs/final/final_covariances.npy", covariances)
    np.save("outputs/final/final_weights.npy", weights)

    # -------------------------------
    # Step 7: Plot convergence
    # -------------------------------
    convergence_checker.plot_log_likelihood()

    print("\nEM completed and outputs saved.")
    print("============================================================")

    print("============================================================")
    print("MODULE 6: CLUSTERING & PREDICTION")
    print("============================================================\n")

    # -------------------------------
    # Load preprocessed data
    # -------------------------------
    data_file = "data/processed/preprocessed_data.csv"
    data = pd.read_csv(data_file)
    X = data.values
    print(f"✅ Preprocessed data loaded: {X.shape} samples, {X.shape[1]} features\n")

    # -------------------------------
    # Quick EM loop to get parameters and responsibilities
    # -------------------------------
    K = 3
    max_iter = 5  # short demo
    e_step = GMMExpectation()
    m_step = GMMMaximization()

    # Initialize GMM parameters
    means = X[:K, :]
    covariances = np.array([np.cov(X.T) + np.eye(X.shape[1])*1e-6 for _ in range(K)])
    weights = np.ones(K)/K

    responsibilities = np.ones((X.shape[0], K))/K  # uniform initialization

    for iteration in range(max_iter):
        # M-Step
        means, covariances, weights, log_likelihood = m_step.run_m_step(X, responsibilities)
        # E-Step
        responsibilities, log_likelihood = e_step.run_e_step(X, means, covariances, weights)
        print(f"Iteration {iteration+1} - Log-Likelihood: {log_likelihood:.4f}")

    print("\n✅ EM Algorithm completed successfully!\n")

    # -------------------------------
    # Module 6: Clustering & Prediction
    # -------------------------------
    clustering = GMMClustering(means, covariances, weights)

    # Assign clusters
    cluster_labels = clustering.assign_cluster(responsibilities)

    # Predict probability for first 5 samples
    clustering.predict_probability(responsibilities, n_samples=5)

    # Generate synthetic samples
    clustering.generate_samples(n_samples=5)

    print("\n✅ Module 6 completed successfully")
    print("============================================================")

    print("============================================================")
    print("MODULE 7: VISUALIZATION MODULE")
    print("============================================================\n")

    # -------------------------------
    # Load data
    # -------------------------------
    data_file = "data/processed/preprocessed_data.csv"
    data = pd.read_csv(data_file)
    X = data.values
    print(f"✅ Data loaded: {X.shape}\n")

    # -------------------------------
    # Train small EM model (demo)
    # -------------------------------
    K = 3
    e_step = GMMExpectation()
    m_step = GMMMaximization()

    means = X[:K, :]
    covariances = np.array([np.cov(X.T) + np.eye(X.shape[1])*1e-6 for _ in range(K)])
    weights = np.ones(K)/K
    responsibilities = np.ones((X.shape[0], K))/K

    for i in range(5):
        means, covariances, weights, ll = m_step.run_m_step(X, responsibilities)
        responsibilities, ll = e_step.run_e_step(X, means, covariances, weights)

    print("✅ EM training completed\n")

    # -------------------------------
    # Clustering
    # -------------------------------
    clustering = GMMClustering(means, covariances, weights)
    labels = clustering.assign_cluster(responsibilities)

    # -------------------------------
    # Visualization
    # -------------------------------
    viz = GMMVisualization(X, means, covariances, labels)

    viz.plot_clusters()        # Cluster plot
    viz.plot_gaussians()       # Gaussian ellipses
    viz.plot_distributions()   # Density plot

    print("\n✅ Module 7 Visualization completed successfully")
    print("============================================================")


    # Load preprocessed dataset
    data = pd.read_csv("data/processed/preprocessed_data.csv")
    X = data.values

    base = "outputs/final"

    # Required EM output files
    files = {
        "responsibilities": "responsibilities.npy",
        "labels": "cluster_labels.npy",
        "means": "final_means.npy",
        "covs": "final_covariances.npy",
        "weights": "final_weights.npy"
    }

    for f in files.values():
        path = os.path.join(base, f)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    # Load EM outputs
    responsibilities = np.load(os.path.join(base, files["responsibilities"]))
    labels = np.load(os.path.join(base, files["labels"]))
    means = np.load(os.path.join(base, files["means"]))
    covariances = np.load(os.path.join(base, files["covs"]))
    weights = np.load(os.path.join(base, files["weights"]))

    # Evaluation
    evaluator = GMMEvaluation(X, labels, responsibilities, means, covariances, weights)

    sil = evaluator.silhouette()
    bic, aic = evaluator.bic_aic()
    gmm_sil, km_sil = evaluator.compare_with_kmeans()

    print("Silhouette Score :", sil)
    print("BIC :", bic)
    print("AIC :", aic)
    print("GMM Silhouette :", gmm_sil)
    print("KMeans Silhouette :", km_sil)


    # After EM, clustering, and evaluation steps
    documentation = GMMDocumentation(output_dir="outputs/final")

    # Example inputs from previous modules
    # 'means', 'covariances', 'weights' are from EM
    # 'cluster_labels' is np.argmax(responsibilities, axis=1)
    # 'log_likelihoods' is list of log-likelihood values over EM iterations

    # Cluster labels

    documentation = GMMDocumentation(output_dir="outputs/final")

    summary_file = documentation.generate_summary(
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


    '''print("\n🚀 PIPELINE STATUS")
    print("------------------------------------------------------")
    print("MODULE 1: COMPLETED")
    print("NEXT MODULES:")
    print("→ MODULE 2: GMM Parameter Initialization")
    print("→ MODULE 3: Expectation Step (E-Step)")
    print("→ MODULE 4: Maximization Step (M-Step)")
    print("→ MODULE 5: Convergence Checking")
    print("→ MODULE 6: Clustering & Prediction")
    print("→ MODULE 7: Visualization")
    print("→ MODULE 8: Evaluation")
    print("\nSystem ready for next module development")'''


if __name__ == "__main__":
    main()
