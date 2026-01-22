import sys
import os
import numpy as np

# -----------------------------
# Ensure project root is in path
# -----------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------
# Module imports
# -----------------------------
from modules.data_processing import DataPreprocessor
from modules.initialization import GMMInitializer
from modules.expectation import GMMExpectation
from modules.maximization import GMMMaximization
from config import Config


def main():
    print("📋 Loading configuration...")
    config = Config.load("config.yaml")
    config.validate()
    print("✅ Configuration loaded successfully")

    # ==================================================
    # MODULE 1: DATA PREPROCESSING
    # ==================================================
    print("\n" + "=" * 60)
    print("MODULE 1: DATA PREPROCESSING")
    print("=" * 60)

    preprocessor = DataPreprocessor(config)
    data, numerical_cols, categorical_cols = preprocessor.get_preprocessed_data()

    os.makedirs("data/processed", exist_ok=True)
    preprocessed_path = "data/processed/preprocessed_data.csv"
    data.to_csv(preprocessed_path, index=False)
    print(f"✅ Preprocessed data saved: {preprocessed_path}")

    # ==================================================
    # MODULE 2: GMM INITIALIZATION
    # ==================================================
    print("\n" + "=" * 60)
    print("MODULE 2: GMM PARAMETER INITIALIZATION")
    print("=" * 60)

    X = data[numerical_cols].values
    initializer = GMMInitializer(random_state=42)
    params = initializer.initialize_all(
        X=X,
        K=config.clustering.n_components,
        mean_method="kmeans",
        cov_method="sample",
        weight_method="uniform"
    )

    os.makedirs("outputs/models", exist_ok=True)
    gmm_params_path = "outputs/models/gmm_initialized_params.npy"
    np.save(gmm_params_path, params)
    print(f"✅ GMM parameters saved: {gmm_params_path}")

    # ==================================================
    # MODULES 3 + 4: EM ALGORITHM LOOP
    # ==================================================
    print("\n" + "=" * 60)
    print("RUNNING EM ALGORITHM (E-STEP + M-STEP)")
    print("=" * 60)

    e_step = GMMExpectation()
    m_step = GMMMaximization()

    max_iter = config.clustering.max_iterations
    tol = config.clustering.tolerance
    prev_log_likelihood = -np.inf

    for iteration in range(1, max_iter + 1):
        # -------- E-STEP --------
        gamma, log_likelihood = e_step.run_e_step(
            X,
            params["means"],
            params["covariances"],
            params["weights"]
        )

        # -------- M-STEP --------
        means, covariances, weights, log_likelihood = m_step.run_m_step(X, gamma)
        params.update({"means": means, "covariances": covariances, "weights": weights})

        print(f"Iteration {iteration:03d} - Log-likelihood: {log_likelihood:.6f}")

        # -------- Convergence Check --------
        if abs(log_likelihood - prev_log_likelihood) < tol:
            print(f"✅ Convergence reached at iteration {iteration}")
            break
        prev_log_likelihood = log_likelihood

    # ==================================================
    # SAVE FINAL PARAMETERS AND RESPONSIBILITIES
    # ==================================================
    os.makedirs("outputs/final", exist_ok=True)
    np.save("outputs/final/gmm_final_params.npy", params)
    np.save("outputs/final/responsibilities.npy", gamma)
    with open("outputs/final/log_likelihood.txt", "w") as f:
        f.write(f"{log_likelihood}\n")

    print("\n✅ EM Algorithm Completed Successfully")
    print(f"Final log-likelihood: {log_likelihood:.6f}")
    print(f"Final parameters saved: outputs/final/gmm_final_params.npy")
    print(f"Responsibilities saved: outputs/final/responsibilities.npy")
    print("🚀 READY FOR NEXT MODULE: VISUALIZATION OR CLUSTER ASSIGNMENT")


if __name__ == "__main__":
    main()
