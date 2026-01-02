# src/main.py
import sys
import os
import numpy as np
from dataclasses import asdict

# Fix Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.modules.data_processing import DataPreprocessor
from src.modules.initialization import initialize_gmm_from_data
from src.modules.expectation import GMMExpectation
from src.modules.maximization import GMMMaximization

def main():
    print("📋 Loading configuration...")

    # --------------------------------------------------
    # Load & validate configuration
    # --------------------------------------------------
    try:
        config = Config.load("config.yaml")
        config.validate()
        print("✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    try:
        # ==================================================
        # MODULE 1: DATA COLLECTION & PRE-PROCESSING
        # ==================================================
        print("\n" + "=" * 50)
        print("MODULE 1: DATA PREPROCESSING")
        print("=" * 50)

        preprocessor = DataPreprocessor(config)

        data, numerical_cols, categorical_cols = preprocessor.get_preprocessed_data()

        # --------------------------------------------------
        # Meaningful summary (exam friendly)
        # --------------------------------------------------
        print("\n📊 Preprocessing Summary")
        print(f"   → Total samples      : {data.shape[0]}")
        print(f"   → Total features     : {data.shape[1]}")
        print(f"   → Numerical features : {numerical_cols}")
        print(f"   → Ignored features   : {categorical_cols}")

        # Show example of Z-score understanding
        print("\n📐 Normalization Check (Z-score example)")
        example_col = numerical_cols[0]
        print(f"   Feature : {example_col}")
        print(f"   Mean ≈ {data[example_col].mean():.4f}")
        print(f"   Std  ≈ {data[example_col].std():.4f}")
        print("   (Mean ≈ 0 and Std ≈ 1 confirms Z-score normalization)")

        # --------------------------------------------------
        # Save processed data
        # --------------------------------------------------
        os.makedirs("data/processed", exist_ok=True)
        data.to_csv("data/processed/preprocessed_data.csv", index=False)
        print("\n💾 Preprocessed data saved to data/processed/preprocessed_data.csv")

        print("\n✅ DATA IS READY FOR GMM (Initialization → E-Step → M-Step)")

    except Exception as e:
        print("\n❌ Error during execution:")
        print(e)

        # ==================================================
        # MODULE 2: GMM INITIALIZATION
        # ==================================================
        print("\n" + "=" * 50)
        print("MODULE 2: GMM INITIALIZATION")
        print("=" * 50)

        X = data[numerical_cols].values
        print(f"📊 GMM Input Shape: {X.shape}")

        config_dict = asdict(config)
        initializer, params = initialize_gmm_from_data(X, config_dict)

        print("✅ GMM Initialized")
        print(f"Means shape       : {params['means'].shape}")
        print(f"Covariances shape : {params['covariances'].shape}")
        print(f"Weights           : {params['weights'].round(3)}")

        if X.shape[1] >= 2:
            initializer.visualize_initialization(
                X,
                feature_indices=(0, 1),
                title="GMM Initialization",
                save_path="outputs/plots/gmm_initialization.png"
            )

        os.makedirs("outputs/models", exist_ok=True)
        initializer.save_parameters("outputs/models/initialized_params.npy")

        # ==================================================
        # MODULE 3: EXPECTATION STEP (E-STEP)
        # ==================================================
        print("\n" + "=" * 50)
        print("MODULE 3: EXPECTATION STEP (E-STEP)")
        print("=" * 50)

        e_step = GMMExpectation()
        responsibilities, log_likelihood = e_step.run_e_step(
            X,
            params["means"],
            params["covariances"],
            params["weights"]
        )

        print("✅ E-Step completed")
        print(f"Responsibilities shape: {responsibilities.shape}")
        print(f"Log-Likelihood        : {log_likelihood:.4f}")

        np.save("outputs/models/responsibilities.npy", responsibilities)

        # ==================================================
        # MODULE 4: MAXIMIZATION STEP (M-STEP)
        # ==================================================
        print("\n" + "=" * 50)
        print("MODULE 4: MAXIMIZATION STEP (M-STEP)")
        print("=" * 50)

        m_step = GMMMaximization(reg_covar=1e-6)
        updated_params = m_step.run_m_step(X, responsibilities)

        print("✅ M-Step completed")
        print(f"Updated weights: {updated_params['weights'].round(3)}")

        np.save("outputs/models/updated_params.npy", updated_params)

        print("\n🎉 ONE FULL EM ITERATION COMPLETED SUCCESSFULLY")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
