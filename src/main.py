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
    """Main function to run the GMM clustering pipeline"""

    # ============================================================
    # LOAD CONFIGURATION
    # ============================================================
    print("📋 Loading configuration...")
    try:
        config = Config.load("config.yaml")
        config.validate()
        print("✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    try:
        # ============================================================
        # MODULE 1: DATA PREPROCESSING
        # ============================================================
        print("\n" + "=" * 50)
        print("MODULE 1: DATA PREPROCESSING")
        print("=" * 50)

        preprocessor = DataPreprocessor(config)
        data, numerical_cols, categorical_cols = preprocessor.get_preprocessed_data()

        print("\n📊 DATA SUMMARY:")
        print(f"   Samples: {data.shape[0]}")
        print(f"   Features: {data.shape[1]}")
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Feature names: {list(data.columns)}")

        # Save preprocessed data
        processed_path = "data/processed/preprocessed_data.csv"
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        data.to_csv(processed_path, index=False)
        print(f"\n💾 Preprocessed data saved to: {processed_path}")

        # ============================================================
        # MODULE 2: GMM PARAMETER INITIALIZATION
        # ============================================================
        print("\n" + "=" * 50)
        print("MODULE 2: GMM PARAMETER INITIALIZATION")
        print("=" * 50)

        # Use only numerical columns for GMM
        X = data[numerical_cols].values
        print(f"📊 Data matrix shape for GMM: {X.shape}")

        config_dict = asdict(config)

        initializer, params = initialize_gmm_from_data(X, config_dict)

        print("\n✅ GMM PARAMETERS INITIALIZED:")
        print(f"   Number of components: {params['n_components']}")
        print(f"   Means shape: {params['means'].shape}")
        print(f"   Covariances shape: {params['covariances'].shape}")
        print(f"   Weights: {params['weights'].round(3)}")
        print(f"   Initialization method: {params.get('initialization_method', 'kmeans')}")

        # Visualization
        if X.shape[1] >= 2:
            initializer.visualize_initialization(
                X,
                feature_indices=(0, 1),
                title="GMM Initialization for Customer Segmentation",
                save_path="outputs/plots/gmm_initialization.png"
            )

        # Save initialized parameters
        init_param_path = "outputs/models/initialized_params.npy"
        os.makedirs(os.path.dirname(init_param_path), exist_ok=True)
        initializer.save_parameters(init_param_path)
        print(f"\n💾 Initialized parameters saved to: {init_param_path}")

        # ============================================================
        # MODULE 3: EXPECTATION STEP (E-STEP)
        # ============================================================
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

        print("✅ E-Step completed successfully!")
        print(f"   Responsibilities shape: {responsibilities.shape}")
        print(f"   Log-Likelihood: {log_likelihood:.4f}")

        resp_path = "outputs/models/responsibilities.npy"
        os.makedirs(os.path.dirname(resp_path), exist_ok=True)
        np.save(resp_path, responsibilities)
        print(f"💾 Responsibilities saved to: {resp_path}")

        # ============================================================
        # MODULE 4: MAXIMIZATION STEP (M-STEP)
        # ============================================================
        print("\n" + "=" * 50)
        print("MODULE 4: MAXIMIZATION STEP (M-STEP)")
        print("=" * 50)

        m_step = GMMMaximization(reg_covar=1e-6)

        updated_params = m_step.run_m_step(
            X,
            responsibilities
        )

        print("✅ M-Step completed successfully!")
        print(f"   Updated weights: {updated_params['weights'].round(3)}")
        print(f"   Updated means shape: {updated_params['means'].shape}")
        print(f"   Updated covariances shape: {updated_params['covariances'].shape}")

        mstep_path = "outputs/models/updated_params.npy"
        os.makedirs(os.path.dirname(mstep_path), exist_ok=True)
        np.save(mstep_path, updated_params)

        print(f"💾 Updated GMM parameters saved to: {mstep_path}")
        print("\n🎉 ONE FULL EM ITERATION COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
