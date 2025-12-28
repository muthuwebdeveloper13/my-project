# src/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.modules.data_processing import DataPreprocessor

def main():
    """Main function to run the GMM clustering pipeline"""
    
    # Load configuration
    print("📋 Loading configuration...")
    try:
        config = Config.load("config.yaml")
        config.validate()
        print("✅ Configuration loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Initialize and run Module 1: Data Preprocessing
    print("\n" + "="*50)
    print("MODULE 1: DATA PREPROCESSING")
    print("="*50)
    
    preprocessor = DataPreprocessor(config)
    
    try:
        # Run the complete preprocessing pipeline
        data, numerical_cols, categorical_cols = preprocessor.get_preprocessed_data()
        
        # Display summary
        print("\n📊 DATA SUMMARY:")
        print(f"   Samples: {data.shape[0]}")
        print(f"   Features: {data.shape[1]}")
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        print(f"   Feature names: {list(data.columns)}")
        
        # Save preprocessed data (optional)
        output_path = "data/processed/preprocessed_data.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        print(f"\n💾 Preprocessed data saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()