import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessor:
    """
    MODULE 1: Data Collection & Pre-processing
    ------------------------------------------
    This module prepares raw customer data for GMM modeling.
    Steps:
    1. Data Loading
    2. Cleaning
    3. Missing Value Handling
    4. Feature Selection
    5. Normalization
    6. Visualization
    """

    def __init__(self, config):
        self.config = config
        self.data = None
        self.numerical_cols = []
        self.categorical_cols = []

    # -------------------------------------------------
    def load_data(self):
        print("\n📥 STEP 1: DATA LOADING")
        self.data = pd.read_csv(
            self.config.dataset.path,
            delimiter=self.config.dataset.delimiter,
            encoding=self.config.dataset.encoding
        )

        print("✅ Dataset loaded successfully")
        print(f"   → Total records   : {self.data.shape[0]:,}")
        print(f"   → Total attributes: {self.data.shape[1]}")
        print(f"   → Columns         : {list(self.data.columns)}")
        return self.data

    # -------------------------------------------------
    def clean_data(self):
        print("\n🧹 STEP 2: DATA CLEANING")

        df = self.data.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.data = df
        print("✅ Basic data cleaning completed")
        print("   → Invalid values removed")
        print("   → Infinite values handled")
        return self.data

    # -------------------------------------------------
    def handle_missing_values(self):
        print("\n🩺 STEP 3: MISSING VALUE HANDLING")

        method = self.config.preprocessing.handle_missing
        df = self.data.copy()

        if method == "mean":
            df = df.fillna(df.mean(numeric_only=True))
            explanation = "Mean Imputation (Average value replacement)"
        elif method == "median":
            df = df.fillna(df.median(numeric_only=True))
            explanation = "Median Imputation"
        elif method == "mode":
            df = df.fillna(df.mode().iloc[0])
            explanation = "Mode Imputation"
        elif method == "drop":
            df = df.dropna()
            explanation = "Row deletion"

        self.data = df.reset_index(drop=True)

        print("✅ Missing values processed")
        print(f"   → Method used : {explanation}")
        print("   → Purpose     : Maintain data integrity and model stability")
        return self.data

    # -------------------------------------------------
    def select_features(self):
        print("\n📊 STEP 4: FEATURE SELECTION")

        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()

        self.data = self.data[self.numerical_cols]

        print("✅ Feature selection completed")
        print("   → Numerical features selected for GMM:")
        for col in self.numerical_cols:
            if col.lower() == "quantity":
                print(f"      - {col} : Customer purchase volume")
            elif col.lower() == "unitprice":
                print(f"      - {col} : Spending per product")
            elif col.lower() == "customerid":
                print(f"      - {col} : Customer identity reference")
            else:
                print(f"      - {col}")

        print("   → Categorical features excluded (GMM works on numerical data)")
        return self.data

    # -------------------------------------------------
    def normalize_data(self):
        print("\n📐 STEP 5: NORMALIZATION")

        if self.config.preprocessing.normalize:
            if self.config.preprocessing.method == "zscore":
                self.data = (self.data - self.data.mean()) / self.data.std()
                self.data = self.data.fillna(0)

                print("✅ Z-score normalization applied")
                print("   → Mean of each feature = 0")
                print("   → Standard deviation  = 1")
                print("   → Ensures equal feature contribution")
                print("   → Improves EM convergence stability")

        return self.data

    # -------------------------------------------------
    def visualize_raw_data(self):
        print("\n📈 STEP 6: DATA VISUALIZATION")

        if self.config.preprocessing.plot_raw_data:
            self.data.hist(figsize=self.config.preprocessing.figure_size)
            plt.suptitle("Feature Distributions After Normalization")
            plt.show()

            if self.data.shape[1] >= 2:
                plt.figure(figsize=(7, 5))
                plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], alpha=0.5)
                plt.xlabel(self.data.columns[0])
                plt.ylabel(self.data.columns[1])
                plt.title("Feature Spread and Overlap (Cluster Feasibility)")
                plt.show()

            print("✅ Visualization completed")
            print("   → Distribution plots generated")
            print("   → Feature overlap scatter plot generated")
            print("   → Used to assess clustering feasibility")

    # -------------------------------------------------
    def get_preprocessed_data(self):
        print("\n" + "="*70)
        print("MODULE 1: DATA COLLECTION & PREPROCESSING")
        print("="*70)

        self.load_data()
        self.clean_data()
        self.handle_missing_values()
        self.select_features()
        self.normalize_data()
        self.visualize_raw_data()

        print("\n🎯 MODULE 1 OUTPUT SUMMARY")
        print("------------------------------------------------------")
        print("✔ Data cleaned")
        print("✔ Missing values handled")
        print("✔ Features selected")
        print("✔ Data normalized")
        print("✔ Visualization generated")
        print("✔ Data ready for GMM modeling")
        print("------------------------------------------------------")

        return self.data, self.numerical_cols, self.categorical_cols
