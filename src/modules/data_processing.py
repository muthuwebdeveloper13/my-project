import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.numerical_cols = []
        self.categorical_cols = []

    # -------------------------------------------------
    def load_data(self):
        path = Path(self.config.dataset.path)
        self.data = pd.read_csv(path)

        print("✅ Data loaded successfully")
        print(f"   → Dataset shape: {self.data.shape}")
        print(f"   → Columns: {list(self.data.columns)}")
        return self.data

    # -------------------------------------------------
    def clean_data(self):
        df = self.data.copy()
        before_rows = df.shape[0]

        if "CustomerID" in df.columns:
            df = df.dropna(subset=["CustomerID"])

        if "InvoiceNo" in df.columns:
            df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

        if "UnitPrice" in df.columns:
            df = df[df["UnitPrice"] > 0]

        if "Quantity" in df.columns:
            df = df[df["Quantity"] > 0]

        after_rows = df.shape[0]
        removed = before_rows - after_rows

        self.data = df.reset_index(drop=True)

        print("✅ Invalid records removed")
        print(f"   → Records before: {before_rows}")
        print(f"   → Records after : {after_rows}")
        print(f"   → Removed rows  : {removed}")

        return self.data

    # -------------------------------------------------
    def create_features(self):
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()

        self.data = self.data[self.numerical_cols]

        print("✅ Features created (Numerical only for GMM)")
        print(f"   → Numerical features ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"   → Categorical ignored: {self.categorical_cols}")

        return self.data

    # -------------------------------------------------
    def normalize_features(self):
        means = self.data.mean()
        stds = self.data.std()

        self.data = (self.data - means) / stds
        self.data = self.data.fillna(0)

        print("✅ Features normalized using Z-score")
        print("   Z-score formula: z = (x − μ) / σ")

        # Show example (first feature, first value)
        col = self.data.columns[0]
        example_z = self.data[col].iloc[0]

        print(f"   → Example feature : {col}")
        print(f"   → Mean (μ)        : {means[col]:.4f}")
        print(f"   → Std (σ)         : {stds[col]:.4f}")
        print(f"   → First Z-score   : {example_z:.4f}")

        return self.data

    # -------------------------------------------------
    def visualize_raw_data(self):
        print("📊 Visualizing data distribution (for GMM suitability)")

        self.data.hist(figsize=(10, 6))
        plt.suptitle("Feature Distributions (After Z-score Normalization)")
        plt.tight_layout()
        plt.show()

        if self.data.shape[1] >= 2:
            plt.figure(figsize=(7, 5))
            plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], alpha=0.5)
            plt.xlabel(self.data.columns[0])
            plt.ylabel(self.data.columns[1])
            plt.title("Feature Spread & Overlap (Cluster Feasibility)")
            plt.show()

        print("✅ Data visualization completed")

    # -------------------------------------------------
    def get_preprocessed_data(self):
        self.load_data()
        self.clean_data()
        self.create_features()
        self.normalize_features()
        self.visualize_raw_data()

        return self.data, self.numerical_cols, self.categorical_cols
