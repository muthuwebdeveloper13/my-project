import yaml
import os
from dataclasses import dataclass
from typing import List, Optional

# -----------------------------
# Dataset Configuration
# -----------------------------
@dataclass
class DatasetConfig:
    path: str
    delimiter: str
    encoding: str

# -----------------------------
# Preprocessing Configuration
# -----------------------------
@dataclass
class PreprocessingConfig:
    handle_missing: str
    remove_outliers: bool
    outlier_method: str
    normalize: bool
    method: str
    plot_raw_data: bool
    figure_size: List[int]

# -----------------------------
# Clustering Configuration
# -----------------------------
@dataclass
class ClusteringConfig:
    n_components: int
    max_iterations: int
    tolerance: float
    random_state: int

# -----------------------------
# Main Config Class
# -----------------------------
@dataclass
class Config:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    clustering: ClusteringConfig

    @classmethod
    def load(cls, path="config.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        dataset_cfg = DatasetConfig(**cfg["dataset"])
        preprocessing_cfg = PreprocessingConfig(**cfg["preprocessing"])
        clustering_cfg = ClusteringConfig(**cfg["clustering"])

        return cls(dataset_cfg, preprocessing_cfg, clustering_cfg)

    def validate(self):
        if not os.path.exists(self.dataset.path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset.path}")

        if self.clustering.n_components <= 0:
            raise ValueError("n_components must be > 0")

        if self.clustering.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")

        if self.clustering.tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        print("✔ Configuration validated successfully")
