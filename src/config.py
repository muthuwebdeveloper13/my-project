import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional

# -----------------------------
# Dataset configuration
# -----------------------------
@dataclass
class DatasetConfig:
    path: str
    delimiter: str = ","
    encoding: str = "utf-8"

# -----------------------------
# Preprocessing configuration
# -----------------------------
@dataclass
class PreprocessingConfig:
    handle_missing: str = "mean"              # mean, median, mode, drop
    remove_outliers: bool = True
    outlier_method: str = "zscore"            # zscore, iqr
    normalize: bool = True
    method: str = "zscore"                     # normalization method: zscore, minmax, robust
    use_columns: Optional[List[str]] = None
    exclude_columns: List[str] = field(default_factory=list)
    plot_raw_data: bool = True
    figure_size: List[int] = field(default_factory=lambda: [10, 6])

# -----------------------------
# Clustering (GMM) configuration
# -----------------------------
@dataclass
class ClusteringConfig:
    n_components: int = 3
    covariance_type: str = "full"             # full, tied, diag, spherical
    max_iterations: int = 100
    tolerance: float = 1e-4
    random_state: int = 42

# -----------------------------
# Full project configuration
# -----------------------------
@dataclass
class Config:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    clustering: ClusteringConfig

    @classmethod
    def load(cls, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        # Convert nested dictionaries to dataclasses
        dataset_cfg = DatasetConfig(**cfg_dict.get("dataset", {}))
        preprocessing_cfg = PreprocessingConfig(**cfg_dict.get("preprocessing", {}))
        clustering_cfg = ClusteringConfig(**cfg_dict.get("clustering", {}))

        return cls(
            dataset=dataset_cfg,
            preprocessing=preprocessing_cfg,
            clustering=clustering_cfg
        )

    def validate(self):
        """Validate configuration parameters"""

        # Dataset
        if not os.path.exists(self.dataset.path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset.path}")

        # Preprocessing
        valid_missing = ["mean", "median", "mode", "drop"]
        if self.preprocessing.handle_missing not in valid_missing:
            raise ValueError(f"handle_missing must be one of {valid_missing}")

        valid_methods = ["zscore", "minmax", "robust"]
        if self.preprocessing.method not in valid_methods:
            raise ValueError(f"Normalization method must be one of {valid_methods}")

        # Clustering
        valid_covariance = ["full", "tied", "diag", "spherical"]
        if self.clustering.covariance_type not in valid_covariance:
            raise ValueError(f"covariance_type must be one of {valid_covariance}")

        if self.clustering.n_components <= 0:
            raise ValueError("n_components must be > 0")

        if self.clustering.max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")

        if self.clustering.tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        return True
