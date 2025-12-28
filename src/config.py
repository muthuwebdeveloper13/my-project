# src/config.py
import yaml
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class DatasetConfig:
    path: str
    delimiter: str
    encoding: str

@dataclass
class PreprocessingConfig:
    handle_missing: str
    remove_outliers: bool
    outlier_method: str
    normalize: bool
    method: str
    use_columns: Optional[List[str]]
    exclude_columns: List[str]
    plot_raw_data: bool
    figure_size: List[int]

@dataclass
class ClusteringConfig:
    n_components: int
    covariance_type: str
    max_iter: int
    tolerance: float
    random_state: int

@dataclass
class Config:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    clustering: ClusteringConfig
    
    @classmethod
    def load(cls, config_path: str = "config.yaml"):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Convert nested dictionaries to config objects
        dataset_config = DatasetConfig(**config_dict['dataset'])
        preprocessing_config = PreprocessingConfig(**config_dict['preprocessing'])
        clustering_config = ClusteringConfig(**config_dict['clustering'])
        
        return cls(
            dataset=dataset_config,
            preprocessing=preprocessing_config,
            clustering=clustering_config
        )
    
    def validate(self):
        """Validate configuration parameters"""
        if not os.path.exists(self.dataset.path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset.path}")
        
        valid_handle_missing = ["mean", "median", "mode", "drop"]
        if self.preprocessing.handle_missing not in valid_handle_missing:
            raise ValueError(f"handle_missing must be one of {valid_handle_missing}")
        
        valid_methods = ["zscore", "minmax", "robust"]
        if self.preprocessing.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        
        valid_covariance = ["full", "tied", "diag", "spherical"]
        if self.clustering.covariance_type not in valid_covariance:
            raise ValueError(f"covariance_type must be one of {valid_covariance}")
        
        return True