# src/modules/data_processing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Module 1: Data Collection & Pre-processing Module"""
    
    def __init__(self, config):
        """
        Initialize the data preprocessor with configuration.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.data = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.scaler_params = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV, Excel, or database.
        
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        file_path = Path(self.config.dataset.path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.csv':
            self.data = pd.read_csv(
                file_path,
                delimiter=self.config.dataset.delimiter,
                encoding=self.config.dataset.encoding
            )
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        print(f"✅ Data loaded successfully!")
        print(f"   Shape: {self.data.shape}")
        print(f"   Columns: {list(self.data.columns)}")
        print(f"   Missing values: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Handle missing values and remove noise/outliers.
        
        Args:
            data: Input DataFrame (uses self.data if None)
            
        Returns:
            pandas.DataFrame: Cleaned dataset
        """
        if data is None:
            data = self.data.copy()
        
        df = data.copy()
        
        # Identify column types
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print(f"📊 Data types identified:")
        print(f"   Numerical columns: {len(self.numerical_cols)}")
        print(f"   Categorical columns: {len(self.categorical_cols)}")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers if configured
        if self.config.preprocessing.remove_outliers:
            df = self._remove_outliers(df)
        
        self.data = df
        print(f"✅ Data cleaned successfully!")
        print(f"   New shape: {self.data.shape}")
        
        return self.data
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            print("   No missing values found.")
            return df
        
        print(f"   Found {missing_count} missing values.")
        
        strategy = self.config.preprocessing.handle_missing
        
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.numerical_cols:
                    if strategy == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == "median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == "mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif strategy == "drop":
                        df = df.dropna(subset=[col])
                else:  # Categorical columns
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        print(f"   Missing values handled using '{strategy}' strategy.")
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method"""
        method = self.config.preprocessing.outlier_method
        original_shape = df.shape
        
        if method == "iqr":
            for col in self.numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Keep data within bounds
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == "zscore":
            for col in self.numerical_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]  # Keep within 3 standard deviations
        
        removed_count = original_shape[0] - df.shape[0]
        if removed_count > 0:
            print(f"   Removed {removed_count} outliers using {method} method.")
        
        return df
    
    def normalize_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Standardize features (Z-score) for stable EM performance.
        
        Args:
            data: Input DataFrame (uses self.data if None)
            
        Returns:
            pandas.DataFrame: Normalized dataset
        """
        if data is None:
            data = self.data.copy()
        
        if not self.config.preprocessing.normalize:
            print("⚠️  Normalization is disabled in config.")
            return data
        
        df = data.copy()
        method = self.config.preprocessing.method
        
        print(f"🔧 Normalizing data using '{method}' method...")
        
        # Store original values for inverse transformation if needed
        self.scaler_params = {
            'method': method,
            'means': {},
            'stds': {},
            'mins': {},
            'maxs': {}
        }
        
        for col in self.numerical_cols:
            if method == "zscore":
                # Standardization (Z-score normalization)
                mean_val = df[col].mean()
                std_val = df[col].std()
                self.scaler_params['means'][col] = mean_val
                self.scaler_params['stds'][col] = std_val
                df[col] = (df[col] - mean_val) / std_val if std_val != 0 else 0
            
            elif method == "minmax":
                # Min-Max scaling to [0, 1]
                min_val = df[col].min()
                max_val = df[col].max()
                self.scaler_params['mins'][col] = min_val
                self.scaler_params['maxs'][col] = max_val
                df[col] = (df[col] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
            
            elif method == "robust":
                # Robust scaling using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                median = df[col].median()
                IQR = Q3 - Q1
                self.scaler_params['means'][col] = median
                self.scaler_params['stds'][col] = IQR
                df[col] = (df[col] - median) / IQR if IQR != 0 else 0
        
        self.data = df
        print("✅ Data normalized successfully!")
        
        return self.data
    
    def select_features(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Select specific features for clustering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pandas.DataFrame: Dataset with selected features
        """
        if data is None:
            data = self.data.copy()
        
        df = data.copy()
        
        # Use specified columns
        if self.config.preprocessing.use_columns:
            use_cols = [col for col in self.config.preprocessing.use_columns 
                       if col in df.columns]
            df = df[use_cols]
            print(f"   Using specified columns: {use_cols}")
        
        # Exclude specified columns
        if self.config.preprocessing.exclude_columns:
            exclude_cols = [col for col in self.config.preprocessing.exclude_columns 
                          if col in df.columns]
            df = df.drop(columns=exclude_cols)
            print(f"   Excluded columns: {exclude_cols}")
        
        # Update numerical columns list
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.data = df
        print(f"✅ Feature selection complete!")
        print(f"   Final shape: {self.data.shape}")
        print(f"   Features: {list(self.data.columns)}")
        
        return self.data
    
    def visualize_raw_data(self, save_path: Optional[str] = None) -> None:
        """
        Scatter plots, histograms for understanding distribution.
        
        Args:
            save_path: Path to save the visualization (optional)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data
        
        print("📈 Creating visualizations...")
        
        # Create figure with subplots
        n_cols = len(self.numerical_cols)
        if n_cols == 0:
            print("⚠️  No numerical columns to visualize.")
            return
        
        # Set up subplots
        n_plots = min(6, n_cols * 2)  # Max 6 plots
        n_rows = 2
        n_cols_plots = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols_plots, 
                                figsize=(self.config.preprocessing.figure_size[0],
                                        self.config.preprocessing.figure_size[1]))
        
        # If axes is not 2D, make it 2D
        if n_cols_plots == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        # 1. Histograms for each numerical column
        for i, col in enumerate(self.numerical_cols[:len(axes)//2]):
            if i < len(axes):
                ax = axes[i]
                df[col].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        # 2. Scatter plots for pairs of features
        if len(self.numerical_cols) >= 2:
            scatter_idx = len(axes)//2
            for i in range(min(len(self.numerical_cols)-1, len(axes)-scatter_idx)):
                ax = axes[scatter_idx + i]
                col1 = self.numerical_cols[i]
                col2 = self.numerical_cols[(i + 1) % len(self.numerical_cols)]
                ax.scatter(df[col1], df[col2], alpha=0.6, s=20)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f'{col1} vs {col2}')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.numerical_cols)*2, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_distributions.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        print("✅ Visualizations created successfully!")
    
    def get_preprocessed_data(self) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Complete preprocessing pipeline.
        
        Returns:
            Tuple containing:
            - Preprocessed DataFrame
            - List of numerical columns
            - List of categorical columns
        """
        print("=" * 50)
        print("🚀 STARTING DATA PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Select features
        self.select_features()
        
        # Step 4: Normalize data
        self.normalize_data()
        
        # Step 5: Visualize (if enabled)
        if self.config.preprocessing.plot_raw_data:
            self.visualize_raw_data()
        
        print("=" * 50)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 50)
        
        return self.data, self.numerical_cols, self.categorical_cols