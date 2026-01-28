# =========================================================
# GMM PROJECT - MAIN PIPELINE
# =========================================================

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Create necessary directories
for dir_path in ["outputs/final", "data/processed"]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Import project modules
from config import Config
from modules.data_processing import DataPreprocessor
from modules.initialization import GMMInitializer
from modules.expectation import GMMExpectation
from modules.maximization import GMMMaximization
from modules.convergence import ConvergenceChecker
from modules.clustering import GMMClustering
from modules.visualization import GMMVisualization
from modules.evaluation import GMMEvaluation
from modules.documentation import GMMDocumentation


class GMMPipeline:
    """Main pipeline for Gaussian Mixture Model implementation."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the GMM pipeline with configuration."""
        self.config_path = config_path
        self.config = None
        self.data = None
        self.X = None
        self.K = None
        self.means = None
        self.covariances = None
        self.weights = None
        self.responsibilities = None
        self.cluster_labels = None
        self.log_likelihoods = []
        
    def load_configuration(self):
        """Load and validate configuration."""
        print("\n[MODULE 0] CONFIGURATION")
        print("-" * 70)
        
        self.config = Config.load(self.config_path)
        self.config.validate()
        print("✓ Configuration loaded and validated successfully")
        
        return self
    
    def preprocess_data(self):
        """Preprocess the dataset."""
        print("\n[MODULE 1] DATA PREPROCESSING")
        print("-" * 70)
        
        preprocessor = DataPreprocessor(self.config)
        self.data, num_cols, cat_cols = preprocessor.get_preprocessed_data()
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Numerical columns: {num_cols}")
        print(f"Categorical columns: {cat_cols}")
        
        # Save processed data
        self.data.to_csv("data/processed/preprocessed_data.csv", index=False)
        print("✓ Preprocessed data saved")
        
        # Convert to numpy array
        self.X = self.data.values
        print(f"Feature matrix X shape: {self.X.shape}")
        
        return self
    
    def initialize_parameters(self):
        """Initialize GMM parameters from config."""
        print(f"\n[MODULE 2] PARAMETER INITIALIZATION")
        print("-" * 70)
        
        self.K = self.config.clustering.n_components
        print(f"Number of clusters (K): {self.K}")
        
        initializer = GMMInitializer(self.data)
        
        self.means, self.covariances, self.weights = initializer.initialize_all(
            K=self.K,
            mean_method="kmeans",
            cov_method="identity"
        )
        
        print("Initial parameters set:")
        print(f"  Means shape: {self.means.shape}")
        print(f"  Covariances shape: {self.covariances.shape}")
        print(f"  Weights shape: {self.weights.shape}")
        
        return self
    
    def run_em_algorithm(self):
        """Execute Expectation-Maximization algorithm."""
        print(f"\n[MODULE 3-5] EM ALGORITHM")
        print("-" * 70)
        
        # Get parameters from config
        max_iter = self.config.clustering.max_iterations
        tol = self.config.clustering.tolerance
        
        print(f"Max iterations: {max_iter}, Tolerance: {tol}")
        
        # Initialize EM components
        e_step = GMMExpectation()
        m_step = GMMMaximization()
        convergence = ConvergenceChecker(tol=tol, max_iter=max_iter)
        
        prev_log_likelihood = None
        
        for iteration in range(max_iter):
            print(f"\nIteration {iteration + 1}/{max_iter}")
            
            # E-Step
            self.responsibilities, log_likelihood_e = e_step.run_e_step(
                self.X, self.means, self.covariances, self.weights
            )
            
            # M-Step
            self.means, self.covariances, self.weights, log_likelihood_m = m_step.run_m_step(
                self.X, self.responsibilities
            )
            
            # Store log likelihood (use M-step value)
            self.log_likelihoods.append(log_likelihood_m)
            
            # Print current status
            print(f"  Log-Likelihood: {log_likelihood_m:.4f}")
            
            # Calculate delta (change in log likelihood)
            if prev_log_likelihood is not None:
                delta = abs(log_likelihood_m - prev_log_likelihood)
                print(f"  Δ Log-Likelihood: {delta:.6f}")
            else:
                print(f"  Δ Log-Likelihood: N/A")
            
            # Check convergence - pass all three required parameters
            if convergence.check_convergence(log_likelihood_m, prev_log_likelihood, iteration + 1):
                print(f"\n✓ Convergence achieved after {iteration + 1} iterations")
                break
            
            prev_log_likelihood = log_likelihood_m
        
        print("\n✓ EM algorithm completed")
        
        # Save EM results
        self.save_em_results()
        
        # Plot convergence - The ConvergenceChecker should already have log_likelihoods
        # stored from check_convergence method
        convergence.plot_log_likelihood()
        
        # Also save our own plot
        self.plot_log_likelihood_convergence()
        
        return self
    
    def save_em_results(self):
        """Save EM algorithm results."""
        # Save numpy arrays
        np.save("outputs/final/responsibilities.npy", self.responsibilities)
        np.save("outputs/final/final_means.npy", self.means)
        np.save("outputs/final/final_covariances.npy", self.covariances)
        np.save("outputs/final/final_weights.npy", self.weights)
        
        # Assign and save cluster labels
        self.cluster_labels = np.argmax(self.responsibilities, axis=1)
        np.save("outputs/final/cluster_labels.npy", self.cluster_labels)
        
        # Also save as CSV for readability
        pd.DataFrame(self.cluster_labels, columns=['Cluster']).to_csv(
            "outputs/final/cluster_assignments.csv", index=False
        )
        
        print("✓ EM results saved to outputs/final/")
        
        return self
    
    def plot_log_likelihood_convergence(self):
        """Plot the log-likelihood convergence."""
        if len(self.log_likelihoods) > 1:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.log_likelihoods) + 1), self.log_likelihoods, 
                    marker='o', linestyle='-', color='blue')
            plt.title('EM Algorithm Convergence')
            plt.xlabel('Iteration')
            plt.ylabel('Log-Likelihood')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("outputs/final/log_likelihood_convergence.png", dpi=300)
            plt.close()
            print("✓ Log-likelihood convergence plot saved as outputs/final/log_likelihood_convergence.png")
    
    def perform_clustering(self):
        """Perform clustering based on learned parameters."""
        print("\n[MODULE 6] CLUSTERING")
        print("-" * 70)
        
        clustering = GMMClustering(self.means, self.covariances, self.weights)
        assigned_labels = clustering.assign_cluster(self.responsibilities)
        
        print(f"Cluster distribution:")
        unique, counts = np.unique(assigned_labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            percentage = (count / len(assigned_labels)) * 100
            print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
        
        return self
    
    def visualize_results(self):
        """Visualize clustering results."""
        print("\n[MODULE 7] VISUALIZATION")
        print("-" * 70)
        
        if self.X.shape[1] < 2:
            print("⚠ Visualization requires at least 2 features")
            return self
        
        viz = GMMVisualization(self.X, self.means, self.covariances, self.cluster_labels)
        
        # Create plots
        try:
            viz.plot_clusters()
            viz.plot_gaussians()
            viz.plot_distributions()
            print("✓ Visualizations generated and saved")
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
            print("  Continuing with other modules...")
        
        return self
    
    def evaluate_model(self):
        """Evaluate GMM performance with error handling."""
        print("\n[MODULE 8] MODEL EVALUATION")
        print("-" * 70)
        
        try:
            evaluator = GMMEvaluation(
                self.X,
                self.cluster_labels,
                self.responsibilities,
                self.means,
                self.covariances,
                self.weights
            )
            
            print("Calculating evaluation metrics...")
            
            # Calculate silhouette score
            silhouette = evaluator.silhouette()
            print(f"✓ Silhouette Score: {silhouette:.4f}")
            
            # Calculate BIC and AIC (with timeout protection)
            try:
                print("Calculating BIC and AIC (this may take a moment)...")
                bic, aic = evaluator.bic_aic()
                print(f"✓ Bayesian Information Criterion (BIC): {bic:.2f}")
                print(f"✓ Akaike Information Criterion (AIC): {aic:.2f}")
            except Exception as e:
                print(f"⚠ BIC/AIC calculation skipped: {e}")
                bic, aic = np.nan, np.nan
            
            # Compare with KMeans
            print("Comparing with KMeans...")
            gmm_sil, kmeans_sil = evaluator.compare_with_kmeans()
            print(f"✓ GMM Silhouette: {gmm_sil:.4f}")
            print(f"✓ KMeans Silhouette: {kmeans_sil:.4f}")
            
            # Get cluster statistics
            print("\nCluster Statistics:")
            stats = evaluator.get_cluster_statistics()
            for stat in stats:
                print(f"  Cluster {stat['cluster']}: {stat['size']} samples ({stat['percentage']:.1f}%), Weight: {stat['weight']:.3f}")
            
        except Exception as e:
            print(f"⚠ Evaluation error: {e}")
            print("  Continuing with other modules...")
        
        return self
    
    def generate_documentation(self):
        """Generate documentation and final report."""
        print("\n[MODULE 9] DOCUMENTATION")
        print("-" * 70)
        
        try:
            documentation = GMMDocumentation(output_dir="outputs/final")
            
            # Generate summary report
            documentation.generate_summary(
                means=self.means,
                covariances=self.covariances,
                weights=self.weights,
                log_likelihoods=self.log_likelihoods,
                cluster_labels=self.cluster_labels
            )
            
            # Export model
            documentation.export_model(
                means=self.means,
                covariances=self.covariances,
                weights=self.weights
            )
            
            print("✓ Documentation generated")
            print("✓ Model exported for future use")
        except Exception as e:
            print(f"⚠ Documentation error: {e}")
            print("  Continuing with pipeline summary...")
        
        return self
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline execution."""
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        
        print(f"Dataset: {self.config.dataset.path}")
        print(f"Original shape: {self.data.shape if self.data is not None else 'N/A'}")
        print(f"Number of clusters: {self.K}")
        print(f"EM iterations: {len(self.log_likelihoods)}")
        if self.log_likelihoods:
            print(f"Final log-likelihood: {self.log_likelihoods[-1]:.4f}")
        else:
            print(f"Final log-likelihood: N/A")
        print(f"Output directory: outputs/final/")
        
        if self.cluster_labels is not None:
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            print(f"Cluster distribution: {dict(zip(unique, counts))}")
        
        print("=" * 80)


def main():
    """Execute the complete GMM pipeline."""
    print("\n" + "=" * 80)
    print("GAUSSIAN MIXTURE MODEL - FULL IMPLEMENTATION")
    print("=" * 80)
    
    try:
        # Create and execute pipeline
        pipeline = GMMPipeline(config_path="config.yaml")
        
        pipeline.load_configuration()
        pipeline.preprocess_data()
        pipeline.initialize_parameters()
        pipeline.run_em_algorithm()
        pipeline.perform_clustering()
        pipeline.visualize_results()
        pipeline.evaluate_model()
        pipeline.generate_documentation()
        pipeline.print_pipeline_summary()
        
        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        print("Please check your config.yaml file and ensure the dataset path is correct.")
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Please ensure all required modules are available.")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()