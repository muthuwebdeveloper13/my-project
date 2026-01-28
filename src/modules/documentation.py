import os
import numpy as np
from datetime import datetime

class GMMDocumentation:
    def __init__(self, output_dir="outputs/final"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------
    # Generate project summary report
    # -------------------------------------------------
    def generate_summary(self, means, covariances, weights, log_likelihoods, cluster_labels):
        summary_path = os.path.join(self.output_dir, "gmm_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("============================================================\n")
            f.write("GAUSSIAN MIXTURE MODEL - FINAL PROJECT REPORT\n")
            f.write("============================================================\n\n")

            f.write(f"Report Generated On : {datetime.now()}\n\n")

            f.write("MODEL CONFIGURATION\n")
            f.write("------------------------------------------------------------\n")
            f.write(f"Number of Components (K) : {len(weights)}\n")
            f.write(f"Feature Dimension        : {means.shape[1]}\n\n")

            f.write("TRAINING RESULTS (EM ALGORITHM)\n")
            f.write("------------------------------------------------------------\n")
            f.write(f"Final Log-Likelihood : {log_likelihoods[-1]}\n")
            f.write(f"Total EM Iterations  : {len(log_likelihoods)}\n\n")

            f.write("MIXTURE WEIGHTS (π_k)\n")
            f.write("------------------------------------------------------------\n")
            f.write(str(weights) + "\n\n")

            f.write("MEANS (μ_k)\n")
            f.write("------------------------------------------------------------\n")
            f.write(str(means) + "\n\n")

            f.write("COVARIANCES (Σ_k)\n")
            f.write("------------------------------------------------------------\n")
            f.write(str(covariances) + "\n\n")

            f.write("CLUSTER DISTRIBUTION\n")
            f.write("------------------------------------------------------------\n")
            unique, counts = np.unique(cluster_labels, return_counts=True)
            for u, c in zip(unique, counts):
                f.write(f"Cluster {u} : {c} samples\n")

            f.write("\n============================================================\n")
            f.write("END OF REPORT\n")
            f.write("============================================================\n")

        return summary_path

    # -------------------------------------------------
    # Export trained GMM model
    # -------------------------------------------------
    def export_model(self, means, covariances, weights):
        np.save(os.path.join(self.output_dir, "final_means.npy"), means)
        np.save(os.path.join(self.output_dir, "final_covariances.npy"), covariances)
        np.save(os.path.join(self.output_dir, "final_weights.npy"), weights)
