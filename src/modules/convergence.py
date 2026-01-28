# =========================
# File: convergence.py
# =========================

import matplotlib.pyplot as plt

class ConvergenceChecker:
    """
    Module 5: Convergence Checking for EM Algorithm
    """

    def __init__(self, tol=1e-3, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.log_likelihoods = []

    # -----------------------------
    # Check Convergence
    # -----------------------------
    def check_convergence(self, current_ll, prev_ll, iteration):
        """
        Returns True if convergence criteria are met
        """
        self.log_likelihoods.append(current_ll)
        delta = abs(current_ll - prev_ll) if prev_ll is not None else None

        if delta is not None and delta < self.tol:
            print(f"✅ Convergence reached at iteration {iteration} (ΔLL = {delta:.6f})")
            return True
        if iteration >= self.max_iter:
            print(f"⚠ Maximum iterations reached ({iteration})")
            return True
        return False

    # -----------------------------
    # Plot Log-Likelihood
    # -----------------------------
    def plot_log_likelihood(self):
        """
        Plots log-likelihood over iterations
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.log_likelihoods)+1), self.log_likelihoods, marker='o')
        plt.title("EM Algorithm Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.grid(True)
        plt.show()
        print("✅ Log-likelihood plot displayed")
