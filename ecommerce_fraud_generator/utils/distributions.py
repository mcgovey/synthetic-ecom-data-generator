"""Statistical distribution utilities for realistic data generation."""

import numpy as np
from scipy.stats import pareto, weibull_min, multivariate_normal
from typing import Tuple, List


class RealisticDistributions:
    """Use real-world statistical distributions for data generation."""

    def __init__(self):
        # Transaction amounts follow Pareto distribution (80/20 rule)
        self.amount_dist = pareto(b=1.16)  # Based on real e-commerce data

        # Time between purchases follows Weibull distribution
        self.purchase_interval_dist = weibull_min(c=1.5, scale=30)

    def generate_transaction_amount(self, base_amount: float, variance: float = 0.5) -> float:
        """
        Generate realistic transaction amount using Pareto distribution.

        Args:
            base_amount: Base amount to scale the distribution
            variance: Amount of variance in the distribution

        Returns:
            Generated transaction amount
        """
        raw_amount = self.amount_dist.rvs() * base_amount * variance
        return max(1.0, raw_amount)  # Ensure minimum amount of $1

    def generate_purchase_interval(self) -> int:
        """
        Generate realistic time interval between purchases in days.

        Returns:
            Number of days until next purchase
        """
        interval = self.purchase_interval_dist.rvs()
        return max(1, int(interval))  # Minimum 1 day


class CorrelatedFeatureGenerator:
    """Generate correlated features using multivariate distributions."""

    def __init__(self):
        # Define correlation matrix for related features
        self.correlation_matrix = np.array([
            [1.0, 0.7, 0.3],    # age, income, avg_purchase_amount correlation
            [0.7, 1.0, 0.5],    # income, age, avg_purchase_amount correlation
            [0.3, 0.5, 1.0]     # avg_purchase_amount, income, age correlation
        ])

    def generate_correlated_features(self, num_samples: int) -> np.ndarray:
        """
        Generate correlated customer features.

        Args:
            num_samples: Number of feature sets to generate

        Returns:
            Array of correlated features [age, income, avg_purchase_amount]
        """
        mean = [40, 50000, 100]  # Example means for age, income, avg_purchase_amount
        return multivariate_normal.rvs(
            mean=mean,
            cov=self.correlation_matrix,
            size=num_samples
        )

    def generate_customer_demographics(self) -> Tuple[int, float, float]:
        """
        Generate correlated customer demographics.

        Returns:
            Tuple of (age, income, avg_purchase_amount)
        """
        features = self.generate_correlated_features(1)
        age = max(18, min(80, int(features[0])))  # Age between 18-80
        income = max(20000, features[1])  # Minimum income threshold
        avg_purchase = max(10, features[2])  # Minimum purchase amount

        return age, income, avg_purchase
