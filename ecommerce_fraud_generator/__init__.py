"""
E-commerce Fraud Data Generator

A synthetic data generator for creating realistic e-commerce transaction datasets
with fraud patterns for machine learning model development and testing.
"""

__version__ = "0.0.1"
__author__ = "@mcgovey"

from .generators.data_generator import FraudDataGenerator
from .cli import main

__all__ = ["FraudDataGenerator", "main"]
