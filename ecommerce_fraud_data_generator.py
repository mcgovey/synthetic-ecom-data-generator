#!/usr/bin/env python
"""
E-commerce Fraud Data Generator

Legacy entry point - now uses the modular package structure.
For full functionality, use: uv run ecommerce_fraud_generator generate --config_path=config.yaml
"""

import sys
import os

# Add the current directory to Python path to import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecommerce_fraud_generator.cli import main

if __name__ == "__main__":
    main()

