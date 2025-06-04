"""Command-line interface for the fraud data generator using Fire."""

import os
import fire
from .generators.data_generator import FraudDataGenerator
from .config.settings import create_default_config


def main():
    """Main entry point for the CLI."""
    fire.Fire(FraudDataGeneratorCLI)


class FraudDataGeneratorCLI:
    """
    Command-line interface for the E-commerce Fraud Data Generator.

    This tool creates realistic synthetic e-commerce transaction datasets
    with fraud patterns for machine learning model development and testing.
    """

    def __init__(self):
        self.generator = FraudDataGenerator()

    def generate(self, config_path: str = 'config.yaml'):
        """
        Generate synthetic e-commerce transaction data and save to parquet files.

        Args:
            config_path (str): Path to the configuration YAML file

        Returns:
            str: Path to the directory containing the generated parquet files
        """
        # Create default config if it doesn't exist
        if not os.path.exists(config_path):
            print(f"Config file {config_path} not found. Creating default configuration...")
            self.create_config(config_path)

        return self.generator.generate(config_path)

    def create_config(self, output_path: str = 'config.yaml'):
        """
        Create a default configuration file.

        Args:
            output_path (str): Path where to save the configuration file
        """
        config_content = create_default_config()

        with open(output_path, 'w') as file:
            file.write(config_content)

        print(f"Default configuration created at: {output_path}")
        print("You can modify this file to customize the data generation parameters.")

    def info(self):
        """Display information about the fraud data generator."""
        info_text = """
E-commerce Fraud Data Generator
==============================

This tool generates synthetic e-commerce transaction data with realistic
fraud patterns for machine learning model development and testing.

Features:
- Customer personas with distinct behavior patterns
- Geographic distribution modeling
- Realistic temporal patterns
- Organized fraud campaigns
- Friendly fraud simulation
- Chunked processing for memory efficiency

Usage:
  python -m ecommerce_fraud_generator generate --config_path=config.yaml
  python -m ecommerce_fraud_generator create_config --output_path=my_config.yaml
  python -m ecommerce_fraud_generator info

Configuration:
The generator uses a YAML configuration file to specify dataset parameters.
Use 'create_config' command to generate a default configuration file.

Output:
The generated dataset is saved as parquet files in chunks for efficient
processing with tools like Pandas, Dask, or Spark.
        """
        print(info_text)


if __name__ == '__main__':
    main()
