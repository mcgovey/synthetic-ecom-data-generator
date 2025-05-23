# Synthetic E-commerce Fraud Data Generator

A tool for generating realistic synthetic e-commerce transaction data with fraud labels for developing and testing fraud detection models.

## Features

- Generates realistic customer profiles
- Creates merchant profiles with varying risk levels
- Simulates transaction patterns including both legitimate and fraudulent behaviors
- Outputs data in Parquet format
- Customizable number of customers, merchants, and transactions

## Installation

This project uses UV for dependency management. To set up the project:

```bash
# Clone the repository
git clone <repository-url>
cd synthetic-ecom-data-generator

# Create a virtual environment and install dependencies with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

The script provides a command-line interface using FIRE:

```bash
# Generate dataset with default parameters
python ecommerce_fraud_data_generator.py generate

# Customize the data generation
python ecommerce_fraud_data_generator.py generate --num_customers=1000 --num_merchants=50 --num_transactions=50000 --output_file="my_dataset.parquet"
```

### Command-line Arguments

- `num_customers`: Number of unique customers to generate (default: 5000)
- `num_merchants`: Number of unique merchants to generate (default: 100)
- `num_transactions`: Number of transactions to generate (default: 200000)
- `output_file`: Filename for the output parquet file (default: "ecommerce_fraud_dataset.parquet")

## Output

The generated dataset includes:
- Customer demographics
- Merchant information
- Transaction details
- Device and IP address data
- Fraud labels and risk scores
- Various derived features useful for fraud detection
