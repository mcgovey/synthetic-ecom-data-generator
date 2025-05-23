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

# Run the setup script (Unix/macOS)
chmod +x setup_project.sh
./setup_project.sh

# For Windows users
setup_project.bat
# Or manually install with UV:
# First install UV: https://astral.sh/uv/install
# uv venv
# .venv\Scripts\activate
# uv pip install -r requirements.txt
```

The setup script will:
1. Install UV if it's not already installed
2. Create a virtual environment
3. Activate the virtual environment
4. Install all required dependencies

## Usage

The script provides a command-line interface using FIRE:

```bash
# Generate dataset with default parameters
uv run ecommerce_fraud_data_generator.py generate

# Customize the data generation
uv run ecommerce_fraud_data_generator.py generate --num_customers=1000 --num_merchants=50 --num_transactions=50000 --output_file="my_dataset.parquet"
```

### Command-line Arguments

- `num_customers`: Number of unique customers to generate (default: 5000)
- `num_merchants`: Number of unique merchants to generate (default: 100)
- `num_transactions`: Number of transactions to generate (default: 200000)
- `output_file`: Filename for the output parquet file (default: "ecommerce_fraud_dataset.parquet")

### Examples

Here are some example use cases:

```bash
# Small dataset for quick testing (few customers, merchants, and transactions)
uv run ecommerce_fraud_data_generator.py generate --num_customers=10 --num_merchants=5 --num_transactions=100 --output_file="small_test.parquet"

# Medium dataset with more customers than merchants
uv run ecommerce_fraud_data_generator.py generate --num_customers=500 --num_merchants=20 --num_transactions=10000 --output_file="medium_dataset.parquet"

# Large dataset with default parameters
uv run ecommerce_fraud_data_generator.py generate

# Custom output location
uv run ecommerce_fraud_data_generator.py generate --output_file="data/fraud_data_$(date +%Y%m%d).parquet"

# Only modify transaction count
uv run ecommerce_fraud_data_generator.py generate --num_transactions=1000000
```

## Output

The generated dataset includes:
- Customer demographics (name, email, age, signup date)
- Merchant information (name, category, risk level)
- Transaction details (amount, timestamp, currency)
- Device and IP address data
- Billing and shipping addresses
- Payment information (credit card BIN, last4)
- Fraud labels and risk scores
- Various derived features useful for fraud detection:
  - Day of week, hour of day
  - Is weekend/night flags
  - Address matching
  - Device and IP newness
  - Days since signup/last purchase
  - Customer purchase count
