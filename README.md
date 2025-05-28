# Synthetic E-commerce Fraud Data Generator

A tool for generating realistic synthetic e-commerce transaction data with fraud labels for developing and testing fraud detection models.

## Features

- Generates realistic customer profiles
- Creates merchant profiles with varying risk levels
- Simulates transaction patterns including both legitimate and fraudulent behaviors
- Outputs data in Parquet format
- Configuration-driven generation using YAML
- Supports friendly fraud scenarios
- Modular architecture for extensibility
- Data quality validation and benchmark comparison
- Streaming and parallel data generation for performance

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

The script is now fully configuration-driven using a YAML file. You can customize the generation parameters by editing the `config.yaml` file:

```yaml
customer:
  num_customers: 5000
merchant:
  num_merchants: 100
fraud:
  num_transactions: 200000
friendly_fraud:
  enabled: true
  rate: 0.15
  triggers:
    buyer_remorse: 0.3
    family_dispute: 0.2
    subscription_forgotten: 0.25
    delivery_issues: 0.15
    merchant_dispute: 0.1
```

To generate the dataset, simply run:

```bash
uv run ecommerce_fraud_data_generator.py generate --config_path=config.yaml --output_file=output.parquet
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

## Data Quality and Validation

The generator includes data quality validation and benchmark comparison to ensure the realism and accuracy of the generated data.

## Performance and Scalability

The generator supports streaming and parallel data generation to handle large datasets efficiently.
