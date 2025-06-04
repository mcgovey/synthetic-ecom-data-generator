# E-commerce Fraud Data Generator

A sophisticated synthetic data generator for creating realistic e-commerce transaction datasets with fraud patterns for machine learning model development and testing.

## Features

- **Customer Personas**: Distinct behavior patterns (budget_conscious, average_spender, premium_customer, high_value)
- **Geographic Distribution**: Realistic US metro area distribution with affluence modeling
- **Temporal Patterns**: Seasonal, daily, and hourly transaction patterns
- **Fraud Campaigns**: Organized fraud attacks (card testing, account takeover, friendly fraud, etc.)
- **Realistic Business Logic**: Merchant operating hours, seasonal patterns, customer affinity
- **Memory Efficient**: Chunked processing for large datasets using Dask/Pandas
- **Configurable**: YAML-based configuration for all parameters

## Project Structure

```
ecommerce_fraud_generator/
├── __init__.py                 # Package initialization
├── __main__.py                # Module execution entry point
├── cli.py                     # Command-line interface
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py           # Global settings and personas
├── models/                    # Data models
│   ├── __init__.py
│   ├── customer.py           # Customer model with personas
│   ├── merchant.py           # Merchant model with business patterns
│   └── fraud_campaign.py     # Fraud campaign modeling
├── generators/                # Data generation engines
│   ├── __init__.py
│   ├── data_generator.py     # Main dataset generator
│   └── fraud_generator.py    # Fraud pattern generators
└── utils/                     # Utility functions
    ├── __init__.py
    ├── temporal_patterns.py  # Time-based patterns
    └── distributions.py      # Statistical distributions
```

## Installation

### Prerequisites

- Python 3.8+
- Required packages (install via requirements.txt):

```bash
pip install -r requirements.txt
```

### Required Dependencies

- pandas >= 2.0.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- faker >= 18.0.0
- dask >= 2024.0.0
- pyarrow >= 10.0.0
- fire >= 0.4.0
- PyYAML >= 6.0

## Usage

### Command Line Interface

#### Generate Dataset

```bash
# Using the modular package (recommended)
python -m ecommerce_fraud_generator generate --config_path=config.yaml

# Using the legacy script (backward compatibility)
python ecommerce_fraud_data_generator.py generate --config_path=config.yaml
```

#### Create Configuration File

```bash
python -m ecommerce_fraud_generator create_config --output_path=my_config.yaml
```

#### Get Information

```bash
python -m ecommerce_fraud_generator info
```

### Python API

```python
from ecommerce_fraud_generator import FraudDataGenerator

# Initialize generator
generator = FraudDataGenerator()

# Generate dataset
output_dir = generator.generate('config.yaml')
print(f"Dataset saved to: {output_dir}")
```

### Configuration

The generator uses YAML configuration files. Example `config.yaml`:

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
output:
  file:
    path: "output/ecommerce_transactions_1000"
    chunk_size: 1000
```

## Data Schema

The generated dataset includes the following features:

### Transaction Identifiers
- `transaction_id`: Unique transaction identifier
- `customer_id`: Customer identifier
- `merchant_id`: Merchant identifier
- `timestamp`: Transaction timestamp

### Customer Information
- `customer_name`, `customer_email`, `customer_phone`: PII data
- `customer_age`: Customer age
- `customer_persona`: Assigned persona type
- `customer_metro_area`: Geographic location
- `days_since_signup`: Account age

### Transaction Details
- `amount`: Transaction amount (USD)
- `currency`: Always 'USD'
- `merchant_name`, `merchant_category`: Merchant information
- `merchant_years_in_business`: Business maturity
- `merchant_geographic_scope`: Business reach

### Device & Network
- `device_id`, `os`, `browser`, `user_agent`: Device fingerprinting
- `ip_address`: Transaction IP address

### Address Information
- `billing_*`: Billing address fields
- `shipping_*`: Shipping address fields
- `address_match`: Whether billing and shipping addresses match

### Payment Information
- `cc_bin`: Credit card BIN
- `cc_last4`: Last 4 digits of card
- `cc_expiry`: Card expiration

### Risk Features
- `days_since_last_purchase`: Time since previous transaction
- `customer_purchase_count`: Total customer transactions
- `is_new_device`, `is_new_ip`: New device/IP indicators
- `is_business_hours`: Whether transaction during business hours
- `is_weekend`, `is_night`: Temporal indicators

### Fraud Information
- `active_fraud_campaign`: Active campaign ID (if any)
- `fraud_campaign_type`: Type of fraud campaign
- `is_fraud`: Primary fraud label (0/1)
- `is_friendly_fraud`: Friendly fraud label (0/1)

## Customer Personas

### Budget Conscious (35% of customers)
- **Amount Range**: $15-75
- **Frequency**: 1-4 transactions/month
- **Categories**: Food, Home Goods, Clothing
- **Devices**: Android, Windows
- **Fraud Susceptibility**: Low (30%)

### Average Spender (40% of customers)
- **Amount Range**: $50-200
- **Frequency**: 2-8 transactions/month
- **Categories**: Electronics, Clothing, Health & Beauty
- **Devices**: Mixed platforms
- **Fraud Susceptibility**: Medium (40%)

### Premium Customer (20% of customers)
- **Amount Range**: $150-800
- **Frequency**: 5-15 transactions/month
- **Categories**: Electronics, Jewelry, Travel
- **Devices**: iOS, macOS, Windows
- **Fraud Susceptibility**: Medium-High (50%)

### High Value (5% of customers)
- **Amount Range**: $500-2000
- **Frequency**: 8-25 transactions/month
- **Categories**: Jewelry, Travel, Electronics, Services
- **Devices**: Premium (iOS, macOS)
- **Fraud Susceptibility**: High (70%)

## Fraud Types

### Organized Campaigns
- **Card Testing**: Small amounts, digital products
- **Account Takeover**: High-value customers, new devices
- **Bust Out**: Long-term credit building schemes
- **Refund Fraud**: Return policy exploitation

### Friendly Fraud
- **Buyer's Remorse**: Post-purchase disputes
- **Family Disputes**: Unauthorized family member purchases
- **Subscription Forgotten**: Recurring charge disputes
- **Delivery Issues**: Package theft claims

## Geographic Distribution

Realistic distribution across US metro areas:
- New York (12%), Los Angeles (8%), Chicago (5%)
- Major metros with affluence scoring
- Rural areas (20%) with different spending patterns

## Temporal Patterns

### Hourly Patterns
- Morning lull (2-6 AM)
- Lunch peak (12-1 PM)
- Evening peak (6-9 PM)

### Seasonal Patterns
- Holiday season boost (November-December)
- Back-to-school surge (August)
- Post-holiday lull (January)

## Performance

### Memory Management
- Chunked processing prevents memory overflow
- Configurable chunk sizes (default: 1000 transactions)
- Parallel processing capability (future enhancement)

### Output Formats
- Parquet files for efficient storage and processing
- Dask-compatible partitioned datasets
- Pandas/Spark readable formats

## Examples

### Small Test Dataset
```yaml
customer:
  num_customers: 100
merchant:
  num_merchants: 10
fraud:
  num_transactions: 1000
output:
  file:
    path: "test_output"
    chunk_size: 100
```

### Large Production Dataset
```yaml
customer:
  num_customers: 50000
merchant:
  num_merchants: 1000
fraud:
  num_transactions: 10000000
output:
  file:
    path: "production_dataset"
    chunk_size: 10000
```

## Best Practices

1. **Memory Management**: Use appropriate chunk sizes based on available RAM
2. **Configuration**: Always validate configuration before large dataset generation
3. **Storage**: Ensure sufficient disk space (approximately 1GB per 1M transactions)
4. **Reproducibility**: Use consistent random seeds for reproducible results

## Contributing

1. Follow the modular architecture patterns
2. Add comprehensive docstrings to all functions
3. Include type hints for function parameters and returns
4. Update tests for any new functionality
5. Document configuration changes in README

## Architecture Principles

The refactored codebase follows these principles:

- **Separation of Concerns**: Each module has a single responsibility
- **Configuration-Driven**: All parameters externalized to YAML
- **Type Safety**: Full type hints throughout
- **Memory Efficiency**: Chunked processing for large datasets
- **Extensibility**: Easy to add new personas, fraud types, or features
- **Testability**: Modular design enables comprehensive testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
