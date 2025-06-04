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

## Dataset Schema

The generated dataset contains the following columns:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| `transaction_id` | Integer | Unique identifier for each transaction |
| `customer_id` | Integer | Unique identifier for each customer |
| `merchant_id` | Integer | Unique identifier for each merchant |
| `timestamp` | DateTime | Date and time when the transaction occurred |
| `amount` | Float | Transaction amount in USD (rounded to 2 decimal places) |
| `currency` | String | Currency code (always 'USD' in this dataset) |
| `customer_name` | String | Full name of the customer |
| `customer_email` | String | Email address of the customer |
| `customer_phone` | String | Phone number of the customer |
| `customer_age` | Integer | Age of the customer (18-80 years) |
| `days_since_signup` | Integer | Number of days since customer signed up |
| `device_id` | String | Unique identifier for the device used in transaction |
| `os` | String | Operating system of the device (iOS, Android, Windows, macOS, Linux) |
| `browser` | String | Browser used for the transaction (Chrome, Safari, Firefox, Edge, Opera) |
| `user_agent` | String | Complete user agent string from the browser |
| `ip_address` | String | IP address from which the transaction originated |
| `billing_street` | String | Street address for billing |
| `billing_city` | String | City for billing address |
| `billing_state` | String | State abbreviation for billing address |
| `billing_zip` | String | ZIP code for billing address |
| `billing_country` | String | Country for billing address (always 'US') |
| `shipping_street` | String | Street address for shipping |
| `shipping_city` | String | City for shipping address |
| `shipping_state` | String | State abbreviation for shipping address |
| `shipping_zip` | String | ZIP code for shipping address |
| `shipping_country` | String | Country for shipping address (always 'US') |
| `cc_bin` | String | First 6 digits of the credit card (Bank Identification Number) |
| `cc_last4` | String | Last 4 digits of the credit card |
| `cc_expiry` | String | Credit card expiry date |
| `merchant_name` | String | Name of the merchant |
| `merchant_category` | String | Category of the merchant (Electronics, Clothing, Food, etc.) |
| `days_since_last_purchase` | Integer | Number of days since customer's last purchase |
| `customer_purchase_count` | Integer | Total number of purchases made by the customer |
| `address_match` | Binary (0/1) | Whether billing and shipping addresses match (1=match, 0=different) |
| `is_new_device` | Binary (0/1) | Whether the transaction used a new device for this customer |
| `is_new_ip` | Binary (0/1) | Whether the transaction used a new IP address for this customer |
| `is_international` | Binary (0/1) | Whether the transaction is international (always 0 in this dataset) |
| `is_fraud` | Binary (0/1) | **Target variable**: Whether the transaction is fraudulent (1=fraud, 0=legitimate) |
| `is_friendly_fraud` | Binary (0/1) | Whether the transaction is friendly fraud (customer disputes legitimate charge) |
| `day_of_week` | Integer | Day of the week (0=Monday, 6=Sunday) |
| `hour_of_day` | Integer | Hour of the day when transaction occurred (0-23) |
| `is_weekend` | Binary (0/1) | Whether the transaction occurred on weekend (Saturday/Sunday) |
| `is_night` | Binary (0/1) | Whether the transaction occurred at night (10 PM - 5 AM) |
| `fraud_risk_score` | Float | Calculated risk score (0-1) based on various risk factors |

### Key Target Variables

- **`is_fraud`**: Primary target variable for fraud detection models. Binary classification (0=legitimate, 1=fraudulent)
- **`is_friendly_fraud`**: Secondary target for friendly fraud detection models
- **`fraud_risk_score`**: Continuous risk score that can be used for regression models or risk-based ranking

### Feature Categories

1. **Transaction Features**: `amount`, `timestamp`, `currency`
2. **Customer Features**: `customer_*`, `days_since_signup`, `customer_purchase_count`
3. **Merchant Features**: `merchant_name`, `merchant_category`
4. **Device/Technical Features**: `device_id`, `os`, `browser`, `user_agent`, `ip_address`
5. **Address Features**: `billing_*`, `shipping_*`, `address_match`
6. **Behavioral Features**: `days_since_last_purchase`, `is_new_device`, `is_new_ip`
7. **Temporal Features**: `day_of_week`, `hour_of_day`, `is_weekend`, `is_night`
8. **Payment Features**: `cc_bin`, `cc_last4`, `cc_expiry`

## Data Quality and Validation

The generator includes data quality validation and benchmark comparison to ensure the realism and accuracy of the generated data.

## Performance and Scalability

The generator supports streaming and parallel data generation to handle large datasets efficiently.
