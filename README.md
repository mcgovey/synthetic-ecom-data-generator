# Synthetic E-commerce Fraud Data Generator - Phase 1 Enhanced

A sophisticated tool for generating realistic synthetic e-commerce transaction data with fraud labels for developing and testing fraud detection models. **Phase 1 implementation includes customer personas, geographic distribution, fraud campaigns, and enhanced temporal modeling.**

## Key Features

### Phase 1 Enhancements ✨
- **Customer Personas**: Realistic customer segments (budget-conscious, average, premium, high-value)
- **Geographic Distribution**: US metro area-based customer distribution with timezone awareness
- **Fraud Campaigns**: Organized fraud patterns (card testing, account takeover, bust-out fraud)
- **Enhanced Temporal Modeling**: Realistic hourly, daily, and seasonal transaction patterns
- **Business Hour Modeling**: Merchant-specific operating hours and patterns
- **Customer-Merchant Affinity**: Realistic customer preferences for merchant categories
- **Enhanced Payment Instruments**: Persona-based credit card preferences
- **Realistic Device/IP Generation**: Geographic-consistent IP addresses and device patterns

### Core Features
- Generates realistic customer profiles with distinct behaviors
- Creates merchant profiles with varying risk levels and business patterns
- Simulates transaction patterns including both legitimate and fraudulent behaviors
- Campaign-based fraud modeling for realistic attack patterns
- Outputs data in chunked Parquet format for memory efficiency
- Configuration-driven generation using YAML
- Supports friendly fraud scenarios
- Modular architecture for extensibility
- Data quality validation and benchmark comparison

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
```

The setup script will:
1. Install UV if it's not already installed
2. Create a virtual environment
3. Activate the virtual environment
4. Install all required dependencies

## Configuration

The generator is now fully configuration-driven with enhanced Phase 1 parameters. Edit the `config.yaml` file to customize generation:

```yaml
# Phase 1 Enhanced Configuration

# Customer configuration with persona-based generation
customer:
  num_customers: 5000
  personas:
    budget_conscious: 0.35      # 35% budget-conscious customers
    average_spender: 0.40       # 40% average spenders
    premium_customer: 0.20      # 20% premium customers
    high_value: 0.05           # 5% high-value customers

# Geographic distribution settings
geography:
  enabled: true
  international_rate: 0.0      # US-only for now

# Enhanced merchant configuration
merchant:
  num_merchants: 100
  business_patterns:
    operating_hours: true       # Realistic business hours
    seasonal_effects: true      # Seasonal business cycles
    customer_affinity: true     # Customer-merchant preferences

# Enhanced fraud modeling with campaigns
fraud:
  num_transactions: 1000
  campaigns:
    enabled: true
    types:
      card_testing:
        frequency: 0.30
        duration_days: [3, 14]
      account_takeover:
        frequency: 0.20
        duration_days: [7, 30]
      # ... additional campaign types

# Enhanced temporal patterns
temporal_patterns:
  hourly_distribution: true     # Realistic hourly patterns
  seasonal_effects: true       # Holiday and seasonal effects
  holiday_effects: true        # Special event modeling
```

### Key Configuration Sections

#### Customer Personas
- **budget_conscious**: Lower spending, basic devices, limited merchant categories
- **average_spender**: Moderate spending, mixed devices, broad preferences
- **premium_customer**: Higher spending, premium devices, luxury preferences
- **high_value**: Highest spending, latest devices, exclusive merchants

#### Geographic Distribution
- Realistic US metro area distribution
- Geographic-consistent IP addresses
- Timezone-aware transaction patterns
- Affluence-adjusted spending patterns

#### Fraud Campaigns
- **card_testing**: Automated testing of stolen card numbers
- **account_takeover**: Compromised customer accounts
- **friendly_fraud**: Legitimate customers disputing valid charges
- **bust_out**: Long-term identity fraud schemes
- **refund_fraud**: Return and refund abuse

## Usage

### Basic Generation

```bash
# Generate with default config
uv run ecommerce_fraud_data_generator.py generate

# Use custom configuration
uv run ecommerce_fraud_data_generator.py generate --config_path=my_config.yaml
```

### Phase 1 Data Quality Improvements

The enhanced generator produces significantly more realistic data:

1. **Temporal Realism**: Transactions follow natural daily/weekly/seasonal patterns
2. **Customer Behavior**: Distinct personas with consistent spending and device preferences
3. **Merchant Patterns**: Business hours, seasonal effects, and customer affinity
4. **Fraud Sophistication**: Campaign-based attacks with coordinated patterns
5. **Geographic Consistency**: IP addresses, timezones, and regional spending patterns

## Output Structure

The generator creates chunked parquet files optimized for analysis:

```
output/
├── ecommerce_transactions_1000/
│   ├── results_1.parquet     # First chunk
│   ├── results_2.parquet     # Second chunk
│   └── ...                   # Additional chunks
```

### Enhanced Output Schema (Phase 1)

| Column Name | Data Type | Description | Phase 1 Enhancement |
|-------------|-----------|-------------|-------------------|
| `transaction_id` | Integer | Unique transaction identifier | ✓ |
| `customer_id` | Integer | Unique customer identifier | ✓ |
| `customer_persona` | String | Customer persona type | ✨ **NEW** |
| `customer_metro_area` | String | Customer's metro area | ✨ **NEW** |
| `merchant_category` | String | Enhanced merchant category | ✅ **Enhanced** |
| `merchant_years_in_business` | Float | Merchant maturity | ✨ **NEW** |
| `merchant_geographic_scope` | String | Local/regional/national | ✨ **NEW** |
| `timestamp` | DateTime | Transaction timestamp | ✅ **Enhanced patterns** |
| `amount` | Float | Transaction amount | ✅ **Persona-based** |
| `is_business_hours` | Binary | Within merchant hours | ✨ **NEW** |
| `active_fraud_campaign` | Integer | Campaign ID if active | ✨ **NEW** |
| `fraud_campaign_type` | String | Type of fraud campaign | ✨ **NEW** |
| `quarter` | Integer | Business quarter | ✨ **NEW** |
| `is_holiday_season` | Binary | Holiday shopping period | ✨ **NEW** |
| `billing_metro_area` | String | Geographic consistency | ✨ **NEW** |
| `fraud_risk_score` | Float | Enhanced risk calculation | ✅ **Enhanced** |
| `is_fraud` | Binary | Fraud label | ✅ **Campaign-based** |
| `is_friendly_fraud` | Binary | Friendly fraud label | ✓ |

*Legend: ✨ New in Phase 1, ✅ Enhanced in Phase 1, ✓ Unchanged from original*

## Data Quality Improvements

### Before Phase 1
- Random transaction timing
- Uniform customer behavior
- Independent fraud events
- Limited merchant diversity
- Basic risk scoring

### After Phase 1
- **Realistic temporal patterns** with hourly, daily, and seasonal cycles
- **Customer personas** with distinct spending and behavior patterns
- **Fraud campaigns** with coordinated attack patterns
- **Enhanced merchant modeling** with business hours and seasonal effects
- **Geographic realism** with metro area distribution and IP consistency
- **Sophisticated risk scoring** incorporating multiple contextual factors

## Working with Phase 1 Enhanced Data

```python
import pandas as pd
import dask.dataframe as dd

# Load the enhanced dataset
ddf = dd.read_parquet("output/ecommerce_transactions_1000/")

# Analyze customer personas
persona_stats = ddf.groupby('customer_persona').agg({
    'amount': 'mean',
    'is_fraud': 'mean',
    'transaction_id': 'count'
}).compute()

# Analyze fraud campaigns
campaign_analysis = ddf[ddf['active_fraud_campaign'].notna()].groupby([
    'fraud_campaign_type', 'customer_persona'
])['is_fraud'].mean().compute()

# Temporal patterns
hourly_patterns = ddf.groupby('hour_of_day').agg({
    'transaction_id': 'count',
    'is_fraud': 'mean'
}).compute()

# Geographic distribution
metro_analysis = ddf.groupby('customer_metro_area').agg({
    'amount': 'mean',
    'is_fraud': 'mean',
    'transaction_id': 'count'
}).compute()
```

## Phase 1 Validation

The enhanced generator includes validation metrics to ensure data quality:

- **Temporal Distribution**: Realistic hourly/daily/seasonal patterns
- **Customer Behavior Consistency**: Persona-based spending patterns
- **Geographic Realism**: Metro area population alignment
- **Fraud Campaign Patterns**: Coordinated attack sequences
- **Business Logic**: Operating hours and seasonal effects

## Next Steps: Phase 2 & 3

- **Phase 2**: Advanced feature engineering, network analysis, behavioral patterns
- **Phase 3**: Production-grade enhancements, real-time capabilities, model integration

The Phase 1 implementation provides a solid foundation for realistic fraud detection research and model development with significantly improved data quality and realism.
