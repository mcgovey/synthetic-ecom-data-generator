# E-commerce Fraud Data Generator

A sophisticated synthetic data generator for creating realistic e-commerce transaction datasets with fraud patterns for machine learning model development and testing.

## 🚀 Enhanced Features

**NEW**: Realistic fraud rates, adversarial intelligence, and comprehensive metadata generation for production-ready fraud detection models.

### Key Improvements
- **Realistic Fraud Rates**: Industry-standard fraud rates (0.5-3%) with validation
- **Adversarial Intelligence**: Fraud patterns that actively evade detection systems
- **Enhanced Metadata**: Comprehensive payment processing, device fingerprinting, and network data
- **Detection Evasion**: Sophisticated fraud tactics including pattern mimicry and threshold awareness
- **Industry Configuration**: Configurable fraud rates by industry type (e-commerce, digital goods, travel, financial services)

## Features

### Core Data Generation
- **Customer Personas**: Distinct behavior patterns (budget_conscious, average_spender, premium_customer, high_value)
- **Geographic Distribution**: Realistic US metro area distribution with affluence modeling
- **Temporal Patterns**: Seasonal, daily, and hourly transaction patterns
- **Fraud Campaigns**: Organized fraud attacks (card testing, account takeover, friendly fraud, etc.)
- **Realistic Business Logic**: Merchant operating hours, seasonal patterns, customer affinity
- **Memory Efficient**: Chunked processing for large datasets using Dask/Pandas
- **Configurable**: YAML-based configuration for all parameters

### Enhanced Features
- **Fraud Rate Controller**: Validates and enforces realistic fraud rates with industry-specific multipliers
- **Adversarial Fraud Generator**: Creates sophisticated fraud patterns with detection evasion tactics
- **Enhanced Metadata Generator**: Generates comprehensive transaction metadata including:
  - Payment processing data (authorization codes, processing times, decline reasons)
  - Device fingerprinting (screen resolution, browser language, automation detection)
  - Network metadata (VPN/proxy detection, IP geolocation, connection types)
  - Transaction flow (cart abandonment, checkout duration, payment attempts)
  - Payment instrument data (realistic BINs, issuing banks, verification results)

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
│   ├── data_generator.py     # Main dataset generator (enhanced)
│   ├── fraud_generator.py    # Adversarial fraud pattern generators
│   └── metadata_generator.py # Comprehensive metadata generator
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

#### Generate Dataset with Enhancements

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

The generator uses YAML configuration files with enhanced features. Example `config.yaml`:

```yaml
# Enhanced Configuration
fraud:
  num_transactions: 10000

  # Realistic fraud rate configuration (0.5-3% total fraud rate)
  fraud_rates:
    industry_type: "general_ecommerce"  # Options: general_ecommerce, digital_goods, travel, financial_services
    base_fraud_rate: 0.015              # 1.5% base fraud rate
    friendly_fraud_rate: 0.008          # 0.8% friendly fraud rate
    technical_fraud_rate: 0.007         # 0.7% technical fraud rate

    # Industry-specific multipliers
    industry_multipliers:
      general_ecommerce: 1.0
      digital_goods: 2.5      # Higher fraud rate for digital goods
      travel: 1.8             # Higher fraud rate for travel
      financial_services: 0.6  # Lower fraud rate for financial services

    # Enable fraud rate validation and warnings
    validation:
      enabled: true
      warn_if_above: 0.05     # Warn if fraud rate exceeds 5%
      error_if_above: 0.10    # Error if fraud rate exceeds 10%

  # Enhanced adversarial intelligence
  adversarial_patterns:
    enabled: true
    detection_evasion:
      amount_threshold_awareness: true    # Fraudsters avoid common amount thresholds
      velocity_threshold_awareness: true  # Fraudsters space transactions to avoid velocity checks
      geographic_evasion: true           # Fraudsters use VPNs and proxies strategically

    adaptive_behavior:
      enabled: true
      evolution_rate: 0.1     # How quickly fraud patterns adapt (10% per week)
      mimicry_intelligence: 0.7  # How well fraudsters mimic legitimate behavior (0-1)

    evasion_tactics:
      legitimate_pattern_mimicry: true    # Copy victim's shopping patterns
      gradual_escalation: true           # Start small, gradually increase risk
      mixed_legitimate_fraud: true       # Mix fraud with legitimate transactions
      dormancy_periods: true             # Go dormant when detection risk is high

# Enhanced raw transaction metadata generation
data_realism:
  # Comprehensive payment processing metadata
  payment_processing:
    enabled: true
    generate_authorization_codes: true
    generate_processing_times: true
    generate_decline_reasons: true
    generate_payment_processor_data: true

  # Device fingerprinting data
  device_fingerprinting:
    enabled: true
    generate_screen_resolution: true
    generate_browser_language: true
    generate_session_duration: true
    generate_automation_signatures: true

  # Network and session metadata
  network_metadata:
    enabled: true
    generate_vpn_detection: true
    generate_proxy_detection: true
    generate_ip_geolocation: true
    generate_connection_type: true

  # Transaction flow metadata
  transaction_flow:
    enabled: true
    generate_cart_abandonment: true
    generate_checkout_duration: true
    generate_payment_attempts: true
    generate_session_context: true

  # Enhanced payment instrument data
  payment_instruments:
    enabled: true
    generate_realistic_bins: true
    generate_issuing_banks: true
    generate_card_countries: true
    generate_funding_types: true
    generate_verification_results: true

# Enhanced statistics and validation
output:
  validation:
    enabled: true
    show_correlations: true
    show_distributions: true
    validate_fraud_rates: true
    show_adversarial_metrics: true
```

## Data Schema

The generated dataset includes comprehensive features for production fraud detection:

### Core Transaction Data
- `transaction_id`: Unique transaction identifier
- `customer_id`: Customer identifier
- `merchant_id`: Merchant identifier
- `timestamp`: Transaction timestamp
- `amount`: Transaction amount (USD)
- `currency`: Always 'USD'

### Enhanced Payment Processing Metadata
- `authorization_code`: Payment authorization code
- `processing_time_ms`: Transaction processing time
- `payment_processor`: Payment processor used (stripe, square, paypal, etc.)
- `processor_transaction_id`: Processor-specific transaction ID
- `gateway_response_code`: Gateway response code
- `processor_fee_cents`: Processing fee charged

### Enhanced Device Fingerprinting
- `device_id`, `os`, `browser`, `user_agent`: Device fingerprinting
- `screen_resolution`: Device screen resolution
- `browser_language`: Browser language setting
- `session_duration_seconds`: Session duration
- `automation_detected`: Whether automation was detected
- `automation_signature`: Type of automation detected
- `webdriver_present`: Whether webdriver was detected
- `plugins_count`: Number of browser plugins
- `fonts_count`: Number of system fonts

### Enhanced Network Metadata
- `ip_address`: Transaction IP address
- `vpn_detected`: Whether VPN was detected
- `vpn_confidence_score`: VPN detection confidence
- `vpn_provider`: VPN provider (if detected)
- `proxy_detected`: Whether proxy was detected
- `proxy_type`: Type of proxy detected
- `ip_country`, `ip_region`, `ip_city`: IP geolocation
- `ip_latitude`, `ip_longitude`: IP coordinates
- `connection_type`: Connection type (broadband, mobile, etc.)
- `isp_provider`: Internet service provider

### Enhanced Transaction Flow
- `cart_abandonment_count_30d`: Cart abandonments in last 30 days
- `checkout_duration_seconds`: Time spent in checkout
- `payment_attempts_count`: Number of payment attempts
- `failed_payment_attempts`: Number of failed attempts
- `session_page_views`: Pages viewed in session
- `time_on_site_seconds`: Total time on site
- `referrer_domain`: Referrer domain

### Enhanced Payment Instrument Data
- `card_bin`: Credit card BIN (realistic)
- `issuing_bank`: Card issuing bank
- `card_country`: Card issuing country
- `funding_type`: Card funding type (credit, debit, prepaid)
- `cvv_match`: CVV verification result
- `avs_result`: Address verification result
- `cvv_provided`: Whether CVV was provided
- `avs_provided`: Whether address was provided for verification

### Fraud Intelligence Metadata
- `fraud_intelligence_level`: Intelligence level of fraud (0-1)
- `fraud_evasion_tactics`: Evasion tactics used (comma-separated)
- `fraud_detection_risk_score`: Detection risk score (0-1)

### Traditional Risk Features
- `days_since_last_purchase`: Time since previous transaction
- `customer_purchase_count`: Total customer transactions
- `is_new_device`, `is_new_ip`: New device/IP indicators
- `is_business_hours`: Whether transaction during business hours
- `address_match`: Whether billing and shipping addresses match

### Fraud Labels
- `is_fraud`: Primary fraud label (0/1) - **Realistic rates: 0.5-3%**
- `is_friendly_fraud`: Friendly fraud label (0/1)

## Adversarial Intelligence

### Detection Evasion Tactics
- **Amount Threshold Awareness**: Fraudsters avoid common detection thresholds ($100, $500, $1000)
- **Velocity Evasion**: Transactions spaced to avoid velocity detection
- **Pattern Mimicry**: High-intelligence fraud copies victim's shopping patterns
- **Gradual Escalation**: Account takeover starts small and gradually increases risk
- **Geographic Evasion**: Strategic use of VPNs and proxies

### Fraud Intelligence Levels
- **Card Testing**: 0.8 intelligence (high sophistication)
- **Account Takeover**: 0.9 intelligence (very high sophistication)
- **Friendly Fraud**: 0.3 intelligence (low sophistication - customer dispute)
- **Bust Out**: 0.7 intelligence (high sophistication)
- **Refund Fraud**: 0.6 intelligence (medium sophistication)

## Industry-Specific Fraud Rates

| Industry Type | Base Fraud Rate | Typical Range |
|---------------|----------------|---------------|
| General E-commerce | 1.5% | 1.0-2.0% |
| Digital Goods | 3.75% | 2.5-5.0% |
| Travel | 2.7% | 2.0-3.5% |
| Financial Services | 0.9% | 0.5-1.5% |

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
- **Categories**: Luxury items, Travel, Jewelry
- **Devices**: Latest iOS, macOS
- **Fraud Susceptibility**: High (60%)

## Validation and Quality Assurance

The implementation includes comprehensive validation:

- **Fraud Rate Validation**: Automatic warnings for unrealistic fraud rates
- **Industry Standard Compliance**: Built-in industry-specific fraud rate multipliers
- **Adversarial Pattern Validation**: Ensures fraud patterns use appropriate evasion tactics
- **Metadata Completeness**: Validates that all required metadata fields are generated
- **Correlation Analysis**: Shows relationships between features and fraud labels

## Getting Started

1. **Configure realistic fraud rates** for your industry
2. **Enable adversarial patterns** for sophisticated fraud generation
3. **Activate comprehensive metadata generation** for production-ready features
4. **Run validation** to ensure data quality meets standards
5. **Analyze output** using built-in statistical validation tools

```bash
# Generate with enhancements
uv run ecommerce_fraud_data_generator.py generate --config_path=config.yaml
```

## Contributing

The system represents a significant step toward production-ready synthetic fraud data. Future enhancements will include:

- **Network attack modeling** and fraud ring generation
- **Advanced payment ecosystem integration** and macro-economic modeling

See the expert critique in `refs/critiques_3.md` for detailed analysis and roadmap.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Phase 2: Advanced Behavioral Modeling and Temporal Evolution

### Overview
Phase 2 introduces sophisticated behavioral modeling that captures how both legitimate users and fraudsters behave over time, with advanced evasion tactics and temporal fraud evolution.

### Key Features

#### 1. Temporal Fraud Evolution
- **Fraud Lifecycle**: Campaigns evolve through phases (exploration, exploitation, adaptation, migration)
- **Detection Adaptation**: Fraudsters become more sophisticated as they learn from detection attempts
- **Seasonal Patterns**: Fraud rates adjust based on shopping seasons (e.g., 1.8x during Black Friday)

#### 2. Behavioral Modeling
- **User Consistency**: Legitimate users show consistent behavior patterns (0.85 default consistency)
- **Fraud Mimicry**: Advanced fraudsters mimic legitimate user patterns (0.70 mimicry strength)
- **Privacy-Conscious Users**: Legitimate users using VPNs, ad blockers (25% rate)

#### 3. Advanced Evasion Tactics
- **ML Model Awareness**: Sophisticated fraudsters understand detection thresholds (0.60 awareness)
- **Feature Engineering Awareness**: Fraudsters manipulate specific features to avoid detection
- **Ensemble Evasion**: Multi-model evasion strategies for different detection systems

#### 4. Network Effects
- **Fraud Networks**: Coordinated attacks with 3-15 connected accounts
- **Velocity Patterns**: Burst, sustained, or distributed attack patterns
- **Network Topologies**: Hub-and-spoke, distributed, layered, and hybrid structures

#### 5. Legitimate User Sophistication
- **Tech-Savvy Behaviors**: Multiple browsers, extensions, developer tools (35% rate)
- **Cross-Device Usage**: Users switching between devices (60% rate)
- **Privacy Tools**: VPN usage, tracking protection, incognito browsing

### Configuration Example

```yaml
# Phase 2: Advanced Behavioral Modeling and Temporal Evolution
phase_2:
  enabled: true

  # Temporal fraud evolution
  temporal_evolution:
    enabled: true
    fraud_lifecycle_days: 90  # How long fraud patterns evolve
    detection_adaptation_rate: 0.15  # How quickly fraudsters adapt
    seasonal_patterns: true

  # Behavioral modeling
  behavioral_modeling:
    enabled: true
    user_behavior_consistency: 0.85  # How consistent legitimate users are
    fraud_behavior_mimicry: 0.70     # How well fraud mimics legitimate behavior
    anomaly_detection_threshold: 0.25

  # Advanced evasion tactics
  advanced_evasion:
    enabled: true
    ml_model_awareness: 0.60         # How aware fraudsters are of ML detection
    feature_engineering_awareness: 0.45
    ensemble_evasion: true           # Multi-model evasion tactics

  # Network effects
  network_effects:
    enabled: true
    fraud_network_size: [3, 15]     # Range of connected fraud accounts
    velocity_patterns: true         # Coordinated velocity attacks
    account_similarity_scores: true

  # Legitimate user sophistication
  legitimate_sophistication:
    enabled: true
    privacy_conscious_rate: 0.25     # Users with VPNs, ad blockers, etc.
    tech_savvy_behaviors: 0.35
    cross_device_usage: 0.60
```

### New Features in Generated Data

#### Temporal Evolution Features
- `fraud_campaign_age_days`: How long the fraud campaign has been running
- `fraud_phase`: Current phase (exploration, exploitation, adaptation, migration)
- `evolved_sophistication`: Intelligence level adjusted for temporal learning
- `seasonal_fraud_multiplier`: Seasonal adjustment factor

#### Behavioral Consistency Features
- `behavior_consistency_score`: How consistent the user's behavior is
- `tech_savviness_score`: User's technical sophistication level
- `privacy_consciousness_score`: How privacy-focused the user is

#### Mimicry and Evasion Features
- `mimicry_patterns`: Which legitimate patterns fraud is copying
- `evasion_tactics`: Specific ML evasion techniques being used
- `legitimate_sophistication`: Advanced legitimate user behaviors

#### Network Analysis Features
- `network_id`: Identifier for coordinated fraud networks
- `network_role`: Role within fraud network (hub, spoke, controller, etc.)
- `coordination_score`: How well coordinated the network attack is
- `velocity_pattern`: Type of coordinated attack (burst, sustained, distributed)

### Impact on Data Quality

Phase 2 dramatically improves dataset realism by:

1. **Reducing False Positives**: Legitimate users now exhibit sophisticated behaviors that might previously appear suspicious
2. **Increasing Detection Difficulty**: Fraud patterns evolve and adapt, mimicking real-world adversarial evolution
3. **Temporal Realism**: Fraud campaigns show realistic evolution over time rather than static patterns
4. **Network Behavior**: Coordinated attacks reflect real fraud network operations
5. **Behavioral Nuance**: Users show realistic inconsistencies and adaptations over time
