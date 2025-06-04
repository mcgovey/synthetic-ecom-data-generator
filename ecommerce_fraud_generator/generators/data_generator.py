"""Main data generator for creating synthetic e-commerce fraud datasets."""

import os
import random
import datetime
import pandas as pd
import numpy as np
import dask.dataframe as dd
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from scipy.stats import bernoulli

from ..config.settings import GlobalSettings, load_config
from ..models.customer import Customer
from ..models.merchant import Merchant
from ..models.fraud_campaign import generate_fraud_campaigns
from ..utils.temporal_patterns import TemporalPatterns
from .fraud_generator import FriendlyFraudGenerator

# Set random seeds for reproducibility
np.random.seed(GlobalSettings.NUMPY_SEED)
random.seed(GlobalSettings.RANDOM_SEED)


class FraudDataGenerator:
    """
    Generate synthetic e-commerce transaction data with fraud labels.

    This tool creates a realistic dataset of e-commerce transactions
    with both legitimate and fraudulent patterns for use in fraud detection
    model development and testing.
    """

    def __init__(self):
        self.customers = {}
        self.merchants = {}
        self.fraud_campaigns = []
        self.friendly_fraud_generator = FriendlyFraudGenerator()
        self.temporal_patterns = TemporalPatterns()

    def generate(self, config_path: str = 'config.yaml') -> str:
        """
        Generate synthetic e-commerce transaction data and save to parquet files.

        Args:
            config_path (str): Path to the configuration YAML file

        Returns:
            str: Path to the directory containing the generated parquet files
        """
        # Load configuration
        config = load_config(config_path)

        # Get configuration parameters
        num_customers = config['customer']['num_customers']
        num_merchants = config['merchant']['num_merchants']
        num_transactions = config['fraud']['num_transactions']

        # Get output configuration
        output_config = config.get('output', {}).get('file', {})
        output_path = output_config.get('path', 'output/ecommerce_fraud_dataset')
        chunk_size = output_config.get('chunk_size', 100)

        # Extract directory and filename pattern from the path
        if output_path.endswith('.parquet'):
            # Remove .parquet extension and use the directory and base filename
            output_dir = os.path.dirname(output_path)
            filename_pattern = os.path.basename(output_path).replace('.parquet', '')
        else:
            output_dir = output_path
            filename_pattern = 'results_'

        print("Initializing data generation...")
        self._initialize_entities(num_customers, num_merchants)

        print("Generating fraud campaigns...")
        self.fraud_campaigns = generate_fraud_campaigns(
            GlobalSettings.START_DATE,
            GlobalSettings.END_DATE,
            num_merchants
        )
        print(f"Generated {len(self.fraud_campaigns)} fraud campaigns")

        # Generate the dataset
        generated_output_dir = self._generate_dataset(
            num_transactions=num_transactions,
            chunk_size=chunk_size,
            output_dir=output_dir,
            filename_pattern=filename_pattern
        )

        print(f"Dataset generated in chunks in directory: {generated_output_dir}")

        # Display statistics
        self._display_statistics(generated_output_dir)

        return generated_output_dir

    def _initialize_entities(self, num_customers: int, num_merchants: int) -> None:
        """Initialize customers and merchants."""
        print("Generating customers with personas and geographic distribution...")
        self.customers = {
            customer_id: Customer(customer_id)
            for customer_id in range(1, num_customers + 1)
        }

        print("Generating merchants with realistic business patterns...")
        self.merchants = {
            merchant_id: Merchant(merchant_id)
            for merchant_id in range(1, num_merchants + 1)
        }

    def _generate_dataset(self, num_transactions: int, chunk_size: int,
                         output_dir: str, filename_pattern: str) -> str:
        """Generate the complete dataset in chunks."""
        print("Generating transactions in chunks with enhanced realism...")

        # Customer purchase history tracking
        customer_purchase_counts = defaultdict(int)

        # Generate all transaction timestamps with realistic temporal patterns
        all_timestamps = self._generate_timestamps(num_transactions)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Remove any existing parquet files from previous runs
        self._cleanup_output_directory(output_dir)

        # Generate data in chunks
        for i in range(0, len(all_timestamps), chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, len(all_timestamps))
            current_timestamps = all_timestamps[chunk_start:chunk_end]

            print(f"Generating transactions for chunk {i // chunk_size + 1}/"
                  f"{(len(all_timestamps) + chunk_size - 1) // chunk_size} "
                  f"({chunk_start+1}-{chunk_end})")

            chunk_df = self._generate_chunk(
                current_timestamps, chunk_start, customer_purchase_counts
            )

            # Save chunk to parquet
            chunk_filename = f"{filename_pattern}{i // chunk_size + 1}.parquet"
            chunk_filepath = os.path.join(output_dir, chunk_filename)
            chunk_df.to_parquet(chunk_filepath, index=False, engine='pyarrow')

        return output_dir

    def _generate_timestamps(self, num_transactions: int) -> List[datetime.datetime]:
        """Generate realistic timestamps for all transactions."""
        all_timestamps = []

        # Get temporal weights
        hourly_weights = self.temporal_patterns.get_hourly_weights()
        daily_weights = self.temporal_patterns.get_daily_weights()

        for _ in range(num_transactions):
            # Generate realistic timestamp
            random_date = GlobalSettings.START_DATE + datetime.timedelta(
                seconds=random.uniform(0, (GlobalSettings.END_DATE - GlobalSettings.START_DATE).total_seconds())
            )

            # Apply seasonal multiplier to decide if transaction should occur
            seasonal_mult = self.temporal_patterns.get_seasonal_multiplier(random_date)
            if random.random() > seasonal_mult * 0.7:  # Base acceptance rate
                continue

            # Select hour and day based on realistic patterns
            hour = random.choices(range(24), weights=hourly_weights)[0]
            day_of_week = random.choices(range(7), weights=daily_weights)[0]

            # Adjust the random date to have realistic hour
            final_timestamp = random_date.replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            all_timestamps.append(final_timestamp)

        # Sort timestamps and trim to exact count needed
        return sorted(all_timestamps)[:num_transactions]

    def _generate_chunk(self, timestamps: List[datetime.datetime],
                       chunk_start: int, customer_purchase_counts: Dict[int, int]) -> pd.DataFrame:
        """Generate a chunk of transaction data."""
        chunk_transactions_list = []

        for idx, timestamp in enumerate(timestamps):
            transaction_id = chunk_start + idx + 1

            # Select customer and merchant
            customer_id, customer = self._select_customer(timestamp)
            merchant_id, merchant = self._select_merchant(customer, timestamp)

            # Generate transaction details
            transaction_data = self._generate_transaction(
                transaction_id, customer, merchant, timestamp, customer_purchase_counts
            )

            customer.update_last_purchase(timestamp)
            chunk_transactions_list.append(transaction_data)

        # Convert to DataFrame and add derived features
        chunk_df = pd.DataFrame(chunk_transactions_list)
        return self._add_derived_features(chunk_df)

    def _select_customer(self, timestamp: datetime.datetime) -> Tuple[int, Customer]:
        """Select a customer for the transaction based on timing patterns."""
        # Enhanced customer selection based on temporal patterns
        eligible_customers = []
        for cust_id, customer in self.customers.items():
            timing_prob = customer.get_transaction_timing_probability(
                timestamp.hour, timestamp.weekday()
            )
            if random.random() < timing_prob * 0.1:  # Base probability adjustment
                eligible_customers.append(cust_id)

        if not eligible_customers:
            # Fallback to any customer
            customer_id = random.randint(1, len(self.customers))
        else:
            customer_id = random.choice(eligible_customers)

        return customer_id, self.customers[customer_id]

    def _select_merchant(self, customer: Customer, timestamp: datetime.datetime) -> Tuple[int, Merchant]:
        """Select a merchant based on customer affinity and business patterns."""
        merchant_candidates = []
        for merch_id, merchant in self.merchants.items():
            affinity = merchant.get_customer_affinity(customer)
            business_hour_mult = merchant.get_business_hour_multiplier(
                timestamp.hour, timestamp.weekday()
            )
            seasonal_mult = merchant.get_seasonal_multiplier(timestamp)

            total_prob = affinity * business_hour_mult * seasonal_mult
            merchant_candidates.append((merch_id, total_prob))

        # Select merchant based on weighted probabilities
        if merchant_candidates:
            merchant_ids, weights = zip(*merchant_candidates)
            try:
                merchant_id = random.choices(merchant_ids, weights=weights)[0]
            except ValueError:
                merchant_id = random.choice(merchant_ids)
        else:
            merchant_id = random.randint(1, len(self.merchants))

        return merchant_id, self.merchants[merchant_id]

    def _generate_transaction(self, transaction_id: int, customer: Customer,
                            merchant: Merchant, timestamp: datetime.datetime,
                            customer_purchase_counts: Dict[int, int]) -> Dict[str, Any]:
        """Generate a complete transaction with fraud detection."""
        # Calculate days since last purchase
        days_since_last_purchase = customer.get_time_since_last_purchase(timestamp)

        # Generate transaction amount
        amount_generator = merchant.get_transaction_amount_distribution()
        amount = amount_generator()

        # Enhanced fraud detection with campaign integration
        is_fraud, is_friendly_fraud, active_campaign = self._determine_fraud_status(
            customer, merchant, timestamp, amount, days_since_last_purchase,
            customer_purchase_counts[customer.customer_id]
        )

        # Device, IP, shipping determination
        device, ip_address, shipping_address = self._generate_transaction_context(
            customer, is_fraud or is_friendly_fraud
        )

        # Update customer state
        customer_purchase_counts[customer.customer_id] += 1

        # Build transaction data
        return self._build_transaction_data(
            transaction_id, customer, merchant, timestamp, amount,
            device, ip_address, shipping_address,
            days_since_last_purchase, customer_purchase_counts[customer.customer_id],
            is_fraud, is_friendly_fraud, active_campaign
        )

    def _determine_fraud_status(self, customer: Customer, merchant: Merchant,
                              timestamp: datetime.datetime, amount: float,
                              days_since_last_purchase: int,
                              customer_purchase_count: int) -> Tuple[bool, bool, Any]:
        """Determine if transaction is fraudulent."""
        # Check if any fraud campaign affects this transaction
        active_campaigns = [c for c in self.fraud_campaigns if c.is_active(timestamp)]

        # Base fraud probability calculation
        base_fraud_prob = (customer.risk_score * 0.3 +
                          merchant.risk_level * 0.2 +
                          (1 - merchant.security_level) * 0.1)

        # Campaign influence on fraud probability
        campaign_fraud_prob = base_fraud_prob
        active_fraud_campaign = None

        for campaign in active_campaigns:
            campaign_prob = campaign.get_fraud_probability(customer, merchant, base_fraud_prob)
            if campaign_prob > campaign_fraud_prob:
                campaign_fraud_prob = campaign_prob
                active_fraud_campaign = campaign

        # Additional risk factors
        time_risk = min(0.5, days_since_last_purchase / 180)

        amount_risk = 0
        if customer_purchase_count > 0:
            typical_amount = customer.avg_purchase_amount
            amount_deviation = abs(amount - typical_amount) / max(1, typical_amount)
            amount_risk = min(0.2, amount_deviation * 0.5)

        # Final fraud probability
        fraud_prob = min(0.95, campaign_fraud_prob + time_risk + amount_risk)
        is_fraud = bernoulli.rvs(fraud_prob)

        # Friendly fraud check (only if not regular fraud)
        is_friendly_fraud = False
        if not is_fraud:
            preliminary_transaction_data = {'amount': amount, 'timestamp': timestamp}
            friendly_fraud_prob = self.friendly_fraud_generator.generate_friendly_fraud(
                customer, preliminary_transaction_data, merchant
            )
            is_friendly_fraud = bernoulli.rvs(friendly_fraud_prob)

        return bool(is_fraud), bool(is_friendly_fraud), active_fraud_campaign

    def _generate_transaction_context(self, customer: Customer,
                                    is_fraudulent: bool) -> Tuple[Dict[str, str], str, Dict[str, str]]:
        """Generate device, IP, and shipping context for transaction."""
        use_new_device = random.random() < 0.1
        use_new_ip = random.random() < 0.1
        shipping_address = customer.shipping_address

        if use_new_device:
            device = customer.get_new_device()
        else:
            device = customer.get_common_device()

        if use_new_ip:
            ip_address = customer.get_new_ip()
        else:
            ip_address = customer.get_common_ip()

        # Fraud pattern modifications
        if is_fraudulent:
            if random.random() < 0.8:
                device = customer.get_new_device()
            if random.random() < 0.8:
                ip_address = customer.get_new_ip()
            if random.random() < 0.7:
                from faker import Faker
                fake = Faker()
                shipping_address = {
                    'street': fake.street_address(),
                    'city': fake.city(),
                    'state': fake.state_abbr(),
                    'zip': fake.zipcode(),
                    'country': 'US'
                }

        return device, ip_address, shipping_address

    def _build_transaction_data(self, transaction_id: int, customer: Customer,
                              merchant: Merchant, timestamp: datetime.datetime,
                              amount: float, device: Dict[str, str], ip_address: str,
                              shipping_address: Dict[str, str], days_since_last_purchase: int,
                              customer_purchase_count: int, is_fraud: bool,
                              is_friendly_fraud: bool, active_campaign: Any) -> Dict[str, Any]:
        """Build the complete transaction data dictionary."""
        return {
            'transaction_id': transaction_id,
            'customer_id': customer.customer_id,
            'merchant_id': merchant.merchant_id,
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'currency': 'USD',

            # Enhanced customer data
            'customer_name': customer.name,
            'customer_email': customer.email,
            'customer_phone': customer.phone,
            'customer_age': customer.age,
            'customer_persona': customer.persona_type,
            'customer_metro_area': customer.metro_area,
            'days_since_signup': (timestamp.date() - customer.signup_date).days,

            # Device and IP
            'device_id': device['device_id'],
            'os': device['os'],
            'browser': device['browser'],
            'user_agent': device['user_agent'],
            'ip_address': ip_address,

            # Enhanced addresses
            'billing_street': customer.billing_address['street'],
            'billing_city': customer.billing_address['city'],
            'billing_state': customer.billing_address['state'],
            'billing_zip': customer.billing_address['zip'],
            'billing_country': customer.billing_address['country'],
            'billing_metro_area': customer.billing_address.get('metro_area', ''),

            'shipping_street': shipping_address['street'],
            'shipping_city': shipping_address['city'],
            'shipping_state': shipping_address['state'],
            'shipping_zip': shipping_address['zip'],
            'shipping_country': shipping_address['country'],

            # Payment info
            'cc_bin': customer.cc_bin,
            'cc_last4': customer.cc_number[-4:],
            'cc_expiry': customer.cc_expiry,

            # Enhanced merchant info
            'merchant_name': merchant.name,
            'merchant_category': merchant.category,
            'merchant_years_in_business': round(merchant.years_in_business, 1),
            'merchant_geographic_scope': merchant.geographic_scope,

            # Transaction metadata
            'days_since_last_purchase': days_since_last_purchase,
            'customer_purchase_count': customer_purchase_count,

            # Enhanced binary features
            'address_match': int(customer.billing_address == shipping_address),
            'is_new_device': int(random.random() < 0.1),  # This should be passed from context
            'is_new_ip': int(random.random() < 0.1),      # This should be passed from context
            'is_international': 0,
            'is_business_hours': int(merchant.get_business_hour_multiplier(timestamp.hour, timestamp.weekday()) > 0.5),

            # Campaign information
            'active_fraud_campaign': active_campaign.campaign_id if active_campaign else None,
            'fraud_campaign_type': active_campaign.fraud_type if active_campaign else None,

            # Fraud labels
            'is_fraud': int(is_fraud),
            'is_friendly_fraud': int(is_friendly_fraud)
        }

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived temporal and categorical features."""
        # Enhanced temporal features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
        df['is_holiday_season'] = df['month'].apply(lambda x: 1 if x in [11, 12] else 0)

        return df

    def _cleanup_output_directory(self, output_dir: str) -> None:
        """Remove existing parquet files from output directory."""
        if os.path.exists(output_dir):
            for file_name in os.listdir(output_dir):
                if file_name.endswith('.parquet'):
                    os.remove(os.path.join(output_dir, file_name))

    def _display_statistics(self, output_dir: str) -> None:
        """Display dataset statistics using Dask."""
        try:
            print("Reading data from chunks for statistics using Dask...")
            ddf = dd.read_parquet(output_dir)

            # Display statistics using Dask
            total_transactions = len(ddf)
            print(f"Total transactions: {total_transactions}")
            fraud_rate = ddf['is_fraud'].mean().compute() * 100
            print(f"Fraud rate: {fraud_rate:.2f}%")

            # Display sample of the data
            print("\nSample data (from Dask):")
            print(ddf.head())

            # Show feature correlations with fraud
            print("\nFeature correlations with fraud (using Dask for a subset of columns):")
            numeric_cols_candidates = [
                'amount', 'customer_age', 'days_since_signup', 'days_since_last_purchase',
                'customer_purchase_count', 'address_match', 'is_new_device', 'is_new_ip',
                'day_of_week', 'hour_of_day', 'is_weekend', 'is_night', 'is_fraud', 'is_friendly_fraud'
            ]

            # Filter for columns actually present in the ddf
            numeric_cols_present = [col for col in numeric_cols_candidates if col in ddf.columns]

            if 'is_fraud' in numeric_cols_present:
                correlations_dask = ddf[numeric_cols_present].corr()['is_fraud'].compute().sort_values(ascending=False)
                print(correlations_dask.head(15))
            else:
                print("'is_fraud' column not found in the generated data for correlation calculation.")

        except Exception as e:
            print(f"Could not read data from directory {output_dir} for statistics using Dask: {e}")
            print("The dataset was likely generated in chunks, but statistics calculation failed.")
