import pandas as pd
import numpy as np
import random
import datetime
import hashlib
import uuid
from faker import Faker
from collections import defaultdict
import ipaddress
import string
from scipy.stats import lognorm, gamma, poisson, bernoulli, pareto, weibull_min, multivariate_normal
import fire
import yaml
from multiprocessing import Pool
import dask.dataframe as dd
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Initialize faker for generating realistic data
fake = Faker()
Faker.seed(42)

# Global constants
NUM_CUSTOMERS = 5000
NUM_MERCHANTS = 100
NUM_TRANSACTIONS = 200000
START_DATE = datetime.datetime(2023, 1, 1)
END_DATE = datetime.datetime(2023, 12, 31)

# Customer demographics - will be used to generate consistent data
class Customer:
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.name = fake.name()
        self.email = self._generate_email()
        self.phone = fake.phone_number()
        self.age = random.randint(18, 80)
        self.signup_date = fake.date_between_dates(
            date_start=START_DATE - datetime.timedelta(days=365*5),
            date_end=END_DATE
        )

        # Address generation
        self.billing_address = {
            'street': fake.street_address(),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip': fake.zipcode(),
            'country': 'US'
        }

        # 95% of customers have same billing and shipping address
        if random.random() < 0.95:
            self.shipping_address = self.billing_address.copy()
        else:
            self.shipping_address = {
                'street': fake.street_address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zip': fake.zipcode(),
                'country': 'US'
            }

        # Credit card info
        cc_prefixes = ['4', '51', '52', '53', '54', '55', '34', '37', '6011', '65']
        prefix = random.choice(cc_prefixes)
        self.cc_bin = prefix + ''.join(random.choices('0123456789', k=6-len(prefix) if len(prefix) < 6 else 0))
        self.cc_number = self.cc_bin + ''.join(random.choices('0123456789', k=10))
        self.cc_expiry = fake.credit_card_expire()

        # Device and behavior characteristics
        self.usual_devices = random.randint(1, 3)
        self.usual_ips = random.randint(1, 5)
        self.devices = self._generate_devices(self.usual_devices)
        self.ips = self._generate_ips(self.usual_ips)

        # Purchase behavior
        self.avg_purchase_amount = random.uniform(30, 500)
        self.purchase_frequency = random.uniform(0.1, 10)  # Purchases per month
        self.last_purchase_date = None
        self.purchase_history = []

        # Risk factors
        self.risk_score = random.random()  # Base risk score between 0 and 1

    def _generate_email(self):
        # 80% use name-based email, 20% use random string
        if random.random() < 0.8:
            name_parts = self.name.lower().replace(' ', '.').split('.')
            if len(name_parts) > 1:
                email = f"{name_parts[0]}.{name_parts[-1]}"
            else:
                email = name_parts[0]

            if random.random() < 0.3:
                email += str(random.randint(1, 99))

            domain = random.choice(['gmail.com', 'yahoo.com', 'hotmail.com',
                                   'outlook.com', 'icloud.com', 'aol.com'])
            return f"{email}@{domain}"
        else:
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 10)))
            domain = random.choice(['gmail.com', 'yahoo.com', 'hotmail.com',
                                   'outlook.com', 'icloud.com', 'aol.com'])
            return f"{random_str}@{domain}"

    def _generate_devices(self, num_devices):
        devices = []
        os_types = ['iOS', 'Android', 'Windows', 'macOS', 'Linux']
        browser_types = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera']

        for _ in range(num_devices):
            os = random.choice(os_types)
            browser = random.choice(browser_types)
            device_id = str(uuid.uuid4())
            devices.append({
                'device_id': device_id,
                'os': os,
                'browser': browser,
                'user_agent': f"{browser}/{random.randint(50, 100)} ({os}; {random.choice(['Mobile', 'Desktop'])})"
            })
        return devices

    def _generate_ips(self, num_ips):
        ips = []
        for _ in range(num_ips):
            # Generate IPs that look realistic
            ip = ipaddress.IPv4Address(random.randint(0, 2**32 - 1))
            ips.append(str(ip))
        return ips

    def get_common_device(self):
        if self.devices:
            return random.choice(self.devices)
        return None

    def get_common_ip(self):
        if self.ips:
            return random.choice(self.ips)
        return None

    def get_new_device(self):
        os_types = ['iOS', 'Android', 'Windows', 'macOS', 'Linux']
        browser_types = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera']

        os = random.choice(os_types)
        browser = random.choice(browser_types)
        device_id = str(uuid.uuid4())
        return {
            'device_id': device_id,
            'os': os,
            'browser': browser,
            'user_agent': f"{browser}/{random.randint(50, 100)} ({os}; {random.choice(['Mobile', 'Desktop'])})"
        }

    def get_new_ip(self):
        ip = ipaddress.IPv4Address(random.randint(0, 2**32 - 1))
        return str(ip)

    def update_last_purchase(self, date):
        self.last_purchase_date = date

    def get_time_since_last_purchase(self, current_date):
        if self.last_purchase_date is None:
            return 365  # Arbitrary large number for first purchase
        delta = current_date - self.last_purchase_date
        return delta.days

# Merchant profiles with different risk levels
class Merchant:
    def __init__(self, merchant_id):
        self.merchant_id = merchant_id
        self.name = fake.company()
        self.category = random.choice([
            'Electronics', 'Clothing', 'Food', 'Home Goods', 'Health & Beauty',
            'Sports & Outdoors', 'Books & Media', 'Toys & Games', 'Jewelry',
            'Digital Products', 'Services', 'Travel'
        ])

        # Merchant profile characteristics
        self.avg_transaction_amount = random.uniform(20, 1000)
        self.transaction_volume = random.uniform(10, 10000)  # Monthly transactions

        # Risk factors - inherent risk in this merchant category
        self.risk_level = random.betavariate(2, 5)  # Higher risk level = more fraudulent transactions

        # How much the merchant enforces security measures
        self.security_level = random.betavariate(5, 2)  # Higher = better security

        # Typical transaction patterns
        self.min_transaction = max(1, self.avg_transaction_amount * 0.1)
        self.max_transaction = self.avg_transaction_amount * random.uniform(3, 10)

    def get_transaction_amount_distribution(self):
        # Returns a function that generates transaction amounts for this merchant
        mean = np.log(self.avg_transaction_amount)
        sigma = random.uniform(0.3, 0.8)  # Controls the spread

        def transaction_amount_generator():
            return min(self.max_transaction, max(self.min_transaction, lognorm.rvs(s=sigma, scale=np.exp(mean))))

        return transaction_amount_generator

class FriendlyFraudGenerator:
    """Generate friendly fraud patterns - legitimate customers disputing valid charges"""

    def __init__(self):
        self.friendly_fraud_triggers = {
            'buyer_remorse': 0.3,      # Customer regrets purchase
            'family_dispute': 0.2,     # Family member made purchase
            'subscription_forgotten': 0.25,  # Forgot about recurring charge
            'delivery_issues': 0.15,   # Package not received/damaged
            'merchant_dispute': 0.1    # Dissatisfied with service
        }

    def generate_friendly_fraud(self, customer, transaction, merchant):
        """Determine if transaction becomes friendly fraud"""
        base_prob = 0.02  # 2% base rate

        # Buyer's remorse factors
        if transaction['amount'] > customer.avg_purchase_amount * 3:
            base_prob *= 2.5

        # Merchant category risk
        high_dispute_categories = ['Digital Products', 'Services', 'Travel']
        if merchant.category in high_dispute_categories:
            base_prob *= 1.8

        # Customer tenure (longer customers more likely to dispute)
        if (transaction['timestamp'].date() - customer.signup_date).days > 365:
            base_prob *= 1.3

        # Subscription/recurring billing
        if hasattr(transaction, 'is_recurring') and transaction.is_recurring:
            base_prob *= 2.0

        return min(0.25, base_prob)  # Cap at 25%

class RealisticDistributions:
    """Use real-world statistical distributions"""
    def __init__(self):
        # Transaction amounts follow Pareto distribution (80/20 rule)
        self.amount_dist = pareto(b=1.16)  # Based on real e-commerce data

        # Time between purchases follows Weibull distribution
        self.purchase_interval_dist = weibull_min(c=1.5, scale=30)

class CorrelatedFeatureGenerator:
    """Generate correlated features using multivariate distributions"""
    def __init__(self):
        # Define correlation matrix for related features
        self.correlation_matrix = np.array([
            [1.0, 0.7, 0.3],    # age, income, avg_purchase_amount
            [0.7, 1.0, 0.5],    # income, age, avg_purchase_amount
            [0.3, 0.5, 1.0]     # avg_purchase_amount, income, age
        ])

    def generate_correlated_features(self, num_samples):
        mean = [40, 50000, 100]  # Example means for age, income, avg_purchase_amount
        return multivariate_normal.rvs(mean=mean, cov=self.correlation_matrix, size=num_samples)

# Function to generate the dataset
def generate_ecommerce_dataset(num_customers=NUM_CUSTOMERS, num_merchants=NUM_MERCHANTS,
                             num_transactions=NUM_TRANSACTIONS,
                             start_date=START_DATE, end_date=END_DATE,
                             chunk_size=5000,
                             output_dir="output"): # Renamed and now represents directory
    print("Generating customers...")
    customers = {customer_id: Customer(customer_id) for customer_id in range(1, num_customers + 1)}

    print("Generating merchants...")
    merchants = {merchant_id: Merchant(merchant_id) for merchant_id in range(1, num_merchants + 1)}

    print("Generating transactions in chunks...")

    # Customer purchase history tracking
    customer_purchase_counts = defaultdict(int)
    customer_last_amounts = defaultdict(float)

    # Generate all transaction timestamps first, then process in chunks
    time_range = (end_date - start_date).total_seconds()
    all_timestamps = sorted([
        start_date + datetime.timedelta(seconds=random.uniform(0, time_range))
        for _ in range(num_transactions)
    ])

    friendly_fraud_generator = FriendlyFraudGenerator()
    realistic_distributions = RealisticDistributions() # Not used in generate_ecommerce_dataset currently, but keeping for context
    correlated_feature_generator = CorrelatedFeatureGenerator() # Not used directly in transaction loop

    # Generate correlated features for customers once
    correlated_features = correlated_feature_generator.generate_correlated_features(num_customers)
    # Update customer objects with these features (this assumes correlated_features order matches customer_id order)
    for customer_id, (age, income, avg_purchase_amount) in zip(range(1, num_customers + 1), correlated_features):
        customers[customer_id].age = age
        customers[customer_id].avg_purchase_amount = avg_purchase_amount
        # If using income, add it to customer object as well
        # customers[customer_id].income = income


    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Remove any existing parquet files in the output directory from previous runs
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file_name))

    for i in range(0, num_transactions, chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, num_transactions)
        current_timestamps = all_timestamps[chunk_start:chunk_end]

        chunk_transactions_list = [] # List to hold transactions for the current chunk

        print(f"Generating transactions for chunk {i // chunk_size + 1}/{(num_transactions + chunk_size - 1) // chunk_size} ({chunk_start+1}-{chunk_end})")

        # Process timestamps in the current chunk
        for idx, timestamp in enumerate(current_timestamps):
            # Calculate the transaction_id based on chunk position
            transaction_id = chunk_start + idx + 1 # transaction_id is 1-based index

            # Randomly pick a customer (weighted towards more active customers if needed, but simple random for now)
            customer_id = random.randint(1, num_customers)
            customer = customers[customer_id]

            # Merchants with higher transaction volumes get more transactions
            merchant_id = random.choices(
                list(merchants.keys()),
                weights=[m.transaction_volume for m in merchants.values()],
                k=1
            )[0]
            merchant = merchants[merchant_id]

            # Calculate days since last purchase
            days_since_last_purchase = customer.get_time_since_last_purchase(timestamp)

            # Determine initial device, IP, shipping address, and amount (non-fraud case)
            # Use a seed based on transaction_id or timestamp to ensure reproducibility within chunking
            # random.seed(hash(str(transaction_id) + str(timestamp))) # Optional: add more fine-grained seed control
            use_new_device = random.random() < 0.1  # Occasionally uses new device
            use_new_ip = random.random() < 0.1      # Occasionally uses new IP
            shipping_address = customer.shipping_address # Default to customer's shipping address

            if use_new_device:
                device = customer.get_new_device()
            else:
                device = customer.get_common_device()

            if use_new_ip:
                ip_address = customer.get_new_ip()
            else:
                ip_address = customer.get_common_ip()

            amount_generator = merchant.get_transaction_amount_distribution()
            amount = amount_generator()

            # Calculate fraud probability
            # Base probability from combined risk factors
            base_fraud_prob = (customer.risk_score * 0.3 +  # Customer risk contributes 30%
                               merchant.risk_level * 0.2 +   # Merchant risk contributes 20%
                               (1 - merchant.security_level) * 0.1)  # Security measures reduce fraud

            # Additional risk factors:
            # 1. Long time since last purchase increases fraud risk
            time_risk = min(0.5, days_since_last_purchase / 180)  # Maxes at 6 months

            # 2. Unusual transaction amount
            customer_purchase_count = customer_purchase_counts[customer_id]

            # Calculate amount risk based on deviation from customer's average
            amount_risk = 0
            # Check if customer_purchase_count is greater than 0 before calculating typical amount
            if customer_purchase_count > 0:
                 # Use the customer's average purchase amount for comparison
                typical_amount = customers[customer_id].avg_purchase_amount # Use the pre-calculated average
                amount_deviation = abs(amount - typical_amount) / max(1, typical_amount)
                amount_risk = min(0.2, amount_deviation * 0.5) # Cap amount risk


            # Store this amount for next time (before potential fraud modification)
            # Note: This updates the _last_ amount seen for the customer, which influences the *next* transaction's amount_risk
            customer_last_amounts[customer_id] = amount
            customer_purchase_counts[customer_id] += 1

            # Final fraud probability
            fraud_prob = min(0.95, base_fraud_prob + time_risk + amount_risk)

            # Decide if this transaction is fraudulent
            is_fraud = bernoulli.rvs(fraud_prob)

            # Generate a preliminary transaction_data for friendly fraud check
            preliminary_transaction_data = {
                'amount': amount,
                'timestamp': timestamp,
                # Include other necessary fields for friendly fraud check if any
            }

            # Determine if this transaction is friendly fraud (only if not regular fraud)
            is_friendly_fraud = False
            if not is_fraud:
                friendly_fraud_prob = friendly_fraud_generator.generate_friendly_fraud(customer, preliminary_transaction_data, merchant)
                is_friendly_fraud = bernoulli.rvs(friendly_fraud_prob)

            # For fraud transactions, modify the pattern (device, IP, shipping, amount etc.)
            if is_fraud or is_friendly_fraud:
                # 80% of fraud uses a new device and IP
                use_new_device_fraud = random.random() < 0.8
                use_new_ip_fraud = random.random() < 0.8

                if use_new_device_fraud:
                    device = customer.get_new_device()
                # else: device remains the initially determined one

                if use_new_ip_fraud:
                    ip_address = customer.get_new_ip()
                # else: ip_address remains the initially determined one

                # Fraud often has mismatched billing and shipping
                if random.random() < 0.7:
                    shipping_address = {
                        'street': fake.street_address(),
                        'city': fake.city(),
                        'state': fake.state_abbr(),
                        'zip': fake.zipcode(),
                        'country': 'US'
                    }
                # else: shipping_address remains the initially determined one

                # Often uses high value merchants (re-select merchant if fraud)
                if random.random() < 0.6:
                    merchant_id = random.choices(
                        list(merchants.keys()),
                        weights=[max(50, m.avg_transaction_amount) for m in merchants.values()], # Weight by average amount for higher value merchants
                        k=1
                    )[0]
                    merchant = merchants[merchant_id]
                    amount_generator = merchant.get_transaction_amount_distribution()

                    # Higher amount than normal for this merchant (for fraud)
                    amount = amount_generator() * random.uniform(1.0, 2.0)
                # else: amount remains the initially determined one

            # Generate final transaction data using potentially modified variables
            transaction_data = {
                'transaction_id': transaction_id, # Use the global transaction ID
                'customer_id': customer_id,
                'merchant_id': merchant_id,
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'currency': 'USD',

                # Customer data
                'customer_name': customer.name,
                'customer_email': customer.email,
                'customer_phone': customer.phone,
                'customer_age': customer.age,
                'days_since_signup': (timestamp.date() - customer.signup_date).days,

                # Device and IP
                'device_id': device['device_id'],
                'os': device['os'],
                'browser': device['browser'],
                'user_agent': device['user_agent'],
                'ip_address': ip_address,

                # Addresses
                'billing_street': customer.billing_address['street'],
                'billing_city': customer.billing_address['city'],
                'billing_state': customer.billing_address['state'],
                'billing_zip': customer.billing_address['zip'],
                'billing_country': customer.billing_address['country'],

                'shipping_street': shipping_address['street'],
                'shipping_city': shipping_address['city'],
                'shipping_state': shipping_address['state'],
                'shipping_zip': shipping_address['zip'],
                'shipping_country': shipping_address['country'],

                # Payment info
                'cc_bin': customer.cc_bin,
                'cc_last4': customer.cc_number[-4:],
                'cc_expiry': customer.cc_expiry,

                # Merchant info
                'merchant_name': merchant.name,
                'merchant_category': merchant.category,

                # Transaction metadata
                'days_since_last_purchase': days_since_last_purchase,
                'customer_purchase_count': customer_purchase_counts[customer_id], # Use the updated count for the current transaction

                # Binary features
                'address_match': int(customer.billing_address == shipping_address),
                'is_new_device': int(use_new_device),
                'is_new_ip': int(use_new_ip),
                'is_international': 0,  # Simplified for this example

                # The fraud label (ground truth)
                'is_fraud': int(is_fraud),
                'is_friendly_fraud': int(is_friendly_fraud)
            }

            # Update customer's last purchase date AFTER generating the transaction data
            customer.update_last_purchase(timestamp)
            chunk_transactions_list.append(transaction_data)


        # Convert chunk to DataFrame and calculate chunk-specific features
        chunk_df = pd.DataFrame(chunk_transactions_list)

        # Add derived features that would be useful for fraud detection (can be done per chunk)
        chunk_df['day_of_week'] = chunk_df['timestamp'].dt.dayofweek
        chunk_df['hour_of_day'] = chunk_df['timestamp'].dt.hour
        chunk_df['is_weekend'] = chunk_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        chunk_df['is_night'] = chunk_df['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)

        # Velocity features like transactions per minute/hour/day might be inaccurate per chunk.
        # For simplicity, I will skip calculating these across chunk boundaries for now.
        # The existing 'days_since_last_purchase' and 'customer_purchase_count' are sufficient per transaction.

        # Generate the fraud risk score (our novel label) - can be done per chunk
        def calculate_risk_score_chunk(row, customers_dict, merchants_dict):
            # Start with base customer and merchant risk
            customer = customers_dict[row['customer_id']]
            merchant = merchants_dict[row['merchant_id']]

            # Base risk components
            base_risk = customer.risk_score * 0.2 + merchant.risk_level * 0.1

            # Transaction specific risks
            # New device/IP risk
            device_ip_risk = (row['is_new_device'] * 0.15 + row['is_new_ip'] * 0.1)

            # Address mismatch risk
            address_risk = (1 - row['address_match']) * 0.1

            # Time since last purchase risk - higher time gap increases risk
            time_gap_risk = min(0.15, row['days_since_last_purchase'] / 180)

            # Purchase amount anomaly
            amount_risk = 0
            # Use the customer's average purchase amount for deviation calculation
            typical_amount = customers_dict[row['customer_id']].avg_purchase_amount
            deviation = abs(row['amount'] - typical_amount) / max(1, typical_amount)
            amount_risk = min(0.2, deviation * 0.4)


            # Time of day risk
            time_risk = row['is_night'] * 0.05 + row['is_weekend'] * 0.02

            # Combine risk factors with some randomness
            combined_risk = base_risk + device_ip_risk + address_risk + time_gap_risk + amount_risk + time_risk

            # Add some randomness to make the relationship less obvious
            # Use a seeded random number for reproducibility within the chunk
            # This random state should ideally be tied to the transaction or customer+timestamp
            # For simplicity here, a general random state is used per chunk
            # Ensure seed is within valid range for numpy.random.RandomState (0 to 2**32 - 1)
            seed_value = hash(f"{row['transaction_id']}_{row['customer_id']}") & 0xFFFFFFFF # Use bitwise AND to get a 32-bit unsigned int
            chunk_noise_state = np.random.RandomState(seed=seed_value)
            noise = chunk_noise_state.normal(0, 0.05)


            # Ensure risk score is between 0 and 1
            risk_score = max(0, min(1, combined_risk + noise))

            return risk_score

        # Apply the risk score calculation to the chunk
        # Pass customers and merchants dicts to the apply function
        chunk_df['fraud_risk_score'] = chunk_df.apply(calculate_risk_score_chunk, axis=1, args=(customers, merchants))


        # Save chunk to parquet
        # Write each chunk as a separate file in the output directory
        chunk_filename = f"chunk_{i // chunk_size:04d}.parquet"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        chunk_df.to_parquet(chunk_filepath, index=False, engine='pyarrow')

    # The function no longer returns a single large DataFrame
    # It will return the path to the incrementally written file
    return output_dir # Return the output directory path

class FraudDataGenerator:
    """
    Generate synthetic e-commerce transaction data with fraud labels.

    This tool creates a realistic dataset of e-commerce transactions
    with both legitimate and fraudulent patterns for use in fraud detection
    model development and testing.
    """

    def generate(self, config_path='config.yaml', output_file="ecommerce_fraud_dataset.parquet", chunk_size=100): # Added chunk_size arg
        """
        Generate synthetic e-commerce transaction data and save to a parquet file.

        Args:
            config_path (str): Path to the configuration YAML file
            output_file (str): Directory name for the output parquet files (will be created if it doesn't exist)
            chunk_size (int): Number of transactions per chunk (default 100)

        Returns:
            str: Path to the directory containing the generated parquet files
        """
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Generate the dataset in chunks, writing directly to the output file
        # Pass chunk_size to the generation function
        generated_output_dir = generate_ecommerce_dataset(
            num_customers=config['customer']['num_customers'],
            num_merchants=config['merchant']['num_merchants'],
            num_transactions=config['fraud']['num_transactions'],
            start_date=START_DATE, # Pass start_date and end_date explicitly
            end_date=END_DATE,
            chunk_size=chunk_size,
            output_dir=output_file # Pass the output directory name
        )

        print(f"Dataset generated in chunks in directory: {generated_output_dir}")

        # For statistics and sample display, read from the directory using Dask
        # Dask can read partitioned parquet files without loading everything into memory
        try:
            print("Reading data from chunks for statistics using Dask...")
            ddf = dd.read_parquet(generated_output_dir)

            # Display statistics using Dask
            total_transactions = len(ddf)
            print(f"Total transactions: {total_transactions}") # Remove .compute() since len() already returns a scalar
            fraud_rate = ddf['is_fraud'].mean().compute() * 100
            print(f"Fraud rate: {fraud_rate:.2f}%")

            # Display sample of the data (Dask head loads a small portion)
            print("\nSample data (from Dask):")
            print(ddf.head())

            # Correlation calculation might still be memory intensive depending on the number of columns.
            # Dask can help, but for simplicity and to avoid potential issues with many columns,
            # we will skip the full correlation matrix and just calculate the correlation with 'is_fraud' using Dask.
            print(f"\nCorrelation between fraud_risk_score and is_fraud (from Dask): {ddf['fraud_risk_score'].corr(ddf['is_fraud']).compute():.4f}")

            # Show feature correlations with fraud (calculate for a subset of columns or use sampling if memory is an issue)
            print("\nFeature correlations with fraud (using Dask for a subset of columns):")
            # Select numeric columns that are likely candidates for correlation analysis and are present in chunks
            # Need to ensure these columns exist in the Dask DataFrame
            numeric_cols_candidates = ['amount', 'customer_age', 'days_since_signup', 'days_since_last_purchase', 'customer_purchase_count', 'address_match', 'is_new_device', 'is_new_ip', 'day_of_week', 'hour_of_day', 'is_weekend', 'is_night', 'fraud_risk_score', 'is_fraud', 'is_friendly_fraud']
            # Filter for columns actually present in the ddf
            numeric_cols_present = [col for col in numeric_cols_candidates if col in ddf.columns]

            if 'is_fraud' in numeric_cols_present:
                 # Calculate correlations with 'is_fraud' for the present numeric columns
                correlations_dask = ddf[numeric_cols_present].corr()['is_fraud'].compute().sort_values(ascending=False)
                print(correlations_dask.head(15)) # Display top 15 correlated features
            else:
                print("'is_fraud' column not found in the generated data for correlation calculation.")

        except Exception as e:
            print(f"Could not read data from directory {generated_output_dir} for statistics using Dask: {e}")
            print("The dataset was likely generated in chunks, but statistics calculation failed.")

        return generated_output_dir # Return the output directory path

# Example configuration file (config.yaml)
config_yaml = '''
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
'''

# Save the example configuration to a file if it doesn't exist
if not os.path.exists('config.yaml'):
    with open('config.yaml', 'w') as file:
        file.write(config_yaml)

class StreamingDataGenerator:
    """Generate data in chunks for memory efficiency"""
    # This class seems redundant now that generate_ecommerce_dataset handles chunking.
    # It can be kept if there's a future plan to use it for a different streaming approach,
    # but the current chunking is implemented in the main generation logic.
    def __init__(self, chunk_size=10000):
        self.chunk_size = chunk_size

    def generate_chunks(self, num_chunks):
        """Yield data chunks instead of loading everything in memory"""
        # This method would need to be updated to call the chunked generation logic
        # or replaced entirely.
        print("StreamingDataGenerator.generate_chunks is not fully implemented for the new chunking logic.")
        print("Please use the FraudDataGenerator.generate method which now supports chunking.")
        # Placeholder for actual chunk generation logic
        # yield pd.DataFrame() # This would yield empty DataFrames

    def _generate_chunk(self, chunk_id):
        # Placeholder for actual chunk generation logic
        return pd.DataFrame()

class ParallelDataGenerator:
    """Use multiprocessing for faster generation"""
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.pool = Pool(num_workers)

    def generate_parallel(self, func, iterable):
        # This could potentially be used to parallelize the chunk generation,
        # but requires careful management of the shared customer/merchant state and file writing.
        # For now, the chunking is sequential.
        print("ParallelDataGenerator.generate_parallel is not integrated with the chunked generation logic.")
        print("The current chunking implementation is sequential.")
        return self.pool.map(func, iterable)

if __name__ == "__main__":
    fire.Fire(FraudDataGenerator) # Now calls the FraudDataGenerator with chunking
