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
from scipy.stats import lognorm, gamma, poisson, bernoulli

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
        self.cc_bin = random.choice([
            '4', '51', '52', '53', '54', '55', '34', '37', '6011', '65'
        ]) + ''.join(random.choices('0123456789', k=6-len(self.cc_bin) if len(self.cc_bin) < 6 else 0))
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

# Function to generate the dataset
def generate_ecommerce_dataset():
    print("Generating customers...")
    customers = {customer_id: Customer(customer_id) for customer_id in range(1, NUM_CUSTOMERS + 1)}
    
    print("Generating merchants...")
    merchants = {merchant_id: Merchant(merchant_id) for merchant_id in range(1, NUM_MERCHANTS + 1)}
    
    print("Generating transactions...")
    transactions = []
    
    # Customer purchase history tracking
    customer_purchase_counts = defaultdict(int)
    customer_last_amounts = defaultdict(float)
    
    # Generate transaction timestamps
    time_range = (END_DATE - START_DATE).total_seconds()
    timestamps = sorted([
        START_DATE + datetime.timedelta(seconds=random.uniform(0, time_range))
        for _ in range(NUM_TRANSACTIONS)
    ])
    
    for transaction_id, timestamp in enumerate(timestamps, 1):
        # Some customers make more purchases than others following a Pareto distribution
        customer_id = random.choices(
            list(customers.keys()),
            weights=[c.purchase_frequency for c in customers.values()],
            k=1
        )[0]
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
        
        # Determine if this will be a fraudulent transaction
        # Base probability from combined risk factors
        base_fraud_prob = (customer.risk_score * 0.3 +  # Customer risk contributes 30%
                           merchant.risk_level * 0.2 +   # Merchant risk contributes 20%
                           (1 - merchant.security_level) * 0.1)  # Security measures reduce fraud
        
        # Additional risk factors:
        # 1. Long time since last purchase increases fraud risk
        time_risk = min(0.5, days_since_last_purchase / 180)  # Maxes at 6 months
        
        # 2. Unusual transaction amount
        customer_purchase_count = customer_purchase_counts[customer_id]
        amount_generator = merchant.get_transaction_amount_distribution()
        amount = amount_generator()
        
        # Calculate amount risk based on deviation from customer's average
        amount_risk = 0
        if customer_purchase_count > 0:
            avg_amount = customer_last_amounts[customer_id]
            amount_deviation = abs(amount - avg_amount) / max(1, avg_amount)
            amount_risk = min(0.3, amount_deviation * 0.5)
        
        # Store this amount for next time
        customer_last_amounts[customer_id] = amount
        customer_purchase_counts[customer_id] += 1
        
        # Final fraud probability
        fraud_prob = min(0.95, base_fraud_prob + time_risk + amount_risk)
        
        # Decide if this transaction is fraudulent
        is_fraud = bernoulli.rvs(fraud_prob)
        
        # For fraud transactions, modify the pattern
        if is_fraud:
            # 80% of fraud uses a new device and IP
            use_new_device = random.random() < 0.8
            use_new_ip = random.random() < 0.8
            
            # Fraud often has mismatched billing and shipping
            if random.random() < 0.7:
                shipping_address = {
                    'street': fake.street_address(),
                    'city': fake.city(),
                    'state': fake.state_abbr(),
                    'zip': fake.zipcode(),
                    'country': 'US'
                }
            else:
                shipping_address = customer.shipping_address
            
            # Often uses high value merchants
            if random.random() < 0.6:
                merchant_id = random.choices(
                    list(merchants.keys()),
                    weights=[max(50, m.avg_transaction_amount) for m in merchants.values()],
                    k=1
                )[0]
                merchant = merchants[merchant_id]
                amount_generator = merchant.get_transaction_amount_distribution()
                
                # Higher amount than normal for this merchant
                amount = amount_generator() * random.uniform(1.0, 2.0)
        else:
            # Non-fraud usually uses known device and IP
            use_new_device = random.random() < 0.1  # Occasionally uses new device
            use_new_ip = random.random() < 0.1      # Occasionally uses new IP
            shipping_address = customer.shipping_address
        
        # Get device and IP
        if use_new_device:
            device = customer.get_new_device()
        else:
            device = customer.get_common_device()
        
        if use_new_ip:
            ip_address = customer.get_new_ip()
        else:
            ip_address = customer.get_common_ip()
        
        # Generate transaction data
        transaction_data = {
            'transaction_id': transaction_id,
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
            'customer_purchase_count': customer_purchase_count,
            
            # Binary features
            'address_match': int(customer.billing_address == shipping_address),
            'is_new_device': int(use_new_device),
            'is_new_ip': int(use_new_ip),
            'is_international': 0,  # Simplified for this example
            
            # The fraud label (ground truth)
            'is_fraud': int(is_fraud)
        }
        
        transactions.append(transaction_data)
        
        # Update customer's last purchase date
        customer.update_last_purchase(timestamp)
        
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Add derived features that would be useful for fraud detection
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
    
    # Calculate velocity features (transactions per unit time)
    df = df.sort_values(['customer_id', 'timestamp'])
    
    # Generate the fraud risk score (our novel label)
    # This will be a continuous value between 0-1 that correlates with but isn't exactly the fraud label
    def calculate_risk_score(row):
        # Start with base customer and merchant risk
        customer = customers[row['customer_id']]
        merchant = merchants[row['merchant_id']]
        
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
        if row['customer_purchase_count'] > 0:
            typical_amount = customer.avg_purchase_amount
            deviation = abs(row['amount'] - typical_amount) / max(1, typical_amount)
            amount_risk = min(0.2, deviation * 0.4)
        
        # Time of day risk
        time_risk = row['is_night'] * 0.05 + row['is_weekend'] * 0.02
        
        # Combine risk factors with some randomness
        combined_risk = base_risk + device_ip_risk + address_risk + time_gap_risk + amount_risk + time_risk
        
        # Add some randomness to make the relationship less obvious
        noise = np.random.normal(0, 0.05)
        
        # Ensure risk score is between 0 and 1
        risk_score = max(0, min(1, combined_risk + noise))
        
        return risk_score
    
    # Apply the risk score calculation
    df['fraud_risk_score'] = df.apply(calculate_risk_score, axis=1)
    
    return df

# Generate the dataset
if __name__ == "__main__":
    df = generate_ecommerce_dataset()
    
    # Display statistics
    print(f"Total transactions: {len(df)}")
    fraud_rate = df['is_fraud'].mean() * 100
    print(f"Fraud rate: {fraud_rate:.2f}%")
    
    # Save to CSV
    df.to_csv('ecommerce_fraud_dataset.csv', index=False)
    
    # Display sample of the data
    print("\nSample data:")
    print(df.head())
    
    # Display correlation between risk score and fraud
    print(f"\nCorrelation between fraud_risk_score and is_fraud: {df['fraud_risk_score'].corr(df['is_fraud']):.4f}")
    
    # Show feature importance
    print("\nFeature correlations with fraud:")
    correlations = df.corr()['is_fraud'].sort_values(ascending=False)
    print(correlations.head(10))
