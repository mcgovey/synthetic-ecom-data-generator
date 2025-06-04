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
from scipy.stats import lognorm, gamma, poisson, bernoulli, pareto, weibull_min, multivariate_normal, beta
import fire
import yaml
from multiprocessing import Pool
import dask.dataframe as dd
import os
import math

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

# Phase 1 Enhancement: Customer Personas and Realistic Distributions
class CustomerPersona:
    """Define realistic customer personas with distinct behavior patterns"""

    PERSONAS = {
        'budget_conscious': {
            'weight': 0.35,
            'avg_amount_range': (15, 75),
            'frequency_range': (1, 4),  # transactions per month
            'preferred_categories': ['Food', 'Home Goods', 'Clothing'],
            'device_preference': ['Android', 'Windows'],
            'payment_stability': 0.9,  # how often they use same payment method
            'geographic_mobility': 0.1,  # how often they travel/move
            'fraud_susceptibility': 0.3
        },
        'average_spender': {
            'weight': 0.40,
            'avg_amount_range': (50, 200),
            'frequency_range': (2, 8),
            'preferred_categories': ['Electronics', 'Clothing', 'Health & Beauty', 'Sports & Outdoors'],
            'device_preference': ['iOS', 'Android', 'Windows', 'macOS'],
            'payment_stability': 0.8,
            'geographic_mobility': 0.2,
            'fraud_susceptibility': 0.4
        },
        'premium_customer': {
            'weight': 0.20,
            'avg_amount_range': (150, 800),
            'frequency_range': (5, 15),
            'preferred_categories': ['Electronics', 'Jewelry', 'Travel', 'Digital Products'],
            'device_preference': ['iOS', 'macOS', 'Windows'],
            'payment_stability': 0.85,
            'geographic_mobility': 0.4,
            'fraud_susceptibility': 0.5
        },
        'high_value': {
            'weight': 0.05,
            'avg_amount_range': (500, 2000),
            'frequency_range': (8, 25),
            'preferred_categories': ['Jewelry', 'Travel', 'Electronics', 'Services'],
            'device_preference': ['iOS', 'macOS'],
            'payment_stability': 0.9,
            'geographic_mobility': 0.6,
            'fraud_susceptibility': 0.7
        }
    }

class GeographicDistribution:
    """Realistic geographic distribution for customers and transactions"""

    # US metro areas with realistic population weights
    METRO_AREAS = {
        'New York-Newark-Jersey City': {'weight': 0.12, 'timezone': 'America/New_York', 'affluence': 0.7},
        'Los Angeles-Long Beach-Anaheim': {'weight': 0.08, 'timezone': 'America/Los_Angeles', 'affluence': 0.6},
        'Chicago-Naperville-Elgin': {'weight': 0.05, 'timezone': 'America/Chicago', 'affluence': 0.6},
        'Dallas-Fort Worth-Arlington': {'weight': 0.04, 'timezone': 'America/Chicago', 'affluence': 0.5},
        'Houston-The Woodlands-Sugar Land': {'weight': 0.04, 'timezone': 'America/Chicago', 'affluence': 0.5},
        'Washington-Arlington-Alexandria': {'weight': 0.04, 'timezone': 'America/New_York', 'affluence': 0.8},
        'Miami-Fort Lauderdale-West Palm Beach': {'weight': 0.03, 'timezone': 'America/New_York', 'affluence': 0.6},
        'Philadelphia-Camden-Wilmington': {'weight': 0.03, 'timezone': 'America/New_York', 'affluence': 0.6},
        'Atlanta-Sandy Springs-Roswell': {'weight': 0.03, 'timezone': 'America/New_York', 'affluence': 0.5},
        'Boston-Cambridge-Newton': {'weight': 0.03, 'timezone': 'America/New_York', 'affluence': 0.8},
        'Other_Metro': {'weight': 0.35, 'timezone': 'America/Chicago', 'affluence': 0.4},
        'Rural': {'weight': 0.20, 'timezone': 'America/Chicago', 'affluence': 0.3}
    }

class FraudCampaign:
    """Model fraud as organized campaigns rather than independent events"""

    def __init__(self, campaign_id, start_date, end_date, fraud_type, target_merchants=None):
        self.campaign_id = campaign_id
        self.start_date = start_date
        self.end_date = end_date
        self.fraud_type = fraud_type
        self.target_merchants = target_merchants or []
        self.active_fraudsters = []
        self.tools_used = self._select_fraud_tools()
        self.intensity = random.betavariate(2, 5)  # Campaign intensity

    def _select_fraud_tools(self):
        """Select fraud tools based on campaign type"""
        tool_sets = {
            'card_testing': ['bot_network', 'proxy_rotation', 'card_generator'],
            'account_takeover': ['credential_stuffing', 'social_engineering', 'sim_swapping'],
            'friendly_fraud': ['dispute_automation', 'chargeback_farming'],
            'bust_out': ['identity_synthesis', 'credit_building', 'coordinated_spending'],
            'refund_fraud': ['return_fraud_automation', 'fake_tracking']
        }
        return tool_sets.get(self.fraud_type, ['basic_fraud'])

    def is_active(self, date):
        return self.start_date <= date <= self.end_date

    def get_fraud_probability(self, customer, merchant, base_prob):
        if not self.is_active(datetime.datetime.now()):
            return base_prob

        # Campaign-specific probability modifications
        multiplier = 1.0

        if merchant.merchant_id in self.target_merchants:
            multiplier *= 3.0

        if self.fraud_type == 'card_testing' and merchant.category in ['Digital Products', 'Services']:
            multiplier *= 2.0

        if 'bot_network' in self.tools_used:
            multiplier *= 1.5

        return min(0.9, base_prob * multiplier * self.intensity)

class TemporalPatterns:
    """Realistic temporal patterns for transactions"""

    @staticmethod
    def get_hourly_weights():
        """Transaction probability by hour of day"""
        # Peak during lunch (12-1) and evening (6-9)
        weights = np.array([
            0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8,  # 0-7 AM
            1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.3,  # 8-15 (8 AM - 3 PM)
            1.5, 1.8, 2.2, 2.5, 2.2, 1.8, 1.2, 0.8   # 16-23 (4-11 PM)
        ])
        return weights / weights.sum()

    @staticmethod
    def get_daily_weights():
        """Transaction probability by day of week (Mon=0, Sun=6)"""
        # Higher on weekends and Friday
        weights = np.array([0.12, 0.12, 0.13, 0.14, 0.16, 0.18, 0.15])
        return weights / weights.sum()

    @staticmethod
    def get_seasonal_multiplier(date):
        """Seasonal transaction multiplier"""
        day_of_year = date.timetuple().tm_yday

        # Holiday seasons (higher activity)
        # Black Friday/Cyber Monday (late November)
        if 320 <= day_of_year <= 335:
            return 1.8
        # Christmas season
        if 340 <= day_of_year <= 365:
            return 1.6
        # Back to school (August)
        if 213 <= day_of_year <= 243:
            return 1.3
        # Summer vacation (June-July)
        if 152 <= day_of_year <= 212:
            return 1.2
        # Post-holiday lull (January)
        if 1 <= day_of_year <= 31:
            return 0.7

        return 1.0

# Enhanced Customer class with personas and realistic behavior
class Customer:
    def __init__(self, customer_id):
        self.customer_id = customer_id

        # Assign persona based on weights
        persona_names = list(CustomerPersona.PERSONAS.keys())
        persona_weights = [CustomerPersona.PERSONAS[p]['weight'] for p in persona_names]
        self.persona_type = random.choices(persona_names, weights=persona_weights)[0]
        self.persona = CustomerPersona.PERSONAS[self.persona_type]

        # Geographic assignment
        metro_names = list(GeographicDistribution.METRO_AREAS.keys())
        metro_weights = [GeographicDistribution.METRO_AREAS[m]['weight'] for m in metro_names]
        self.metro_area = random.choices(metro_names, weights=metro_weights)[0]
        self.metro_info = GeographicDistribution.METRO_AREAS[self.metro_area]

        # Basic demographics influenced by persona and geography
        self.name = fake.name()
        self.email = self._generate_email()
        self.phone = fake.phone_number()

        # Age influenced by persona (premium customers tend to be older)
        age_ranges = {
            'budget_conscious': (18, 35),
            'average_spender': (25, 55),
            'premium_customer': (30, 65),
            'high_value': (35, 70)
        }
        age_min, age_max = age_ranges[self.persona_type]
        self.age = random.randint(age_min, age_max)

        self.signup_date = fake.date_between_dates(
            date_start=START_DATE - datetime.timedelta(days=365*5),
            date_end=END_DATE
        )

        # Geographic-aware address generation
        self.billing_address = self._generate_address()

        # 95% of customers have same billing and shipping address
        if random.random() < 0.95:
            self.shipping_address = self.billing_address.copy()
        else:
            self.shipping_address = self._generate_address()

        # Realistic credit card info based on persona
        self.cc_bin, self.cc_number, self.cc_expiry = self._generate_payment_info()

        # Device and behavior characteristics based on persona
        self.devices = self._generate_devices()
        self.ips = self._generate_ips()

        # Purchase behavior from persona
        amount_min, amount_max = self.persona['avg_amount_range']
        freq_min, freq_max = self.persona['frequency_range']

        # Adjust for geographic affluence
        affluence_multiplier = self.metro_info['affluence']
        self.avg_purchase_amount = random.uniform(amount_min, amount_max) * affluence_multiplier
        self.purchase_frequency = random.uniform(freq_min, freq_max)

        # Transaction timing preferences (work schedule, timezone)
        self.preferred_hours = self._generate_activity_pattern()
        self.timezone = self.metro_info['timezone']

        self.last_purchase_date = None
        self.purchase_history = []
        self.preferred_categories = self.persona['preferred_categories']

        # Risk factors
        self.risk_score = self.persona['fraud_susceptibility'] * random.betavariate(2, 3)

    def _generate_address(self):
        """Generate address consistent with metro area"""
        # Simplified - in production, would use real geographic data
        fake_locale = fake
        if self.metro_area != 'Rural':
            city = self.metro_area.split('-')[0]  # Extract main city name
        else:
            city = fake.city()

        return {
            'street': fake_locale.street_address(),
            'city': city,
            'state': fake_locale.state_abbr(),
            'zip': fake_locale.zipcode(),
            'country': 'US',
            'metro_area': self.metro_area
        }

    def _generate_payment_info(self):
        """Generate realistic payment info based on persona"""
        # Different card types for different personas
        card_preferences = {
            'budget_conscious': ['4', '51', '52'],  # Visa, basic Mastercard
            'average_spender': ['4', '51', '52', '53', '54', '55'],  # Visa, Mastercard
            'premium_customer': ['4', '51', '52', '53', '54', '55', '34', '37'],  # Include Amex
            'high_value': ['34', '37', '4', '51', '52']  # Prefer Amex, premium Visa/MC
        }

        prefixes = card_preferences[self.persona_type]
        prefix = random.choice(prefixes)

        # Generate realistic BIN (Bank Identification Number)
        if prefix in ['34', '37']:  # Amex
            cc_bin = prefix + ''.join(random.choices('0123456789', k=4))
            cc_number = cc_bin + ''.join(random.choices('0123456789', k=9))
        else:  # Visa/Mastercard
            cc_bin = prefix + ''.join(random.choices('0123456789', k=5))
            cc_number = cc_bin + ''.join(random.choices('0123456789', k=10))

        cc_expiry = fake.credit_card_expire()

        return cc_bin, cc_number, cc_expiry

    def _generate_devices(self):
        """Generate devices based on persona preferences"""
        device_count = random.randint(1, 3)
        devices = []

        preferred_os = self.persona['device_preference']

        for _ in range(device_count):
            os = random.choice(preferred_os)

            # Realistic browser distribution by OS
            browser_by_os = {
                'iOS': ['Safari', 'Chrome', 'Firefox'],
                'Android': ['Chrome', 'Samsung Internet', 'Firefox'],
                'Windows': ['Chrome', 'Edge', 'Firefox', 'Opera'],
                'macOS': ['Safari', 'Chrome', 'Firefox'],
                'Linux': ['Firefox', 'Chrome', 'Opera']
            }

            browser = random.choice(browser_by_os.get(os, ['Chrome']))
            device_id = str(uuid.uuid4())

            # More realistic user agent generation
            if os == 'iOS':
                device_type = random.choice(['iPhone', 'iPad'])
                user_agent = f"Mozilla/5.0 ({device_type}; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
            elif os == 'Android':
                user_agent = f"Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36"
            else:
                user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

            devices.append({
                'device_id': device_id,
                'os': os,
                'browser': browser,
                'user_agent': user_agent
            })

        return devices

    def _generate_ips(self):
        """Generate geographic-consistent IP addresses"""
        ip_count = random.randint(1, 4)
        ips = []

        # IP ranges roughly corresponding to geographic areas
        # This is simplified - in production would use real GeoIP data
        metro_ip_ranges = {
            'New York-Newark-Jersey City': [(72, 21), (173, 252), (96, 47)],
            'Los Angeles-Long Beach-Anaheim': [(64, 58), (173, 239), (207, 241)],
            'Chicago-Naperville-Elgin': [(64, 107), (173, 205), (96, 43)],
            # ... other metros would have their ranges
        }

        ip_bases = metro_ip_ranges.get(self.metro_area, [(192, 168), (10, 0), (172, 16)])

        for _ in range(ip_count):
            base_octets = random.choice(ip_bases)
            ip = f"{base_octets[0]}.{base_octets[1]}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            ips.append(ip)

        return ips

    def _generate_activity_pattern(self):
        """Generate customer-specific activity hours based on persona and demographics"""
        if self.age < 30:
            # Younger customers: more evening/night activity
            preferred = list(range(18, 24)) + list(range(0, 2))
        elif self.age > 55:
            # Older customers: more daytime activity
            preferred = list(range(9, 17))
        else:
            # Working age: lunch and evening peaks
            preferred = list(range(12, 14)) + list(range(18, 22))

        return preferred

    def _generate_email(self):
        """Enhanced email generation with realistic patterns"""
        if random.random() < 0.8:
            name_parts = self.name.lower().replace(' ', '.').split('.')
            if len(name_parts) > 1:
                email = f"{name_parts[0]}.{name_parts[-1]}"
            else:
                email = name_parts[0]

            if random.random() < 0.3:
                email += str(random.randint(1, 99))

            # Domain preferences by persona
            domain_preferences = {
                'budget_conscious': ['gmail.com', 'yahoo.com', 'hotmail.com'],
                'average_spender': ['gmail.com', 'yahoo.com', 'outlook.com', 'icloud.com'],
                'premium_customer': ['gmail.com', 'icloud.com', 'outlook.com'],
                'high_value': ['gmail.com', 'icloud.com', 'company.com']
            }

            domains = domain_preferences[self.persona_type]
            domain = random.choice(domains)
            return f"{email}@{domain}"
        else:
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(5, 10)))
            domain = random.choice(['gmail.com', 'yahoo.com', 'hotmail.com'])
            return f"{random_str}@{domain}"

    def get_transaction_timing_probability(self, hour, day_of_week):
        """Get probability of transaction at specific time based on customer pattern"""
        base_prob = 1.0

        # Hour preference
        if hour in self.preferred_hours:
            base_prob *= 2.0
        elif hour in range(2, 6):  # Very early morning
            base_prob *= 0.1

        # Day of week preference (higher on weekends for most)
        if day_of_week >= 5:  # Weekend
            base_prob *= 1.3

        return base_prob

    def get_preferred_categories(self, merchant_categories):
        """Get customer's preferred merchant categories"""
        return [cat for cat in merchant_categories if cat in self.preferred_categories]

    def get_common_device(self):
        if self.devices:
            return random.choice(self.devices)
        return None

    def get_common_ip(self):
        if self.ips:
            return random.choice(self.ips)
        return None

    def get_new_device(self):
        # Use persona preferences for new devices too
        os_types = self.persona['device_preference']
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
        # Generate new IP from same geographic region
        if self.ips:
            # Use similar pattern to existing IPs
            existing_ip = random.choice(self.ips)
            octets = existing_ip.split('.')
            # Keep first two octets, randomize last two
            new_ip = f"{octets[0]}.{octets[1]}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            return new_ip
        else:
            ip = ipaddress.IPv4Address(random.randint(0, 2**32 - 1))
            return str(ip)

    def update_last_purchase(self, date):
        self.last_purchase_date = date

    def get_time_since_last_purchase(self, current_date):
        if self.last_purchase_date is None:
            return 365  # Arbitrary large number for first purchase
        delta = current_date - self.last_purchase_date
        return delta.days

# Merchant profiles with different risk levels and realistic business patterns
class Merchant:
    def __init__(self, merchant_id):
        self.merchant_id = merchant_id
        self.name = fake.company()

        # Enhanced category system with realistic distribution
        category_weights = {
            'Electronics': 0.12,
            'Clothing': 0.15,
            'Food & Restaurants': 0.18,
            'Home Goods': 0.12,
            'Health & Beauty': 0.10,
            'Sports & Outdoors': 0.08,
            'Books & Media': 0.05,
            'Toys & Games': 0.06,
            'Jewelry': 0.03,
            'Digital Products': 0.08,
            'Services': 0.02,
            'Travel': 0.01
        }

        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        self.category = random.choices(categories, weights=weights)[0]

        # Business characteristics based on category
        self._set_category_characteristics()

        # Geographic presence (some merchants are local, others national)
        self.geographic_scope = random.choices(
            ['local', 'regional', 'national', 'international'],
            weights=[0.4, 0.3, 0.25, 0.05]
        )[0]

        # Business maturity affects transaction patterns
        self.years_in_business = random.uniform(0.5, 20)
        self.maturity_factor = min(1.0, self.years_in_business / 10)  # Mature at 10+ years

        # Operating hours and patterns
        self.operating_hours = self._generate_operating_hours()
        self.seasonal_patterns = self._generate_seasonal_patterns()

        # Security and risk characteristics
        self.security_level = random.betavariate(5, 2) * self.maturity_factor  # Mature businesses have better security
        self.fraud_victimization_history = random.betavariate(2, 8)  # Most merchants have low fraud history

        # Customer base characteristics
        self.customer_loyalty = random.betavariate(3, 2)  # How often customers return
        self.average_customer_value = self._calculate_customer_value()

    def _set_category_characteristics(self):
        """Set merchant characteristics based on business category"""
        category_profiles = {
            'Electronics': {
                'avg_amount_range': (50, 800),
                'volume_range': (100, 5000),
                'risk_level': 0.6,  # High value items attract fraud
                'seasonality_strength': 0.8,  # Strong holiday patterns
                'operating_pattern': 'standard_retail'
            },
            'Clothing': {
                'avg_amount_range': (25, 200),
                'volume_range': (200, 8000),
                'risk_level': 0.4,
                'seasonality_strength': 0.9,  # Very seasonal
                'operating_pattern': 'standard_retail'
            },
            'Food & Restaurants': {
                'avg_amount_range': (15, 80),
                'volume_range': (500, 15000),
                'risk_level': 0.2,  # Lower fraud risk
                'seasonality_strength': 0.3,
                'operating_pattern': 'food_service'
            },
            'Digital Products': {
                'avg_amount_range': (5, 100),
                'volume_range': (50, 2000),
                'risk_level': 0.8,  # High fraud risk
                'seasonality_strength': 0.4,
                'operating_pattern': '24_7'
            },
            'Travel': {
                'avg_amount_range': (200, 2000),
                'volume_range': (20, 500),
                'risk_level': 0.7,  # High value, high risk
                'seasonality_strength': 0.9,
                'operating_pattern': 'travel_agency'
            },
            'Jewelry': {
                'avg_amount_range': (100, 2000),
                'volume_range': (10, 200),
                'risk_level': 0.8,  # Very high value, high risk
                'seasonality_strength': 0.95,  # Extremely seasonal
                'operating_pattern': 'luxury_retail'
            }
        }

        # Default profile for categories not explicitly defined
        default_profile = {
            'avg_amount_range': (30, 300),
            'volume_range': (100, 2000),
            'risk_level': 0.4,
            'seasonality_strength': 0.5,
            'operating_pattern': 'standard_retail'
        }

        profile = category_profiles.get(self.category, default_profile)

        # Apply profile characteristics
        amount_min, amount_max = profile['avg_amount_range']
        volume_min, volume_max = profile['volume_range']

        self.avg_transaction_amount = random.uniform(amount_min, amount_max)
        self.transaction_volume = random.uniform(volume_min, volume_max)
        self.risk_level = profile['risk_level'] * random.betavariate(2, 3)
        self.seasonality_strength = profile['seasonality_strength']
        self.operating_pattern = profile['operating_pattern']

        # Transaction bounds
        self.min_transaction = max(1, self.avg_transaction_amount * 0.1)
        self.max_transaction = self.avg_transaction_amount * random.uniform(3, 8)

    def _generate_operating_hours(self):
        """Generate realistic operating hours based on business type"""
        patterns = {
            'standard_retail': {'open': 9, 'close': 21, 'weekend_factor': 0.8},
            'food_service': {'open': 11, 'close': 23, 'weekend_factor': 1.2},
            '24_7': {'open': 0, 'close': 24, 'weekend_factor': 1.0},
            'travel_agency': {'open': 8, 'close': 18, 'weekend_factor': 0.6},
            'luxury_retail': {'open': 10, 'close': 19, 'weekend_factor': 1.1}
        }

        pattern = patterns.get(self.operating_pattern, patterns['standard_retail'])

        return {
            'weekday_open': pattern['open'],
            'weekday_close': pattern['close'],
            'weekend_factor': pattern['weekend_factor'],
            'holiday_factor': random.uniform(0.5, 1.5)
        }

    def _generate_seasonal_patterns(self):
        """Generate seasonal business patterns"""
        base_pattern = {
            'Q1': 0.8,   # Post-holiday slowdown
            'Q2': 1.0,   # Normal
            'Q3': 1.1,   # Summer
            'Q4': 1.4    # Holiday season
        }

        # Adjust based on category
        if self.category in ['Travel', 'Sports & Outdoors']:
            base_pattern['Q3'] = 1.5  # Summer peak
        elif self.category in ['Clothing', 'Jewelry']:
            base_pattern['Q4'] = 1.8  # Strong holiday peak
        elif self.category in ['Food & Restaurants']:
            # More stable across seasons
            base_pattern = {'Q1': 0.95, 'Q2': 1.0, 'Q3': 1.05, 'Q4': 1.1}

        return base_pattern

    def _calculate_customer_value(self):
        """Calculate average customer lifetime value"""
        base_value = self.avg_transaction_amount * self.customer_loyalty * 10
        return base_value * random.uniform(0.8, 1.2)

    def get_business_hour_multiplier(self, hour, day_of_week):
        """Get transaction probability multiplier based on business hours"""
        is_weekend = day_of_week >= 5

        if self.operating_pattern == '24_7':
            return 1.0

        open_hour = self.operating_hours['weekday_open']
        close_hour = self.operating_hours['weekday_close']

        # Check if within operating hours
        if open_hour <= hour < close_hour:
            multiplier = 1.0

            # Peak hours vary by business type
            if self.operating_pattern == 'food_service':
                # Lunch and dinner peaks
                if hour in [12, 13, 18, 19, 20]:
                    multiplier = 1.5
            elif self.operating_pattern == 'standard_retail':
                # Evening shopping peak
                if hour in [17, 18, 19, 20]:
                    multiplier = 1.3

            if is_weekend:
                multiplier *= self.operating_hours['weekend_factor']

            return multiplier
        else:
            # Outside business hours - very low activity
            return 0.1 if self.operating_pattern != '24_7' else 1.0

    def get_seasonal_multiplier(self, date):
        """Get seasonal transaction multiplier"""
        month = date.month
        if month in [1, 2, 3]:
            quarter = 'Q1'
        elif month in [4, 5, 6]:
            quarter = 'Q2'
        elif month in [7, 8, 9]:
            quarter = 'Q3'
        else:
            quarter = 'Q4'

        base_multiplier = self.seasonal_patterns[quarter]

        # Apply seasonal strength
        adjusted_multiplier = 1.0 + (base_multiplier - 1.0) * self.seasonality_strength

        # Special events
        if month == 11 and date.day >= 24:  # Black Friday week
            adjusted_multiplier *= 1.5
        elif month == 12 and date.day >= 20:  # Christmas week
            adjusted_multiplier *= 1.3

        return adjusted_multiplier

    def get_transaction_amount_distribution(self):
        """Returns a function that generates realistic transaction amounts"""
        # Use log-normal distribution with category-specific parameters
        mean = np.log(self.avg_transaction_amount)

        # Variance depends on business type
        if self.category in ['Jewelry', 'Travel', 'Electronics']:
            sigma = 0.8  # Higher variance for luxury/high-value items
        elif self.category in ['Food & Restaurants', 'Digital Products']:
            sigma = 0.4  # Lower variance for everyday purchases
        else:
            sigma = 0.6  # Medium variance

        def transaction_amount_generator():
            amount = lognorm.rvs(s=sigma, scale=np.exp(mean))
            return min(self.max_transaction, max(self.min_transaction, amount))

        return transaction_amount_generator

    def get_customer_affinity(self, customer):
        """Calculate affinity between merchant and customer based on profiles"""
        base_affinity = 0.5

        # Category preference match
        if self.category in customer.preferred_categories:
            base_affinity += 0.3

        # Geographic proximity (simplified)
        if customer.metro_info['affluence'] > 0.6 and self.category in ['Jewelry', 'Travel']:
            base_affinity += 0.2
        elif customer.metro_info['affluence'] < 0.4 and self.category in ['Food & Restaurants']:
            base_affinity += 0.2

        # Customer persona alignment
        if customer.persona_type == 'high_value' and self.category in ['Jewelry', 'Travel', 'Electronics']:
            base_affinity += 0.3
        elif customer.persona_type == 'budget_conscious' and self.category in ['Food & Restaurants', 'Clothing']:
            base_affinity += 0.2

        return min(1.0, base_affinity)

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

# Enhanced fraud generation with campaign integration
def generate_fraud_campaigns(start_date, end_date, num_merchants):
    """Generate realistic fraud campaigns over the time period"""
    campaigns = []

    # Campaign types and their characteristics
    campaign_types = {
        'card_testing': {
            'duration_days': (3, 14),
            'frequency': 0.3,  # How often this type occurs
            'merchant_targeting': 'digital_high_volume'
        },
        'account_takeover': {
            'duration_days': (7, 30),
            'frequency': 0.2,
            'merchant_targeting': 'high_value'
        },
        'friendly_fraud': {
            'duration_days': (1, 7),
            'frequency': 0.25,
            'merchant_targeting': 'any'
        },
        'bust_out': {
            'duration_days': (30, 90),
            'frequency': 0.15,
            'merchant_targeting': 'credit_based'
        },
        'refund_fraud': {
            'duration_days': (5, 21),
            'frequency': 0.1,
            'merchant_targeting': 'return_policy'
        }
    }

    total_days = (end_date - start_date).days
    campaign_id = 1

    for fraud_type, characteristics in campaign_types.items():
        # Determine number of campaigns of this type
        expected_campaigns = int(total_days / 30 * characteristics['frequency'])

        for _ in range(expected_campaigns):
            # Random start date
            campaign_start = start_date + datetime.timedelta(
                days=random.randint(0, total_days - 30)
            )

            # Campaign duration
            duration_min, duration_max = characteristics['duration_days']
            duration = random.randint(duration_min, duration_max)
            campaign_end = campaign_start + datetime.timedelta(days=duration)

            # Target merchants based on campaign type
            target_merchants = []
            if characteristics['merchant_targeting'] == 'digital_high_volume':
                # Target digital and high-volume merchants
                target_merchants = list(range(1, min(10, num_merchants)))
            elif characteristics['merchant_targeting'] == 'high_value':
                # Target merchants with high transaction amounts
                target_merchants = list(range(1, min(5, num_merchants)))

            campaign = FraudCampaign(
                campaign_id=campaign_id,
                start_date=campaign_start,
                end_date=campaign_end,
                fraud_type=fraud_type,
                target_merchants=target_merchants
            )

            campaigns.append(campaign)
            campaign_id += 1

    return campaigns

# Function to generate the dataset with Phase 1 enhancements
def generate_ecommerce_dataset(num_customers=NUM_CUSTOMERS, num_merchants=NUM_MERCHANTS,
                             num_transactions=NUM_TRANSACTIONS,
                             start_date=START_DATE, end_date=END_DATE,
                             chunk_size=5000,
                             output_dir="output", filename_pattern="results_"):
    print("Generating customers with personas and geographic distribution...")
    customers = {customer_id: Customer(customer_id) for customer_id in range(1, num_customers + 1)}

    print("Generating merchants with realistic business patterns...")
    merchants = {merchant_id: Merchant(merchant_id) for merchant_id in range(1, num_merchants + 1)}

    print("Generating fraud campaigns...")
    fraud_campaigns = generate_fraud_campaigns(start_date, end_date, num_merchants)
    print(f"Generated {len(fraud_campaigns)} fraud campaigns")

    print("Generating transactions in chunks with enhanced realism...")

    # Customer purchase history tracking
    customer_purchase_counts = defaultdict(int)
    customer_last_amounts = defaultdict(float)

    # Generate all transaction timestamps with realistic temporal patterns
    all_timestamps = []
    temporal_patterns = TemporalPatterns()

    # Get temporal weights
    hourly_weights = temporal_patterns.get_hourly_weights()
    daily_weights = temporal_patterns.get_daily_weights()

    for _ in range(num_transactions):
        # Generate realistic timestamp
        random_date = start_date + datetime.timedelta(
            seconds=random.uniform(0, (end_date - start_date).total_seconds())
        )

        # Apply seasonal multiplier to decide if transaction should occur
        seasonal_mult = temporal_patterns.get_seasonal_multiplier(random_date)
        if random.random() > seasonal_mult * 0.7:  # Base acceptance rate
            continue

        # Select hour and day based on realistic patterns
        hour = random.choices(range(24), weights=hourly_weights)[0]
        day_of_week = random.choices(range(7), weights=daily_weights)[0]

        # Adjust the random date to have realistic hour
        final_timestamp = random_date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
        all_timestamps.append(final_timestamp)

    # Sort timestamps and trim to exact count needed
    all_timestamps = sorted(all_timestamps)[:num_transactions]

    friendly_fraud_generator = FriendlyFraudGenerator()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Remove any existing parquet files in the output directory from previous runs
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file_name))

    for i in range(0, len(all_timestamps), chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, len(all_timestamps))
        current_timestamps = all_timestamps[chunk_start:chunk_end]

        chunk_transactions_list = []

        print(f"Generating transactions for chunk {i // chunk_size + 1}/{(len(all_timestamps) + chunk_size - 1) // chunk_size} ({chunk_start+1}-{chunk_end})")

        for idx, timestamp in enumerate(current_timestamps):
            transaction_id = chunk_start + idx + 1

            # Enhanced customer selection based on temporal patterns
            # Some customers are more active at certain times
            eligible_customers = []
            for cust_id, customer in customers.items():
                timing_prob = customer.get_transaction_timing_probability(
                    timestamp.hour, timestamp.weekday()
                )
                if random.random() < timing_prob * 0.1:  # Base probability adjustment
                    eligible_customers.append(cust_id)

            if not eligible_customers:
                # Fallback to any customer
                customer_id = random.randint(1, num_customers)
            else:
                customer_id = random.choice(eligible_customers)

            customer = customers[customer_id]

            # Enhanced merchant selection with customer affinity
            merchant_candidates = []
            for merch_id, merchant in merchants.items():
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
                except:
                    merchant_id = random.choice(merchant_ids)
            else:
                merchant_id = random.randint(1, num_merchants)

            merchant = merchants[merchant_id]

            # Calculate days since last purchase
            days_since_last_purchase = customer.get_time_since_last_purchase(timestamp)

            # Enhanced fraud detection with campaign integration
            # Check if any fraud campaign affects this transaction
            active_campaigns = [c for c in fraud_campaigns if c.is_active(timestamp)]

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

            customer_purchase_count = customer_purchase_counts[customer_id]
            amount_risk = 0
            if customer_purchase_count > 0:
                typical_amount = customers[customer_id].avg_purchase_amount
                # Generate amount first to calculate risk
                amount_generator = merchant.get_transaction_amount_distribution()
                amount = amount_generator()
                amount_deviation = abs(amount - typical_amount) / max(1, typical_amount)
                amount_risk = min(0.2, amount_deviation * 0.5)
            else:
                amount_generator = merchant.get_transaction_amount_distribution()
                amount = amount_generator()

            # Final fraud probability
            fraud_prob = min(0.95, campaign_fraud_prob + time_risk + amount_risk)

            # Decide if this transaction is fraudulent
            is_fraud = bernoulli.rvs(fraud_prob)

            # Device, IP, shipping determination
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

            # Friendly fraud check (only if not regular fraud)
            is_friendly_fraud = False
            if not is_fraud:
                preliminary_transaction_data = {
                    'amount': amount,
                    'timestamp': timestamp,
                }
                friendly_fraud_prob = friendly_fraud_generator.generate_friendly_fraud(
                    customer, preliminary_transaction_data, merchant
                )
                is_friendly_fraud = bernoulli.rvs(friendly_fraud_prob)

            # Fraud pattern modifications
            if is_fraud or is_friendly_fraud:
                if random.random() < 0.8:
                    device = customer.get_new_device()
                if random.random() < 0.8:
                    ip_address = customer.get_new_ip()
                if random.random() < 0.7:
                    shipping_address = {
                        'street': fake.street_address(),
                        'city': fake.city(),
                        'state': fake.state_abbr(),
                        'zip': fake.zipcode(),
                        'country': 'US'
                    }

            # Store transaction data
            customer_last_amounts[customer_id] = amount
            customer_purchase_counts[customer_id] += 1

            # Generate final transaction data
            transaction_data = {
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'merchant_id': merchant_id,
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
                'customer_purchase_count': customer_purchase_counts[customer_id],

                # Enhanced binary features
                'address_match': int(customer.billing_address == shipping_address),
                'is_new_device': int(use_new_device),
                'is_new_ip': int(use_new_ip),
                'is_international': 0,
                'is_business_hours': int(merchant.get_business_hour_multiplier(timestamp.hour, timestamp.weekday()) > 0.5),

                # Campaign information
                'active_fraud_campaign': active_fraud_campaign.campaign_id if active_fraud_campaign else None,
                'fraud_campaign_type': active_fraud_campaign.fraud_type if active_fraud_campaign else None,

                # Fraud labels
                'is_fraud': int(is_fraud),
                'is_friendly_fraud': int(is_friendly_fraud)
            }

            customer.update_last_purchase(timestamp)
            chunk_transactions_list.append(transaction_data)

        # Convert chunk to DataFrame and add derived features
        chunk_df = pd.DataFrame(chunk_transactions_list)

        # Enhanced temporal features
        chunk_df['day_of_week'] = chunk_df['timestamp'].dt.dayofweek
        chunk_df['hour_of_day'] = chunk_df['timestamp'].dt.hour
        chunk_df['month'] = chunk_df['timestamp'].dt.month
        chunk_df['quarter'] = chunk_df['timestamp'].dt.quarter
        chunk_df['is_weekend'] = chunk_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        chunk_df['is_night'] = chunk_df['hour_of_day'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
        chunk_df['is_holiday_season'] = chunk_df['month'].apply(lambda x: 1 if x in [11, 12] else 0)

        # Save chunk to parquet
        chunk_filename = f"{filename_pattern}{i // chunk_size + 1}.parquet"
        chunk_filepath = os.path.join(output_dir, chunk_filename)
        chunk_df.to_parquet(chunk_filepath, index=False, engine='pyarrow')

    return output_dir

class FraudDataGenerator:
    """
    Generate synthetic e-commerce transaction data with fraud labels.

    This tool creates a realistic dataset of e-commerce transactions
    with both legitimate and fraudulent patterns for use in fraud detection
    model development and testing.
    """

    def generate(self, config_path='config.yaml'): # Removed output_file and chunk_size CLI parameters
        """
        Generate synthetic e-commerce transaction data and save to a parquet file.

        Args:
            config_path (str): Path to the configuration YAML file

        Returns:
            str: Path to the directory containing the generated parquet files
        """
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Get output path and chunk_size from config
        output_config = config.get('output', {}).get('file', {})
        output_path = output_config.get('path', 'output/ecommerce_fraud_dataset')
        chunk_size = output_config.get('chunk_size', 100)  # Default to 100 if not specified

        # Extract directory and filename pattern from the path
        if output_path.endswith('.parquet'):
            # Remove .parquet extension and use the directory and base filename
            output_file = os.path.dirname(output_path)
            filename_pattern = os.path.basename(output_path).replace('.parquet', '')
        else:
            output_file = output_path
            filename_pattern = 'results_'

        # Generate the dataset in chunks, writing directly to the output file
        # Pass chunk_size and filename_pattern to the generation function
        generated_output_dir = generate_ecommerce_dataset(
            num_customers=config['customer']['num_customers'],
            num_merchants=config['merchant']['num_merchants'],
            num_transactions=config['fraud']['num_transactions'],
            start_date=START_DATE, # Pass start_date and end_date explicitly
            end_date=END_DATE,
            chunk_size=chunk_size,
            output_dir=output_file, # Pass the output directory name
            filename_pattern=filename_pattern # Pass the filename pattern
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

            # we will skip the full correlation matrix and just calculate correlations with 'is_fraud' using Dask.
            print(f"\nCorrelation between fraud_risk_score and is_fraud (from Dask): {ddf['fraud_risk_score'].corr(ddf['is_fraud']).compute():.4f}")

            # Show feature correlations with fraud
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

