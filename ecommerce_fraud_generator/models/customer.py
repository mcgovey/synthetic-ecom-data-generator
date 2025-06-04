"""Customer model with personas and realistic behavior patterns."""

import random
import string
import uuid
import datetime
import ipaddress
from typing import Dict, List, Any, Optional, Tuple
from faker import Faker

from ..config.settings import CustomerPersona, GeographicDistribution, GlobalSettings

fake = Faker()
Faker.seed(GlobalSettings.FAKER_SEED)


class Customer:
    """Enhanced Customer class with personas and realistic behavior."""

    def __init__(self, customer_id: int):
        self.customer_id = customer_id

        # Assign persona based on weights
        self.persona_type = CustomerPersona.select_random_persona()
        self.persona = CustomerPersona.PERSONAS[self.persona_type]

        # Geographic assignment
        self.metro_area = GeographicDistribution.select_random_metro()
        self.metro_info = GeographicDistribution.get_metro_info(self.metro_area)

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
            date_start=GlobalSettings.START_DATE - datetime.timedelta(days=365*5),
            date_end=GlobalSettings.END_DATE
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

    def _generate_address(self) -> Dict[str, str]:
        """Generate address consistent with metro area."""
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

    def _generate_payment_info(self) -> Tuple[str, str, str]:
        """Generate realistic payment info based on persona."""
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

    def _generate_devices(self) -> List[Dict[str, str]]:
        """Generate devices based on persona preferences."""
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

    def _generate_ips(self) -> List[str]:
        """Generate geographic-consistent IP addresses."""
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

    def _generate_activity_pattern(self) -> List[int]:
        """Generate customer-specific activity hours based on persona and demographics."""
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

    def _generate_email(self) -> str:
        """Enhanced email generation with realistic patterns."""
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

    def get_transaction_timing_probability(self, hour: int, day_of_week: int) -> float:
        """Get probability of transaction at specific time based on customer pattern."""
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

    def get_preferred_categories(self, merchant_categories: List[str]) -> List[str]:
        """Get customer's preferred merchant categories."""
        return [cat for cat in merchant_categories if cat in self.preferred_categories]

    def get_common_device(self) -> Optional[Dict[str, str]]:
        """Get a common device for this customer."""
        if self.devices:
            return random.choice(self.devices)
        return None

    def get_common_ip(self) -> Optional[str]:
        """Get a common IP address for this customer."""
        if self.ips:
            return random.choice(self.ips)
        return None

    def get_new_device(self) -> Dict[str, str]:
        """Generate a new device using persona preferences."""
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

    def get_new_ip(self) -> str:
        """Generate new IP from same geographic region."""
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

    def update_last_purchase(self, date: datetime.datetime) -> None:
        """Update the last purchase date."""
        self.last_purchase_date = date

    def get_time_since_last_purchase(self, current_date: datetime.datetime) -> int:
        """Get number of days since last purchase."""
        if self.last_purchase_date is None:
            return 365  # Arbitrary large number for first purchase
        delta = current_date - self.last_purchase_date
        return delta.days
