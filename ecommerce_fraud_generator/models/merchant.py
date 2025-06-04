"""Merchant model with different risk levels and realistic business patterns."""

import random
import numpy as np
from typing import Dict, List, Any, Callable
from faker import Faker
from scipy.stats import lognorm

from ..config.settings import GlobalSettings

fake = Faker()
Faker.seed(GlobalSettings.FAKER_SEED)


class Merchant:
    """Merchant profiles with different risk levels and realistic business patterns."""

    def __init__(self, merchant_id: int):
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

    def _set_category_characteristics(self) -> None:
        """Set merchant characteristics based on business category."""
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

    def _generate_operating_hours(self) -> Dict[str, float]:
        """Generate realistic operating hours based on business type."""
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

    def _generate_seasonal_patterns(self) -> Dict[str, float]:
        """Generate seasonal business patterns."""
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

    def _calculate_customer_value(self) -> float:
        """Calculate average customer lifetime value."""
        base_value = self.avg_transaction_amount * self.customer_loyalty * 10
        return base_value * random.uniform(0.8, 1.2)

    def get_business_hour_multiplier(self, hour: int, day_of_week: int) -> float:
        """Get transaction probability multiplier based on business hours."""
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

    def get_seasonal_multiplier(self, date) -> float:
        """Get seasonal transaction multiplier."""
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

    def get_transaction_amount_distribution(self) -> Callable[[], float]:
        """Returns a function that generates realistic transaction amounts."""
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

    def get_customer_affinity(self, customer) -> float:
        """Calculate affinity between merchant and customer based on profiles."""
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
