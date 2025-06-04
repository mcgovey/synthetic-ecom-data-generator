"""Configuration settings and personas for the fraud data generator."""

import datetime
import random
import yaml
from typing import Dict, List, Tuple, Any, Optional


class GlobalSettings:
    """Global constants and settings for the data generator."""

    # Default dataset parameters
    NUM_CUSTOMERS = 5000
    NUM_MERCHANTS = 100
    NUM_TRANSACTIONS = 200000
    START_DATE = datetime.datetime(2023, 1, 1)
    END_DATE = datetime.datetime(2023, 12, 31)

    # Random seeds for reproducibility
    NUMPY_SEED = 42
    RANDOM_SEED = 42
    FAKER_SEED = 42


class CustomerPersona:
    """Define realistic customer personas with distinct behavior patterns."""

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

    @classmethod
    def get_persona_names(cls) -> List[str]:
        """Get list of persona names."""
        return list(cls.PERSONAS.keys())

    @classmethod
    def get_persona_weights(cls) -> List[float]:
        """Get list of persona weights for random selection."""
        return [cls.PERSONAS[p]['weight'] for p in cls.get_persona_names()]

    @classmethod
    def select_random_persona(cls) -> str:
        """Select a random persona based on weights."""
        return random.choices(cls.get_persona_names(), weights=cls.get_persona_weights())[0]


class GeographicDistribution:
    """Realistic geographic distribution for customers and transactions."""

    # US metro areas with realistic population weights
    METRO_AREAS = {
        'New York-Newark-Jersey City': {
            'weight': 0.12,
            'timezone': 'America/New_York',
            'affluence': 0.7
        },
        'Los Angeles-Long Beach-Anaheim': {
            'weight': 0.08,
            'timezone': 'America/Los_Angeles',
            'affluence': 0.6
        },
        'Chicago-Naperville-Elgin': {
            'weight': 0.05,
            'timezone': 'America/Chicago',
            'affluence': 0.6
        },
        'Dallas-Fort Worth-Arlington': {
            'weight': 0.04,
            'timezone': 'America/Chicago',
            'affluence': 0.5
        },
        'Houston-The Woodlands-Sugar Land': {
            'weight': 0.04,
            'timezone': 'America/Chicago',
            'affluence': 0.5
        },
        'Washington-Arlington-Alexandria': {
            'weight': 0.04,
            'timezone': 'America/New_York',
            'affluence': 0.8
        },
        'Miami-Fort Lauderdale-West Palm Beach': {
            'weight': 0.03,
            'timezone': 'America/New_York',
            'affluence': 0.6
        },
        'Philadelphia-Camden-Wilmington': {
            'weight': 0.03,
            'timezone': 'America/New_York',
            'affluence': 0.6
        },
        'Atlanta-Sandy Springs-Roswell': {
            'weight': 0.03,
            'timezone': 'America/New_York',
            'affluence': 0.5
        },
        'Boston-Cambridge-Newton': {
            'weight': 0.03,
            'timezone': 'America/New_York',
            'affluence': 0.8
        },
        'Other_Metro': {
            'weight': 0.35,
            'timezone': 'America/Chicago',
            'affluence': 0.4
        },
        'Rural': {
            'weight': 0.20,
            'timezone': 'America/Chicago',
            'affluence': 0.3
        }
    }

    @classmethod
    def get_metro_names(cls) -> List[str]:
        """Get list of metro area names."""
        return list(cls.METRO_AREAS.keys())

    @classmethod
    def get_metro_weights(cls) -> List[float]:
        """Get list of metro area weights for random selection."""
        return [cls.METRO_AREAS[m]['weight'] for m in cls.get_metro_names()]

    @classmethod
    def select_random_metro(cls) -> str:
        """Select a random metro area based on weights."""
        return random.choices(cls.get_metro_names(), weights=cls.get_metro_weights())[0]

    @classmethod
    def get_metro_info(cls, metro_name: str) -> Dict[str, Any]:
        """Get information about a specific metro area."""
        return cls.METRO_AREAS.get(metro_name, cls.METRO_AREAS['Other_Metro'])


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_default_config() -> str:
    """Create default configuration YAML content."""
    return '''
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
'''
