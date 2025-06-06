"""
Advanced Behavioral Modeling Engine
Implements sophisticated user behavior patterns, temporal evolution, and fraud mimicry
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from scipy import stats


@dataclass
class BehaviorProfile:
    """User behavior profile with consistency metrics"""
    user_id: int
    behavior_consistency: float
    tech_savviness: float
    privacy_consciousness: float
    spending_volatility: float
    time_pattern_consistency: float
    device_loyalty: float

    # Temporal patterns
    preferred_hours: List[int]
    preferred_days: List[int]
    seasonal_preferences: Dict[str, float]

    # Network characteristics
    social_network_size: int
    influence_susceptibility: float


class TemporalEvolutionEngine:
    """Models how fraud patterns evolve over time"""

    def __init__(self, config: Dict):
        self.lifecycle_days = config.get('fraud_lifecycle_days', 90)
        self.adaptation_rate = config.get('detection_adaptation_rate', 0.15)
        self.seasonal_patterns = config.get('seasonal_patterns', True)

        # Fraud evolution phases
        self.phases = {
            'exploration': (0, 14),      # Days 0-14: Testing defenses
            'exploitation': (15, 45),    # Days 15-45: Full operation
            'adaptation': (46, 75),      # Days 46-75: Evolving tactics
            'migration': (76, 90)        # Days 76-90: Moving to new methods
        }

    def get_fraud_phase(self, days_since_start: int) -> str:
        """Determine current fraud campaign phase"""
        for phase, (start, end) in self.phases.items():
            if start <= days_since_start <= end:
                return phase
        return 'dormant'

    def calculate_sophistication_multiplier(self, days_since_start: int,
                                          base_intelligence: float) -> float:
        """Calculate how fraud sophistication changes over time"""
        phase = self.get_fraud_phase(days_since_start)

        phase_multipliers = {
            'exploration': 0.6,  # Lower sophistication during testing
            'exploitation': 1.0, # Full sophistication
            'adaptation': 1.3,   # Higher sophistication after adaptation
            'migration': 0.8,    # Reduced during transition
            'dormant': 0.3
        }

        base_multiplier = phase_multipliers.get(phase, 1.0)

        # Add adaptation learning curve
        adaptation_boost = min(days_since_start * self.adaptation_rate / 30, 0.4)

        return base_intelligence * (base_multiplier + adaptation_boost)

    def get_seasonal_fraud_multiplier(self, timestamp: datetime) -> float:
        """Calculate seasonal fraud rate adjustments"""
        if not self.seasonal_patterns:
            return 1.0

        month = timestamp.month

        # Higher fraud rates during shopping seasons
        seasonal_multipliers = {
            11: 1.8,  # November (Black Friday)
            12: 1.6,  # December (Christmas)
            1: 1.4,   # January (New Year/Returns)
            2: 1.2,   # February (Valentine's)
            3: 0.9,   # March
            4: 0.8,   # April
            5: 1.0,   # May (Mother's Day)
            6: 0.9,   # June
            7: 1.1,   # July (Summer shopping)
            8: 0.8,   # August
            9: 1.0,   # September (Back to school)
            10: 1.1   # October (Halloween)
        }

        return seasonal_multipliers.get(month, 1.0)


class BehaviorMimicryEngine:
    """Advanced fraud behavior that mimics legitimate users"""

    def __init__(self, config: Dict):
        self.mimicry_strength = config.get('fraud_behavior_mimicry', 0.70)
        self.ml_awareness = config.get('ml_model_awareness', 0.60)
        self.feature_awareness = config.get('feature_engineering_awareness', 0.45)

    def generate_mimicry_pattern(self, legitimate_profile: BehaviorProfile,
                                fraud_intelligence: float) -> Dict:
        """Generate fraud behavior that mimics legitimate patterns"""

        # Base mimicry strength adjusted by intelligence
        effective_mimicry = min(self.mimicry_strength * fraud_intelligence, 0.95)

        # Sophisticated fraudsters mimic specific aspects
        mimicry_aspects = {
            'timing_patterns': effective_mimicry * 0.9,
            'amount_patterns': effective_mimicry * 0.8,
            'device_consistency': effective_mimicry * 0.7,
            'geographic_patterns': effective_mimicry * 0.85,
            'velocity_patterns': effective_mimicry * 0.6  # Harder to mimic
        }

        return mimicry_aspects

    def apply_ml_evasion_tactics(self, transaction_data: Dict,
                                fraud_intelligence: float) -> Dict:
        """Apply ML model evasion tactics"""

        if fraud_intelligence < self.ml_awareness:
            return transaction_data  # Not sophisticated enough

        evasion_tactics = {}

        # Feature value manipulation to avoid common thresholds
        if 'amount' in transaction_data:
            amount = transaction_data['amount']

            # Avoid common fraud detection thresholds
            suspicious_thresholds = [100, 250, 500, 1000, 2500, 5000]
            for threshold in suspicious_thresholds:
                if abs(amount - threshold) < 10:
                    # Adjust to avoid threshold
                    adjustment = random.choice([-15, -12, -8, 8, 12, 15])
                    evasion_tactics['amount_adjustment'] = adjustment
                    break

        # Velocity evasion
        if fraud_intelligence > 0.8:
            evasion_tactics['velocity_obfuscation'] = {
                'time_dispersal': True,
                'amount_variation': True,
                'geographic_distribution': True
            }

        # Feature noise injection
        if fraud_intelligence > 0.7:
            evasion_tactics['feature_noise'] = {
                'minor_geographic_variations': True,
                'timing_jitter': True,
                'device_fingerprint_rotation': True
            }

        return evasion_tactics


class NetworkEffectsEngine:
    """Models fraud networks and coordinated attacks"""

    def __init__(self, config: Dict):
        self.network_size_range = config.get('fraud_network_size', [3, 15])
        self.velocity_patterns = config.get('velocity_patterns', True)
        self.similarity_scores = config.get('account_similarity_scores', True)

        # Network topologies
        self.network_types = {
            'hub_and_spoke': 0.4,    # One central account, many satellites
            'distributed': 0.3,      # Evenly connected network
            'layered': 0.2,          # Hierarchical structure
            'hybrid': 0.1           # Mixed topology
        }

    def generate_fraud_network(self, network_id: int) -> Dict:
        """Generate a coordinated fraud network"""

        network_size = random.randint(*self.network_size_range)
        network_type = np.random.choice(
            list(self.network_types.keys()),
            p=list(self.network_types.values())
        )

        # Generate network characteristics
        network = {
            'network_id': network_id,
            'size': network_size,
            'topology': network_type,
            'coordination_level': random.uniform(0.6, 0.95),
            'sophistication_variance': random.uniform(0.1, 0.3),
            'geographic_spread': random.choice(['local', 'regional', 'international']),
            'operational_timeframe': random.randint(7, 60),  # Days
            'target_merchants': random.randint(1, 5)
        }

        # Generate member characteristics
        network['members'] = []
        base_intelligence = random.uniform(0.4, 0.9)

        for i in range(network_size):
            # Intelligence varies within network
            variance = random.uniform(-network['sophistication_variance'],
                                    network['sophistication_variance'])
            member_intelligence = np.clip(base_intelligence + variance, 0.2, 0.95)

            member = {
                'member_id': i,
                'role': self._assign_network_role(i, network_type, network_size),
                'intelligence_level': member_intelligence,
                'activity_correlation': random.uniform(0.3, 0.8),
                'risk_tolerance': random.uniform(0.4, 0.9)
            }
            network['members'].append(member)

        return network

    def _assign_network_role(self, member_id: int, topology: str,
                           network_size: int) -> str:
        """Assign role based on network topology"""

        if topology == 'hub_and_spoke':
            return 'hub' if member_id == 0 else 'spoke'
        elif topology == 'layered':
            if member_id < network_size * 0.2:
                return 'controller'
            elif member_id < network_size * 0.5:
                return 'operator'
            else:
                return 'executor'
        else:
            return random.choice(['operator', 'executor', 'specialist'])

    def calculate_network_velocity_pattern(self, network: Dict,
                                         current_time: datetime) -> Dict:
        """Calculate coordinated velocity patterns"""

        if not self.velocity_patterns:
            return {}

        coordination = network['coordination_level']

        # Time windows for coordinated attacks
        attack_windows = {
            'burst': timedelta(minutes=random.randint(5, 30)),
            'sustained': timedelta(hours=random.randint(2, 8)),
            'distributed': timedelta(days=random.randint(1, 5))
        }

        # Select attack pattern based on sophistication
        if coordination > 0.8:
            pattern = 'distributed'
        elif coordination > 0.6:
            pattern = 'sustained'
        else:
            pattern = 'burst'

        return {
            'attack_pattern': pattern,
            'time_window': attack_windows[pattern],
            'coordination_score': coordination,
            'expected_velocity': network['size'] * coordination,
            'geographic_dispersion': network['geographic_spread']
        }


class LegitimateUserSophistication:
    """Models legitimate users who use privacy tools and sophisticated behaviors"""

    def __init__(self, config: Dict):
        self.privacy_conscious_rate = config.get('privacy_conscious_rate', 0.25)
        self.tech_savvy_rate = config.get('tech_savvy_behaviors', 0.35)
        self.cross_device_rate = config.get('cross_device_usage', 0.60)

    def generate_sophisticated_legitimate_behavior(self, user_profile: BehaviorProfile) -> Dict:
        """Generate sophisticated but legitimate user behaviors"""

        behaviors = {}

        # Privacy-conscious behaviors
        if random.random() < self.privacy_conscious_rate:
            behaviors['privacy_tools'] = {
                'vpn_usage': random.choice([True, False]),
                'ad_blocker': random.choice([True, False]),
                'tracking_protection': True,
                'incognito_mode_frequency': random.uniform(0.1, 0.4)
            }

        # Tech-savvy behaviors
        if random.random() < self.tech_savvy_rate:
            behaviors['tech_savvy'] = {
                'multiple_browsers': True,
                'browser_customization': True,
                'developer_tools_usage': random.choice([True, False]),
                'extension_count': random.randint(5, 20)
            }

        # Cross-device usage patterns
        if random.random() < self.cross_device_rate:
            behaviors['cross_device'] = {
                'device_count': random.randint(2, 5),
                'device_switching_frequency': random.uniform(0.2, 0.8),
                'session_continuity': random.uniform(0.3, 0.9)
            }

        # Sophisticated shopping behaviors
        if user_profile.tech_savviness > 0.7:
            behaviors['shopping_sophistication'] = {
                'price_comparison': True,
                'review_analysis': True,
                'deal_hunting': random.choice([True, False]),
                'social_commerce': random.choice([True, False])
            }

        return behaviors


class BehavioralModelingEngine:
    """Main engine coordinating all behavioral modeling components"""

    def __init__(self, config: Dict):
        self.config = config
        self.temporal_engine = TemporalEvolutionEngine(config.get('temporal_evolution', {}))
        self.mimicry_engine = BehaviorMimicryEngine(config.get('advanced_evasion', {}))
        self.network_engine = NetworkEffectsEngine(config.get('network_effects', {}))
        self.legitimate_engine = LegitimateUserSophistication(config.get('legitimate_sophistication', {}))

        self.behavior_consistency = config.get('behavioral_modeling', {}).get('user_behavior_consistency', 0.85)

        # Cache for generated profiles
        self.user_profiles = {}
        self.fraud_networks = {}

    def generate_user_behavior_profile(self, user_id: int, is_fraudster: bool = False) -> BehaviorProfile:
        """Generate comprehensive user behavior profile"""

        if user_id in self.user_profiles:
            return self.user_profiles[user_id]

        # Base behavior characteristics
        if is_fraudster:
            # Fraudsters have different behavior patterns
            consistency = random.uniform(0.3, 0.7)
            tech_savviness = random.uniform(0.6, 0.95)
            privacy_consciousness = random.uniform(0.7, 0.95)
        else:
            # Legitimate users
            consistency = random.uniform(0.7, 0.95)
            tech_savviness = random.uniform(0.2, 0.8)
            privacy_consciousness = random.uniform(0.1, 0.6)

        profile = BehaviorProfile(
            user_id=user_id,
            behavior_consistency=consistency,
            tech_savviness=tech_savviness,
            privacy_consciousness=privacy_consciousness,
            spending_volatility=random.uniform(0.1, 0.8),
            time_pattern_consistency=random.uniform(0.6, 0.9),
            device_loyalty=random.uniform(0.4, 0.9),
            preferred_hours=sorted(random.sample(range(24), random.randint(4, 12))),
            preferred_days=sorted(random.sample(range(7), random.randint(3, 7))),
            seasonal_preferences={
                'spring': random.uniform(0.5, 1.5),
                'summer': random.uniform(0.5, 1.5),
                'fall': random.uniform(0.5, 1.5),
                'winter': random.uniform(0.5, 1.5)
            },
            social_network_size=random.randint(5, 200),
            influence_susceptibility=random.uniform(0.1, 0.8)
        )

        self.user_profiles[user_id] = profile
        return profile

    def enhance_transaction_with_behavioral_context(self, transaction: Dict,
                                                  timestamp: datetime) -> Dict:
        """Enhance transaction with behavioral modeling context"""

        enhanced = transaction.copy()
        user_id = transaction.get('customer_id')
        is_fraud = transaction.get('is_fraud', False)

        # Generate or retrieve user profile
        profile = self.generate_user_behavior_profile(user_id, is_fraud)

        # Add temporal evolution context
        if is_fraud:
            days_since_campaign_start = random.randint(0, self.temporal_engine.lifecycle_days)
            base_intelligence = transaction.get('fraud_intelligence_level', 0.5)

            enhanced['fraud_campaign_age_days'] = days_since_campaign_start
            enhanced['fraud_phase'] = self.temporal_engine.get_fraud_phase(days_since_campaign_start)
            enhanced['evolved_sophistication'] = self.temporal_engine.calculate_sophistication_multiplier(
                days_since_campaign_start, base_intelligence
            )
            enhanced['seasonal_fraud_multiplier'] = self.temporal_engine.get_seasonal_fraud_multiplier(timestamp)

            # Apply mimicry patterns
            mimicry = self.mimicry_engine.generate_mimicry_pattern(profile, base_intelligence)
            enhanced['mimicry_patterns'] = mimicry

            # Apply ML evasion tactics
            evasion = self.mimicry_engine.apply_ml_evasion_tactics(transaction, base_intelligence)
            enhanced['evasion_tactics'] = evasion
        else:
            # Legitimate user sophistication
            sophisticated_behavior = self.legitimate_engine.generate_sophisticated_legitimate_behavior(profile)
            enhanced['legitimate_sophistication'] = sophisticated_behavior

        # Add behavioral consistency scores
        enhanced['behavior_consistency_score'] = profile.behavior_consistency
        enhanced['tech_savviness_score'] = profile.tech_savviness
        enhanced['privacy_consciousness_score'] = profile.privacy_consciousness

        return enhanced
