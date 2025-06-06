"""
Enhanced Fraud Pattern Generators with Adversarial Intelligence.

This module implements sophisticated fraud generators that model real-world adversarial
behavior, detection evasion tactics, and adaptive fraud patterns.
"""

import random
import datetime
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import bernoulli
import numpy as np


class FraudRateController:
    """Controls realistic fraud rates and validates against industry standards."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('fraud_rates', {})
        self.industry_type = self.config.get('industry_type', 'general_ecommerce')
        self.base_fraud_rate = self.config.get('base_fraud_rate', 0.015)
        self.friendly_fraud_rate = self.config.get('friendly_fraud_rate', 0.008)
        self.technical_fraud_rate = self.config.get('technical_fraud_rate', 0.007)

        # Industry multipliers
        self.industry_multipliers = self.config.get('industry_multipliers', {
            'general_ecommerce': 1.0,
            'digital_goods': 2.5,
            'travel': 1.8,
            'financial_services': 0.6
        })

        # Validation settings
        self.validation_config = self.config.get('validation', {})
        self.warn_threshold = self.validation_config.get('warn_if_above', 0.05)
        self.error_threshold = self.validation_config.get('error_if_above', 0.10)

    def get_effective_fraud_rate(self) -> float:
        """Calculate the effective fraud rate including industry multipliers."""
        multiplier = self.industry_multipliers.get(self.industry_type, 1.0)
        effective_rate = self.base_fraud_rate * multiplier

        # Validate fraud rate
        if self.validation_config.get('enabled', True):
            if effective_rate > self.error_threshold:
                raise ValueError(f"Fraud rate {effective_rate:.3f} exceeds error threshold {self.error_threshold}")
            elif effective_rate > self.warn_threshold:
                print(f"WARNING: Fraud rate {effective_rate:.3f} exceeds recommended threshold {self.warn_threshold}")

        return effective_rate

    def get_friendly_fraud_rate(self) -> float:
        """Get the friendly fraud rate."""
        multiplier = self.industry_multipliers.get(self.industry_type, 1.0)
        return self.friendly_fraud_rate * multiplier

    def get_technical_fraud_rate(self) -> float:
        """Get the technical fraud rate."""
        multiplier = self.industry_multipliers.get(self.industry_type, 1.0)
        return self.technical_fraud_rate * multiplier


class AdversarialFraudGenerator:
    """
    Generate sophisticated fraud patterns that mimic legitimate behavior
    and actively evade detection systems.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('adversarial_patterns', {})
        self.detection_evasion = self.config.get('detection_evasion', {})
        self.adaptive_behavior = self.config.get('adaptive_behavior', {})
        self.evasion_tactics = self.config.get('evasion_tactics', {})

        # Track detection patterns to adapt to
        self.detected_patterns = {
            'amount_thresholds': [100, 500, 1000, 2500],
            'velocity_limits': {'daily': 5, 'hourly': 2},
            'geographic_restrictions': ['unusual_location', 'vpn_detected']
        }

        # Intelligence levels for different fraud types
        self.intelligence_levels = {
            'card_testing': 0.8,
            'account_takeover': 0.9,
            'friendly_fraud': 0.3,
            'bust_out': 0.7,
            'refund_fraud': 0.6
        }

    def generate_evasive_fraud_pattern(self, customer, merchant, fraud_type: str,
                                     timestamp: datetime.datetime) -> Dict[str, Any]:
        """Generate fraud pattern with sophisticated evasion tactics."""
        intelligence_level = self.intelligence_levels.get(fraud_type, 0.5)

        # Base fraud pattern
        fraud_pattern = {
            'fraud_type': fraud_type,
            'intelligence_level': intelligence_level,
            'evasion_tactics_used': [],
            'detection_risk_score': 0.0
        }

        # Apply adversarial intelligence based on fraud type
        if fraud_type == 'card_testing':
            fraud_pattern.update(self._generate_evasive_card_testing(customer, merchant, intelligence_level))
        elif fraud_type == 'account_takeover':
            fraud_pattern.update(self._generate_gradual_takeover(customer, merchant, intelligence_level))
        elif fraud_type == 'friendly_fraud':
            fraud_pattern.update(self._generate_disguised_friendly_fraud(customer, merchant, intelligence_level))
        elif fraud_type == 'bust_out':
            fraud_pattern.update(self._generate_strategic_bust_out(customer, merchant, intelligence_level))
        elif fraud_type == 'refund_fraud':
            fraud_pattern.update(self._generate_sophisticated_refund_fraud(customer, merchant, intelligence_level))

        return fraud_pattern

    def _generate_evasive_card_testing(self, customer, merchant, intelligence: float) -> Dict[str, Any]:
        """Generate card testing that avoids detection thresholds."""
        pattern = {
            'evasion_tactics_used': [],
            'detection_risk_score': 0.2  # Start with low risk
        }

        if intelligence > 0.7 and self.detection_evasion.get('amount_threshold_awareness', False):
            # Avoid common amount thresholds
            pattern['amount_strategy'] = 'threshold_aware'
            pattern['target_amounts'] = [9.99, 49.99, 89.99]  # Just below common thresholds
            pattern['evasion_tactics_used'].append('amount_threshold_evasion')

        if intelligence > 0.6 and self.detection_evasion.get('velocity_threshold_awareness', False):
            # Space transactions to avoid velocity detection
            pattern['timing_strategy'] = 'velocity_aware'
            pattern['transaction_spacing_hours'] = random.uniform(8, 24)  # Space transactions
            pattern['evasion_tactics_used'].append('velocity_evasion')

        if intelligence > 0.8 and self.evasion_tactics.get('legitimate_pattern_mimicry', False):
            # Copy victim's shopping patterns
            pattern['mimicry_strategy'] = 'customer_pattern_copy'
            pattern['mimic_customer_times'] = True
            pattern['mimic_customer_merchants'] = True
            pattern['evasion_tactics_used'].append('pattern_mimicry')
            pattern['detection_risk_score'] = 0.1  # Very low risk due to mimicry

        return pattern

    def _generate_gradual_takeover(self, customer, merchant, intelligence: float) -> Dict[str, Any]:
        """Generate account takeover with gradual escalation."""
        pattern = {
            'phases': [],
            'evasion_tactics_used': [],
            'detection_risk_score': 0.1  # Start very low
        }

        if intelligence > 0.8:
            # Phase 1: Reconnaissance (appears completely legitimate)
            pattern['phases'].append({
                'phase': 'reconnaissance',
                'duration_days': random.randint(3, 7),
                'behavior': 'mimic_legitimate_exactly',
                'amounts': 'customer_typical_range',
                'merchants': 'customer_frequent_only',
                'detection_risk': 0.05
            })

            # Phase 2: Small tests (still low risk)
            pattern['phases'].append({
                'phase': 'small_tests',
                'duration_days': random.randint(2, 5),
                'behavior': 'slightly_elevated_spending',
                'amounts': 'customer_range_upper_bound',
                'detection_risk': 0.15
            })

            # Phase 3: Full exploitation (higher risk but after trust established)
            pattern['phases'].append({
                'phase': 'exploitation',
                'duration_days': random.randint(1, 3),
                'behavior': 'high_value_rapid_extraction',
                'detection_risk': 0.8
            })

            pattern['evasion_tactics_used'].extend([
                'gradual_escalation', 'trust_building', 'pattern_establishment'
            ])

        return pattern

    def _generate_disguised_friendly_fraud(self, customer, merchant, intelligence: float) -> Dict[str, Any]:
        """Generate friendly fraud that appears completely legitimate."""
        pattern = {
            'evasion_tactics_used': [],
            'detection_risk_score': 0.0  # Appears completely legitimate
        }

        # Friendly fraud should appear as normal transactions
        pattern['transaction_appearance'] = 'completely_legitimate'
        pattern['dispute_timing'] = random.randint(30, 90)  # Days after transaction
        pattern['dispute_reason'] = random.choice([
            'did_not_authorize', 'item_not_received', 'defective_product',
            'service_not_provided', 'billing_error'
        ])

        if intelligence > 0.4:
            # Even low intelligence friendly fraudsters avoid obvious patterns
            pattern['amount_strategy'] = 'within_normal_range'
            pattern['timing_strategy'] = 'normal_shopping_hours'

        return pattern

    def _generate_strategic_bust_out(self, customer, merchant, intelligence: float) -> Dict[str, Any]:
        """Generate strategic bust-out fraud with credit limit building."""
        pattern = {
            'phases': [],
            'evasion_tactics_used': [],
            'detection_risk_score': 0.1
        }

        if intelligence > 0.6:
            # Phase 1: Build trust and payment history
            pattern['phases'].append({
                'phase': 'trust_building',
                'duration_days': random.randint(30, 60),
                'behavior': 'consistent_small_payments',
                'payment_reliability': 1.0,  # Always pay on time
                'detection_risk': 0.05
            })

            # Phase 2: Gradual limit increases
            pattern['phases'].append({
                'phase': 'limit_building',
                'duration_days': random.randint(20, 40),
                'behavior': 'strategic_limit_requests',
                'payment_reliability': 0.95,  # Still very reliable
                'detection_risk': 0.1
            })

            # Phase 3: Maximum extraction
            pattern['phases'].append({
                'phase': 'extraction',
                'duration_days': random.randint(3, 7),
                'behavior': 'rapid_maximum_utilization',
                'payment_reliability': 0.0,  # Never pay
                'detection_risk': 0.9
            })

            pattern['evasion_tactics_used'].extend([
                'trust_establishment', 'credit_building', 'strategic_timing'
            ])

        return pattern

    def _generate_sophisticated_refund_fraud(self, customer, merchant, intelligence: float) -> Dict[str, Any]:
        """Generate refund fraud with sophisticated cover stories."""
        pattern = {
            'evasion_tactics_used': [],
            'detection_risk_score': 0.2
        }

        if intelligence > 0.5:
            # Sophisticated refund fraud scenarios
            pattern['fraud_scenario'] = random.choice([
                'wardrobing_with_tags_intact',
                'return_empty_box_weight_matched',
                'receipt_fraud_legitimate_purchase',
                'serial_number_swap',
                'damage_claim_self_inflicted'
            ])

            pattern['timing_strategy'] = 'within_return_policy'
            pattern['documentation'] = 'complete_and_convincing'
            pattern['evasion_tactics_used'].append('sophisticated_cover_story')

        return pattern


class EnhancedFriendlyFraudGenerator:
    """Enhanced friendly fraud generator with realistic patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('friendly_fraud', {})
        self.triggers = self.config.get('triggers', {
            'buyer_remorse': 0.3,
            'family_dispute': 0.2,
            'subscription_forgotten': 0.25,
            'delivery_issues': 0.15,
            'merchant_dispute': 0.1
        })

    def generate_friendly_fraud(self, customer, transaction: Dict[str, Any],
                              merchant, fraud_rate_controller: FraudRateController) -> Tuple[float, Dict[str, Any]]:
        """Generate realistic friendly fraud probability and metadata."""
        base_rate = fraud_rate_controller.get_friendly_fraud_rate()

        # Customer factors
        risk_multiplier = 1.0

        # High-value customers more likely to dispute
        if hasattr(customer, 'persona_type'):
            if customer.persona_type in ['premium_customer', 'high_value']:
                risk_multiplier *= 1.4
            elif customer.persona_type == 'budget_conscious':
                risk_multiplier *= 1.2

        # Merchant category risk
        high_dispute_categories = ['Digital Products', 'Services', 'Travel', 'Clothing']
        if merchant.category in high_dispute_categories:
            risk_multiplier *= 1.6

        # Transaction amount (higher amounts more likely to be disputed)
        if transaction['amount'] > 500:
            risk_multiplier *= 1.8
        elif transaction['amount'] > 200:
            risk_multiplier *= 1.3

        final_probability = min(0.20, base_rate * risk_multiplier)

        # Generate metadata
        metadata = {
            'fraud_type': 'friendly_fraud',
            'trigger': self._select_trigger(),
            'dispute_timing_days': random.randint(15, 90),
            'likelihood_of_chargeback': random.uniform(0.6, 0.9)
        }

        return final_probability, metadata

    def _select_trigger(self) -> str:
        """Select a random fraud trigger based on weights."""
        triggers = list(self.triggers.keys())
        weights = list(self.triggers.values())
        return random.choices(triggers, weights=weights)[0]


class TechnicalFraudGenerator:
    """Enhanced technical fraud patterns with adversarial intelligence."""

    def __init__(self, config: Dict[str, Any], adversarial_generator: AdversarialFraudGenerator):
        self.config = config
        self.adversarial_generator = adversarial_generator

    def generate_card_testing_probability(self, customer, merchant, recent_transactions,
                                        fraud_rate_controller: FraudRateController) -> Tuple[float, Dict[str, Any]]:
        """Generate sophisticated card testing probability."""
        base_rate = fraud_rate_controller.get_technical_fraud_rate() * 0.4  # 40% of technical fraud is card testing

        # High-risk scenarios
        risk_multiplier = 1.0

        # Digital goods and services are prime targets
        if merchant.category in ['Digital Products', 'Services']:
            risk_multiplier *= 3.0

        # Low-value items preferred for testing
        if hasattr(merchant, 'avg_transaction_amount') and merchant.avg_transaction_amount < 50:
            risk_multiplier *= 2.0

        final_probability = min(0.8, base_rate * risk_multiplier)

        # Generate adversarial pattern
        metadata = self.adversarial_generator.generate_evasive_fraud_pattern(
            customer, merchant, 'card_testing', datetime.datetime.now()
        )

        return final_probability, metadata

    def generate_account_takeover_probability(self, customer, transaction_data,
                                           fraud_rate_controller: FraudRateController) -> Tuple[float, Dict[str, Any]]:
        """Generate sophisticated account takeover probability."""
        base_rate = fraud_rate_controller.get_technical_fraud_rate() * 0.3  # 30% of technical fraud is ATO

        # Risk factors
        risk_multiplier = 1.0

        # High-value customers are prime targets
        if hasattr(customer, 'persona_type') and customer.persona_type in ['premium_customer', 'high_value']:
            risk_multiplier *= 2.5

        # New device/IP indicators
        if transaction_data.get('is_new_device', False):
            risk_multiplier *= 1.8
        if transaction_data.get('is_new_ip', False):
            risk_multiplier *= 1.6

        final_probability = min(0.6, base_rate * risk_multiplier)

        # Generate adversarial pattern
        metadata = self.adversarial_generator.generate_evasive_fraud_pattern(
            customer, None, 'account_takeover', datetime.datetime.now()
        )

        return final_probability, metadata
