"""Fraud pattern generators for different types of fraudulent behavior."""

import random
from typing import Dict, Any
from scipy.stats import bernoulli


class FriendlyFraudGenerator:
    """Generate friendly fraud patterns - legitimate customers disputing valid charges."""

    def __init__(self):
        self.friendly_fraud_triggers = {
            'buyer_remorse': 0.3,      # Customer regrets purchase
            'family_dispute': 0.2,     # Family member made purchase
            'subscription_forgotten': 0.25,  # Forgot about recurring charge
            'delivery_issues': 0.15,   # Package not received/damaged
            'merchant_dispute': 0.1    # Dissatisfied with service
        }

    def generate_friendly_fraud(self, customer, transaction: Dict[str, Any], merchant) -> float:
        """
        Determine probability that a transaction becomes friendly fraud.

        Args:
            customer: Customer object
            transaction: Transaction data dictionary
            merchant: Merchant object

        Returns:
            Probability of friendly fraud (0.0 to 1.0)
        """
        base_prob = 0.02  # 2% base rate

        # Buyer's remorse factors
        if transaction['amount'] > customer.avg_purchase_amount * 3:
            base_prob *= 2.5

        # Merchant category risk
        high_dispute_categories = ['Digital Products', 'Services', 'Travel']
        if merchant.category in high_dispute_categories:
            base_prob *= 1.8

        # Customer tenure (longer customers more likely to dispute)
        if hasattr(transaction, 'timestamp') and hasattr(customer, 'signup_date'):
            customer_tenure_days = (transaction['timestamp'].date() - customer.signup_date).days
            if customer_tenure_days > 365:
                base_prob *= 1.3

        # Subscription/recurring billing
        if hasattr(transaction, 'is_recurring') and transaction.get('is_recurring', False):
            base_prob *= 2.0

        return min(0.25, base_prob)  # Cap at 25%

    def select_fraud_trigger(self) -> str:
        """Select a random fraud trigger based on weights."""
        triggers = list(self.friendly_fraud_triggers.keys())
        weights = list(self.friendly_fraud_triggers.values())
        return random.choices(triggers, weights=weights)[0]


class TechnicalFraudGenerator:
    """Generate technical fraud patterns like card testing and bot attacks."""

    def __init__(self):
        self.card_testing_patterns = {
            'rapid_small_amounts': 0.4,  # Small amounts in rapid succession
            'sequential_cards': 0.3,     # Sequential card numbers
            'same_ip_multiple_cards': 0.2,  # Same IP, different cards
            'bot_user_agents': 0.1       # Automated user agents
        }

    def generate_card_testing_probability(self, customer, merchant, recent_transactions) -> float:
        """Generate probability of card testing fraud."""
        base_prob = 0.01

        # High-risk merchant categories
        if merchant.category in ['Digital Products', 'Services']:
            base_prob *= 3.0

        # Check for rapid transactions from same customer
        if len(recent_transactions) > 5:  # More than 5 transactions recently
            base_prob *= 2.0

        # Small amount pattern
        if hasattr(merchant, 'avg_transaction_amount'):
            if merchant.avg_transaction_amount < 50:  # Small amounts
                base_prob *= 1.5

        return min(0.8, base_prob)


class AccountTakeoverGenerator:
    """Generate account takeover fraud patterns."""

    def __init__(self):
        self.takeover_indicators = {
            'unusual_location': 0.3,     # Login from unusual location
            'new_device': 0.25,          # New device/browser
            'rapid_purchases': 0.2,      # Rapid high-value purchases
            'address_change': 0.15,      # Shipping address change
            'payment_method_change': 0.1 # New payment method
        }

    def generate_takeover_probability(self, customer, transaction_data) -> float:
        """Generate probability of account takeover."""
        base_prob = 0.005  # 0.5% base rate

        # New device indicator
        if transaction_data.get('is_new_device', False):
            base_prob *= 3.0

        # Address mismatch
        if not transaction_data.get('address_match', True):
            base_prob *= 2.0

        # High-value customer targets
        if customer.persona_type in ['premium_customer', 'high_value']:
            base_prob *= 1.5

        return min(0.6, base_prob)


class RefundFraudGenerator:
    """Generate refund fraud patterns."""

    def __init__(self):
        self.refund_patterns = {
            'fake_returns': 0.4,         # Claiming returns that weren't made
            'wardrobing': 0.3,           # Using items then returning
            'receipt_fraud': 0.2,        # Using fake receipts
            'package_theft_claims': 0.1  # Claiming packages were stolen
        }

    def generate_refund_fraud_probability(self, customer, merchant) -> float:
        """Generate probability of refund fraud."""
        base_prob = 0.008  # 0.8% base rate

        # Merchant categories with return policies
        if merchant.category in ['Clothing', 'Electronics']:
            base_prob *= 2.0

        # Customer risk factors
        if customer.persona_type == 'budget_conscious':
            base_prob *= 1.3

        return min(0.3, base_prob)
