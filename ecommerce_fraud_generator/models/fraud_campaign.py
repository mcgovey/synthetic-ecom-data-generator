"""Fraud campaign model for simulating organized fraud attacks."""

import random
import datetime
from typing import List, Optional, Dict, Any


class FraudCampaign:
    """Model fraud as organized campaigns rather than independent events."""

    def __init__(self, campaign_id: int, start_date: datetime.datetime,
                 end_date: datetime.datetime, fraud_type: str,
                 target_merchants: Optional[List[int]] = None):
        self.campaign_id = campaign_id
        self.start_date = start_date
        self.end_date = end_date
        self.fraud_type = fraud_type
        self.target_merchants = target_merchants or []
        self.active_fraudsters = []
        self.tools_used = self._select_fraud_tools()
        self.intensity = random.betavariate(2, 5)  # Campaign intensity

    def _select_fraud_tools(self) -> List[str]:
        """Select fraud tools based on campaign type."""
        tool_sets = {
            'card_testing': ['bot_network', 'proxy_rotation', 'card_generator'],
            'account_takeover': ['credential_stuffing', 'social_engineering', 'sim_swapping'],
            'friendly_fraud': ['dispute_automation', 'chargeback_farming'],
            'bust_out': ['identity_synthesis', 'credit_building', 'coordinated_spending'],
            'refund_fraud': ['return_fraud_automation', 'fake_tracking']
        }
        return tool_sets.get(self.fraud_type, ['basic_fraud'])

    def is_active(self, date: datetime.datetime) -> bool:
        """Check if campaign is active on given date."""
        return self.start_date <= date <= self.end_date

    def get_fraud_probability(self, customer, merchant, base_prob: float) -> float:
        """Calculate fraud probability with campaign-specific modifications."""
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

    def add_fraudster(self, customer_id: int) -> None:
        """Add a customer to the active fraudsters list."""
        if customer_id not in self.active_fraudsters:
            self.active_fraudsters.append(customer_id)

    def remove_fraudster(self, customer_id: int) -> None:
        """Remove a customer from the active fraudsters list."""
        if customer_id in self.active_fraudsters:
            self.active_fraudsters.remove(customer_id)

    def get_campaign_info(self) -> Dict[str, Any]:
        """Get comprehensive campaign information."""
        return {
            'campaign_id': self.campaign_id,
            'fraud_type': self.fraud_type,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'duration_days': (self.end_date - self.start_date).days,
            'target_merchants': self.target_merchants,
            'tools_used': self.tools_used,
            'intensity': self.intensity,
            'active_fraudsters_count': len(self.active_fraudsters)
        }


def generate_fraud_campaigns(start_date: datetime.datetime,
                           end_date: datetime.datetime,
                           num_merchants: int) -> List[FraudCampaign]:
    """Generate realistic fraud campaigns over the time period."""
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
