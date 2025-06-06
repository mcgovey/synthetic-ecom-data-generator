"""
Enhanced Metadata Generator for Raw Transaction Data.

This module generates comprehensive transaction metadata including payment processing data,
device fingerprinting, network context, and transaction flow information.
"""

import random
import secrets
import datetime
from typing import Dict, Any, Optional, List
from faker import Faker
import uuid

fake = Faker()


class PaymentProcessingMetadataGenerator:
    """Generate realistic payment processing metadata."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('payment_processing', {})
        self.enabled = self.config.get('enabled', True)

        # Common payment processors
        self.payment_processors = [
            'stripe', 'square', 'paypal', 'authorize_net', 'adyen', 'braintree'
        ]

        # Common decline reasons
        self.decline_reasons = [
            'insufficient_funds', 'card_declined', 'invalid_cvv', 'expired_card',
            'invalid_card_number', 'suspected_fraud', 'limit_exceeded', 'lost_stolen'
        ]

        # Processing time ranges (milliseconds)
        self.processing_times = {
            'fast': (50, 200),
            'normal': (200, 800),
            'slow': (800, 3000),
            'timeout': (3000, 10000)
        }

    def generate_authorization_code(self) -> str:
        """Generate realistic authorization code."""
        if not self.config.get('generate_authorization_codes', True):
            return None

        # 6-8 digit alphanumeric authorization codes
        length = random.randint(6, 8)
        return ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=length))

    def generate_processing_time(self, is_fraud: bool = False) -> int:
        """Generate realistic processing time in milliseconds."""
        if not self.config.get('generate_processing_times', True):
            return None

        # Fraud transactions often have longer processing times
        if is_fraud and random.random() < 0.3:
            time_range = self.processing_times['slow']
        else:
            time_range = self.processing_times['normal']

        return random.randint(*time_range)

    def generate_decline_reason(self) -> Optional[str]:
        """Generate decline reason (only for declined transactions)."""
        if not self.config.get('generate_decline_reasons', True):
            return None

        return random.choice(self.decline_reasons)

    def generate_payment_processor_data(self) -> Dict[str, Any]:
        """Generate payment processor specific data."""
        if not self.config.get('generate_payment_processor_data', True):
            return {}

        processor = random.choice(self.payment_processors)

        return {
            'payment_processor': processor,
            'processor_transaction_id': str(uuid.uuid4())[:16],
            'gateway_response_code': random.choice(['00', '05', '12', '14', '51', '54', '61', '62']),
            'processor_fee_cents': random.randint(29, 250)  # Typical processing fees
        }


class DeviceFingerprintingGenerator:
    """Generate realistic device fingerprinting data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('device_fingerprinting', {})
        self.enabled = self.config.get('enabled', True)

        # Common screen resolutions
        self.screen_resolutions = [
            '1920x1080', '1366x768', '1440x900', '1280x1024', '1024x768',
            '1680x1050', '1600x900', '1280x800', '2560x1440', '3840x2160'
        ]

        # Browser languages
        self.browser_languages = [
            'en-US', 'en-GB', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR',
            'ja-JP', 'ko-KR', 'zh-CN', 'ru-RU', 'ar-SA'
        ]

        # Automation signatures (for fraud detection)
        self.automation_signatures = [
            'phantomjs', 'selenium', 'headless_chrome', 'puppeteer',
            'webdriver', 'chromedriver', 'automation'
        ]

    def generate_screen_resolution(self) -> str:
        """Generate realistic screen resolution."""
        if not self.config.get('generate_screen_resolution', True):
            return None

        return random.choice(self.screen_resolutions)

    def generate_browser_language(self, customer_location: str = 'US') -> str:
        """Generate browser language based on customer location."""
        if not self.config.get('generate_browser_language', True):
            return None

        # Weight languages based on location
        if customer_location == 'US':
            return random.choices(
                ['en-US', 'es-ES', 'en-GB'],
                weights=[0.8, 0.15, 0.05]
            )[0]
        else:
            return random.choice(self.browser_languages)

    def generate_session_duration(self, is_fraud: bool = False) -> int:
        """Generate session duration in seconds."""
        if not self.config.get('generate_session_duration', True):
            return None

        if is_fraud:
            # Fraudulent sessions tend to be shorter and more focused
            return random.randint(30, 300)  # 30 seconds to 5 minutes
        else:
            # Legitimate sessions vary widely
            return random.randint(120, 1800)  # 2 minutes to 30 minutes

    def generate_automation_signatures(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate automation detection signatures."""
        if not self.config.get('generate_automation_signatures', True):
            return {}

        signatures = {}

        # Fraud more likely to use automation
        if is_fraud and random.random() < 0.4:
            signatures['automation_detected'] = True
            signatures['automation_signature'] = random.choice(self.automation_signatures)
            signatures['webdriver_present'] = True
        else:
            signatures['automation_detected'] = False
            signatures['webdriver_present'] = random.random() < 0.02  # 2% false positive rate

        # Additional automation indicators
        signatures['plugins_count'] = random.randint(0, 15) if not signatures.get('automation_detected') else 0
        signatures['fonts_count'] = random.randint(20, 200) if not signatures.get('automation_detected') else random.randint(5, 30)

        return signatures


class NetworkMetadataGenerator:
    """Generate network and session metadata."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('network_metadata', {})
        self.enabled = self.config.get('enabled', True)

        # VPN/Proxy providers (for detection)
        self.vpn_providers = [
            'nordvpn', 'expressvpn', 'surfshark', 'cyberghost', 'protonvpn',
            'tunnelbear', 'hotspot_shield', 'pia', 'windscribe'
        ]

        # Connection types
        self.connection_types = [
            'broadband', 'mobile', 'satellite', 'corporate', 'university', 'government'
        ]

        # ISP providers
        self.isp_providers = [
            'comcast', 'verizon', 'att', 'spectrum', 'cox', 't-mobile', 'sprint'
        ]

    def generate_vpn_detection(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate VPN detection results."""
        if not self.config.get('generate_vpn_detection', True):
            return {}

        # Fraud more likely to use VPN
        vpn_probability = 0.3 if is_fraud else 0.05
        using_vpn = random.random() < vpn_probability

        result = {
            'vpn_detected': using_vpn,
            'vpn_confidence_score': random.uniform(0.8, 0.99) if using_vpn else random.uniform(0.0, 0.2)
        }

        if using_vpn:
            result['vpn_provider'] = random.choice(self.vpn_providers)
            result['exit_country'] = fake.country_code()

        return result

    def generate_proxy_detection(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate proxy detection results."""
        if not self.config.get('generate_proxy_detection', True):
            return {}

        # Fraud more likely to use proxy
        proxy_probability = 0.2 if is_fraud else 0.02
        using_proxy = random.random() < proxy_probability

        return {
            'proxy_detected': using_proxy,
            'proxy_type': random.choice(['http', 'socks4', 'socks5', 'transparent']) if using_proxy else None,
            'proxy_confidence_score': random.uniform(0.7, 0.95) if using_proxy else random.uniform(0.0, 0.15)
        }

    def generate_ip_geolocation(self, customer_location: Dict[str, str]) -> Dict[str, Any]:
        """Generate IP geolocation data."""
        if not self.config.get('generate_ip_geolocation', True):
            return {}

        # Usually matches customer location, but fraud may not
        return {
            'ip_country': customer_location.get('country', 'US'),
            'ip_region': customer_location.get('state', fake.state()),
            'ip_city': customer_location.get('city', fake.city()),
            'ip_latitude': round(random.uniform(25.0, 49.0), 6),
            'ip_longitude': round(random.uniform(-125.0, -66.0), 6),
            'ip_accuracy_radius_km': random.randint(1, 100)
        }

    def generate_connection_type(self) -> Dict[str, Any]:
        """Generate connection type information."""
        if not self.config.get('generate_connection_type', True):
            return {}

        connection_type = random.choices(
            self.connection_types,
            weights=[0.6, 0.25, 0.05, 0.05, 0.03, 0.02]
        )[0]

        return {
            'connection_type': connection_type,
            'isp_provider': random.choice(self.isp_providers),
            'connection_speed_mbps': random.randint(5, 1000)
        }


class TransactionFlowMetadataGenerator:
    """Generate transaction flow and session context metadata."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('transaction_flow', {})
        self.enabled = self.config.get('enabled', True)

    def generate_cart_abandonment_data(self, customer_history: List[Dict]) -> Dict[str, Any]:
        """Generate cart abandonment history."""
        if not self.config.get('generate_cart_abandonment', True):
            return {}

        # Analyze customer's abandonment patterns
        abandonment_count = random.randint(0, 5)

        return {
            'cart_abandonment_count_30d': abandonment_count,
            'last_abandonment_days_ago': random.randint(1, 30) if abandonment_count > 0 else None,
            'average_abandoned_cart_value': round(random.uniform(50, 300), 2) if abandonment_count > 0 else None
        }

    def generate_checkout_duration(self, is_fraud: bool = False) -> int:
        """Generate checkout duration in seconds."""
        if not self.config.get('generate_checkout_duration', True):
            return None

        if is_fraud:
            # Fraudulent checkouts tend to be faster (automated) or much slower (manual testing)
            if random.random() < 0.6:
                return random.randint(10, 45)  # Very fast (automated)
            else:
                return random.randint(600, 1800)  # Very slow (manual testing)
        else:
            # Normal checkout times
            return random.randint(60, 300)  # 1-5 minutes

    def generate_payment_attempts(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate payment attempt history."""
        if not self.config.get('generate_payment_attempts', True):
            return {}

        if is_fraud:
            # Fraud often involves multiple payment attempts
            attempts = random.randint(2, 8)
            failed_attempts = random.randint(1, attempts - 1)
        else:
            # Legitimate customers usually succeed on first try
            attempts = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
            failed_attempts = max(0, attempts - 1)

        return {
            'payment_attempts_count': attempts,
            'failed_payment_attempts': failed_attempts,
            'payment_retry_delay_seconds': random.randint(5, 120) if failed_attempts > 0 else None
        }

    def generate_session_context(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate broader session context."""
        if not self.config.get('generate_session_context', True):
            return {}

        # Page views and interaction patterns
        if is_fraud:
            page_views = random.randint(1, 5)  # Focused on target
            time_on_site = random.randint(30, 300)  # Quick in and out
        else:
            page_views = random.randint(3, 20)  # Browse around
            time_on_site = random.randint(300, 3600)  # Longer browsing

        return {
            'session_page_views': page_views,
            'time_on_site_seconds': time_on_site,
            'referrer_domain': random.choice(['google.com', 'facebook.com', 'direct', 'email', 'twitter.com']),
            'utm_source': random.choice(['google', 'facebook', 'email', 'direct', None]),
            'utm_campaign': f"campaign_{random.randint(1, 100)}" if random.random() < 0.3 else None
        }


class PaymentInstrumentMetadataGenerator:
    """Generate enhanced payment instrument metadata."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('payment_instruments', {})
        self.enabled = self.config.get('enabled', True)

        # Real credit card BIN ranges (first 6 digits)
        self.bin_ranges = {
            'visa': ['424242', '400000', '401288', '411111', '444444'],
            'mastercard': ['555555', '545454', '535353', '512345', '522222'],
            'amex': ['378282', '371449', '370000', '343434'],
            'discover': ['601111', '644444', '655555']
        }

        # Issuing banks
        self.issuing_banks = [
            'chase', 'citi', 'bank_of_america', 'wells_fargo', 'capital_one',
            'american_express', 'discover', 'synchrony', 'barclays'
        ]

        # Card countries
        self.card_countries = ['US', 'CA', 'GB', 'AU', 'DE', 'FR', 'JP']

        # Funding types
        self.funding_types = ['credit', 'debit', 'prepaid']

    def generate_realistic_bin(self, card_type: str = None) -> str:
        """Generate realistic BIN number."""
        if not self.config.get('generate_realistic_bins', True):
            return None

        if not card_type:
            card_type = random.choices(
                ['visa', 'mastercard', 'amex', 'discover'],
                weights=[0.6, 0.25, 0.1, 0.05]
            )[0]

        return random.choice(self.bin_ranges[card_type])

    def generate_issuing_bank(self) -> str:
        """Generate issuing bank."""
        if not self.config.get('generate_issuing_banks', True):
            return None

        return random.choice(self.issuing_banks)

    def generate_card_country(self, customer_country: str = 'US') -> str:
        """Generate card issuing country."""
        if not self.config.get('generate_card_countries', True):
            return None

        # 90% chance card matches customer country
        if random.random() < 0.9:
            return customer_country
        else:
            return random.choice(self.card_countries)

    def generate_funding_type(self) -> str:
        """Generate card funding type."""
        if not self.config.get('generate_funding_types', True):
            return None

        return random.choices(
            self.funding_types,
            weights=[0.7, 0.25, 0.05]  # Credit most common
        )[0]

    def generate_verification_results(self, is_fraud: bool = False) -> Dict[str, Any]:
        """Generate CVV and AVS verification results."""
        if not self.config.get('generate_verification_results', True):
            return {}

        if is_fraud:
            # Fraud more likely to fail verification
            cvv_match = random.choices(['Y', 'N', 'U'], weights=[0.3, 0.6, 0.1])[0]
            avs_result = random.choices(['Y', 'N', 'A', 'Z', 'U'], weights=[0.2, 0.5, 0.1, 0.1, 0.1])[0]
        else:
            # Legitimate transactions usually pass
            cvv_match = random.choices(['Y', 'N', 'U'], weights=[0.9, 0.05, 0.05])[0]
            avs_result = random.choices(['Y', 'N', 'A', 'Z', 'U'], weights=[0.8, 0.1, 0.05, 0.03, 0.02])[0]

        return {
            'cvv_match': cvv_match,
            'avs_result': avs_result,
            'cvv_provided': random.random() < 0.95,  # 95% provide CVV
            'avs_provided': random.random() < 0.90   # 90% provide address for verification
        }


class EnhancedMetadataGenerator:
    """Main metadata generator that coordinates all sub-generators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('data_realism', {})

        # Initialize sub-generators
        self.payment_processor = PaymentProcessingMetadataGenerator(self.config)
        self.device_fingerprint = DeviceFingerprintingGenerator(self.config)
        self.network_metadata = NetworkMetadataGenerator(self.config)
        self.transaction_flow = TransactionFlowMetadataGenerator(self.config)
        self.payment_instrument = PaymentInstrumentMetadataGenerator(self.config)

    def generate_comprehensive_metadata(self, customer, merchant, transaction_data: Dict[str, Any],
                                       is_fraud: bool = False) -> Dict[str, Any]:
        """Generate comprehensive transaction metadata."""
        metadata = {}

        # Payment processing metadata
        payment_meta = self.payment_processor.generate_payment_processor_data()
        metadata.update(payment_meta)

        metadata['authorization_code'] = self.payment_processor.generate_authorization_code()
        metadata['processing_time_ms'] = self.payment_processor.generate_processing_time(is_fraud)

        # Device fingerprinting
        metadata['screen_resolution'] = self.device_fingerprint.generate_screen_resolution()
        metadata['browser_language'] = self.device_fingerprint.generate_browser_language(
            customer.billing_address.get('country', 'US')
        )
        metadata['session_duration_seconds'] = self.device_fingerprint.generate_session_duration(is_fraud)

        # Automation detection
        automation_data = self.device_fingerprint.generate_automation_signatures(is_fraud)
        metadata.update(automation_data)

        # Network metadata
        vpn_data = self.network_metadata.generate_vpn_detection(is_fraud)
        metadata.update(vpn_data)

        proxy_data = self.network_metadata.generate_proxy_detection(is_fraud)
        metadata.update(proxy_data)

        ip_geo_data = self.network_metadata.generate_ip_geolocation(customer.billing_address)
        metadata.update(ip_geo_data)

        connection_data = self.network_metadata.generate_connection_type()
        metadata.update(connection_data)

        # Transaction flow metadata
        cart_data = self.transaction_flow.generate_cart_abandonment_data([])
        metadata.update(cart_data)

        metadata['checkout_duration_seconds'] = self.transaction_flow.generate_checkout_duration(is_fraud)

        payment_attempts = self.transaction_flow.generate_payment_attempts(is_fraud)
        metadata.update(payment_attempts)

        session_context = self.transaction_flow.generate_session_context(is_fraud)
        metadata.update(session_context)

        # Payment instrument metadata
        metadata['card_bin'] = self.payment_instrument.generate_realistic_bin()
        metadata['issuing_bank'] = self.payment_instrument.generate_issuing_bank()
        metadata['card_country'] = self.payment_instrument.generate_card_country(
            customer.billing_address.get('country', 'US')
        )
        metadata['funding_type'] = self.payment_instrument.generate_funding_type()

        verification_data = self.payment_instrument.generate_verification_results(is_fraud)
        metadata.update(verification_data)

        return metadata
