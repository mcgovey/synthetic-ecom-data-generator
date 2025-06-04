"""Temporal pattern utilities for realistic transaction timing."""

import numpy as np
from typing import Tuple


class TemporalPatterns:
    """Realistic temporal patterns for transactions."""

    @staticmethod
    def get_hourly_weights() -> np.ndarray:
        """
        Get transaction probability by hour of day.

        Returns:
            Array of hourly weights with peaks during lunch (12-1) and evening (6-9)
        """
        # Peak during lunch (12-1) and evening (6-9)
        weights = np.array([
            0.2, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8,  # 0-7 AM
            1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.3,  # 8-15 (8 AM - 3 PM)
            1.5, 1.8, 2.2, 2.5, 2.2, 1.8, 1.2, 0.8   # 16-23 (4-11 PM)
        ])
        return weights / weights.sum()

    @staticmethod
    def get_daily_weights() -> np.ndarray:
        """
        Get transaction probability by day of week.

        Returns:
            Array of daily weights (Mon=0, Sun=6) with higher activity on weekends
        """
        # Higher on weekends and Friday
        weights = np.array([0.12, 0.12, 0.13, 0.14, 0.16, 0.18, 0.15])
        return weights / weights.sum()

    @staticmethod
    def get_seasonal_multiplier(date) -> float:
        """
        Get seasonal transaction multiplier based on date.

        Args:
            date: Date object to calculate multiplier for

        Returns:
            Seasonal multiplier (higher during holiday seasons)
        """
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

    @staticmethod
    def is_business_hours(hour: int, day_of_week: int) -> bool:
        """
        Check if given time is during typical business hours.

        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)

        Returns:
            True if during business hours
        """
        # Weekday business hours: 9 AM to 6 PM
        if 0 <= day_of_week <= 4:  # Monday to Friday
            return 9 <= hour <= 18
        # Weekend reduced hours: 10 AM to 4 PM
        else:
            return 10 <= hour <= 16
