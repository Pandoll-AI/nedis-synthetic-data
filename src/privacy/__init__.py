"""
Privacy Enhancement Module for NEDIS Synthetic Data

This module provides privacy-preserving transformations including:
- Identifier management and anonymization
- K-anonymity and l-diversity enforcement
- Differential privacy mechanisms
- Age and geographic generalization
"""

from .identifier_manager import IdentifierManager
from .generalization import AgeGeneralizer, GeographicGeneralizer
from .k_anonymity import KAnonymityValidator, KAnonymityEnforcer
from .differential_privacy import DifferentialPrivacy
from .privacy_validator import PrivacyValidator

__all__ = [
    'IdentifierManager',
    'AgeGeneralizer',
    'GeographicGeneralizer',
    'KAnonymityValidator',
    'KAnonymityEnforcer',
    'DifferentialPrivacy',
    'PrivacyValidator'
]