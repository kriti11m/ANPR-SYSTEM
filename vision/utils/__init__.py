"""
Vision utilities package for ANPR system.

This package contains utility functions and classes for:
- License plate normalization and validation
- Text processing and cleaning
- Indian number plate format validation
- OCR result processing

Available modules:
- plate_normalizer: Indian license plate normalization and validation
"""

from .plate_normalizer import (
    normalize_license_plate,
    is_valid_indian_plate,
    get_state_name,
    format_plate_display,
    IndianPlateNormalizer,
    PlateValidationResult,
    PlateFormat
)

__all__ = [
    'normalize_license_plate',
    'is_valid_indian_plate', 
    'get_state_name',
    'format_plate_display',
    'IndianPlateNormalizer',
    'PlateValidationResult',
    'PlateFormat'
]

__version__ = "1.0.0"
__author__ = "ANPR System"
