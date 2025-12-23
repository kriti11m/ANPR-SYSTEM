"""
License Plate Normalization and Validation Utility

This module provides utilities to normalize license plate text,
remove noise, convert to uppercase, and validate Indian number plate formats.

Author: ANPR System
Date: December 23, 2025
"""

import re
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class PlateFormat(Enum):
    """Indian license plate format types."""
    STANDARD_OLD = "standard_old"  # Example: DL 01 AB 1234
    STANDARD_NEW = "standard_new"  # Example: DL01AB1234
    BHARAT_SERIES = "bharat_series"  # Example: 22 BH 1234 AB
    TEMPORARY = "temporary"  # Example: T 1234
    UNKNOWN = "unknown"


@dataclass
class PlateValidationResult:
    """Result of license plate validation."""
    is_valid: bool
    normalized_text: str
    original_text: str
    format_type: PlateFormat
    state_code: Optional[str] = None
    district_code: Optional[str] = None
    series: Optional[str] = None
    number: Optional[str] = None
    confidence_score: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class IndianPlateNormalizer:
    """
    Utility class for normalizing and validating Indian license plates.
    """
    
    # Indian state codes mapping
    STATE_CODES = {
        'AN': 'Andaman and Nicobar Islands',
        'AP': 'Andhra Pradesh',
        'AR': 'Arunachal Pradesh',
        'AS': 'Assam',
        'BR': 'Bihar',
        'CH': 'Chandigarh',
        'CG': 'Chhattisgarh',
        'DN': 'Dadra and Nagar Haveli',
        'DD': 'Daman and Diu',
        'DL': 'Delhi',
        'GA': 'Goa',
        'GJ': 'Gujarat',
        'HR': 'Haryana',
        'HP': 'Himachal Pradesh',
        'JK': 'Jammu and Kashmir',
        'JH': 'Jharkhand',
        'KA': 'Karnataka',
        'KL': 'Kerala',
        'LD': 'Lakshadweep',
        'MP': 'Madhya Pradesh',
        'MH': 'Maharashtra',
        'MN': 'Manipur',
        'ML': 'Meghalaya',
        'MZ': 'Mizoram',
        'NL': 'Nagaland',
        'OR': 'Odisha',
        'PY': 'Puducherry',
        'PB': 'Punjab',
        'RJ': 'Rajasthan',
        'SK': 'Sikkim',
        'TN': 'Tamil Nadu',
        'TG': 'Telangana',
        'TR': 'Tripura',
        'UP': 'Uttar Pradesh',
        'UK': 'Uttarakhand',
        'WB': 'West Bengal',
        'BH': 'Bharat Series'
    }
    
    # Common OCR misreadings and their corrections
    OCR_CORRECTIONS = {
        '0': 'O',  # Zero to O in state codes
        '1': 'I',  # One to I in state codes
        '5': 'S',  # Five to S in state codes
        '8': 'B',  # Eight to B in state codes
        'Q': '0',  # Q to zero in numbers
        'I': '1',  # I to one in numbers
        'O': '0',  # O to zero in numbers
        'S': '5',  # S to five in numbers
        'Z': '2',  # Z to two in numbers
        'G': '6',  # G to six in numbers
    }
    
    def __init__(self):
        """Initialize the plate normalizer."""
        self._compile_regex_patterns()
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for different Indian plate formats."""
        
        # Standard old format: XX 00 XX 0000 (e.g., DL 01 AB 1234)
        self.old_format_pattern = re.compile(
            r'^([A-Z]{2})\s*([0-9]{2})\s*([A-Z]{1,2})\s*([0-9]{1,4})$',
            re.IGNORECASE
        )
        
        # Standard new format: XX00XX0000 (e.g., DL01AB1234)
        self.new_format_pattern = re.compile(
            r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{1,4})$',
            re.IGNORECASE
        )
        
        # Bharat Series: 00 BH 0000 XX (e.g., 22 BH 1234 AB)
        self.bharat_format_pattern = re.compile(
            r'^([0-9]{2})\s*BH\s*([0-9]{1,4})\s*([A-Z]{1,2})$',
            re.IGNORECASE
        )
        
        # Temporary format: T 0000 (e.g., T 1234)
        self.temp_format_pattern = re.compile(
            r'^T\s*([0-9]{1,4})$',
            re.IGNORECASE
        )
    
    def remove_noise(self, text: str) -> str:
        """
        Remove noise characters from license plate text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text with noise removed
        """
        if not text:
            return ""
        
        # Remove common noise characters
        noise_chars = r'[^\w\s]'  # Keep only alphanumeric and spaces
        cleaned = re.sub(noise_chars, '', text)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove leading/trailing dots, hyphens, etc.
        cleaned = re.sub(r'^[.\-_\s]+|[.\-_\s]+$', '', cleaned)
        
        return cleaned
    
    def apply_ocr_corrections(self, text: str, context: str = "mixed") -> str:
        """
        Apply common OCR error corrections.
        
        Args:
            text: Text to correct
            context: Context for corrections ("letters", "numbers", "mixed")
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        corrected = text
        
        # Apply context-specific corrections
        if context == "letters":
            # When we expect letters, correct common number->letter mistakes
            corrections = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
            for wrong, right in corrections.items():
                corrected = corrected.replace(wrong, right)
                
        elif context == "numbers":
            # When we expect numbers, correct common letter->number mistakes
            corrections = {'Q': '0', 'O': '0', 'I': '1', 'l': '1', 'S': '5', 'Z': '2', 'G': '6'}
            for wrong, right in corrections.items():
                corrected = corrected.replace(wrong, right)
        
        return corrected
    
    def normalize_spacing(self, text: str) -> str:
        """
        Normalize spacing in license plate text.
        
        Args:
            text: Text with irregular spacing
            
        Returns:
            Text with normalized spacing
        """
        if not text:
            return ""
        
        # Remove all spaces first
        no_spaces = re.sub(r'\s', '', text)
        
        # Add spaces according to Indian format patterns
        # Try to match and format different patterns
        
        # Check for standard format: XXNNXNNNN -> XX NN X NNNN
        match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{1,4})$', no_spaces, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
        
        # Check for Bharat series: NNBHNNNNXX -> NN BH NNNN XX
        match = re.match(r'^([0-9]{2})BH([0-9]{1,4})([A-Z]{1,2})$', no_spaces, re.IGNORECASE)
        if match:
            return f"{match.group(1)} BH {match.group(2)} {match.group(3)}"
        
        # If no pattern matches, return with minimal spacing
        return no_spaces
    
    def validate_state_code(self, state_code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Indian state code.
        
        Args:
            state_code: Two-letter state code
            
        Returns:
            Tuple of (is_valid, state_name)
        """
        if not state_code or len(state_code) != 2:
            return False, None
        
        state_code = state_code.upper()
        if state_code in self.STATE_CODES:
            return True, self.STATE_CODES[state_code]
        
        return False, None
    
    def validate_district_code(self, district_code: str) -> bool:
        """
        Validate district code (should be 2 digits, 01-99).
        
        Args:
            district_code: District code string
            
        Returns:
            True if valid district code
        """
        if not district_code:
            return False
        
        try:
            code_int = int(district_code)
            return 1 <= code_int <= 99 and len(district_code) == 2
        except ValueError:
            return False
    
    def detect_format(self, text: str) -> PlateFormat:
        """
        Detect the format of Indian license plate.
        
        Args:
            text: Normalized license plate text
            
        Returns:
            Detected format type
        """
        if not text:
            return PlateFormat.UNKNOWN
        
        # Remove spaces for pattern matching
        no_space_text = re.sub(r'\s', '', text)
        
        # Check Bharat series first (as it has unique BH identifier)
        if self.bharat_format_pattern.match(text) or re.search(r'BH', text, re.IGNORECASE):
            return PlateFormat.BHARAT_SERIES
        
        # Check temporary format
        if self.temp_format_pattern.match(text):
            return PlateFormat.TEMPORARY
        
        # Check new format (no spaces)
        if self.new_format_pattern.match(no_space_text):
            return PlateFormat.STANDARD_NEW
        
        # Check old format (with spaces)
        if self.old_format_pattern.match(text):
            return PlateFormat.STANDARD_OLD
        
        return PlateFormat.UNKNOWN
    
    def parse_plate_components(self, text: str, format_type: PlateFormat) -> Dict[str, str]:
        """
        Parse license plate into components based on format.
        
        Args:
            text: Normalized license plate text
            format_type: Detected format type
            
        Returns:
            Dictionary with parsed components
        """
        components = {}
        
        if format_type == PlateFormat.STANDARD_OLD:
            match = self.old_format_pattern.match(text)
            if match:
                components = {
                    'state_code': match.group(1).upper(),
                    'district_code': match.group(2),
                    'series': match.group(3).upper(),
                    'number': match.group(4)
                }
        
        elif format_type == PlateFormat.STANDARD_NEW:
            no_space_text = re.sub(r'\s', '', text)
            match = self.new_format_pattern.match(no_space_text)
            if match:
                components = {
                    'state_code': match.group(1).upper(),
                    'district_code': match.group(2),
                    'series': match.group(3).upper(),
                    'number': match.group(4)
                }
        
        elif format_type == PlateFormat.BHARAT_SERIES:
            match = self.bharat_format_pattern.match(text)
            if match:
                components = {
                    'state_code': 'BH',
                    'district_code': match.group(1),
                    'series': match.group(3).upper(),
                    'number': match.group(2)
                }
        
        elif format_type == PlateFormat.TEMPORARY:
            match = self.temp_format_pattern.match(text)
            if match:
                components = {
                    'state_code': 'T',
                    'district_code': None,
                    'series': None,
                    'number': match.group(1)
                }
        
        return components
    
    def calculate_confidence_score(self, text: str, validation_result: PlateValidationResult) -> float:
        """
        Calculate confidence score for the normalized plate.
        
        Args:
            text: Original text
            validation_result: Validation result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not validation_result.is_valid:
            return 0.0
        
        score = 0.0
        
        # Base score for valid format
        score += 0.4
        
        # Bonus for valid state code
        if validation_result.state_code and validation_result.state_code in self.STATE_CODES:
            score += 0.2
        
        # Bonus for valid district code
        if validation_result.district_code and self.validate_district_code(validation_result.district_code):
            score += 0.1
        
        # Bonus for proper length
        normalized_length = len(validation_result.normalized_text.replace(' ', ''))
        if 6 <= normalized_length <= 10:
            score += 0.1
        
        # Bonus for alphanumeric characters only
        if re.match(r'^[A-Z0-9\s]+$', validation_result.normalized_text):
            score += 0.1
        
        # Penalty for too many corrections
        original_clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        normalized_clean = re.sub(r'[^A-Z0-9]', '', validation_result.normalized_text.upper())
        
        if original_clean and normalized_clean:
            similarity = len(set(original_clean) & set(normalized_clean)) / len(set(original_clean + normalized_clean))
            score += similarity * 0.1
        
        return min(score, 1.0)
    
    def normalize_and_validate(self, text: str) -> PlateValidationResult:
        """
        Complete normalization and validation of license plate text.
        
        Args:
            text: Raw license plate text from OCR
            
        Returns:
            PlateValidationResult with all validation details
        """
        original_text = text or ""
        errors = []
        
        if not text or not text.strip():
            return PlateValidationResult(
                is_valid=False,
                normalized_text="",
                original_text=original_text,
                format_type=PlateFormat.UNKNOWN,
                errors=["Empty or null input text"]
            )
        
        # Step 1: Remove noise
        cleaned = self.remove_noise(text)
        if not cleaned:
            return PlateValidationResult(
                is_valid=False,
                normalized_text="",
                original_text=original_text,
                format_type=PlateFormat.UNKNOWN,
                errors=["Text became empty after noise removal"]
            )
        
        # Step 2: Convert to uppercase
        upper_text = cleaned.upper()
        
        # Step 3: Normalize spacing
        normalized = self.normalize_spacing(upper_text)
        
        # Step 4: Detect format
        format_type = self.detect_format(normalized)
        
        if format_type == PlateFormat.UNKNOWN:
            errors.append("Unknown or invalid license plate format")
        
        # Step 5: Parse components
        components = self.parse_plate_components(normalized, format_type)
        
        # Step 6: Validate components
        is_valid = True
        
        if format_type != PlateFormat.UNKNOWN and components:
            # Validate state code
            state_code = components.get('state_code')
            if state_code:
                valid_state, _ = self.validate_state_code(state_code)
                if not valid_state:
                    errors.append(f"Invalid state code: {state_code}")
                    is_valid = False
            
            # Validate district code
            district_code = components.get('district_code')
            if district_code and not self.validate_district_code(district_code):
                errors.append(f"Invalid district code: {district_code}")
                is_valid = False
        else:
            is_valid = False
        
        # Create result
        result = PlateValidationResult(
            is_valid=is_valid,
            normalized_text=normalized,
            original_text=original_text,
            format_type=format_type,
            state_code=components.get('state_code'),
            district_code=components.get('district_code'),
            series=components.get('series'),
            number=components.get('number'),
            errors=errors
        )
        
        # Step 7: Calculate confidence score
        result.confidence_score = self.calculate_confidence_score(original_text, result)
        
        return result


def normalize_license_plate(text: str) -> PlateValidationResult:
    """
    Convenience function to normalize and validate license plate text.
    
    Args:
        text: Raw license plate text from OCR
        
    Returns:
        PlateValidationResult with normalized and validated plate information
        
    Example:
        >>> result = normalize_license_plate("dl01ab1234")
        >>> print(f"Valid: {result.is_valid}")
        >>> print(f"Normalized: {result.normalized_text}")
        >>> print(f"State: {result.state_code}")
    """
    normalizer = IndianPlateNormalizer()
    return normalizer.normalize_and_validate(text)


# Additional utility functions
def is_valid_indian_plate(text: str) -> bool:
    """
    Quick check if text is a valid Indian license plate.
    
    Args:
        text: License plate text to validate
        
    Returns:
        True if valid Indian license plate format
    """
    result = normalize_license_plate(text)
    return result.is_valid


def get_state_name(state_code: str) -> Optional[str]:
    """
    Get full state name from state code.
    
    Args:
        state_code: Two-letter state code
        
    Returns:
        Full state name or None if invalid
    """
    normalizer = IndianPlateNormalizer()
    is_valid, state_name = normalizer.validate_state_code(state_code)
    return state_name if is_valid else None


def format_plate_display(text: str) -> str:
    """
    Format license plate for display with proper spacing.
    
    Args:
        text: License plate text
        
    Returns:
        Formatted plate text for display
    """
    result = normalize_license_plate(text)
    return result.normalized_text if result.is_valid else text.upper()


if __name__ == "__main__":
    # Test the utility functions
    test_plates = [
        "DL01AB1234",
        "dl 01 ab 1234",
        "22BH1234AB",
        "22 BH 1234 AB",
        "MH12DE3456",
        "T1234",
        "INVALID123",
        "DL@01#AB$1234",
        "dl01ab12345"  # Too long
    ]
    
    print("🚗 Indian License Plate Normalization Test")
    print("=" * 50)
    
    for plate in test_plates:
        result = normalize_license_plate(plate)
        
        print(f"\n📄 Original: '{plate}'")
        print(f"✅ Valid: {result.is_valid}")
        print(f"🔤 Normalized: '{result.normalized_text}'")
        print(f"🏷️  Format: {result.format_type.value}")
        print(f"🏛️  State: {result.state_code}")
        print(f"📊 Confidence: {result.confidence_score:.3f}")
        
        if result.errors:
            print(f"❌ Errors: {', '.join(result.errors)}")
