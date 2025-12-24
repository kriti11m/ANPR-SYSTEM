"""
OCR Pre-Normalization Repair Module

This module provides intelligent OCR error correction based on position context:
- O → 0 only in numeric zones (district codes, numbers)
- I → 1 only in district/number zones  
- Confusion matrix-based correction for common OCR errors
- Position-aware corrections to avoid false positives

Author: AI Assistant
Date: December 2025
"""

import re
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class OCRRepairEngine:
    """
    Intelligent OCR error correction engine that applies context-aware fixes
    to improve license plate recognition accuracy.
    """
    
    def __init__(self):
        """Initialize the OCR repair engine with confusion matrices"""
        
        # Common OCR confusion pairs (OCR_char → Correct_char)
        self.confusion_matrix = {
            # Numbers that look like letters
            'O': '0',  # O → 0 in numeric contexts
            'o': '0',  # lowercase o → 0  
            'I': '1',  # I → 1 in numeric contexts
            'l': '1',  # lowercase l → 1
            'S': '5',  # S → 5 in numeric contexts
            'Z': '2',  # Z → 2 in numeric contexts
            'G': '6',  # G → 6 in numeric contexts
            'B': '8',  # B → 8 in numeric contexts
            
            # Letters that look like numbers
            '0': 'O',  # 0 → O in letter contexts (rarely used)
            '1': 'I',  # 1 → I in letter contexts (rarely used)
            '5': 'S',  # 5 → S in letter contexts (rarely used)
            '8': 'B',  # 8 → B in letter contexts (rarely used)
        }
        
        # Indian license plate patterns for context detection
        self.patterns = {
            'standard_new': re.compile(r'^([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})$'),
            'standard_spaced': re.compile(r'^([A-Z]{2})\s*(\d{2})\s*([A-Z]{2})\s*(\d{4})$'),
            'bharat_series': re.compile(r'^(\d{2})\s*BH\s*(\d{4})\s*([A-Z]{2})$'),
        }
        
        # Valid Indian state codes for validation
        self.valid_state_codes = {
            'AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK',
            'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ',
            'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DD', 'DL',
            'LD', 'PY'
        }
    
    def repair_ocr_errors(self, text: str) -> List[str]:
        """
        Apply intelligent OCR error correction to generate repair candidates.
        
        Args:
            text: Raw OCR text that may contain errors
            
        Returns:
            List of corrected text candidates, ordered by confidence
        """
        logger.debug(f"Repairing OCR errors in: '{text}'")
        
        # Clean input text
        cleaned_text = self._clean_input(text)
        
        # Generate repair candidates
        candidates = []
        
        # Original text (cleaned)
        candidates.append(cleaned_text)
        
        # Apply position-aware corrections
        position_corrected = self._apply_position_aware_corrections(cleaned_text)
        if position_corrected != cleaned_text:
            candidates.append(position_corrected)
        
        # Apply pattern-based corrections
        pattern_candidates = self._apply_pattern_based_corrections(cleaned_text)
        candidates.extend(pattern_candidates)
        
        # Apply aggressive corrections (for very noisy inputs)
        aggressive_candidates = self._apply_aggressive_corrections(cleaned_text)
        candidates.extend(aggressive_candidates)
        
        # Remove duplicates while preserving order
        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate not in seen:
                unique_candidates.append(candidate)
                seen.add(candidate)
        
        logger.debug(f"Generated {len(unique_candidates)} repair candidates")
        return unique_candidates[:6]  # Limit to top 6 candidates
    
    def _clean_input(self, text: str) -> str:
        """Basic cleaning of input text"""
        # Remove extra spaces and convert to uppercase
        cleaned = re.sub(r'\s+', '', text.strip().upper())
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('.', '').replace(',', '').replace('-', '')
        
        return cleaned
    
    def _apply_position_aware_corrections(self, text: str) -> str:
        """
        Apply corrections based on expected position of characters in Indian plates.
        
        Indian plate format: XX ## XX ####
        - Positions 0-1: State code (letters)
        - Positions 2-3: District code (numbers)  
        - Positions 4-5: Series letters (letters)
        - Positions 6-9: Number (numbers)
        """
        if len(text) < 8:
            return text
        
        corrected = list(text)
        
        try:
            # For standard format without spaces (8-10 chars)
            if 8 <= len(text) <= 10:
                
                # State code positions (0-1): Should be letters
                for i in [0, 1]:
                    if i < len(corrected):
                        if corrected[i] in ['0', '1', '5', '8']:
                            corrected[i] = self.confusion_matrix.get(corrected[i], corrected[i])
                
                # District code positions (2-3): Should be numbers
                for i in [2, 3]:
                    if i < len(corrected):
                        if corrected[i] in ['O', 'o', 'I', 'l', 'S', 'Z', 'G', 'B']:
                            corrected[i] = self.confusion_matrix.get(corrected[i], corrected[i])
                
                # Series positions (4-5): Should be letters  
                for i in [4, 5]:
                    if i < len(corrected):
                        if corrected[i] in ['0', '1', '5', '8']:
                            corrected[i] = self.confusion_matrix.get(corrected[i], corrected[i])
                
                # Number positions (6+): Should be numbers
                for i in range(6, len(corrected)):
                    if corrected[i] in ['O', 'o', 'I', 'l', 'S', 'Z', 'G', 'B']:
                        corrected[i] = self.confusion_matrix.get(corrected[i], corrected[i])
        
        except Exception as e:
            logger.warning(f"Error in position-aware correction: {e}")
            return text
        
        return ''.join(corrected)
    
    def _apply_pattern_based_corrections(self, text: str) -> List[str]:
        """Apply corrections based on known Indian license plate patterns"""
        candidates = []
        
        # Try to match against known patterns and fix accordingly
        for pattern_name, pattern in self.patterns.items():
            
            # Try direct match first
            if pattern.match(text):
                continue  # Already matches, no correction needed
            
            # Try with common substitutions
            test_text = text
            
            # Apply O→0 and I→1 corrections
            test_text = test_text.replace('O', '0').replace('o', '0')
            test_text = test_text.replace('I', '1').replace('l', '1')
            
            if pattern.match(test_text) and test_text != text:
                candidates.append(test_text)
            
            # Try with partial corrections (state code area)
            if len(text) >= 2:
                state_corrected = text
                # Fix common state code errors
                if text[:2] in ['DLO', 'DL0']:
                    state_corrected = 'DL' + text[2:]
                elif text[:2] in ['MHO', 'MH0']:
                    state_corrected = 'MH' + text[2:]
                elif text[:2] in ['KAO', 'KA0']:
                    state_corrected = 'KA' + text[2:]
                
                if state_corrected != text:
                    candidates.append(state_corrected)
        
        return candidates
    
    def _apply_aggressive_corrections(self, text: str) -> List[str]:
        """Apply more aggressive corrections for very noisy inputs"""
        candidates = []
        
        # Apply all possible O→0, I→1 corrections
        aggressive = text
        for ocr_char, correct_char in self.confusion_matrix.items():
            if ocr_char in ['O', 'o', 'I', 'l']:  # Focus on most common errors
                test_candidate = aggressive.replace(ocr_char, correct_char)
                if test_candidate != aggressive:
                    candidates.append(test_candidate)
                    aggressive = test_candidate
        
        # Try fixing common state code corruptions
        common_state_fixes = {
            'DLO': 'DL0', 'DL0': 'DL0',
            'MHO': 'MH0', 'MH0': 'MH0', 
            'KAO': 'KA0', 'KA0': 'KA0',
            'UPO': 'UP0', 'UP0': 'UP0',
        }
        
        for wrong, right in common_state_fixes.items():
            if text.startswith(wrong):
                fixed = right + text[len(wrong):]
                candidates.append(fixed)
        
        return candidates
    
    def get_repair_confidence(self, original: str, repaired: str) -> float:
        """
        Calculate confidence score for a repair operation.
        
        Returns:
            Float between 0.0 and 1.0 indicating repair confidence
        """
        if original == repaired:
            return 1.0  # No changes needed
        
        # Calculate edit distance
        edit_distance = self._calculate_edit_distance(original, repaired)
        
        # Penalize for too many changes
        if edit_distance > len(original) * 0.3:  # More than 30% changed
            return 0.3
        
        # Check if repaired text matches known patterns
        pattern_bonus = 0.0
        for pattern in self.patterns.values():
            if pattern.match(repaired):
                pattern_bonus = 0.4
                break
        
        # Base confidence starts high for small changes
        base_confidence = max(0.5, 1.0 - (edit_distance / len(original)))
        
        return min(1.0, base_confidence + pattern_bonus)
    
    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        return prev_row[-1]


# Convenience function for easy integration
def repair_ocr_text(text: str) -> Tuple[str, float]:
    """
    Convenience function to repair OCR errors in license plate text.
    
    Args:
        text: Raw OCR text that may contain errors
        
    Returns:
        Tuple of (best_repaired_text, confidence_score)
    """
    from vision.utils.plate_normalizer import normalize_license_plate
    
    engine = OCRRepairEngine()
    candidates = engine.repair_ocr_errors(text)
    
    if not candidates:
        return text, 0.5  # Return original if no candidates
    
    # Find the best candidate that can be normalized
    best_candidate = text
    best_score = 0.0
    
    for candidate in candidates:
        # Calculate repair confidence
        repair_confidence = engine.get_repair_confidence(text, candidate)
        
        # Test if candidate can be normalized
        try:
            norm_result = normalize_license_plate(candidate)
            normalization_success = 1.0 if norm_result.is_valid else 0.0
            validation_confidence = norm_result.confidence_score if norm_result.is_valid else 0.0
        except:
            normalization_success = 0.0
            validation_confidence = 0.0
        
        # Combined score: repair confidence + normalization bonus
        combined_score = (repair_confidence * 0.4) + (validation_confidence * 0.3) + (normalization_success * 0.3)
        
        if combined_score > best_score:
            best_candidate = candidate
            best_score = combined_score
    
    return best_candidate, best_score


# Integration example
def enhanced_normalize_with_repair(text: str):
    """
    Enhanced normalization that includes OCR repair before validation.
    This is what should be used in the main ANPR pipeline.
    """
    from vision.utils.plate_normalizer import normalize_license_plate
    
    # Generate repair candidates
    repair_engine = OCRRepairEngine()
    candidates = repair_engine.repair_ocr_errors(text)
    
    best_result = None
    best_score = 0.0
    
    for candidate in candidates:
        # Try normalization on repaired text
        norm_result = normalize_license_plate(candidate)
        
        # Calculate combined score
        repair_confidence = repair_engine.get_repair_confidence(text, candidate)
        validation_confidence = norm_result.confidence_score if norm_result.is_valid else 0.0
        
        combined_score = (repair_confidence * 0.4) + (validation_confidence * 0.6)
        
        if norm_result.is_valid:
            combined_score += 0.3  # Bonus for valid plates
        
        if combined_score > best_score:
            best_result = norm_result
            best_score = combined_score
            best_result.original_text = text
            best_result.repaired_text = candidate
    
    return best_result, best_score


if __name__ == "__main__":
    # Test the OCR repair engine
    engine = OCRRepairEngine()
    
    test_cases = [
        "DLO1AB1234",   # O → 0
        "MHI2DE3456",   # I → 1  
        "DLOI AB I234", # Multiple errors
        "KAOSBC789O",   # Very noisy
        "DL01AB1234",   # Already clean
    ]
    
    print("🔧 OCR REPAIR ENGINE TEST")
    print("=" * 50)
    
    for test_text in test_cases:
        print(f"\n📥 Input: '{test_text}'")
        candidates = engine.repair_ocr_errors(test_text)
        
        for i, candidate in enumerate(candidates):
            confidence = engine.get_repair_confidence(test_text, candidate)
            print(f"   {i+1}. '{candidate}' (confidence: {confidence:.3f})")
