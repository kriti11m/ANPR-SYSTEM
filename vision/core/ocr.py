"""
OCR (Optical Character Recognition) module for license plate text extraction.

This module provides multiple OCR engines for extracting text from cropped license plate images:
- Tesseract OCR (traditional, fast)
- EasyOCR (deep learning based, accurate)
- PaddleOCR (multilingual, robust)

Features:
- Multiple OCR engine support with fallback
- Image preprocessing for better OCR accuracy
- Text cleaning and validation
- License plate format validation
- Performance monitoring and statistics
"""

import re
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# Import our advanced normalization utility
from ..utils.plate_normalizer import normalize_license_plate, PlateValidationResult

# OCR engine imports (with graceful fallback)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logging.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddleocr")


class OCREngine(Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AUTO = "auto"  # Try all available engines


@dataclass
class OCRResult:
    """OCR extraction result."""
    text: str                           # Raw extracted text
    cleaned_text: str                   # Cleaned alphanumeric text
    confidence: float                   # Confidence score (0-1)
    engine_used: str                    # OCR engine that produced this result
    processing_time: float              # Time taken for OCR processing
    valid_plate: bool                   # Whether text matches plate patterns
    bounding_boxes: List[Tuple] = None  # Character-level bounding boxes (if available)


class LicensePlateOCR:
    """
    License Plate OCR processor with multiple engine support.
    
    Features:
    - Multiple OCR engines (Tesseract, EasyOCR, PaddleOCR)
    - Image preprocessing for better accuracy
    - Text cleaning and validation
    - Performance monitoring
    """
    
    def __init__(
        self,
        engines: List[OCREngine] = None,
        tesseract_config: str = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        confidence_threshold: float = 0.6,
        enable_preprocessing: bool = True
    ):
        """
        Initialize OCR processor.
        
        Args:
            engines: List of OCR engines to use (default: all available)
            tesseract_config: Tesseract configuration string
            confidence_threshold: Minimum confidence threshold for results
            enable_preprocessing: Whether to apply image preprocessing
        """
        self.tesseract_config = tesseract_config
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize available engines
        self.available_engines = []
        if engines is None:
            engines = [OCREngine.AUTO]
            
        if OCREngine.AUTO in engines:
            # Use all available engines
            if TESSERACT_AVAILABLE:
                self.available_engines.append(OCREngine.TESSERACT)
            if EASYOCR_AVAILABLE:
                self.available_engines.append(OCREngine.EASYOCR)
            if PADDLEOCR_AVAILABLE:
                self.available_engines.append(OCREngine.PADDLEOCR)
        else:
            for engine in engines:
                if engine == OCREngine.TESSERACT and TESSERACT_AVAILABLE:
                    self.available_engines.append(engine)
                elif engine == OCREngine.EASYOCR and EASYOCR_AVAILABLE:
                    self.available_engines.append(engine)
                elif engine == OCREngine.PADDLEOCR and PADDLEOCR_AVAILABLE:
                    self.available_engines.append(engine)
        
        # Initialize OCR engines
        self._init_engines()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'engine_usage': {engine.value: 0 for engine in self.available_engines},
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0
        }
        
        # License plate patterns (can be extended for different countries)
        self.plate_patterns = [
            r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$',  # ABC123, AB1234C, etc.
            r'^\d{1,3}[A-Z]{1,4}\d{0,3}$',    # 123ABC, 12AB34, etc.
            r'^[A-Z]\d{1,3}[A-Z]{1,3}$',      # A123BC, B12CD, etc.
            r'^[A-Z]{2}\d{2}[A-Z]{3}$',       # AB12CDE
            r'^[A-Z]{3}\d{3,4}$',             # ABC123, XYZ1234
            r'^\d{3}[A-Z]{3}$',               # 123ABC
        ]
        
        logging.info(f"LicensePlateOCR initialized with engines: {[e.value for e in self.available_engines]}")
    
    def _init_engines(self):
        """Initialize OCR engines."""
        self.easyocr_reader = None
        self.paddleocr_reader = None
        
        # Initialize EasyOCR
        if OCREngine.EASYOCR in self.available_engines:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)  # CPU mode for compatibility
                logging.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize EasyOCR: {e}")
                self.available_engines = [e for e in self.available_engines if e != OCREngine.EASYOCR]
        
        # Initialize PaddleOCR
        if OCREngine.PADDLEOCR in self.available_engines:
            try:
                self.paddleocr_reader = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                logging.info("PaddleOCR reader initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize PaddleOCR: {e}")
                self.available_engines = [e for e in self.available_engines if e != OCREngine.PADDLEOCR]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR accuracy.
        
        Args:
            image: Input license plate image (BGR format)
            
        Returns:
            Preprocessed image optimized for OCR
        """
        if not self.enable_preprocessing:
            return image
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if image is too small
        height, width = gray.shape
        if width < 100 or height < 30:
            scale_factor = max(100/width, 30/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply image enhancements
        enhanced = gray.copy()
        
        # 1. Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
        
        # 3. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 4. Sharpen the image
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # 5. Threshold to binary image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract OCR."""
        start_time = time.time()
        
        try:
            # First try image_to_string for text extraction
            raw_text = pytesseract.image_to_string(image, config=self.tesseract_config).strip()
            
            # If we got text, try to get detailed data for confidence
            confidences = []
            bounding_boxes = []
            
            if raw_text:
                try:
                    # Get detailed results with confidence
                    data = pytesseract.image_to_data(image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
                    
                    # Extract confidence values for valid text
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > 0:  # Valid detection
                            text = data['text'][i].strip()
                            if text:
                                confidences.append(int(conf))
                                bounding_boxes.append((data['left'][i], data['top'][i], 
                                                     data['width'][i], data['height'][i]))
                except Exception as e:
                    logging.debug(f"Could not get detailed Tesseract data: {e}")
            
            # Calculate average confidence
            if confidences:
                avg_confidence = np.mean(confidences) / 100.0
            elif raw_text:
                # If we have text but no confidence data, assume reasonable confidence
                avg_confidence = 0.8
            else:
                avg_confidence = 0.0
            
        except Exception as e:
            logging.error(f"Tesseract OCR failed: {e}")
            raw_text = ""
            avg_confidence = 0.0
            bounding_boxes = []
        
        processing_time = time.time() - start_time
        cleaned_text = self.clean_text(raw_text)
        
        return OCRResult(
            text=raw_text,
            cleaned_text=cleaned_text,
            confidence=avg_confidence,
            engine_used="tesseract",
            processing_time=processing_time,
            valid_plate=self.validate_plate_text(cleaned_text),
            bounding_boxes=bounding_boxes
        )
    
    def extract_text_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR."""
        start_time = time.time()
        
        try:
            results = self.easyocr_reader.readtext(image)
            
            if results:
                # Combine all detected text
                text_parts = []
                confidences = []
                bounding_boxes = []
                
                for (bbox, text, conf) in results:
                    if conf > 0.1:  # Filter very low confidence
                        text_parts.append(text)
                        confidences.append(conf)
                        bounding_boxes.append(bbox)
                
                raw_text = ' '.join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
            else:
                raw_text = ""
                avg_confidence = 0.0
                bounding_boxes = []
                
        except Exception as e:
            logging.error(f"EasyOCR failed: {e}")
            raw_text = ""
            avg_confidence = 0.0
            bounding_boxes = []
        
        processing_time = time.time() - start_time
        cleaned_text = self.clean_text(raw_text)
        
        return OCRResult(
            text=raw_text,
            cleaned_text=cleaned_text,
            confidence=avg_confidence,
            engine_used="easyocr",
            processing_time=processing_time,
            valid_plate=self.validate_plate_text(cleaned_text),
            bounding_boxes=bounding_boxes
        )
    
    def extract_text_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR."""
        start_time = time.time()
        
        try:
            results = self.paddleocr_reader.ocr(image, cls=True)
            
            if results and results[0]:
                # Combine all detected text
                text_parts = []
                confidences = []
                bounding_boxes = []
                
                for line in results[0]:
                    bbox, (text, conf) = line
                    if conf > 0.1:  # Filter very low confidence
                        text_parts.append(text)
                        confidences.append(conf)
                        bounding_boxes.append(bbox)
                
                raw_text = ' '.join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
            else:
                raw_text = ""
                avg_confidence = 0.0
                bounding_boxes = []
                
        except Exception as e:
            logging.error(f"PaddleOCR failed: {e}")
            raw_text = ""
            avg_confidence = 0.0
            bounding_boxes = []
        
        processing_time = time.time() - start_time
        cleaned_text = self.clean_text(raw_text)
        
        return OCRResult(
            text=raw_text,
            cleaned_text=cleaned_text,
            confidence=avg_confidence,
            engine_used="paddleocr",
            processing_time=processing_time,
            valid_plate=self.validate_plate_text(cleaned_text),
            bounding_boxes=bounding_boxes
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text using advanced Indian license plate normalization.
        
        Args:
            text: Raw OCR extracted text
            
        Returns:
            Cleaned and normalized license plate text
        """
        if not text:
            return ""
        
        # Use our advanced normalization utility
        result = normalize_license_plate(text)
        
        # Return the normalized text even if validation failed
        # This preserves the cleaned format for further processing
        return result.normalized_text if result.normalized_text else text.upper()
    
    def validate_plate_text(self, text: str) -> bool:
        """
        Validate if text matches Indian license plate patterns using advanced normalization.
        
        Args:
            text: Text to validate as license plate
            
        Returns:
            True if text is a valid Indian license plate format
        """
        if not text or len(text.strip()) < 3:
            return False
        
        # Use our advanced normalization and validation
        result = normalize_license_plate(text)
        return result.is_valid and result.confidence_score > 0.5
    
    def extract_text(self, image: np.ndarray, engine: OCREngine = None) -> OCRResult:
        """
        Extract text from license plate image using specified or best available engine.
        
        Args:
            image: License plate image (BGR format)
            engine: Specific OCR engine to use (None for auto-selection)
            
        Returns:
            OCRResult with extracted text and metadata
        """
        self.stats['total_processed'] += 1
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Determine engines to try
        engines_to_try = []
        if engine and engine in self.available_engines:
            engines_to_try = [engine]
        else:
            # Try all available engines, prioritize by accuracy
            if OCREngine.EASYOCR in self.available_engines:
                engines_to_try.append(OCREngine.EASYOCR)
            if OCREngine.PADDLEOCR in self.available_engines:
                engines_to_try.append(OCREngine.PADDLEOCR)
            if OCREngine.TESSERACT in self.available_engines:
                engines_to_try.append(OCREngine.TESSERACT)
        
        best_result = None
        best_score = 0.0
        
        # Try each engine and select best result
        for engine_type in engines_to_try:
            try:
                if engine_type == OCREngine.TESSERACT:
                    result = self.extract_text_tesseract(processed_image)
                elif engine_type == OCREngine.EASYOCR:
                    result = self.extract_text_easyocr(processed_image)
                elif engine_type == OCREngine.PADDLEOCR:
                    result = self.extract_text_paddleocr(processed_image)
                else:
                    continue
                
                # Score based on confidence and validity
                score = result.confidence
                if result.valid_plate:
                    score += 0.3  # Bonus for valid plate pattern
                if len(result.cleaned_text) >= 5:
                    score += 0.1  # Bonus for reasonable length
                
                if score > best_score:
                    best_result = result
                    best_score = score
                
                # Update engine usage stats
                self.stats['engine_usage'][engine_type.value] += 1
                
                # If we found a high-confidence valid plate, stop trying other engines
                if result.confidence > 0.8 and result.valid_plate:
                    break
                    
            except Exception as e:
                logging.error(f"OCR engine {engine_type.value} failed: {e}")
                continue
        
        # Use best result or create empty result
        if best_result is None:
            best_result = OCRResult(
                text="",
                cleaned_text="",
                confidence=0.0,
                engine_used="none",
                processing_time=0.0,
                valid_plate=False
            )
        
        # Update statistics
        if best_result.cleaned_text:
            self.stats['successful_extractions'] += 1
        else:
            self.stats['failed_extractions'] += 1
        
        # Update rolling averages
        total = self.stats['total_processed']
        self.stats['avg_processing_time'] = (
            (self.stats['avg_processing_time'] * (total - 1) + best_result.processing_time) / total
        )
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (total - 1) + best_result.confidence) / total
        )
        
        return best_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OCR processing statistics."""
        stats = self.stats.copy()
        stats['success_rate'] = (
            self.stats['successful_extractions'] / max(1, self.stats['total_processed'])
        )
        stats['available_engines'] = [engine.value for engine in self.available_engines]
        return stats


def create_test_license_plate_image() -> np.ndarray:
    """
    Create a synthetic license plate image for testing.
    
    Returns:
        Synthetic license plate image
    """
    # Create white background
    image = np.ones((60, 200, 3), dtype=np.uint8) * 255
    
    # Add black border
    cv2.rectangle(image, (5, 5), (195, 55), (0, 0, 0), 2)
    
    # Add license plate text
    cv2.putText(image, 'ABC123', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    return image


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize OCR processor
    ocr_processor = LicensePlateOCR()
    
    # Test with synthetic image
    test_image = create_test_license_plate_image()
    
    print("Testing License Plate OCR...")
    print(f"Available engines: {[engine.value for engine in ocr_processor.available_engines]}")
    
    # Extract text
    result = ocr_processor.extract_text(test_image)
    
    print(f"\nOCR Results:")
    print(f"Raw text: '{result.text}'")
    print(f"Cleaned text: '{result.cleaned_text}'")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Engine used: {result.engine_used}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Valid plate: {result.valid_plate}")
    
    # Show statistics
    stats = ocr_processor.get_statistics()
    print(f"\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
