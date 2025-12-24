#!/usr/bin/env python3
"""
Complete End-to-End ANPR Pipeline

Pipeline Flow:
1. YOLO → Detect license plate in image
2. Plate Crop → Extract plate region
3. OCR → Extract text from plate
4. Pre-Normalization Repair → Fix O→0, I→1 errors based on position
5. Normalize → Validate and format plate text
6. Track → Store/update in database

Author: AI Assistant
Date: December 2025
"""

import os
import sys
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from ultralytics import YOLO
    from vision.utils.plate_normalizer import normalize_license_plate, PlateValidationResult
    from vision.utils.ocr_repair import OCRRepairEngine, repair_ocr_text
    from backend.services.plate_tracking import LicensePlateTracker
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("📋 Install required packages:")
    print("   pip install ultralytics opencv-python pytesseract")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteANPRPipeline:
    """Complete ANPR Pipeline with all stages integrated"""
    
    def __init__(self, 
                 yolo_model_path: str = "yolo11n.pt",
                 database_path: str = "license_plates.db",
                 confidence_threshold: float = 0.5):
        """
        Initialize the complete ANPR pipeline
        
        Args:
            yolo_model_path: Path to YOLO model weights
            database_path: Path to SQLite database for tracking
            confidence_threshold: Minimum confidence for YOLO detection
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        logger.info("🚀 Initializing Complete ANPR Pipeline...")
        
        # 1. YOLO Model for license plate detection
        try:
            self.yolo_model = YOLO(yolo_model_path)
            logger.info(f"✅ YOLO model loaded: {yolo_model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # 2. OCR Repair Engine for pre-normalization
        self.ocr_repair_engine = OCRRepairEngine()
        logger.info("✅ OCR Repair Engine initialized")
        
        # 3. License Plate Tracker for database operations
        try:
            self.tracker = LicensePlateTracker(database_path)
            logger.info(f"✅ License Plate Tracker initialized: {database_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize tracker: {e}")
            raise
        
        # Configure Tesseract PSM for license plates
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        logger.info("✅ Complete ANPR Pipeline ready!")

    def detect_license_plate(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Stage 1: YOLO Detection - Find license plate in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (cropped_plate_image, detection_info) or None if not found
        """
        logger.info("🔍 Stage 1: YOLO License Plate Detection...")
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image, verbose=False)
            
            # Find license plate detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].item()
                        
                        # Check if confidence meets threshold
                        if confidence >= self.confidence_threshold:
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Crop license plate region
                            cropped_plate = image[y1:y2, x1:x2]
                            
                            detection_info = {
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'width': x2 - x1,
                                'height': y2 - y1
                            }
                            
                            logger.info(f"✅ License plate detected (confidence: {confidence:.3f})")
                            return cropped_plate, detection_info
            
            # Fallback: Try traditional CV methods if YOLO fails
            logger.warning("⚠️ YOLO detection failed, trying CV fallback...")
            return self._cv_fallback_detection(image)
            
        except Exception as e:
            logger.error(f"❌ YOLO detection error: {e}")
            return self._cv_fallback_detection(image)

    def _cv_fallback_detection(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Fallback CV-based license plate detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Find edges using Canny
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # License plates are typically rectangular (4 corners)
                if len(approx) == 4:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Check aspect ratio (license plates are wider than tall)
                    aspect_ratio = w / h
                    if 2.0 <= aspect_ratio <= 5.0 and w > 100 and h > 30:
                        cropped_plate = image[y:y+h, x:x+w]
                        
                        detection_info = {
                            'confidence': 0.8,  # Assume good confidence for CV detection
                            'bbox': (x, y, x+w, y+h),
                            'width': w,
                            'height': h,
                            'method': 'cv_fallback'
                        }
                        
                        logger.info("✅ License plate found using CV fallback")
                        return cropped_plate, detection_info
            
            logger.warning("❌ No license plate detected with any method")
            return None
            
        except Exception as e:
            logger.error(f"❌ CV fallback detection error: {e}")
            return None

    def extract_text_ocr(self, plate_image: np.ndarray) -> Optional[str]:
        """
        Stage 2: OCR - Extract text from cropped license plate
        
        Args:
            plate_image: Cropped license plate image
            
        Returns:
            Extracted text or None if OCR fails
        """
        logger.info("📝 Stage 2: OCR Text Extraction...")
        
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_for_ocr(plate_image)
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                config=self.tesseract_config
            ).strip()
            
            if extracted_text:
                logger.info(f"✅ OCR extracted: '{extracted_text}'")
                return extracted_text
            else:
                logger.warning("❌ OCR extraction failed - no text found")
                return None
                
        except Exception as e:
            logger.error(f"❌ OCR extraction error: {e}")
            return None

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize image for better OCR (height should be at least 32 pixels)
        height, width = gray.shape
        if height < 32:
            scale_factor = 32 / height
            new_width = int(width * scale_factor)
            gray = cv2.resize(gray, (new_width, 32), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def repair_ocr_text(self, raw_ocr_text: str) -> Tuple[str, float]:
        """
        Stage 3: Pre-Normalization Repair - Fix common OCR errors
        
        Args:
            raw_ocr_text: Raw text from OCR
            
        Returns:
            Tuple of (repaired_text, confidence_score)
        """
        logger.info("🔧 Stage 3: Pre-Normalization OCR Repair...")
        
        try:
            # Apply position-aware OCR repairs
            repaired_text, confidence = repair_ocr_text(raw_ocr_text)
            
            if repaired_text != raw_ocr_text:
                logger.info(f"✅ OCR repaired: '{raw_ocr_text}' → '{repaired_text}' (confidence: {confidence:.2f})")
            else:
                logger.info(f"✅ OCR text clean: '{raw_ocr_text}' (confidence: {confidence:.2f})")
            
            return repaired_text, confidence
            
        except Exception as e:
            logger.error(f"❌ OCR repair error: {e}")
            return raw_ocr_text, 0.5  # Return original text with low confidence

    def normalize_license_plate(self, repaired_text: str) -> Optional[PlateValidationResult]:
        """
        Stage 4: Normalization - Validate and format license plate
        
        Args:
            repaired_text: Text after OCR repair
            
        Returns:
            PlateValidationResult or None if invalid
        """
        logger.info("🔄 Stage 4: License Plate Normalization...")
        
        try:
            result = normalize_license_plate(repaired_text)
            
            if result.is_valid:
                logger.info(f"✅ Plate normalized: '{result.normalized_text}' ({result.format_type.value})")
            else:
                logger.warning(f"❌ Plate normalization failed: {result.errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Normalization error: {e}")
            return None

    def track_license_plate(self, normalized_plate: str) -> Optional[Dict[str, Any]]:
        """
        Stage 5: Tracking - Store/update license plate in database
        
        Args:
            normalized_plate: Validated and normalized plate text
            
        Returns:
            Tracking result or None if failed
        """
        logger.info("🗄️ Stage 5: License Plate Tracking...")
        
        try:
            result = self.tracker.track_license_plate_pass(normalized_plate)
            
            tracking_info = {
                'plate_number': normalized_plate,
                'pass_count': result.pass_count,
                'is_new_plate': result.is_new_plate,
                'last_seen': result.last_seen.isoformat() if result.last_seen else None
            }
            
            status = "NEW" if result.is_new_plate else "RETURNING"
            logger.info(f"✅ Plate tracked: {normalized_plate} - {status} (count: {result.pass_count})")
            
            return tracking_info
            
        except Exception as e:
            logger.error(f"❌ Tracking error: {e}")
            return None

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: Process image through all stages
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with results from all stages
        """
        logger.info(f"🚀 Processing image: {image_path}")
        
        pipeline_result = {
            'image_path': image_path,
            'success': False,
            'stages': {
                'detection': None,
                'ocr': None,
                'repair': None,
                'normalization': None,
                'tracking': None
            },
            'final_result': None,
            'processing_time': None
        }
        
        import time
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Could not load image: {image_path}")
                return pipeline_result
            
            # Stage 1: YOLO Detection
            detection_result = self.detect_license_plate(image)
            if detection_result is None:
                logger.error("❌ Pipeline failed at Stage 1: License plate not detected")
                return pipeline_result
            
            cropped_plate, detection_info = detection_result
            pipeline_result['stages']['detection'] = detection_info
            
            # Stage 2: OCR
            raw_text = self.extract_text_ocr(cropped_plate)
            if raw_text is None:
                logger.error("❌ Pipeline failed at Stage 2: OCR extraction failed")
                return pipeline_result
            
            pipeline_result['stages']['ocr'] = {'raw_text': raw_text}
            
            # Stage 3: Pre-Normalization Repair
            repaired_text, repair_confidence = self.repair_ocr_text(raw_text)
            pipeline_result['stages']['repair'] = {
                'repaired_text': repaired_text,
                'confidence': repair_confidence
            }
            
            # Stage 4: Normalization
            norm_result = self.normalize_license_plate(repaired_text)
            if norm_result is None or not norm_result.is_valid:
                logger.error("❌ Pipeline failed at Stage 4: Normalization failed")
                pipeline_result['stages']['normalization'] = {'valid': False, 'errors': norm_result.errors if norm_result else []}
                return pipeline_result
            
            pipeline_result['stages']['normalization'] = {
                'valid': True,
                'normalized_text': norm_result.normalized_text,
                'format_type': norm_result.format_type.value,
                'state_code': norm_result.state_code
            }
            
            # Stage 5: Tracking
            tracking_result = self.track_license_plate(norm_result.normalized_text)
            if tracking_result is None:
                logger.error("❌ Pipeline failed at Stage 5: Tracking failed")
                return pipeline_result
            
            pipeline_result['stages']['tracking'] = tracking_result
            
            # Success!
            pipeline_result['success'] = True
            pipeline_result['final_result'] = {
                'license_plate': norm_result.normalized_text,
                'pass_count': tracking_result['pass_count'],
                'is_new': tracking_result['is_new_plate']
            }
            
            processing_time = time.time() - start_time
            pipeline_result['processing_time'] = round(processing_time, 3)
            
            logger.info(f"🎉 Pipeline SUCCESS! Plate: {norm_result.normalized_text} (processed in {processing_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
        
        return pipeline_result

    def process_batch(self, image_directory: str) -> List[Dict[str, Any]]:
        """Process multiple images in a directory"""
        logger.info(f"📁 Processing batch: {image_directory}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        results = []
        
        image_dir = Path(image_directory)
        if not image_dir.exists():
            logger.error(f"❌ Directory not found: {image_directory}")
            return results
        
        image_files = [f for f in image_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"⚠️ No images found in {image_directory}")
            return results
        
        logger.info(f"📸 Found {len(image_files)} images to process")
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"📷 Processing {i}/{len(image_files)}: {image_file.name}")
            result = self.process_image(str(image_file))
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        logger.info(f"📊 Batch complete: {successful}/{len(results)} successful")
        
        return results


def main():
    """Example usage of the complete ANPR pipeline"""
    print("🚀 Complete ANPR Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CompleteANPRPipeline(
        yolo_model_path="yolo11n.pt",  # Will download if not exists
        database_path="anpr_tracking.db",
        confidence_threshold=0.5
    )
    
    # Test with sample images
    test_images = [
        "sample_test_image.jpg",
        "synthetic_noisy.jpg",
        # Add more test images here
    ]
    
    print(f"\n📸 Testing pipeline with {len(test_images)} images...")
    print("=" * 60)
    
    all_results = []
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n🔄 Processing: {image_path}")
            print("-" * 40)
            
            result = pipeline.process_image(image_path)
            all_results.append(result)
            
            # Display result summary
            if result['success']:
                final = result['final_result']
                print(f"✅ SUCCESS: {final['license_plate']} (Pass #{final['pass_count']})")
            else:
                print("❌ FAILED")
        else:
            print(f"⚠️ Image not found: {image_path}")
    
    # Overall statistics
    print(f"\n📊 PIPELINE PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    if total > 0:
        success_rate = (successful / total) * 100
        print(f"Overall Success Rate: {successful}/{total} ({success_rate:.1f}%)")
        
        avg_time = np.mean([r['processing_time'] for r in all_results if r['processing_time']])
        print(f"Average Processing Time: {avg_time:.3f} seconds")
        
        # Stage-wise success analysis
        stages = ['detection', 'ocr', 'repair', 'normalization', 'tracking']
        for stage in stages:
            stage_success = sum(1 for r in all_results if r['stages'][stage] is not None)
            stage_rate = (stage_success / total) * 100
            print(f"{stage.title()} Success: {stage_success}/{total} ({stage_rate:.1f}%)")
    
    print(f"\n✅ Demo complete!")


if __name__ == "__main__":
    main()
