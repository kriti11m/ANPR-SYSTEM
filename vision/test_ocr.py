#!/usr/bin/env python3
"""
Test script for OCR functionality in the ANPR system.

This script tests:
1. OCR engine initialization
2. Text extraction from license plate images
3. Text cleaning and validation
4. Performance comparison between engines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from typing import List, Tuple
from vision.core.ocr import LicensePlateOCR, OCREngine, create_test_license_plate_image
from vision.core.detector import YOLOv11PlateDetector


def create_various_test_plates() -> List[Tuple[str, np.ndarray]]:
    """Create various test license plate images."""
    test_plates = []
    
    # Test plate 1: Clean synthetic plate
    plate1 = create_test_license_plate_image()
    test_plates.append(("Synthetic Clean", plate1))
    
    # Test plate 2: Different format
    plate2 = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(plate2, (5, 5), (195, 55), (0, 0, 0), 2)
    cv2.putText(plate2, 'XYZ789', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    test_plates.append(("Synthetic XYZ789", plate2))
    
    # Test plate 3: Mixed case (should be cleaned)
    plate3 = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(plate3, (5, 5), (195, 55), (0, 0, 0), 2)
    cv2.putText(plate3, 'Ab12Cd', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    test_plates.append(("Mixed Case", plate3))
    
    # Test plate 4: With noise/artifacts
    plate4 = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(plate4, (5, 5), (195, 55), (0, 0, 0), 2)
    cv2.putText(plate4, 'DEF456', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    # Add some noise
    noise = np.random.randint(0, 50, plate4.shape, dtype=np.uint8)
    plate4 = cv2.add(plate4, noise)
    test_plates.append(("With Noise", plate4))
    
    return test_plates


def test_ocr_engines():
    """Test individual OCR engines."""
    print("🔍 Testing OCR Engines")
    print("=" * 50)
    
    # Create test images
    test_plates = create_various_test_plates()
    
    # Test each available engine
    engines = [OCREngine.TESSERACT, OCREngine.EASYOCR, OCREngine.PADDLEOCR]
    
    for engine in engines:
        print(f"\n🤖 Testing {engine.value.upper()} Engine")
        print("-" * 30)
        
        try:
            # Initialize OCR with specific engine
            ocr_processor = LicensePlateOCR(engines=[engine])
            
            if not ocr_processor.available_engines:
                print(f"❌ {engine.value} not available")
                continue
            
            print(f"✅ {engine.value} initialized successfully")
            
            # Test each plate
            total_time = 0
            successful_extractions = 0
            
            for plate_name, plate_image in test_plates:
                result = ocr_processor.extract_text(plate_image)
                total_time += result.processing_time
                
                if result.cleaned_text:
                    successful_extractions += 1
                    status = "✅" if result.valid_plate else "⚠️"
                else:
                    status = "❌"
                
                print(f"  {status} {plate_name}: '{result.cleaned_text}' (conf: {result.confidence:.3f}, {result.processing_time:.3f}s)")
            
            # Summary
            avg_time = total_time / len(test_plates)
            success_rate = successful_extractions / len(test_plates)
            
            print(f"\n  📊 {engine.value} Summary:")
            print(f"     Success rate: {success_rate:.1%}")
            print(f"     Average time: {avg_time:.3f}s")
            
        except Exception as e:
            print(f"❌ {engine.value} failed: {e}")


def test_auto_engine_selection():
    """Test automatic engine selection."""
    print("\n\n🚀 Testing Auto Engine Selection")
    print("=" * 50)
    
    # Initialize with auto selection
    ocr_processor = LicensePlateOCR(engines=[OCREngine.AUTO])
    
    if not ocr_processor.available_engines:
        print("❌ No OCR engines available")
        return
    
    print(f"✅ Auto selection initialized")
    print(f"Available engines: {[e.value for e in ocr_processor.available_engines]}")
    
    # Test with various plates
    test_plates = create_various_test_plates()
    
    print(f"\n🧪 Testing {len(test_plates)} license plates:")
    
    for plate_name, plate_image in test_plates:
        result = ocr_processor.extract_text(plate_image)
        
        status = "✅" if result.cleaned_text else "❌"
        validity = "✅ Valid" if result.valid_plate else "⚠️ Invalid" if result.cleaned_text else "❌ No text"
        
        print(f"  {status} {plate_name}:")
        print(f"     Text: '{result.cleaned_text}'")
        print(f"     Confidence: {result.confidence:.3f}")
        print(f"     Engine: {result.engine_used}")
        print(f"     Validity: {validity}")
        print(f"     Time: {result.processing_time:.3f}s")
        print()
    
    # Show final statistics
    stats = ocr_processor.get_statistics()
    print(f"📈 Final Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")


def test_integrated_anpr_system():
    """Test integrated YOLO + OCR system."""
    print("\n\n🔧 Testing Integrated ANPR System")
    print("=" * 50)
    
    try:
        # Initialize detector with OCR enabled
        detector = YOLOv11PlateDetector(
            model_path="yolo11n.pt",
            enable_ocr=True,
            ocr_engines=["auto"]
        )
        
        if not detector.enable_ocr:
            print("❌ OCR not enabled in detector")
            return
            
        print("✅ ANPR system initialized with OCR")
        print(f"OCR engines available: {detector.ocr_processor.available_engines}")
        
        # Create test image with a "license plate" (rectangle with text)
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add a car-like shape
        cv2.rectangle(test_frame, (150, 200), (490, 350), (100, 50, 20), -1)
        
        # Add license plate area
        cv2.rectangle(test_frame, (250, 280), (390, 320), (255, 255, 255), -1)
        cv2.rectangle(test_frame, (250, 280), (390, 320), (0, 0, 0), 2)
        cv2.putText(test_frame, "ABC123", (260, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Run detection
        print("\n🔍 Running YOLO detection...")
        detections = detector.detect_plates(test_frame)
        
        print(f"Found {len(detections)} detections:")
        for i, detection in enumerate(detections):
            print(f"\n  Detection {i+1}:")
            print(f"    Bbox: {detection.bbox}")
            print(f"    YOLO confidence: {detection.confidence:.3f}")
            print(f"    License plate text: '{detection.license_plate}'")
            print(f"    OCR confidence: {detection.ocr_confidence:.3f}")
            print(f"    Valid plate: {detection.valid_plate}")
        
        # Save test image for inspection
        cv2.imwrite("test_anpr_frame.jpg", test_frame)
        print(f"\n💾 Test frame saved as 'test_anpr_frame.jpg'")
        
    except Exception as e:
        print(f"❌ Integrated test failed: {e}")
        import traceback
        traceback.print_exc()


def test_text_cleaning():
    """Test text cleaning functionality."""
    print("\n\n🧹 Testing Text Cleaning")
    print("=" * 50)
    
    from vision.core.ocr import LicensePlateOCR
    
    ocr_processor = LicensePlateOCR()
    
    # Test cases
    test_cases = [
        ("ABC123", "Simple valid case"),
        ("abc123", "Lowercase (should be uppercase)"),
        ("A BC 1 23", "With spaces"),
        ("A-BC-123", "With dashes"),
        ("ABC.123", "With dots"),
        ("AB C1 23XY", "Mixed with spaces"),
        ("O123", "Letter O (might be digit 0)"),
        ("I234", "Letter I (might be digit 1)"),
        ("AB12CD", "Standard format"),
        ("123ABC", "Numeric first"),
        ("", "Empty string"),
        ("!@#$%", "Only special characters"),
        ("ABC12345678", "Too long"),
    ]
    
    print("Testing text cleaning:")
    for test_text, description in test_cases:
        cleaned = ocr_processor.clean_text(test_text)
        valid = ocr_processor.validate_plate_text(cleaned)
        
        status = "✅" if valid else "⚠️" if cleaned else "❌"
        
        print(f"  {status} '{test_text}' -> '{cleaned}' ({description})")


def main():
    """Run all OCR tests."""
    print("🏁 Starting ANPR OCR Testing Suite")
    print("=" * 60)
    
    try:
        # Test 1: Individual engine testing
        test_ocr_engines()
        
        # Test 2: Auto engine selection
        test_auto_engine_selection()
        
        # Test 3: Text cleaning
        test_text_cleaning()
        
        # Test 4: Integrated ANPR system
        test_integrated_anpr_system()
        
        print("\n🎉 All OCR tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
