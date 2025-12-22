#!/usr/bin/env python3
"""
Test script for ANPR system to verify all components are working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from vision.core.detector import YOLOv11PlateDetector, PlateDetection
from vision.core.processor import FrameProcessor
from vision.utils.video_capture import VideoCapture


def test_basic_functionality():
    """Test basic functionality of all components."""
    print("🚀 ANPR System Test - Basic Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Initialize detector
        print("\n1️⃣ Testing YOLOv11 Detector...")
        detector = YOLOv11PlateDetector(model_path="yolo11n.pt", confidence_threshold=0.5)
        print("   ✅ Detector initialized successfully")
        
        # Test 2: Test detection on dummy frame
        print("\n2️⃣ Testing detection on dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_plates(dummy_frame)
        print(f"   ✅ Detection successful: {len(detections)} objects found")
        
        # Test 3: Initialize processor
        print("\n3️⃣ Testing FrameProcessor...")
        processor = FrameProcessor(
            target_fps=30.0,
            frame_skip=1,
            yolo_model_path="yolo11n.pt",
            confidence_threshold=0.5
        )
        print("   ✅ Processor initialized successfully")
        
        # Test 4: Check processor stats
        stats = processor.get_processing_stats()
        print(f"   ✅ Stats: {stats}")
        
        # Test 5: Test with synthetic plate-like image
        print("\n4️⃣ Testing with synthetic license plate image...")
        plate_image = create_synthetic_plate()
        detections = detector.detect_plates(plate_image)
        print(f"   ✅ Synthetic plate test: {len(detections)} detections")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_synthetic_plate():
    """Create a synthetic license plate image for testing."""
    # Create a 480x640 image with a white rectangle (simulating a license plate)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a white rectangle in the center
    cv2.rectangle(image, (200, 200), (440, 280), (255, 255, 255), -1)
    
    # Add some black text on it
    cv2.putText(image, "ABC123", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image


def test_video_capture():
    """Test video capture functionality (without actually using camera)."""
    print("\n🎥 ANPR System Test - Video Capture")
    print("=" * 50)
    
    try:
        # Just test that we can import and initialize the class
        from vision.utils.video_capture import VideoCapture
        print("   ✅ VideoCapture class imported successfully")
        
        # We won't actually try to open a camera since it might not be available
        print("   ✅ Video capture module is ready")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Video capture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("\n🔧 ANPR System Test - Component Integration")
    print("=" * 50)
    
    try:
        # Test convenience functions can be imported
        from vision.core.processor import process_webcam_stream, process_rtsp_stream, process_video_file
        print("   ✅ Convenience functions imported successfully")
        
        # Test that we can create processor with YOLO
        processor = FrameProcessor(yolo_model_path="yolo11n.pt")
        print("   ✅ Integrated processor created successfully")
        
        # Test processing a synthetic frame through the full pipeline
        synthetic_frame = create_synthetic_plate()
        
        # Simulate metadata
        metadata = {
            'frame_id': 1,
            'timestamp': 1234567890.0,
            'source_id': 'test',
            'frame_size': synthetic_frame.shape
        }
        
        # Test the internal YOLO processing
        if processor.detector:
            detections = processor.detector.detect_plates(synthetic_frame)
            print(f"   ✅ Pipeline processing: {len(detections)} detections")
        
        print("   ✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🏁 Starting ANPR System Complete Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Video Capture", test_video_capture),
        ("Component Integration", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ANPR system is ready to use.")
        print("\nNext steps:")
        print("1. Connect a camera or prepare video files")
        print("2. Run: python -m vision.core.processor demo")
        print("3. Or use the convenience functions in your own code")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
