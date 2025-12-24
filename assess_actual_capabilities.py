#!/usr/bin/env python3
"""
ANPR System Actual Capability Assessment

This test checks what your ANPR system CAN and CANNOT do in reality.
Not a demo - actual capability testing.

Author: AI Assistant
Date: December 2025
"""

import os
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from vision.utils.plate_normalizer import normalize_license_plate
    from backend.services.plate_tracking import LicensePlateTracker
    from vision.utils.ocr_repair import OCRRepairEngine, enhanced_normalize_with_repair
except ImportError as e:
    print(f"❌ Cannot import system components: {e}")
    sys.exit(1)


def test_actual_normalization_capability():
    """Test actual normalization capabilities with realistic OCR errors"""
    print("\n🔍 Testing Actual Normalization Capability...")
    
    # Test cases with common OCR errors that occur in real scenarios
    test_cases = [
        # Clean text (should work)
        ("MH12AB1234", "Standard format"),
        ("DL8CAF9754", "Delhi format"),
        ("GJ05BZ9999", "Gujarat format"),
        
        # Common OCR errors (O→0, I→1, etc.)
        ("MH1ZAB1234", "O→0 error in district"),
        ("MHI2AB1234", "I→1 error in district"), 
        ("MH12AB1Z34", "O→0 error in number"),
        ("MH12ABI234", "I→1 error in number"),
        ("MHIZABI234", "Multiple I/O errors"),
        ("0L8CAF9754", "0→O error at start"),
        ("DL8CAFQ754", "9→Q error"),
        ("GJ05BZ99Q9", "9→Q at end"),
        
        # Spacing/formatting issues
        ("MH 12 AB 1234", "Spacing errors"),
        ("MH12 AB1234", "Partial spacing"),
        ("mh12ab1234", "Lowercase"),
        ("MH12AB 1234", "Number spacing"),
        
        # Noise and partial corruption
        ("MH12AB123", "Missing digit"),
        ("XH12AB1234", "M→X corruption"),
        ("MH12XB1234", "A→X corruption"),
        ("MH12AB12345", "Extra digit"),
        
        # Bharat Series
        ("22BH1234AB", "Bharat series"),
        ("22BHI234AB", "Bharat with I error"),
        ("Z2BH1234AB", "Bharat with corruption"),
    ]
    
    success_count = 0
    repair_success_count = 0
    repair_engine = OCRRepairEngine()
    
    print("\n� Testing Standard Normalization vs. With Pre-Repair:")
    print("=" * 80)
    
    for ocr_text, description in test_cases:
        # Test standard normalization
        try:
            result = normalize_license_plate(ocr_text)
            standard_success = result.is_valid and result.normalized_text
        except:
            standard_success = False
        
        # Test with pre-repair
        try:
            repaired_result_tuple = enhanced_normalize_with_repair(ocr_text)
            repaired_result = repaired_result_tuple[0]  # Extract the result from tuple
            repair_success = repaired_result.is_valid and repaired_result.normalized_text
        except:
            repair_success = False
        
        if standard_success:
            success_count += 1
        if repair_success:
            repair_success_count += 1
            
        # Visual status
        standard_status = "✅" if standard_success else "❌"
        repair_status = "✅" if repair_success else "❌"
        improvement = "�" if (repair_success and not standard_success) else ""
        
        print(f"{ocr_text:15} | {standard_status} Standard | {repair_status} Pre-Repair {improvement} | {description}")
    
    total_tests = len(test_cases)
    standard_rate = (success_count / total_tests) * 100
    repair_rate = (repair_success_count / total_tests) * 100
    improvement = repair_rate - standard_rate
    
    print("=" * 80)
    print(f"📈 RESULTS:")
    print(f"   Standard Normalization: {success_count}/{total_tests} ({standard_rate:.1f}%)")
    print(f"   With Pre-Repair Stage: {repair_success_count}/{total_tests} ({repair_rate:.1f}%)")
    print(f"   Improvement: +{improvement:.1f} percentage points")
    
    # Assessment
    if repair_rate >= 80:
        print("🎯 SUCCESS: Target 80% success rate achieved!")
    elif repair_rate >= 75:
        print("✅ GOOD: Close to target, minor tweaking needed")
    elif improvement >= 20:
        print("� PROGRESS: Significant improvement, continue optimization")
    else:
        print("⚠️  NEEDS WORK: Repair stage needs enhancement")
        
    return {
        'standard_success_rate': standard_rate,
        'repair_success_rate': repair_rate,
        'improvement': improvement,
        'total_tests': total_tests
    }


def test_actual_tracking_capability():
    """Test what the tracking system can actually handle"""
    print("\n🗄️ Testing Actual Tracking Capability...")
    
    # Use temporary database for testing
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    tracker = LicensePlateTracker(temp_db.name)
    
    test_plates = [
        "DL01AB1234",
        "MH12DE3456", 
        "DL01AB1234",  # Same plate - should increment
        "KA05BC7890",
        "MH12DE3456",  # Same plate - should increment
        "DL01AB1234",  # Same plate - should increment to 3
    ]
    
    success_count = 0
    expected_counts = {"DL01AB1234": 3, "MH12DE3456": 2, "KA05BC7890": 1}
    
    for plate in test_plates:
        try:
            result = tracker.track_license_plate_pass(plate)
            success_count += 1
        except Exception as e:
            print(f"❌ {plate} → ERROR: {e}")
    
    # Verify counts are correct
    validation_success = 0
    for plate, expected_count in expected_counts.items():
        try:
            # Get actual count from database
            stats = tracker.get_license_plate_stats(limit=100)
            actual_count = next((s['pass_count'] for s in stats if s['plate_number'] == plate), 0)
            if actual_count == expected_count:
                validation_success += 1
                print(f"✅ {plate}: {actual_count} passes (expected {expected_count})")
            else:
                print(f"❌ {plate}: {actual_count} passes (expected {expected_count})")
        except Exception as e:
            print(f"❌ {plate} validation error: {e}")
    
    # Cleanup
    os.unlink(temp_db.name)
    
    total_tests = len(test_plates) + len(expected_counts)
    overall_success = success_count + validation_success
    success_rate = (overall_success / total_tests) * 100
    
    print(f"📊 Tracking Success Rate: {overall_success}/{total_tests} ({success_rate:.1f}%)")
    return success_rate


def test_end_to_end_realistic_workflow():
    """Test realistic end-to-end workflow"""
    print("\n🔄 TESTING END-TO-END REALISTIC WORKFLOW")
    print("=" * 60)
    
    # Simulate real OCR outputs (with potential errors)
    simulated_ocr_outputs = [
        {"text": "DL01AB1234", "source": "Clean OCR"},
        {"text": "MH 12 DE 3456", "source": "Spaced OCR"},
        {"text": "DL01AB1234", "source": "Same plate again"},
        {"text": "KAOSBC789O", "source": "Noisy OCR (should fail)"},
        {"text": "UP16XY9876", "source": "Clean OCR - UP plate"},
    ]
    
    # Use temporary database
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    tracker = LicensePlateTracker(temp_db.name)
    
    valid_processed = 0
    
    for i, ocr_data in enumerate(simulated_ocr_outputs, 1):
        text = ocr_data["text"]
        source = ocr_data["source"]
        
        print(f"\n📸 Processing {i}: {source}")
        print(f"   📥 Raw OCR: '{text}'")
        
        try:
            # Step 1: Normalize
            norm_result = normalize_license_plate(text)
            print(f"   🔄 Normalized: '{norm_result.normalized_text}'")
            print(f"   ✅ Valid: {norm_result.is_valid}")
            
            if norm_result.is_valid:
                # Step 2: Track
                track_result = tracker.track_license_plate_pass(norm_result.normalized_text)
                print(f"   📊 Pass Count: {track_result.pass_count}")
                print(f"   🎯 Status: {'NEW' if track_result.is_new_plate else 'RETURNING'}")
                valid_processed += 1
            else:
                print(f"   ❌ Skipped tracking (invalid plate)")
                
        except Exception as e:
            print(f"   🚨 Exception: {e}")
    
    print(f"\n📊 END-TO-END RESULTS:")
    print(f"   ✅ Valid Plates Processed: {valid_processed}/{len(simulated_ocr_outputs)}")
    
    # Cleanup
    os.unlink(temp_db.name)
    
    return valid_processed


def main():
    """Run comprehensive capability assessment"""
    print("� ANPR System - Actual Capability Assessment")
    print("=" * 60)
    
    # Test 1: Normalization capability (with and without pre-repair)
    norm_results = test_actual_normalization_capability()
    
    # Test 2: Tracking capability  
    track_rate = test_actual_tracking_capability()
    
    # Overall assessment
    print(f"\n� OVERALL SYSTEM ASSESSMENT")
    print("=" * 60)
    print(f"Standard Normalization: {norm_results['standard_success_rate']:.1f}%")
    print(f"Enhanced Normalization: {norm_results['repair_success_rate']:.1f}%")
    print(f"License Plate Tracking: {track_rate:.1f}%")
    
    # Production readiness assessment
    if (norm_results['repair_success_rate'] >= 80 and track_rate >= 90):
        print("\n🚀 PRODUCTION READY: System meets performance targets")
    elif (norm_results['repair_success_rate'] >= 70 and track_rate >= 80):
        print("\n⚠️  PILOT READY: Good for controlled testing")
    else:
        print("\n� DEVELOPMENT: Needs more optimization")
        
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if norm_results['repair_success_rate'] < 80:
        print("   - Enhance OCR repair patterns")
        print("   - Add more position-aware corrections")
    if norm_results['improvement'] > 0:
        print(f"   - Pre-repair stage provides +{norm_results['improvement']:.1f}% improvement")
    if track_rate < 90:
        print("   - Review database operations")
        
    print(f"\n✅ Assessment Complete - System functional at {norm_results['repair_success_rate']:.1f}% success rate")


if __name__ == "__main__":
    main()
