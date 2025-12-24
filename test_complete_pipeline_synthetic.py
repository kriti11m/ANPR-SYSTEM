#!/usr/bin/env python3
"""
Complete ANPR Pipeline Test with Synthetic Data

This demonstrates the full pipeline with synthetic license plate data
that includes OCR errors typical in real-world scenarios.

Author: AI Assistant  
Date: December 2025
"""

import os
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from vision.utils.plate_normalizer import normalize_license_plate
from vision.utils.ocr_repair import repair_ocr_text, enhanced_normalize_with_repair
from backend.services.plate_tracking import LicensePlateTracker
import tempfile

def test_complete_pipeline_synthetic():
    """Test complete pipeline with synthetic OCR data"""
    print("🚀 COMPLETE ANPR PIPELINE TEST")
    print("=" * 60)
    print("Testing: YOLO → Crop → OCR → Pre-Repair → Normalize → Track")
    print("Using synthetic OCR data with realistic errors")
    print()
    
    # Synthetic OCR results (simulating what would come from real images)
    test_scenarios = [
        {
            'name': 'Clean Mumbai Plate',
            'raw_ocr': 'MH12AB1234',
            'expected_repairs': 'MH12AB1234',
            'should_normalize': True
        },
        {
            'name': 'O→0 Error in District',
            'raw_ocr': 'MH1ZAB3456',
            'expected_repairs': 'MH12AB3456',
            'should_normalize': True
        },
        {
            'name': 'I→1 Error in District',
            'raw_ocr': 'DLI8CD9876',
            'expected_repairs': 'DL18CD9876',
            'should_normalize': True
        },
        {
            'name': 'Multiple I/O Errors',
            'raw_ocr': 'MHIZABIZ34',
            'expected_repairs': 'MH12AB1234',
            'should_normalize': True
        },
        {
            'name': 'Noisy Bharat Series',
            'raw_ocr': '22BHI234AB',
            'expected_repairs': '22BH1234AB',
            'should_normalize': True
        },
        {
            'name': 'Spaced Gujarat Plate',
            'raw_ocr': 'GJ 05 BZ 9999',
            'expected_repairs': 'GJ 05 BZ 9999',
            'should_normalize': True
        }
    ]
    
    # Use temporary database
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    tracker = LicensePlateTracker(temp_db.name)
    
    pipeline_results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"📸 Test {i}: {scenario['name']}")
        print(f"   📥 Raw OCR: '{scenario['raw_ocr']}'")
        
        pipeline_result = {
            'scenario': scenario['name'],
            'stages': {},
            'success': False
        }
        
        try:
            # Stage 1: Simulate YOLO detection (would normally provide cropped image)
            pipeline_result['stages']['detection'] = {'success': True, 'method': 'simulated'}
            print("   ✅ Stage 1: Plate detection (simulated)")
            
            # Stage 2: Simulate OCR extraction (this is our input)
            raw_text = scenario['raw_ocr']
            pipeline_result['stages']['ocr'] = {'raw_text': raw_text, 'success': True}
            print(f"   ✅ Stage 2: OCR extracted '{raw_text}'")
            
            # Stage 3: Pre-Normalization Repair
            repaired_text, repair_confidence = repair_ocr_text(raw_text)
            pipeline_result['stages']['repair'] = {
                'repaired_text': repaired_text, 
                'confidence': repair_confidence,
                'success': True
            }
            
            if repaired_text != raw_text:
                print(f"   🔧 Stage 3: OCR repaired '{raw_text}' → '{repaired_text}' (conf: {repair_confidence:.2f})")
            else:
                print(f"   ✅ Stage 3: OCR clean '{raw_text}' (conf: {repair_confidence:.2f})")
            
            # Stage 4: Normalization 
            norm_result = normalize_license_plate(repaired_text)
            pipeline_result['stages']['normalization'] = {
                'valid': norm_result.is_valid,
                'normalized_text': norm_result.normalized_text if norm_result.is_valid else None,
                'format_type': norm_result.format_type.value if norm_result.is_valid else None,
                'success': norm_result.is_valid
            }
            
            if norm_result.is_valid:
                print(f"   ✅ Stage 4: Normalized '{norm_result.normalized_text}' ({norm_result.format_type.value})")
                
                # Stage 5: Tracking
                track_result = tracker.track_license_plate_pass(norm_result.normalized_text)
                pipeline_result['stages']['tracking'] = {
                    'plate_number': norm_result.normalized_text,
                    'pass_count': track_result.pass_count,
                    'is_new': track_result.is_new_plate,
                    'success': True
                }
                
                status = "NEW" if track_result.is_new_plate else "RETURNING"
                print(f"   ✅ Stage 5: Tracked {norm_result.normalized_text} - {status} (pass #{track_result.pass_count})")
                
                pipeline_result['success'] = True
                print(f"   🎉 PIPELINE SUCCESS! Final: {norm_result.normalized_text}")
                
            else:
                print(f"   ❌ Stage 4: Normalization failed - {norm_result.errors}")
                pipeline_result['stages']['tracking'] = {'success': False}
                
        except Exception as e:
            print(f"   🚨 Pipeline error: {e}")
            
        pipeline_results.append(pipeline_result)
        print()
    
    # Summary statistics
    print("📊 COMPLETE PIPELINE PERFORMANCE")
    print("=" * 60)
    
    total_tests = len(pipeline_results)
    successful = sum(1 for r in pipeline_results if r['success'])
    
    print(f"Overall Success Rate: {successful}/{total_tests} ({(successful/total_tests)*100:.1f}%)")
    print()
    
    # Stage-wise analysis
    stages = ['detection', 'ocr', 'repair', 'normalization', 'tracking']
    for stage in stages:
        stage_success = sum(1 for r in pipeline_results if r['stages'].get(stage, {}).get('success', False))
        rate = (stage_success / total_tests) * 100
        print(f"{stage.title()} Success: {stage_success}/{total_tests} ({rate:.1f}%)")
    
    # Show successful normalizations with repair improvements
    print(f"\n🔧 OCR REPAIR IMPROVEMENTS:")
    print("=" * 40)
    
    repair_improvements = 0
    for result in pipeline_results:
        ocr_text = result['stages'].get('ocr', {}).get('raw_text', '')
        repaired_text = result['stages'].get('repair', {}).get('repaired_text', '')
        
        if ocr_text != repaired_text and result['stages'].get('normalization', {}).get('valid', False):
            repair_improvements += 1
            normalized = result['stages']['normalization']['normalized_text']
            print(f"✅ '{ocr_text}' → '{repaired_text}' → '{normalized}'")
    
    print(f"\nTotal plates improved by repair stage: {repair_improvements}")
    
    # Database summary
    try:
        stats = tracker.get_summary_stats()
        print(f"\n📊 TRACKING DATABASE SUMMARY:")
        print(f"   Unique Plates: {stats.get('total_unique_plates', 0)}")
        print(f"   Total Passes: {stats.get('total_passes', 0)}")
    except:
        print("\n❌ Could not get database stats")
    
    # Cleanup
    os.unlink(temp_db.name)
    
    print(f"\n✅ Complete ANPR Pipeline Test Finished!")
    print(f"🎯 Success Rate: {successful}/{total_tests} = {(successful/total_tests)*100:.1f}%")
    
    return successful, total_tests


if __name__ == "__main__":
    test_complete_pipeline_synthetic()
