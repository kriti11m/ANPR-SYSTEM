#!/usr/bin/env python3
"""
Test PostgreSQL License Plate Tracking Database

This script tests the PostgreSQL database schema and core functionality
for license plate pass tracking.
"""

import asyncio
import asyncpg
import os
import json
from datetime import datetime
from typing import Dict, Any

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'user': os.getenv('DB_USER', 'anpr_user'),
    'password': os.getenv('DB_PASSWORD', 'anpr_password'),
    'database': os.getenv('DB_NAME', 'anpr_db')
}

class DatabaseTester:
    """Test PostgreSQL database functionality"""
    
    def __init__(self):
        self.conn = None
    
    async def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = await asyncpg.connect(**DB_CONFIG)
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            await self.conn.close()
            print("🔌 Disconnected from database")
    
    async def test_table_exists(self):
        """Test if main table exists"""
        try:
            result = await self.conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'license_plate_passes'
            """)
            
            if result > 0:
                print("✅ Table 'license_plate_passes' exists")
                return True
            else:
                print("❌ Table 'license_plate_passes' not found")
                return False
        except Exception as e:
            print(f"❌ Error checking table: {e}")
            return False
    
    async def test_core_function(self):
        """Test core tracking function"""
        try:
            # Test tracking a new plate
            test_plate = f"TEST{datetime.now().strftime('%H%M%S')}"
            
            result = await self.conn.fetchval(
                "SELECT track_license_plate_pass($1)",
                test_plate
            )
            
            if result:
                data = result  # Result is already parsed JSON from asyncpg
                print(f"✅ Core function works: {test_plate}")
                print(f"   📊 Pass count: {data['pass_count']}")
                print(f"   🆕 New plate: {data['is_new_plate']}")
                
                # Test incrementing the same plate
                result2 = await self.conn.fetchval(
                    "SELECT track_license_plate_pass($1)",
                    test_plate
                )
                
                data2 = result2
                if data2['pass_count'] == 2 and not data2['is_new_plate']:
                    print(f"✅ Increment function works: pass count = {data2['pass_count']}")
                    return True
                else:
                    print("❌ Increment function failed")
                    return False
            else:
                print("❌ Core function returned no result")
                return False
                
        except Exception as e:
            print(f"❌ Error testing core function: {e}")
            return False
    
    async def test_utility_functions(self):
        """Test utility functions"""
        try:
            # Test summary function
            summary = await self.conn.fetchrow("SELECT * FROM get_tracking_summary()")
            if summary:
                print("✅ Summary function works")
                print(f"   📊 Total vehicles: {summary['total_unique_vehicles']}")
                print(f"   📊 Total passes: {summary['total_passes']}")
            
            # Test search function
            search_results = await self.conn.fetch("SELECT * FROM search_vehicles('TEST')")
            if search_results:
                print(f"✅ Search function works: found {len(search_results)} results")
            
            # Test recent activity function
            recent = await self.conn.fetch("SELECT * FROM get_recent_activity(24)")
            print(f"✅ Recent activity function works: {len(recent)} vehicles in last 24h")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing utility functions: {e}")
            return False
    
    async def test_views(self):
        """Test database views"""
        try:
            # Test analytics view
            analytics = await self.conn.fetch("SELECT * FROM license_plate_analytics LIMIT 3")
            print(f"✅ Analytics view works: {len(analytics)} records")
            
            # Test daily summary view
            daily = await self.conn.fetch("SELECT * FROM daily_pass_summary LIMIT 3")
            print(f"✅ Daily summary view works: {len(daily)} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing views: {e}")
            return False
    
    async def test_batch_processing(self):
        """Test batch processing function"""
        try:
            test_plates = ['BATCH01', 'BATCH02', 'BATCH03']
            
            result = await self.conn.fetchval(
                "SELECT batch_track_plates($1)",
                test_plates
            )
            
            if result:
                data = result
                print("✅ Batch processing works")
                print(f"   📊 Processed: {data['processed_count']} plates")
                print(f"   🆕 New plates: {data['new_plates_count']}")
                print(f"   🔄 Existing plates: {data['existing_plates_count']}")
                return True
            else:
                print("❌ Batch processing failed")
                return False
                
        except Exception as e:
            print(f"❌ Error testing batch processing: {e}")
            return False
    
    async def test_indexes(self):
        """Test if indexes are created"""
        try:
            indexes = await self.conn.fetch("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'license_plate_passes'
            """)
            
            print(f"✅ Found {len(indexes)} indexes:")
            for idx in indexes:
                print(f"   📋 {idx['indexname']}")
            
            return len(indexes) > 0
            
        except Exception as e:
            print(f"❌ Error checking indexes: {e}")
            return False
    
    async def performance_test(self):
        """Run basic performance test"""
        try:
            import time
            
            # Test single insert performance
            start_time = time.time()
            test_plates = [f"PERF{i:04d}" for i in range(100)]
            
            for plate in test_plates:
                await self.conn.fetchval(
                    "SELECT track_license_plate_pass($1)",
                    plate
                )
            
            single_time = time.time() - start_time
            
            # Test batch insert performance
            start_time = time.time()
            batch_plates = [f"BATCH{i:04d}" for i in range(100)]
            
            await self.conn.fetchval(
                "SELECT batch_track_plates($1)",
                batch_plates
            )
            
            batch_time = time.time() - start_time
            
            print("📈 Performance Test Results:")
            print(f"   ⏱️  Single inserts (100): {single_time:.3f}s ({100/single_time:.1f} ops/sec)")
            print(f"   ⏱️  Batch insert (100): {batch_time:.3f}s ({100/batch_time:.1f} ops/sec)")
            print(f"   🚀 Batch is {single_time/batch_time:.1f}x faster")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data"""
        try:
            deleted = await self.conn.execute("""
                DELETE FROM license_plate_passes 
                WHERE plate_number LIKE 'TEST%' 
                   OR plate_number LIKE 'BATCH%' 
                   OR plate_number LIKE 'PERF%'
            """)
            
            print(f"🧹 Cleaned up test data: {deleted.split()[-1]} records deleted")
            return True
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

async def main():
    """Run all database tests"""
    print("🧪 PostgreSQL License Plate Tracking Database Tests")
    print("=" * 60)
    
    tester = DatabaseTester()
    
    # Connect to database
    if not await tester.connect():
        return
    
    try:
        tests_passed = 0
        total_tests = 8
        
        # Run tests
        print("\n📋 Running Tests...")
        print("-" * 40)
        
        if await tester.test_table_exists():
            tests_passed += 1
        
        if await tester.test_core_function():
            tests_passed += 1
        
        if await tester.test_utility_functions():
            tests_passed += 1
        
        if await tester.test_views():
            tests_passed += 1
        
        if await tester.test_batch_processing():
            tests_passed += 1
        
        if await tester.test_indexes():
            tests_passed += 1
        
        if await tester.performance_test():
            tests_passed += 1
        
        if await tester.cleanup_test_data():
            tests_passed += 1
        
        # Results
        print("\n" + "=" * 60)
        print(f"🧪 Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! Database is working correctly.")
            print("\n🚀 Your PostgreSQL ANPR database is ready for production!")
            print("\n📖 Next steps:")
            print("   1. Use track_license_plate_pass() to track vehicles")
            print("   2. Use get_tracking_summary() for statistics")
            print("   3. Check README_SQL_QUERIES.md for more examples")
        else:
            print("❌ Some tests failed. Please check the setup.")
            print("\n🔧 Troubleshooting:")
            print("   1. Ensure PostgreSQL is running")
            print("   2. Check database connection details")
            print("   3. Run setup script: ./setup_postgresql.sh")
    
    finally:
        await tester.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
