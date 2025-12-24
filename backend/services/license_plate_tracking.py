"""
License Plate Pass Tracking Service

This module implements the core business logic for tracking license plate passes:
- If plate already exists → increment count and update last_seen
- If plate doesn't exist → insert new record with count = 1
- Track: plate_number, pass_count, first_seen, last_seen

Author: AI Assistant
Date: December 2025
"""

import os
import asyncio
import asyncpg
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LicensePlatePassResult:
    """Result of license plate pass tracking operation"""
    plate_number: str
    pass_count: int
    first_seen: datetime
    last_seen: datetime
    is_new_plate: bool
    processed_at: datetime


@dataclass  
class LicensePlateStats:
    """License plate statistics"""
    plate_number: str
    pass_count: int
    first_seen: datetime
    last_seen: datetime
    days_since_first_seen: int
    hours_since_last_seen: float


class LicensePlateTrackingService:
    """
    Core service for license plate pass tracking business logic.
    
    This service handles the fundamental requirement:
    - If plate exists → increment pass_count, update last_seen
    - If plate doesn't exist → insert new record with pass_count = 1
    """
    
    def __init__(self, database_url: str = None):
        """
        Initialize the license plate tracking service.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or self._get_default_database_url()
        self._pool = None
    
    def _get_default_database_url(self) -> str:
        """Get default database URL from environment variables"""
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        user = os.getenv('DB_USER', 'anpr_user')
        password = os.getenv('DB_PASSWORD', 'anpr_password')
        database = os.getenv('DB_NAME', 'anpr_db')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("🔌 Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._pool:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            yield connection
    
    async def track_license_plate_pass(
        self,
        plate_number: str,
        detection_timestamp: datetime = None
    ) -> LicensePlatePassResult:
        """
        Core business logic: Track a license plate pass.
        
        If plate already exists → increment pass_count and update last_seen
        If plate doesn't exist → insert new record with pass_count = 1
        
        Args:
            plate_number: Normalized license plate number (e.g., "DL 01 AB 1234")
            detection_timestamp: When the plate was detected (defaults to now)
            
        Returns:
            LicensePlatePassResult with updated tracking information
        """
        if detection_timestamp is None:
            detection_timestamp = datetime.now(timezone.utc)
        
        logger.info(f"🚗 Tracking license plate pass: {plate_number}")
        
        async with self.get_connection() as conn:
            try:
                # Call the database function that implements the core logic
                result = await conn.fetchval(
                    "SELECT track_license_plate_pass($1, $2)",
                    plate_number,
                    detection_timestamp
                )
                
                # Parse the JSON result
                data = result
                
                result_obj = LicensePlatePassResult(
                    plate_number=data['plate_number'],
                    pass_count=data['pass_count'], 
                    first_seen=data['first_seen'],
                    last_seen=data['last_seen'],
                    is_new_plate=data['is_new_plate'],
                    processed_at=data['processed_at']
                )
                
                if result_obj.is_new_plate:
                    logger.info(f"🆕 New license plate registered: {plate_number} (pass_count: 1)")
                else:
                    logger.info(f"🔄 Existing plate updated: {plate_number} (pass_count: {result_obj.pass_count})")
                
                return result_obj
                
            except Exception as e:
                logger.error(f"❌ Failed to track license plate pass: {e}")
                raise
    
    async def get_license_plate_stats(
        self,
        plate_number: str = None,
        limit: int = 100
    ) -> List[LicensePlateStats]:
        """
        Get license plate statistics.
        
        Args:
            plate_number: Specific plate to get stats for (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of LicensePlateStats objects
        """
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch(
                    "SELECT * FROM get_license_plate_stats($1, $2)",
                    plate_number,
                    limit
                )
                
                results = []
                for row in rows:
                    stats = LicensePlateStats(
                        plate_number=row['plate_number'],
                        pass_count=row['pass_count'],
                        first_seen=row['first_seen'],
                        last_seen=row['last_seen'],
                        days_since_first_seen=row['days_since_first_seen'],
                        hours_since_last_seen=float(row['hours_since_last_seen'])
                    )
                    results.append(stats)
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get license plate stats: {e}")
                raise
    
    async def get_recent_passes(self, hours: int = 24) -> List[Dict]:
        """Get recent license plate passes within specified hours"""
        async with self.get_connection() as conn:
            try:
                query = """
                SELECT 
                    plate_number,
                    pass_count,
                    first_seen,
                    last_seen,
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600 as hours_ago
                FROM license_plate_passes
                WHERE last_seen >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
                ORDER BY last_seen DESC
                """ % hours
                
                rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    results.append({
                        'plate_number': row['plate_number'],
                        'pass_count': row['pass_count'],
                        'first_seen': row['first_seen'],
                        'last_seen': row['last_seen'],
                        'hours_ago': round(float(row['hours_ago']), 2)
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get recent passes: {e}")
                raise
    
    async def get_top_frequent_plates(self, limit: int = 20) -> List[Dict]:
        """Get most frequently seen license plates"""
        async with self.get_connection() as conn:
            try:
                query = """
                SELECT 
                    plate_number,
                    pass_count,
                    first_seen,
                    last_seen,
                    EXTRACT(DAYS FROM (last_seen - first_seen)) as days_active
                FROM license_plate_passes
                ORDER BY pass_count DESC, last_seen DESC
                LIMIT $1
                """
                
                rows = await conn.fetch(query, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'plate_number': row['plate_number'],
                        'pass_count': row['pass_count'],
                        'first_seen': row['first_seen'],
                        'last_seen': row['last_seen'],
                        'days_active': int(row['days_active']) if row['days_active'] else 0
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get top frequent plates: {e}")
                raise
    
    async def get_daily_summary(self, days: int = 7) -> List[Dict]:
        """Get daily pass tracking summary"""
        async with self.get_connection() as conn:
            try:
                query = """
                SELECT * FROM daily_pass_summary
                WHERE pass_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY pass_date DESC
                """ % days
                
                rows = await conn.fetch(query)
                
                results = []
                for row in rows:
                    results.append({
                        'date': row['pass_date'],
                        'total_plates_seen': row['total_plates_seen'],
                        'new_plates': row['new_plates'],
                        'returning_plates': row['returning_plates'],
                        'avg_pass_count': round(float(row['avg_pass_count']), 2),
                        'max_pass_count': row['max_pass_count']
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get daily summary: {e}")
                raise


# Convenience function for single-use operations
async def track_single_plate_pass(
    plate_number: str,
    detection_timestamp: datetime = None,
    database_url: str = None
) -> LicensePlatePassResult:
    """
    Convenience function to track a single license plate pass.
    
    This implements the core business logic:
    - If plate exists → increment count and update last_seen  
    - If plate doesn't exist → insert new record with count = 1
    
    Args:
        plate_number: Normalized license plate number
        detection_timestamp: When detected (defaults to now)
        database_url: Database connection string (optional)
        
    Returns:
        LicensePlatePassResult with tracking information
    """
    service = LicensePlateTrackingService(database_url)
    try:
        result = await service.track_license_plate_pass(plate_number, detection_timestamp)
        return result
    finally:
        await service.close()


# Example usage and testing
async def demo_license_plate_tracking():
    """Demonstrate the license plate tracking system"""
    print("🚗 LICENSE PLATE PASS TRACKING DEMO")
    print("=" * 50)
    
    service = LicensePlateTrackingService()
    await service.initialize()
    
    try:
        # Test plates
        test_plates = [
            "DL 01 AB 1234",  # Delhi plate
            "MH 12 DE 3456",  # Maharashtra plate  
            "DL 01 AB 1234",  # Same Delhi plate (should increment)
            "KA 05 BC 7890",  # Karnataka plate
            "DL 01 AB 1234",  # Same Delhi plate again (should increment to 3)
        ]
        
        print("\n🔄 Processing license plate passes...")
        
        for i, plate in enumerate(test_plates, 1):
            print(f"\n📱 Pass {i}: {plate}")
            result = await service.track_license_plate_pass(plate)
            
            print(f"   ✅ Pass Count: {result.pass_count}")
            print(f"   🆕 New Plate: {result.is_new_plate}")
            print(f"   📅 First Seen: {result.first_seen.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   🕒 Last Seen: {result.last_seen.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📊 License Plate Statistics:")
        stats = await service.get_license_plate_stats()
        
        for stat in stats:
            print(f"   🚗 {stat.plate_number}: {stat.pass_count} passes, "
                  f"last seen {stat.hours_since_last_seen:.1f} hours ago")
        
        print(f"\n🏆 Top Frequent Plates:")
        frequent = await service.get_top_frequent_plates(5)
        
        for plate in frequent:
            print(f"   🥇 {plate['plate_number']}: {plate['pass_count']} passes")
        
        print(f"\n📈 Daily Summary:")
        daily = await service.get_daily_summary(3)
        
        for day in daily:
            print(f"   📅 {day['date']}: {day['total_plates_seen']} plates seen, "
                  f"{day['new_plates']} new, {day['returning_plates']} returning")
    
    finally:
        await service.close()
    
    print(f"\n🎉 License plate tracking demo complete!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_license_plate_tracking())
