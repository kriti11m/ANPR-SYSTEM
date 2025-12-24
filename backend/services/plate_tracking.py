"""
License Plate Pass Tracking - Synchronous Implementation

This module implements the core business logic for tracking license plate passes:
- If plate already exists → increment count and update last_seen
- If plate doesn't exist → insert new record with count = 1
- Track: plate_number, pass_count, first_seen, last_seen

Integrates with the ANPR pipeline for real-time tracking.

Author: AI Assistant  
Date: December 2025
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, NamedTuple
from contextlib import contextmanager
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlatePassResult(NamedTuple):
    """Result of license plate pass tracking operation"""
    plate_number: str
    pass_count: int
    first_seen: str
    last_seen: str
    is_new_plate: bool
    processed_at: str


class LicensePlateTracker:
    """
    Core service for license plate pass tracking business logic.
    
    This service handles the fundamental requirement:
    - If plate exists → increment pass_count, update last_seen
    - If plate doesn't exist → insert new record with pass_count = 1
    """
    
    def __init__(self, db_path: str = "anpr_tracking.db"):
        """
        Initialize the license plate tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                # Create license_plate_passes table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS license_plate_passes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL UNIQUE,
                        pass_count INTEGER NOT NULL DEFAULT 1,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_plate_number 
                    ON license_plate_passes(plate_number)
                """)
                
                # Create index for last_seen queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_seen 
                    ON license_plate_passes(last_seen DESC)
                """)
                
                conn.commit()
                logger.info("✅ Database initialized successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def track_license_plate_pass(
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
        
        timestamp_str = detection_timestamp.isoformat()
        processed_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"🚗 Tracking license plate pass: {plate_number}")
        
        with self.get_connection() as conn:
            try:
                # Check if plate already exists
                existing = conn.execute(
                    "SELECT plate_number, pass_count, first_seen, last_seen FROM license_plate_passes WHERE plate_number = ?",
                    (plate_number,)
                ).fetchone()
                
                if existing:
                    # Plate exists → increment pass_count and update last_seen
                    new_count = existing['pass_count'] + 1
                    conn.execute("""
                        UPDATE license_plate_passes 
                        SET pass_count = ?, last_seen = ?, updated_at = ?
                        WHERE plate_number = ?
                    """, (new_count, timestamp_str, processed_at, plate_number))
                    
                    result = LicensePlatePassResult(
                        plate_number=plate_number,
                        pass_count=new_count,
                        first_seen=existing['first_seen'],
                        last_seen=timestamp_str,
                        is_new_plate=False,
                        processed_at=processed_at
                    )
                    
                    logger.info(f"🔄 Existing plate updated: {plate_number} (pass_count: {new_count})")
                    
                else:
                    # Plate doesn't exist → insert new record with pass_count = 1
                    conn.execute("""
                        INSERT INTO license_plate_passes 
                        (plate_number, pass_count, first_seen, last_seen, created_at, updated_at)
                        VALUES (?, 1, ?, ?, ?, ?)
                    """, (plate_number, timestamp_str, timestamp_str, processed_at, processed_at))
                    
                    result = LicensePlatePassResult(
                        plate_number=plate_number,
                        pass_count=1,
                        first_seen=timestamp_str,
                        last_seen=timestamp_str,
                        is_new_plate=True,
                        processed_at=processed_at
                    )
                    
                    logger.info(f"🆕 New license plate registered: {plate_number} (pass_count: 1)")
                
                conn.commit()
                return result
                
            except Exception as e:
                logger.error(f"❌ Failed to track license plate pass: {e}")
                conn.rollback()
                raise
    
    def get_license_plate_stats(
        self,
        plate_number: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get license plate statistics.
        
        Args:
            plate_number: Specific plate to get stats for (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of license plate statistics
        """
        with self.get_connection() as conn:
            try:
                if plate_number:
                    query = """
                        SELECT plate_number, pass_count, first_seen, last_seen,
                               julianday('now') - julianday(first_seen) as days_since_first_seen,
                               (julianday('now') - julianday(last_seen)) * 24 as hours_since_last_seen
                        FROM license_plate_passes 
                        WHERE plate_number = ?
                        ORDER BY last_seen DESC
                        LIMIT ?
                    """
                    rows = conn.execute(query, (plate_number, limit)).fetchall()
                else:
                    query = """
                        SELECT plate_number, pass_count, first_seen, last_seen,
                               julianday('now') - julianday(first_seen) as days_since_first_seen,
                               (julianday('now') - julianday(last_seen)) * 24 as hours_since_last_seen
                        FROM license_plate_passes 
                        ORDER BY last_seen DESC
                        LIMIT ?
                    """
                    rows = conn.execute(query, (limit,)).fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'plate_number': row['plate_number'],
                        'pass_count': row['pass_count'],
                        'first_seen': row['first_seen'],
                        'last_seen': row['last_seen'],
                        'days_since_first_seen': round(float(row['days_since_first_seen']), 1),
                        'hours_since_last_seen': round(float(row['hours_since_last_seen']), 2)
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get license plate stats: {e}")
                raise
    
    def get_recent_passes(self, hours: int = 24) -> List[Dict]:
        """Get recent license plate passes within specified hours"""
        with self.get_connection() as conn:
            try:
                query = """
                    SELECT plate_number, pass_count, first_seen, last_seen,
                           (julianday('now') - julianday(last_seen)) * 24 as hours_ago
                    FROM license_plate_passes
                    WHERE julianday('now') - julianday(last_seen) <= ?/24.0
                    ORDER BY last_seen DESC
                """
                
                rows = conn.execute(query, (hours,)).fetchall()
                
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
    
    def get_top_frequent_plates(self, limit: int = 20) -> List[Dict]:
        """Get most frequently seen license plates"""
        with self.get_connection() as conn:
            try:
                query = """
                    SELECT plate_number, pass_count, first_seen, last_seen,
                           julianday(last_seen) - julianday(first_seen) as days_active
                    FROM license_plate_passes
                    ORDER BY pass_count DESC, last_seen DESC
                    LIMIT ?
                """
                
                rows = conn.execute(query, (limit,)).fetchall()
                
                results = []
                for row in rows:
                    results.append({
                        'plate_number': row['plate_number'],
                        'pass_count': row['pass_count'],
                        'first_seen': row['first_seen'],
                        'last_seen': row['last_seen'],
                        'days_active': round(float(row['days_active']), 1) if row['days_active'] else 0
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Failed to get top frequent plates: {e}")
                raise
    
    def get_summary_stats(self) -> Dict:
        """Get overall summary statistics"""
        with self.get_connection() as conn:
            try:
                # Total plates and passes
                totals = conn.execute("""
                    SELECT 
                        COUNT(*) as total_unique_plates,
                        SUM(pass_count) as total_passes,
                        AVG(pass_count) as avg_passes_per_plate,
                        MAX(pass_count) as max_passes
                    FROM license_plate_passes
                """).fetchone()
                
                # Recent activity (last 24 hours)
                recent = conn.execute("""
                    SELECT COUNT(*) as recent_plates
                    FROM license_plate_passes
                    WHERE julianday('now') - julianday(last_seen) <= 1.0
                """).fetchone()
                
                return {
                    'total_unique_plates': totals['total_unique_plates'],
                    'total_passes': totals['total_passes'],
                    'avg_passes_per_plate': round(float(totals['avg_passes_per_plate'] or 0), 2),
                    'max_passes': totals['max_passes'],
                    'recent_plates_24h': recent['recent_plates']
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to get summary stats: {e}")
                raise


class ANPRIntegration:
    """Integration class to connect ANPR detection with license plate tracking"""
    
    def __init__(self, db_path: str = "anpr_tracking.db"):
        self.tracker = LicensePlateTracker(db_path)
    
    def process_detection_result(self, detection_result: Dict) -> Dict:
        """
        Process an ANPR detection result and update tracking database.
        
        Args:
            detection_result: Result from ANPR pipeline with normalized_text
            
        Returns:
            Enhanced result with tracking information
        """
        try:
            # Extract normalized plate number
            plate_number = detection_result.get('normalized_text')
            
            if not plate_number:
                logger.warning("⚠️ No normalized_text in detection result")
                return detection_result
            
            # Track the license plate pass
            tracking_result = self.tracker.track_license_plate_pass(plate_number)
            
            # Add tracking information to the result
            enhanced_result = detection_result.copy()
            enhanced_result.update({
                'tracking': {
                    'plate_number': tracking_result.plate_number,
                    'pass_count': tracking_result.pass_count,
                    'first_seen': tracking_result.first_seen,
                    'last_seen': tracking_result.last_seen,
                    'is_new_plate': tracking_result.is_new_plate,
                    'processed_at': tracking_result.processed_at
                }
            })
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"❌ Failed to process detection result: {e}")
            # Return original result if tracking fails
            return detection_result


def demo_license_plate_tracking():
    """Demonstrate the license plate tracking system"""
    print("🚗 LICENSE PLATE PASS TRACKING DEMO")
    print("=" * 50)
    
    # Initialize tracker
    tracker = LicensePlateTracker("demo_tracking.db")
    
    # Test plates
    test_plates = [
        "DL 01 AB 1234",  # Delhi plate
        "MH 12 DE 3456",  # Maharashtra plate  
        "DL 01 AB 1234",  # Same Delhi plate (should increment)
        "KA 05 BC 7890",  # Karnataka plate
        "DL 01 AB 1234",  # Same Delhi plate again (should increment to 3)
        "MH 12 DE 3456",  # Same Maharashtra plate (should increment to 2)
    ]
    
    print("\n🔄 Processing license plate passes...")
    
    for i, plate in enumerate(test_plates, 1):
        print(f"\n📱 Pass {i}: {plate}")
        result = tracker.track_license_plate_pass(plate)
        
        print(f"   ✅ Pass Count: {result.pass_count}")
        print(f"   🆕 New Plate: {'Yes' if result.is_new_plate else 'No'}")
        print(f"   📅 First Seen: {result.first_seen[:19]}")
        print(f"   🕒 Last Seen: {result.last_seen[:19]}")
    
    print(f"\n📊 Current License Plate Statistics:")
    stats = tracker.get_license_plate_stats(limit=10)
    
    for stat in stats:
        print(f"   🚗 {stat['plate_number']}: {stat['pass_count']} passes, "
              f"last seen {stat['hours_since_last_seen']:.1f} hours ago")
    
    print(f"\n🏆 Top Frequent Plates:")
    frequent = tracker.get_top_frequent_plates(5)
    
    for plate in frequent:
        print(f"   🥇 {plate['plate_number']}: {plate['pass_count']} passes")
    
    print(f"\n📈 Summary Statistics:")
    summary = tracker.get_summary_stats()
    print(f"   📊 Total Unique Plates: {summary['total_unique_plates']}")
    print(f"   📊 Total Passes: {summary['total_passes']}")
    print(f"   📊 Average Passes per Plate: {summary['avg_passes_per_plate']}")
    print(f"   📊 Most Frequent Plate: {summary['max_passes']} passes")
    print(f"   📊 Recent Activity (24h): {summary['recent_plates_24h']} plates")
    
    print(f"\n🎉 License plate tracking demo complete!")
    print(f"💾 Database saved as: demo_tracking.db")


if __name__ == "__main__":
    # Run the demo
    demo_license_plate_tracking()
