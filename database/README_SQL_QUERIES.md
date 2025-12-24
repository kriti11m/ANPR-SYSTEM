# PostgreSQL Schema and SQL Queries for Vehicle License Plate Tracking

## Table Structure

### Main Table: `license_plate_passes`

```sql
CREATE TABLE license_plate_passes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    pass_count INTEGER NOT NULL DEFAULT 1 CHECK (pass_count > 0),
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Fields:**
- `id`: Unique identifier (UUID)
- `plate_number`: Vehicle license plate number (e.g., "DL 01 AB 1234")
- `pass_count`: Number of times this vehicle has passed through
- `first_seen`: First detection timestamp
- `last_seen`: Most recent detection timestamp

## Core SQL Queries

### 1. Track Vehicle Pass (Core Business Logic)

```sql
-- Function call to track a vehicle pass
SELECT track_license_plate_pass('DL 01 AB 1234', CURRENT_TIMESTAMP);

-- Manual implementation (if not using function)
INSERT INTO license_plate_passes (plate_number, pass_count, first_seen, last_seen)
VALUES ('DL 01 AB 1234', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (plate_number) 
DO UPDATE SET 
    pass_count = license_plate_passes.pass_count + 1,
    last_seen = CURRENT_TIMESTAMP,
    updated_at = CURRENT_TIMESTAMP;
```

### 2. Get Vehicle Statistics

```sql
-- Get all vehicle statistics
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - first_seen)) as days_since_first,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600, 2) as hours_since_last
FROM license_plate_passes
ORDER BY last_seen DESC;

-- Get specific vehicle stats
SELECT * FROM get_license_plate_stats('DL 01 AB 1234');
```

### 3. Most Frequent Vehicles

```sql
-- Top 10 most frequent vehicles
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (last_seen - first_seen)) as days_active
FROM license_plate_passes
ORDER BY pass_count DESC, last_seen DESC
LIMIT 10;
```

### 4. Recent Activity

```sql
-- Vehicles seen in last 24 hours
SELECT 
    plate_number,
    pass_count,
    last_seen,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600, 2) as hours_ago
FROM license_plate_passes
WHERE last_seen >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY last_seen DESC;

-- Vehicles seen today
SELECT 
    plate_number,
    pass_count,
    last_seen
FROM license_plate_passes
WHERE DATE(last_seen) = CURRENT_DATE
ORDER BY last_seen DESC;
```

### 5. New vs Returning Vehicles

```sql
-- New vehicles (first time seen today)
SELECT 
    plate_number,
    first_seen,
    last_seen,
    pass_count
FROM license_plate_passes
WHERE DATE(first_seen) = CURRENT_DATE
ORDER BY first_seen DESC;

-- Returning vehicles (seen multiple times)
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (last_seen - first_seen)) as days_between_visits
FROM license_plate_passes
WHERE pass_count > 1
ORDER BY pass_count DESC;
```

### 6. Time-based Analytics

```sql
-- Daily summary
SELECT * FROM daily_pass_summary;

-- Hourly activity for today
SELECT 
    EXTRACT(HOUR FROM last_seen) as hour_of_day,
    COUNT(*) as vehicles_seen,
    COUNT(DISTINCT plate_number) as unique_vehicles
FROM license_plate_passes
WHERE DATE(last_seen) = CURRENT_DATE
GROUP BY EXTRACT(HOUR FROM last_seen)
ORDER BY hour_of_day;

-- Weekly summary
SELECT 
    DATE_TRUNC('week', last_seen) as week_start,
    COUNT(*) as total_passes,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    AVG(pass_count) as avg_pass_count
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '4 weeks'
GROUP BY DATE_TRUNC('week', last_seen)
ORDER BY week_start DESC;
```

### 7. Search and Filter Queries

```sql
-- Search by partial plate number
SELECT * FROM license_plate_passes
WHERE plate_number ILIKE '%DL%'
ORDER BY last_seen DESC;

-- Filter by state code (first 2 letters)
SELECT * FROM license_plate_passes
WHERE LEFT(plate_number, 2) = 'DL'
ORDER BY pass_count DESC;

-- Filter by pass count range
SELECT * FROM license_plate_passes
WHERE pass_count BETWEEN 5 AND 20
ORDER BY last_seen DESC;

-- Vehicles not seen in X days
SELECT 
    plate_number,
    pass_count,
    last_seen,
    EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - last_seen)) as days_since_last
FROM license_plate_passes
WHERE last_seen < CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY days_since_last DESC;
```

### 8. Performance and Monitoring Queries

```sql
-- Database statistics
SELECT 
    COUNT(*) as total_unique_vehicles,
    SUM(pass_count) as total_passes_recorded,
    AVG(pass_count) as avg_passes_per_vehicle,
    MAX(pass_count) as max_passes_single_vehicle,
    MIN(first_seen) as oldest_record,
    MAX(last_seen) as newest_record
FROM license_plate_passes;

-- Activity by time periods
SELECT 
    'Today' as period,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    SUM(CASE WHEN DATE(first_seen) = CURRENT_DATE THEN 1 ELSE 0 END) as new_vehicles
FROM license_plate_passes
WHERE DATE(last_seen) = CURRENT_DATE

UNION ALL

SELECT 
    'This Week' as period,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    SUM(CASE WHEN first_seen >= DATE_TRUNC('week', CURRENT_DATE) THEN 1 ELSE 0 END) as new_vehicles
FROM license_plate_passes
WHERE last_seen >= DATE_TRUNC('week', CURRENT_DATE)

UNION ALL

SELECT 
    'This Month' as period,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    SUM(CASE WHEN first_seen >= DATE_TRUNC('month', CURRENT_DATE) THEN 1 ELSE 0 END) as new_vehicles
FROM license_plate_passes
WHERE last_seen >= DATE_TRUNC('month', CURRENT_DATE);
```

### 9. Advanced Analytics Queries

```sql
-- Vehicle frequency categories
SELECT * FROM license_plate_analytics
ORDER BY pass_count DESC;

-- Peak hours analysis
SELECT 
    EXTRACT(HOUR FROM last_seen) as hour,
    COUNT(*) as total_detections,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    ROUND(COUNT(*)::NUMERIC / COUNT(DISTINCT plate_number), 2) as avg_detections_per_vehicle
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY EXTRACT(HOUR FROM last_seen)
ORDER BY total_detections DESC;

-- State-wise analysis (assuming Indian license plates)
SELECT 
    LEFT(plate_number, 2) as state_code,
    COUNT(*) as unique_vehicles,
    SUM(pass_count) as total_passes,
    AVG(pass_count) as avg_passes_per_vehicle,
    MAX(pass_count) as max_passes
FROM license_plate_passes
GROUP BY LEFT(plate_number, 2)
ORDER BY unique_vehicles DESC;
```

### 10. Maintenance and Cleanup Queries

```sql
-- Delete old records (older than 1 year)
DELETE FROM license_plate_passes
WHERE last_seen < CURRENT_TIMESTAMP - INTERVAL '1 year';

-- Archive old data
CREATE TABLE license_plate_passes_archive AS
SELECT * FROM license_plate_passes
WHERE last_seen < CURRENT_TIMESTAMP - INTERVAL '6 months';

-- Update statistics
ANALYZE license_plate_passes;

-- Check table size
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE tablename = 'license_plate_passes';
```

## Indexes for Performance

```sql
-- Core indexes (already created in migration)
CREATE INDEX idx_license_plate_passes_plate_number ON license_plate_passes(plate_number);
CREATE INDEX idx_license_plate_passes_last_seen ON license_plate_passes(last_seen DESC);
CREATE INDEX idx_license_plate_passes_pass_count ON license_plate_passes(pass_count DESC);
CREATE INDEX idx_license_plate_passes_first_seen ON license_plate_passes(first_seen);

-- Additional useful indexes
CREATE INDEX idx_license_plate_passes_state_code ON license_plate_passes(LEFT(plate_number, 2));
CREATE INDEX idx_license_plate_passes_last_seen_date ON license_plate_passes(DATE(last_seen));
CREATE INDEX idx_license_plate_passes_composite ON license_plate_passes(pass_count DESC, last_seen DESC);
```

## Views for Common Queries

### Analytics View
```sql
-- Already created: license_plate_analytics
SELECT * FROM license_plate_analytics LIMIT 10;
```

### Daily Summary View
```sql
-- Already created: daily_pass_summary
SELECT * FROM daily_pass_summary LIMIT 7;
```

### Custom View Examples
```sql
-- Recent activity view
CREATE VIEW recent_activity AS
SELECT 
    plate_number,
    pass_count,
    last_seen,
    CASE 
        WHEN last_seen >= CURRENT_TIMESTAMP - INTERVAL '1 hour' THEN 'Just Now'
        WHEN last_seen >= CURRENT_TIMESTAMP - INTERVAL '1 day' THEN 'Today'
        WHEN last_seen >= CURRENT_TIMESTAMP - INTERVAL '1 week' THEN 'This Week'
        ELSE 'Older'
    END as recency
FROM license_plate_passes
WHERE last_seen >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY last_seen DESC;

-- Frequent visitors view
CREATE VIEW frequent_visitors AS
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    ROUND(EXTRACT(DAYS FROM (last_seen - first_seen))::NUMERIC / NULLIF(pass_count - 1, 0), 2) as avg_days_between_visits
FROM license_plate_passes
WHERE pass_count > 5
ORDER BY pass_count DESC;
```

## Example Usage

### Tracking a New Vehicle Pass
```sql
-- Track a vehicle pass (automatically handles new/existing)
SELECT track_license_plate_pass('KA 05 MN 1234');

-- Result: {"plate_number": "KA 05 MN 1234", "pass_count": 1, "is_new_plate": true, ...}
```

### Getting Vehicle History
```sql
-- Get complete history for a specific vehicle
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - first_seen)) as days_tracked,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600, 1) as hours_since_last
FROM license_plate_passes
WHERE plate_number = 'KA 05 MN 1234';
```

This comprehensive set of SQL queries covers all common use cases for vehicle license plate tracking, from basic CRUD operations to advanced analytics and reporting.
