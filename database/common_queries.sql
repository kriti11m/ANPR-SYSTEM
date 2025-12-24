-- Common SQL Queries for License Plate Tracking System
-- Ready-to-use queries for daily operations and reporting

-- =============================================================================
-- BASIC QUERIES
-- =============================================================================

-- 1. Track a single vehicle pass
SELECT track_license_plate_pass('DL 01 AB 1234');

-- 2. Get all vehicles with their statistics
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - first_seen)) as days_tracked
FROM license_plate_passes
ORDER BY last_seen DESC;

-- 3. Get specific vehicle information
SELECT * FROM license_plate_passes 
WHERE plate_number = 'DL 01 AB 1234';

-- =============================================================================
-- REPORTING QUERIES
-- =============================================================================

-- 4. Daily summary report
SELECT 
    CURRENT_DATE as report_date,
    COUNT(*) as vehicles_today,
    COUNT(CASE WHEN DATE(first_seen) = CURRENT_DATE THEN 1 END) as new_vehicles_today,
    COUNT(CASE WHEN pass_count > 1 THEN 1 END) as returning_vehicles_today,
    SUM(CASE WHEN DATE(last_seen) = CURRENT_DATE THEN 1 ELSE 0 END) as total_passes_today
FROM license_plate_passes
WHERE DATE(last_seen) = CURRENT_DATE;

-- 5. Weekly summary report
SELECT 
    DATE_TRUNC('week', last_seen)::DATE as week_start,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    SUM(pass_count) as total_passes,
    AVG(pass_count) as avg_passes_per_vehicle
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '4 weeks'
GROUP BY DATE_TRUNC('week', last_seen)
ORDER BY week_start DESC;

-- 6. Top 10 most frequent vehicles
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    CASE 
        WHEN pass_count >= 50 THEN 'Very Frequent'
        WHEN pass_count >= 20 THEN 'Frequent'
        WHEN pass_count >= 10 THEN 'Regular'
        WHEN pass_count >= 5 THEN 'Occasional'
        ELSE 'Rare'
    END as frequency_category
FROM license_plate_passes
ORDER BY pass_count DESC, last_seen DESC
LIMIT 10;

-- =============================================================================
-- SEARCH AND FILTER QUERIES
-- =============================================================================

-- 7. Search vehicles by partial plate number
SELECT * FROM license_plate_passes
WHERE plate_number ILIKE '%AB%'
ORDER BY last_seen DESC;

-- 8. Get vehicles from specific state (e.g., Delhi)
SELECT * FROM license_plate_passes
WHERE LEFT(plate_number, 2) = 'DL'
ORDER BY pass_count DESC;

-- 9. Get vehicles seen in last X hours
SELECT 
    plate_number,
    pass_count,
    last_seen,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600, 1) as hours_ago
FROM license_plate_passes
WHERE last_seen >= CURRENT_TIMESTAMP - INTERVAL '6 hours'
ORDER BY last_seen DESC;

-- 10. Get new vehicles (first time seen)
SELECT 
    plate_number,
    first_seen,
    last_seen,
    pass_count
FROM license_plate_passes
WHERE pass_count = 1 AND DATE(first_seen) = CURRENT_DATE
ORDER BY first_seen DESC;

-- =============================================================================
-- ANALYTICS QUERIES
-- =============================================================================

-- 11. Peak hours analysis
SELECT 
    EXTRACT(HOUR FROM last_seen) as hour,
    COUNT(*) as detections,
    COUNT(DISTINCT plate_number) as unique_vehicles
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY EXTRACT(HOUR FROM last_seen)
ORDER BY detections DESC;

-- 12. State-wise vehicle distribution
SELECT 
    LEFT(plate_number, 2) as state_code,
    COUNT(*) as vehicle_count,
    SUM(pass_count) as total_passes,
    AVG(pass_count) as avg_passes_per_vehicle
FROM license_plate_passes
GROUP BY LEFT(plate_number, 2)
ORDER BY vehicle_count DESC;

-- 13. Vehicle activity patterns
SELECT 
    CASE 
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 6 AND 11 THEN 'Morning (6-11 AM)'
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 12 AND 17 THEN 'Afternoon (12-5 PM)'
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 18 AND 23 THEN 'Evening (6-11 PM)'
        ELSE 'Night (12-5 AM)'
    END as time_period,
    COUNT(*) as vehicle_count,
    COUNT(DISTINCT plate_number) as unique_vehicles
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY 
    CASE 
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 6 AND 11 THEN 'Morning (6-11 AM)'
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 12 AND 17 THEN 'Afternoon (12-5 PM)'
        WHEN EXTRACT(HOUR FROM last_seen) BETWEEN 18 AND 23 THEN 'Evening (6-11 PM)'
        ELSE 'Night (12-5 AM)'
    END
ORDER BY vehicle_count DESC;

-- =============================================================================
-- FUNCTION USAGE EXAMPLES
-- =============================================================================

-- 14. Get system summary
SELECT * FROM get_tracking_summary();

-- 15. Get vehicles from Karnataka state
SELECT * FROM get_vehicles_by_state('KA');

-- 16. Get recent activity (last 12 hours)
SELECT * FROM get_recent_activity(12);

-- 17. Get top 5 frequent vehicles
SELECT * FROM get_top_frequent_vehicles(5);

-- 18. Search for plates containing "1234"
SELECT * FROM search_vehicles('1234');

-- 19. Get hourly stats for last 3 days
SELECT * FROM get_hourly_activity_stats(3);

-- =============================================================================
-- MAINTENANCE QUERIES
-- =============================================================================

-- 20. Check database health
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT plate_number) as unique_plates,
    MIN(first_seen) as oldest_record,
    MAX(last_seen) as newest_record,
    AVG(pass_count) as avg_passes,
    MAX(pass_count) as max_passes
FROM license_plate_passes;

-- 21. Find potential duplicate or problematic records
SELECT 
    plate_number,
    COUNT(*) as record_count
FROM license_plate_passes
GROUP BY plate_number
HAVING COUNT(*) > 1;

-- 22. Get table size information
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    most_common_vals,
    most_common_freqs
FROM pg_stats
WHERE tablename = 'license_plate_passes'
ORDER BY attname;

-- =============================================================================
-- BATCH OPERATIONS
-- =============================================================================

-- 23. Track multiple vehicles at once (using function)
SELECT batch_track_plates(
    ARRAY['DL 01 AB 1234', 'MH 12 CD 5678', 'KA 03 EF 9012'],
    CURRENT_TIMESTAMP
);

-- 24. Update all records with current timestamp (maintenance)
UPDATE license_plate_passes 
SET updated_at = CURRENT_TIMESTAMP
WHERE updated_at IS NULL OR updated_at < CURRENT_TIMESTAMP - INTERVAL '1 day';

-- =============================================================================
-- EXPORT QUERIES
-- =============================================================================

-- 25. Export data for specific date range
SELECT * FROM export_data_range('2025-12-01', '2025-12-31');

-- 26. Export CSV format for reporting
COPY (
    SELECT 
        plate_number,
        pass_count,
        TO_CHAR(first_seen, 'YYYY-MM-DD HH24:MI:SS') as first_seen,
        TO_CHAR(last_seen, 'YYYY-MM-DD HH24:MI:SS') as last_seen,
        EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - first_seen)) as days_tracked
    FROM license_plate_passes
    ORDER BY last_seen DESC
) TO '/tmp/vehicle_tracking_report.csv' WITH CSV HEADER;

-- =============================================================================
-- PERFORMANCE MONITORING
-- =============================================================================

-- 27. Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE tablename = 'license_plate_passes';

-- 28. Query performance analysis
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM license_plate_passes 
WHERE plate_number = 'DL 01 AB 1234';

-- 29. Refresh materialized view (run periodically)
SELECT refresh_vehicle_statistics();

-- 30. View materialized statistics
SELECT * FROM mv_vehicle_statistics 
ORDER BY activity_date DESC
LIMIT 30;
