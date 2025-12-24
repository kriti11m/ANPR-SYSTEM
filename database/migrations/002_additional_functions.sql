-- Additional SQL Functions and Procedures for License Plate Tracking
-- This file contains utility functions to extend the core tracking functionality

-- Function to get vehicle tracking summary
CREATE OR REPLACE FUNCTION get_tracking_summary()
RETURNS TABLE (
    total_unique_vehicles INTEGER,
    total_passes INTEGER,
    avg_passes_per_vehicle NUMERIC,
    most_frequent_vehicle VARCHAR(20),
    max_passes INTEGER,
    vehicles_today INTEGER,
    new_vehicles_today INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_unique_vehicles,
        SUM(lpp.pass_count)::INTEGER as total_passes,
        ROUND(AVG(lpp.pass_count), 2) as avg_passes_per_vehicle,
        (SELECT plate_number FROM license_plate_passes ORDER BY pass_count DESC LIMIT 1) as most_frequent_vehicle,
        MAX(lpp.pass_count)::INTEGER as max_passes,
        (SELECT COUNT(*)::INTEGER FROM license_plate_passes WHERE DATE(last_seen) = CURRENT_DATE) as vehicles_today,
        (SELECT COUNT(*)::INTEGER FROM license_plate_passes WHERE DATE(first_seen) = CURRENT_DATE) as new_vehicles_today
    FROM license_plate_passes lpp;
END;
$$ LANGUAGE plpgsql;

-- Function to get vehicles by state code
CREATE OR REPLACE FUNCTION get_vehicles_by_state(p_state_code VARCHAR(2))
RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    days_active INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.first_seen,
        lpp.last_seen,
        EXTRACT(DAYS FROM (lpp.last_seen - lpp.first_seen))::INTEGER as days_active
    FROM license_plate_passes lpp
    WHERE LEFT(lpp.plate_number, 2) = UPPER(p_state_code)
    ORDER BY lpp.pass_count DESC, lpp.last_seen DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get recent activity within specified hours
CREATE OR REPLACE FUNCTION get_recent_activity(p_hours INTEGER DEFAULT 24)
RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    last_seen TIMESTAMP WITH TIME ZONE,
    hours_ago NUMERIC,
    is_new_vehicle BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.last_seen,
        ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - lpp.last_seen)) / 3600, 2) as hours_ago,
        (lpp.pass_count = 1 AND DATE(lpp.first_seen) = CURRENT_DATE) as is_new_vehicle
    FROM license_plate_passes lpp
    WHERE lpp.last_seen >= CURRENT_TIMESTAMP - (p_hours || ' hours')::INTERVAL
    ORDER BY lpp.last_seen DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get top frequent vehicles
CREATE OR REPLACE FUNCTION get_top_frequent_vehicles(p_limit INTEGER DEFAULT 10)
RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    frequency_category TEXT,
    days_active INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.first_seen,
        lpp.last_seen,
        CASE 
            WHEN lpp.pass_count = 1 THEN 'Single Visit'
            WHEN lpp.pass_count BETWEEN 2 AND 5 THEN 'Occasional'
            WHEN lpp.pass_count BETWEEN 6 AND 20 THEN 'Regular'
            WHEN lpp.pass_count BETWEEN 21 AND 50 THEN 'Frequent'
            ELSE 'Very Frequent'
        END as frequency_category,
        EXTRACT(DAYS FROM (lpp.last_seen - lpp.first_seen))::INTEGER as days_active
    FROM license_plate_passes lpp
    ORDER BY lpp.pass_count DESC, lpp.last_seen DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to search vehicles by partial plate number
CREATE OR REPLACE FUNCTION search_vehicles(p_search_term VARCHAR(20))
RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    match_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.first_seen,
        lpp.last_seen,
        CASE 
            WHEN lpp.plate_number = UPPER(p_search_term) THEN 'Exact Match'
            WHEN lpp.plate_number LIKE UPPER(p_search_term) || '%' THEN 'Starts With'
            WHEN lpp.plate_number LIKE '%' || UPPER(p_search_term) || '%' THEN 'Contains'
            ELSE 'Other'
        END as match_type
    FROM license_plate_passes lpp
    WHERE lpp.plate_number ILIKE '%' || p_search_term || '%'
    ORDER BY 
        CASE 
            WHEN lpp.plate_number = UPPER(p_search_term) THEN 1
            WHEN lpp.plate_number LIKE UPPER(p_search_term) || '%' THEN 2
            ELSE 3
        END,
        lpp.last_seen DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get hourly activity statistics
CREATE OR REPLACE FUNCTION get_hourly_activity_stats(p_days INTEGER DEFAULT 7)
RETURNS TABLE (
    hour_of_day INTEGER,
    total_detections BIGINT,
    unique_vehicles BIGINT,
    avg_detections_per_vehicle NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        EXTRACT(HOUR FROM lpp.last_seen)::INTEGER as hour_of_day,
        COUNT(*) as total_detections,
        COUNT(DISTINCT lpp.plate_number) as unique_vehicles,
        ROUND(COUNT(*)::NUMERIC / NULLIF(COUNT(DISTINCT lpp.plate_number), 0), 2) as avg_detections_per_vehicle
    FROM license_plate_passes lpp
    WHERE lpp.last_seen >= CURRENT_DATE - (p_days || ' days')::INTERVAL
    GROUP BY EXTRACT(HOUR FROM lpp.last_seen)
    ORDER BY hour_of_day;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old records
CREATE OR REPLACE FUNCTION cleanup_old_records(p_days_to_keep INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    v_deleted_count INTEGER;
BEGIN
    DELETE FROM license_plate_passes
    WHERE last_seen < CURRENT_DATE - (p_days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    
    -- Update table statistics after cleanup
    ANALYZE license_plate_passes;
    
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to export data for a specific date range
CREATE OR REPLACE FUNCTION export_data_range(
    p_start_date DATE,
    p_end_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    total_days_active INTEGER,
    state_code VARCHAR(2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.first_seen,
        lpp.last_seen,
        EXTRACT(DAYS FROM (lpp.last_seen - lpp.first_seen))::INTEGER as total_days_active,
        LEFT(lpp.plate_number, 2) as state_code
    FROM license_plate_passes lpp
    WHERE DATE(lpp.last_seen) BETWEEN p_start_date AND p_end_date
    ORDER BY lpp.last_seen DESC;
END;
$$ LANGUAGE plpgsql;

-- Stored procedure for batch tracking (useful for processing multiple detections)
CREATE OR REPLACE FUNCTION batch_track_plates(
    p_plates TEXT[], -- Array of plate numbers
    p_detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
)
RETURNS JSONB AS $$
DECLARE
    v_plate TEXT;
    v_results JSONB := '[]'::JSONB;
    v_single_result JSONB;
    v_processed_count INTEGER := 0;
    v_new_plates_count INTEGER := 0;
BEGIN
    -- Process each plate in the array
    FOREACH v_plate IN ARRAY p_plates
    LOOP
        -- Track individual plate
        SELECT track_license_plate_pass(v_plate, p_detection_timestamp) INTO v_single_result;
        
        -- Add to results array
        v_results := v_results || v_single_result;
        
        -- Count statistics
        v_processed_count := v_processed_count + 1;
        IF (v_single_result->>'is_new_plate')::BOOLEAN THEN
            v_new_plates_count := v_new_plates_count + 1;
        END IF;
    END LOOP;
    
    -- Return summary with individual results
    RETURN jsonb_build_object(
        'processed_count', v_processed_count,
        'new_plates_count', v_new_plates_count,
        'existing_plates_count', v_processed_count - v_new_plates_count,
        'processed_at', p_detection_timestamp,
        'individual_results', v_results
    );
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for performance (refresh periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vehicle_statistics AS
SELECT 
    DATE_TRUNC('day', last_seen) as activity_date,
    COUNT(*) as vehicles_seen,
    COUNT(DISTINCT plate_number) as unique_vehicles,
    SUM(pass_count) as total_passes,
    AVG(pass_count) as avg_passes_per_vehicle,
    COUNT(CASE WHEN pass_count = 1 THEN 1 END) as new_vehicles,
    COUNT(CASE WHEN pass_count > 1 THEN 1 END) as returning_vehicles
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE_TRUNC('day', last_seen)
ORDER BY activity_date DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_mv_vehicle_statistics_date ON mv_vehicle_statistics(activity_date DESC);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_vehicle_statistics()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_vehicle_statistics;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions on all new functions
GRANT EXECUTE ON FUNCTION get_tracking_summary() TO anpr_user;
GRANT EXECUTE ON FUNCTION get_vehicles_by_state(VARCHAR) TO anpr_user;
GRANT EXECUTE ON FUNCTION get_recent_activity(INTEGER) TO anpr_user;
GRANT EXECUTE ON FUNCTION get_top_frequent_vehicles(INTEGER) TO anpr_user;
GRANT EXECUTE ON FUNCTION search_vehicles(VARCHAR) TO anpr_user;
GRANT EXECUTE ON FUNCTION get_hourly_activity_stats(INTEGER) TO anpr_user;
GRANT EXECUTE ON FUNCTION cleanup_old_records(INTEGER) TO anpr_user;
GRANT EXECUTE ON FUNCTION export_data_range(DATE, DATE) TO anpr_user;
GRANT EXECUTE ON FUNCTION batch_track_plates(TEXT[], TIMESTAMP WITH TIME ZONE) TO anpr_user;
GRANT EXECUTE ON FUNCTION refresh_vehicle_statistics() TO anpr_user;

-- Grant permissions on materialized view
GRANT SELECT ON mv_vehicle_statistics TO anpr_user;

-- Add comments for documentation
COMMENT ON FUNCTION get_tracking_summary() IS 'Get overall system statistics and summary';
COMMENT ON FUNCTION get_vehicles_by_state(VARCHAR) IS 'Get all vehicles from a specific state (e.g., DL, MH, KA)';
COMMENT ON FUNCTION get_recent_activity(INTEGER) IS 'Get vehicles seen within specified hours';
COMMENT ON FUNCTION get_top_frequent_vehicles(INTEGER) IS 'Get most frequently seen vehicles with categories';
COMMENT ON FUNCTION search_vehicles(VARCHAR) IS 'Search vehicles by partial plate number';
COMMENT ON FUNCTION batch_track_plates(TEXT[], TIMESTAMP WITH TIME ZONE) IS 'Process multiple vehicle detections in one call';
COMMENT ON MATERIALIZED VIEW mv_vehicle_statistics IS 'Pre-computed daily statistics for better performance';
