-- Migration: License Plate Pass Tracking System
-- This implements the core business logic for tracking license plate passes
-- 
-- Business Logic:
-- - If plate already exists → increment pass_count and update last_seen
-- - If plate doesn't exist → insert new record with pass_count = 1
-- - Track: plate_number, pass_count, first_seen, last_seen

-- Create license_plate_passes table for core business logic
CREATE TABLE IF NOT EXISTS license_plate_passes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_number VARCHAR(20) NOT NULL UNIQUE,
    pass_count INTEGER NOT NULL DEFAULT 1 CHECK (pass_count > 0),
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_license_plate_passes_plate_number ON license_plate_passes(plate_number);
CREATE INDEX IF NOT EXISTS idx_license_plate_passes_last_seen ON license_plate_passes(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_license_plate_passes_pass_count ON license_plate_passes(pass_count DESC);
CREATE INDEX IF NOT EXISTS idx_license_plate_passes_first_seen ON license_plate_passes(first_seen);

-- Create trigger for updated_at
CREATE TRIGGER update_license_plate_passes_updated_at
    BEFORE UPDATE ON license_plate_passes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to handle license plate pass tracking
-- This implements the core business logic
CREATE OR REPLACE FUNCTION track_license_plate_pass(
    p_plate_number VARCHAR(20),
    p_detection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_pass_count INTEGER;
    v_first_seen TIMESTAMP WITH TIME ZONE;
    v_last_seen TIMESTAMP WITH TIME ZONE;
    v_is_new_plate BOOLEAN;
BEGIN
    -- Try to increment existing plate pass count
    UPDATE license_plate_passes 
    SET 
        pass_count = pass_count + 1,
        last_seen = p_detection_timestamp,
        updated_at = CURRENT_TIMESTAMP
    WHERE plate_number = p_plate_number
    RETURNING pass_count, first_seen, last_seen INTO v_pass_count, v_first_seen, v_last_seen;
    
    -- If plate exists, we updated it
    IF FOUND THEN
        v_is_new_plate := FALSE;
    ELSE
        -- Plate doesn't exist, insert new record
        INSERT INTO license_plate_passes (
            plate_number,
            pass_count,
            first_seen,
            last_seen
        ) VALUES (
            p_plate_number,
            1,
            p_detection_timestamp,
            p_detection_timestamp
        )
        RETURNING pass_count, first_seen, last_seen INTO v_pass_count, v_first_seen, v_last_seen;
        
        v_is_new_plate := TRUE;
    END IF;
    
    -- Return structured result
    v_result := jsonb_build_object(
        'plate_number', p_plate_number,
        'pass_count', v_pass_count,
        'first_seen', v_first_seen,
        'last_seen', v_last_seen,
        'is_new_plate', v_is_new_plate,
        'processed_at', CURRENT_TIMESTAMP
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Create function to get license plate statistics
CREATE OR REPLACE FUNCTION get_license_plate_stats(
    p_plate_number VARCHAR(20) DEFAULT NULL,
    p_limit INTEGER DEFAULT 100
) RETURNS TABLE (
    plate_number VARCHAR(20),
    pass_count INTEGER,
    first_seen TIMESTAMP WITH TIME ZONE,
    last_seen TIMESTAMP WITH TIME ZONE,
    days_since_first_seen INTEGER,
    hours_since_last_seen NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lpp.plate_number,
        lpp.pass_count,
        lpp.first_seen,
        lpp.last_seen,
        EXTRACT(DAYS FROM (CURRENT_TIMESTAMP - lpp.first_seen))::INTEGER as days_since_first_seen,
        ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - lpp.last_seen)) / 3600, 2) as hours_since_last_seen
    FROM license_plate_passes lpp
    WHERE (p_plate_number IS NULL OR lpp.plate_number = p_plate_number)
    ORDER BY lpp.last_seen DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Create view for pass tracking analytics
CREATE OR REPLACE VIEW license_plate_analytics AS
SELECT 
    plate_number,
    pass_count,
    first_seen,
    last_seen,
    EXTRACT(DAYS FROM (last_seen - first_seen)) as days_active,
    CASE 
        WHEN pass_count = 1 THEN 'Single Pass'
        WHEN pass_count BETWEEN 2 AND 5 THEN 'Occasional'
        WHEN pass_count BETWEEN 6 AND 20 THEN 'Regular'
        WHEN pass_count BETWEEN 21 AND 50 THEN 'Frequent'
        ELSE 'Very Frequent'
    END as frequency_category,
    ROUND(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600, 2) as hours_since_last_seen,
    CASE 
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600 < 1 THEN 'Just Now'
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600 < 24 THEN 'Today'
        WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - last_seen)) / 3600 < 168 THEN 'This Week'
        ELSE 'Older'
    END as recency_category
FROM license_plate_passes
ORDER BY last_seen DESC;

-- Create view for daily pass summary
CREATE OR REPLACE VIEW daily_pass_summary AS
SELECT 
    DATE(last_seen) as pass_date,
    COUNT(*) as total_plates_seen,
    SUM(CASE WHEN pass_count = 1 THEN 1 ELSE 0 END) as new_plates,
    SUM(CASE WHEN pass_count > 1 THEN 1 ELSE 0 END) as returning_plates,
    AVG(pass_count) as avg_pass_count,
    MAX(pass_count) as max_pass_count
FROM license_plate_passes
WHERE last_seen >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(last_seen)
ORDER BY pass_date DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON license_plate_passes TO anpr_user;
GRANT EXECUTE ON FUNCTION track_license_plate_pass(VARCHAR, TIMESTAMP WITH TIME ZONE) TO anpr_user;
GRANT EXECUTE ON FUNCTION get_license_plate_stats(VARCHAR, INTEGER) TO anpr_user;

COMMENT ON TABLE license_plate_passes IS 'Core business logic table for tracking license plate passes';
COMMENT ON COLUMN license_plate_passes.plate_number IS 'Normalized license plate number (e.g., DL 01 AB 1234)';
COMMENT ON COLUMN license_plate_passes.pass_count IS 'Number of times this plate has been detected';
COMMENT ON COLUMN license_plate_passes.first_seen IS 'First time this plate was detected';
COMMENT ON COLUMN license_plate_passes.last_seen IS 'Most recent detection of this plate';
COMMENT ON FUNCTION track_license_plate_pass IS 'Core function: increment pass count or insert new plate record';
