-- Initial database setup for ANPR system
-- Run this script to create the initial database structure

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create feeds table
CREATE TABLE IF NOT EXISTS feeds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('webcam', 'rtsp', 'file')),
    source_url VARCHAR(1000) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_url)
);

-- Create detections table
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    license_plate VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    bounding_box JSONB NOT NULL,
    image_path VARCHAR(500),
    feed_id UUID NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create processing_jobs table
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('real_time', 'batch', 'single_image')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    input_data JSONB NOT NULL,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create system_logs table for application logging
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    extra_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_detections_feed_id ON detections(feed_id);
CREATE INDEX IF NOT EXISTS idx_detections_license_plate ON detections(license_plate);
CREATE INDEX IF NOT EXISTS idx_detections_feed_timestamp ON detections(feed_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_created_at ON processing_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_feeds_updated_at
    BEFORE UPDATE ON feeds
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for detection statistics
CREATE OR REPLACE VIEW detection_stats AS
SELECT 
    f.id as feed_id,
    f.name as feed_name,
    f.source_type,
    COUNT(d.id) as total_detections,
    AVG(d.confidence) as avg_confidence,
    MAX(d.timestamp) as last_detection,
    MIN(d.timestamp) as first_detection
FROM feeds f
LEFT JOIN detections d ON f.id = d.feed_id
GROUP BY f.id, f.name, f.source_type;

-- Create view for daily detection summary
CREATE OR REPLACE VIEW daily_detection_summary AS
SELECT 
    DATE(timestamp) as detection_date,
    COUNT(*) as total_detections,
    COUNT(DISTINCT license_plate) as unique_plates,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT feed_id) as active_feeds
FROM detections
GROUP BY DATE(timestamp)
ORDER BY detection_date DESC;

-- Insert initial admin user (password: admin123)
INSERT INTO users (email, password_hash, full_name) 
VALUES (
    'admin@anpr.com', 
    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',
    'System Administrator'
) ON CONFLICT (email) DO NOTHING;

-- Grant privileges
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO anpr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO anpr_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anpr_user;
