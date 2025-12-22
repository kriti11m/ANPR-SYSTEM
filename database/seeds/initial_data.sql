-- Seed data for ANPR system testing
-- This file contains sample data for development and testing

-- Insert sample feeds
INSERT INTO feeds (name, source_type, source_url, description, is_active) VALUES
('Main Entrance Camera', 'rtsp', 'rtsp://192.168.1.100:554/stream1', 'Primary entrance monitoring camera', true),
('Exit Gate Camera', 'rtsp', 'rtsp://192.168.1.101:554/stream1', 'Exit gate monitoring camera', true),
('Parking Area Camera', 'webcam', '0', 'USB camera monitoring parking area', false),
('Sample Video File', 'file', '/uploads/sample_traffic.mp4', 'Sample video file for testing', false);

-- Insert sample detections (using the feed IDs from above)
WITH feed_ids AS (
  SELECT id, name FROM feeds
),
main_feed AS (SELECT id FROM feed_ids WHERE name = 'Main Entrance Camera' LIMIT 1),
exit_feed AS (SELECT id FROM feed_ids WHERE name = 'Exit Gate Camera' LIMIT 1)

INSERT INTO detections (license_plate, confidence, bounding_box, image_path, feed_id, timestamp) 
SELECT * FROM (
  VALUES
    ('ABC123', 0.95, '{"x1": 100, "y1": 200, "x2": 300, "y2": 250}', '/images/detection_001.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '5 minutes'),
    ('XYZ789', 0.87, '{"x1": 150, "y1": 180, "x2": 350, "y2": 230}', '/images/detection_002.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '10 minutes'),
    ('DEF456', 0.92, '{"x1": 120, "y1": 190, "x2": 320, "y2": 240}', '/images/detection_003.jpg', (SELECT id FROM exit_feed), NOW() - INTERVAL '15 minutes'),
    ('GHI789', 0.88, '{"x1": 110, "y1": 195, "x2": 310, "y2": 245}', '/images/detection_004.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '20 minutes'),
    ('JKL012', 0.90, '{"x1": 140, "y1": 185, "x2": 340, "y2": 235}', '/images/detection_005.jpg', (SELECT id FROM exit_feed), NOW() - INTERVAL '25 minutes'),
    ('MNO345', 0.94, '{"x1": 130, "y1": 175, "x2": 330, "y2": 225}', '/images/detection_006.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '30 minutes'),
    ('PQR678', 0.86, '{"x1": 125, "y1": 205, "x2": 325, "y2": 255}', '/images/detection_007.jpg', (SELECT id FROM exit_feed), NOW() - INTERVAL '35 minutes'),
    ('STU901', 0.91, '{"x1": 135, "y1": 170, "x2": 335, "y2": 220}', '/images/detection_008.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '40 minutes'),
    ('VWX234', 0.89, '{"x1": 145, "y1": 200, "x2": 345, "y2": 250}', '/images/detection_009.jpg', (SELECT id FROM exit_feed), NOW() - INTERVAL '45 minutes'),
    ('YZA567', 0.93, '{"x1": 115, "y1": 188, "x2": 315, "y2": 238}', '/images/detection_010.jpg', (SELECT id FROM main_feed), NOW() - INTERVAL '50 minutes')
) AS t(license_plate, confidence, bounding_box, image_path, feed_id, timestamp);

-- Insert sample processing jobs
INSERT INTO processing_jobs (job_type, status, input_data, result_data, created_at, started_at, completed_at) VALUES
(
  'single_image',
  'completed',
  '{"image_path": "/uploads/test_image_1.jpg", "confidence_threshold": 0.5}',
  '{"detections": [{"license_plate": "ABC123", "confidence": 0.95}], "processing_time": 0.34}',
  NOW() - INTERVAL '1 hour',
  NOW() - INTERVAL '1 hour' + INTERVAL '2 seconds',
  NOW() - INTERVAL '1 hour' + INTERVAL '5 seconds'
),
(
  'batch',
  'completed',
  '{"image_paths": ["/uploads/batch_1.jpg", "/uploads/batch_2.jpg"], "confidence_threshold": 0.6}',
  '{"detections": [{"license_plate": "XYZ789", "confidence": 0.87}, {"license_plate": "DEF456", "confidence": 0.92}], "processing_time": 0.68}',
  NOW() - INTERVAL '2 hours',
  NOW() - INTERVAL '2 hours' + INTERVAL '1 second',
  NOW() - INTERVAL '2 hours' + INTERVAL '8 seconds'
),
(
  'real_time',
  'processing',
  '{"feed_id": "main_entrance", "duration": 3600}',
  NULL,
  NOW() - INTERVAL '10 minutes',
  NOW() - INTERVAL '10 minutes' + INTERVAL '3 seconds',
  NULL
);

-- Insert sample system logs
INSERT INTO system_logs (level, module, message, extra_data, timestamp) VALUES
('INFO', 'anpr.processing', 'Successfully processed license plate detection', '{"license_plate": "ABC123", "confidence": 0.95, "processing_time": 0.34}', NOW() - INTERVAL '5 minutes'),
('INFO', 'anpr.feeds', 'Feed started successfully', '{"feed_id": "main_entrance", "feed_name": "Main Entrance Camera"}', NOW() - INTERVAL '10 minutes'),
('WARNING', 'anpr.feeds', 'RTSP stream connection timeout, retrying', '{"feed_id": "exit_gate", "attempt": 2, "timeout": 30}', NOW() - INTERVAL '15 minutes'),
('ERROR', 'anpr.database', 'Database connection failed, retrying', '{"error": "connection timeout", "retry_count": 1}', NOW() - INTERVAL '20 minutes'),
('INFO', 'anpr.auth', 'User login successful', '{"user_email": "admin@anpr.com", "ip_address": "192.168.1.50"}', NOW() - INTERVAL '25 minutes'),
('INFO', 'anpr.processing', 'Batch processing completed', '{"job_id": "batch_001", "images_processed": 50, "detections_found": 23, "total_time": 15.6}', NOW() - INTERVAL '30 minutes'),
('WARNING', 'anpr.vision', 'Low detection confidence', '{"license_plate": "UNCLEAR1", "confidence": 0.45, "threshold": 0.5}', NOW() - INTERVAL '35 minutes'),
('INFO', 'anpr.system', 'System backup completed', '{"backup_size": "150MB", "backup_location": "/backups/anpr_20231201.sql"}', NOW() - INTERVAL '40 minutes');

-- Create additional test user
INSERT INTO users (email, password_hash, full_name, is_active) VALUES
('operator@anpr.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'System Operator', true)
ON CONFLICT (email) DO NOTHING;
