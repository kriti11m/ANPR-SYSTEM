# Database Configuration and Migration README

PostgreSQL database setup for the ANPR system with schemas, migrations, and seed data.

## 🏗️ Structure

```
database/
├── migrations/        # Database migration scripts
├── seeds/            # Seed data for testing
├── schemas/          # SQL schema definitions
├── init.sql          # Initial database setup
├── docker-compose.yml # PostgreSQL container
└── README.md         # This file
```

## 📊 Database Schema

### Tables

#### users
- `id` - Primary key (UUID)
- `email` - User email (unique)
- `password_hash` - Hashed password
- `full_name` - User's full name
- `is_active` - Account status
- `created_at` - Account creation timestamp
- `updated_at` - Last update timestamp

#### feeds
- `id` - Primary key (UUID)
- `name` - Feed display name
- `source_type` - Type: webcam, rtsp, file
- `source_url` - Source URL/path
- `description` - Optional description
- `is_active` - Feed status
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp

#### detections
- `id` - Primary key (UUID)
- `license_plate` - Detected license plate text
- `confidence` - Detection confidence (0.0 - 1.0)
- `bounding_box` - JSON with x1, y1, x2, y2 coordinates
- `image_path` - Path to saved image
- `feed_id` - Foreign key to feeds table
- `timestamp` - Detection timestamp
- `created_at` - Record creation timestamp

#### processing_jobs
- `id` - Primary key (UUID)
- `job_type` - Type: real_time, batch, single_image
- `status` - Status: pending, processing, completed, failed
- `input_data` - JSON with job parameters
- `result_data` - JSON with job results
- `error_message` - Error details if failed
- `created_at` - Job creation timestamp
- `started_at` - Processing start timestamp
- `completed_at` - Processing completion timestamp

## 🛠️ Setup

### Local PostgreSQL
```bash
# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Create database
createdb anpr_db

# Create user
createuser -P anpr_user
# Password: anpr_pass

# Grant privileges
psql -d anpr_db -c "GRANT ALL PRIVILEGES ON DATABASE anpr_db TO anpr_user;"
```

### Docker Setup
```bash
cd database
docker-compose up -d
```

### Run Migrations
```bash
# From backend directory
cd backend
alembic upgrade head
```

## 🔧 Migration Commands

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Show current revision
alembic current

# Show migration history
alembic history
```

## 🌱 Seed Data

```bash
# Load seed data
psql -d anpr_db -f database/seeds/initial_data.sql
```

## 📊 Database Monitoring

### Useful Queries

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(tablename::text)) as size
FROM pg_tables 
WHERE schemaname = 'public';

-- Recent detections
SELECT 
    license_plate,
    confidence,
    f.name as feed_name,
    timestamp
FROM detections d
JOIN feeds f ON d.feed_id = f.id
ORDER BY timestamp DESC
LIMIT 10;

-- Detection statistics by feed
SELECT 
    f.name as feed_name,
    COUNT(*) as detection_count,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as last_detection
FROM detections d
JOIN feeds f ON d.feed_id = f.id
GROUP BY f.id, f.name
ORDER BY detection_count DESC;
```

## 🔐 Security

### Connection String Format
```
postgresql://username:password@host:port/database
```

### Environment Variables
- `DATABASE_URL` - Full connection string
- `DB_HOST` - Database host
- `DB_PORT` - Database port (default: 5432)
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password

## 🚀 Performance Optimization

### Indexes
```sql
-- Index on detections timestamp for fast queries
CREATE INDEX idx_detections_timestamp ON detections(timestamp DESC);

-- Index on detections feed_id for joins
CREATE INDEX idx_detections_feed_id ON detections(feed_id);

-- Index on license plate for search
CREATE INDEX idx_detections_license_plate ON detections(license_plate);

-- Composite index for common queries
CREATE INDEX idx_detections_feed_timestamp ON detections(feed_id, timestamp DESC);
```

### Connection Pooling
Configure connection pooling in the backend:
- Pool size: 5-10 connections
- Max overflow: 10 connections
- Pool timeout: 30 seconds

## 📦 Backup and Restore

### Backup
```bash
# Full backup
pg_dump -h localhost -U anpr_user -d anpr_db > backup.sql

# Schema only
pg_dump -h localhost -U anpr_user -d anpr_db --schema-only > schema.sql

# Data only
pg_dump -h localhost -U anpr_user -d anpr_db --data-only > data.sql
```

### Restore
```bash
# Restore from backup
psql -h localhost -U anpr_user -d anpr_db < backup.sql
```

### Automated Backups
Set up daily backups with cron:
```bash
# Add to crontab
0 2 * * * pg_dump -h localhost -U anpr_user anpr_db | gzip > /backups/anpr_$(date +\%Y\%m\%d).sql.gz
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection refused**
   - Check PostgreSQL is running
   - Verify connection parameters
   - Check firewall settings

2. **Authentication failed**
   - Verify username/password
   - Check pg_hba.conf settings

3. **Database doesn't exist**
   - Create database: `createdb anpr_db`

4. **Permission denied**
   - Grant privileges to user
   - Check database ownership

### Useful Commands
```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Connect to database
psql -h localhost -U anpr_user -d anpr_db

# List databases
\l

# List tables
\dt

# Describe table
\d table_name

# Show current connections
SELECT * FROM pg_stat_activity;
```
