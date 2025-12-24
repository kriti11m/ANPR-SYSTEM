# PostgreSQL Database Quick Start Guide

## 🚀 Quick Setup Options

### Option 1: Docker Compose (Recommended)

```bash
# Navigate to database directory
cd /Users/kritimaheshwari/Desktop/anpr-system/database

# Start PostgreSQL with Docker
docker-compose up -d

# Check if running
docker-compose ps

# View logs
docker-compose logs postgres
```

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `anpr_db`
- Username: `anpr_user`
- Password: `anpr_password`

### Option 2: Manual PostgreSQL Setup

```bash
# Run setup script
./setup_postgresql.sh

# Or set custom values
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=anpr_db
export DB_USER=anpr_user
export DB_PASSWORD=anpr_password
export DB_ADMIN_PASSWORD=your_postgres_password
./setup_postgresql.sh
```

## 🔧 Database Schema

### Core Table: `license_plate_passes`

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `plate_number` | VARCHAR(20) | License plate number |
| `pass_count` | INTEGER | Number of passes |
| `first_seen` | TIMESTAMP | First detection time |
| `last_seen` | TIMESTAMP | Latest detection time |
| `created_at` | TIMESTAMP | Record creation time |
| `updated_at` | TIMESTAMP | Last update time |

### Core Business Logic

```sql
-- Track a vehicle pass (automatically handles new/existing)
SELECT track_license_plate_pass('DL 01 AB 1234');

-- Get vehicle statistics
SELECT * FROM get_license_plate_stats();

-- Get system summary
SELECT * FROM get_tracking_summary();
```

## 📊 Common Operations

### 1. Track Vehicle Passes

```sql
-- Single vehicle
SELECT track_license_plate_pass('DL 01 AB 1234');

-- Multiple vehicles
SELECT batch_track_plates(
    ARRAY['DL 01 AB 1234', 'MH 12 CD 5678', 'KA 03 EF 9012']
);
```

### 2. Query Vehicle Data

```sql
-- All vehicles
SELECT * FROM license_plate_passes ORDER BY last_seen DESC;

-- Vehicles from Delhi
SELECT * FROM get_vehicles_by_state('DL');

-- Recent activity (last 6 hours)
SELECT * FROM get_recent_activity(6);

-- Top frequent vehicles
SELECT * FROM get_top_frequent_vehicles(10);

-- Search by partial plate
SELECT * FROM search_vehicles('1234');
```

### 3. Reporting Queries

```sql
-- Daily summary
SELECT * FROM daily_pass_summary LIMIT 7;

-- Hourly activity stats
SELECT * FROM get_hourly_activity_stats(7);

-- System overview
SELECT * FROM get_tracking_summary();
```

## 🔍 Testing the Setup

### Test Core Functionality

```sql
-- Test database connection
\conninfo

-- Test core tracking function
SELECT track_license_plate_pass('TEST 01 AB 1234');

-- Verify result
SELECT * FROM license_plate_passes WHERE plate_number = 'TEST 01 AB 1234';

-- Test increment functionality
SELECT track_license_plate_pass('TEST 01 AB 1234');
SELECT * FROM license_plate_passes WHERE plate_number = 'TEST 01 AB 1234';
```

### Test Analytics Functions

```sql
-- Test summary
SELECT * FROM get_tracking_summary();

-- Test search
SELECT * FROM search_vehicles('TEST');

-- Test state filter
SELECT * FROM get_vehicles_by_state('TE');
```

## 🛠️ Administration

### Database Management

```sql
-- Check table size
SELECT 
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes
FROM pg_stat_user_tables 
WHERE tablename = 'license_plate_passes';

-- Refresh materialized view (run daily)
SELECT refresh_vehicle_statistics();

-- Clean old records (older than 1 year)
SELECT cleanup_old_records(365);
```

### Backup and Restore

```bash
# Backup
pg_dump -h localhost -U anpr_user -d anpr_db > anpr_backup.sql

# Restore
psql -h localhost -U anpr_user -d anpr_db < anpr_backup.sql
```

## 📁 File Structure

```
database/
├── README_QUICKSTART.md          # This file
├── README_SQL_QUERIES.md         # Comprehensive SQL guide
├── docker-compose.yml            # Docker setup
├── setup_postgresql.sh           # Manual setup script
├── init.sql                      # Base schema
├── common_queries.sql             # Ready-to-use queries
├── migrations/
│   ├── 001_license_plate_tracking.sql
│   └── 002_additional_functions.sql
└── seeds/
    └── initial_data.sql
```

## 🔗 Integration with Application

### Python Example

```python
import asyncpg
import asyncio

async def track_vehicle(plate_number: str):
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='anpr_user',
        password='anpr_password',
        database='anpr_db'
    )
    
    result = await conn.fetchval(
        "SELECT track_license_plate_pass($1)",
        plate_number
    )
    
    await conn.close()
    return result

# Usage
result = asyncio.run(track_vehicle('DL 01 AB 1234'))
print(f"Pass count: {result['pass_count']}")
```

### Environment Variables

```bash
# .env file
DB_HOST=localhost
DB_PORT=5432
DB_NAME=anpr_db
DB_USER=anpr_user
DB_PASSWORD=anpr_password
DATABASE_URL=postgresql://anpr_user:anpr_password@localhost:5432/anpr_db
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection refused**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps
   # Or for manual install
   sudo systemctl status postgresql
   ```

2. **Permission denied**
   ```bash
   # Reset permissions
   docker-compose down
   docker volume rm database_postgres_data
   docker-compose up -d
   ```

3. **Function not found**
   ```bash
   # Re-run migrations
   psql -h localhost -U anpr_user -d anpr_db -f migrations/001_license_plate_tracking.sql
   ```

### Performance Optimization

```sql
-- Check index usage
SELECT * FROM pg_stat_user_indexes WHERE tablename = 'license_plate_passes';

-- Analyze table statistics
ANALYZE license_plate_passes;

-- Update table statistics
REINDEX TABLE license_plate_passes;
```

## 🎯 Next Steps

1. **Setup Database**: Choose Docker or manual setup
2. **Test Functionality**: Run test queries
3. **Integrate with Application**: Use connection details in your ANPR app
4. **Monitor Performance**: Set up regular maintenance tasks
5. **Setup Backups**: Schedule regular database backups

## 📞 Support

For issues or questions:
- Check the comprehensive SQL guide: `README_SQL_QUERIES.md`
- Review common queries: `common_queries.sql`
- Test with provided examples above
