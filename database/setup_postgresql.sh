#!/bin/bash

# PostgreSQL Setup Script for ANPR License Plate Tracking System
# This script creates the complete database schema for vehicle tracking

set -e  # Exit on any error

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-anpr_db}"
DB_USER="${DB_USER:-anpr_user}"
DB_PASSWORD="${DB_PASSWORD:-anpr_password}"
DB_ADMIN_USER="${DB_ADMIN_USER:-postgres}"

echo "🚀 Setting up PostgreSQL database for ANPR License Plate Tracking"
echo "=================================================="
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Host: $DB_HOST:$DB_PORT"
echo ""

# Function to execute SQL with admin privileges
execute_sql_admin() {
    local sql_file=$1
    echo "📄 Executing $sql_file as admin..."
    PGPASSWORD="$DB_ADMIN_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_ADMIN_USER" -d postgres -f "$sql_file"
}

# Function to execute SQL with app user
execute_sql_user() {
    local sql_file=$1
    echo "📄 Executing $sql_file as $DB_USER..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file"
}

# Create database and user setup script
cat > /tmp/setup_db.sql << EOF
-- Create database and user for ANPR system
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Connect to the new database and grant permissions
\c $DB_NAME;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant permissions to user
GRANT ALL PRIVILEGES ON SCHEMA public TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO $DB_USER;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO $DB_USER;
EOF

# Step 1: Create database and user (requires admin privileges)
echo "🔧 Step 1: Creating database and user..."
if [ ! -z "$DB_ADMIN_PASSWORD" ]; then
    execute_sql_admin /tmp/setup_db.sql
else
    echo "⚠️  DB_ADMIN_PASSWORD not set. Please create database and user manually:"
    echo "   CREATE DATABASE $DB_NAME;"
    echo "   CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASSWORD';"
    echo "   GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    echo ""
    echo "Press Enter when done..."
    read
fi

# Step 2: Run initial schema setup
echo "🔧 Step 2: Setting up initial schema..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "init.sql"

# Step 3: Run license plate tracking migration
echo "🔧 Step 3: Setting up license plate tracking..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "migrations/001_license_plate_tracking.sql"

# Step 4: Run additional functions migration
echo "🔧 Step 4: Adding utility functions..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "migrations/002_additional_functions.sql"

# Step 5: Verify setup
echo "🔍 Step 5: Verifying database setup..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
SELECT 'Database setup verification:' as status;
SELECT 'Tables created: ' || COUNT(*) as tables FROM information_schema.tables WHERE table_schema = 'public';
SELECT 'Functions created: ' || COUNT(*) as functions FROM information_schema.routines WHERE routine_schema = 'public';

-- Test core functionality
SELECT 'Testing core function...' as test;
SELECT track_license_plate_pass('TEST 01 AB 1234');

-- Show test result
SELECT plate_number, pass_count, first_seen, last_seen 
FROM license_plate_passes 
WHERE plate_number = 'TEST 01 AB 1234';
"

echo ""
echo "✅ PostgreSQL database setup completed successfully!"
echo ""
echo "🔗 Connection Details:"
echo "   Host: $DB_HOST"
echo "   Port: $DB_PORT"
echo "   Database: $DB_NAME"
echo "   User: $DB_USER"
echo ""
echo "📊 Core Functions Available:"
echo "   • track_license_plate_pass(plate_number, timestamp)"
echo "   • get_license_plate_stats(plate_number, limit)"
echo "   • get_tracking_summary()"
echo "   • get_vehicles_by_state(state_code)"
echo "   • search_vehicles(search_term)"
echo ""
echo "📖 Documentation:"
echo "   • See README_SQL_QUERIES.md for usage examples"
echo "   • See common_queries.sql for ready-to-use queries"
echo ""
echo "🚀 Your ANPR License Plate Tracking Database is ready!"

# Clean up temporary files
rm -f /tmp/setup_db.sql
