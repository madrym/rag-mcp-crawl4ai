#!/bin/bash
set -e

[ -f .env ] && export $(grep -v '^#' .env | xargs)

DB_HOST="${DB_HOST:-your-rds-endpoint.rds.amazonaws.com}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-crawl4ai_db}"
DB_USER="${DB_USER:-mcp_admin}"
DB_SECRET_NAME="${DB_SECRET_NAME:-mcp/crawl4ai/db_credentials}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"

DB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id "$DB_SECRET_NAME" --region "$AWS_REGION" --query SecretString --output text | jq -r .password)

echo "Enabling pgvector extension..."
PGPASSWORD="$DB_PASSWORD" psql --host="$DB_HOST" --port="$DB_PORT" --username="$DB_USER" --dbname="$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" 