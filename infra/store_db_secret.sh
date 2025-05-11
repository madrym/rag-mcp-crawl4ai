#!/bin/bash
set -e

# Usage: ./store_db_secret.sh db_secret.json
if [ $# -ne 1 ]; then
  echo "Usage: $0 db_secret.json"
  exit 1
fi

DB_SECRET_NAME="${DB_SECRET_NAME:-mcp/crawl4ai/db_credentials}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"

aws secretsmanager create-secret --name "$DB_SECRET_NAME" --region "$AWS_REGION" --secret-string file://$1 