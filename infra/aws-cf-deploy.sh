#!/bin/bash

STACK_NAME="mcp-crawl4ai-stack"
TEMPLATE_FILE="infra/rds-pgvector-cloudformation.yaml"

# Load .env if it exists
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# Prompt for required parameters if not set, or use defaults
if [ -z "$DBMasterUserPassword" ]; then
  read -p "Enter DB Master Password (min 8 chars): " -s DBMasterUserPassword
  echo
fi
if [ -z "$AccessCIDR" ]; then
  read -p "Enter Access CIDR (e.g., $(curl -s https://checkip.amazonaws.com)/32): " AccessCIDR
  AccessCIDR=${AccessCIDR:-"$(curl -s https://checkip.amazonaws.com)/32"}
fi

# Optional/with defaults
DBInstanceIdentifier="${DBInstanceIdentifier:-mcp-crawl4ai-rds}"
DBName="${DBName:-crawl4ai_db}"
DBMasterUsername="${DBMasterUsername:-mcp_admin}"
DBInstanceClass="${DBInstanceClass:-db.t3.micro}"
DBAllocatedStorage="${DBAllocatedStorage:-20}"
PostgresEngineVersion="${PostgresEngineVersion:-17.5}"
EnableIAMDatabaseAuthentication="${EnableIAMDatabaseAuthentication:-true}"
BackupRetentionPeriod="${BackupRetentionPeriod:-7}"
MultiAZDeployment="${MultiAZDeployment:-false}"

PARAMS="DBMasterUserPassword=$DBMasterUserPassword AccessCIDR=$AccessCIDR DBInstanceIdentifier=$DBInstanceIdentifier DBName=$DBName DBMasterUsername=$DBMasterUsername DBInstanceClass=$DBInstanceClass DBAllocatedStorage=$DBAllocatedStorage PostgresEngineVersion=$PostgresEngineVersion EnableIAMDatabaseAuthentication=$EnableIAMDatabaseAuthentication BackupRetentionPeriod=$BackupRetentionPeriod MultiAZDeployment=$MultiAZDeployment"

if [ -n "$AppServerSecurityGroupId" ]; then
  PARAMS="$PARAMS AppServerSecurityGroupId=$AppServerSecurityGroupId"
fi

aws cloudformation deploy \
  --template-file $TEMPLATE_FILE \
  --stack-name $STACK_NAME \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides $PARAMS