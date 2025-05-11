# Makefile for AWS CloudFormation RDS Stack Management

STACK_NAME ?= mcp-crawl4ai-stack
TEMPLATE_FILE ?= infra/rds-pgvector-cloudformation.yaml
DEPLOY_SCRIPT ?= infra/aws-cf-deploy.sh
UPDATE_SG_IP_SCRIPT ?= infra/aws-update-sg-ip.sh

# Set these variables in your environment or .env file
AWS_REGION ?= ap-southeast-2
DB_HOST ?= mcp-crawl4ai-rds.<rds-instance-id>.ap-southeast-2.rds.amazonaws.com
DB_PORT ?= 5432
DB_NAME ?= crawl4ai_db
DB_USER ?= mcp_admin

# Use AWS CLI to get password from Secrets Manager
DB_SECRET_NAME ?= mcp/crawl4ai/db_credentials

# Deploy or update the stack using the deploy script
.PHONY: deploy
deploy:
	bash $(DEPLOY_SCRIPT)

# Delete the CloudFormation stack
.PHONY: delete
delete:
	aws cloudformation delete-stack --stack-name $(STACK_NAME)

# Describe the stack (full details)
.PHONY: describe
describe:
	aws cloudformation describe-stacks --stack-name $(STACK_NAME)

# Show stack outputs (endpoints, secrets, etc.)
.PHONY: outputs
outputs:
	aws cloudformation describe-stacks --stack-name $(STACK_NAME) --query "Stacks[0].Outputs"

# Update the RDS security group to allow your current IP
.PHONY: update-sg-ip
update-sg-ip:
	bash $(UPDATE_SG_IP_SCRIPT)

# Validate the CloudFormation template
.PHONY: validate
validate:
	aws cloudformation validate-template --template-body file://$(TEMPLATE_FILE)

# Show recent stack events (for troubleshooting)
.PHONY: events
events:
	aws cloudformation describe-stack-events --stack-name $(STACK_NAME) --max-items 10

.PHONY: psql connect enable-pgvector verify-pgvector store-secret get-secret get-secret-password init-schema run

psql:
	@echo "Connecting to RDS with psql..."
	PGPASSWORD=$$(make get-secret-password) psql --host=$(DB_HOST) --port=$(DB_PORT) --username=$(DB_USER) --dbname=$(DB_NAME)

connect: psql

enable-pgvector:
	@echo "Enabling pgvector extension..."
	PGPASSWORD=$$(make get-secret-password) psql --host=$(DB_HOST) --port=$(DB_PORT) --username=$(DB_USER) --dbname=$(DB_NAME) -c "CREATE EXTENSION IF NOT EXISTS vector;"

verify-pgvector:
	@echo "Verifying pgvector extension..."
	PGPASSWORD=$$(make get-secret-password) psql --host=$(DB_HOST) --port=$(DB_PORT) --username=$(DB_USER) --dbname=$(DB_NAME) -c "\\dx vector"

store-secret:
	@echo "Storing secret in AWS Secrets Manager..."
	aws secretsmanager create-secret --name $(DB_SECRET_NAME) --region $(AWS_REGION) --secret-string file://infra/db_secret.json

delete-secret:
	@echo "Deleting secret from AWS Secrets Manager..."
	aws secretsmanager delete-secret --secret-id $(DB_SECRET_NAME) --region $(AWS_REGION)

get-secret:
	@aws secretsmanager get-secret-value --secret-id $(DB_SECRET_NAME) --region $(AWS_REGION) --query SecretString --output text

get-secret-password:
	@aws secretsmanager get-secret-value --secret-id $(DB_SECRET_NAME) --region $(AWS_REGION) --query SecretString --output text | jq -r .password

init-schema:
	@echo "Initializing crawled_pages schema in RDS..."
	PGPASSWORD=$$(make get-secret-password) psql --host=$(DB_HOST) --port=$(DB_PORT) --username=$(DB_USER) --dbname=$(DB_NAME) -f infra/rds-scripts/create_crawled_pages_schema.sql

run:
	uv run src/crawl4ai_mcp.py 