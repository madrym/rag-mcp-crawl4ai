# MCP Crawl4AI Infrastructure Setup

This README provides step-by-step instructions for provisioning and managing the AWS infrastructure for the MCP Crawl4AI server using CloudFormation, RDS, and supporting scripts. All commands can be run via the Makefile or directly as bash scripts.

---

## Prerequisites

- **AWS CLI** installed and configured (`aws configure`).
- **jq** installed (`brew install jq` or `sudo apt-get install jq`).
- **psql** (PostgreSQL client) installed.
- **Your AWS credentials** must have permissions for CloudFormation, RDS, EC2, and Secrets Manager.
- **.env file** (optional, but recommended) in the project root with any custom values for parameters (see below).

---

## 1. Configure Environment Variables

Create a `.env` file in the project root (or export variables in your shell) for any parameters you want to override. Example:

```
DBMasterUserPassword=yourStrongPassword
AccessCIDR=YOUR_IP/32
DBInstanceIdentifier=mcp-crawl4ai-rds
DBName=crawl4ai_db
DBMasterUsername=mcp_admin
DBInstanceClass=db.t3.micro
DBAllocatedStorage=20
PostgresEngineVersion=17.5
EnableIAMDatabaseAuthentication=true
BackupRetentionPeriod=7
MultiAZDeployment=false
AWS_REGION=ap-southeast-2
```

> **Tip:** The deploy script will prompt for any required values not set in your environment.

---

## 2. Deploy the CloudFormation Stack

This will provision the VPC, subnets, security group, and RDS instance.

```
make deploy
```
- This runs `infra/aws-cf-deploy.sh`, which will prompt for any missing required parameters.
- The stack may take several minutes to complete.

---

## 3. Check Stack Status and Outputs

- **Check stack status:**
  ```
  make describe
  ```
- **Show stack outputs (endpoints, secrets, etc.):**
  ```
  make outputs
  ```
- **Show recent stack events:**
  ```
  make events
  ```

---

## 4. Update Security Group for Your IP (if needed)

If your IP changes, update the RDS security group:
```
make update-sg-ip
```

---

## 5. Store RDS Credentials in AWS Secrets Manager

1. Edit `infra/db_secret.json` with your actual credentials (if not using CloudFormation-managed secret).
2. Store the secret:
   ```
   make store-secret
   # or
   bash infra/store_db_secret.sh infra/db_secret.json
   ```

---

## 6. Connect to the RDS Database

- **Connect using psql:**
  ```
  make psql
  # or
  bash infra/psql_connect.sh
  ```

---

## 7. Enable and Verify pgvector Extension

- **Enable pgvector:**
  ```
  make enable-pgvector
  # or
  bash infra/enable_pgvector.sh
  ```
- **Verify pgvector:**
  ```
  make verify-pgvector
  # or
  bash infra/verify_pgvector.sh
  ```

---

## 8. Clean Up

- **Delete the CloudFormation stack:**
  ```
  make delete
  ```

---

## Notes
- All Makefile targets can be run directly as `make <target>`.
- All scripts in this folder can be run directly as bash scripts if you prefer.
- The region defaults to `ap-southeast-2` but can be overridden in your `.env` or shell.
- For advanced configuration, edit the CloudFormation template (`infra/rds-pgvector-cloudformation.yaml`).

---

## Troubleshooting
- If you get permission errors, check your AWS credentials and IAM permissions.
- If you get connection errors, check your security group rules and `AccessCIDR`.
- For any issues, check stack events: `make events`. 