#!/bin/bash

# Load .env if it exists
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

STACK_NAME="${STACK_NAME:-mcp-crawl4ai-stack}"

# Requires jq: install with 'brew install jq' if not present

# Get your new IP
NEW_IP=$(curl -s https://checkip.amazonaws.com)

# Get the security group ID from stack outputs
SG_ID=$(aws cloudformation describe-stacks --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='RDSSecurityGroupId'].OutputValue" \
    --output text)

# Remove old ingress rules for port 5432 with description 'Home IP Access'
aws ec2 describe-security-groups --group-ids $SG_ID \
  --query "SecurityGroups[0].IpPermissions[?FromPort==\`5432\` && ToPort==\`5432\`].IpRanges" \
  --output json | \
  jq -r '.[] | .[] | select(.Description=="Home IP Access") | .CidrIp' | \
  while read OLD_CIDR; do
    aws ec2 revoke-security-group-ingress --group-id $SG_ID --protocol tcp --port 5432 --cidr $OLD_CIDR
done

# Add new ingress rule
aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --protocol tcp \
    --port 5432 \
    --cidr ${NEW_IP}/32 \
    --description "Home IP Access"