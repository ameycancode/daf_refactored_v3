# =============================================================================
# GITHUB ACTIONS CONFIGURATION TEMPLATES
# =============================================================================

# Repository Variables Template - .github/templates/repository-variables.md
---
# Energy Forecasting MLOps - Repository Variables Configuration

## Required Repository Variables

Configure these variables in your GitHub repository settings under **Settings > Secrets and variables > Actions > Variables**:

### AWS Configuration
| Variable Name | Description | Example Value |
|---------------|-------------|---------------|
| `AWS_ACCOUNT_ID` | AWS Account ID | `123456789012` |
| `AWS_REGION` | Primary AWS Region | `us-west-2` |

### Environment-Specific Variables (if using environment protection rules)
| Variable Name | Description | Dev Value | Preprod Value | Prod Value |
|---------------|-------------|-----------|---------------|------------|
| `SAGEMAKER_ROLE_NAME` | SageMaker execution role | `sdcp-dev-sagemaker-...` | `sdcp-preprod-sagemaker-...` | `sdcp-prod-sagemaker-...` |
| `DATA_BUCKET` | S3 data bucket name | `sdcp-dev-sagemaker-...` | `sdcp-preprod-sagemaker-...` | `sdcp-prod-sagemaker-...` |
| `MODEL_BUCKET` | S3 model bucket name | `sdcp-dev-sagemaker-...` | `sdcp-preprod-sagemaker-...` | `sdcp-prod-sagemaker-...` |

## Required Repository Secrets

Configure these secrets in your GitHub repository settings under **Settings > Secrets and variables > Actions > Secrets**:

### OIDC Configuration (No secrets required)
The pipeline uses OpenID Connect (OIDC) for keyless authentication. No AWS access keys or secrets are needed.

## OIDC Setup Instructions

### 1. Create OIDC Identity Provider in AWS
```bash
# Run this in your AWS account
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 2. Create Trust Relationship for SageMaker Roles
Add this trust relationship to each environment's SageMaker role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:*"
        }
      }
    }
  ]
}
```

### 3. Required IAM Permissions
Each SageMaker role needs these additional permissions for CI/CD:
- `states:*` (Step Functions)
- `lambda:*` (Lambda Functions)
- `events:*` (EventBridge)
- `ecr:*` (Container Registry)
- `codebuild:*` (Container Builds)
