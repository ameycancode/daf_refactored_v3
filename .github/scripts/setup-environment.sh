# Environment Setup Script - .github/scripts/setup-environment.sh
#!/bin/bash

set -euo pipefail

echo "🚀 Energy Forecasting MLOps - Environment Setup"
echo "================================================="

# Configuration
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-west-2}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo "Account: $ACCOUNT_ID"
echo ""

# Validate environment
case "$ENVIRONMENT" in
  dev|preprod|prod)
    echo "✅ Valid environment: $ENVIRONMENT"
    ;;
  *)
    echo "❌ Invalid environment. Must be: dev, preprod, or prod"
    exit 1
    ;;
esac

# Set environment-specific variables
ROLE_NAME="sdcp-$ENVIRONMENT-sagemaker-energy-forecasting-datascientist-role"
DATA_BUCKET="sdcp-$ENVIRONMENT-sagemaker-energy-forecasting-data"
MODEL_BUCKET="sdcp-$ENVIRONMENT-sagemaker-energy-forecasting-models"

echo "🔍 Checking Prerequisites..."
echo "=============================="

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is required but not installed"
    exit 1
fi
echo "✅ AWS CLI available"

# Check credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured"
    exit 1
fi
echo "✅ AWS credentials configured"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi
echo "✅ Python 3 available"

echo ""
echo "🏗️ Setting up Environment Resources..."
echo "======================================"

# Create S3 buckets
echo "📦 Creating S3 buckets..."
for BUCKET in "$DATA_BUCKET" "$MODEL_BUCKET"; do
    if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
        echo "✅ S3 bucket already exists: $BUCKET"
    else
        echo "🔨 Creating S3 bucket: $BUCKET"
        if [ "$AWS_REGION" = "us-east-1" ]; then
            aws s3api create-bucket --bucket "$BUCKET"
        else
            aws s3api create-bucket \
                --bucket "$BUCKET" \
                --create-bucket-configuration LocationConstraint="$AWS_REGION"
        fi
        echo "✅ S3 bucket created: $BUCKET"
    fi
done

# Check IAM role
echo "🔐 Checking IAM role..."
if aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
    echo "✅ IAM role exists: $ROLE_NAME"
else
    echo "⚠️ IAM role does not exist: $ROLE_NAME"
    echo "   Please create the SageMaker execution role manually or via CloudFormation"
fi

# Create ECR repositories
echo "🐳 Creating ECR repositories..."
for REPO in "energy-preprocessing" "energy-training" "energy-prediction"; do
    if aws ecr describe-repositories --repository-names "$REPO" &> /dev/null; then
        echo "✅ ECR repository already exists: $REPO"
    else
        echo "🔨 Creating ECR repository: $REPO"
        aws ecr create-repository --repository-name "$REPO"
        echo "✅ ECR repository created: $REPO"
    fi
done

# Setup CodeBuild project
echo "🏗️ Setting up CodeBuild project..."
CODEBUILD_PROJECT="energy-forecasting-container-builds"
if aws codebuild batch-get-projects --names "$CODEBUILD_PROJECT" --query 'projects[0].name' --output text 2>/dev/null | grep -q "$CODEBUILD_PROJECT"; then
    echo "✅ CodeBuild project already exists: $CODEBUILD_PROJECT"
else
    echo "🔨 Creating CodeBuild project: $CODEBUILD_PROJECT"
    python3 sdcp_code/scripts/build_via_codebuild.py --create-only --region "$AWS_REGION"
    echo "✅ CodeBuild project created: $CODEBUILD_PROJECT"
fi

echo ""
echo "✅ Environment Setup Complete!"
echo "=============================="
echo "Environment: $ENVIRONMENT"
echo "Data Bucket: $DATA_BUCKET"
echo "Model Bucket: $MODEL_BUCKET"
echo "SageMaker Role: $ROLE_NAME"
echo ""
echo "🎯 Next Steps:"
echo "1. Configure GitHub repository variables with the above values"
echo "2. Set up OIDC trust relationship for the SageMaker role"
echo "3. Run the CI/CD pipeline to deploy the MLOps infrastructure"
echo ""
echo "📚 For detailed instructions, see:"
echo "   .github/templates/repository-variables.md"