# =============================================================================
# CONTAINER CONFIGURATION MANAGER - deployment/container_config_manager.py
# =============================================================================
"""
Container Configuration Manager for Energy Forecasting MLOps
Manages environment-aware container configuration generation
Updated for sdcp_code/ folder structure
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import boto3
from datetime import datetime

class ContainerConfigManager:
    def __init__(self, environment: str = "dev", region: str = "us-west-2"):
        self.environment = environment
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.logger = self._setup_logging()
       
        # Load environment configuration
        self.config = self._load_environment_config()
       
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f'ContainerConfigManager-{self.environment}')
   
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        config_path = f"sdcp_code/config/environments/{self.environment}.yml"
       
        if not os.path.exists(config_path):
            self.logger.warning(f"Environment config not found: {config_path}, using defaults")
            return self._get_default_config()
       
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
           
        # Replace template variables
        config_str = yaml.dump(config)
        config_str = config_str.replace("{{ AWS_ACCOUNT_ID }}", self.account_id)
        return yaml.safe_load(config_str)
   
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if environment config is missing"""
        return {
            'environment': self.environment,
            'aws_account_id': self.account_id,
            'aws_region': self.region,
            's3': {
                'data_bucket': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                'model_bucket': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models'
            },
            'redshift': {
                'bi_schema': f'edp_bi_{"" if self.environment == "prod" else self.environment}',
                'cluster_identifier': f'sdcp-edp-backend-{"prod" if self.environment == "prod" else self.environment}',
                'database': 'sdcp',
                'db_user': 'ds_service_user',
                'input_schema': f'edp_cust_{"" if self.environment == "prod" else self.environment}',
                'input_table': 'caiso_sqmd',
                'operational_schema': f'edp_forecasting_{"" if self.environment == "prod" else self.environment}',
                'operational_table': 'dayahead_load_forecasts'
            },
            'containers': {
                'preprocessing': {
                    'environment': self.environment,
                    'debug_mode': self.environment == 'dev',
                    'log_level': 'DEBUG' if self.environment == 'dev' else 'INFO'
                },
                'training': {
                    'environment': self.environment,
                    'debug_mode': self.environment == 'dev',
                    'log_level': 'DEBUG' if self.environment == 'dev' else 'INFO'
                }
            }
        }
   
    def generate_container_configs(self) -> Dict[str, str]:
        """Generate environment-aware container configuration files"""
        self.logger.info(f"Generating container configs for environment: {self.environment}")
       
        configs_generated = {}
       
        # Generate preprocessing container config
        preprocessing_config = self._generate_preprocessing_json()
        preprocessing_path = "sdcp_code/containers/preprocessing/src/config.json"
        self._write_json_file(preprocessing_path, preprocessing_config)
        configs_generated['preprocessing'] = preprocessing_path
       
        # Generate training container config
        training_config = self._generate_training_json()
        training_path = "sdcp_code/containers/training/src/config.json"
        self._write_json_file(training_path, training_config)
        configs_generated['training'] = training_path
       
        # Generate buildspec.yml with environment variables (keep at root)
        buildspec_config = self._generate_buildspec_config()
        buildspec_path = "buildspec.yml"
        self._write_buildspec_file(buildspec_path, buildspec_config)
        configs_generated['buildspec'] = buildspec_path
       
        self.logger.info(f"Generated {len(configs_generated)} container configurations")
        return configs_generated
   
    def _generate_preprocessing_json(self) -> Dict[str, Any]:
        """Generate preprocessing container JSON configuration"""
        container_config = self.config['containers']['preprocessing']
        s3_config = self.config['s3']
        redshift_config = self.config.get('redshift', {})
        s3_prefix_model = s3_config.get('prefix', 'sdcp_modeling')

        config_json = {
            # Environment Configuration
            "ENVIRONMENT": self.environment,
            "AWS_REGION": self.region,
            "AWS_ACCOUNT_ID": self.account_id,
           
            # Logging Configuration
            "DEBUG_MODE": container_config.get('debug_mode', False),
            "LOG_LEVEL": container_config.get('log_level', 'INFO'),
           
            # S3 Configuration
            "DATA_BUCKET": s3_config['data_bucket'],
            "MODEL_BUCKET": s3_config['model_bucket'],
            "PREFIX": s3_prefix_model,

            # Redshift Configuration
            "REDSHIFT_BI_SCHEMA": redshift_config.get('bi_schema', 'edp_bi'),
            "REDSHIFT_CLUSTER_IDENTIFIER": redshift_config.get('cluster_identifier', 'sdcp-edp-backend'),
            "REDSHIFT_DATABASE": redshift_config.get('database', 'sdcp'),
            "REDSHIFT_DB_USER": redshift_config.get('db_user', 'ds_service_user'),
            "REDSHIFT_INPUT_SCHEMA": redshift_config.get('input_schema', 'edp_cust'),
            "REDSHIFT_INPUT_TABLE": redshift_config.get('input_table', 'caiso_sqmd'),
            "REDSHIFT_OPERATIONAL_SCHEMA": redshift_config.get('operational_schema', 'edp_forecasting'),
            "REDSHIFT_OPERATIONAL_TABLE": redshift_config.get('operational_table', 'dayahead_load_forecasts'),
            "DATA_READING_PERIOD_DAYS": redshift_config.get('data_reading_period_days', None),
           
            # Data Paths
            "S3_PATHS": {
                "raw_data": f"{s3_prefix_model}/forecasting/data/raw/",
                "processed_data": f"{s3_prefix_model}/forecasting/data/xgboost/processed/",
                "model_input": f"{s3_prefix_model}/forecasting/data/xgboost/input/",
                "model_output": f"{s3_prefix_model}/forecasting/models/",
                "output_data": f"{s3_prefix_model}/forecasting/data/xgboost/output/",
                "train_results": f"{s3_prefix_model}/forecasting/data/xgboost/train_results/",
                "temp_data": f"{s3_prefix_model}/temp/preprocessing/"
            },
           
            # Processing Configuration
            "PROCESSING_CONFIG": {
                "chunk_size": 50000, # if self.environment != "dev" else 50000,
                "parallel_processing": True,
                "validation_enabled": True,
                "debug_mode": container_config.get('debug_mode', False)
            },
           
            # Customer Profiles
            "CUSTOMER_PROFILES": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        }
       
        return config_json
   
    def _generate_training_json(self) -> Dict[str, Any]:
        """Generate training container JSON configuration"""
        container_config = self.config['containers']['training']
        s3_config = self.config['s3']
        s3_prefix_model = s3_config.get('prefix', 'sdcp_modeling')
       
        # Environment-specific training parameters
        n_estimators = 50 if self.environment == "dev" else (100 if self.environment == "preprod" else 200)
        early_stopping = 5 if self.environment == "dev" else (10 if self.environment == "preprod" else 20)
       
        config_json = {
            # Environment Configuration
            "ENVIRONMENT": self.environment,
            "AWS_REGION": self.region,
            "AWS_ACCOUNT_ID": self.account_id,
           
            # Logging Configuration
            "DEBUG_MODE": container_config.get('debug_mode', False),
            "LOG_LEVEL": container_config.get('log_level', 'INFO'),
           
            # S3 Configuration
            "DATA_BUCKET": s3_config['data_bucket'],
            "MODEL_BUCKET": s3_config['model_bucket'],
           
            # Model Training Configuration
            "TRAINING_CONFIG": {
                "xgboost": {
                    # "n_estimators": n_estimators,
                    "n_estimators": [150, 200, 300],
                    "max_depth": [4, 5, 6, 7],
                    "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    # "subsample": 0.8,
                    # "colsample_bytree": 0.8,
                    # "random_state": 42,
                    # "performance_threshold": None,
                },
                "validation_split": 0.2,
                "early_stopping_rounds": early_stopping,
                "eval_metric": "rmse"
            },
           
            # Model Registry Configuration
            "MODEL_REGISTRY_CONFIG": {
                "model_package_group_name": "energy-forecasting-models",
                "approval_status": "Approved",
                "performance_thresholds": {
                    "RNN": {"min_r2": 0.85, "max_mape": 5.0, "max_rmse": 0.1},
                    "RN": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12},
                    "M": {"min_r2": 0.85, "max_mape": 4.0, "max_rmse": 0.08},
                    "S": {"min_r2": 0.82, "max_mape": 5.5, "max_rmse": 0.10},
                    "AGR": {"min_r2": 0.80, "max_mape": 7.0, "max_rmse": 0.15},
                    "L": {"min_r2": 0.75, "max_mape": 8.0, "max_rmse": 0.20},
                    "A6": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12}
                }
            },
           
            # Data Paths
            "S3_PATHS": {
                "processed_data": f"{s3_prefix_model}/forecasting/data/xgboost/processed/",
                "model_input": f"{s3_prefix_model}/forecasting/data/xgboost/input/",
                "model_output": f"{s3_prefix_model}/forecasting/models/",
                "temp_models": f"{s3_prefix_model}/temp/training/"
            },
           
            # Customer Profiles
            "CUSTOMER_PROFILES": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        }
       
        return config_json

    def _write_json_file(self, file_path: str, content: Dict[str, Any]) -> None:
        """Write JSON configuration file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
       
        # Write JSON configuration file
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
       
        self.logger.info(f"Generated JSON configuration file: {file_path}")
       
    def _generate_buildspec_config(self) -> str:
        """Generate buildspec.yml with environment-aware configuration for sdcp_code structure"""
        buildspec_content = f'''version: 0.2

env:
  variables:
    ENVIRONMENT: {self.environment}
    AWS_DEFAULT_REGION: {self.region}
    AWS_ACCOUNT_ID: {self.account_id}
    IMAGE_TAG: latest
    ECR_PREFIX: energy
 
phases:
  pre_build:
    commands:
      - echo "=== PRE-BUILD PHASE ==="
      - echo "Environment: $ENVIRONMENT"
      - echo "Account ID: $AWS_ACCOUNT_ID"
      - echo "Region: $AWS_DEFAULT_REGION"
      - echo "Working Directory: $(pwd)"
      - echo "Directory Contents:"
      - ls -la
      - echo "SDCP Code Directory Contents:"
      - ls -la sdcp_code/ || echo "sdcp_code directory not found"
      - echo "Logging in to Amazon ECR..."
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
     
      # Create ECR repositories if they don't exist
      - |
        for repo in energy-preprocessing energy-training energy-prediction; do
          echo "Creating ECR repository: $repo"
          aws ecr create-repository --repository-name $repo --region $AWS_DEFAULT_REGION || echo "Repository $repo already exists"
        done
 
  build:
    commands:
      - echo "=== BUILD PHASE ==="
      - echo "Current working directory: $(pwd)"
      - echo "Available directories:"
      - ls -la
     
      # Generate container configurations
      - echo "Generating container configurations..."
      - python sdcp_code/deployment/container_config_manager.py --environment $ENVIRONMENT --generate-configs || echo "Config generation failed, continuing with existing configs"
     
      # Build preprocessing container
      - echo "Building preprocessing container..."
      - echo "Checking for preprocessing container directory..."
      - ls -la sdcp_code/containers/preprocessing/ || echo "Preprocessing container directory not found"
      - cd sdcp_code/containers/preprocessing
      - echo "Contents of preprocessing directory:"
      - ls -la
      - docker build -t $ECR_PREFIX-preprocessing:$IMAGE_TAG .
      - docker tag $ECR_PREFIX-preprocessing:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-preprocessing:$IMAGE_TAG
      - docker tag $ECR_PREFIX-preprocessing:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-preprocessing:$ENVIRONMENT-$IMAGE_TAG
      - cd ../../..
     
      # Build training container
      - echo "Building training container..."
      - echo "Checking for training container directory..."
      - ls -la sdcp_code/containers/training/ || echo "Training container directory not found"
      - cd sdcp_code/containers/training
      - echo "Contents of training directory:"
      - ls -la
      - docker build -t $ECR_PREFIX-training:$IMAGE_TAG .
      - docker tag $ECR_PREFIX-training:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-training:$IMAGE_TAG
      - docker tag $ECR_PREFIX-training:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-training:$ENVIRONMENT-$IMAGE_TAG
      - cd ../../..
     
      # Build prediction container (if exists)
      - |
        if [ -d "sdcp_code/containers/prediction" ]; then
          echo "Building prediction container..."
          cd sdcp_code/containers/prediction
          echo "Contents of prediction directory:"
          ls -la
          docker build -t $ECR_PREFIX-prediction:$IMAGE_TAG .
          docker tag $ECR_PREFIX-prediction:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-prediction:$IMAGE_TAG
          docker tag $ECR_PREFIX-prediction:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-prediction:$ENVIRONMENT-$IMAGE_TAG
          cd ../../..
        else
          echo "Prediction container directory not found, skipping..."
        fi
 
  post_build:
    commands:
      - echo "=== POST-BUILD PHASE ==="
     
      # Push preprocessing container
      - echo "Pushing preprocessing container..."
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-preprocessing:$IMAGE_TAG
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-preprocessing:$ENVIRONMENT-$IMAGE_TAG
     
      # Push training container
      - echo "Pushing training container..."
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-training:$IMAGE_TAG
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-training:$ENVIRONMENT-$IMAGE_TAG
     
      # Push prediction container (if exists)
      - |
        if docker images | grep -q $ECR_PREFIX-prediction; then
          echo "Pushing prediction container..."
          docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-prediction:$IMAGE_TAG
          docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$ECR_PREFIX-prediction:$ENVIRONMENT-$IMAGE_TAG
        else
          echo "Prediction container not built, skipping push..."
        fi
     
      # Create build summary
      - |
        cat > container-build-summary.json << EOF
        {{
          "build_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
          "environment": "$ENVIRONMENT",
          "account_id": "$AWS_ACCOUNT_ID",
          "region": "$AWS_DEFAULT_REGION",
          "image_tag": "$IMAGE_TAG",
          "folder_structure": "sdcp_code",
          "containers_built": [
            {{
              "name": "energy-preprocessing",
              "repository": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/energy-preprocessing",
              "tags": ["$IMAGE_TAG", "$ENVIRONMENT-$IMAGE_TAG"],
              "path": "sdcp_code/containers/preprocessing"
            }},
            {{
              "name": "energy-training",
              "repository": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/energy-training",
              "tags": ["$IMAGE_TAG", "$ENVIRONMENT-$IMAGE_TAG"],
              "path": "sdcp_code/containers/training"
            }}
          ]
        }}
        EOF
     
      - echo "=== BUILD COMPLETE ==="
      - echo "Container build summary created: container-build-summary.json"
      - echo "Final directory listing:"
      - ls -la

artifacts:
  files:
    - container-build-summary.json
'''
        return buildspec_content
   
    def _write_buildspec_file(self, file_path: str, content: str) -> None:
        """Write buildspec.yml file"""
        with open(file_path, 'w') as f:
            f.write(content)
       
        self.logger.info(f"Generated buildspec file: {file_path}")
   
    def backup_existing_configs(self) -> Dict[str, str]:
        """Backup existing configuration files"""
        backup_dir = f"backup/container-configs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
       
        backed_up = {}
       
        # Updated paths for sdcp_code structure
        config_files = [
            "sdcp_code/containers/preprocessing/src/config.json",
            "sdcp_code/containers/training/src/config.json",
            "buildspec.yml"
        ]
       
        for config_file in config_files:
            if os.path.exists(config_file):
                backup_path = os.path.join(backup_dir, os.path.basename(config_file))
                os.system(f"cp {config_file} {backup_path}")
                backed_up[config_file] = backup_path
                self.logger.info(f"Backed up {config_file} to {backup_path}")
       
        return backed_up
   
    def validate_container_configs(self) -> Dict[str, Any]:
        """Validate generated container configurations"""
        validation_results = {
            "preprocessing_config": False,
            "training_config": False,
            "buildspec_config": False,
            "errors": []
        }
       
        # Validate preprocessing config (JSON now, not Python)
        try:
            preprocessing_path = "sdcp_code/containers/preprocessing/src/config.json"
            if os.path.exists(preprocessing_path):
                # JSON syntax check
                with open(preprocessing_path, 'r') as f:
                    json.load(f)
                validation_results["preprocessing_config"] = True
                self.logger.info("Preprocessing JSON config validation: PASSED")
            else:
                validation_results["errors"].append("Preprocessing config JSON file not found")
        except Exception as e:
            validation_results["errors"].append(f"Preprocessing config validation error: {str(e)}")
       
        # Validate training config (JSON now, not Python)
        try:
            training_path = "sdcp_code/containers/training/src/config.json"
            if os.path.exists(training_path):
                # JSON syntax check
                with open(training_path, 'r') as f:
                    json.load(f)
                validation_results["training_config"] = True
                self.logger.info("Training JSON config validation: PASSED")
            else:
                validation_results["errors"].append("Training config JSON file not found")
        except Exception as e:
            validation_results["errors"].append(f"Training config validation error: {str(e)}")
       
        # Validate buildspec
        try:
            buildspec_path = "buildspec.yml"
            if os.path.exists(buildspec_path):
                with open(buildspec_path, 'r') as f:
                    yaml.safe_load(f)
                validation_results["buildspec_config"] = True
                self.logger.info("Buildspec validation: PASSED")
            else:
                validation_results["errors"].append("Buildspec file not found")
        except Exception as e:
            validation_results["errors"].append(f"Buildspec validation error: {str(e)}")
       
        return validation_results

def main():
    """Main function for container configuration management"""
    parser = argparse.ArgumentParser(description='Container Configuration Manager for SDCP Code Structure')
    parser.add_argument('--environment', default='dev',
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--generate-configs', action='store_true',
                       help='Generate container configuration files')
    parser.add_argument('--backup-existing', action='store_true',
                       help='Backup existing configuration files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate generated configurations')
   
    args = parser.parse_args()
   
    # Initialize manager
    manager = ContainerConfigManager(
        environment=args.environment,
        region=args.region
    )
   
    try:
        print(f"Container Configuration Manager for Environment: {args.environment}")
        print(f"Working with sdcp_code/ folder structure")
       
        # Backup existing configs if requested
        if args.backup_existing:
            backed_up = manager.backup_existing_configs()
            print(f"Backed up {len(backed_up)} configuration files")
       
        # Generate configs if requested
        if args.generate_configs:
            configs_generated = manager.generate_container_configs()
            print(f"Generated {len(configs_generated)} configuration files:")
            for name, path in configs_generated.items():
                print(f"  {name}: {path}")
       
        # Validate configs if requested
        if args.validate:
            validation_results = manager.validate_container_configs()
            print("Configuration validation results:")
            for config_name, result in validation_results.items():
                if config_name != "errors":
                    status = "PASSED" if result else "FAILED"
                    print(f"  {config_name}: {status}")
           
            if validation_results["errors"]:
                print("Validation errors:")
                for error in validation_results["errors"]:
                    print(f"  - {error}")
                return 1
       
        print("Container configuration management completed successfully!")
        return 0
       
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
