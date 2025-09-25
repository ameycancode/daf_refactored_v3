"""
Simplified Configuration Management for Energy Forecasting System
Container-ready version that doesn't require external config files
"""

import os
import json
import boto3
from datetime import datetime
import pytz
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment-specific configuration from JSON
def load_environment_config():
    """Load environment-specific configuration from JSON file"""
    config_path = Path(__file__).parent / "config.json"
   
    if config_path.exists():
        with open(config_path, 'r') as f:
            env_config = json.load(f)
            logger.info(f"Loaded environment configuration from: {config_path}")
            logger.info(f"Environment: {env_config.get('ENVIRONMENT', 'unknown')}")
            logger.debug(f"Configuration details: {json.dumps(env_config, indent=2)}")
            return env_config
    else:
        # Fallback for local development
        logger.warning("config.json not found, using fallback configuration for local development")
        return {
            "ENVIRONMENT": "dev",
            "AWS_REGION": "us-west-2",
            "DEBUG_MODE": True,
            "LOG_LEVEL": "DEBUG",
            "DATA_BUCKET": "sdcp-dev-sagemaker-energy-forecasting-data",
            "MODEL_BUCKET": "sdcp-dev-sagemaker-energy-forecasting-models",
            "TRAINING_CONFIG": {
                "xgboost": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.1
                }
            },
            "MODEL_REGISTRY_CONFIG": {
                "model_package_group_name": "energy-forecasting-models",
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
            "S3_PATHS": {
                "processed_data": "sdcp_modeling/forecasting/data/xgboost/processed/",
                "model_input": "sdcp_modeling/forecasting/data/xgboost/input/",
                "model_output": "sdcp_modeling/forecasting/models/",
                "temp_models": "sdcp_modeling/temp/training/"
            },
            "CUSTOMER_PROFILES": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
        }

# Load configuration from JSON
ENV_CONFIG = load_environment_config()

# Export configuration values
ENVIRONMENT = ENV_CONFIG["ENVIRONMENT"]
AWS_REGION = ENV_CONFIG["AWS_REGION"]
AWS_ACCOUNT_ID = ENV_CONFIG.get("AWS_ACCOUNT_ID", "")
DEBUG_MODE = ENV_CONFIG["DEBUG_MODE"]
LOG_LEVEL = ENV_CONFIG["LOG_LEVEL"]
DATA_BUCKET = ENV_CONFIG["DATA_BUCKET"]
MODEL_BUCKET = ENV_CONFIG["MODEL_BUCKET"]

# Training Configuration
TRAINING_CONFIG = ENV_CONFIG["TRAINING_CONFIG"]
MODEL_REGISTRY_CONFIG = ENV_CONFIG.get("MODEL_REGISTRY_CONFIG", {})

# S3 Paths and Customer Profiles from JSON
S3_PATHS = ENV_CONFIG.get("S3_PATHS", {})
CUSTOMER_PROFILES = ENV_CONFIG.get("CUSTOMER_PROFILES", ["RNN", "RN", "M", "S", "AGR", "L", "A6"])


class EnergyForecastingConfig:
    """Simplified configuration management for containers"""
   
    def __init__(self, config_file=None):
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = datetime.now(self.pacific_tz).strftime("%Y%m%d")
       
        # Initialize boto3 clients
        try:
            self.s3_client = boto3.client('s3')
            self.region = boto3.Session().region_name or AWS_REGION
            self.account_id = boto3.client('sts').get_caller_identity()['Account']
            logger.info(f"AWS connection successful. Region: {self.region}, Account: {self.account_id}")
        except Exception as e:
            logger.warning(f"AWS connection failed: {str(e)}")
            self.s3_client = None
            self.region = AWS_REGION
            self.account_id = AWS_ACCOUNT_ID or "123456789012"
       
        # Load configuration using JSON-based approach
        self.config = self._load_configuration()
       
        logger.info(f"Configuration initialized for date: {self.current_date}")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Data bucket: {DATA_BUCKET}")
        logger.info(f"Model bucket: {MODEL_BUCKET}")
   
    def _load_configuration(self):
        """Load configuration using JSON-based approach"""
        logger.info("Loading configuration from environment-specific JSON")
       
        return {
            # S3 Configuration (matches our original structure)
            "s3": {
                "data_bucket": DATA_BUCKET,
                "model_bucket": MODEL_BUCKET,
                "raw_data_prefix": "sdcp_modeling/forecasting/data/raw/",
                "processed_data_prefix": S3_PATHS.get("processed_data", "sdcp_modeling/forecasting/data/xgboost/processed/"),
                "input_data_prefix": S3_PATHS.get("model_input", "sdcp_modeling/forecasting/data/xgboost/input/"),
                "output_data_prefix": S3_PATHS.get("output_data", "sdcp_modeling/forecasting/data/xgboost/output/"),
                # "model_prefix": "xgboost/",
                "model_prefix": f"{S3_PATHS.get('model_output', 'sdcp_modeling/forecasting/models/')}xgboost/",
                "train_results_prefix": S3_PATHS.get("train_results", "sdcp_modeling/forecasting/data/xgboost/train_results/")
            },
           
            # Data Processing Configuration (from your original code)
            "data_processing": {
                "split_date": "2025-06-24",
                "profile_start_dates": {
                    "df_RNN": "2022-03-01",
                    "df_RN": "2022-03-01",
                    "df_M": "2021-07-01",
                    "df_S": "2021-07-01",
                    "df_AGR": "2023-05-01",
                    "df_L": "2021-07-01",
                    "df_A6": "2021-07-10"
                },
                "lag_features": {
                    "load_i_lag_days": 14,
                    "load_lag_days": 70
                },
                "profile_mappings": {
                    "RES_Non_NEM": "df_RNN",
                    "RES_NEM": "df_RN",
                    "MEDCI": "df_M",
                    "SMLCOM": "df_S",
                    "AGR": "df_AGR",
                    "LIGHT": "df_L",
                    "A6": "df_A6"
                },
                "holidays": [
                    "2021-01-01", "2021-02-15", "2021-05-31", "2021-07-05", "2021-09-06",
                    "2021-11-11", "2021-11-25", "2021-12-25",
                    "2022-01-01", "2022-02-21", "2022-05-30", "2022-07-04", "2022-09-05",
                    "2022-11-11", "2022-11-24", "2022-12-26",
                    "2023-01-02", "2023-02-20", "2023-05-29", "2023-07-04", "2023-09-04",
                    "2023-11-11", "2023-11-23", "2023-12-25",
                    "2024-01-01", "2024-02-19", "2024-05-27", "2024-07-04", "2024-09-02",
                    "2024-11-11", "2024-11-28", "2024-12-25",
                    "2025-01-01", "2025-02-17", "2025-05-26", "2025-07-04", "2025-09-01",
                    "2025-11-11", "2025-11-27", "2025-12-25"
                ]
            },
           
            # Training Configuration (from your original code)
            "training": {
                "train_cutoff": "2025-05-24",
                "cv_splits": 10,
                "xgboost_params": TRAINING_CONFIG.get("xgboost", {
                    "n_estimators": [150, 200, 300],
                    "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "max_depth": [4, 5, 6, 7]
                }),
                "performance_threshold": None,
                "random_state": 42
            },
           
            # API Configuration for Weather/Radiation
            "apis": {
                "weather": {
                    "base_url": "https://api.weather.gov",
                    "station": "KSAN",  # San Diego
                    "location": {
                        "latitude": 32.7157,
                        "longitude": -117.1611
                    }
                },
                "radiation": {
                    "base_url": "https://api.open-meteo.com/v1/forecast",
                    "location": {
                        "latitude": 32.7157,
                        "longitude": -117.1611
                    }
                }
            },
           
            # File naming patterns (from your original code)
            "file_patterns": {
                "raw_files": {
                    "load_data": "SQMD.csv",
                    "temperature_data": "Temperature.csv",
                    "radiation_data": "Radiation.csv"
                },
                "processed_files": {
                    "profile_lagged": "{profile}_lagged_{date}.csv",
                    "profile_train": "{profile}_train_{date}.csv",
                    "profile_test": "{profile}_test_{date}{suffix}.csv"
                },
                "model_files": {
                    "xgboost_model": "{profile}_best_xgboost_{date}.pkl"
                },
                "prediction_files": {
                    "profile_predictions": "{profile}_predictions_{date}.csv",
                    "combined_load": "Combined_Load_{date}.csv",
                    "aggregated_load": "Aggregated_Load_{date}.csv",
                    "weather_forecast": "T_{date}.csv",
                    "radiation_forecast": "shortwave_radiation_{date}.csv"
                }
            },
           
            # Container paths (SageMaker specific)
            "container_paths": {
                "input_path": "/opt/ml/processing/input",
                "output_path": "/opt/ml/processing/output",
                "model_path": "/opt/ml/processing/models",
                "code_path": "/opt/ml/processing/code",
                "config_path": "/opt/ml/processing/config"
            }
        }
   
    # S3 Path Generators
    def get_s3_path(self, path_type, **kwargs):
        """Generate S3 paths based on configuration"""
        bucket = self.config["s3"]["data_bucket"]
       
        if path_type == "raw_data":
            return f"s3://{bucket}/{self.config['s3']['raw_data_prefix']}"
        elif path_type == "processed_data":
            return f"s3://{bucket}/{self.config['s3']['processed_data_prefix']}"
        elif path_type == "input_data":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}"
        elif path_type == "output_data":
            return f"s3://{bucket}/{self.config['s3']['output_data_prefix']}"
        elif path_type == "train_results":
            return f"s3://{bucket}/{self.config['s3']['train_results_prefix']}"
        elif path_type == "models":
            bucket = self.config["s3"]["model_bucket"]
            return f"s3://{bucket}/{self.config['s3']['model_prefix']}"
        elif path_type == "temperature_input":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}temperature/"
        elif path_type == "radiation_input":
            return f"s3://{bucket}/{self.config['s3']['input_data_prefix']}radiation/"
        else:
            raise ValueError(f"Unknown path type: {path_type}")
   
    def get_file_path(self, file_type, **kwargs):
        """Generate file paths based on configuration patterns"""
        patterns = self.config["file_patterns"]
       
        # Use current date if not provided
        date = kwargs.get('date', self.current_date)
        profile = kwargs.get('profile', '')
        suffix = kwargs.get('suffix', '')
       
        if file_type in patterns["raw_files"]:
            return patterns["raw_files"][file_type]
        elif file_type in patterns["processed_files"]:
            return patterns["processed_files"][file_type].format(
                profile=profile, date=date, suffix=suffix
            )
        elif file_type in patterns["model_files"]:
            return patterns["model_files"][file_type].format(
                profile=profile, date=date
            )
        elif file_type in patterns["prediction_files"]:
            return patterns["prediction_files"][file_type].format(
                profile=profile, date=date
            )
        else:
            raise ValueError(f"Unknown file type: {file_type}")
   
    def get_full_s3_key(self, path_type, file_type, **kwargs):
        """Get complete S3 key combining path and filename"""
        s3_path = self.get_s3_path(path_type, **kwargs)
        filename = self.get_file_path(file_type, **kwargs)
       
        # Remove s3:// and bucket name to get just the key
        s3_key = s3_path.split('/', 3)[-1] + filename
        return s3_key
   
    # Configuration getters
    def get_profiles(self):
        """Get all profile codes"""
        return list(self.config["data_processing"]["profile_mappings"].values())
   
    def get_profile_start_date(self, profile):
        """Get start date for a specific profile"""
        return self.config["data_processing"]["profile_start_dates"].get(profile)
   
    def get_training_config(self):
        """Get training configuration"""
        return self.config["training"]
   
    def get_api_config(self, api_name):
        """Get API configuration"""
        return self.config["apis"].get(api_name, {})
   
    def get_container_paths(self):
        """Get container path configuration"""
        return self.config["container_paths"]
   
    def get_data_processing_config(self):
        """Get data processing configuration"""
        return self.config["data_processing"]
   
    # S3 bucket getters
    @property
    def data_bucket(self):
        return self.config["s3"]["data_bucket"]
   
    @property
    def model_bucket(self):
        return self.config["s3"]["model_bucket"]
   
    @property
    def current_date_str(self):
        return self.current_date
   
    # Save configuration
    def save_config(self, filepath):
        """Save current configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {filepath}: {str(e)}")
            return False

class S3FileManager:
    """S3 file manager using configuration"""
   
    def __init__(self, config: EnergyForecastingConfig):
        self.config = config
        self.s3_client = config.s3_client
   
    def upload_file(self, local_path, s3_key, bucket=None):
        """Upload file to S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        bucket = bucket or self.config.data_bucket
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
   
    def upload_dataframe(self, df, s3_key, bucket=None):
        """Upload DataFrame as CSV to S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        import pandas as pd
        from io import StringIO
       
        bucket = bucket or self.config.data_bucket
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Uploaded DataFrame to s3://{bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {str(e)}")
            return False
   
    def download_file(self, s3_key, local_path, bucket=None):
        """Download file from S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return False
           
        bucket = bucket or self.config.data_bucket
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download s3://{bucket}/{s3_key}: {str(e)}")
            return False
   
    def save_and_upload_dataframe(self, df, local_path, s3_key, bucket=None):
        """Save DataFrame locally and upload to S3"""
        bucket = bucket or self.config.data_bucket
       
        # Save locally first
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            df.to_csv(local_path, index=False)
            logger.info(f"Saved DataFrame locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame locally: {str(e)}")
            return False
       
        # Upload to S3
        return self.upload_file(local_path, s3_key, bucket)
   
    def save_and_upload_file(self, content, local_path, s3_key, bucket=None):
        """Save content to local file and upload to S3"""
        bucket = bucket or self.config.data_bucket
       
        # Save locally first
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                if isinstance(content, dict):
                    json.dump(content, f, indent=2)
                else:
                    f.write(content)
            logger.info(f"Saved file locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save file locally: {str(e)}")
            return False
       
        # Upload to S3
        return self.upload_file(local_path, s3_key, bucket)
