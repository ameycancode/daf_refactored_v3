"""
Simplified Configuration Management for Energy Forecasting System
Container-ready version with Redshift integration
"""

import os
import gc
import psutil
import json
import boto3
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import logging
import numpy as np
import pandas as pd

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
            "REDSHIFT_BI_SCHEMA": "edp_bi_dev",
            "REDSHIFT_CLUSTER_IDENTIFIER": "sdcp-edp-backend-dev",
            "REDSHIFT_DATABASE": "sdcp",
            "REDSHIFT_DB_USER": "ds_service_user",
            "REDSHIFT_INPUT_SCHEMA": "edp_cust_dev",
            "REDSHIFT_INPUT_TABLE": "caiso_sqmd",
            "REDSHIFT_OPERATIONAL_SCHEMA": "edp_forecasting_dev",
            "REDSHIFT_OPERATIONAL_TABLE": "dayahead_load_forecasts",
            "DATA_READING_PERIOD_DAYS": 365,
            "S3_PATHS": {
                "raw_data": "sdcp_modeling/forecasting/data/raw/",
                "processed_data": "sdcp_modeling/forecasting/data/xgboost/processed/",
                "model_input": "sdcp_modeling/forecasting/data/xgboost/input/",
                "output_data": "sdcp_modeling/forecasting/data/xgboost/output/",
                "train_results": "sdcp_modeling/forecasting/data/xgboost/train_results/",
                "temp_data": "sdcp_modeling/temp/preprocessing/"
            },
            "PROCESSING_CONFIG": {
                "chunk_size": 50000,
                "parallel_processing": True,
                "validation_enabled": True,
                "debug_mode": True
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

# Redshift Configuration
REDSHIFT_BI_SCHEMA = ENV_CONFIG["REDSHIFT_BI_SCHEMA"]
REDSHIFT_CLUSTER_IDENTIFIER = ENV_CONFIG["REDSHIFT_CLUSTER_IDENTIFIER"]
REDSHIFT_DATABASE = ENV_CONFIG["REDSHIFT_DATABASE"]
REDSHIFT_DB_USER = ENV_CONFIG["REDSHIFT_DB_USER"]
REDSHIFT_INPUT_SCHEMA = ENV_CONFIG["REDSHIFT_INPUT_SCHEMA"]
REDSHIFT_INPUT_TABLE = ENV_CONFIG["REDSHIFT_INPUT_TABLE"]
REDSHIFT_OPERATIONAL_SCHEMA = ENV_CONFIG["REDSHIFT_OPERATIONAL_SCHEMA"]
REDSHIFT_OPERATIONAL_TABLE = ENV_CONFIG["REDSHIFT_OPERATIONAL_TABLE"]
DATA_READING_PERIOD_DAYS = ENV_CONFIG["DATA_READING_PERIOD_DAYS"]

# S3 Paths and Processing Config from JSON
S3_PATHS = ENV_CONFIG.get("S3_PATHS", {})
PROCESSING_CONFIG = ENV_CONFIG.get("PROCESSING_CONFIG", {})
CUSTOMER_PROFILES = ENV_CONFIG.get("CUSTOMER_PROFILES", ["RNN", "RN", "M", "S", "AGR", "L", "A6"])

class EnergyForecastingConfig:
    """Simplified configuration management for containers with Redshift support"""
   
    def __init__(self, config_file=None):
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = datetime.now(self.pacific_tz).strftime("%Y%m%d")
       
        # Initialize boto3 clients
        try:
            self.s3_client = boto3.client('s3')
            self.redshift_data_client = boto3.client('redshift-data', region_name=AWS_REGION)
            self.region = boto3.Session().region_name or AWS_REGION
            self.account_id = boto3.client('sts').get_caller_identity()['Account']
            logger.info(f"AWS connection successful. Region: {self.region}, Account: {self.account_id}")
        except Exception as e:
            logger.warning(f"AWS connection failed: {str(e)}")
            self.s3_client = None
            self.redshift_data_client = None
            self.region = AWS_REGION
            self.account_id = AWS_ACCOUNT_ID or "123456789012"
       
        # Load configuration using JSON-based approach
        self.config = self._load_configuration()
       
        logger.info(f"Configuration initialized for date: {self.current_date}")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Data bucket: {DATA_BUCKET}")
        logger.info(f"Redshift cluster: {REDSHIFT_CLUSTER_IDENTIFIER}")
   
    def _load_configuration(self):
        """Load configuration using JSON-based approach"""
        logger.info("Loading configuration from environment-specific JSON")
        
        return {
            # S3 Configuration
            "s3": {
                "data_bucket": DATA_BUCKET,
                "model_bucket": MODEL_BUCKET,
                "raw_data_prefix": S3_PATHS.get("raw_data", "sdcp_modeling/forecasting/data/raw/"),
                "processed_data_prefix": S3_PATHS.get("processed_data", "sdcp_modeling/forecasting/data/xgboost/processed/"),
                "input_data_prefix": S3_PATHS.get("model_input", "sdcp_modeling/forecasting/data/xgboost/input/"),
                "output_data_prefix": S3_PATHS.get("output_data", "sdcp_modeling/forecasting/data/xgboost/output/"),
                "model_prefix": "xgboost/",
                "train_results_prefix": S3_PATHS.get("train_results", "sdcp_modeling/forecasting/data/xgboost/train_results/"),
            },
            
            # Redshift Configuration
            "redshift": {
                "database": REDSHIFT_DATABASE,
                "cluster_identifier": REDSHIFT_CLUSTER_IDENTIFIER,
                "db_user": REDSHIFT_DB_USER,
                "region": AWS_REGION,
                "schema": REDSHIFT_INPUT_SCHEMA,
                "table": REDSHIFT_INPUT_TABLE,
                "bi_schema": REDSHIFT_BI_SCHEMA,
                "operational_schema": REDSHIFT_OPERATIONAL_SCHEMA,
                "operational_table": REDSHIFT_OPERATIONAL_TABLE,
                "query_timeout_seconds": 1800,
                "use_redshift": True,
                "data_reading_period_days": DATA_READING_PERIOD_DAYS
            },
           
            # Data Processing Configuration
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
           
            # Training Configuration
            "training": {
                "train_cutoff": "2025-05-24",
                "cv_splits": 10,
                "xgboost_params": {
                    "n_estimators": [150, 200, 300],
                    "learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "max_depth": [4, 5, 6, 7]
                },
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
           
            # File naming patterns
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
           
            # Container paths
            "container_paths": {
                "input_path": "/opt/ml/processing/input",
                "output_path": "/opt/ml/processing/output",
                "model_path": "/opt/ml/processing/models",
                "code_path": "/opt/ml/processing/code",
                "config_path": "/opt/ml/processing/config"
            }
        }
   
    # Redshift Configuration getters
    def get_redshift_config(self):
        """Get Redshift configuration"""
        return self.config["redshift"]
    
    def is_redshift_enabled(self):
        """Check if Redshift is enabled"""
        return self.config["redshift"].get("use_redshift", True)
    
    def get_data_reading_period_days(self):
        """Get data reading period in days"""
        return self.config["redshift"].get("data_reading_period_days", None)
   
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


class RedshiftDataManager:
    """Redshift data operations using Data API"""
    
    def __init__(self, config: EnergyForecastingConfig):
        self.config = config
        self.redshift_config = config.get_redshift_config()
        self.redshift_client = config.redshift_data_client
        
    def execute_query(self, query, query_limit=None):
        """Execute Redshift query using Data API with pagination support"""
        try:
            if not self.redshift_client:
                raise Exception("Redshift client not available")
                
            if query_limit and query_limit > 0:
                query += f" LIMIT {query_limit}"
                
            logger.info(f"Executing query via Data API on cluster: {self.redshift_config['cluster_identifier']}")
            logger.info(f"Database: {self.redshift_config['database']}, User: {self.redshift_config['db_user']}")
            logger.info(f"Query: {query}")
            
            # Execute the query
            response = self.redshift_client.execute_statement(
                ClusterIdentifier=self.redshift_config['cluster_identifier'],
                Database=self.redshift_config['database'],
                DbUser=self.redshift_config['db_user'],
                Sql=query
            )
            
            query_id = response['Id']
            logger.info(f"Query submitted with ID: {query_id}")
            
            # Wait for completion
            self._wait_for_completion(query_id)
            
            # Get all results with pagination
            df = self._get_paginated_results(query_id)
            
            logger.info(f"Query completed successfully. Retrieved {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query via Data API: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _wait_for_completion(self, query_id):
        """Wait for query completion"""
        max_wait = self.redshift_config.get('query_timeout_seconds', 1800)
        waited = 0
        
        logger.info(f"Waiting for query {query_id} to complete...")
        
        while waited < max_wait:
            try:
                status_response = self.redshift_client.describe_statement(Id=query_id)
                status = status_response['Status']
                
                if status == 'FINISHED':
                    logger.info(f"Query {query_id} completed successfully")
                    return
                elif status == 'FAILED':
                    error_msg = status_response.get('Error', 'Unknown error')
                    logger.error(f"Query {query_id} failed: {error_msg}")
                    raise Exception(f'Query failed: {error_msg}')
                elif status == 'ABORTED':
                    logger.error(f"Query {query_id} was aborted")
                    raise Exception(f'Query was aborted')
                
                # Still running
                if waited % 60 == 0 and waited > 0:  # Log every minute
                    logger.info(f"Query still running... waited {waited}s (status: {status})")
                
                time.sleep(10)
                waited += 10
                
            except Exception as e:
                if 'failed:' in str(e) or 'aborted' in str(e):
                    raise
                else:
                    logger.warning(f"Error checking query status: {str(e)}")
                    time.sleep(10)
                    waited += 10
                    continue
        
        raise Exception(f'Query timed out after {max_wait} seconds')
    
    def _get_paginated_results(self, query_id):
        """Get all results with proper pagination"""
        import pandas as pd
        
        all_records = []
        column_metadata = None
        next_token = None
        page_count = 0
        
        try:
            while True:
                page_count += 1
                logger.info(f"Fetching results page {page_count}...")
                
                # Prepare request parameters
                request_params = {'Id': query_id}
                if next_token:
                    request_params['NextToken'] = next_token
                
                # Get results page
                result_response = self.redshift_client.get_statement_result(**request_params)
                
                # Get column metadata from first page only
                if column_metadata is None:
                    column_metadata = result_response.get('ColumnMetadata', [])
                    logger.info(f"Query has {len(column_metadata)} columns")
                
                # Get records from this page
                page_records = result_response.get('Records', [])
                all_records.extend(page_records)
                
                logger.info(f"Page {page_count}: Retrieved {len(page_records)} records (Total: {len(all_records)})")
                
                # Check if there are more pages
                next_token = result_response.get('NextToken')
                if not next_token:
                    logger.info(f"Pagination complete. Total pages: {page_count}, Total records: {len(all_records)}")
                    break
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(column_metadata, all_records)
            return df
            
        except Exception as e:
            logger.error(f"Error in paginated result retrieval: {str(e)}")
            raise
    
    def _convert_to_dataframe(self, column_metadata, all_records):
        """Convert Redshift Data API results to DataFrame"""
        import pandas as pd
        
        try:
            # Get column names
            column_names = [col['name'] for col in column_metadata]
            
            logger.info(f"Converting {len(all_records)} records with {len(column_names)} columns")
            
            if not all_records:
                return pd.DataFrame(columns=column_names)
            
            # Convert records to list of lists
            data_rows = []
            for record in all_records:
                row = []
                for field in record:
                    # Extract value based on type
                    if 'stringValue' in field:
                        row.append(field['stringValue'])
                    elif 'longValue' in field:
                        row.append(field['longValue'])
                    elif 'doubleValue' in field:
                        row.append(field['doubleValue'])
                    elif 'booleanValue' in field:
                        row.append(field['booleanValue'])
                    elif 'isNull' in field and field['isNull']:
                        row.append(None)
                    else:
                        row.append(str(field))  # Fallback
                data_rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=column_names)
            
            logger.info(f"DataFrame created: {len(df)} rows, {len(column_names)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error converting results to DataFrame: {str(e)}")
            raise
    
    def query_sqmd_data(self, current_date=None, query_limit=None):
        """Query SQMD data from Redshift with time filtering"""
        try:
            if current_date is None:
                current_date = datetime.now()
            
            schema_name = self.redshift_config['schema']
            table_name = self.redshift_config['table']
            
            # Calculate time range based on data reading period
            data_period_days = self.config.get_data_reading_period_days()
            
            if data_period_days:
                start_date = current_date - timedelta(days=data_period_days)
                logger.info(f"Filtering data from {start_date.strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')}")
                
                query = f"""
                SELECT
                    tradedatelocal as tradedate,
                    tradehourstartlocal as tradetime,
                    loadprofile, rategroup, baseload, lossadjustedload, metercount,
                    loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                    submission, createddate as created
                FROM {schema_name}.{table_name}
                WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'
                ORDER BY tradedatelocal, tradehourstartlocal
                """
            else:
                logger.info("No time limit set - fetching all available data")
                query = f"""
                SELECT
                    tradedatelocal as tradedate,
                    tradehourstartlocal as tradetime,
                    loadprofile, rategroup, baseload, lossadjustedload, metercount,
                    loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
                    submission, createddate as created
                FROM {schema_name}.{table_name}
                ORDER BY tradedatelocal, tradehourstartlocal
                """
            
            logger.info(f"Executing SQMD data query")
            df = self.execute_query(query, query_limit)
            
            logger.info(f"Retrieved {len(df)} rows of SQMD data from Redshift")
            return df
            
        except Exception as e:
            logger.error(f"Error querying SQMD data: {e}")
            logger.error(traceback.format_exc())
            raise

class MemoryOptimizedRedshiftDataManager(RedshiftDataManager):
    """Memory-optimized version for large datasets"""
    
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size = PROCESSING_CONFIG.get('chunk_size', 50000)
        self.memory_threshold = 0.8
        
    def query_sqmd_data_chunked(self, current_date=None, chunk_size=None):
        """Query SQMD data in chunks with per-chunk processing"""
        try:
            if chunk_size is None:
                chunk_size = self.chunk_size
                
            logger.info(f"Starting optimized chunked query with per-chunk processing")
            logger.info(f"Chunk size: {chunk_size:,}")
            
            # Get total row count first
            count_query = self._build_count_query(current_date)
            total_rows = self._get_total_row_count(count_query)
            
            if total_rows == 0:
                raise ValueError("No SQMD data available for the specified period")
                
            logger.info(f"Total rows to process: {total_rows:,}")
            
            # Calculate number of chunks needed
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            logger.info(f"Will process data in {num_chunks} chunks with per-chunk optimization")
            
            # Process data in chunks with per-chunk processing
            processed_chunks = []
            for chunk_num in range(num_chunks):
                offset = chunk_num * chunk_size
                
                logger.info(f"Processing chunk {chunk_num + 1}/{num_chunks} (rows {offset:,} to {min(offset + chunk_size, total_rows):,})")
                
                # Check memory before processing each chunk
                self._check_memory_usage()
                
                # Get raw chunk
                chunk_query = self._build_chunk_query(current_date, chunk_size, offset)
                raw_chunk = self.execute_query(chunk_query)
                
                if raw_chunk.empty:
                    logger.warning(f"Chunk {chunk_num + 1} returned no data")
                    continue
                
                # Per-chunk processing (stages 1-5)
                processed_chunk = self._process_chunk_optimized(raw_chunk, chunk_num + 1)
                
                if not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)
                    logger.info(f"Chunk {chunk_num + 1} processed: {len(processed_chunk):,} rows")
                
                # Clear raw chunk from memory
                del raw_chunk
                gc.collect()
            
            if not processed_chunks:
                raise ValueError("No data retrieved from any chunks")
            
            # Combine processed chunks
            logger.info("Combining all processed chunks...")
            df_combined = pd.concat(processed_chunks, ignore_index=True)
            
            # Clear chunk data from memory
            del processed_chunks
            gc.collect()
            
            logger.info(f"Successfully combined all chunks: {len(df_combined):,} total rows")
            
            # Full-dataset processing (stages 6-8)
            logger.info("Starting full-dataset processing (aggregation and final steps)...")
            df_final = self._process_full_dataset(df_combined)
            
            # Clear combined data from memory
            del df_combined
            gc.collect()
            
            logger.info(f"Final processing completed: {len(df_final):,} records")
            return df_final
            
        except Exception as e:
            logger.error(f"Error in optimized chunked query: {str(e)}")
            # Clean up memory on error
            for var_name in ['raw_chunk', 'processed_chunks', 'df_combined']:
                if var_name in locals():
                    del locals()[var_name]
            gc.collect()
            raise

    def _process_chunk_optimized(self, chunk_df, chunk_num):
        """Process individual chunk with stages 1-5"""
        try:
            logger.info(f"  Chunk {chunk_num}: Starting per-chunk processing...")
            
            # Stage 1: Process datetime columns
            logger.info("Stage 1: Process datetime columns")
            chunk_df = self._process_datetime_columns_chunk(chunk_df)
            
            # Stage 2: Create profile classifications  
            logger.info("Stage 2: Create profile classifications")
            chunk_df = self._create_profile_classifications_chunk(chunk_df)
            
            # Stage 3: Clean numeric data
            logger.info("Stage 3: Clean numeric data")
            chunk_df = self._clean_numeric_data_chunk(chunk_df)
            
            # Stage 4: Select required columns
            logger.info("Stage 4: Select required columns")
            chunk_df = self._select_required_columns_chunk(chunk_df)
            
            # Stage 5: Separate submissions (can be done per chunk)
            logger.info("Stage 5: Separate submissions (can be done per chunk)")
            df_final_chunk, df_initial_chunk = self._separate_submissions_chunk(chunk_df)
            
            # Add submission type back to chunks for later processing
            df_final_chunk['submission_type'] = 'Final'
            df_initial_chunk['submission_type'] = 'Initial'
            
            # Combine final and initial for this chunk
            processed_chunk = pd.concat([df_final_chunk, df_initial_chunk], ignore_index=True)
            
            logger.info(f"  Chunk {chunk_num}: Per-chunk processing completed")
            return processed_chunk
            
        except Exception as e:
            logger.error(f"  Chunk {chunk_num}: Per-chunk processing failed: {str(e)}")
            raise
    
    # Per-chunk processing methods (stages 1-5)
    
    def _process_datetime_columns_chunk(self, df):
        """Optimized datetime processing for chunk"""
        try:
            # Check if we have the expected format in first few rows
            sample_tradedate = str(df['tradedate'].iloc[0]) if not df.empty else ''
            sample_tradetime = str(df['tradetime'].iloc[0]) if not df.empty else ''
            
            # Most efficient approach for your format (tradedate: "2021-03-03", tradetime: "00")
            if len(sample_tradetime) <= 2:  # Your case: tradetime is just hour
                # Vectorized operation - much faster than apply
                df['tradetime_formatted'] = df['tradetime'].astype(str).str.zfill(2) + ':00:00'
                df['TradeDateTime'] = pd.to_datetime(
                    df['tradedate'].astype(str) + ' ' + df['tradetime_formatted'], 
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
                df.drop('tradetime_formatted', axis=1, inplace=True)
            else:
                # Standard format
                df['TradeDateTime'] = pd.to_datetime(
                    df['tradedate'].astype(str) + ' ' + df['tradetime'].astype(str), 
                    format='%Y-%m-%d %H:%M:%S',
                    errors='coerce'
                )
            
            # Handle any NaT values
            nat_count = df['TradeDateTime'].isna().sum()
            if nat_count > 0:
                logger.warning(f"    Found {nat_count} invalid datetime values in chunk, removing...")
                df = df.dropna(subset=['TradeDateTime'])
            
            return df
            
        except Exception as e:
            logger.error(f"    Chunk datetime processing failed: {str(e)}")
            raise
    
    def _create_profile_classifications_chunk(self, df):
        """Create profile and NEM classifications for chunk"""
        df['RateGroup'] = df['rategroup'].astype(str)
        
        # Vectorized operations
        df['NEM'] = df['RateGroup'].str.startswith(('NEM', 'SBP')).map({True: 'NEM', False: 'Non_NEM'})
        df['Profile'] = df.apply(
            lambda row: row['loadprofile'] + '_' + row['NEM'] if row['loadprofile'] == 'RES' else row['loadprofile'], 
            axis=1
        )
        
        return df
    
    def _clean_numeric_data_chunk(self, df):
        """Clean numeric data for chunk"""
        # Convert to numeric with error handling
        df['lossadjustedload'] = pd.to_numeric(df['lossadjustedload'], errors='coerce')
        df['metercount'] = pd.to_numeric(df['metercount'], errors='coerce')
        
        # Remove rows with invalid numeric values
        invalid_count = df[['lossadjustedload', 'metercount']].isna().any(axis=1).sum()
        if invalid_count > 0:
            logger.warning(f"    Found {invalid_count} rows with invalid numeric values in chunk, removing...")
            df = df.dropna(subset=['lossadjustedload', 'metercount'])
        
        return df
    
    def _select_required_columns_chunk(self, df):
        """Select only required columns to reduce memory usage"""
        required_columns = ['TradeDateTime', 'tradedate', 'tradetime', 'Profile', 'lossadjustedload', 'metercount', 'submission']
        df = df[required_columns].copy()
        
        # Rename columns
        df.columns = ['TradeDateTime', 'TradeDate', 'TradeTime', 'Profile', 'LossAdjustedLoad', 'MeterCount', 'Submission']
        return df
    
    def _separate_submissions_chunk(self, df):
        """Separate Final and Initial submissions for chunk"""
        df_final = df[df['Submission'] == 'Final'].copy()
        df_initial = df[df['Submission'] == 'Initial'].copy()
        
        return df_final, df_initial
    
    # Full-dataset processing methods (stages 6-8)  
    
    def _process_full_dataset(self, df_combined):
        """Process full combined dataset (stages 6-8)"""
        try:
            logger.info("Starting full-dataset aggregation and processing...")
            
            # Separate final and initial submissions from combined data
            df_final = df_combined[df_combined['submission_type'] == 'Final'].copy()
            df_initial = df_combined[df_combined['submission_type'] == 'Initial'].copy()
            
            # Remove submission_type column
            df_final.drop('submission_type', axis=1, inplace=True)
            df_initial.drop('submission_type', axis=1, inplace=True)
            
            # Clear combined data from memory
            del df_combined
            gc.collect()
            
            # Stage 6: Aggregate hourly data
            logger.info("Stage 6: Aggregating hourly data...")
            df_hour_final = self._aggregate_hourly_data_optimized(df_final, 'final')
            df_hour_initial = self._aggregate_hourly_data_optimized(df_initial, 'initial')
            
            # Clear submission dataframes from memory
            del df_final, df_initial
            gc.collect()
            
            # Stage 7: Calculate metrics and merge
            logger.info("Stage 7: Calculating metrics and merging...")
            df_processed = self._calculate_metrics_and_merge_optimized(df_hour_final, df_hour_initial)
            
            # Clear hourly dataframes from memory  
            del df_hour_final, df_hour_initial
            gc.collect()
            
            # Stage 8: Extend dataset and add features
            logger.info("Stage 8: Extending dataset and adding features...")
            df_extended = self._extend_dataset_optimized(df_processed)
            del df_processed
            gc.collect()
            
            df_final = self._add_date_features_optimized(df_extended)
            del df_extended
            gc.collect()
            
            logger.info("Full-dataset processing completed successfully")
            return df_final
            
        except Exception as e:
            logger.error(f"Full-dataset processing failed: {str(e)}")
            raise
    
    def _aggregate_hourly_data_optimized(self, df, submission_type):
        """Memory-optimized hourly aggregation"""
        if df.empty:
            logger.warning(f"No data to aggregate for {submission_type} submissions")
            return pd.DataFrame(columns=['TradeDateTime', 'Profile', 'LoadHour', 'Count'])
        
        # Group by hourly with memory-optimized aggregation
        df_hour = df.groupby(['TradeDateTime', 'Profile']).agg(
            LoadHour=('LossAdjustedLoad', 'sum'),
            Count=('MeterCount', 'sum')
        ).reset_index()
        
        logger.info(f"Aggregated {submission_type} data: {len(df_hour)} hourly records")
        return df_hour
    
    def _calculate_metrics_and_merge_optimized(self, df_hour_final, df_hour_initial):
        """Memory-optimized metrics calculation and merging"""
        # Calculate Load_Per_Meter for both datasets
        df_hour_final['Load_Per_Meter'] = df_hour_final['LoadHour'] / df_hour_final['Count']
        df_hour_initial['Load_Per_Meter'] = df_hour_initial['LoadHour'] / df_hour_initial['Count']
        
        # Replace inf values with NaN
        df_hour_final['Load_Per_Meter'] = df_hour_final['Load_Per_Meter'].replace([np.inf, -np.inf], np.nan)
        df_hour_initial['Load_Per_Meter'] = df_hour_initial['Load_Per_Meter'].replace([np.inf, -np.inf], np.nan)
        
        # Rename initial columns
        df_hour_initial = df_hour_initial.rename(columns={
            'LoadHour': 'LoadHour_I',
            'Count': 'Count_I',
            'Load_Per_Meter': 'Load_Per_Meter_I'
        })
        
        # Merge final and initial data
        df_merged = pd.merge(df_hour_final, df_hour_initial, on=['TradeDateTime', 'Profile'], how='right')
        df_processed = df_merged[['TradeDateTime', 'Profile', 'Count', 'Load_Per_Meter', 'Count_I', 'Load_Per_Meter_I']].copy()
        
        return df_processed
    
    def _extend_dataset_optimized(self, df):
        """Memory-optimized dataset extension"""
        max_date = df['TradeDateTime'].max()
        extended_dates = pd.date_range(start=max_date + pd.Timedelta(hours=1), periods=40 * 24, freq='h')
        profiles = df['Profile'].unique()
        
        extended_df = pd.DataFrame({'TradeDateTime': extended_dates}).merge(
            pd.DataFrame(profiles, columns=['Profile']), how='cross'
        )
        
        df_extended = pd.concat([df, extended_df], ignore_index=True)
        logger.info(f"Extended dataset: {len(df_extended)} records")
        
        return df_extended
    
    def _add_date_features_optimized(self, df):
        """Memory-optimized date feature addition"""
        df['Year'] = df['TradeDateTime'].dt.year
        df['Month'] = df['TradeDateTime'].dt.month
        df['Day'] = df['TradeDateTime'].dt.day
        df['Hour'] = df['TradeDateTime'].dt.hour
        df['Weekday'] = df['TradeDateTime'].dt.day_name()
        df['Season'] = df['Month'].map({
            1: 'Winter', 2: 'Winter', 3: 'Winter', 4: 'Winter', 5: 'Winter',
            6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Summer', 10: 'Summer',
            11: 'Winter', 12: 'Winter'
        })
        
        # Add holidays and workday features
        holidays = [
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
        
        df['TradeDate'] = df['TradeDateTime'].dt.date.astype(str)
        df['Holiday'] = df['TradeDate'].isin(holidays).astype(int)
        df['Workday'] = ((df['Holiday'] == 0) & (~df['Weekday'].isin(['Saturday', 'Sunday']))).astype(int)
        
        logger.info("Added date features and holiday information")
        return df
    
    def _build_count_query(self, current_date=None):
        """Build query to get total row count"""
        schema_name = self.redshift_config['schema']
        table_name = self.redshift_config['table']
        
        if current_date is None:
            current_date = datetime.now()
        
        data_period_days = self.config.get_data_reading_period_days()
        
        if data_period_days:
            start_date = current_date - timedelta(days=data_period_days)
            where_clause = f"WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'"
        else:
            where_clause = ""
        
        return f"SELECT COUNT(*) as total_rows FROM {schema_name}.{table_name} {where_clause}"
    
    def _get_total_row_count(self, count_query):
        """Get total row count"""
        try:
            result_df = self.execute_query(count_query)
            total_rows = int(result_df.iloc[0, 0])
            return total_rows
        except Exception as e:
            logger.error(f"Error getting row count: {str(e)}")
            # Fallback: assume large dataset and use chunking anyway
            return 1000000  # Default assumption
    
    def _build_chunk_query(self, current_date=None, limit=50000, offset=0):
        """Build query for a specific chunk"""
        schema_name = self.redshift_config['schema']
        table_name = self.redshift_config['table']
        
        if current_date is None:
            current_date = datetime.now()
        
        data_period_days = self.config.get_data_reading_period_days()
        
        if data_period_days:
            start_date = current_date - timedelta(days=data_period_days)
            where_clause = f"WHERE tradedatelocal >= '{start_date.strftime('%Y-%m-%d')}'"
        else:
            where_clause = ""
        
        return f"""
        SELECT
            tradedatelocal as tradedate,
            tradehourstartlocal as tradetime,
            loadprofile, rategroup, baseload, lossadjustedload, metercount,
            loadbl, loadlal, loadmetercount, genbl, genlal, genmetercount,
            submission, createddate as created
        FROM {schema_name}.{table_name}
        {where_clause}
        ORDER BY tradedatelocal, tradehourstartlocal, loadprofile
        LIMIT {limit} OFFSET {offset}
        """
    
    def _check_memory_usage(self):
        """Check current memory usage and warn if high"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            available_gb = psutil.virtual_memory().available / (1024**3)
            
            logger.info(f"Memory usage: {memory_percent:.1%}, Available: {available_gb:.1f} GB")
            
            if memory_percent > self.memory_threshold:
                logger.warning(f"High memory usage detected: {memory_percent:.1%}")
                logger.warning("Consider reducing chunk size or upgrading instance type")
                
                # Force aggressive garbage collection
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Could not check memory usage: {str(e)}")
 
 
class MemoryOptimizedEnergyForecastingConfig(EnergyForecastingConfig):
    """Memory-optimized configuration with environment variable support"""
    
    def __init__(self, config_file=None):
        super().__init__(config_file)
        
        # Apply memory optimization settings from JSON config
        self._apply_memory_optimization_settings()
        
        # Override with environment variables if available
        self._apply_environment_overrides()
    
    def _apply_memory_optimization_settings(self):
        """Apply memory optimization settings from JSON configuration"""
        processing_config = PROCESSING_CONFIG
        
        # Set chunk size based on environment and JSON config
        # default_chunk_size = processing_config.get('chunk_size', 10000)
        if ENVIRONMENT == 'dev':
            default_chunk_size = 50000
        elif ENVIRONMENT == 'preprod':
            default_chunk_size = 60000
        elif ENVIRONMENT == 'prod':
            default_chunk_size = 70000
        
        chunk_size = processing_config.get('chunk_size', default_chunk_size)
            
        self.config['redshift']['chunk_size'] = chunk_size
        
        # Enable memory optimization if specified
        if processing_config.get('parallel_processing', True):
            self.config['memory_optimization'] = {
                'enabled': True,
                'chunk_processing': True,
                'garbage_collection': True,
                'memory_monitoring': True
            }
            
        logger.info(f"Memory optimization applied: chunk_size={default_chunk_size}")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides for container optimization"""

        # Data reading period override
        env_data_period = os.getenv('DATA_READING_PERIOD_DAYS')
        if env_data_period:
            try:
                self.config['redshift']['data_reading_period_days'] = float(env_data_period)
                logger.info(f"Override: data_reading_period_days = {env_data_period}")
            except ValueError:
                logger.warning(f"Invalid DATA_READING_PERIOD_DAYS: {env_data_period}")
        
        # Query limit override
        env_query_limit = os.getenv('QUERY_LIMIT')
        if env_query_limit:
            try:
                self.config['redshift']['query_limit'] = int(env_query_limit)
                logger.info(f"Override: query_limit = {env_query_limit}")
            except ValueError:
                logger.warning(f"Invalid QUERY_LIMIT: {env_query_limit}")        
        
        # Chunk size for memory optimization
        env_chunk_size = os.getenv('CHUNK_SIZE')
        if env_chunk_size:
            try:
                self.config['redshift']['chunk_size'] = int(env_chunk_size)
                logger.info(f"Override: chunk_size = {env_chunk_size}")
            except ValueError:
                logger.warning(f"Invalid CHUNK_SIZE: {env_chunk_size}")
        
        # Memory optimization mode
        if os.getenv('MEMORY_OPTIMIZATION') == '1':
            logger.info("Memory optimization mode enabled")
            self.config['memory_optimization'] = {
                'enabled': True,
                'chunk_processing': True,
                'garbage_collection': True,
                'memory_monitoring': True
            }
    
    def get_query_limit(self):
        """Get query limit for large datasets"""
        return self.config['redshift'].get('query_limit', None)
    
    def get_chunk_size(self):
        """Get chunk size for memory optimization"""
        return self.config['redshift'].get('chunk_size', 50000)
    
    def is_memory_optimization_enabled(self):
        """Check if memory optimization is enabled"""
        return self.config.get('memory_optimization', {}).get('enabled', False)
 
 
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