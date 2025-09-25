"""
Enhanced Model Registry Lambda Function
Integrates with Step Functions pipeline and handles automatic model registration
Based on existing working version with Step Functions integration enhancements
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import tempfile
import tarfile
import os
import re
from datetime import datetime
from typing import Dict, Any

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

# Security: Input sanitization patterns
LOG_SANITIZER = {
    'newlines': re.compile(r'[\r\n]+'),
    'tabs': re.compile(r'[\t]+'),
    'control_chars': re.compile(r'[\x00-\x1f\x7f-\x9f]'),
    'excessive_whitespace': re.compile(r'\s{3,}')
}

def sanitize_for_logging(value, max_length=500):
    """
    Sanitize input for safe logging to prevent log injection
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length for logged values
    
    Returns:
        Sanitized string safe for logging
    """
    if value is None:
        return "None"
    
    str_value = str(value)
    
    # Truncate if too long
    if len(str_value) > max_length:
        str_value = str_value[:max_length] + "...[truncated]"
    
    # Remove dangerous characters
    str_value = LOG_SANITIZER['newlines'].sub(' ', str_value)
    str_value = LOG_SANITIZER['tabs'].sub(' ', str_value)
    str_value = LOG_SANITIZER['control_chars'].sub('', str_value)
    str_value = LOG_SANITIZER['excessive_whitespace'].sub(' ', str_value)
    
    return str_value.strip()

def sanitize_event_for_logging(event):
    """Create a sanitized version of event for safe logging"""
    sanitized_event = {}
    
    safe_fields = ['training_date', 'model_bucket', 'data_bucket']
    
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
    
    # Handle training metadata safely
    if 'training_metadata' in event and isinstance(event['training_metadata'], dict):
        sanitized_event['training_metadata_keys'] = list(event['training_metadata'].keys())[:10]  # Max 10 keys
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_training_metadata': 'training_metadata' in event,
        'has_buckets': bool(event.get('model_bucket') or event.get('data_bucket'))
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Enhanced Lambda handler for model registry with Step Functions integration
    Handles both direct invocation and Step Functions pipeline integration
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting enhanced model registry process [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
        
        # Extract information from event (supports both formats)
        training_metadata = event.get('training_metadata', {})
        training_date = event.get('training_date', datetime.now().strftime('%Y%m%d'))
        model_bucket = event.get('model_bucket', os.environ.get('MODEL_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-models'))
        data_bucket = event.get('data_bucket', os.environ.get('DATA_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data'))
        model_prefix = event.get('model_prefix', "sdcp_modeling/forecasting/models/xgboost/")
        registry_prefix = event.get('registry_prefix', "sdcp_modeling/forecasting/registry/")
        
        # Enhanced configuration
        config = {
            "model_bucket": model_bucket,
            "data_bucket": data_bucket,
            "model_prefix": model_prefix,
            "registry_prefix": registry_prefix,
            "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"],
            "customer_profile": "SDCP",
            "region": os.environ.get('REGION', 'us-west-2'),
            "account_id": os.environ.get('ACCOUNT_ID', context.invoked_function_arn.split(':')[4])
        }
        
        # SECURITY FIX: Sanitize config values for logging
        safe_training_date = sanitize_for_logging(training_date, 20)
        safe_model_bucket = sanitize_for_logging(config["model_bucket"], 100)
        safe_data_bucket = sanitize_for_logging(config["data_bucket"], 100)
        safe_model_prefix = sanitize_for_logging(config["model_prefix"], 100)

        logger.info(f"Processing models training_date={safe_training_date} model_bucket={safe_model_bucket} data_bucket={safe_data_bucket} model_prefix={safe_model_prefix}")

        # Step 1: Find latest models for each profile
        latest_models = find_latest_models(config, training_date)
        
        if not latest_models:
            # Check if this is expected (e.g., no new training)
            logger.warning("No models found for registration - checking if this is expected")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'No models found for registration',
                    'execution_id': execution_id,
                    'successful_count': 0,
                    'total_models': 0,
                    'approved_models': {},
                    'training_metadata': training_metadata,
                    'training_date': training_date,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'no_models_found'
                }
            }
        
        models_found_count = len(latest_models)
        logger.info(f"Found models for registration count={models_found_count}")
        
        # Step 2: Create model package groups
        registry_groups = create_model_package_groups(config['profiles'], config['customer_profile'])
        
        # Step 3: Process each model
        approved_models = {}
        successful_count = 0
        processing_results = {}
        
        for profile, model_info in latest_models.items():
            try:
                # SECURITY FIX: Sanitize profile name for logging
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Processing model profile={safe_profile}")
                
                # Package and register model
                result = process_single_model(
                    profile=profile,
                    model_info=model_info,
                    config=config,
                    registry_group=registry_groups.get(profile),
                    execution_id=execution_id
                )
                
                processing_results[profile] = result
                
                if result['status'] == 'success':
                    approved_models[profile] = result
                    successful_count += 1
                    logger.info(f"Successfully processed model profile={safe_profile}")
                else:
                    error_msg = sanitize_for_logging(result.get('error', 'Unknown error'))
                    logger.error(f"Failed to process model profile={safe_profile} error={error_msg}")
                    
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Exception processing profile={safe_profile} error={error_msg}")
                processing_results[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile
                }
                continue
        
        # Step 4: Generate summary and metrics
        summary_metrics = generate_processing_summary(
            latest_models, processing_results, successful_count, execution_id
        )
        
        # Prepare enhanced response for Step Functions
        response = {
            'statusCode': 200,
            'body': {
                'message': f'Enhanced model registry completed for {len(latest_models)} models',
                'execution_id': execution_id,
                'successful_count': successful_count,
                'failed_count': len(latest_models) - successful_count,
                'total_models': len(latest_models),
                'success_rate': (successful_count / len(latest_models)) * 100 if latest_models else 0,
                'approved_models': approved_models,
                'processing_results': processing_results,
                'training_metadata': training_metadata,
                'training_date': training_date,
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'summary_metrics': summary_metrics,
                'next_step_ready': successful_count > 0
            }
        }
        
        logger.info(f"Enhanced model registry process completed [execution_id={sanitize_for_logging(execution_id)}] successful={successful_count} total={len(latest_models)}")
        return response
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Enhanced model registry process failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'message': 'Enhanced model registry process failed',
                'timestamp': datetime.now().isoformat(),
                'next_step_ready': False
            }
        }

def find_latest_models(config: Dict[str, Any], training_date: str = None) -> Dict[str, Dict]:
    """
    Enhanced model discovery with better pattern matching and date handling
    """
    latest_models = {}
    
    try:
        # SECURITY FIX: Sanitize bucket and prefix for logging
        safe_bucket = sanitize_for_logging(config["model_bucket"], 100)
        safe_prefix = sanitize_for_logging(config["model_prefix"], 50)
        logger.info(f"Searching for models bucket={safe_bucket} prefix={safe_prefix}")
        
        response = s3_client.list_objects_v2(
            Bucket=config["model_bucket"],
            Prefix=config["model_prefix"]
        )
        
        if 'Contents' not in response:
            logger.warning(f"No objects found in S3 bucket={safe_bucket} prefix={safe_prefix}")
            return {}
        
        objects_count = len(response['Contents'])
        logger.info(f"Found S3 objects count={objects_count}")
        
        # Parse model files by profile with enhanced pattern matching
        profile_models = {profile: [] for profile in config["profiles"]}
        
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.basename(key)
            
            # Enhanced pattern matching for different naming conventions
            patterns = [
                r'(?:df_)?([A-Z0-9]+)_best_xgboost_(\d{8})\.pkl',  # Current pattern
                r'([A-Z0-9]+)_best_xgboost_(\d{8})\.pkl',          # Alternative pattern
                r'model_([A-Z0-9]+)_(\d{8})\.pkl'                  # Future pattern
            ]
            
            for pattern in patterns:
                match = re.match(pattern, filename)
                if match:
                    profile = match.group(1)
                    date_str = match.group(2)
                    
                    if profile in profile_models:
                        model_info = {
                            'profile': profile,
                            'date': date_str,
                            's3_key': key,
                            'filename': filename,
                            'last_modified': obj['LastModified'],
                            'size': obj['Size'],
                            'bucket': config["model_bucket"]
                        }
                        profile_models[profile].append(model_info)
                        
                        # SECURITY FIX: Sanitize filename and profile for logging
                        safe_filename = sanitize_for_logging(filename, 100)
                        safe_profile = sanitize_for_logging(profile, 20)
                        logger.debug(f"Found model filename={safe_filename} profile={safe_profile}")
                    break
        
        # Find appropriate model for each profile
        for profile, models in profile_models.items():
            safe_profile = sanitize_for_logging(profile, 20)
            
            if models:
                if training_date:
                    # Look for models from specific training date first
                    date_models = [m for m in models if m['date'] == training_date]
                    if date_models:
                        # If multiple models from same date, get the most recent by modification time
                        latest_models[profile] = sorted(date_models, key=lambda x: x['last_modified'], reverse=True)[0]
                        safe_filename = sanitize_for_logging(latest_models[profile]['filename'], 100)
                        safe_training_date = sanitize_for_logging(training_date, 20)
                        logger.info(f"Found model from training date profile={safe_profile} date={safe_training_date} filename={safe_filename}")
                    else:
                        # Fallback to most recent model if no model from specific date
                        latest_model = sorted(models, key=lambda x: x['date'], reverse=True)[0]
                        latest_models[profile] = latest_model
                        safe_filename = sanitize_for_logging(latest_model['filename'], 100)
                        safe_training_date = sanitize_for_logging(training_date, 20)
                        logger.warning(f"No model found from specific date, using latest profile={safe_profile} requested_date={safe_training_date} latest_filename={safe_filename}")
                else:
                    # Get most recent model
                    latest_model = sorted(models, key=lambda x: x['date'], reverse=True)[0]
                    latest_models[profile] = latest_model
                    safe_filename = sanitize_for_logging(latest_model['filename'], 100)
                    logger.info(f"Found latest model profile={safe_profile} filename={safe_filename}")
            else:
                logger.warning(f"No models found profile={safe_profile}")
        
        found_count = len(latest_models)
        logger.info(f"Model discovery completed models_found={found_count}")
        return latest_models
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Error finding latest models error={error_msg}")
        return {}

def create_model_package_groups(profiles: list, customer_profile: str) -> Dict[str, str]:
    """
    Enhanced model package group creation with better error handling
    """
    registry_groups = {}
    
    for profile in profiles:
        group_name = f"EnergyForecastModels-{customer_profile}-{profile}"
        safe_group_name = sanitize_for_logging(group_name, 100)
        safe_profile = sanitize_for_logging(profile, 20)
        
        try:
            # Check if group exists
            sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            logger.info(f"Model package group already exists profile={safe_profile} group={safe_group_name}")
            
        except sagemaker_client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ValidationException' and "does not exist" in str(e):
                # Create new group
                logger.info(f"Creating model package group profile={safe_profile} group={safe_group_name}")
                try:
                    sagemaker_client.create_model_package_group(
                        ModelPackageGroupName=group_name,
                        ModelPackageGroupDescription=f"Energy load forecasting models for {customer_profile} customer profile {profile}"
                    )
                    logger.info(f"Successfully created model package group profile={safe_profile} group={safe_group_name}")
                except Exception as create_error:
                    error_msg = sanitize_for_logging(str(create_error))
                    logger.error(f"Failed to create model package group profile={safe_profile} group={safe_group_name} error={error_msg}")
                    continue
            else:
                error_msg = sanitize_for_logging(str(e))
                logger.error(f"Error checking model package group profile={safe_profile} group={safe_group_name} error={error_msg}")
                continue
        
        registry_groups[profile] = group_name
    
    groups_ready_count = len(registry_groups)
    logger.info(f"Model package groups ready count={groups_ready_count}")
    return registry_groups

def create_enhanced_inference_script(profile: str) -> str:
    """
    Create enhanced SageMaker-compatible inference script with improved error handling
    """
    
    # Enhanced profile-specific feature configurations
    profile_features = {
        'RNN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'RN': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days', 'shortwave_radiation'],
        'M': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'S': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'AGR': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'L': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days'],
        'A6': ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
    }
    
    expected_features = profile_features.get(profile, profile_features['RNN'])
    
    inference_script = f'''
import joblib
import json
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime

# Setup enhanced logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Enhanced profile-specific configuration
PROFILE = "{profile}"
EXPECTED_FEATURES = {expected_features}
MODEL_VERSION = "enhanced_v1.0"

def model_fn(model_dir):
    """Enhanced model loading with error handling and validation"""
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {{model_path}}")
        
        model = joblib.load(model_path)
        
        # Enhanced model validation
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model expects {{len(model.feature_names_in_)}} features")
        
        logger.info(f"Enhanced model loaded successfully for profile {{PROFILE}}")
        logger.info(f"Model type: {{type(model).__name__}}")
        logger.info(f"Expected features: {{len(EXPECTED_FEATURES)}}")
        
        return model
        
    except Exception as e:
        logger.error(f"Enhanced model loading failed: {{str(e)}}")
        raise

def input_fn(request_body, content_type):
    """Enhanced input processing with SIMPLE COLUMN ASSIGNMENT FIX"""
    try:
        logger.info(f"Processing input for profile {{PROFILE}}, content_type: {{content_type}}")
       
        if content_type == 'application/json':
            data = json.loads(request_body)
           
            # Handle different input formats
            if 'instances' in data:
                instances = data['instances']
                logger.info("Processing 'instances' format")
            elif isinstance(data, list) and len(data) > 0:
                instances = data
                logger.info("Processing list format")
            else:
                raise ValueError(f"Unsupported data format: {{type(data)}}")

            # Create DataFrame from instances
            df = pd.DataFrame(instances)
            logger.info(f"Created DataFrame with {{len(df)}} rows and {{len(df.columns)}} columns")

            # KEY FIX: Simply assign the correct column names based on profile
            if PROFILE == 'RN':
                # RN profile has 13 features (includes shortwave_radiation)
                expected_columns = ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
                                  'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days',
                                  'Load_lag_70_days', 'shortwave_radiation']
            else:
                # All other profiles have 12 features
                expected_columns = ['Count', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season',
                                  'Holiday', 'Workday', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days']
           
            # Validate column count matches expected
            if len(df.columns) != len(expected_columns):
                raise ValueError(f"Column count mismatch: got {{len(df.columns)}}, expected {{len(expected_columns)}} for profile {{PROFILE}}")

            # SIMPLE FIX: Assign correct column names
            df.columns = expected_columns
            logger.info(f"Assigned column names: {{list(df.columns)}}")

            # Now enhanced_preprocess_features will work perfectly!
            df = enhanced_preprocess_features(df)
           
            # Select features in correct order
            df_features = df[EXPECTED_FEATURES]
           
            # Enhanced data validation
            validate_input_data(df_features)

            logger.info(f"Input processing completed: {{len(df_features)}} rows, {{len(df_features.columns)}} features")
            logger.info(f"Sample row: {{df_features.iloc[0].to_dict()}}")

            return df_features.values
           
        else:
            raise ValueError(f"Unsupported content type: {{content_type}}")

    except Exception as e:
        logger.error(f"Input processing failed: {{str(e)}}")
        raise

def enhanced_preprocess_features(df):
    """Enhanced feature preprocessing with comprehensive transformations"""
    try:
        logger.info("Starting enhanced feature preprocessing")
        
        # Enhanced categorical encoding
        if 'Weekday' in df.columns:
            if df['Weekday'].dtype == 'object':
                weekday_map = {{
                    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7,
                    'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7
                }}
                df['Weekday'] = df['Weekday'].map(weekday_map).fillna(1)
                logger.info("Enhanced weekday encoding completed")
       
        if 'Season' in df.columns:
            if df['Season'].dtype == 'object':
                season_map = {{
                    'Summer': 1, 'Winter': 0, 'summer': 1, 'winter': 0,
                    'SUMMER': 1, 'WINTER': 0
                }}
                df['Season'] = df['Season'].map(season_map).fillna(0)
                logger.info("Enhanced season encoding completed")
       
        # Enhanced missing value handling with profile-specific defaults
        default_values = get_profile_defaults()
        df = df.fillna(default_values)
        
        # Enhanced data type validation
        for col in df.columns:
            if col in EXPECTED_FEATURES:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info("Enhanced feature preprocessing completed")
        return df
        
    except Exception as e:
        logger.error(f"Enhanced preprocessing failed: {{str(e)}}")
        raise

def get_default_value(feature_name):
    """Get enhanced default values for missing features"""
    defaults = {{
        'Count': 1000,
        'Year': datetime.now().year,
        'Month': datetime.now().month,
        'Day': datetime.now().day,
        'Hour': 12,
        'Weekday': 1,
        'Season': 1,
        'Holiday': 0,
        'Workday': 1,
        'Temperature': 70.0,
        'Load_I_lag_14_days': 0.5,
        'Load_lag_70_days': 0.5,
        'shortwave_radiation': 200.0
    }}
    return defaults.get(feature_name, 0)

def get_profile_defaults():
    """Get profile-specific default values"""
    base_defaults = {{
        'Count': 1000,
        'Temperature': 70.0,
        'Load_I_lag_14_days': 0.5,
        'Load_lag_70_days': 0.5,
        'shortwave_radiation': 200.0,
        'Holiday': 0,
        'Workday': 1,
        'Season': 1,
        'Weekday': 1
    }}
    
    # Profile-specific adjustments
    if PROFILE == 'RN':  # Residential with solar
        base_defaults['shortwave_radiation'] = 300.0
    elif PROFILE in ['M', 'S']:  # Commercial profiles
        base_defaults['Count'] = 500
        base_defaults['Load_I_lag_14_days'] = 0.7
    
    return base_defaults

def validate_input_data(df):
    """Enhanced input data validation"""
    try:
        # Check for infinite values
        if np.isinf(df.values).any():
            raise ValueError("Input data contains infinite values")
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.5:
            raise ValueError(f"Too many missing values: {{missing_pct:.1%}}")
        
        # Check data ranges
        for col in df.columns:
            if col == 'Temperature' and (df[col].min() < -50 or df[col].max() > 150):
                logger.warning(f"Temperature values outside expected range: {{df[col].min()}} to {{df[col].max()}}")
            elif col == 'Hour' and (df[col].min() < 0 or df[col].max() > 23):
                raise ValueError(f"Hour values outside valid range: {{df[col].min()}} to {{df[col].max()}}")
        
        logger.info("Enhanced input validation passed")
        
    except Exception as e:
        logger.error(f"Enhanced input validation failed: {{str(e)}}")
        raise

def predict_fn(input_data, model):
    """Enhanced prediction with comprehensive error handling"""
    try:
        logger.info(f"Starting enhanced prediction for {{len(input_data)}} samples")
        
        # Enhanced prediction with validation
        predictions = model.predict(input_data)
        
        # Enhanced prediction validation
        if len(predictions) != len(input_data):
            raise ValueError(f"Prediction count mismatch: expected {{len(input_data)}}, got {{len(predictions)}}")
        
        # Check for invalid predictions
        if np.isnan(predictions).any():
            logger.warning("Some predictions are NaN, replacing with median")
            median_pred = np.nanmedian(predictions)
            predictions = np.where(np.isnan(predictions), median_pred, predictions)
        
        if (predictions < 0).any():
            logger.warning("Some predictions are negative, clipping to zero")
            predictions = np.maximum(predictions, 0)
        
        logger.info(f"Enhanced prediction completed: {{len(predictions)}} predictions generated")
        logger.info(f"Prediction range: {{predictions.min():.4f}} to {{predictions.max():.4f}}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed: {{str(e)}}")
        raise

def output_fn(prediction, accept):
    """Enhanced output formatting with comprehensive metadata"""
    try:
        logger.info(f"Formatting enhanced output for {{len(prediction)}} predictions")
        
        if accept == 'application/json':
            # Convert numpy arrays to lists for JSON serialization
            if hasattr(prediction, 'tolist'):
                prediction_list = prediction.tolist()
            else:
                prediction_list = list(prediction)
           
            # Enhanced response with comprehensive metadata
            response = {{
                "predictions": prediction_list,
                "metadata": {{
                    "profile": PROFILE,
                    "model_version": MODEL_VERSION,
                    "prediction_count": len(prediction_list),
                    "timestamp": datetime.now().isoformat(),
                    "statistics": {{
                        "min": float(min(prediction_list)),
                        "max": float(max(prediction_list)),
                        "mean": float(sum(prediction_list) / len(prediction_list)),
                        "total": float(sum(prediction_list))
                    }},
                    "expected_features": EXPECTED_FEATURES,
                    "feature_count": len(EXPECTED_FEATURES)
                }}
            }}
           
            logger.info("Enhanced output formatting completed")
            return json.dumps(response)
            
        else:
            raise ValueError(f"Unsupported accept type: {{accept}}")
           
    except Exception as e:
        logger.error(f"Enhanced output formatting failed: {{str(e)}}")
        raise
'''
    
    return inference_script

def create_enhanced_sagemaker_model_archive(profile: str, model_info: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Create enhanced SageMaker-compatible tar.gz archive with comprehensive packaging
    """
    
    try:
        # SECURITY FIX: Sanitize profile for logging
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Creating enhanced SageMaker model archive profile={safe_profile}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Download the pickle file
            pkl_local_path = os.path.join(temp_dir, 'model.pkl')
            
            # SECURITY FIX: Sanitize S3 info for logging
            safe_bucket = sanitize_for_logging(model_info['bucket'], 100)
            safe_s3_key = sanitize_for_logging(model_info['s3_key'], 200)
            logger.info(f"Downloading model bucket={safe_bucket} key={safe_s3_key}")
            
            s3_client.download_file(
                model_info['bucket'], 
                model_info['s3_key'], 
                pkl_local_path
            )
            logger.info(f"Successfully downloaded model file profile={safe_profile}")
            
            # Step 2: Create enhanced inference script
            inference_script = create_enhanced_inference_script(profile)
            inference_path = os.path.join(temp_dir, 'inference.py')
            with open(inference_path, 'w') as f:
                f.write(inference_script)
            logger.info(f"Created enhanced inference script profile={safe_profile}")
            
            # Step 3: Create enhanced requirements.txt
            requirements_content = """joblib>=1.1.0
scikit-learn==1.3.2
xgboost==1.7.6
numpy>=1.21.0
pandas>=1.5.0
"""
            requirements_path = os.path.join(temp_dir, 'requirements.txt')
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            # Step 4: Create enhanced model metadata
            metadata_content = {
                "profile": profile,
                "model_type": "XGBoost",
                "framework": "scikit-learn",
                "training_date": model_info['date'],
                "created_at": datetime.now().isoformat(),
                "sagemaker_compatible": True,
                "inference_script": "inference.py",
                "model_file": "model.pkl",
                "enhancement_version": "v2.0",
                "original_s3_location": f"s3://{model_info['bucket']}/{model_info['s3_key']}",
                "model_size_bytes": model_info['size'],
                "last_modified": model_info['last_modified'].isoformat() if isinstance(model_info['last_modified'], datetime) else str(model_info['last_modified'])
            }
            
            metadata_path = os.path.join(temp_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata_content, f, indent=2)
            
            # Step 5: Create enhanced tar.gz archive
            archive_path = os.path.join(temp_dir, 'model.tar.gz')
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(pkl_local_path, arcname='model.pkl')
                tar.add(inference_path, arcname='inference.py')
                tar.add(requirements_path, arcname='requirements.txt')
                tar.add(metadata_path, arcname='model_metadata.json')
            
            logger.info(f"Created enhanced tar.gz archive profile={safe_profile}")
            
            # Step 6: Upload archive to S3 with enhanced naming
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            archive_s3_key = f"{config['registry_prefix']}{profile}/enhanced_model_{model_info['date']}_{timestamp}.tar.gz"
            
            s3_client.upload_file(archive_path, config["model_bucket"], archive_s3_key)
            
            archive_s3_uri = f"s3://{config['model_bucket']}/{archive_s3_key}"
            safe_archive_uri = sanitize_for_logging(archive_s3_uri, 200)
            logger.info(f"Uploaded enhanced model archive profile={safe_profile} uri={safe_archive_uri}")
            
            return archive_s3_uri
            
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to create enhanced model archive profile={safe_profile} error={error_msg}")
        return None

def process_single_model(profile: str, model_info: Dict[str, Any], config: Dict[str, Any], 
                        registry_group: str, execution_id: str) -> Dict[str, Any]:
    """
    Enhanced single model processing with comprehensive error handling and metrics
    """
    
    result = {
        'profile': profile,
        'status': 'failed',
        'execution_id': execution_id,
        'processing_start_time': datetime.now().isoformat()
    }
    
    try:
        # SECURITY FIX: Sanitize profile for logging
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Starting enhanced processing profile={safe_profile}")
        
        # Step 1: Create enhanced SageMaker model archive
        archive_s3_uri = create_enhanced_sagemaker_model_archive(profile, model_info, config)
        
        if not archive_s3_uri:
            result['error'] = "Failed to create enhanced model archive"
            result['processing_end_time'] = datetime.now().isoformat()
            return result
        
        result['archive_s3_uri'] = archive_s3_uri
        
        # Step 2: Register model in SageMaker Model Registry with enhancements
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        model_package_response = sagemaker_client.create_model_package(
            ModelPackageGroupName=registry_group,
            ModelPackageDescription=f"Enhanced energy forecasting model for {profile} profile - {timestamp}",
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                        'ModelDataUrl': archive_s3_uri,
                        'Environment': {
                            'SAGEMAKER_PROGRAM': 'inference.py',
                            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model',
                            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                            'SAGEMAKER_REGION': config['region'],
                            'MODEL_PROFILE': profile,
                            'MODEL_VERSION': 'enhanced_v2.0'
                        }
                    }
                ],
                'SupportedTransformInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge'],
                'SupportedRealtimeInferenceInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge', 'ml.t2.medium'],
                'SupportedContentTypes': ['application/json'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            ModelApprovalStatus='Approved'  # Auto-approve enhanced models
        )
        
        model_package_arn = model_package_response['ModelPackageArn']
        
        # Enhanced result with comprehensive information
        result.update({
            'model_package_arn': model_package_arn,
            'model_package_group': registry_group,
            'status': 'success',
            'approval_status': 'Approved',
            'registration_time': datetime.now().isoformat(),
            'processing_end_time': datetime.now().isoformat(),
            'model_metadata': {
                'original_s3_key': model_info['s3_key'],
                'model_size_bytes': model_info['size'],
                'training_date': model_info['date'],
                'filename': model_info['filename']
            },
            'enhancement_features': [
                'comprehensive_error_handling',
                'enhanced_input_validation',
                'profile_specific_defaults',
                'advanced_preprocessing',
                'detailed_logging'
            ]
        })
        
        safe_model_package_arn = sanitize_for_logging(model_package_arn, 200)
        logger.info(f"Successfully processed enhanced model profile={safe_profile} arn={safe_model_package_arn}")
        
        return result
        
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to process enhanced model profile={safe_profile} error={error_msg}")
        result.update({
            'error': error_msg,
            'processing_end_time': datetime.now().isoformat()
        })
        return result

def generate_processing_summary(latest_models: Dict, processing_results: Dict, 
                              successful_count: int, execution_id: str) -> Dict:
    """
    Generate comprehensive processing summary with enhanced metrics
    """
    
    summary = {
        'execution_id': execution_id,
        'total_profiles_expected': len(['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6']),
        'models_found': len(latest_models),
        'models_processed': len(processing_results),
        'successful_registrations': successful_count,
        'failed_registrations': len(processing_results) - successful_count,
        'success_rate_percent': (successful_count / len(processing_results)) * 100 if processing_results else 0,
        'processing_timestamp': datetime.now().isoformat(),
        'profile_status': {}
    }
    
    # Add detailed profile status
    all_profiles = ['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6']
    for profile in all_profiles:
        if profile in processing_results:
            result = processing_results[profile]
            summary['profile_status'][profile] = {
                'status': result['status'],
                'model_found': profile in latest_models,
                'registered': result['status'] == 'success',
                'error': result.get('error', None)
            }
            if result['status'] == 'success':
                summary['profile_status'][profile]['model_package_arn'] = result.get('model_package_arn')
        else:
            summary['profile_status'][profile] = {
                'status': 'not_processed',
                'model_found': profile in latest_models,
                'registered': False,
                'error': 'Model not found or not processed'
            }
    
    return summary
