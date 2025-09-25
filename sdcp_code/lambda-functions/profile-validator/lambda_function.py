"""
lambda-functions/profile-validator/lambda_function.py
Validates and filters profiles based on S3 configurations
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import re
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')

# Security: Input sanitization patterns
LOG_SANITIZER = {
    # Remove/replace potentially dangerous characters for logging
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
   
    # Convert to string
    str_value = str(value)
   
    # Truncate if too long
    if len(str_value) > max_length:
        str_value = str_value[:max_length] + "...[truncated]"
   
    # Remove dangerous characters
    str_value = LOG_SANITIZER['newlines'].sub(' ', str_value)
    str_value = LOG_SANITIZER['tabs'].sub(' ', str_value)
    str_value = LOG_SANITIZER['control_chars'].sub('', str_value)
    str_value = LOG_SANITIZER['excessive_whitespace'].sub(' ', str_value)
   
    # Strip whitespace
    str_value = str_value.strip()
   
    return str_value

def sanitize_event_for_logging(event):
    """
    Create a sanitized version of event for safe logging
   
    Args:
        event: Lambda event object
   
    Returns:
        Dictionary with sanitized event data safe for logging
    """
    sanitized_event = {}
   
    # Only log safe, non-sensitive event fields
    safe_fields = [
        'operation', 'profiles', 'data_bucket', 'model_bucket'
    ]
   
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
   
    # Add metadata without sensitive info
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profiles': 'profiles' in event,
        'has_buckets': bool(event.get('data_bucket') or event.get('model_bucket'))
    }
   
    return sanitized_event

def lambda_handler(event, context):
    """
    Validate and filter profiles based on available S3 configurations
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting profile validation [execution_id={sanitize_for_logging(execution_id)}]")
       
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Extract profiles and configuration with input validation
        requested_profiles = event.get('profiles', [])
        data_bucket = event.get('data_bucket')
        model_bucket = event.get('model_bucket')
        operation = event.get('operation', 'validate_and_filter_profiles')
       
        # Input validation
        if not isinstance(requested_profiles, list):
            requested_profiles = []
           
        # Sanitize profile names for logging
        safe_profiles_for_log = [sanitize_for_logging(p, 20) for p in requested_profiles[:10]]  # Max 10 profiles in log
       
        if not requested_profiles:
            requested_profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]  # Default all
            safe_profiles_for_log = requested_profiles
       
        logger.info(f"Validating profiles count={len(requested_profiles)} sample_profiles={safe_profiles_for_log}")
       
        # Validate profiles
        if operation == 'validate_and_filter_profiles':
            validation_result = validate_profiles_with_s3_configs(
                requested_profiles, data_bucket, execution_id
            )
        else:
            raise ValueError(f"Unknown operation: {sanitize_for_logging(operation)}")
       
        return {
            'statusCode': 200,
            'body': validation_result
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Profile validation failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'message': 'Profile validation failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def validate_profiles_with_s3_configs(profiles: List[str], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Validate that each profile has a valid S3 endpoint configuration
    """
   
    try:
        logger.info(f"Checking S3 configurations for profile_count={len(profiles)} bucket={sanitize_for_logging(data_bucket, 100)}")
       
        valid_profiles = []
        invalid_profiles = []
       
        for profile in profiles:
            # SECURITY FIX: Sanitize profile name before logging
            safe_profile = sanitize_for_logging(profile, 50)
           
            try:
                # Validate profile name (additional security)
                if not profile or not isinstance(profile, str) or len(profile) > 100:
                    invalid_profiles.append({
                        'profile': safe_profile,
                        'reason': 'Invalid profile name format',
                        'validation_status': 'invalid_format'
                    })
                    continue
               
                # Check if S3 configuration exists
                try:
                    response = s3_client.list_objects_v2(
                        Bucket=data_bucket,
                        Prefix=f"sdcp_modeling/endpoint-configurations/{profile}",
                        MaxKeys=1000
                    )
                   
                    if not response.get('Contents'):
                        invalid_profiles.append({
                            'profile': safe_profile,
                            'reason': 'No endpoint configuration files found',
                            'validation_status': 'not_found'
                        })
                        continue
                   
                    # Get the latest file
                    sorted_files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
                    config_key = sorted_files[0]['Key']
                   
                except Exception as e:
                    error_msg = sanitize_for_logging(str(e))
                    invalid_profiles.append({
                        'profile': safe_profile,
                        'reason': f"Error searching for config files: {error_msg}",
                        'validation_status': 'error'
                    })
                    continue

                # SECURITY FIX: Sanitize S3 path for logging
                safe_s3_path = sanitize_for_logging(f"s3://{data_bucket}/{config_key}", 200)
                logger.info(f"Checking S3 config profile={safe_profile} path={safe_s3_path}")
               
                # Try to get the configuration file
                response = s3_client.get_object(Bucket=data_bucket, Key=config_key)
                config_data = json.loads(response['Body'].read())
               
                # Validate required fields
                required_fields = ['endpoint_config_name', 'model_name']
                if all(field in config_data for field in required_fields):
                    valid_profiles.append({
                        'profile': profile,  # Use original profile name in response, not sanitized version
                        's3_config_path': f"s3://{data_bucket}/{config_key}",
                        'config_data': config_data,
                        'validation_status': 'valid'
                    })
                    logger.info(f"Valid configuration found profile={safe_profile}")
                else:
                    missing_fields = [field for field in required_fields if field not in config_data]
                    safe_missing_fields = [sanitize_for_logging(field, 50) for field in missing_fields]
                    invalid_profiles.append({
                        'profile': safe_profile,
                        'reason': f"Missing required fields: {safe_missing_fields}",
                        'validation_status': 'invalid'
                    })
                    logger.warning(f"Missing fields profile={safe_profile} missing={safe_missing_fields}")
               
            except s3_client.exceptions.NoSuchKey:
                invalid_profiles.append({
                    'profile': safe_profile,
                    'reason': 'S3 configuration file not found',
                    'validation_status': 'not_found'
                })
                logger.warning(f"S3 configuration not found profile={safe_profile}")
               
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                invalid_profiles.append({
                    'profile': safe_profile,
                    'reason': f"Configuration validation error: {error_msg}",
                    'validation_status': 'error'
                })
                logger.error(f"Validation error profile={safe_profile} error={error_msg}")
       
        result = {
            'requested_profiles': profiles,  # Return original profile names
            'valid_profiles': valid_profiles,
            'invalid_profiles': invalid_profiles,
            'valid_profiles_count': len(valid_profiles),
            'invalid_profiles_count': len(invalid_profiles),
            'validation_timestamp': datetime.now().isoformat(),
            'execution_id': execution_id
        }
       
        logger.info(f"Validation complete valid_count={len(valid_profiles)} invalid_count={len(invalid_profiles)} execution_id={sanitize_for_logging(execution_id)}")
       
        return result
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Profile validation process failed error={error_msg} execution_id={sanitize_for_logging(execution_id)}")
        raise
