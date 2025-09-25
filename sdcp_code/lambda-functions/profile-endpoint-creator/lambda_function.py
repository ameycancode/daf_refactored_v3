"""
Fixed Profile Endpoint Creator Lambda Function
Handles Model Package vs Direct Container approach correctly
lambda-functions/profile-endpoint-creator/lambda_function.py
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import time
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
   
    safe_fields = ['operation', 'profile', 'data_bucket', 'model_bucket']
   
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
   
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profile': 'profile' in event,
        'has_s3_config': 's3_config_path' in event
    }
   
    return sanitized_event

def lambda_handler(event, context):
    """
    Create or check status of SageMaker endpoints for individual profiles
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting profile endpoint management [execution_id={sanitize_for_logging(execution_id)}]")
       
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Extract operation and profile details
        operation = event.get('operation', 'create_endpoint')
        profile = event.get('profile')
       
        if not profile:
            raise ValueError("Profile is required for endpoint operations")
       
        # SECURITY FIX: Sanitize profile name before logging
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Processing operation={sanitize_for_logging(operation)} profile={safe_profile}")
       
        # Handle different operations
        if operation == 'create_endpoint':
            result = create_endpoint_from_s3_config(event, execution_id)
        elif operation == 'check_endpoint_status':
            result = check_endpoint_status(event, execution_id)
        else:
            raise ValueError(f"Unknown operation: {sanitize_for_logging(operation)}")
       
        return {
            'statusCode': 200,
            'body': result
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Profile endpoint management failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'message': 'Profile endpoint management failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def create_endpoint_from_s3_config(event: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Create SageMaker endpoint from S3 configuration for a specific profile
    Creates an exact replica of the previously deleted endpoint
    """
   
    try:
        profile = event['profile']
        s3_config_path = event.get('s3_config_path')
        data_bucket = event.get('data_bucket')
       
        # SECURITY FIX: Sanitize values for logging
        safe_profile = sanitize_for_logging(profile, 50)
        safe_s3_path = sanitize_for_logging(s3_config_path, 200) if s3_config_path else "default"
       
        logger.info(f"Creating endpoint profile={safe_profile} s3_path={safe_s3_path}")
       
        # Load configuration from S3
        if s3_config_path:
            # Extract bucket and key from S3 path
            s3_parts = s3_config_path.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
        else:
            # Default path for profile-specific configurations
            bucket = data_bucket
            key = f"endpoint-configurations/{profile}/endpoint_config.json"
       
        safe_bucket = sanitize_for_logging(bucket, 100)
        safe_key = sanitize_for_logging(key, 200)
        logger.info(f"Loading config from bucket={safe_bucket} key={safe_key}")
       
        response = s3_client.get_object(Bucket=bucket, Key=key)
        config_data = json.loads(response['Body'].read())
       
        logger.info("Successfully loaded endpoint configuration from S3")
       
        # Generate unique names for new resources
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_suffix = f"{timestamp}-{execution_id[:8]}"
       
        # Generate new resource names
        new_model_name = f"energy-forecasting-{profile.lower()}-model-{unique_suffix}"
        new_endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-{unique_suffix}"
        new_endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{unique_suffix}"
       
        # SECURITY FIX: Sanitize resource names for logging
        safe_model_name = sanitize_for_logging(new_model_name, 100)
        safe_endpoint_config_name = sanitize_for_logging(new_endpoint_config_name, 100)
        safe_endpoint_name = sanitize_for_logging(new_endpoint_name, 100)
       
        logger.info(f"Generated resource names model={safe_model_name} config={safe_endpoint_config_name} endpoint={safe_endpoint_name}")
       
        # Step 1: Create Model using the correct approach based on available configuration
        logger.info("Step 1: Creating SageMaker Model...")
        model_config = config_data.get('model_config', {})
       
        if not model_config:
            raise ValueError("model_config not found in S3 configuration")
       
        # Determine which approach to use: Model Package vs Direct Container
        primary_container = model_config.get('primary_container', {})
        model_package_name = model_config.get('model_package_name') or primary_container.get('ModelPackageName')
       
        if model_package_name:
            # Approach 1: Use Model Package (Preferred for MLOps)
            safe_package_name = sanitize_for_logging(model_package_name, 200)
            logger.info(f"Using Model Package approach package={safe_package_name}")
           
            create_model_params = {
                'ModelName': new_model_name,
                'ExecutionRoleArn': model_config['execution_role_arn'],
                'PrimaryContainer': {
                    'ModelPackageName': model_package_name
                },
                'Tags': model_config.get('tags', [])
            }
           
            # Add any additional environment variables if specified
            if 'Environment' in primary_container:
                create_model_params['PrimaryContainer']['Environment'] = primary_container['Environment']
                env_keys = list(primary_container['Environment'].keys())
                safe_env_keys = [sanitize_for_logging(k, 50) for k in env_keys[:5]]  # Log max 5 keys
                logger.info(f"Added environment variables keys={safe_env_keys}")
       
        else:
            # Approach 2: Use Direct Container specification
            logger.info("Using Direct Container approach")
           
            if not primary_container.get('Image') or not primary_container.get('ModelDataUrl'):
                raise ValueError("Either ModelPackageName or both Image and ModelDataUrl must be specified")
           
            safe_image = sanitize_for_logging(primary_container['Image'], 200)
            safe_model_url = sanitize_for_logging(primary_container['ModelDataUrl'], 200)
            logger.info(f"Using container image={safe_image} model_url={safe_model_url}")
           
            create_model_params = {
                'ModelName': new_model_name,
                'ExecutionRoleArn': model_config['execution_role_arn'],
                'PrimaryContainer': {
                    'Image': primary_container['Image'],
                    'ModelDataUrl': primary_container['ModelDataUrl'],
                    'Mode': primary_container.get('Mode', 'SingleModel')
                },
                'Tags': model_config.get('tags', [])
            }
           
            # Add optional container configuration
            if 'Environment' in primary_container:
                create_model_params['PrimaryContainer']['Environment'] = primary_container['Environment']
           
            if 'ModelDataSource' in primary_container:
                create_model_params['PrimaryContainer']['ModelDataSource'] = primary_container['ModelDataSource']
       
        logger.info("Creating model with provided configuration")
       
        sagemaker_client.create_model(**create_model_params)
        logger.info(f"Successfully created model name={safe_model_name}")
       
        # Step 2: Create Endpoint Configuration using the exact endpoint_config from JSON
        logger.info("Step 2: Creating Endpoint Configuration...")
        endpoint_config = config_data.get('endpoint_config', {})
       
        if not endpoint_config:
            raise ValueError("endpoint_config not found in S3 configuration")
       
        # Prepare production variants with new model name
        production_variants = []
        for variant in endpoint_config.get('production_variants', []):
            new_variant = variant.copy()
            new_variant['ModelName'] = new_model_name  # Use the new model name
            production_variants.append(new_variant)
       
        create_endpoint_config_params = {
            'EndpointConfigName': new_endpoint_config_name,
            'ProductionVariants': production_variants,
            'Tags': endpoint_config.get('tags', [])
        }
       
        logger.info("Creating endpoint configuration with provided parameters")
       
        sagemaker_client.create_endpoint_config(**create_endpoint_config_params)
        logger.info(f"Successfully created endpoint configuration name={safe_endpoint_config_name}")
       
        # Step 3: Create Endpoint
        logger.info("Step 3: Creating Endpoint...")
       
        create_endpoint_params = {
            'EndpointName': new_endpoint_name,
            'EndpointConfigName': new_endpoint_config_name,
            'Tags': [
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'ExecutionId', 'Value': execution_id},
                {'Key': 'CreatedBy', 'Value': 'EnhancedPredictionPipeline'},
                {'Key': 'Purpose', 'Value': 'OnDemandPrediction'},
                {'Key': 'CostOptimized', 'Value': 'True'},
                {'Key': 'OriginalProfile', 'Value': profile},
                {'Key': 'RecreatedFrom', 'Value': 'S3Config'}
            ]
        }
       
        create_response = sagemaker_client.create_endpoint(**create_endpoint_params)
        logger.info(f"Successfully initiated endpoint creation name={safe_endpoint_name}")
       
        # Extract configuration details for response
        instance_type = config_data.get('instance_type', 'ml.m5.large')
        instance_count = config_data.get('instance_count', 1)
       
        # If not in top level, extract from endpoint_config
        if 'production_variants' in endpoint_config and endpoint_config['production_variants']:
            first_variant = endpoint_config['production_variants'][0]
            instance_type = first_variant.get('InstanceType', instance_type)
            instance_count = first_variant.get('InitialInstanceCount', instance_count)
       
        result = {
            'profile': profile,
            'endpoint_name': new_endpoint_name,
            'endpoint_config_name': new_endpoint_config_name,
            'model_name': new_model_name,
            'instance_type': instance_type,
            'instance_count': instance_count,
            'status': 'Creating',
            'creation_time': datetime.now().isoformat(),
            'execution_id': execution_id,
            'endpoint_arn': create_response['EndpointArn'],
            'recreation_details': {
                'original_config_loaded': True,
                'model_approach': 'ModelPackage' if model_package_name else 'DirectContainer',
                'model_package_used': model_package_name if model_package_name else None,
                'model_created': True,
                'endpoint_config_created': True,
                'endpoint_creation_initiated': True,
                'config_source': f"s3://{bucket}/{key}",
                'replica_of_deleted_endpoint': True
            }
        }
       
        logger.info(f"Successfully completed endpoint creation process profile={safe_profile}")
        logger.info("New endpoint is being created using configured approach")
       
        return result
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Failed to create endpoint profile={safe_profile} error={error_msg}")
       
        # Enhanced error information
        error_details = {
            'error': error_msg,
            'error_type': type(e).__name__,
            'profile': event.get('profile'),
            'execution_id': execution_id,
            'step_failed': 'endpoint_creation',
            'config_source': event.get('s3_config_path', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
       
        # Add more specific error context
        if 'ValidationException' in str(e):
            if 'ModelPackageName' in str(e):
                error_details['likely_cause'] = 'Model Package configuration conflict'
            else:
                error_details['likely_cause'] = 'AWS resource validation failed'
        elif 'ResourceNotFoundException' in str(e):
            error_details['likely_cause'] = 'Referenced AWS resource not found'
        elif 'NoSuchKey' in str(e):
            error_details['likely_cause'] = 'S3 configuration file not found'
       
        raise Exception(json.dumps(error_details))

def check_endpoint_status(event: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Check the status of a SageMaker endpoint
    """
   
    try:
        profile = event['profile']
        endpoint_name = event['endpoint_name']
       
        # SECURITY FIX: Sanitize values for logging
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
       
        logger.info(f"Checking endpoint status profile={safe_profile} endpoint={safe_endpoint_name}")
       
        # Get endpoint status
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
       
        status = response['EndpointStatus']
       
        result = {
            'profile': profile,
            'endpoint_name': endpoint_name,
            'status': status,
            'creation_time': response.get('CreationTime', '').isoformat() if response.get('CreationTime') else None,
            'last_modified_time': response.get('LastModifiedTime', '').isoformat() if response.get('LastModifiedTime') else None,
            'execution_id': execution_id,
            'check_timestamp': datetime.now().isoformat()
        }
       
        if status == 'Failed':
            failure_reason = sanitize_for_logging(response.get('FailureReason', 'Unknown failure'))
            result['failure_reason'] = failure_reason
            logger.error(f"Endpoint failed endpoint={safe_endpoint_name} reason={failure_reason}")
        elif status == 'InService':
            logger.info(f"Endpoint is InService and ready endpoint={safe_endpoint_name}")
        else:
            logger.info(f"Endpoint status endpoint={safe_endpoint_name} status={status}")
       
        return result
       
    except sagemaker_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException' and 'does not exist' in str(e):
            # Endpoint doesn't exist
            safe_endpoint_name = sanitize_for_logging(event.get('endpoint_name', 'unknown'), 100)
            logger.error(f"Endpoint does not exist endpoint={safe_endpoint_name}")
            return {
                'profile': event.get('profile'),
                'endpoint_name': event.get('endpoint_name'),
                'status': 'NotFound',
                'error': 'Endpoint does not exist',
                'execution_id': execution_id,
                'check_timestamp': datetime.now().isoformat()
            }
        else:
            raise
   
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Failed to check endpoint status profile={safe_profile} error={error_msg}")
        return {
            'profile': event.get('profile'),
            'endpoint_name': event.get('endpoint_name'),
            'status': 'Error',
            'error': error_msg,
            'execution_id': execution_id,
            'check_timestamp': datetime.now().isoformat()
        }
