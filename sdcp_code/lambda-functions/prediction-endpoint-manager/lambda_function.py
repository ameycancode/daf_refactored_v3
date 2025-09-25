"""
Prediction Endpoint Manager Lambda Function - Environment-Aware Version
This version uses environment variables for all configuration values
instead of hardcoded dev environment values.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

def sanitize_for_logging(value: str, max_length: int = 50) -> str:
    """Sanitize string values for safe logging"""
    if not isinstance(value, str):
        value = str(value)
    # Remove potentially sensitive characters and limit length
    sanitized = ''.join(c for c in value if c.isalnum() or c in '-_.')[:max_length]
    return sanitized if sanitized else 'unknown'

def sanitize_event_for_logging(event: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize event data for logging"""
    sanitized_event = {}
   
    # Safe keys to include
    safe_keys = ['operation', 'execution_id']
    for key in safe_keys:
        if key in event:
            sanitized_event[key] = sanitize_for_logging(str(event[key]), 100)
   
    # Include metadata about profiles
    if 'profiles' in event and isinstance(event['profiles'], list):
        sanitized_event['profiles_count'] = len(event['profiles'])
        sanitized_event['profiles'] = [sanitize_for_logging(p, 20) for p in event['profiles'][:5]]  # Max 5 profiles
   
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profiles': 'profiles' in event
    }
   
    return sanitized_event

def lambda_handler(event, context):
    """
    Main Lambda handler for prediction endpoint management
   
    Expected event structure:
    {
        "operation": "recreate_all_endpoints",
        "profiles": ["RNN", "RN", "M"],
        "execution_id": "12345-abcde"
    }
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting prediction endpoint management [execution_id={sanitize_for_logging(execution_id)}]")
       
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Environment-aware configuration - no hardcoded values
        data_bucket = os.environ.get('DATA_BUCKET')
        if not data_bucket:
            raise ValueError("DATA_BUCKET environment variable is required but not set")
       
        config = {
            'data_bucket': data_bucket,
            'config_prefix': 'endpoint-configurations/',
            'max_wait_time': 600,  # 10 minutes
            'wait_interval': 30   # 30 seconds
        }
       
        # Extract operation details
        operation = event.get('operation', 'recreate_all_endpoints')
        profiles = event.get('profiles', [])
        step_functions_execution_id = event.get('execution_id', execution_id)
       
        # Validate inputs
        if not profiles:
            raise ValueError("Profiles list is required for endpoint operations")
       
        # SECURITY FIX: Sanitize values before logging
        safe_operation = sanitize_for_logging(operation)
        safe_data_bucket = sanitize_for_logging(data_bucket, 100)
       
        logger.info(f"Processing operation={safe_operation} data_bucket={safe_data_bucket}")
        logger.info(f"Profiles requested: {len(profiles)}")
       
        # Handle different operations
        if operation == 'recreate_all_endpoints':
            result = recreate_all_endpoints(profiles, config, execution_id)
        elif operation == 'check_endpoints_status':
            result = check_endpoints_status(profiles, execution_id)
        elif operation == 'cleanup_endpoints':
            result = cleanup_endpoints(profiles, execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
       
        return {
            'statusCode': 200,
            'body': result
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e), 200)
        logger.error(f"Prediction endpoint management failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profiles': event.get('profiles', []),
                'message': 'Prediction endpoint management failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def recreate_all_endpoints(profiles: List[str], config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoints for all specified profiles from saved S3 configurations
    """
   
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Recreating endpoints profiles_count={profiles_count}")
       
        endpoint_details = {}
        successful_creations = 0
       
        # Create endpoints for each profile
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile name for logging
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Recreating endpoint profile={safe_profile}")
               
                endpoint_result = recreate_endpoint_from_config(profile, config, execution_id)
                endpoint_details[profile] = endpoint_result
               
                if endpoint_result['status'] == 'success':
                    successful_creations += 1
                   
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                logger.error(f"Failed to recreate endpoint profile={safe_profile} error={error_msg}")
                endpoint_details[profile] = {
                    'status': 'failed',
                    'error': error_msg,
                    'profile': profile
                }
       
        # Wait for all endpoints to be ready
        if successful_creations > 0:
            logger.info(f"Waiting for endpoints to be ready successful_count={successful_creations}")
            wait_results = wait_for_endpoints_ready(endpoint_details, config['max_wait_time'])
        else:
            wait_results = {}
       
        return {
            'operation': 'recreate_all_endpoints',
            'execution_id': execution_id,
            'profiles_requested': profiles,
            'successful_creations': successful_creations,
            'endpoint_details': endpoint_details,
            'wait_results': wait_results,
            'timestamp': datetime.now().isoformat()
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to recreate endpoints error={error_msg}")
        return {
            'operation': 'recreate_all_endpoints',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def recreate_endpoint_from_config(profile: str, config: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Recreate endpoint from saved S3 configuration
    """
   
    try:
        # Find latest configuration file for this profile
        config_data = find_latest_endpoint_config(profile, config['data_bucket'], config['config_prefix'])
       
        if not config_data:
            safe_profile = sanitize_for_logging(profile, 50)
            logger.error(f"No configuration found for profile profile={safe_profile}")
            return {
                'status': 'failed',
                'error': 'No configuration found',
                'profile': profile
            }
       
        logger.info(f"Found configuration for {profile}")
       
        # Generate unique names for new resources
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_suffix = f"{timestamp}-{execution_id[:8]}"
       
        model_name = f"energy-forecasting-{profile.lower()}-model-{unique_suffix}"
        endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-{unique_suffix}"
        endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{unique_suffix}"
       
        # Extract configuration details
        model_config = config_data.get('model_config', {})
        endpoint_config = config_data.get('endpoint_config', {})
       
        if not model_config or not endpoint_config:
            return {
                'status': 'failed',
                'error': 'Invalid configuration format',
                'profile': profile
            }
       
        # Get instance type from original config or use default
        original_variants = endpoint_config.get('ProductionVariants', [])
        instance_type = 'ml.t2.medium'  # Default
        if original_variants:
            instance_type = original_variants[0].get('InstanceType', 'ml.t2.medium')
       
        # Get SageMaker execution role from environment
        sagemaker_role = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
        if not sagemaker_role:
            raise ValueError("SAGEMAKER_EXECUTION_ROLE environment variable is required")
       
        # Step 1: Create Model using saved configuration
        containers = model_config.get('Containers', [])
        if not containers:
            return {
                'status': 'failed',
                'error': 'No container configuration found',
                'profile': profile
            }
       
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': containers[0].get('Image'),
                'ModelDataUrl': containers[0].get('ModelDataUrl'),
                'Environment': containers[0].get('Environment', {})
            },
            ExecutionRoleArn=sagemaker_role
        )
       
        logger.info(f"Created model for prediction: {model_name}")
       
        # Step 2: Create Endpoint Configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
       
        # SECURITY FIX: Sanitize config name for logging
        safe_config_name = sanitize_for_logging(endpoint_config_name, 100)
        logger.info(f"Created endpoint configuration for prediction name={safe_config_name}")
       
        # Step 3: Create Endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
       
        # SECURITY FIX: Sanitize endpoint name for logging
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Created endpoint for prediction name={safe_endpoint_name}")
       
        return {
            'status': 'success',
            'profile': profile,
            'endpoint_name': endpoint_name,
            'endpoint_config_name': endpoint_config_name,
            'model_name': model_name,
            'instance_type': instance_type,
            'source_config': config_data.get('creation_timestamp', 'unknown'),
            'created_for': 'prediction'
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Failed to recreate endpoint profile={safe_profile} error={error_msg}")
        return {
            'status': 'failed',
            'error': error_msg,
            'profile': profile
        }

def find_latest_endpoint_config(profile: str, bucket: str, config_prefix: str) -> Optional[Dict[str, Any]]:
    """
    Find the latest endpoint configuration for a profile
    """
   
    try:
        # List all configuration files for this profile
        profile_prefix = f"{config_prefix}{profile}/"
       
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=profile_prefix
        )
       
        if 'Contents' not in response:
            safe_profile = sanitize_for_logging(profile, 50)
            logger.warning(f"No configuration files found profile={safe_profile}")
            return None
       
        # Find the most recent configuration file
        config_files = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                config_files.append({
                    'key': obj['Key'],
                    'last_modified': obj['LastModified']
                })
       
        if not config_files:
            safe_profile = sanitize_for_logging(profile, 50)
            logger.warning(f"No JSON configuration files found profile={safe_profile}")
            return None
       
        # Get the most recent file
        latest_config = max(config_files, key=lambda x: x['last_modified'])
       
        # Load configuration data
        response = s3_client.get_object(Bucket=bucket, Key=latest_config['key'])
        config_data = json.loads(response['Body'].read())
       
        safe_profile = sanitize_for_logging(profile, 50)
        safe_key = sanitize_for_logging(latest_config['key'], 150)
        logger.info(f"Loaded latest config profile={safe_profile} key={safe_key}")
       
        return config_data
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(profile, 50)
        logger.error(f"Error finding configuration profile={safe_profile} error={error_msg}")
        return None

def wait_for_endpoints_ready(endpoint_details: Dict[str, Dict[str, Any]], max_wait_time: int) -> Dict[str, Any]:
    """
    Wait for all endpoints to be ready
    """
   
    wait_results = {}
    start_time = time.time()
   
    # Get list of endpoints to wait for
    endpoints_to_wait = {}
    for profile, details in endpoint_details.items():
        if details.get('status') == 'success':
            endpoints_to_wait[profile] = details['endpoint_name']
   
    if not endpoints_to_wait:
        return {'message': 'No endpoints to wait for'}
   
    logger.info(f"Waiting for {len(endpoints_to_wait)} endpoints to be ready")
   
    # Wait for endpoints
    while endpoints_to_wait and (time.time() - start_time) < max_wait_time:
        completed_profiles = []
       
        for profile, endpoint_name in endpoints_to_wait.items():
            try:
                response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
               
                safe_profile = sanitize_for_logging(profile, 50)
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
               
                if status == 'InService':
                    wait_results[profile] = {
                        'status': 'ready',
                        'endpoint_name': endpoint_name,
                        'wait_time_seconds': int(time.time() - start_time)
                    }
                    completed_profiles.append(profile)
                    logger.info(f"Endpoint ready profile={safe_profile} endpoint={safe_endpoint_name}")
                   
                elif status in ['Failed', 'RollingBack']:
                    wait_results[profile] = {
                        'status': 'failed',
                        'endpoint_name': endpoint_name,
                        'endpoint_status': status,
                        'wait_time_seconds': int(time.time() - start_time)
                    }
                    completed_profiles.append(profile)
                    logger.error(f"Endpoint failed profile={safe_profile} status={status}")
                   
                else:
                    # Still in progress
                    logger.info(f"Endpoint still creating profile={safe_profile} status={status}")
                   
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                wait_results[profile] = {
                    'status': 'error',
                    'endpoint_name': endpoint_name,
                    'error': error_msg,
                    'wait_time_seconds': int(time.time() - start_time)
                }
                completed_profiles.append(profile)
                logger.error(f"Error checking endpoint profile={safe_profile} error={error_msg}")
       
        # Remove completed endpoints from waiting list
        for profile in completed_profiles:
            endpoints_to_wait.pop(profile, None)
       
        # Sleep before next check if there are still endpoints to wait for
        if endpoints_to_wait:
            time.sleep(30)
   
    # Handle any remaining endpoints that didn't complete
    for profile, endpoint_name in endpoints_to_wait.items():
        wait_results[profile] = {
            'status': 'timeout',
            'endpoint_name': endpoint_name,
            'wait_time_seconds': int(time.time() - start_time)
        }
        safe_profile = sanitize_for_logging(profile, 50)
        logger.warning(f"Endpoint creation timeout profile={safe_profile}")
   
    return wait_results

def check_endpoints_status(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Check the status of endpoints for specified profiles
    """
   
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Checking endpoint status profiles_count={profiles_count}")
       
        endpoint_status = {}
       
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile name for logging
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Checking endpoint status profile={safe_profile}")
               
                # List endpoints that match this profile
                endpoints = sagemaker_client.list_endpoints(
                    StatusEquals='InService',
                    NameContains=f"energy-forecasting-{profile.lower()}"
                )
               
                if endpoints['Endpoints']:
                    # Get the most recent endpoint
                    latest_endpoint = sorted(endpoints['Endpoints'],
                                           key=lambda x: x['CreationTime'], reverse=True)[0]
                   
                    safe_endpoint_name = sanitize_for_logging(latest_endpoint['EndpointName'], 100)
                    endpoint_status[profile] = {
                        'status': 'active',
                        'endpoint_name': latest_endpoint['EndpointName'],
                        'endpoint_status': latest_endpoint['EndpointStatus'],
                        'creation_time': latest_endpoint['CreationTime'].isoformat(),
                        'instance_type': 'unknown'  # Would need to describe endpoint config to get this
                    }
                    logger.info(f"Found active endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
                else:
                    endpoint_status[profile] = {
                        'status': 'not_found',
                        'message': f'No active endpoints found for {profile}'
                    }
                    logger.info(f"No active endpoints found profile={safe_profile}")
                   
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                endpoint_status[profile] = {
                    'status': 'error',
                    'error': error_msg
                }
                logger.error(f"Error checking endpoint status profile={safe_profile} error={error_msg}")
       
        return {
            'operation': 'check_endpoints_status',
            'execution_id': execution_id,
            'endpoint_status': endpoint_status,
            'timestamp': datetime.now().isoformat()
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to check endpoint status error={error_msg}")
        return {
            'operation': 'check_endpoints_status',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }

def cleanup_endpoints(profiles: List[str], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup endpoints after predictions are complete
    """
   
    try:
        profiles_count = len(profiles) if isinstance(profiles, list) else 0
        logger.info(f"Cleaning up endpoints profiles_count={profiles_count}")
       
        cleanup_results = {}
        total_cost_saved = 0.0
       
        for profile in profiles:
            try:
                # SECURITY FIX: Sanitize profile name for logging
                safe_profile = sanitize_for_logging(profile, 50)
                logger.info(f"Cleaning up endpoints profile={safe_profile}")
               
                # Find active endpoints for this profile
                endpoints = sagemaker_client.list_endpoints(
                    StatusEquals='InService',
                    NameContains=f"energy-forecasting-{profile.lower()}"
                )
               
                deleted_endpoints = []
                profile_cost_saved = 0.0
               
                for endpoint in endpoints['Endpoints']:
                    endpoint_name = endpoint['EndpointName']
                   
                    try:
                        # Get endpoint details for cost calculation
                        endpoint_details = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                       
                        # Delete endpoint
                        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                        deleted_endpoints.append(endpoint_name)
                       
                        # Estimate cost saved (rough calculation)
                        # This is a simplified calculation - actual costs vary
                        hours_saved = 24  # Assume we save 24 hours of running time
                        instance_cost_per_hour = 0.05  # Rough estimate for ml.t2.medium
                        cost_saved = hours_saved * instance_cost_per_hour
                        profile_cost_saved += cost_saved
                       
                        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                        logger.info(f"Deleted endpoint profile={safe_profile} endpoint={safe_endpoint_name}")
                       
                    except Exception as e:
                        error_msg = sanitize_for_logging(str(e))
                        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
                        logger.error(f"Failed to delete endpoint profile={safe_profile} endpoint={safe_endpoint_name} error={error_msg}")
               
                cleanup_results[profile] = {
                    'status': 'success' if deleted_endpoints else 'no_endpoints_found',
                    'deleted_endpoints': deleted_endpoints,
                    'endpoints_deleted_count': len(deleted_endpoints),
                    'estimated_cost_saved_usd': round(profile_cost_saved, 2)
                }
               
                total_cost_saved += profile_cost_saved
               
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                safe_profile = sanitize_for_logging(profile, 50)
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': error_msg
                }
                logger.error(f"Failed to cleanup endpoints profile={safe_profile} error={error_msg}")
       
        return {
            'operation': 'cleanup_endpoints',
            'execution_id': execution_id,
            'cleanup_results': cleanup_results,
            'total_cost_saved_usd': round(total_cost_saved, 2),
            'summary': {
                'profiles_processed': len(profiles),
                'total_endpoints_deleted': sum(r.get('endpoints_deleted_count', 0) for r in cleanup_results.values()),
                'successful_cleanups': len([r for r in cleanup_results.values() if r.get('status') == 'success'])
            },
            'timestamp': datetime.now().isoformat()
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to cleanup endpoints error={error_msg}")
        return {
            'operation': 'cleanup_endpoints',
            'execution_id': execution_id,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
