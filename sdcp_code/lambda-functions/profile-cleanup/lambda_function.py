"""
lambda-functions/profile-cleanup/lambda_function.py
Cleans up resources for individual profiles
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import re
from datetime import datetime
from typing import Dict, Any

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')

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
    
    safe_fields = ['operation', 'profile', 'delete_model']
    
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
    
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_profile': 'profile' in event,
        'has_endpoint_details': any(k in event for k in ['endpoint_name', 'endpoint_config_name', 'model_name'])
    }
    
    return sanitized_event

def lambda_handler(event, context):
    """
    Clean up SageMaker resources for individual profiles
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting profile cleanup [execution_id={sanitize_for_logging(execution_id)}]")
        
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Extract cleanup details
        operation = event.get('operation', 'cleanup_profile_resources')
        profile = event.get('profile')
       
        if not profile:
            raise ValueError("Profile is required for cleanup operations")
        
        # SECURITY FIX: Sanitize profile name before logging
        safe_profile = sanitize_for_logging(profile, 50)
        logger.info(f"Processing operation={sanitize_for_logging(operation)} profile={safe_profile}")
       
        # Handle cleanup operation
        if operation == 'cleanup_profile_resources':
            result = cleanup_profile_resources(event, execution_id)
        else:
            raise ValueError(f"Unknown operation: {sanitize_for_logging(operation)}")
       
        return {
            'statusCode': 200,
            'body': result
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Profile cleanup failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'error': error_msg,
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'message': 'Profile cleanup failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def cleanup_profile_resources(event: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Clean up all SageMaker resources for a specific profile
    """
   
    try:
        profile = event['profile']
        endpoint_name = event.get('endpoint_name')
        endpoint_config_name = event.get('endpoint_config_name')
        model_name = event.get('model_name')
        
        # SECURITY FIX: Sanitize values for logging
        safe_profile = sanitize_for_logging(profile, 50)
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100) if endpoint_name else None
        safe_endpoint_config_name = sanitize_for_logging(endpoint_config_name, 100) if endpoint_config_name else None
        safe_model_name = sanitize_for_logging(model_name, 100) if model_name else None
       
        logger.info(f"Cleaning up resources profile={safe_profile}")
       
        cleanup_actions = []
        cleanup_start_time = datetime.now().isoformat()
       
        # 1. Delete endpoint
        if endpoint_name:
            try:
                logger.info(f"Deleting endpoint name={safe_endpoint_name}")
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                cleanup_actions.append(f"endpoint:{endpoint_name}")
                logger.info(f"Successfully deleted endpoint name={safe_endpoint_name}")
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                if 'does not exist' in str(e).lower():
                    logger.info(f"Endpoint already deleted or doesn't exist name={safe_endpoint_name}")
                else:
                    logger.warning(f"Could not delete endpoint name={safe_endpoint_name} error={error_msg}")
       
        # 2. Delete endpoint configuration
        if endpoint_config_name:
            try:
                logger.info(f"Deleting endpoint configuration name={safe_endpoint_config_name}")
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                cleanup_actions.append(f"endpoint_config:{endpoint_config_name}")
                logger.info(f"Successfully deleted endpoint config name={safe_endpoint_config_name}")
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                if 'does not exist' in str(e).lower():
                    logger.info(f"Endpoint config already deleted or doesn't exist name={safe_endpoint_config_name}")
                else:
                    logger.warning(f"Could not delete endpoint config name={safe_endpoint_config_name} error={error_msg}")
       
        # 3. Delete model (optional - models can be reused)
        if model_name and event.get('delete_model', False):
            try:
                logger.info(f"Deleting model name={safe_model_name}")
                sagemaker_client.delete_model(ModelName=model_name)
                cleanup_actions.append(f"model:{model_name}")
                logger.info(f"Successfully deleted model name={safe_model_name}")
            except Exception as e:
                error_msg = sanitize_for_logging(str(e))
                if 'does not exist' in str(e).lower():
                    logger.info(f"Model already deleted or doesn't exist name={safe_model_name}")
                else:
                    logger.warning(f"Could not delete model name={safe_model_name} error={error_msg}")
       
        # Calculate cost savings (approximate)
        cost_per_hour = 0.115  # Approximate cost for ml.m5.large
        estimated_hourly_savings = cost_per_hour
       
        result = {
            'profile': profile,
            'cleanup_status': 'success',
            'cleanup_start_time': cleanup_start_time,
            'cleanup_end_time': datetime.now().isoformat(),
            'resources_cleaned': cleanup_actions,
            'resources_cleaned_count': len(cleanup_actions),
            'execution_id': execution_id,
            'cost_impact': f"Estimated savings: ${estimated_hourly_savings:.3f}/hour",
            'message': f'Successfully cleaned up {len(cleanup_actions)} resources for {profile}'
        }
       
        logger.info(f"Successfully cleaned up all resources profile={safe_profile} resources_count={len(cleanup_actions)}")
       
        return result
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_profile = sanitize_for_logging(event.get('profile', 'unknown'), 50)
        logger.error(f"Failed to cleanup resources profile={safe_profile} error={error_msg}")
       
        return {
            'profile': event.get('profile'),
            'cleanup_status': 'failed',
            'cleanup_start_time': cleanup_start_time if 'cleanup_start_time' in locals() else datetime.now().isoformat(),
            'cleanup_end_time': datetime.now().isoformat(),
            'error': error_msg,
            'execution_id': execution_id,
            'message': f'Cleanup failed for {event.get("profile")}: {error_msg}'
        }
