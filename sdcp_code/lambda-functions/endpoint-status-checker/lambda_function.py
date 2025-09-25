"""
Endpoint Status Checker Lambda Function
Waits for all parallel endpoints to reach InService status
SECURITY FIX: Log injection vulnerability (CWE-117, CWE-93) patched
"""

import json
import boto3
import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, List

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
   
    safe_fields = ['operation', 'max_wait_time']
   
    for field in safe_fields:
        if field in event:
            sanitized_event[field] = sanitize_for_logging(event[field])
   
    # Handle endpoint creation results safely
    if 'endpoint_creation_results' in event and isinstance(event['endpoint_creation_results'], list):
        sanitized_event['endpoint_creation_results_count'] = len(event['endpoint_creation_results'])
        # Log sample of first few results (safely)
        sample_results = []
        for i, result in enumerate(event['endpoint_creation_results'][:3]):  # Max 3 samples
            if isinstance(result, dict):
                sample_result = {}
                if 'status' in result:
                    sample_result['status'] = sanitize_for_logging(result['status'], 20)
                if 'profile' in result:
                    sample_result['profile'] = sanitize_for_logging(result['profile'], 20)
                sample_results.append(sample_result)
        sanitized_event['endpoint_creation_results_sample'] = sample_results
   
    sanitized_event['_metadata'] = {
        'event_keys_count': len(event.keys()),
        'has_endpoint_results': 'endpoint_creation_results' in event,
        'has_execution_id': 'execution_id' in event
    }
   
    return sanitized_event

def lambda_handler(event, context):
    """
    Wait for all parallel endpoints to reach InService status
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting endpoint status checking [execution_id={sanitize_for_logging(execution_id)}]")
       
        # SECURITY FIX: Sanitize event data before logging
        sanitized_event = sanitize_event_for_logging(event)
        logger.info(f"Event metadata: {json.dumps(sanitized_event)}")
       
        # Extract parameters
        operation = event.get('operation', 'wait_for_all_endpoints')
        endpoint_creation_results = event.get('endpoint_creation_results', [])
        max_wait_time = event.get('max_wait_time', 900)  # 15 minutes default
        pipeline_execution_id = event.get('execution_id', execution_id)
       
        # SECURITY FIX: Sanitize values for logging
        safe_operation = sanitize_for_logging(operation)
        safe_max_wait_time = sanitize_for_logging(max_wait_time)
        safe_pipeline_execution_id = sanitize_for_logging(pipeline_execution_id)
        results_count = len(endpoint_creation_results) if isinstance(endpoint_creation_results, list) else 0
       
        logger.info(f"Operation: {safe_operation}, Results count: {results_count}, Max wait: {safe_max_wait_time}s, Pipeline ID: {safe_pipeline_execution_id}")
       
        if operation == 'wait_for_all_endpoints':
            result = wait_for_all_endpoints_ready(endpoint_creation_results, max_wait_time, pipeline_execution_id)
        elif operation == 'check_endpoint_status':
            endpoint_name = event.get('endpoint_name')
            result = check_single_endpoint_status(endpoint_name, pipeline_execution_id)
        else:
            raise ValueError(f"Unknown operation: {safe_operation}")
       
        return {
            'statusCode': 200,
            'body': result
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Endpoint status checking failed [execution_id={sanitize_for_logging(execution_id)}] error={error_msg}")
        return {
            'statusCode': 500,
            'body': {
                'operation': event.get('operation', 'unknown'),
                'status': 'failed',
                'error': error_msg,
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        }

def wait_for_all_endpoints_ready(endpoint_creation_results: List[Dict[str, Any]], max_wait_time: int, execution_id: str) -> Dict[str, Any]:
    """
    Wait for all endpoints from parallel creation to reach InService status
    """
   
    try:
        wait_minutes = max_wait_time / 60
        safe_wait_minutes = sanitize_for_logging(wait_minutes)
        logger.info(f"Waiting for endpoints to be ready max_wait_minutes={safe_wait_minutes:.1f}")
       
        # Extract endpoint details from Step Functions parallel execution results
        endpoints_to_check = {}
       
        for result in endpoint_creation_results:
            if isinstance(result, dict):
                if result.get('status') == 'success' and 'endpoint_details' in result:
                    endpoint_info = result['endpoint_details']
                    profile = endpoint_info.get('profile')
                    endpoint_name = endpoint_info.get('endpoint_name')
                   
                    if profile and endpoint_name:
                        endpoints_to_check[profile] = {
                            'endpoint_name': endpoint_name,
                            'status': 'checking',
                            'profile': profile
                        }
                       
        if not endpoints_to_check:
            logger.warning("No valid endpoints found to check")
            return {
                'operation': 'wait_for_all_endpoints',
                'status': 'no_endpoints_to_check',
                'ready_endpoints': {},
                'failed_endpoints': {},
                'execution_id': execution_id
            }
       
        logger.info(f"Checking status for endpoints_count={len(endpoints_to_check)} profiles={list(endpoints_to_check.keys())[:5]}")  # Max 5 profile names in log
       
        # Wait for all endpoints to be ready
        start_time = time.time()
        ready_endpoints = {}
        failed_endpoints = {}
       
        while time.time() - start_time < max_wait_time:
            all_ready = True
           
            for profile, endpoint_info in endpoints_to_check.items():
                if profile in ready_endpoints or profile in failed_endpoints:
                    continue  # Already processed
               
                endpoint_name = endpoint_info['endpoint_name']
               
                # SECURITY FIX: Sanitize values for logging
                safe_profile = sanitize_for_logging(profile, 50)
                safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
               
                try:
                    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                    status = response['EndpointStatus']
                   
                    if status == 'InService':
                        ready_endpoints[profile] = {
                            'endpoint_name': endpoint_name,
                            'status': 'ready',
                            'ready_time': datetime.now().isoformat(),
                            'profile': profile
                        }
                        logger.info(f"Endpoint ready profile={safe_profile} endpoint={safe_endpoint_name}")
                       
                    elif status == 'Failed':
                        failure_reason = sanitize_for_logging(response.get('FailureReason', 'Unknown failure'))
                        failed_endpoints[profile] = {
                            'endpoint_name': endpoint_name,
                            'status': 'failed',
                            'error': failure_reason,
                            'profile': profile
                        }
                        logger.error(f"Endpoint failed profile={safe_profile} endpoint={safe_endpoint_name} reason={failure_reason}")
                       
                    elif status in ['Creating', 'Updating']:
                        all_ready = False
                        logger.info(f"Endpoint still {status} profile={safe_profile} endpoint={safe_endpoint_name}")
                       
                    else:
                        all_ready = False
                        logger.warning(f"Endpoint in unexpected status profile={safe_profile} endpoint={safe_endpoint_name} status={status}")
               
                except Exception as e:
                    error_msg = sanitize_for_logging(str(e))
                    logger.warning(f"Could not check status profile={safe_profile} endpoint={safe_endpoint_name} error={error_msg}")
                    all_ready = False
           
            # Check if we're done
            total_processed = len(ready_endpoints) + len(failed_endpoints)
            if total_processed == len(endpoints_to_check):
                break
           
            if all_ready:
                break
           
            # Wait before next check
            time.sleep(30)
       
        # Final status
        total_ready = len(ready_endpoints)
        total_failed = len(failed_endpoints)
        total_requested = len(endpoints_to_check)
       
        elapsed_time = time.time() - start_time
       
        logger.info(f"Endpoint status check completed elapsed_seconds={elapsed_time:.1f}")
        logger.info(f"Summary ready={total_ready} failed={total_failed} total={total_requested}")
       
        # Determine overall status
        if total_ready == total_requested:
            overall_status = 'all_ready'
        elif total_ready > 0:
            overall_status = 'partial_ready'
        else:
            overall_status = 'none_ready'
       
        return {
            'operation': 'wait_for_all_endpoints',
            'status': overall_status,
            'execution_id': execution_id,
            'summary': {
                'total_requested': total_requested,
                'ready_count': total_ready,
                'failed_count': total_failed,
                'elapsed_time_seconds': elapsed_time
            },
            'ready_endpoints': ready_endpoints,
            'failed_endpoints': failed_endpoints,
            'timestamp': datetime.now().isoformat()
        }
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        logger.error(f"Failed to wait for endpoints error={error_msg}")
        return {
            'operation': 'wait_for_all_endpoints',
            'status': 'error',
            'error': error_msg,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }

def check_single_endpoint_status(endpoint_name: str, execution_id: str) -> Dict[str, Any]:
    """
    Check status of a single endpoint
    """
   
    try:
        if not endpoint_name:
            raise ValueError("endpoint_name is required")
       
        # SECURITY FIX: Sanitize endpoint name for logging
        safe_endpoint_name = sanitize_for_logging(endpoint_name, 100)
        logger.info(f"Checking status for endpoint name={safe_endpoint_name}")
       
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
       
        status = response['EndpointStatus']
       
        result = {
            'operation': 'check_endpoint_status',
            'endpoint_name': endpoint_name,
            'status': status,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
       
        # Add additional info based on status
        if status == 'Failed':
            failure_reason = sanitize_for_logging(response.get('FailureReason', 'Unknown failure'))
            result['failure_reason'] = failure_reason
            logger.error(f"Endpoint failed endpoint={safe_endpoint_name} reason={failure_reason}")
        elif status == 'InService':
            result['creation_time'] = response.get('CreationTime', '').isoformat() if response.get('CreationTime') else None
            result['last_modified_time'] = response.get('LastModifiedTime', '').isoformat() if response.get('LastModifiedTime') else None
            logger.info(f"Endpoint is InService endpoint={safe_endpoint_name}")
        else:
            logger.info(f"Endpoint status endpoint={safe_endpoint_name} status={status}")
       
        return result
       
    except Exception as e:
        error_msg = sanitize_for_logging(str(e))
        safe_endpoint_name = sanitize_for_logging(endpoint_name or 'unknown', 100)
        logger.error(f"Failed to check endpoint status endpoint={safe_endpoint_name} error={error_msg}")
        return {
            'operation': 'check_endpoint_status',
            'endpoint_name': endpoint_name,
            'status': 'error',
            'error': error_msg,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        }
