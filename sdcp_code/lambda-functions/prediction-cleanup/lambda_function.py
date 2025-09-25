"""
Enhanced Prediction Cleanup Lambda Function
Cleans up prediction endpoints after predictions are completed
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    """
    Main handler for prediction cleanup
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting prediction cleanup [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Extract endpoint details from various possible event structures
        endpoint_details = extract_endpoint_details(event)
        operation = event.get('operation', 'cleanup_endpoints')
        
        if not endpoint_details:
            logger.warning("No endpoint details provided for cleanup")
            return {
                'statusCode': 200,
                'body': {
                    'message': 'No endpoints to cleanup',
                    'execution_id': execution_id,
                    'timestamp': datetime.now().isoformat(),
                    'operation': operation
                }
            }
        
        # Perform cleanup based on operation
        if operation == 'cleanup_endpoints':
            cleanup_results = cleanup_prediction_endpoints(endpoint_details, execution_id)
        elif operation == 'emergency_cleanup':
            cleanup_results = emergency_cleanup_all_prediction_endpoints(execution_id)
        else:
            raise ValueError(f"Unknown cleanup operation: {operation}")
        
        return {
            'statusCode': 200,
            'body': cleanup_results
        }
        
    except Exception as e:
        logger.error(f"Prediction cleanup failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Prediction cleanup failed',
                'timestamp': datetime.now().isoformat()
            }
        }

def extract_endpoint_details(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract endpoint details from various event structures
    """
    
    try:
        # Try different possible locations for endpoint details
        endpoint_details = {}
        
        # Direct from event
        if 'endpoint_details' in event:
            endpoint_details = event['endpoint_details']
        
        # From prediction results
        elif 'prediction_results' in event:
            pred_results = event['prediction_results']
            if isinstance(pred_results, dict) and 'endpoint_details' in pred_results:
                endpoint_details = pred_results['endpoint_details']
            elif isinstance(pred_results, dict) and 'body' in pred_results:
                body = pred_results['body']
                if isinstance(body, dict) and 'endpoint_details' in body:
                    endpoint_details = body['endpoint_details']
        
        # From Step Functions input
        elif 'prediction_input' in event and 'endpoint_details' in event['prediction_input']:
            endpoint_details = event['prediction_input']['endpoint_details']
        
        # From nested body structure
        elif 'body' in event:
            body = event['body']
            if isinstance(body, dict) and 'endpoint_details' in body:
                endpoint_details = body['endpoint_details']
        
        # If still empty, try to extract from any nested dictionary
        if not endpoint_details:
            endpoint_details = find_endpoint_details_recursive(event)
        
        logger.info(f"Extracted endpoint details for {len(endpoint_details)} profiles")
        return endpoint_details
        
    except Exception as e:
        logger.warning(f"Could not extract endpoint details: {str(e)}")
        return {}

def find_endpoint_details_recursive(obj: Any) -> Dict[str, Any]:
    """
    Recursively search for endpoint details in nested structures
    """
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'endpoint_details' and isinstance(value, dict):
                return value
            elif isinstance(value, dict):
                result = find_endpoint_details_recursive(value)
                if result:
                    return result
    
    return {}

def cleanup_prediction_endpoints(endpoint_details: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """
    Cleanup all prediction endpoints and related resources
    """
    
    try:
        logger.info(f"Cleaning up prediction endpoints for {len(endpoint_details)} profiles")
        
        cleanup_results = {}
        successful_cleanups = 0
        total_resources_cleaned = 0
        
        for profile, details in endpoint_details.items():
            try:
                logger.info(f"Cleaning up resources for profile: {profile}")
                
                profile_cleanup = cleanup_profile_resources(profile, details)
                cleanup_results[profile] = profile_cleanup
                
                if profile_cleanup['status'] == 'success':
                    successful_cleanups += 1
                    total_resources_cleaned += len(profile_cleanup.get('resources_cleaned', []))
                    
            except Exception as e:
                logger.error(f"Failed to cleanup resources for {profile}: {str(e)}")
                cleanup_results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
        
        return {
            'message': f'Cleanup completed for {len(endpoint_details)} profiles',
            'execution_id': execution_id,
            'successful_cleanups': successful_cleanups,
            'failed_cleanups': len(endpoint_details) - successful_cleanups,
            'total_profiles': len(endpoint_details),
            'total_resources_cleaned': total_resources_cleaned,
            'cleanup_results': cleanup_results,
            'cost_savings': f"${calculate_cost_savings(successful_cleanups):.2f}/hour saved",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup process failed: {str(e)}")
        raise

def cleanup_profile_resources(profile: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleanup all resources for a single profile
    """
    
    result = {
        'profile': profile,
        'status': 'failed',
        'cleanup_start_time': datetime.now().isoformat(),
        'resources_cleaned': []
    }
    
    try:
        # Extract resource names from details
        endpoint_name = details.get('endpoint_name')
        endpoint_config_name = details.get('endpoint_config_name')
        model_name = details.get('model_name')
        
        # Track what gets cleaned up
        cleanup_actions = []
        
        # 1. Delete endpoint (highest priority)
        if endpoint_name:
            try:
                logger.info(f"Deleting endpoint: {endpoint_name}")
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                cleanup_actions.append(f"endpoint:{endpoint_name}")
                logger.info(f"Successfully deleted endpoint: {endpoint_name}")
            except Exception as e:
                if 'does not exist' in str(e).lower() or 'ValidationException' in str(e):
                    logger.info(f"Endpoint {endpoint_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete endpoint {endpoint_name}: {str(e)}")
        
        # 2. Delete endpoint configuration
        if endpoint_config_name:
            try:
                logger.info(f"Deleting endpoint configuration: {endpoint_config_name}")
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                cleanup_actions.append(f"endpoint_config:{endpoint_config_name}")
                logger.info(f"Successfully deleted endpoint config: {endpoint_config_name}")
            except Exception as e:
                if 'does not exist' in str(e).lower() or 'ValidationException' in str(e):
                    logger.info(f"Endpoint config {endpoint_config_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete endpoint config {endpoint_config_name}: {str(e)}")
        
        # 3. Delete model
        if model_name:
            try:
                logger.info(f"Deleting model: {model_name}")
                sagemaker_client.delete_model(ModelName=model_name)
                cleanup_actions.append(f"model:{model_name}")
                logger.info(f"Successfully deleted model: {model_name}")
            except Exception as e:
                if 'does not exist' in str(e).lower() or 'ValidationException' in str(e):
                    logger.info(f"Model {model_name} already deleted or doesn't exist")
                else:
                    logger.warning(f"Could not delete model {model_name}: {str(e)}")
        
        result.update({
            'status': 'success',
            'resources_cleaned': cleanup_actions,
            'cleanup_end_time': datetime.now().isoformat(),
            'message': f'Successfully cleaned up {len(cleanup_actions)} resources for {profile}',
            'cost_impact': f"Endpoint hosting costs eliminated for {profile}"
        })
        
        logger.info(f"Successfully cleaned up all resources for {profile}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to cleanup resources for {profile}: {str(e)}")
        result.update({
            'error': str(e),
            'cleanup_end_time': datetime.now().isoformat()
        })
        return result

def emergency_cleanup_all_prediction_endpoints(execution_id: str) -> Dict[str, Any]:
    """
    Emergency cleanup function - finds and deletes all prediction-related endpoints
    """
    
    try:
        logger.info("Starting emergency cleanup of all prediction endpoints")
        
        cleanup_results = {}
        total_cleaned = 0
        
        # Find all endpoints with prediction naming pattern
        paginator = sagemaker_client.get_paginator('list_endpoints')
        
        for page in paginator.paginate():
            for endpoint in page['Endpoints']:
                endpoint_name = endpoint['EndpointName']
                
                # Check if it's a prediction endpoint
                if 'energy-forecasting' in endpoint_name and any(
                    pattern in endpoint_name.lower() 
                    for pattern in ['rnn', 'rn', '-m-', '-s-', 'agr', '-l-', 'a6']
                ):
                    try:
                        # Get endpoint details for cleanup
                        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                        
                        # Extract profile from endpoint name
                        profile = extract_profile_from_endpoint_name(endpoint_name)
                        
                        # Cleanup this endpoint
                        details = {
                            'endpoint_name': endpoint_name,
                            'endpoint_config_name': response.get('EndpointConfigName'),
                            'model_name': None  # Would need to extract from endpoint config
                        }
                        
                        cleanup_result = cleanup_profile_resources(profile, details)
                        cleanup_results[profile] = cleanup_result
                        
                        if cleanup_result['status'] == 'success':
                            total_cleaned += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to cleanup endpoint {endpoint_name}: {str(e)}")
                        cleanup_results[endpoint_name] = {
                            'status': 'failed',
                            'error': str(e)
                        }
        
        return {
            'message': f'Emergency cleanup completed - {total_cleaned} endpoints cleaned',
            'execution_id': execution_id,
            'total_cleaned': total_cleaned,
            'cleanup_results': cleanup_results,
            'operation': 'emergency_cleanup',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {str(e)}")
        raise

def extract_profile_from_endpoint_name(endpoint_name: str) -> str:
    """
    Extract profile name from endpoint name
    """
    
    # Map endpoint patterns to profiles
    pattern_map = {
        'rnn': 'RNN',
        'rn': 'RN',
        '-m-': 'M',
        '-s-': 'S',
        'agr': 'AGR',
        '-l-': 'L',
        'a6': 'A6'
    }
    
    endpoint_lower = endpoint_name.lower()
    for pattern, profile in pattern_map.items():
        if pattern in endpoint_lower:
            return profile
    
    return 'Unknown'

def calculate_cost_savings(successful_cleanups: int) -> float:
    """
    Calculate estimated cost savings from endpoint cleanup
    ml.t2.medium costs approximately $0.047/hour
    """
    
    hourly_cost_per_endpoint = 0.047
    return successful_cleanups * hourly_cost_per_endpoint
