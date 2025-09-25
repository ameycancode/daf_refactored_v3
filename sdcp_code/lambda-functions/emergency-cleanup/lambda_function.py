"""
Emergency Cleanup Lambda Function
Handles resource cleanup when the prediction pipeline fails
Provides safety net for cost protection
"""

import json
import boto3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
stepfunctions_client = boto3.client('stepfunctions')
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Emergency cleanup of resources when pipeline fails
    """
    
    execution_id = context.aws_request_id
    
    try:
        logger.info(f"Starting emergency cleanup [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Extract parameters
        operation = event.get('operation', 'emergency_cleanup')
        pipeline_execution_id = event.get('execution_id', execution_id)
        cleanup_reason = event.get('cleanup_reason', 'Pipeline failure')
        
        logger.warning(f"EMERGENCY CLEANUP INITIATED: {cleanup_reason}")
        
        if operation == 'emergency_cleanup':
            result = perform_emergency_cleanup(pipeline_execution_id, cleanup_reason)
        elif operation == 'cleanup_by_pattern':
            pattern = event.get('pattern', 'energy-forecasting')
            result = cleanup_resources_by_pattern(pattern, pipeline_execution_id)
        elif operation == 'cleanup_old_resources':
            max_age_hours = event.get('max_age_hours', 24)
            result = cleanup_old_resources(max_age_hours, pipeline_execution_id)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            'statusCode': 200,
            'body': result
        }
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'operation': event.get('operation', 'unknown'),
                'status': 'emergency_cleanup_failed',
                'error': str(e),
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'warning': 'Manual cleanup may be required'
            }
        }

def perform_emergency_cleanup(execution_id: str, cleanup_reason: str) -> Dict[str, Any]:
    """
    Perform comprehensive emergency cleanup
    """
    
    cleanup_results = {
        'execution_id': execution_id,
        'cleanup_reason': cleanup_reason,
        'timestamp': datetime.now().isoformat(),
        'cleanup_operations': {},
        'resources_cleaned': [],
        'cleanup_errors': [],
        'cost_savings_per_hour': 0.0,
        'recommendations': []
    }
    
    try:
        logger.warning(f"Performing emergency cleanup for execution: {execution_id}")
        
        # Step 1: Find and cleanup prediction endpoints
        logger.info("Step 1: Cleaning up prediction endpoints...")
        endpoint_cleanup = cleanup_prediction_endpoints(execution_id)
        cleanup_results['cleanup_operations']['endpoints'] = endpoint_cleanup
        cleanup_results['resources_cleaned'].extend(endpoint_cleanup.get('cleaned_resources', []))
        cleanup_results['cleanup_errors'].extend(endpoint_cleanup.get('errors', []))
        cleanup_results['cost_savings_per_hour'] += endpoint_cleanup.get('cost_savings', 0.0)
        
        # Step 2: Cleanup any running Step Functions executions
        logger.info("Step 2: Checking for running Step Functions executions...")
        stepfunctions_cleanup = cleanup_running_executions(execution_id)
        cleanup_results['cleanup_operations']['step_functions'] = stepfunctions_cleanup
        
        # Step 3: Cleanup orphaned resources by naming pattern
        logger.info("Step 3: Cleaning up orphaned resources...")
        orphaned_cleanup = cleanup_orphaned_resources(execution_id)
        cleanup_results['cleanup_operations']['orphaned_resources'] = orphaned_cleanup
        cleanup_results['resources_cleaned'].extend(orphaned_cleanup.get('cleaned_resources', []))
        cleanup_results['cleanup_errors'].extend(orphaned_cleanup.get('errors', []))
        cleanup_results['cost_savings_per_hour'] += orphaned_cleanup.get('cost_savings', 0.0)
        
        # Step 4: Generate cleanup report
        logger.info("Step 4: Generating cleanup report...")
        report_info = generate_cleanup_report(cleanup_results)
        cleanup_results['cleanup_operations']['reporting'] = report_info
        
        # Step 5: Generate recommendations
        cleanup_results['recommendations'] = generate_cleanup_recommendations(cleanup_results)
        
        # Determine overall status
        total_cleaned = len(cleanup_results['resources_cleaned'])
        total_errors = len(cleanup_results['cleanup_errors'])
        
        if total_errors == 0:
            cleanup_results['status'] = 'emergency_cleanup_successful'
            logger.info(f"✓ Emergency cleanup completed successfully: {total_cleaned} resources cleaned")
        elif total_cleaned > 0:
            cleanup_results['status'] = 'emergency_cleanup_partial'
            logger.warning(f" Emergency cleanup partially successful: {total_cleaned} cleaned, {total_errors} errors")
        else:
            cleanup_results['status'] = 'emergency_cleanup_failed'
            logger.error(f"✗ Emergency cleanup failed: {total_errors} errors, no resources cleaned")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Emergency cleanup operation failed: {str(e)}")
        cleanup_results['status'] = 'emergency_cleanup_error'
        cleanup_results['cleanup_errors'].append(f"Emergency cleanup error: {str(e)}")
        return cleanup_results

def cleanup_prediction_endpoints(execution_id: str) -> Dict[str, Any]:
    """
    Find and cleanup all prediction-related endpoints
    """
    
    result = {
        'operation': 'cleanup_prediction_endpoints',
        'cleaned_resources': [],
        'errors': [],
        'cost_savings': 0.0
    }
    
    try:
        # Search for endpoints that match prediction patterns
        response = sagemaker_client.list_endpoints()
        
        prediction_endpoints = []
        for endpoint in response['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            
            # Look for prediction-related endpoints
            if (('energy-forecasting' in endpoint_name) and 
                (('pred' in endpoint_name) or 
                 ('prediction' in endpoint_name) or 
                 (execution_id[:8] in endpoint_name))):
                prediction_endpoints.append(endpoint_name)
        
        logger.info(f"Found {len(prediction_endpoints)} prediction endpoints to cleanup")
        
        # Cleanup each endpoint
        for endpoint_name in prediction_endpoints:
            try:
                endpoint_cleanup_result = cleanup_single_endpoint_complete(endpoint_name)
                
                if endpoint_cleanup_result['success']:
                    result['cleaned_resources'].extend(endpoint_cleanup_result['resources'])
                    result['cost_savings'] += 0.115  # Approximate ml.m5.large cost per hour
                    logger.info(f"✓ Cleaned up endpoint: {endpoint_name}")
                else:
                    result['errors'].append(f"Failed to cleanup endpoint {endpoint_name}: {endpoint_cleanup_result['error']}")
                    logger.error(f"✗ Failed to cleanup endpoint: {endpoint_name}")
                    
            except Exception as e:
                error_msg = f"Error cleaning endpoint {endpoint_name}: {str(e)}"
                result['errors'].append(error_msg)
                logger.error(error_msg)
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Endpoint cleanup search failed: {str(e)}")
        logger.error(f"Failed to search for prediction endpoints: {str(e)}")
        return result

def cleanup_single_endpoint_complete(endpoint_name: str) -> Dict[str, Any]:
    """
    Completely cleanup a single endpoint and all associated resources
    """
    
    result = {
        'success': False,
        'resources': [],
        'error': None
    }
    
    try:
        # Get endpoint details
        endpoint_details = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_details['EndpointConfigName']
        
        # Get model names from endpoint configuration
        config_details = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        
        model_names = [variant['ModelName'] for variant in config_details['ProductionVariants']]
        
        # Delete endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        result['resources'].append(f"endpoint: {endpoint_name}")
        
        # Delete endpoint configuration
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        result['resources'].append(f"endpoint_config: {endpoint_config_name}")
        
        # Delete models
        for model_name in model_names:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                result['resources'].append(f"model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not delete model {model_name}: {str(e)}")
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        return result

def cleanup_running_executions(execution_id: str) -> Dict[str, Any]:
    """
    Find and stop any running Step Functions executions related to this pipeline
    """
    
    result = {
        'operation': 'cleanup_step_functions',
        'stopped_executions': [],
        'errors': []
    }
    
    try:
        # List state machines
        response = stepfunctions_client.list_state_machines()
        
        prediction_pipelines = []
        for machine in response['stateMachines']:
            if 'prediction' in machine['name'].lower() and 'energy-forecasting' in machine['name']:
                prediction_pipelines.append(machine['stateMachineArn'])
        
        # Check for running executions
        for pipeline_arn in prediction_pipelines:
            try:
                executions_response = stepfunctions_client.list_executions(
                    stateMachineArn=pipeline_arn,
                    statusFilter='RUNNING',
                    maxResults=50
                )
                
                for execution in executions_response['executions']:
                    execution_arn = execution['executionArn']
                    execution_name = execution['name']
                    
                    # Check if this execution is related to our failed pipeline
                    if (execution_id in execution_name or 
                        'container-integration-test' in execution_name or
                        'prediction' in execution_name):
                        
                        try:
                            stepfunctions_client.stop_execution(
                                executionArn=execution_arn,
                                error="EmergencyCleanup",
                                cause=f"Emergency cleanup for failed pipeline {execution_id}"
                            )
                            
                            result['stopped_executions'].append({
                                'execution_arn': execution_arn,
                                'execution_name': execution_name
                            })
                            logger.info(f"✓ Stopped Step Functions execution: {execution_name}")
                            
                        except Exception as e:
                            error_msg = f"Failed to stop execution {execution_name}: {str(e)}"
                            result['errors'].append(error_msg)
                            logger.warning(error_msg)
                            
            except Exception as e:
                error_msg = f"Error checking executions for {pipeline_arn}: {str(e)}"
                result['errors'].append(error_msg)
                logger.warning(error_msg)
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Step Functions cleanup failed: {str(e)}")
        logger.error(f"Failed to cleanup Step Functions executions: {str(e)}")
        return result

def cleanup_orphaned_resources(execution_id: str) -> Dict[str, Any]:
    """
    Cleanup any orphaned resources that might have been left behind
    """
    
    result = {
        'operation': 'cleanup_orphaned_resources',
        'cleaned_resources': [],
        'errors': [],
        'cost_savings': 0.0
    }
    
    try:
        # Find orphaned endpoints (older than 2 hours)
        cutoff_time = datetime.now() - timedelta(hours=2)
        
        response = sagemaker_client.list_endpoints()
        
        for endpoint in response['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            creation_time = endpoint['CreationTime'].replace(tzinfo=None)
            
            # Check if this is an energy forecasting endpoint that's been running too long
            if (('energy-forecasting' in endpoint_name) and 
                (creation_time < cutoff_time) and
                (any(keyword in endpoint_name for keyword in ['pred', 'test', 'temp']))):
                
                try:
                    cleanup_result = cleanup_single_endpoint_complete(endpoint_name)
                    
                    if cleanup_result['success']:
                        result['cleaned_resources'].extend(cleanup_result['resources'])
                        result['cost_savings'] += 0.115
                        logger.info(f"✓ Cleaned up orphaned endpoint: {endpoint_name}")
                    else:
                        result['errors'].append(f"Failed to cleanup orphaned endpoint {endpoint_name}: {cleanup_result['error']}")
                        
                except Exception as e:
                    error_msg = f"Error cleaning orphaned endpoint {endpoint_name}: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Orphaned resource cleanup failed: {str(e)}")
        logger.error(f"Failed to cleanup orphaned resources: {str(e)}")
        return result

def cleanup_resources_by_pattern(pattern: str, execution_id: str) -> Dict[str, Any]:
    """
    Cleanup resources matching a specific pattern
    """
    
    result = {
        'operation': 'cleanup_by_pattern',
        'pattern': pattern,
        'cleaned_resources': [],
        'errors': [],
        'cost_savings': 0.0
    }
    
    try:
        response = sagemaker_client.list_endpoints()
        
        matching_endpoints = []
        for endpoint in response['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            if pattern in endpoint_name:
                matching_endpoints.append(endpoint_name)
        
        logger.info(f"Found {len(matching_endpoints)} endpoints matching pattern '{pattern}'")
        
        for endpoint_name in matching_endpoints:
            try:
                cleanup_result = cleanup_single_endpoint_complete(endpoint_name)
                
                if cleanup_result['success']:
                    result['cleaned_resources'].extend(cleanup_result['resources'])
                    result['cost_savings'] += 0.115
                else:
                    result['errors'].append(f"Pattern cleanup failed for {endpoint_name}: {cleanup_result['error']}")
                    
            except Exception as e:
                result['errors'].append(f"Error in pattern cleanup for {endpoint_name}: {str(e)}")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Pattern cleanup failed: {str(e)}")
        return result

def cleanup_old_resources(max_age_hours: int, execution_id: str) -> Dict[str, Any]:
    """
    Cleanup resources older than specified age
    """
    
    result = {
        'operation': 'cleanup_old_resources',
        'max_age_hours': max_age_hours,
        'cleaned_resources': [],
        'errors': [],
        'cost_savings': 0.0
    }
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        response = sagemaker_client.list_endpoints()
        
        old_endpoints = []
        for endpoint in response['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            creation_time = endpoint['CreationTime'].replace(tzinfo=None)
            
            if (creation_time < cutoff_time and 
                'energy-forecasting' in endpoint_name):
                old_endpoints.append(endpoint_name)
        
        logger.info(f"Found {len(old_endpoints)} endpoints older than {max_age_hours} hours")
        
        for endpoint_name in old_endpoints:
            try:
                cleanup_result = cleanup_single_endpoint_complete(endpoint_name)
                
                if cleanup_result['success']:
                    result['cleaned_resources'].extend(cleanup_result['resources'])
                    result['cost_savings'] += 0.115
                else:
                    result['errors'].append(f"Age-based cleanup failed for {endpoint_name}: {cleanup_result['error']}")
                    
            except Exception as e:
                result['errors'].append(f"Error in age-based cleanup for {endpoint_name}: {str(e)}")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Age-based cleanup failed: {str(e)}")
        return result

def generate_cleanup_report(cleanup_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate and save emergency cleanup report
    """
    
    try:
        current_date = datetime.now().strftime('%Y%m%d')
        execution_id = cleanup_results.get('execution_id', 'unknown')
        
        # Create comprehensive report
        report = {
            'emergency_cleanup_report': cleanup_results,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_resources_cleaned': len(cleanup_results.get('resources_cleaned', [])),
                'total_errors': len(cleanup_results.get('cleanup_errors', [])),
                'estimated_cost_savings_per_hour': cleanup_results.get('cost_savings_per_hour', 0.0),
                'cleanup_reason': cleanup_results.get('cleanup_reason', 'Unknown')
            }
        }
        
        # Save report to S3
        report_key = f"archived_folders/forecasting/emergency_reports/{current_date}/emergency_cleanup_{execution_id}.json"
        data_bucket = os.environ.get('DATA_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
        
        s3_client.put_object(
            Bucket=data_bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2, default=str),
            ContentType='application/json'
        )
        
        logger.info(f"Emergency cleanup report saved: {report_key}")
        
        return {
            'report_generated': True,
            'report_location': report_key
        }
        
    except Exception as e:
        logger.error(f"Failed to generate cleanup report: {str(e)}")
        return {
            'report_generated': False,
            'error': str(e)
        }

def generate_cleanup_recommendations(cleanup_results: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations based on cleanup results
    """
    
    recommendations = []
    
    try:
        total_cleaned = len(cleanup_results.get('resources_cleaned', []))
        total_errors = len(cleanup_results.get('cleanup_errors', []))
        cost_savings = cleanup_results.get('cost_savings_per_hour', 0.0)
        
        # Success recommendations
        if total_cleaned > 0:
            recommendations.append(f"✓ Successfully cleaned {total_cleaned} resources")
            
        if cost_savings > 0:
            recommendations.append(f" Estimated cost savings: ${cost_savings:.3f}/hour")
        
        # Error recommendations
        if total_errors > 0:
            recommendations.append(f" {total_errors} cleanup errors occurred - manual review needed")
            recommendations.append(" Check CloudWatch logs for detailed error information")
        
        # Prevention recommendations
        recommendations.append(" Consider implementing automated resource tagging for better tracking")
        recommendations.append(" Review pipeline timeout settings to prevent resource leaks")
        recommendations.append(" Monitor pipeline execution patterns for early failure detection")
        
        # Cost optimization recommendations
        if cost_savings > 5.0:  # If we saved significant cost
            recommendations.append(" Consider reviewing endpoint lifecycle management")
            recommendations.append(" Implement stricter resource cleanup policies")
        
        return recommendations
        
    except Exception as e:
        logger.warning(f"Could not generate recommendations: {str(e)}")
        return [" Cleanup completed - manual review recommended"]
