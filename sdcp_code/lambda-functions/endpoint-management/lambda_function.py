"""
Enhanced Endpoint Management Lambda Function - FIXED VERSION
This version correctly stores complete SageMaker resource creation parameters
for proper endpoint recreation during predictions.

Key Fix: The save_endpoint_configuration function now captures the actual
SageMaker API parameters needed to recreate Model, EndpointConfig, and Endpoint
instead of just storing metadata references.
"""

import json
import boto3
import logging
import time
from datetime import datetime
from typing import Dict, Any
import uuid
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

# Configuration
# ENDPOINT_CONFIG_BUCKET = "sdcp-dev-sagemaker-energy-forecasting-data"
ENDPOINT_CONFIG_PREFIX = "sdcp_modeling/endpoint-configurations/"
EXECUTION_LOCK_PREFIX = "sdcp_modeling/execution-locks/"

def lambda_handler(event, context):
    """
    Enhanced Lambda handler supporting both single profile and batch operations
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting endpoint management process [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
       
        # Determine operation type
        operation = event.get('operation', 'create_all_endpoints')
       
        if operation == 'create_endpoint':
            # Handle single profile operation for parallel Step Functions
            return handle_single_profile_endpoint(event, context, execution_id)
        else:
            # Handle batch operation (your original logic)
            return handle_batch_endpoints(event, context, execution_id)
       
    except Exception as e:
        logger.error(f"Endpoint management process failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Endpoint management process failed'
            }
        }

def handle_single_profile_endpoint(event, context, execution_id):
    """
    Handle single profile endpoint creation - FIXED to process ONLY the intended profile
    """
   
    try:
        logger.info(f"Processing single profile endpoint creation")
        logger.info(f"Event received: {json.dumps(event, default=str)}")
       
        # FIXED: Extract the SINGLE profile that this Lambda execution should process
        operation = event.get('operation', 'create_endpoint')
       
        # Extract the profile this execution should handle
        profile = event.get('profile')
        training_metadata = event.get('training_metadata', {})
        approved_models = event.get('approved_models', {})
       
        # CRITICAL FIX: Process ONLY the single profile, not all profiles
        if not profile:
            raise ValueError("Profile parameter is required but not found in event")
       
        if profile not in approved_models:
            raise ValueError(f"Profile '{profile}' not found in approved_models")
       
        logger.info(f"Processing SINGLE profile endpoint lifecycle for: {profile}")
       
        # Get model info for THIS specific profile only
        model_info = approved_models[profile]
       
        # Extract the model_info from the approved_models structure
        profile_model_info = {
            'model_package_arn': model_info.get('model_package_arn'),
            'status': model_info.get('status'),
            'approval_status': model_info.get('approval_status'),
            'model_package_group': model_info.get('model_package_group'),
            'registration_time': model_info.get('registration_time'),
            'model_metadata': model_info.get('model_metadata', {})
        }
       
        # Process ONLY this single profile
        result = process_profile_endpoint_lifecycle(profile, profile_model_info, training_metadata, execution_id)
       
        if result.get('status') == 'success':
            logger.info(f"✓ {profile} endpoint lifecycle completed successfully")
            return {
                'statusCode': 200,
                'body': result
            }
        else:
            logger.error(f"✗ {profile} endpoint lifecycle failed: {result.get('error')}")
            return {
                'statusCode': 500,
                'body': result
            }
       
    except Exception as e:
        logger.error(f"Single profile endpoint creation failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'profile': event.get('profile'),
                'message': 'Single profile endpoint creation failed',
                'event_received': str(event)  # For debugging
            }
        }

def handle_batch_endpoints(event, context, execution_id):
    """
    Handle batch endpoint creation (backward compatibility with your original logic)
    """
   
    try:
        logger.info(f"Processing batch endpoint creation for multiple profiles")
       
        # Extract batch information
        approved_models = event.get('approved_models', {})
        training_metadata = event.get('training_metadata', {})
       
        if not approved_models:
            raise ValueError("No approved_models provided for batch endpoint creation")
       
        # Process each profile (THIS is where we loop through all profiles)
        results = {}
        successful_profiles = 0
       
        for profile, model_info in approved_models.items():
            try:
                logger.info(f"Processing batch endpoint for profile: {profile}")
               
                # Extract the model_info from the approved_models structure
                profile_model_info = {
                    'model_package_arn': model_info.get('model_package_arn'),
                    'status': model_info.get('status'),
                    'approval_status': model_info.get('approval_status'),
                    'model_package_group': model_info.get('model_package_group'),
                    'registration_time': model_info.get('registration_time'),
                    'model_metadata': model_info.get('model_metadata', {})
                }
               
                result = process_profile_endpoint_lifecycle(profile, profile_model_info, training_metadata, execution_id)
                results[profile] = result
               
                if result.get('status') == 'success':
                    successful_profiles += 1
                    logger.info(f"✓ {profile} endpoint lifecycle completed successfully")
                else:
                    logger.error(f"✗ {profile} endpoint lifecycle failed: {result.get('error')}")
                   
            except Exception as e:
                logger.error(f"✗ Profile {profile} failed: {str(e)}")
                results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
       
        return {
            'statusCode': 200,
            'body': {
                'message': f'Batch endpoint creation completed: {successful_profiles}/{len(approved_models)} successful',
                'successful_profiles': successful_profiles,
                'total_profiles': len(approved_models),
                'detailed_results': results,
                'execution_id': execution_id
            }
        }
       
    except Exception as e:
        logger.error(f"Batch endpoint creation failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Batch endpoint creation failed'
            }
        }

def handle_batch_endpoints(event, context, execution_id):
    """
    Handle batch endpoint creation (backward compatibility with your original logic)
    """
   
    try:
        logger.info(f"Processing batch endpoint creation for multiple profiles")
       
        # Extract batch information
        approved_models = event.get('approved_models', {})
        training_metadata = event.get('training_metadata', {})
       
        if not approved_models:
            raise ValueError("No approved models provided for batch endpoint creation")
       
        # Process each profile
        results = {}
        successful_profiles = 0
       
        for profile, model_info in approved_models.items():
            try:
                logger.info(f"Processing batch endpoint for profile: {profile}")
               
                result = process_profile_endpoint_lifecycle(profile, model_info, training_metadata, execution_id)
                results[profile] = result
               
                if result.get('status') == 'success':
                    successful_profiles += 1
                    logger.info(f"✓ {profile} endpoint lifecycle completed successfully")
                else:
                    logger.error(f"✗ {profile} endpoint lifecycle failed: {result.get('error')}")
                   
            except Exception as e:
                logger.error(f"✗ Profile {profile} failed: {str(e)}")
                results[profile] = {
                    'status': 'failed',
                    'error': str(e),
                    'profile': profile
                }
       
        return {
            'statusCode': 200,
            'body': {
                'message': f'Batch endpoint creation completed: {successful_profiles}/{len(approved_models)} successful',
                'successful_profiles': successful_profiles,
                'total_profiles': len(approved_models),
                'detailed_results': results,
                'execution_id': execution_id
            }
        }
       
    except Exception as e:
        logger.error(f"Batch endpoint creation failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'error': str(e),
                'execution_id': execution_id,
                'message': 'Batch endpoint creation failed'
            }
        }

def process_profile_endpoint_lifecycle(profile: str, model_info: Dict[str, Any],
                                     training_metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
    """Process complete endpoint lifecycle for a single profile"""
   
    result = {
        'profile': profile,
        'status': 'failed',
        'execution_id': execution_id,
        'steps_completed': []
    }
   
    endpoint_name = None
    endpoint_config_name = None
    model_name = None
   
    try:
        # Step 1: Create model
        logger.info(f"Step 1: Creating Model for {profile}")
        model_name = create_model_for_profile(profile, model_info, execution_id)
       
        if not model_name:
            result['error'] = "Failed to create model"
            return result
       
        result['model_name'] = model_name
        result['steps_completed'].append('model_created')
        logger.info(f"Created model: {model_name}")
       
        # Step 2: Create endpoint configuration
        logger.info(f"Step 2: Creating Endpoint Configuration for {profile}")
        endpoint_config_name = create_endpoint_config_for_profile(profile, model_name, execution_id)
       
        if not endpoint_config_name:
            result['error'] = "Failed to create endpoint configuration"
            return result
       
        result['endpoint_config_name'] = endpoint_config_name
        result['steps_completed'].append('endpoint_config_created')
        logger.info(f"Created endpoint configuration: {endpoint_config_name}")
       
        # Step 3: Create Endpoint
        logger.info(f"Step 3: Creating Endpoint for {profile}")
        endpoint_name = create_endpoint(profile, endpoint_config_name, execution_id)
       
        if not endpoint_name:
            result['error'] = "Failed to create endpoint"
            return result
       
        result['endpoint_name'] = endpoint_name
        result['steps_completed'].append('endpoint_created')
        logger.info(f"Created endpoint: {endpoint_name}")
       
        # Step 4: Wait for endpoint to be InService
        logger.info(f"Step 4: Waiting for endpoint to be InService")
        if not wait_for_endpoint_inservice(endpoint_name, timeout_minutes=15):
            result['error'] = "Endpoint failed to reach InService status"
            return result
       
        result['steps_completed'].append('endpoint_inservice')
        logger.info(f"Endpoint {endpoint_name} is InService")
       
        # Step 5: Test endpoint
        logger.info(f"Step 5: Testing endpoint inference")
        inference_success = test_endpoint_inference(endpoint_name, profile)
       
        if not inference_success:
            logger.warning(f"Endpoint inference test failed for {profile}, but continuing")
            result['inference_warning'] = "Inference test failed"
        else:
            result['steps_completed'].append('endpoint_tested')
            logger.info(f"Endpoint inference test successful")
       
        # Step 6: Save COMPLETE endpoint configuration to S3 (FIXED VERSION)
        logger.info(f"Step 6: Saving COMPLETE endpoint configuration to S3")
        config_s3_info = save_complete_endpoint_configuration(
            endpoint_name, endpoint_config_name, model_name, profile, model_info, training_metadata
        )
       
        if not config_s3_info:
            result['error'] = "Failed to save endpoint configuration"
            return result
       
        result['configuration_s3'] = config_s3_info
        result['steps_completed'].append('configuration_saved')
        logger.info(f"Saved COMPLETE endpoint configuration to S3")
       
        # Step 7: Delete endpoint for cost optimization
        logger.info(f"Step 7: Deleting endpoint for cost optimization")
        deletion_success = delete_endpoint_and_resources(endpoint_name, endpoint_config_name, model_name)
       
        if deletion_success:
            result['endpoint_deleted'] = True
            result['steps_completed'].append('endpoint_deleted')
            logger.info(f"Successfully deleted endpoint {endpoint_name}")
        else:
            logger.warning(f"Failed to delete endpoint {endpoint_name}")
            result['deletion_warning'] = "Failed to delete endpoint"
       
        result['status'] = 'success'
        return result
       
    except Exception as e:
        logger.error(f"Error in endpoint lifecycle for {profile}: {str(e)}")
        result['error'] = str(e)
       
        # Cleanup on failure
        if endpoint_name:
            try:
                logger.info(f"Cleaning up failed endpoint: {endpoint_name}")
                delete_endpoint_and_resources(endpoint_name, endpoint_config_name, model_name)
            except Exception:
                pass
       
        return result

def create_model_for_profile(profile: str, model_info: Dict[str, Any], execution_id: str) -> str:
    """Create SageMaker model for a specific profile"""
   
    try:
        # Generate unique model name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"energy-forecasting-{profile.lower()}-model-{timestamp}-{execution_id[:8]}"
       
        # Get execution role
        role_arn = get_sagemaker_execution_role()
       
        # Get model package details from Model Registry
        model_package_arn = model_info.get('model_package_arn')
        if not model_package_arn:
            raise ValueError(f"No model package ARN found for profile {profile}")
       
        # Create model from Model Registry package
        sagemaker_client.create_model(
            ModelName=model_name,
            Containers=[
                {
                    'ModelPackageName': model_package_arn
                }
            ],
            ExecutionRoleArn=role_arn,
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'EnergyForecasting'},
                {'Key': 'CreatedBy', 'Value': 'EndpointManagementLambda'},
                {'Key': 'ExecutionId', 'Value': execution_id}
            ]
        )
       
        logger.info(f"Successfully created model: {model_name}")
        return model_name
       
    except Exception as e:
        logger.error(f"Failed to create model for {profile}: {str(e)}")
        return None

def create_endpoint_config_for_profile(profile: str, model_name: str, execution_id: str) -> str:
    """Create endpoint configuration for a specific profile"""
   
    try:
        # Generate unique endpoint config name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_config_name = f"energy-forecasting-{profile.lower()}-config-{timestamp}-{execution_id[:8]}"
       
        # Create endpoint configuration
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1.0
                }
            ],
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'EnergyForecasting'},
                {'Key': 'CreatedBy', 'Value': 'EndpointManagementLambda'},
                {'Key': 'ExecutionId', 'Value': execution_id}
            ]
        )
       
        logger.info(f"Successfully created endpoint configuration: {endpoint_config_name}")
        return endpoint_config_name
       
    except Exception as e:
        logger.error(f"Failed to create endpoint configuration for {profile}: {str(e)}")
        return None

def create_endpoint(profile: str, endpoint_config_name: str, execution_id: str) -> str:
    """Create SageMaker endpoint"""
   
    try:
        # Generate unique endpoint name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"energy-forecasting-{profile.lower()}-endpoint-{timestamp}-{execution_id[:8]}"
       
        # Create endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Profile', 'Value': profile},
                {'Key': 'Purpose', 'Value': 'EnergyForecasting'},
                {'Key': 'CreatedBy', 'Value': 'EndpointManagementLambda'},
                {'Key': 'ExecutionId', 'Value': execution_id}
            ]
        )
       
        logger.info(f"Successfully created endpoint: {endpoint_name}")
        return endpoint_name
       
    except Exception as e:
        logger.error(f"Failed to create endpoint for {profile}: {str(e)}")
        return None

def wait_for_endpoint_inservice(endpoint_name: str, timeout_minutes: int = 15) -> bool:
    """Wait for endpoint to reach InService status"""
   
    timeout_seconds = timeout_minutes * 60
    start_time = time.time()
   
    while time.time() - start_time < timeout_seconds:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
           
            logger.info(f"Endpoint {endpoint_name} status: {status}")
           
            if status == 'InService':
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown error')
                logger.error(f"Endpoint {endpoint_name} failed: {failure_reason}")
                return False
           
            # Wait before checking again
            time.sleep(30)
           
        except Exception as e:
            logger.warning(f"Error checking endpoint status: {str(e)}")
            time.sleep(30)
   
    logger.error(f"Timeout waiting for endpoint {endpoint_name} to be ready")
    return False

def test_endpoint_inference(endpoint_name: str, profile: str) -> bool:
    """Test endpoint inference with sample data - FIXED validation logic"""
   
    try:
        # Create sample input data based on profile
        sample_data = {
            "instances": [
                [1000, 2025, 1, 29, 12, 3, 0, 0, 1, 75.5, 0.85, 0.80]  # Basic features
            ]
        }
       
        # Add radiation for RN profile
        if profile == 'RN':
            sample_data["instances"][0].append(500.0)  # shortwave_radiation
       
        # Invoke endpoint
        runtime_client = boto3.client('sagemaker-runtime')
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_data)
        )
       
        result = json.loads(response['Body'].read().decode())
       
        # FIXED: Accept the enhanced response format
        # The response shows: {'predictions': [32.56577682495117], 'metadata': {...}}
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
            if isinstance(predictions, list) and len(predictions) > 0:
                logger.info(f"Endpoint inference test successful for {profile}: {predictions[0]}")
                return True
            else:
                logger.error(f"Empty predictions for {profile}: {result}")
                return False
        elif isinstance(result, list) and len(result) > 0:
            logger.info(f"Endpoint inference test successful for {profile}: {result[0]}")
            return True
        else:
            logger.error(f"Invalid inference response for {profile}: {result}")
            return False
           
    except Exception as e:
        logger.error(f"Endpoint inference test failed for {profile}: {str(e)}")
        return False

def save_complete_endpoint_configuration(endpoint_name: str, endpoint_config_name: str, model_name: str,
                                       profile: str, model_info: Dict[str, Any], training_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED VERSION: Save COMPLETE endpoint configuration with ACTUAL container details from Model Package
    """
   
    try:
        logger.info(f"Capturing complete configuration details for {profile} with actual container details")
       
        # 1. Get the actual SageMaker resource details before they're deleted
        endpoint_config_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        model_response = sagemaker_client.describe_model(ModelName=model_name)
        endpoint_response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
       
        # 2. FIXED: Get ACTUAL container details from the Model Package
        model_package_arn = model_info.get('model_package_arn') or model_info.get('ModelPackageArn')
       
        if not model_package_arn:
            raise ValueError(f"No model_package_arn found in model_info for {profile}")
       
        # Get the Model Package details to extract actual container information
        logger.info(f"Fetching Model Package details from: {model_package_arn}")
        model_package_response = sagemaker_client.describe_model_package(ModelPackageName=model_package_arn)
       
        # Extract ACTUAL container details from Model Package
        if 'InferenceSpecification' in model_package_response:
            inference_spec = model_package_response['InferenceSpecification']
            containers = inference_spec.get('Containers', [])
            logger.info(f"Containers: {containers}")
           
            if containers and len(containers) > 0:
                # Use the first container (primary container)
                primary_container_spec = containers[0]
               
                # Extract ACTUAL values
                actual_image = primary_container_spec.get('Image', 'UNKNOWN_IMAGE')
                actual_model_data_url = primary_container_spec.get('ModelDataUrl', 'UNKNOWN_MODEL_DATA_URL')
                container_environment = primary_container_spec.get('Environment', {})
               
                # Combine with standard SageMaker environment variables
                complete_environment = {
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model",
                    **container_environment  # Add any Model Package specific environment variables
                }
               
                # Create the model config with ACTUAL values
                model_config = {
                    "execution_role_arn": model_response['ExecutionRoleArn'],
                    "primary_container": {
                        "Image": actual_image,
                        "Mode": "SingleModel",
                        "ModelDataUrl": actual_model_data_url,
                        "ModelDataSource": {
                            "S3DataSource": {
                                "S3Uri": actual_model_data_url,
                                "S3DataType": "S3Object",
                                "CompressionType": "Gzip"
                            }
                        },
                        "Environment": complete_environment
                    },
                    "model_package_name": model_package_arn,
                    "tags": []
                }
               
                logger.info(f"Successfully extracted actual container details:")
                logger.info(f"  Image: {actual_image}")
                logger.info(f"  ModelDataUrl: {actual_model_data_url}")
                logger.info(f"  Environment vars: {len(complete_environment)}")
               
            else:
                raise ValueError(f"No containers found in Model Package InferenceSpecification for {profile}")
        else:
            raise ValueError(f"No InferenceSpecification found in Model Package for {profile}")
       
        # 3. Generate current timestamp
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       
        # 4. Create configuration in YOUR PROVEN WORKING FORMAT with ACTUAL values
        complete_config_data = {
            # Basic identification (matching your format)
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "endpoint_config_name": endpoint_config_name,
            "model_package_arn": model_package_arn,
            "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "instance_type": endpoint_config_response['ProductionVariants'][0]['InstanceType'],
            "instance_count": endpoint_config_response['ProductionVariants'][0]['InitialInstanceCount'],
            "environment": "dev",
            "customer_profile": profile,
            "customer_segment": "FORECASTING",
            "cost_optimized": True,
            "delete_recreate_enabled": True,
            "created_at": current_timestamp,
           
            # Endpoint configuration (matching your format exactly)
            "endpoint_config": {
                "production_variants": [
                    {
                        "VariantName": variant['VariantName'],
                        "ModelName": variant['ModelName'],  # Will be replaced during recreation
                        "InitialInstanceCount": variant['InitialInstanceCount'],
                        "InstanceType": variant['InstanceType'],
                        "InitialVariantWeight": variant['InitialVariantWeight']
                    } for variant in endpoint_config_response['ProductionVariants']
                ],
                "tags": []
            },
           
            # Model configuration with ACTUAL values (matching your format exactly)
            "model_config": model_config,
           
            # Recreation notes (matching your format)
            "recreation_notes": {
                "approach": "delete_recreate",
                "cost_optimization": "endpoint_deleted_after_deployment_and_predictions",
                "recreation_method": "lambda_function_recreates_from_this_config",
                "estimated_startup_time": "3-5_minutes",
                "container_details_source": "model_package_inference_specification"
            },
           
            # Additional metadata for our pipeline
            "training_metadata": training_metadata,
            "model_info": model_info,
            "model_package_details": {
                "model_package_arn": model_package_arn,
                "model_package_status": model_package_response.get('ModelPackageStatus', 'UNKNOWN'),
                "creation_time": model_package_response.get('CreationTime', '').isoformat() if model_package_response.get('CreationTime') else None
            },
            "creation_timestamp": datetime.now().isoformat(),
            "created_by": "training-pipeline-enhanced",
            "configuration_version": "4.0_actual_container_details"
        }
       
        # 5. Save to S3 with profile-specific folder structure
        current_date = datetime.now().strftime("%Y%m%d")
        s3_key = f"{ENDPOINT_CONFIG_PREFIX}{profile}/endpoint_config.json"

        data_bucket = os.environ.get('DATA_BUCKET', 'sdcp-dev-sagemaker-energy-forecasting-data')
       
        # Upload to S3
        s3_client.put_object(
            Bucket=data_bucket,
            Key=s3_key,
            Body=json.dumps(complete_config_data, indent=2, default=str),
            ContentType='application/json'
        )
       
        logger.info(f"Saved COMPLETE endpoint configuration with ACTUAL container details to S3: s3://{data_bucket}/{s3_key}")
       
        return {
            's3_bucket': data_bucket,
            's3_key': s3_key,
            'profile': profile,
            'timestamp': datetime.now().isoformat(),
            'configuration_version': '4.0_actual_container_details',
            'format_source': 'model_package_inference_specification',
            'actual_image': actual_image,
            'actual_model_data_url': actual_model_data_url
        }
       
    except Exception as e:
        logger.error(f"Failed to save complete endpoint configuration for {profile}: {str(e)}")
        logger.error(f"Model response keys: {list(model_response.keys()) if 'model_response' in locals() else 'No model_response'}")
        if 'model_package_response' in locals():
            logger.error(f"Model package response keys: {list(model_package_response.keys())}")
        return None

def delete_endpoint_and_resources(endpoint_name: str, endpoint_config_name: str = None, model_name: str = None) -> bool:
    """Delete endpoint and associated resources for cost optimization"""
   
    try:
        deletion_results = []
       
        # Delete endpoint
        if endpoint_name:
            try:
                sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                deletion_results.append(f"endpoint:{endpoint_name}")
                logger.info(f"Deleted endpoint: {endpoint_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint {endpoint_name}: {str(e)}")
       
        # Delete endpoint configuration
        if endpoint_config_name:
            try:
                sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
                deletion_results.append(f"endpoint_config:{endpoint_config_name}")
                logger.info(f"Deleted endpoint configuration: {endpoint_config_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint config {endpoint_config_name}: {str(e)}")
       
        # Delete model
        if model_name:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
                deletion_results.append(f"model:{model_name}")
                logger.info(f"Deleted model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to delete model {model_name}: {str(e)}")
       
        logger.info(f"Deletion completed: {deletion_results}")
        return True
       
    except Exception as e:
        logger.error(f"Error during resource cleanup: {str(e)}")
        return False

def get_sagemaker_execution_role() -> str:
    """Get SageMaker execution role ARN"""
   
    try:
        # Try to get from environment variable first
        role_arn = os.environ.get('SAGEMAKER_EXECUTION_ROLE')
       
        if not role_arn:
            # Construct default role ARN
            account_id = boto3.client('sts').get_caller_identity()['Account']
            role_arn = f"arn:aws:iam::{account_id}:role/sdcp-dev-sagemaker-energy-forecasting-datascientist-role"
       
        logger.info(f"Using SageMaker execution role: {role_arn}")
        return role_arn
       
    except Exception as e:
        logger.error(f"Failed to get SageMaker execution role: {str(e)}")
        raise
