#!/usr/bin/env python3
"""
Complete MLOps Cleanup Script
deployment/cleanup_enhanced_mlops.py

Deletes ALL resources created by the Enhanced MLOps deployment script.
This script reverses all 7 steps of the deployment in reverse order:

7. Remove integration test artifacts
6. Remove deployment validation resources
5. Delete EventBridge rules and targets
4. Delete ECR repositories and container images
3. Delete Step Functions state machines
2. Delete Lambda functions and layers
1. Clean up S3 artifacts (optional, keeps buckets)

CAUTION: This script will DELETE resources. Use with care!
"""

import json
import boto3
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMLOpsCleanup:
    """
    Complete cleanup manager for the enhanced MLOps pipeline
    """
    
    def __init__(self, region: str = "us-west-2", environment: str = "dev"):
        """Initialize the enhanced MLOps cleanup"""
        
        self.region = region
        self.environment = environment
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.iam_client = boto3.client('iam')
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Configuration
        self.config = {
            'region': region,
            'account_id': self.account_id,
            'data_bucket': f'sdcp-{environment}-sagemaker-energy-forecasting-data',
            'model_bucket': f'sdcp-{environment}-sagemaker-energy-forecasting-models',
            'lambda_functions': [
                'energy-forecasting-model-registry',
                'energy-forecasting-endpoint-management',
                'energy-forecasting-prediction-endpoint-manager',
                'energy-forecasting-prediction-cleanup',
                'energy-forecasting-profile-validator',
                'energy-forecasting-profile-endpoint-creator',
                'energy-forecasting-profile-cleanup',
                'energy-forecasting-endpoint-status-checker',
                'energy-forecasting-prediction-summary',
                'energy-forecasting-emergency-cleanup',
                'energy-forecasting-profile-predictor'
            ],
            'step_functions': [
                'energy-forecasting-training-pipeline',
                'energy-forecasting-enhanced-prediction-pipeline',
                'energy-forecasting-daily-predictions'  # Legacy pipeline if exists
            ],
            'eventbridge_rules': [
                'energy-forecasting-enhanced-daily-predictions',
                'energy-forecasting-monthly-training-pipeline',
                'energy-forecasting-daily-predictions',  # Legacy rule if exists
                'energy-forecasting-monthly-training'   # Legacy rule if exists
            ],
            'ecr_repositories': [
                'energy-preprocessing',
                'energy-training',
                'energy-prediction'
            ],
            'lambda_layers': [
                'SecureEnergyForecastingLayer2025'
            ]
        }
        
        logger.info(f"Enhanced MLOps Cleanup initialized for {environment} environment")
        logger.info(f"Account: {self.account_id}, Region: {region}")

    def cleanup_complete_mlops_pipeline(self, skip_s3: bool = True, confirm: bool = False) -> Dict[str, Any]:
        """
        Complete cleanup of the enhanced MLOps pipeline
        
        Args:
            skip_s3: If True, preserves S3 buckets and data (recommended)
            confirm: If True, proceeds with deletion. If False, shows what would be deleted.
        """
        
        if not confirm:
            logger.warning("="*80)
            logger.warning("DRY RUN MODE - No resources will be deleted")
            logger.warning("Use --confirm flag to actually delete resources")
            logger.warning("="*80)
        
        logger.info("="*100)
        logger.info("ENHANCED MLOPS PIPELINE CLEANUP")
        logger.info("="*100)
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Skip S3 Cleanup: {skip_s3}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        cleanup_start_time = time.time()
        cleanup_results = {}
        
        try:
            # Step 7 (Reverse): Clean up integration test artifacts
            logger.info("\n" + "="*60)
            logger.info("STEP 1: CLEANING UP INTEGRATION TEST ARTIFACTS")
            logger.info("="*60)
            
            test_cleanup = self._cleanup_integration_tests(confirm)
            cleanup_results['integration_tests_cleanup'] = test_cleanup
            
            # Step 6 (Reverse): Clean up deployment validation resources
            logger.info("\n" + "="*60)
            logger.info("STEP 2: CLEANING UP DEPLOYMENT VALIDATION RESOURCES")
            logger.info("="*60)
            
            validation_cleanup = self._cleanup_deployment_validation(confirm)
            cleanup_results['deployment_validation_cleanup'] = validation_cleanup
            
            # Step 5 (Reverse): Delete EventBridge rules and targets
            logger.info("\n" + "="*60)
            logger.info("STEP 3: DELETING EVENTBRIDGE RULES AND TARGETS")
            logger.info("="*60)
            
            eventbridge_cleanup = self._cleanup_eventbridge_rules(confirm)
            cleanup_results['eventbridge_cleanup'] = eventbridge_cleanup
            
            # Step 4 (Reverse): Delete ECR repositories and container images
            logger.info("\n" + "="*60)
            logger.info("STEP 4: DELETING ECR REPOSITORIES AND IMAGES")
            logger.info("="*60)
            
            ecr_cleanup = self._cleanup_ecr_repositories(confirm)
            cleanup_results['ecr_cleanup'] = ecr_cleanup
            
            # Step 3 (Reverse): Delete Step Functions state machines
            logger.info("\n" + "="*60)
            logger.info("STEP 5: DELETING STEP FUNCTIONS STATE MACHINES")
            logger.info("="*60)
            
            stepfunctions_cleanup = self._cleanup_step_functions(confirm)
            cleanup_results['stepfunctions_cleanup'] = stepfunctions_cleanup
            
            # Step 2 (Reverse): Delete Lambda functions and layers
            logger.info("\n" + "="*60)
            logger.info("STEP 6: DELETING LAMBDA FUNCTIONS AND LAYERS")
            logger.info("="*60)
            
            lambda_cleanup = self._cleanup_lambda_functions(confirm)
            cleanup_results['lambda_cleanup'] = lambda_cleanup
            
            # Step 1 (Reverse): Clean up S3 artifacts (optional)
            logger.info("\n" + "="*60)
            logger.info("STEP 7: CLEANING UP S3 ARTIFACTS")
            logger.info("="*60)
            
            if skip_s3:
                logger.info("Skipping S3 cleanup (preserving buckets and data)")
                cleanup_results['s3_cleanup'] = {'status': 'skipped', 'reason': 'skip_s3=True'}
            else:
                s3_cleanup = self._cleanup_s3_artifacts(confirm)
                cleanup_results['s3_cleanup'] = s3_cleanup
            
            # Clean up SageMaker endpoints (if any are running)
            logger.info("\n" + "="*60)
            logger.info("STEP 8: CLEANING UP SAGEMAKER ENDPOINTS")
            logger.info("="*60)
            
            sagemaker_cleanup = self._cleanup_sagemaker_resources(confirm)
            cleanup_results['sagemaker_cleanup'] = sagemaker_cleanup
            
            # Generate cleanup summary
            cleanup_time = time.time() - cleanup_start_time
            summary = self._generate_cleanup_summary(cleanup_results, cleanup_time, confirm)
            
            logger.info(f"CLEANUP RESULTS: {cleanup_results}")
            logger.info(f"CLEANUP SUMMARY: {summary}")
            
            logger.info("\n" + "="*100)
            logger.info("ENHANCED MLOPS CLEANUP COMPLETED")
            logger.info("="*100)
            logger.info(f"Total cleanup time: {cleanup_time / 60:.2f} minutes")
            logger.info(f"Mode: {'ACTUAL DELETION' if confirm else 'DRY RUN'}")
            
            return {'cleanup_summary': summary, 'cleanup_results': cleanup_results}
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {
                'cleanup_failed': True,
                'error': str(e),
                'partial_cleanup_results': cleanup_results
            }

    def _cleanup_integration_tests(self, confirm: bool) -> Dict[str, Any]:
        """Clean up integration test artifacts"""
        
        try:
            results = {'status': 'success', 'cleaned_items': []}
            
            # Integration tests don't typically leave persistent resources
            # but we can clean up any test executions or logs if needed
            
            logger.info("Integration test cleanup completed (no persistent resources)")
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_deployment_validation(self, confirm: bool) -> Dict[str, Any]:
        """Clean up deployment validation resources"""
        
        try:
            results = {'status': 'success', 'cleaned_items': []}
            
            # Deployment validation doesn't create persistent resources
            logger.info("Deployment validation cleanup completed (no persistent resources)")
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_eventbridge_rules(self, confirm: bool) -> Dict[str, Any]:
        """Delete EventBridge rules and targets"""
        
        try:
            results = {'status': 'success', 'deleted_rules': [], 'errors': []}
            
            for rule_name in self.config['eventbridge_rules']:
                try:
                    # Check if rule exists
                    try:
                        rule_info = self.events_client.describe_rule(Name=rule_name)
                        logger.info(f"Found EventBridge rule: {rule_name}")
                        
                        if confirm:
                            # Remove targets first
                            try:
                                targets = self.events_client.list_targets_by_rule(Rule=rule_name)
                                if targets['Targets']:
                                    target_ids = [target['Id'] for target in targets['Targets']]
                                    self.events_client.remove_targets(Rule=rule_name, Ids=target_ids)
                                    logger.info(f"  Removed {len(target_ids)} targets from {rule_name}")
                            except Exception as e:
                                logger.warning(f"  Could not remove targets from {rule_name}: {str(e)}")
                            
                            # Delete the rule
                            self.events_client.delete_rule(Name=rule_name)
                            logger.info(f"  ✓ Deleted EventBridge rule: {rule_name}")
                            results['deleted_rules'].append(rule_name)
                        else:
                            logger.info(f"  Would delete EventBridge rule: {rule_name}")
                            results['deleted_rules'].append(f"{rule_name} (dry-run)")
                    
                    except self.events_client.exceptions.ResourceNotFoundException:
                        logger.info(f"  EventBridge rule not found: {rule_name}")
                
                except Exception as e:
                    error_msg = f"Error processing EventBridge rule {rule_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_ecr_repositories(self, confirm: bool) -> Dict[str, Any]:
        """Delete ECR repositories and container images"""
        
        try:
            results = {'status': 'success', 'deleted_repositories': [], 'errors': []}
            
            for repo_name in self.config['ecr_repositories']:
                try:
                    # Check if repository exists
                    try:
                        repo_info = self.ecr_client.describe_repositories(repositoryNames=[repo_name])
                        logger.info(f"Found ECR repository: {repo_name}")
                        
                        if confirm:
                            # Delete all images in the repository first
                            try:
                                images = self.ecr_client.describe_images(repositoryName=repo_name)
                                if images['imageDetails']:
                                    image_ids = [{'imageDigest': img['imageDigest']} for img in images['imageDetails']]
                                    self.ecr_client.batch_delete_image(repositoryName=repo_name, imageIds=image_ids)
                                    logger.info(f"  Deleted {len(image_ids)} images from {repo_name}")
                            except Exception as e:
                                logger.warning(f"  Could not delete images from {repo_name}: {str(e)}")
                            
                            # Delete the repository
                            self.ecr_client.delete_repository(repositoryName=repo_name, force=True)
                            logger.info(f"  ✓ Deleted ECR repository: {repo_name}")
                            results['deleted_repositories'].append(repo_name)
                        else:
                            logger.info(f"  Would delete ECR repository: {repo_name}")
                            results['deleted_repositories'].append(f"{repo_name} (dry-run)")
                    
                    except self.ecr_client.exceptions.RepositoryNotFoundException:
                        logger.info(f"  ECR repository not found: {repo_name}")
                
                except Exception as e:
                    error_msg = f"Error processing ECR repository {repo_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_step_functions(self, confirm: bool) -> Dict[str, Any]:
        """Delete Step Functions state machines"""
        
        try:
            results = {'status': 'success', 'deleted_state_machines': [], 'errors': []}
            
            for sm_name in self.config['step_functions']:
                try:
                    # Check if state machine exists
                    try:
                        sm_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sm_name}"
                        sm_info = self.stepfunctions_client.describe_state_machine(stateMachineArn=sm_arn)
                        logger.info(f"Found Step Functions state machine: {sm_name}")
                        
                        if confirm:
                            # Stop any running executions first
                            try:
                                executions = self.stepfunctions_client.list_executions(
                                    stateMachineArn=sm_arn,
                                    statusFilter='RUNNING',
                                    maxResults=100
                                )
                                for execution in executions['executions']:
                                    try:
                                        self.stepfunctions_client.stop_execution(
                                            executionArn=execution['executionArn']
                                        )
                                        logger.info(f"  Stopped execution: {execution['name']}")
                                    except Exception as e:
                                        logger.warning(f"  Could not stop execution {execution['name']}: {str(e)}")
                            except Exception as e:
                                logger.warning(f"  Could not list/stop executions for {sm_name}: {str(e)}")
                            
                            # Wait a moment for executions to stop
                            time.sleep(2)
                            
                            # Delete the state machine
                            self.stepfunctions_client.delete_state_machine(stateMachineArn=sm_arn)
                            logger.info(f"  ✓ Deleted Step Functions state machine: {sm_name}")
                            results['deleted_state_machines'].append(sm_name)
                        else:
                            logger.info(f"  Would delete Step Functions state machine: {sm_name}")
                            results['deleted_state_machines'].append(f"{sm_name} (dry-run)")
                    
                    except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                        logger.info(f"  Step Functions state machine not found: {sm_name}")
                
                except Exception as e:
                    error_msg = f"Error processing Step Functions state machine {sm_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_lambda_functions(self, confirm: bool) -> Dict[str, Any]:
        """Delete Lambda functions and layers"""
        
        try:
            results = {
                'status': 'success', 
                'deleted_functions': [], 
                'deleted_layers': [], 
                'errors': []
            }
            
            # Delete Lambda functions
            for func_name in self.config['lambda_functions']:
                try:
                    # Check if function exists
                    try:
                        func_info = self.lambda_client.get_function(FunctionName=func_name)
                        logger.info(f"Found Lambda function: {func_name}")
                        
                        if confirm:
                            # Delete the function
                            self.lambda_client.delete_function(FunctionName=func_name)
                            logger.info(f"  ✓ Deleted Lambda function: {func_name}")
                            results['deleted_functions'].append(func_name)
                        else:
                            logger.info(f"  Would delete Lambda function: {func_name}")
                            results['deleted_functions'].append(f"{func_name} (dry-run)")
                    
                    except self.lambda_client.exceptions.ResourceNotFoundException:
                        logger.info(f"  Lambda function not found: {func_name}")
                
                except Exception as e:
                    error_msg = f"Error processing Lambda function {func_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Delete Lambda layers
            for layer_name in self.config['lambda_layers']:
                try:
                    # List layer versions
                    try:
                        layer_versions = self.lambda_client.list_layer_versions(LayerName=layer_name)
                        
                        if layer_versions['LayerVersions']:
                            logger.info(f"Found Lambda layer: {layer_name} with {len(layer_versions['LayerVersions'])} versions")
                            
                            if confirm:
                                # Delete all versions of the layer
                                for version_info in layer_versions['LayerVersions']:
                                    version_number = version_info['Version']
                                    try:
                                        self.lambda_client.delete_layer_version(
                                            LayerName=layer_name,
                                            VersionNumber=version_number
                                        )
                                        logger.info(f"  ✓ Deleted layer version {layer_name}:{version_number}")
                                    except Exception as e:
                                        logger.warning(f"  Could not delete layer version {layer_name}:{version_number}: {str(e)}")
                                
                                results['deleted_layers'].append(layer_name)
                            else:
                                logger.info(f"  Would delete Lambda layer: {layer_name} (all versions)")
                                results['deleted_layers'].append(f"{layer_name} (dry-run)")
                        else:
                            logger.info(f"  Lambda layer has no versions: {layer_name}")
                    
                    except self.lambda_client.exceptions.ResourceNotFoundException:
                        logger.info(f"  Lambda layer not found: {layer_name}")
                
                except Exception as e:
                    error_msg = f"Error processing Lambda layer {layer_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_s3_artifacts(self, confirm: bool) -> Dict[str, Any]:
        """Clean up S3 artifacts (preserves buckets by default)"""
        
        try:
            results = {
                'status': 'success', 
                'cleaned_prefixes': [], 
                'preserved_buckets': [],
                'errors': []
            }
            
            buckets_to_clean = [self.config['data_bucket'], self.config['model_bucket']]
            
            for bucket_name in buckets_to_clean:
                try:
                    # Check if bucket exists
                    try:
                        self.s3_client.head_bucket(Bucket=bucket_name)
                        logger.info(f"Found S3 bucket: {bucket_name}")
                        
                        # Define prefixes to clean (ML artifacts, not source data)
                        cleanup_prefixes = [
                            'models/endpoint-configs/',
                            'models/temp/',
                            'predictions/temp/',
                            'executions/',
                            'lambda-deployments/',
                            'mlops-artifacts/'
                        ]
                        
                        for prefix in cleanup_prefixes:
                            try:
                                # List objects with this prefix
                                objects = self.s3_client.list_objects_v2(
                                    Bucket=bucket_name, 
                                    Prefix=prefix
                                )
                                
                                if 'Contents' in objects:
                                    logger.info(f"  Found {len(objects['Contents'])} objects under {bucket_name}/{prefix}")
                                    
                                    if confirm:
                                        # Delete objects
                                        delete_keys = [{'Key': obj['Key']} for obj in objects['Contents']]
                                        
                                        # Delete in batches of 1000 (AWS limit)
                                        for i in range(0, len(delete_keys), 1000):
                                            batch = delete_keys[i:i+1000]
                                            self.s3_client.delete_objects(
                                                Bucket=bucket_name,
                                                Delete={'Objects': batch}
                                            )
                                        
                                        logger.info(f"  ✓ Cleaned {len(delete_keys)} objects from {bucket_name}/{prefix}")
                                        results['cleaned_prefixes'].append(f"{bucket_name}/{prefix}")
                                    else:
                                        logger.info(f"  Would clean {len(objects['Contents'])} objects from {bucket_name}/{prefix}")
                                        results['cleaned_prefixes'].append(f"{bucket_name}/{prefix} (dry-run)")
                                else:
                                    logger.info(f"  No objects found under {bucket_name}/{prefix}")
                            
                            except Exception as e:
                                error_msg = f"Error cleaning {bucket_name}/{prefix}: {str(e)}"
                                logger.warning(error_msg)
                                results['errors'].append(error_msg)
                        
                        results['preserved_buckets'].append(bucket_name)
                        logger.info(f"  ✓ Preserved bucket: {bucket_name}")
                    
                    except Exception as e:
                        logger.info(f"  S3 bucket not accessible: {bucket_name} ({str(e)})")
                
                except Exception as e:
                    error_msg = f"Error processing S3 bucket {bucket_name}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _cleanup_sagemaker_resources(self, confirm: bool) -> Dict[str, Any]:
        """Clean up any running SageMaker endpoints"""
        
        try:
            results = {
                'status': 'success',
                'deleted_endpoints': [],
                'deleted_endpoint_configs': [],
                'deleted_models': [],
                'errors': []
            }
            
            # Find energy forecasting endpoints
            try:
                endpoints = self.sagemaker_client.list_endpoints()
                energy_endpoints = [ep for ep in endpoints['Endpoints'] 
                                 if 'energy-forecasting' in ep['EndpointName']]
                
                for endpoint_info in energy_endpoints:
                    endpoint_name = endpoint_info['EndpointName']
                    logger.info(f"Found SageMaker endpoint: {endpoint_name}")
                    
                    if confirm:
                        try:
                            # Get endpoint config name before deleting endpoint
                            endpoint_details = self.sagemaker_client.describe_endpoint(
                                EndpointName=endpoint_name
                            )
                            endpoint_config_name = endpoint_details['EndpointConfigName']
                            
                            # Delete endpoint
                            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
                            logger.info(f"  ✓ Deleted endpoint: {endpoint_name}")
                            results['deleted_endpoints'].append(endpoint_name)
                            
                            # Delete endpoint config
                            try:
                                self.sagemaker_client.delete_endpoint_config(
                                    EndpointConfigName=endpoint_config_name
                                )
                                logger.info(f"  ✓ Deleted endpoint config: {endpoint_config_name}")
                                results['deleted_endpoint_configs'].append(endpoint_config_name)
                            except Exception as e:
                                logger.warning(f"  Could not delete endpoint config {endpoint_config_name}: {str(e)}")
                            
                        except Exception as e:
                            error_msg = f"Error deleting endpoint {endpoint_name}: {str(e)}"
                            logger.error(error_msg)
                            results['errors'].append(error_msg)
                    else:
                        logger.info(f"  Would delete SageMaker endpoint: {endpoint_name}")
                        results['deleted_endpoints'].append(f"{endpoint_name} (dry-run)")
                
                # Clean up models
                models = self.sagemaker_client.list_models()
                energy_models = [model for model in models['Models'] 
                               if 'energy-forecasting' in model['ModelName']]
                
                for model_info in energy_models:
                    model_name = model_info['ModelName']
                    logger.info(f"Found SageMaker model: {model_name}")
                    
                    if confirm:
                        try:
                            self.sagemaker_client.delete_model(ModelName=model_name)
                            logger.info(f"  ✓ Deleted model: {model_name}")
                            results['deleted_models'].append(model_name)
                        except Exception as e:
                            error_msg = f"Error deleting model {model_name}: {str(e)}"
                            logger.error(error_msg)
                            results['errors'].append(error_msg)
                    else:
                        logger.info(f"  Would delete SageMaker model: {model_name}")
                        results['deleted_models'].append(f"{model_name} (dry-run)")
            
            except Exception as e:
                error_msg = f"Error listing SageMaker resources: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            if results['errors']:
                results['status'] = 'partial'
            
            return results
            
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def _generate_cleanup_summary(self, cleanup_results: Dict, cleanup_time: float, confirm: bool) -> Dict[str, Any]:
        """Generate comprehensive cleanup summary"""
        
        try:
            successful_components = sum(1 for result in cleanup_results.values() 
                                      if isinstance(result, dict) and result.get('status') == 'success')
            total_components = len(cleanup_results)
            
            # Count deleted resources
            deleted_resources = {}
            for component, result in cleanup_results.items():
                if isinstance(result, dict) and result.get('status') in ['success', 'partial']:
                    for key, value in result.items():
                        if 'deleted' in key and isinstance(value, list):
                            deleted_resources[key] = len(value)
            
            summary = {
                'cleanup_mode': 'ACTUAL_DELETION' if confirm else 'DRY_RUN',
                'overall_success': successful_components == total_components,
                'cleanup_time_minutes': cleanup_time / 60,
                'successful_components': successful_components,
                'total_components': total_components,
                'success_rate': (successful_components / total_components * 100) if total_components > 0 else 0,
                'cleanup_timestamp': datetime.now().isoformat(),
                'resources_deleted': deleted_resources,
                'environment': self.environment,
                'region': self.region
            }
            
            return summary
            
        except Exception as e:
            return {
                'summary_generation_error': str(e),
                'partial_cleanup_results': cleanup_results
            }

    def get_resource_inventory(self) -> Dict[str, Any]:
        """Get inventory of all resources that would be affected by cleanup"""
        
        inventory = {
            'lambda_functions': [],
            'step_functions': [],
            'eventbridge_rules': [],
            'ecr_repositories': [],
            'sagemaker_endpoints': [],
            'sagemaker_models': [],
            's3_artifact_prefixes': []
        }
        
        try:
            # Check Lambda functions
            for func_name in self.config['lambda_functions']:
                try:
                    func_info = self.lambda_client.get_function(FunctionName=func_name)
                    inventory['lambda_functions'].append({
                        'name': func_name,
                        'status': func_info['Configuration']['State'],
                        'last_modified': func_info['Configuration']['LastModified']
                    })
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    pass
            
            # Check Step Functions
            for sm_name in self.config['step_functions']:
                try:
                    sm_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sm_name}"
                    sm_info = self.stepfunctions_client.describe_state_machine(stateMachineArn=sm_arn)
                    inventory['step_functions'].append({
                        'name': sm_name,
                        'status': sm_info['status'],
                        'creation_date': sm_info['creationDate'].isoformat()
                    })
                except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                    pass
            
            # Check EventBridge rules
            for rule_name in self.config['eventbridge_rules']:
                try:
                    rule_info = self.events_client.describe_rule(Name=rule_name)
                    inventory['eventbridge_rules'].append({
                        'name': rule_name,
                        'state': rule_info['State'],
                        'schedule': rule_info.get('ScheduleExpression', 'None')
                    })
                except self.events_client.exceptions.ResourceNotFoundException:
                    pass
            
            # Check ECR repositories
            for repo_name in self.config['ecr_repositories']:
                try:
                    repo_info = self.ecr_client.describe_repositories(repositoryNames=[repo_name])
                    if repo_info['repositories']:
                        repo = repo_info['repositories'][0]
                        inventory['ecr_repositories'].append({
                            'name': repo_name,
                            'uri': repo['repositoryUri'],
                            'created': repo['createdAt'].isoformat()
                        })
                except self.ecr_client.exceptions.RepositoryNotFoundException:
                    pass
            
            # Check SageMaker endpoints
            endpoints = self.sagemaker_client.list_endpoints()
            energy_endpoints = [ep for ep in endpoints['Endpoints'] 
                             if 'energy-forecasting' in ep['EndpointName']]
            for ep in energy_endpoints:
                inventory['sagemaker_endpoints'].append({
                    'name': ep['EndpointName'],
                    'status': ep['EndpointStatus'],
                    'creation_time': ep['CreationTime'].isoformat()
                })
            
            # Check SageMaker models
            models = self.sagemaker_client.list_models()
            energy_models = [model for model in models['Models'] 
                           if 'energy-forecasting' in model['ModelName']]
            for model in energy_models:
                inventory['sagemaker_models'].append({
                    'name': model['ModelName'],
                    'creation_time': model['CreationTime'].isoformat()
                })
            
            # S3 prefixes that would be cleaned
            inventory['s3_artifact_prefixes'] = [
                f"{self.config['data_bucket']}/models/endpoint-configs/",
                f"{self.config['data_bucket']}/models/temp/",
                f"{self.config['data_bucket']}/predictions/temp/",
                f"{self.config['data_bucket']}/executions/",
                f"{self.config['model_bucket']}/lambda-deployments/",
                f"{self.config['model_bucket']}/mlops-artifacts/"
            ]
            
        except Exception as e:
            inventory['error'] = str(e)
        
        return inventory


def main():
    """Main cleanup function"""
    
    parser = argparse.ArgumentParser(
        description='Complete MLOps Pipeline Cleanup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be deleted
  python deployment/cleanup_enhanced_mlops.py --dry-run
  
  # Actually delete resources (CAUTION!)
  python deployment/cleanup_enhanced_mlops.py --confirm
  
  # Delete resources but preserve S3 data
  python deployment/cleanup_enhanced_mlops.py --confirm --preserve-s3
  
  # Just show inventory of resources
  python deployment/cleanup_enhanced_mlops.py --inventory-only
  
  # Clean up specific environment
  python deployment/cleanup_enhanced_mlops.py --confirm --environment prod
        """
    )
    
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--environment', default='dev', help='Environment (dev, staging, prod)')
    parser.add_argument('--confirm', action='store_true', 
                       help='Actually delete resources (required for deletion)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be deleted without deleting')
    parser.add_argument('--preserve-s3', action='store_true', 
                       help='Skip S3 cleanup (recommended)')
    parser.add_argument('--inventory-only', action='store_true', 
                       help='Just show inventory of resources')
    parser.add_argument('--force', action='store_true', 
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    # Default to dry-run if not confirmed
    if not args.confirm and not args.dry_run and not args.inventory_only:
        args.dry_run = True
        logger.warning("No --confirm flag provided, running in dry-run mode")
    
    cleanup_manager = EnhancedMLOpsCleanup(
        region=args.region,
        environment=args.environment
    )
    
    if args.inventory_only:
        logger.info("Getting resource inventory...")
        inventory = cleanup_manager.get_resource_inventory()
        
        print("\n" + "="*80)
        print("RESOURCE INVENTORY")
        print("="*80)
        
        for resource_type, resources in inventory.items():
            if isinstance(resources, list) and resources:
                print(f"\n{resource_type.upper().replace('_', ' ')} ({len(resources)}):")
                for resource in resources:
                    if isinstance(resource, dict):
                        print(f"  • {resource.get('name', 'Unknown')}")
                        for key, value in resource.items():
                            if key != 'name':
                                print(f"    {key}: {value}")
                    else:
                        print(f"  • {resource}")
            elif isinstance(resources, list):
                print(f"\n{resource_type.upper().replace('_', ' ')}: None found")
        
        if 'error' in inventory:
            print(f"\nError getting inventory: {inventory['error']}")
        
        return
    
    # Confirmation prompt for actual deletion
    if args.confirm and not args.force:
        print("\n" + "="*80)
        print("⚠️  WARNING: DESTRUCTIVE OPERATION")
        print("="*80)
        print("This will DELETE the following types of resources:")
        print("• Lambda functions (11 functions)")
        print("• Step Functions state machines (2-3 machines)")  
        print("• EventBridge rules and targets (2-4 rules)")
        print("• ECR repositories and images (3 repositories)")
        print("• SageMaker endpoints, configs, and models")
        if not args.preserve_s3:
            print("• S3 artifacts (NOT source data)")
        print(f"\nEnvironment: {args.environment}")
        print(f"Region: {args.region}")
        
        response = input("\nType 'DELETE' to confirm resource deletion: ")
        if response != 'DELETE':
            print("Cleanup cancelled.")
            return
    
    # Run cleanup
    skip_s3 = args.preserve_s3 or not args.confirm
    
    logger.info("Starting MLOps pipeline cleanup...")
    result = cleanup_manager.cleanup_complete_mlops_pipeline(
        skip_s3=skip_s3, 
        confirm=args.confirm
    )
    
    # Print final summary
    summary = result.get('cleanup_summary', {})
    
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"Mode: {summary.get('cleanup_mode', 'Unknown')}")
    print(f"Overall Success: {summary.get('overall_success', False)}")
    print(f"Components Cleaned: {summary.get('successful_components', 0)}/{summary.get('total_components', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    print(f"Cleanup Time: {summary.get('cleanup_time_minutes', 0):.2f} minutes")
    
    resources_deleted = summary.get('resources_deleted', {})
    if resources_deleted:
        print("\nResources Affected:")
        for resource_type, count in resources_deleted.items():
            if count > 0:
                print(f"• {resource_type.replace('_', ' ').title()}: {count}")
    
    if args.confirm:
        print(f"\n✅ Cleanup completed for {args.environment} environment")
        print("You can now run deploy_enhanced_mlops.py from scratch")
    else:
        print(f"\n📋 Dry run completed for {args.environment} environment")
        print("Use --confirm flag to actually delete resources")


if __name__ == "__main__":
    main()
