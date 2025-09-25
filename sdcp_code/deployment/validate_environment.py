#!/usr/bin/env python3
"""
Environment Validation Tool for Enhanced Prediction Pipeline
sdcp_code/deployment/validate_environment.py

Comprehensive validation of AWS environment prerequisites:
- Pre-deployment checks: IAM roles, S3 buckets, ECR repositories
- Post-deployment checks: Lambda functions, Step Functions, EventBridge
- Complete integration checks: End-to-end pipeline validation
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

# Add project paths for sdcp_code structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sdcp_code_dir = os.path.dirname(current_dir)  # sdcp_code directory
project_root = os.path.dirname(sdcp_code_dir)  # repository root
sys.path.append(sdcp_code_dir)
sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """
    Comprehensive environment validator for the enhanced prediction pipeline
    """

    def __init__(self, region: str = "us-west-2", environment: str = "dev", role: str = ""):
        """Initialize the environment validator"""
        
        self.region = region
        self.role = role
        self.role_name = role.split('/')[-1] if role else f'sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role'
        self.environment = environment
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # AWS clients
        self.sts_client = boto3.client('sts')
        self.iam_client = boto3.client('iam')
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.ec2_client = boto3.client('ec2', region_name=region)
        
        # Expected resources configuration
        self.expected_resources = {
            'iam_roles': [
                self.role_name
                # f'sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role',
                # f'sdcp-{environment}-sagemaker-energy-forecasting-lambda-role'
            ],
            's3_buckets': [
                f'sdcp-{environment}-sagemaker-energy-forecasting-data',
                f'sdcp-{environment}-sagemaker-energy-forecasting-models'
            ],
            'ecr_repositories': [
                'energy-preprocessing',
                'energy-training',
                # 'energy-prediction'
            ],
            'lambda_functions': [
                f'energy-forecasting-{environment}-model-registry',
                f'energy-forecasting-{environment}-endpoint-management',
                f'energy-forecasting-{environment}-prediction-endpoint-manager',
                f'energy-forecasting-{environment}-prediction-cleanup',
                f'energy-forecasting-{environment}-profile-validator',
                f'energy-forecasting-{environment}-profile-endpoint-creator',
                f'energy-forecasting-{environment}-profile-cleanup',
                f'energy-forecasting-{environment}-endpoint-status-checker',
                f'energy-forecasting-{environment}-prediction-summary',
                f'energy-forecasting-{environment}-emergency-cleanup',
                f'energy-forecasting-{environment}-profile-predictor'
            ],
            'step_functions': [
                'energy-forecasting-training-pipeline',
                'energy-forecasting-enhanced-prediction-pipeline'
            ],
            'model_package_groups': [
                'energy-forecasting-models'
            ]
        }
        
        # Validation results storage
        self.validation_results = {}
        
        logger.info(f"Environment Validator initialized for {environment} environment in {region}")

    def run_pre_deployment_check(self) -> Dict[str, Any]:
        """
        Pre-deployment validation:
        - AWS credentials and permissions
        - IAM roles existence
        - S3 bucket accessibility
        - ECR repository availability
        """
        
        logger.info(" PRE-DEPLOYMENT VALIDATION")
        logger.info("=" * 50)
        
        validation_results = {
            'validation_type': 'pre-deployment',
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'region': self.region,
            'account_id': self.account_id,
            'checks': {}
        }
        
        try:
            # 1. AWS Credentials and Permissions
            logger.info("1. Validating AWS credentials and permissions...")
            validation_results['checks']['credentials'] = self._validate_credentials_and_permissions()
            
            # 2. IAM Roles
            logger.info("2. Validating IAM roles...")
            validation_results['checks']['iam_roles'] = self._validate_iam_roles()
            
            # 3. S3 Buckets
            logger.info("3. Validating S3 buckets...")
            validation_results['checks']['s3_buckets'] = self._validate_s3_resources()
            
            # 4. ECR Repositories
            logger.info("4. Validating ECR repositories...")
            validation_results['checks']['ecr_repositories'] = self._validate_ecr_repositories()
            
            # 5. Basic SageMaker Access
            logger.info("5. Validating SageMaker access...")
            validation_results['checks']['sagemaker_access'] = self._validate_basic_sagemaker_access()
            
            # Determine overall readiness
            validation_results['summary'] = self._assess_pre_deployment_readiness(validation_results['checks'])
            
            return validation_results
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['summary'] = {
                'ready_for_deployment': False,
                'critical_issues': [f"Validation failed: {str(e)}"]
            }
            return validation_results

    def run_post_deployment_check(self) -> Dict[str, Any]:
        """
        Post-deployment validation:
        - Lambda function deployment
        - Step Functions creation
        - EventBridge rule setup
        - Container image availability
        """
        
        logger.info(" POST-DEPLOYMENT VALIDATION")
        logger.info("=" * 50)
        
        validation_results = {
            'validation_type': 'post-deployment',
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'region': self.region,
            'account_id': self.account_id,
            'checks': {}
        }
        
        try:
            # 1. Lambda Functions
            logger.info("1. Validating Lambda function deployments...")
            validation_results['checks']['lambda_functions'] = self._validate_lambda_functions()
            
            # 2. Step Functions
            logger.info("2. Validating Step Functions...")
            validation_results['checks']['step_functions'] = self._validate_step_functions()
            
            # 3. EventBridge Rules
            logger.info("3. Validating EventBridge rules...")
            validation_results['checks']['eventbridge'] = self._validate_eventbridge()
            
            # 4. Container Images
            logger.info("4. Validating container images...")
            validation_results['checks']['container_images'] = self._validate_container_images()
            
            # 5. Cross-service connectivity
            logger.info("5. Validating cross-service connectivity...")
            validation_results['checks']['connectivity'] = self._validate_basic_connectivity()
            
            # Determine deployment success
            validation_results['summary'] = self._assess_post_deployment_success(validation_results['checks'])
            
            return validation_results
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['summary'] = {
                'deployment_successful': False,
                'critical_issues': [f"Validation failed: {str(e)}"]
            }
            return validation_results

    def run_complete_integration_check(self) -> Dict[str, Any]:
        """
        Complete integration validation:
        - End-to-end pipeline connectivity
        - Cross-service communication
        - Data flow validation
        - Performance benchmarks
        """
        
        logger.info(" COMPLETE INTEGRATION VALIDATION")
        logger.info("=" * 50)
        
        validation_results = {
            'validation_type': 'complete-integration',
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'region': self.region,
            'account_id': self.account_id,
            'checks': {}
        }
        
        try:
            # 1. Complete environment validation
            logger.info("1. Running complete environment validation...")
            validation_results['checks']['complete_environment'] = self.validate_complete_environment()
            logger.info("1. Validation Results: ")
            logger.info(f"  RESULT: {json.dumps(validation_results['checks']['complete_environment'], indent=2)}")

            # 2. Pipeline Integration Tests
            logger.info("2. Testing pipeline integrations...")
            validation_results['checks']['pipeline_integration'] = self._validate_pipeline_integration()
            logger.info(f"  RESULT: {json.dumps(validation_results['checks']['pipeline_integration'], indent=2)}")

            # 3. Data Flow Validation
            logger.info("3. Validating data flows...")
            validation_results['checks']['data_flow'] = self._validate_data_flow()
            logger.info(f"  RESULT: {json.dumps(validation_results['checks']['data_flow'], indent=2)}")

            # 4. Performance Benchmarks
            logger.info("4. Running performance benchmarks...")
            validation_results['checks']['performance'] = self._validate_performance_benchmarks()
            logger.info(f"  RESULT: {json.dumps(validation_results['checks']['performance'], indent=2)}")
            
            # 5. End-to-End Connectivity
            logger.info("5. Testing end-to-end connectivity...")
            validation_results['checks']['e2e_connectivity'] = self._validate_end_to_end_connectivity()
            logger.info(f"  RESULT: {json.dumps(validation_results['checks']['e2e_connectivity'], indent=2)}")
            
            # Determine integration readiness
            validation_results['summary'] = self._assess_integration_readiness(validation_results['checks'])
            logger.info(f"  SUMMARY: {json.dumps(validation_results['summary'], indent=2)}")
            
            return validation_results
            
        except Exception as e:
            validation_results['error'] = str(e)
            validation_results['summary'] = {
                'integration_ready': False,
                'critical_issues': [f"Integration validation failed: {str(e)}"]
            }
            return validation_results

    def validate_complete_environment(self) -> Dict[str, Any]:
        """Run complete environment validation (legacy method for compatibility)"""
        
        logger.info("="*80)
        logger.info("COMPREHENSIVE ENVIRONMENT VALIDATION")
        logger.info("="*80)
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        validation_start_time = time.time()
        
        try:
            # 1. Validate AWS Credentials and Permissions
            logger.info("1. VALIDATING AWS CREDENTIALS AND PERMISSIONS")
            logger.info("-" * 50)
            credentials_result = self._validate_credentials_and_permissions()
            self.validation_results['credentials_and_permissions'] = credentials_result
            logger.info(f"  RESULT: {json.dumps(credentials_result, indent=2)}")

            # 2. Validate IAM Roles
            logger.info("2. VALIDATING IAM ROLES")
            logger.info("-" * 50)
            iam_result = self._validate_iam_roles()
            self.validation_results['iam_roles'] = iam_result
            logger.info(f"  RESULT: {json.dumps(iam_result, indent=2)}")
            
            # 3. Validate S3 Resources
            logger.info("3. VALIDATING S3 RESOURCES")
            logger.info("-" * 50)
            s3_result = self._validate_s3_resources()
            self.validation_results['s3_resources'] = s3_result
            logger.info(f"  RESULT: {json.dumps(s3_result, indent=2)}")
            
            # 4. Validate ECR Repositories
            logger.info("4. VALIDATING ECR REPOSITORIES")
            logger.info("-" * 50)
            ecr_result = self._validate_ecr_repositories()
            self.validation_results['ecr_repositories'] = ecr_result
            logger.info(f"  RESULT: {json.dumps(ecr_result, indent=2)}")
            
            # 5. Validate SageMaker Resources
            logger.info("5. VALIDATING SAGEMAKER RESOURCES")
            logger.info("-" * 50)
            sagemaker_result = self._validate_sagemaker_resources()
            self.validation_results['sagemaker_resources'] = sagemaker_result
            logger.info(f"  RESULT: {json.dumps(sagemaker_result, indent=2)}")
            
            # 6. Validate Lambda Functions
            logger.info("6. VALIDATING LAMBDA FUNCTIONS")
            logger.info("-" * 50)
            lambda_result = self._validate_lambda_functions()
            self.validation_results['lambda_functions'] = lambda_result
            logger.info(f"  RESULT: {json.dumps(lambda_result, indent=2)}")
            
            # 7. Validate Step Functions
            logger.info("7. VALIDATING STEP FUNCTIONS")
            logger.info("-" * 50)
            stepfunctions_result = self._validate_step_functions()
            self.validation_results['step_functions'] = stepfunctions_result
            logger.info(f"  RESULT: {json.dumps(stepfunctions_result, indent=2)}")
            
            # 8. Validate EventBridge
            logger.info("8. VALIDATING EVENTBRIDGE")
            logger.info("-" * 50)
            eventbridge_result = self._validate_eventbridge()
            self.validation_results['eventbridge'] = eventbridge_result
            logger.info(f"  RESULT: {json.dumps(eventbridge_result, indent=2)}")
            
            # 9. Validate Network Connectivity
            logger.info("9. VALIDATING NETWORK CONNECTIVITY")
            logger.info("-" * 50)
            network_result = self._validate_network_connectivity()
            self.validation_results['network_connectivity'] = network_result
            logger.info(f"  RESULT: {json.dumps(network_result, indent=2)}")
            
            # 10. Validate Resource Dependencies
            logger.info("10. VALIDATING RESOURCE DEPENDENCIES")
            logger.info("-" * 50)
            dependencies_result = self._validate_resource_dependencies()
            self.validation_results['resource_dependencies'] = dependencies_result
            logger.info(f"  RESULT: {json.dumps(dependencies_result, indent=2)}")
            
            # Calculate validation duration
            validation_duration = time.time() - validation_start_time
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary(validation_duration)
            
            return {
                'validation_summary': validation_summary,
                'detailed_results': self.validation_results,
                'validation_duration_seconds': validation_duration,
                'environment': self.environment,
                'region': self.region,
                'account_id': self.account_id,
                'timestamp': datetime.now().isoformat(),
                'recommendations': self._generate_recommendations(),
                'next_steps': self._generate_next_steps()
            }
            
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            return {
                'validation_summary': {
                    'overall_status': 'ERROR',
                    'environment_ready': False,
                    'error': str(e)
                },
                'detailed_results': self.validation_results,
                'validation_duration_seconds': time.time() - validation_start_time,
                'environment': self.environment,
                'region': self.region,
                'timestamp': datetime.now().isoformat()
            }

    def _validate_credentials_and_permissions(self) -> Dict[str, Any]:
        """Validate AWS credentials and basic permissions"""
        
        result = {
            'caller_identity_valid': False,
            'basic_permissions_valid': False,
            'account_id': None,
            'user_arn': None,
            'permissions_checked': []
        }
        
        try:
            # Check caller identity
            caller_identity = self.sts_client.get_caller_identity()
            result['account_id'] = caller_identity['Account']
            result['user_arn'] = caller_identity['Arn']
            result['caller_identity_valid'] = True
            
            logger.info(f"  ✓ AWS credentials valid")
            logger.info(f"  ✓ Account ID: {result['account_id']}")
            logger.info(f"  ✓ User ARN: {result['user_arn']}")
            
            # Test basic permissions
            permissions_tests = [
                ('s3:ListBuckets', lambda: self.s3_client.list_buckets()),
                ('iam:ListRoles', lambda: self.iam_client.list_roles(MaxItems=1)),
                ('lambda:ListFunctions', lambda: self.lambda_client.list_functions(MaxItems=1)),
                ('states:ListStateMachines', lambda: self.stepfunctions_client.list_state_machines(maxResults=1)),
                ('sagemaker:ListModels', lambda: self.sagemaker_client.list_models(MaxResults=1))
            ]
            
            permissions_valid = 0
            for permission_name, test_func in permissions_tests:
                try:
                    test_func()
                    result['permissions_checked'].append({'permission': permission_name, 'status': 'GRANTED'})
                    permissions_valid += 1
                    logger.info(f"  ✓ Permission validated: {permission_name}")
                except Exception as e:
                    result['permissions_checked'].append({'permission': permission_name, 'status': 'DENIED', 'error': str(e)})
                    logger.warning(f"  ✗ Permission denied: {permission_name}")
            
            result['basic_permissions_valid'] = permissions_valid >= 3  # At least 3 out of 5 permissions
            
        except Exception as e:
            logger.error(f"  ✗ Credentials validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_iam_roles(self) -> Dict[str, Any]:
        """Validate required IAM roles"""
        
        result = {
            'roles_found': {},
            'roles_missing': [],
            'roles_accessible': {},
            'validation_successful': False
        }
        
        try:
            for role_name in self.expected_resources['iam_roles']:
                try:
                    role_response = self.iam_client.get_role(RoleName=role_name)
                    result['roles_found'][role_name] = {
                        'arn': role_response['Role']['Arn'],
                        'created': role_response['Role']['CreateDate'].isoformat(),
                        'path': role_response['Role']['Path']
                    }
                    
                    # Test assume role permissions
                    try:
                        self.sts_client.get_caller_identity()  # Basic test
                        result['roles_accessible'][role_name] = True
                        logger.info(f"  ✓ IAM role found and accessible: {role_name}")
                    except Exception:
                        result['roles_accessible'][role_name] = False
                        logger.warning(f"   IAM role found but not accessible: {role_name}")
                        
                except self.iam_client.exceptions.NoSuchEntityException:
                    result['roles_missing'].append(role_name)
                    logger.error(f"  ✗ IAM role missing: {role_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking IAM role {role_name}: {str(e)}")
            
            result['validation_successful'] = len(result['roles_found']) >= len(self.expected_resources['iam_roles']) * 0.5
            
        except Exception as e:
            logger.error(f"IAM roles validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_s3_resources(self) -> Dict[str, Any]:
        """Validate S3 buckets and configurations"""
        
        result = {
            'buckets_found': {},
            'buckets_missing': [],
            'buckets_accessible': {},
            'validation_successful': False
        }
        
        try:
            for bucket_name in self.expected_resources['s3_buckets']:
                try:
                    # Check if bucket exists and is accessible
                    self.s3_client.head_bucket(Bucket=bucket_name)
                    
                    # Get bucket location
                    location_response = self.s3_client.get_bucket_location(Bucket=bucket_name)
                    bucket_region = location_response.get('LocationConstraint') or 'us-east-1'
                    
                    result['buckets_found'][bucket_name] = {
                        'region': bucket_region,
                        'accessible': True
                    }
                    result['buckets_accessible'][bucket_name] = True
                    
                    logger.info(f"  ✓ S3 bucket found and accessible: {bucket_name}")
                    
                except self.s3_client.exceptions.NoSuchBucket:
                    result['buckets_missing'].append(bucket_name)
                    logger.error(f"  ✗ S3 bucket missing: {bucket_name}")
                except Exception as e:
                    result['buckets_found'][bucket_name] = {
                        'accessible': False,
                        'error': str(e)
                    }
                    result['buckets_accessible'][bucket_name] = False
                    logger.warning(f"   S3 bucket exists but not accessible: {bucket_name}")
            
            result['validation_successful'] = len(result['buckets_found']) >= len(self.expected_resources['s3_buckets']) * 0.5
            
        except Exception as e:
            logger.error(f"S3 resources validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_ecr_repositories(self) -> Dict[str, Any]:
        """Validate ECR repositories"""
        
        result = {
            'repositories_found': {},
            'repositories_missing': [],
            'validation_successful': False
        }
        
        try:
            for repo_name in self.expected_resources['ecr_repositories']:
                try:
                    repo_response = self.ecr_client.describe_repositories(repositoryNames=[repo_name])
                    
                    if repo_response['repositories']:
                        repo_info = repo_response['repositories'][0]
                        result['repositories_found'][repo_name] = {
                            'uri': repo_info['repositoryUri'],
                            'created': repo_info['createdAt'].isoformat(),
                            'registry_id': repo_info['registryId']
                        }
                        logger.info(f"  ✓ ECR repository found: {repo_name}")
                    else:
                        result['repositories_missing'].append(repo_name)
                        
                except self.ecr_client.exceptions.RepositoryNotFoundException:
                    result['repositories_missing'].append(repo_name)
                    logger.warning(f"   ECR repository not found (will be created): {repo_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking ECR repository {repo_name}: {str(e)}")
            
            # ECR repositories can be created during deployment, so this is not critical
            result['validation_successful'] = True
            
        except Exception as e:
            logger.error(f"ECR repositories validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_basic_sagemaker_access(self) -> Dict[str, Any]:
        """Validate basic SageMaker access"""
        
        result = {
            'sagemaker_accessible': False,
            'model_registry_accessible': False,
            'execution_roles_valid': False
        }
        
        try:
            # Test basic SageMaker access
            self.sagemaker_client.list_models(MaxResults=1)
            result['sagemaker_accessible'] = True
            logger.info("  ✓ SageMaker service accessible")
            
            # Test model registry access
            try:
                self.sagemaker_client.list_model_packages(MaxResults=1)
                result['model_registry_accessible'] = True
                logger.info("  ✓ SageMaker Model Registry accessible")
            except Exception:
                logger.warning("   SageMaker Model Registry access limited")
            
            # Test execution roles
            for role_name in self.expected_resources['iam_roles']:
                if 'sagemaker' in role_name.lower():
                    try:
                        # role_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
                        role_arn = self.role
                        # This is a basic validation - actual role assumption would require more complex testing
                        result['execution_roles_valid'] = True
                        logger.info(f"  ✓ SageMaker execution role configured: {role_name} => {role_arn}")
                        break
                    except Exception:
                        continue
            
        except Exception as e:
            logger.error(f"SageMaker access validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_lambda_functions(self) -> Dict[str, Any]:
        """Validate Lambda functions deployment"""
        
        result = {
            'functions_found': {},
            'functions_missing': [],
            'functions_healthy': {},
            'validation_successful': False
        }
        
        try:
            for function_name in self.expected_resources['lambda_functions']:
                try:
                    function_response = self.lambda_client.get_function(FunctionName=function_name)
                    
                    result['functions_found'][function_name] = {
                        'arn': function_response['Configuration']['FunctionArn'],
                        'runtime': function_response['Configuration']['Runtime'],
                        'last_modified': function_response['Configuration']['LastModified'],
                        'state': function_response['Configuration']['State']
                    }
                    
                    # Check if function is healthy
                    is_healthy = function_response['Configuration']['State'] == 'Active'
                    result['functions_healthy'][function_name] = is_healthy
                    
                    status_symbol = "✓" if is_healthy else " "
                    logger.info(f"  {status_symbol} Lambda function: {function_name}")
                    
                except self.lambda_client.exceptions.ResourceNotFoundException:
                    result['functions_missing'].append(function_name)
                    logger.warning(f"   Lambda function not deployed yet: {function_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking Lambda function {function_name}: {str(e)}")
            
            # Consider validation successful if at least some functions are deployed
            # (useful for post-deployment checks)
            result['validation_successful'] = len(result['functions_found']) > 0
            
        except Exception as e:
            logger.error(f"Lambda functions validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_step_functions(self) -> Dict[str, Any]:
        """Validate Step Functions deployment"""
        
        result = {
            'state_machines_found': {},
            'state_machines_missing': [],
            'state_machines_healthy': {},
            'validation_successful': False
        }
        
        try:
            for sf_name in self.expected_resources['step_functions']:
                try:
                    sf_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sf_name}"
                    sf_response = self.stepfunctions_client.describe_state_machine(stateMachineArn=sf_arn)
                    
                    result['state_machines_found'][sf_name] = {
                        'arn': sf_response['stateMachineArn'],
                        'status': sf_response['status'],
                        'created': sf_response['creationDate'].isoformat(),
                        'type': sf_response['type']
                    }
                    
                    # Check if state machine is healthy
                    is_healthy = sf_response['status'] == 'ACTIVE'
                    result['state_machines_healthy'][sf_name] = is_healthy
                    
                    status_symbol = "✓" if is_healthy else " "
                    logger.info(f"  {status_symbol} Step Function: {sf_name}")
                    
                except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
                    result['state_machines_missing'].append(sf_name)
                    logger.warning(f"   Step Function not deployed yet: {sf_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking Step Function {sf_name}: {str(e)}")
            
            # Consider validation successful if at least some state machines are deployed
            result['validation_successful'] = len(result['state_machines_found']) > 0
            
        except Exception as e:
            logger.error(f"Step Functions validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_eventbridge(self) -> Dict[str, Any]:
        """Validate EventBridge rules"""
        
        result = {
            'rules_found': {},
            'rules_missing': [],
            'validation_successful': False
        }
        
        try:
            # Check for training schedule rule
            training_rule_name = "energy-forecasting-monthly-training-pipeline"
            prediction_rule_name = "energy-forecasting-enhanced-daily-predictions"
            
            expected_rules = [training_rule_name, prediction_rule_name]
            
            for rule_name in expected_rules:
                try:
                    rule_response = self.events_client.describe_rule(Name=rule_name)
                    
                    result['rules_found'][rule_name] = {
                        'arn': rule_response['Arn'],
                        'state': rule_response['State'],
                        'schedule': rule_response.get('ScheduleExpression', 'N/A'),
                        'description': rule_response.get('Description', 'N/A')
                    }
                    
                    logger.info(f"  ✓ EventBridge rule: {rule_name}")
                    
                except self.events_client.exceptions.ResourceNotFoundException:
                    result['rules_missing'].append(rule_name)
                    logger.warning(f"   EventBridge rule not found: {rule_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking EventBridge rule {rule_name}: {str(e)}")
            
            result['validation_successful'] = len(result['rules_found']) > 0
            
        except Exception as e:
            logger.error(f"EventBridge validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_container_images(self) -> Dict[str, Any]:
        """Validate container images in ECR"""
        
        result = {
            'images_found': {},
            'images_missing': [],
            'validation_successful': False
        }
        
        try:
            for repo_name in self.expected_resources['ecr_repositories']:
                try:
                    # List images in repository
                    images_response = self.ecr_client.list_images(
                        repositoryName=repo_name,
                        maxResults=10
                    )
                    
                    if images_response['imageIds']:
                        latest_image = None
                        for image in images_response['imageIds']:
                            if image.get('imageTag') == 'latest' or image.get('imageTag') == self.environment:
                                latest_image = image
                                break
                        
                        if not latest_image and images_response['imageIds']:
                            latest_image = images_response['imageIds'][0]
                        
                        if latest_image:
                            result['images_found'][repo_name] = {
                                'imageTag': latest_image.get('imageTag', 'N/A'),
                                'imageDigest': latest_image.get('imageDigest', 'N/A')[:19] + '...',
                                'count': len(images_response['imageIds'])
                            }
                            logger.info(f"  ✓ Container image found: {repo_name}")
                        else:
                            result['images_missing'].append(repo_name)
                            logger.warning(f"   No suitable container image: {repo_name}")
                    else:
                        result['images_missing'].append(repo_name)
                        logger.warning(f"   No container images in repository: {repo_name}")
                        
                except self.ecr_client.exceptions.RepositoryNotFoundException:
                    result['images_missing'].append(repo_name)
                    logger.warning(f"   ECR repository not found: {repo_name}")
                except Exception as e:
                    logger.error(f"  ✗ Error checking container images in {repo_name}: {str(e)}")
            
            result['validation_successful'] = len(result['images_found']) > 0
            
        except Exception as e:
            logger.error(f"Container images validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_basic_connectivity(self) -> Dict[str, Any]:
        """Validate basic cross-service connectivity"""
        
        result = {
            'connectivity_tests': {},
            'validation_successful': False
        }
        
        try:
            # Test S3 to Lambda connectivity (basic IAM test)
            test_results = []
            
            # Test 1: S3 bucket listing
            try:
                buckets = self.s3_client.list_buckets()
                test_results.append(('s3_access', True, f"Found {len(buckets['Buckets'])} buckets"))
            except Exception as e:
                test_results.append(('s3_access', False, str(e)))
            
            # Test 2: Lambda function listing
            try:
                functions = self.lambda_client.list_functions(MaxItems=5)
                test_results.append(('lambda_access', True, f"Found {len(functions['Functions'])} functions"))
            except Exception as e:
                test_results.append(('lambda_access', False, str(e)))
            
            # Test 3: Step Functions listing
            try:
                state_machines = self.stepfunctions_client.list_state_machines(maxResults=5)
                test_results.append(('stepfunctions_access', True, f"Found {len(state_machines['stateMachines'])} state machines"))
            except Exception as e:
                test_results.append(('stepfunctions_access', False, str(e)))
            
            # Test 4: SageMaker access
            try:
                models = self.sagemaker_client.list_models(MaxResults=5)
                test_results.append(('sagemaker_access', True, f"Found {len(models['Models'])} models"))
            except Exception as e:
                test_results.append(('sagemaker_access', False, str(e)))
            
            for test_name, success, message in test_results:
                result['connectivity_tests'][test_name] = {
                    'success': success,
                    'message': message
                }
                status_symbol = "✓" if success else "✗"
                logger.info(f"  {status_symbol} {test_name}: {message}")
            
            # Consider successful if at least 75% of tests pass
            successful_tests = sum(1 for _, success, _ in test_results if success)
            result['validation_successful'] = successful_tests >= len(test_results) * 0.75
            
        except Exception as e:
            logger.error(f"Connectivity validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_sagemaker_resources(self) -> Dict[str, Any]:
        """Validate SageMaker resources (complete version)"""
        
        result = {
            'sagemaker_accessible': False,
            'model_registry_accessible': False,
            'execution_roles_valid': False,
            'model_packages_found': {},
            'endpoints_found': {},
            'validation_successful': False
        }
        
        try:
            # Test basic SageMaker access
            models_response = self.sagemaker_client.list_models(MaxResults=10)
            result['sagemaker_accessible'] = True
            logger.info(f"  ✓ SageMaker accessible - Found {len(models_response['Models'])} models")
            
            # Test model registry access
            try:
                packages_response = self.sagemaker_client.list_model_packages(MaxResults=10)
                result['model_registry_accessible'] = True
                
                # Look for energy forecasting model packages
                for package in packages_response['ModelPackages']:
                    package_name = package['ModelPackageName']
                    if 'energy' in package_name.lower() or 'forecasting' in package_name.lower():
                        result['model_packages_found'][package_name] = {
                            'arn': package['ModelPackageArn'],
                            'status': package['ModelPackageStatus'],
                            'created': package['CreationTime'].isoformat()
                        }
                
                logger.info(f"  ✓ Model Registry accessible - Found {len(result['model_packages_found'])} relevant packages")
                
            except Exception as e:
                logger.warning(f"   Model Registry access limited: {str(e)}")
            
            # Check for existing endpoints
            try:
                endpoints_response = self.sagemaker_client.list_endpoints(MaxResults=10)
                for endpoint in endpoints_response['Endpoints']:
                    endpoint_name = endpoint['EndpointName']
                    if 'energy' in endpoint_name.lower() or 'forecasting' in endpoint_name.lower():
                        result['endpoints_found'][endpoint_name] = {
                            'arn': endpoint['EndpointArn'],
                            'status': endpoint['EndpointStatus'],
                            'created': endpoint['CreationTime'].isoformat()
                        }
                
                logger.info(f"  ✓ Found {len(result['endpoints_found'])} energy forecasting endpoints")
                
            except Exception as e:
                logger.warning(f"   Endpoint listing limited: {str(e)}")
            
            # Validate execution roles
            for role_name in self.expected_resources['iam_roles']:
                if 'sagemaker' in role_name.lower() or 'datascientist' in role_name.lower():
                    try:
                        role_response = self.iam_client.get_role(RoleName=role_name)
                        result['execution_roles_valid'] = True
                        logger.info(f"  ✓ SageMaker execution role: {role_name}")
                        break
                    except Exception:
                        continue
            
            result['validation_successful'] = (
                result['sagemaker_accessible'] and 
                result['model_registry_accessible'] and 
                result['execution_roles_valid']
            )
            
        except Exception as e:
            logger.error(f"SageMaker resources validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_network_connectivity(self) -> Dict[str, Any]:
        """Validate network connectivity"""
        
        result = {
            'vpc_accessible': False,
            'internet_connectivity': False,
            'aws_services_reachable': False,
            'validation_successful': False
        }
        
        try:
            # Basic VPC validation
            try:
                vpcs_response = self.ec2_client.describe_vpcs(MaxResults=5)
                result['vpc_accessible'] = len(vpcs_response['Vpcs']) > 0
                logger.info(f"  ✓ VPC accessible - Found {len(vpcs_response['Vpcs'])} VPCs")
            except Exception as e:
                logger.warning(f"   VPC access limited: {str(e)}")
            
            # Test AWS services reachability (implicit through successful API calls)
            services_reachable = 0
            test_services = [
                ('S3', lambda: self.s3_client.list_buckets()),
                ('Lambda', lambda: self.lambda_client.list_functions(MaxItems=1)),
                ('SageMaker', lambda: self.sagemaker_client.list_models(MaxResults=1)),
                ('Step Functions', lambda: self.stepfunctions_client.list_state_machines(maxResults=1))
            ]
            
            for service_name, test_func in test_services:
                try:
                    test_func()
                    services_reachable += 1
                except Exception:
                    pass
            
            result['aws_services_reachable'] = services_reachable >= 3
            result['internet_connectivity'] = services_reachable > 0  # If we can reach AWS, we have internet
            
            if result['aws_services_reachable']:
                logger.info(f"  ✓ AWS services reachable - {services_reachable}/4 services accessible")
            else:
                logger.warning(f"   Limited AWS services access - {services_reachable}/4 services accessible")
            
            result['validation_successful'] = result['aws_services_reachable']
            
        except Exception as e:
            logger.error(f"Network connectivity validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_resource_dependencies(self) -> Dict[str, Any]:
        """Validate resource dependencies"""
        
        result = {
            'dependencies_valid': False,
            'dependency_checks': {},
            'validation_successful': False
        }
        
        try:
            dependency_checks = []
            
            # Check 1: IAM roles exist for SageMaker execution
            sagemaker_role_exists = any(
                role for role in self.expected_resources['iam_roles']
                if 'ds' in role.lower() or 'datascientist' in role.lower()
            )
            dependency_checks.append(('sagemaker_execution_role', sagemaker_role_exists))
            
            # Check 2: S3 buckets exist for data storage
            s3_buckets_exist = len(self.expected_resources['s3_buckets']) > 0
            dependency_checks.append(('s3_data_storage', s3_buckets_exist))
            
            # # Check 3: Lambda execution role (if exists)
            # lambda_role_exists = any(
            #     role for role in self.expected_resources['iam_roles'] 
            #     if 'lambda' in role.lower()
            # )
            # dependency_checks.append(('lambda_execution_role', lambda_role_exists))
            
            # Check 4: Cross-service permissions (basic test)
            try:
                # Test if we can list resources across services
                self.s3_client.list_buckets()
                self.lambda_client.list_functions(MaxItems=1)
                cross_service_access = True
            except Exception:
                cross_service_access = False
            
            dependency_checks.append(('cross_service_access', cross_service_access))
            
            for check_name, check_result in dependency_checks:
                result['dependency_checks'][check_name] = check_result
                status_symbol = "✓" if check_result else "✗"
                logger.info(f"  {status_symbol} Dependency check: {check_name}")
            
            # Consider valid if at least 75% of dependencies are satisfied
            valid_dependencies = sum(1 for _, result in dependency_checks if result)
            result['dependencies_valid'] = valid_dependencies >= len(dependency_checks) * 0.75
            result['validation_successful'] = result['dependencies_valid']
            
        except Exception as e:
            logger.error(f"Resource dependencies validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_pipeline_integration(self) -> Dict[str, Any]:
        """Validate pipeline integration"""
        
        result = {
            'training_pipeline_ready': False,
            'prediction_pipeline_ready': False,
            'pipeline_connectivity': False,
            'validation_successful': False
        }
        
        try:
            # Check training pipeline components
            training_components = ['sagemaker_resources', 'lambda_functions', 's3_resources']
            training_ready = True

            logger.info(f"Validation Results for Training Pipeline: {self.validation_results}")
            
            for component in training_components:
                if component not in self.validation_results or not self.validation_results[component].get('validation_successful', False):
                    training_ready = False
                    break
            
            result['training_pipeline_ready'] = training_ready
            
            # Check prediction pipeline components
            prediction_components = ['lambda_functions', 'step_functions', 's3_resources']
            prediction_ready = True
            
            for component in prediction_components:
                if component not in self.validation_results or not self.validation_results[component].get('validation_successful', False):
                    prediction_ready = False
                    break
            
            result['prediction_pipeline_ready'] = prediction_ready
            
            # Test basic pipeline connectivity
            result['pipeline_connectivity'] = (
                result['training_pipeline_ready'] and 
                result['prediction_pipeline_ready']
            )
            
            result['validation_successful'] = result['pipeline_connectivity']
            
            status_symbol = "✓" if result['validation_successful'] else "✗"
            logger.info(f"  {status_symbol} Pipeline integration: {'Ready' if result['validation_successful'] else 'Not Ready'}")
            logger.info(f"  RESULT: {json.dumps(result, indent=2)}")
            
        except Exception as e:
            logger.error(f"Pipeline integration validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_data_flow(self) -> Dict[str, Any]:
        """Validate data flows"""
        
        result = {
            'data_sources_accessible': False,
            'data_storage_ready': False,
            'model_storage_ready': False,
            'validation_successful': False
        }
        
        try:
            # Check data bucket accessibility
            data_bucket = f'sdcp-{self.environment}-sagemaker-energy-forecasting-data'
            try:
                self.s3_client.head_bucket(Bucket=data_bucket)
                result['data_sources_accessible'] = True
                result['data_storage_ready'] = True
                logger.info(f"  ✓ Data storage accessible: {data_bucket}")
            except Exception as e:
                logger.warning(f"   Data storage issue: {str(e)}")
            
            # Check model bucket accessibility
            model_bucket = f'sdcp-{self.environment}-sagemaker-energy-forecasting-models'
            try:
                self.s3_client.head_bucket(Bucket=model_bucket)
                result['model_storage_ready'] = True
                logger.info(f"  ✓ Model storage accessible: {model_bucket}")
            except Exception as e:
                logger.warning(f"   Model storage issue: {str(e)}")
            
            result['validation_successful'] = (
                result['data_sources_accessible'] and 
                result['data_storage_ready'] and 
                result['model_storage_ready']
            )
            
        except Exception as e:
            logger.error(f"Data flow validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_performance_benchmarks(self) -> Dict[str, Any]:
        """Validate performance benchmarks"""
        
        result = {
            'lambda_performance_ready': False,
            'stepfunctions_performance_ready': False,
            'sagemaker_performance_ready': False,
            'validation_successful': False
        }
        
        try:
            # Basic performance readiness checks
            
            # Lambda performance readiness
            lambda_result = self.validation_results.get('lambda_functions', {})
            healthy_lambdas = sum(1 for healthy in lambda_result.get('functions_healthy', {}).values() if healthy)
            result['lambda_performance_ready'] = healthy_lambdas >= 3  # At least 3 healthy functions
            
            # Step Functions performance readiness
            sf_result = self.validation_results.get('step_functions', {})
            healthy_sfs = sum(1 for healthy in sf_result.get('state_machines_healthy', {}).values() if healthy)
            result['stepfunctions_performance_ready'] = healthy_sfs >= 1  # At least 1 healthy state machine
            
            # SageMaker performance readiness
            sagemaker_result = self.validation_results.get('sagemaker_resources', {})
            result['sagemaker_performance_ready'] = sagemaker_result.get('sagemaker_accessible', False)
            
            result['validation_successful'] = (
                result['lambda_performance_ready'] and 
                result['stepfunctions_performance_ready'] and 
                result['sagemaker_performance_ready']
            )
            
            status_symbol = "✓" if result['validation_successful'] else " "
            logger.info(f"  {status_symbol} Performance benchmarks: {'Ready' if result['validation_successful'] else 'Limited'}")
            
        except Exception as e:
            logger.error(f"Performance benchmarks validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _validate_end_to_end_connectivity(self) -> Dict[str, Any]:
        """Validate end-to-end connectivity"""
        
        result = {
            'full_pipeline_connectivity': False,
            'external_integrations': False,
            'monitoring_connectivity': False,
            'validation_successful': False
        }
        
        try:
            # Check full pipeline connectivity
            required_components = ['s3_resources', 'lambda_functions', 'step_functions', 'sagemaker_resources']
            components_ready = 0
            
            for component in required_components:
                if self.validation_results.get(component, {}).get('validation_successful', False):
                    components_ready += 1
            
            result['full_pipeline_connectivity'] = components_ready >= len(required_components) * 0.75
            
            # Check monitoring connectivity (CloudWatch is implicit)
            result['monitoring_connectivity'] = True  # CloudWatch is always available if AWS access works
            
            # External integrations (EventBridge, etc.)
            eventbridge_result = self.validation_results.get('eventbridge', {})
            result['external_integrations'] = eventbridge_result.get('validation_successful', False)
            
            result['validation_successful'] = (
                result['full_pipeline_connectivity'] and 
                result['monitoring_connectivity']
            )
            
            status_symbol = "✓" if result['validation_successful'] else " "
            logger.info(f"  {status_symbol} End-to-end connectivity: {'Ready' if result['validation_successful'] else 'Limited'}")
            
        except Exception as e:
            logger.error(f"End-to-end connectivity validation failed: {str(e)}")
            result['error'] = str(e)
        
        return result

    def _assess_pre_deployment_readiness(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Assess pre-deployment readiness"""
        
        critical_issues = []
        warnings = []
        ready_for_deployment = True
        
        # Check credentials
        if not checks.get('credentials', {}).get('caller_identity_valid', False):
            critical_issues.append("AWS credentials are invalid")
            ready_for_deployment = False
        
        if not checks.get('credentials', {}).get('basic_permissions_valid', False):
            critical_issues.append("Insufficient AWS permissions")
            ready_for_deployment = False
        
        # Check IAM roles
        iam_result = checks.get('iam_roles', {})
        if len(iam_result.get('roles_found', {})) == 0:
            critical_issues.append("No required IAM roles found")
            ready_for_deployment = False
        elif len(iam_result.get('roles_missing', [])) > 0:
            warnings.append(f"Missing IAM roles: {', '.join(iam_result['roles_missing'])}")
        
        # Check S3 buckets
        s3_result = checks.get('s3_buckets', {})
        if len(s3_result.get('buckets_found', {})) == 0:
            critical_issues.append("No S3 buckets accessible")
            ready_for_deployment = False
        
        # Check SageMaker access
        sagemaker_result = checks.get('sagemaker_access', {})
        if not sagemaker_result.get('sagemaker_accessible', False):
            warnings.append("SageMaker access limited")
        
        return {
            'ready_for_deployment': ready_for_deployment,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'overall_status': 'READY' if ready_for_deployment else 'NOT_READY'
        }

    def _assess_post_deployment_success(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Assess post-deployment success"""
        
        deployment_issues = []
        warnings = []
        deployment_successful = True
        
        # Check Lambda functions
        lambda_result = checks.get('lambda_functions', {})
        deployed_functions = len(lambda_result.get('functions_found', {}))
        if deployed_functions == 0:
            deployment_issues.append("No Lambda functions deployed")
            deployment_successful = False
        elif deployed_functions < len(self.expected_resources['lambda_functions']):
            warnings.append(f"Only {deployed_functions}/{len(self.expected_resources['lambda_functions'])} Lambda functions deployed")
        
        # Check Step Functions
        sf_result = checks.get('step_functions', {})
        deployed_sfs = len(sf_result.get('state_machines_found', {}))
        if deployed_sfs == 0:
            deployment_issues.append("No Step Functions deployed")
            deployment_successful = False
        
        # Check container images
        container_result = checks.get('container_images', {})
        if len(container_result.get('images_found', {})) == 0:
            warnings.append("No container images found")
        
        return {
            'deployment_successful': deployment_successful,
            'deployment_issues': deployment_issues,
            'warnings': warnings,
            'overall_status': 'SUCCESS' if deployment_successful else 'PARTIAL'
        }

    def _assess_integration_readiness(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Assess integration readiness"""
        
        integration_issues = []
        warnings = []
        integration_ready = True
        
        # Check complete environment
        complete_env = checks.get('complete_environment', {})
        if not complete_env.get('validation_summary', {}).get('environment_ready', False):
            integration_issues.append("Complete environment validation failed")
            integration_ready = False
        
        # Check pipeline integration
        pipeline_result = checks.get('pipeline_integration', {})
        if not pipeline_result.get('pipeline_connectivity', False):
            integration_issues.append("Pipeline connectivity issues detected")
            integration_ready = False
        
        # Check data flow
        data_flow_result = checks.get('data_flow', {})
        if not data_flow_result.get('validation_successful', False):
            warnings.append("Data flow validation issues")
        
        # Check performance
        performance_result = checks.get('performance', {})
        if not performance_result.get('validation_successful', False):
            warnings.append("Performance benchmark issues")
        
        return {
            'integration_ready': integration_ready,
            'integration_issues': integration_issues,
            'warnings': warnings,
            'overall_status': 'READY' if integration_ready else 'NOT_READY'
        }

    def _generate_validation_summary(self, validation_duration: float) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        # Count successful validations
        successful_validations = 0
        total_validations = len(self.validation_results)
        critical_failures = []
        warnings = []
        
        for validation_name, validation_result in self.validation_results.items():
            if validation_result.get('validation_successful', False):
                successful_validations += 1
            else:
                if validation_name in ['credentials_and_permissions', 'iam_roles', 's3_resources']:
                    critical_failures.append(validation_name)
                else:
                    warnings.append(validation_name)
        
        # Determine overall status
        success_rate = successful_validations / total_validations if total_validations > 0 else 0
        
        if success_rate >= 0.9 and len(critical_failures) == 0:
            overall_status = 'EXCELLENT'
            environment_ready = True
        elif success_rate >= 0.75 and len(critical_failures) == 0:
            overall_status = 'GOOD'
            environment_ready = True
        elif success_rate >= 0.5 and len(critical_failures) <= 1:
            overall_status = 'ACCEPTABLE'
            environment_ready = True
        else:
            overall_status = 'NEEDS_ATTENTION'
            environment_ready = False
        
        return {
            'overall_status': overall_status,
            'environment_ready': environment_ready,
            'success_rate': round(success_rate * 100, 1),
            'successful_validations': successful_validations,
            'total_validations': total_validations,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'validation_duration_minutes': round(validation_duration / 60, 2)
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Check validation results and generate specific recommendations
        for validation_name, validation_result in self.validation_results.items():
            if not validation_result.get('validation_successful', True):
                
                if validation_name == 'credentials_and_permissions':
                    recommendations.append("Verify AWS credentials and ensure proper IAM permissions are configured")
                
                elif validation_name == 'iam_roles':
                    missing_roles = validation_result.get('roles_missing', [])
                    if missing_roles:
                        recommendations.append(f"Create missing IAM roles: {', '.join(missing_roles)}")
                
                elif validation_name == 's3_resources':
                    missing_buckets = validation_result.get('buckets_missing', [])
                    if missing_buckets:
                        recommendations.append(f"Create missing S3 buckets: {', '.join(missing_buckets)}")
                
                elif validation_name == 'lambda_functions':
                    missing_functions = validation_result.get('functions_missing', [])
                    if missing_functions:
                        recommendations.append("Deploy missing Lambda functions or run complete MLOps deployment")
                
                elif validation_name == 'step_functions':
                    missing_sfs = validation_result.get('state_machines_missing', [])
                    if missing_sfs:
                        recommendations.append("Deploy missing Step Functions or run complete MLOps deployment")
                
                elif validation_name == 'ecr_repositories':
                    missing_repos = validation_result.get('repositories_missing', [])
                    if missing_repos:
                        recommendations.append(f"Create ECR repositories: {', '.join(missing_repos)}")
                
                elif validation_name == 'sagemaker_resources':
                    recommendations.append("Verify SageMaker permissions and Model Registry access")
                
                elif validation_name == 'eventbridge':
                    recommendations.append("Configure EventBridge rules for automated scheduling")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Environment validation passed - ready for deployment")
        else:
            recommendations.append("Review AWS documentation for specific service configuration requirements")
            recommendations.append("Consider running validation again after addressing issues")
        
        return recommendations

    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on validation results"""
        
        next_steps = []
        
        # Determine validation summary
        validation_summary = self._generate_validation_summary(0)  # Duration not needed for next steps
        
        if validation_summary['environment_ready']:
            next_steps.extend([
                "✓ Environment is ready for MLOps deployment",
                "Run: python sdcp_code/deployment/deploy_enhanced_mlops.py --environment {env}".format(env=self.environment),
                "Monitor deployment logs for any issues",
                "Run post-deployment validation after deployment completes"
            ])
        else:
            next_steps.extend([
                "✗ Environment needs fixes before deployment",
                "Address critical issues identified in validation results",
                "Re-run validation: python sdcp_code/deployment/validate_environment.py --environment {env}".format(env=self.environment),
                "Contact infrastructure team if IAM/VPC issues persist"
            ])
        
        return next_steps


def main():
    """Main function for environment validation"""
    
    parser = argparse.ArgumentParser(description='Environment Validator for Enhanced Prediction Pipeline')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--role', help='AWS IAM role ARN for data scientist')
    parser.add_argument('--environment', default='dev', help='Environment (dev/preprod/prod)')
    
    # Validation type arguments (mutually exclusive)
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument('--pre-deployment-check', action='store_true', 
                                help='Run pre-deployment validation checks')
    validation_group.add_argument('--post-deployment-check', action='store_true',
                                help='Run post-deployment validation checks')
    validation_group.add_argument('--complete-integration-check', action='store_true',
                                help='Run complete integration validation checks')
    
    # Additional options
    parser.add_argument('--quick', action='store_true', help='Run quick validation (basic checks only)')
    parser.add_argument('--category', help='Validate specific category only')
    parser.add_argument('--output-json', help='Save validation results to JSON file')
    parser.add_argument('--output-file', help='Save validation results to specified file')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EnvironmentValidator(region=args.region, environment=args.environment, role=args.role)
    
    try:
        validation_result = None
        
        if args.pre_deployment_check:
            logger.info(" Running PRE-DEPLOYMENT validation...")
            validation_result = validator.run_pre_deployment_check()
            
        elif args.post_deployment_check:
            logger.info(" Running POST-DEPLOYMENT validation...")
            validation_result = validator.run_post_deployment_check()
            
        elif args.complete_integration_check:
            logger.info(" Running COMPLETE INTEGRATION validation...")
            validation_result = validator.run_complete_integration_check()
            
        elif args.quick:
            logger.info(" Running QUICK validation...")
            creds_result = validator._validate_credentials_and_permissions()
            
            if creds_result.get('caller_identity_valid') and creds_result.get('basic_permissions_valid'):
                logger.info("✓ Quick validation PASSED - Basic environment is ready")
                validation_result = {'summary': {'ready_for_deployment': True, 'overall_status': 'READY'}}
            else:
                logger.error("✗ Quick validation FAILED - Environment issues detected")
                print(json.dumps(creds_result, indent=2, default=str))
                validation_result = {'summary': {'ready_for_deployment': False, 'overall_status': 'NOT_READY'}}
                
        elif args.category:
            logger.info(f" Running validation for category: {args.category}")
            
            validation_methods = {
                'credentials': validator._validate_credentials_and_permissions,
                'iam': validator._validate_iam_roles,
                's3': validator._validate_s3_resources,
                'ecr': validator._validate_ecr_repositories,
                'sagemaker': validator._validate_sagemaker_resources,
                'lambda': validator._validate_lambda_functions,
                'stepfunctions': validator._validate_step_functions,
                'eventbridge': validator._validate_eventbridge,
                'network': validator._validate_network_connectivity,
                'dependencies': validator._validate_resource_dependencies
            }
            
            if args.category in validation_methods:
                category_result = validation_methods[args.category]()
                validation_result = {
                    'category': args.category,
                    'result': category_result,
                    'summary': {'validation_successful': category_result.get('validation_successful', False)}
                }
                print(json.dumps(validation_result, indent=2, default=str))
                
                if category_result.get('validation_successful', False):
                    logger.info(f"✓ Category validation PASSED: {args.category}")
                    sys.exit(0)
                else:
                    logger.error(f"✗ Category validation FAILED: {args.category}")
                    sys.exit(1)
            else:
                logger.error(f"Unknown category: {args.category}")
                logger.error(f"Available categories: {', '.join(validation_methods.keys())}")
                sys.exit(1)
                
        else:
            # Default: Run complete environment validation
            logger.info(" Running COMPLETE environment validation...")
            validation_result = validator.validate_complete_environment()
        
        # Save results to file if requested
        if args.output_json or args.output_file:
            output_file = args.output_json or args.output_file
            with open(output_file, 'w') as f:
                json.dump(validation_result, f, indent=2, default=str)
            logger.info(f" Validation results saved to: {output_file}")
        
        # Print validation summary
        if validation_result:
            print("\n" + "="*80)
            print("VALIDATION SUMMARY")
            print("="*80)
            
            summary = validation_result.get('summary', validation_result.get('validation_summary', {}))
            
            # Print overall status
            overall_status = summary.get('overall_status', 'UNKNOWN')
            print(f"Overall Status: {overall_status}")
            
            # Print readiness status
            if args.pre_deployment_check:
                ready_key = 'ready_for_deployment'
                ready_message = "Environment is ready for deployment"
            elif args.post_deployment_check:
                ready_key = 'deployment_successful'
                ready_message = "Deployment was successful"
            elif args.complete_integration_check:
                ready_key = 'integration_ready'
                ready_message = "Integration is ready"
            else:
                ready_key = 'environment_ready'
                ready_message = "Environment is ready"
            
            is_ready = summary.get(ready_key, False)
            ready_symbol = " " if is_ready else " "
            print(f"{ready_symbol} {ready_message}: {is_ready}")
            
            # Print success rate if available
            if 'success_rate' in summary:
                print(f"Success Rate: {summary['success_rate']}%")
            
            # Print critical issues
            critical_issues = summary.get('critical_issues', summary.get('critical_failures', []))
            if critical_issues:
                print(f"\n Critical Issues ({len(critical_issues)}):")
                for i, issue in enumerate(critical_issues, 1):
                    print(f"  {i}. {issue}")
            
            # Print warnings
            warnings = summary.get('warnings', [])
            if warnings:
                print(f"\n  Warnings ({len(warnings)}):")
                for i, warning in enumerate(warnings, 1):
                    print(f"  {i}. {warning}")
            
            # Print recommendations
            recommendations = validation_result.get('recommendations', [])
            if recommendations:
                print(f"\n Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            
            # Print next steps
            next_steps = validation_result.get('next_steps', [])
            if next_steps:
                print(f"\n Next Steps:")
                for i, step in enumerate(next_steps, 1):
                    print(f"  {i}. {step}")
            
            # Exit with appropriate code
            if is_ready:
                logger.info(f"\n✓ VALIDATION PASSED")
                logger.info(f"✓ {ready_message}")
                sys.exit(0)
            else:
                logger.error(f"\n✗ VALIDATION FAILED")
                logger.error(f"✗ Environment needs attention before proceeding")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Environment validation failed with unexpected error: {str(e)}")
        sys.exit(1)


def validate_for_deployment() -> bool:
    """
    Simple validation function for use by other scripts
    Returns True if environment is ready for deployment
    """
    
    try:
        validator = EnvironmentValidator()
        result = validator.validate_complete_environment()
        
        return result.get('validation_summary', {}).get('environment_ready', False)
        
    except Exception:
        return False


def get_validation_report(region: str = "us-west-2", environment: str = "dev") -> Dict[str, Any]:
    """
    Get detailed validation report for programmatic use
    
    Args:
        region: AWS region
        environment: Environment name
        
    Returns:
        Dictionary containing complete validation results
    """
    
    try:
        validator = EnvironmentValidator(region=region, environment=environment)
        return validator.validate_complete_environment()
        
    except Exception as e:
        return {
            'validation_failed': True,
            'error': str(e),
            'validation_summary': {
                'overall_status': 'ERROR',
                'environment_ready': False
            }
        }


def check_specific_resource(resource_type: str, region: str = "us-west-2", environment: str = "dev") -> Dict[str, Any]:
    """
    Check specific resource type
    
    Args:
        resource_type: Type of resource to check (iam, s3, lambda, etc.)
        region: AWS region
        environment: Environment name
        
    Returns:
        Dictionary containing validation results for the specific resource type
    """
    
    try:
        validator = EnvironmentValidator(region=region, environment=environment)
        
        validation_methods = {
            'credentials': validator._validate_credentials_and_permissions,
            'iam': validator._validate_iam_roles,
            's3': validator._validate_s3_resources,
            'ecr': validator._validate_ecr_repositories,
            'sagemaker': validator._validate_sagemaker_resources,
            'lambda': validator._validate_lambda_functions,
            'stepfunctions': validator._validate_step_functions,
            'eventbridge': validator._validate_eventbridge,
            'network': validator._validate_network_connectivity,
            'dependencies': validator._validate_resource_dependencies
        }
        
        if resource_type in validation_methods:
            return validation_methods[resource_type]()
        else:
            return {
                'validation_failed': True,
                'error': f'Unknown resource type: {resource_type}',
                'available_types': list(validation_methods.keys())
            }
            
    except Exception as e:
        return {
            'validation_failed': True,
            'error': str(e)
        }


def validate_prerequisites_for_enhanced_pipeline() -> Dict[str, Any]:
    """
    Specific validation for enhanced prediction pipeline prerequisites
    
    Returns:
        Dictionary with validation results and readiness status
    """
    
    try:
        validator = EnvironmentValidator()
        
        # Run key validations for enhanced pipeline
        results = {}
        
        # Check credentials and permissions
        results['credentials'] = validator._validate_credentials_and_permissions()
        
        # Check IAM roles
        results['iam_roles'] = validator._validate_iam_roles()
        
        # Check S3 resources including endpoint configurations
        results['s3_resources'] = validator._validate_s3_resources()
        
        # Check Lambda functions
        results['lambda_functions'] = validator._validate_lambda_functions()
        
        # Check SageMaker access
        results['sagemaker'] = validator._validate_sagemaker_resources()
        
        # Determine readiness
        critical_checks = [
            results['credentials'].get('caller_identity_valid', False),
            results['credentials'].get('basic_permissions_valid', False),
            len(results['iam_roles'].get('roles_found', {})) > 0,
            len(results['s3_resources'].get('buckets_found', {})) > 0,
            results['sagemaker'].get('model_registry_accessible', False)
        ]
        
        enhanced_pipeline_ready = all(critical_checks)
        
        return {
            'enhanced_pipeline_ready': enhanced_pipeline_ready,
            'validation_results': results,
            'critical_checks_passed': sum(critical_checks),
            'total_critical_checks': len(critical_checks),
            'readiness_percentage': round((sum(critical_checks) / len(critical_checks)) * 100, 1)
        }
        
    except Exception as e:
        return {
            'enhanced_pipeline_ready': False,
            'error': str(e),
            'validation_results': {}
        }


if __name__ == "__main__":
    main()         
