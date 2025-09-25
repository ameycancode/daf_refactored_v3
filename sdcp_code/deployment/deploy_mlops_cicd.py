#!/usr/bin/env python3
"""
Unified MLOps CI/CD Deployment Script
sdcp_code/deployment/deploy_unified_mlops_cicd.py

Single file that handles complete MLOps deployment with CI/CD integration.
Consolidates all deployment steps with unified reporting and timing.
"""

import sys
import os
import json
import argparse
import logging
import time
import boto3
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Add sdcp_code to path for proper imports
script_dir = Path(__file__).parent
sdcp_code_dir = script_dir.parent
sys.path.insert(0, str(sdcp_code_dir))

# Import required modules
try:
    from deployment.lambda_deployer import CompleteLambdaDeployer
    from deployment.container_config_manager import ContainerConfigManager
    import subprocess
    import os
except ImportError as e:
    print(f"Error: Could not import required modules: {str(e)}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class UnifiedMLOpsDeployment:
    """
    Unified MLOps Deployment - Single file handling complete deployment pipeline
    """

    def __init__(self, environment: str, sagemaker_role_arn: str, region: str = "us-west-2",
                 ci_cd_mode: bool = False, github_run_id: Optional[str] = None):
        """Initialize unified MLOps deployment"""
        
        self.environment = environment
        self.sagemaker_role_arn = sagemaker_role_arn
        self.region = region
        self.ci_cd_mode = ci_cd_mode
        self.github_run_id = github_run_id
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.deployment_start_time = datetime.now()
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.events_client = boto3.client('events', region_name=region)
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.iam_client = boto3.client('iam')
        
        # Environment-specific configuration
        self.config = self._get_environment_config()
        
        # Initialize lambda deployer
        self.lambda_deployer = CompleteLambdaDeployer(
            region=region,
            environment=environment,
            # datascientist_role_name=f'sdcp-{environment}-sagemaker-energy-forecasting-datascientist-role'
            datascientist_role_name=self.sagemaker_role_arn.split('/')[-1]
        )
        
        # Set up CI/CD logging if needed
        if ci_cd_mode:
            self._setup_cicd_logging()
        
        self.logger.info(f"Unified MLOps Deployment initialized for {environment} environment")
        self.logger.info(f"CI/CD Mode: {ci_cd_mode}")
        self.logger.info(f"GitHub Run ID: {github_run_id}")
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        return {
            "data_bucket": f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
            "model_bucket": f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
            "model_prefix": 'sdcp_modeling/forecasting/models/xgboost/',
            "registry_prefix": 'sdcp_modeling/forecasting/models/registry/',
            "datascientist_role": self.sagemaker_role_arn,
            # "datascientist_role": f'arn:aws:iam::{self.account_id}:role/sdcp-{self.environment}-sagemaker-energy-forecasting-datascientist-role',
            # "eventbridge_role": f'arn:aws:iam::{self.account_id}:role/EnergyForecastingEventBridgeRole',
            "lambda_prefix": f'energy-forecasting-{self.environment}',
            "step_functions_prefix": f'energy-forecasting-{self.environment}',
            "training_state_machine": f'energy-forecasting-training-pipeline',
            "enhanced_prediction_state_machine": f'energy-forecasting-enhanced-prediction-pipeline',
            "containers": ['energy-preprocessing', 'energy-training'] #, 'energy-prediction']
        }
    
    def _setup_cicd_logging(self):
        """Setup CI/CD specific logging"""
        if self.ci_cd_mode and self.github_run_id:
            log_filename = f"deployment-{self.environment}-{self.github_run_id}.log"
            file_handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def deploy_complete_mlops_pipeline(self, skip_environment_validation: bool = False,
                                     deployment_bucket: Optional[str] = None,
                                     model_bucket: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete unified MLOps deployment with single tracking and reporting
        """
        
        # Initialize single deployment summary
        deployment_summary = {
            "deployment_info": {
                "environment": self.environment,
                "region": self.region,
                "account_id": self.account_id,
                "ci_cd_mode": self.ci_cd_mode,
                "github_run_id": self.github_run_id,
                "deployment_start_time": self.deployment_start_time.isoformat(),
                "skip_environment_validation": skip_environment_validation
            },
            "deployment_status": "IN_PROGRESS",
            "steps_executed": [],
            "steps_failed": [],
            "resources_deployed": {},
            "validation_results": {},
            "timing": {},
            "error_details": None
        }
        
        try:
            self.logger.info("="*100)
            self.logger.info("UNIFIED MLOPS CI/CD DEPLOYMENT STARTED")
            self.logger.info("="*100)
            self.logger.info(f"Environment: {self.environment}")
            self.logger.info(f"Region: {self.region}")
            self.logger.info(f"Account: {self.account_id}")
            self.logger.info(f"CI/CD Mode: {self.ci_cd_mode}")
            self.logger.info(f"GitHub Run ID: {self.github_run_id}")
            self.logger.info(f"Skip Environment Validation: {skip_environment_validation}")
            self.logger.info("="*100)
            
            # Update bucket configuration if provided
            if deployment_bucket:
                self.config["data_bucket"] = deployment_bucket
            if model_bucket:
                self.config["model_bucket"] = model_bucket
            
            # STEP 1: Environment Validation (conditional)
            if not skip_environment_validation:
                step_start = time.time()
                self._track_step("Environment Validation", deployment_summary)
               
                validation_result = self._validate_environment()
                deployment_summary["validation_results"]["environment"] = validation_result
                deployment_summary["timing"]["environment_validation"] = time.time() - step_start
               
                if validation_result.get("environment_ready", False):
                    self._complete_step("Environment Validation", deployment_summary, "COMPLETED")
                else:
                    self._complete_step("Environment Validation", deployment_summary, "FAILED")
                    raise Exception("Environment validation failed - cannot proceed")
            else:
                self.logger.info("STEP 1: SKIPPING ENVIRONMENT VALIDATION (handled by workflow)")
                deployment_summary["steps_executed"].append({
                    "step": "Environment Validation",
                    "status": "SKIPPED",
                    "reason": "handled_by_workflow",
                    "timestamp": datetime.now().isoformat()
                })
            
            # STEP 2: Environment-Aware Container Configuration
            step_start = time.time()
            self._track_step("Container Configuration", deployment_summary)
            
            container_config_result = self._setup_environment_aware_containers()
            deployment_summary["resources_deployed"]["container_configs"] = container_config_result
            deployment_summary["timing"]["container_configuration"] = time.time() - step_start
            
            if container_config_result.get("status") == "success":
                self._complete_step("Container Configuration", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Container Configuration", deployment_summary, "FAILED")
                raise Exception(f"Container configuration failed: {container_config_result.get('error')}")
                        
            # STEP 3: Enhanced Lambda Functions Deployment
            step_start = time.time()
            self._track_step("Lambda Functions Deployment", deployment_summary)
            
            lambda_result = self._deploy_lambda_functions()
            deployment_summary["resources_deployed"]["lambda_functions"] = lambda_result
            deployment_summary["timing"]["lambda_deployment"] = time.time() - step_start
            
            if lambda_result.get("status") == "success":
                self._complete_step("Lambda Functions Deployment", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Lambda Functions Deployment", deployment_summary, "FAILED")
                raise Exception(f"Lambda deployment failed: {lambda_result.get('error')}")
            
            # STEP 4: Enhanced Step Functions Deployment
            step_start = time.time()
            self._track_step("Step Functions Deployment", deployment_summary)
            
            stepfunctions_result = self._deploy_step_functions()
            deployment_summary["resources_deployed"]["step_functions"] = stepfunctions_result
            deployment_summary["timing"]["step_functions_deployment"] = time.time() - step_start
            
            if stepfunctions_result.get("status") == "success":
                self._complete_step("Step Functions Deployment", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Step Functions Deployment", deployment_summary, "FAILED")
                raise Exception(f"Step Functions deployment failed: {stepfunctions_result.get('error')}")
            
            # STEP 5: Container Build and Push
            step_start = time.time()
            self._track_step("Container Build and Push", deployment_summary)
            
            self.logger.info("  Building and pushing all containers locally...")
            containers_result = self._build_and_push_containers()
            deployment_summary["resources_deployed"]["containers"] = containers_result
            deployment_summary["timing"]["container_build"] = time.time() - step_start
            
            if containers_result.get("status") in ["success", "partial"]:
                self._complete_step("Container Build and Push", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Container Build and Push", deployment_summary, "FAILED")
                # Container build failure is not critical for core functionality
                self.logger.warning("  Container build failed, but continuing with deployment")
            
            # STEP 6: Enhanced EventBridge Setup
            step_start = time.time()
            self._track_step("EventBridge Setup", deployment_summary)
            
            eventbridge_result = self._setup_eventbridge()
            deployment_summary["resources_deployed"]["eventbridge"] = eventbridge_result
            deployment_summary["timing"]["eventbridge_setup"] = time.time() - step_start
            
            if eventbridge_result.get("status") == "success":
                self._complete_step("EventBridge Setup", deployment_summary, "COMPLETED")
            else:
                self._complete_step("EventBridge Setup", deployment_summary, "FAILED")
                # EventBridge failure is not critical
                self.logger.warning("  EventBridge setup failed, but core deployment can continue")
            
            # STEP 7: Complete Deployment Validation
            step_start = time.time()
            self._track_step("Deployment Validation", deployment_summary)
            
            deployment_validation = self._validate_complete_deployment()
            deployment_summary["validation_results"]["deployment"] = deployment_validation
            deployment_summary["timing"]["deployment_validation"] = time.time() - step_start
            
            if deployment_validation.get("status") in ["SUCCESS", "PARTIAL"]:
                self._complete_step("Deployment Validation", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Deployment Validation", deployment_summary, "FAILED")
                self.logger.warning("  Deployment validation failed")
            
            # STEP 8: Integration Tests
            step_start = time.time()
            self._track_step("Integration Tests", deployment_summary)
            
            integration_result = self._run_integration_tests()
            deployment_summary["validation_results"]["integration"] = integration_result
            deployment_summary["timing"]["integration_tests"] = time.time() - step_start
            
            if integration_result.get("status") in ["SUCCESS", "PARTIAL"]:
                self._complete_step("Integration Tests", deployment_summary, "COMPLETED")
            else:
                self._complete_step("Integration Tests", deployment_summary, "FAILED")
                self.logger.warning("  Integration tests failed")
            

            deployment_summary["timing"]["artifacts_generation"] = time.time() - step_start
            
            # Calculate total deployment time
            total_deployment_time = time.time() - self.deployment_start_time.timestamp()
            deployment_summary["timing"]["total_deployment_seconds"] = total_deployment_time
            deployment_summary["timing"]["total_deployment_minutes"] = total_deployment_time / 60
            
            # Finalize deployment summary
            deployment_summary["deployment_status"] = "SUCCESS"
            deployment_summary["deployment_end_time"] = datetime.now().isoformat()
            deployment_summary["deployment_duration"] = str(
                datetime.now() - self.deployment_start_time
            )
            
            # Assess overall success
            overall_success = self._assess_overall_success(deployment_summary)
            deployment_summary["overall_success"] = overall_success

            # STEP 9: Generate Final Artifacts and Summary
            step_start = time.time()
            self._track_step("Final Artifacts Generation", deployment_summary)
            
            try:
                self._generate_deployment_artifacts(deployment_summary)
                self._complete_step("Final Artifacts Generation", deployment_summary, "COMPLETED")
            except Exception as e:
                self.logger.error(f" Artifact generation failed: {str(e)}")
                self._complete_step("Final Artifacts Generation", deployment_summary, "FAILED")
            
            self.logger.info("="*100)
            self.logger.info("UNIFIED MLOPS DEPLOYMENT COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total Duration: {deployment_summary['timing']['total_deployment_minutes']:.2f} minutes")
            self.logger.info(f"Overall Success: {overall_success}")
            self.logger.info("="*100)
            
            return deployment_summary
            
        except Exception as e:
            total_deployment_time = time.time() - self.deployment_start_time.timestamp()
            deployment_summary["deployment_status"] = "FAILED"
            deployment_summary["error_details"] = str(e)
            deployment_summary["deployment_end_time"] = datetime.now().isoformat()
            deployment_summary["timing"]["total_deployment_seconds"] = total_deployment_time
            deployment_summary["timing"]["total_deployment_minutes"] = total_deployment_time / 60
            deployment_summary["overall_success"] = False
            
            self.logger.error(f"Unified MLOps deployment failed: {str(e)}")
            
            # Generate failure report
            self._generate_failure_report(deployment_summary, e)
            
            raise

    def _complete_step(self, step_name: str, deployment_summary: Dict[str, Any], status: str = "COMPLETED"):
        """Mark step as completed"""
        for step in reversed(deployment_summary["steps_executed"]):
            if step["step"] == step_name and step["status"] == "IN_PROGRESS":
                step["status"] = status
                step["end_time"] = datetime.now().isoformat()
                break
       
        if status == "COMPLETED":
            self.logger.info(f" {step_name} - COMPLETED")
        else:
            self.logger.error(f" {step_name} - {status}")
            deployment_summary["steps_failed"].append({
                "step": step_name,
                "status": status,
                "timestamp": datetime.now().isoformat()
            })

    def _track_step(self, step_name: str, deployment_summary: Dict[str, Any]):
        """Track deployment step execution"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"EXECUTING: {step_name}")
        self.logger.info(f"{'='*80}")
        
        deployment_summary["steps_executed"].append({
            "step": step_name,
            "status": "IN_PROGRESS",
            "start_time": datetime.now().isoformat()
        })

    def _verify_container_images(self) -> Dict[str, Any]:
        """Verify that container images exist in ECR"""
        self.logger.info("Verifying container images in ECR...")
       
        verification_results = {
            "images_checked": [],
            "images_available": [],
            "images_missing": [],
            "all_available": False
        }
       
        try:
            for repo_name in self.config['containers']:
                self.logger.info(f"   Checking repository: {repo_name}")
               
                try:
                    # List images in repository
                    response = self.ecr_client.list_images(repositoryName=repo_name)
                    images = response.get('imageIds', [])
                   
                    verification_results["images_checked"].append(repo_name)
                   
                    if images:
                        self.logger.info(f"   {repo_name}: {len(images)} images found")
                        verification_results["images_available"].append({
                            "repository": repo_name,
                            "image_count": len(images),
                            "latest_images": [img.get('imageTag', 'untagged') for img in images[:3]]  # Show first 3 tags
                        })
                    else:
                        self.logger.warning(f"    {repo_name}: No images found")
                        verification_results["images_missing"].append({
                            "repository": repo_name,
                            "reason": "No images in repository"
                        })
                       
                except self.ecr_client.exceptions.RepositoryNotFoundException:
                    self.logger.warning(f"   {repo_name}: Repository not found")
                    verification_results["images_missing"].append({
                        "repository": repo_name,
                        "reason": "Repository not found"
                    })
                except Exception as e:
                    self.logger.error(f"   {repo_name}: Error checking repository: {str(e)}")
                    verification_results["images_missing"].append({
                        "repository": repo_name,
                        "reason": str(e)
                    })
           
            verification_results["all_available"] = len(verification_results["images_missing"]) == 0
           
            # Summary logging
            available_count = len(verification_results["images_available"])
            missing_count = len(verification_results["images_missing"])
            total_count = len(self.config['containers'])
           
            if verification_results["all_available"]:
                self.logger.info(f" All {total_count} container images verified in ECR")
            else:
                self.logger.warning(f"  Container verification: {available_count}/{total_count} available, {missing_count} missing")
               
                # Log missing details
                for missing in verification_results["images_missing"]:
                    self.logger.warning(f"     Missing: {missing['repository']} - {missing['reason']}")
           
            return verification_results
           
        except Exception as e:
            self.logger.error(f"Container image verification failed: {str(e)}")
            return {
                "images_checked": [],
                "images_available": [],
                "images_missing": [],
                "all_available": False,
                "error": str(e)
            }
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Environment validation using existing framework"""
        self.logger.info("Running environment validation...")
        
        try:
            # Import and use existing validation
            from deployment.validate_environment import EnvironmentValidator
            
            validator = EnvironmentValidator(
                region=self.region,
                environment=self.environment
            )
            
            validation_result = validator.validate_complete_environment()
            
            return {
                "environment_ready": validation_result.get('validation_summary', {}).get('environment_ready', False),
                "validation_details": validation_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except ImportError:
            self.logger.warning("Environment validator not available - running basic validation")
            return self._basic_environment_validation()
        except Exception as e:
            self.logger.error(f"Environment validation failed: {str(e)}")
            return {
                "environment_ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _basic_environment_validation(self) -> Dict[str, Any]:
        """Basic environment validation fallback"""
        try:
            # Check basic AWS connectivity
            self.iam_client.get_user()
            
            # Check S3 buckets
            self.s3_client.head_bucket(Bucket=self.config['data_bucket'])
            self.s3_client.head_bucket(Bucket=self.config['model_bucket'])
            
            return {
                "environment_ready": True,
                "validation_type": "basic",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "environment_ready": False,
                "error": str(e),
                "validation_type": "basic",
                "timestamp": datetime.now().isoformat()
            }
    
    def _setup_environment_aware_containers(self) -> Dict[str, Any]:
        """Setup environment-aware container configurations"""
        self.logger.info("Setting up environment-aware container configurations...")
       
        try:
            config_manager = ContainerConfigManager(
                environment=self.environment,
                region=self.region
            )
           
            configs_generated = config_manager.generate_container_configs()
            self.logger.info(f"Generated {len(configs_generated)} container configurations")
           
            # Print generated config files content
            for config_name, config_path in configs_generated.items():
                self.logger.info(f"Generated config: {config_name} -> {config_path}")
               
                # Print config.json contents if it exists
                if 'config' in config_path.lower() and config_path.endswith('.json') and os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config_content = f.read()
                        self.logger.info(f" {config_name} content:\n{config_content}")
                    except Exception as e:
                        self.logger.warning(f"Could not read {config_path}: {str(e)}")
               
                # Print buildspec.yml content if it exists
                elif config_name == 'buildspec' and config_path.endswith('.yml') and os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            buildspec_content = f.read()
                        self.logger.info(f" buildspec.yml content:\n{buildspec_content}")
                    except Exception as e:
                        self.logger.warning(f"Could not read buildspec.yml: {str(e)}")
               
                # For other config files, just show they were generated
                else:
                    if os.path.exists(config_path):
                        self.logger.info(f" {config_name} successfully generated")
                    else:
                        self.logger.warning(f"  {config_name} file not found at {config_path}")
           
            # Validate configurations
            validation_results = config_manager.validate_container_configs()
           
            if validation_results.get("errors"):
                self.logger.warning("Configuration validation issues:")
                for error in validation_results["errors"]:
                    self.logger.warning(f"    {error}")
            else:
                self.logger.info(" All container configurations validated successfully")
           
            # Summary of what was generated
            self.logger.info(" Configuration Summary:")
            for config_name, config_path in configs_generated.items():
                file_size = "N/A"
                if os.path.exists(config_path):
                    try:
                        file_size = f"{os.path.getsize(config_path)} bytes"
                    except:
                        pass
                self.logger.info(f"  - {config_name}: {config_path} ({file_size})")
           
            return {
                "status": "success",
                "configs_generated": configs_generated,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
           
        except Exception as e:
            self.logger.error(f"Container configuration failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _deploy_lambda_functions(self) -> Dict[str, Any]:
        """Deploy Lambda functions using enhanced deployer"""
        self.logger.info("Deploying Lambda functions...")
        
        try:
            # Get lambda functions directory
            lambda_functions_dir = sdcp_code_dir / "lambda-functions"
            
            if not lambda_functions_dir.exists():
                raise Exception(f"Lambda functions directory not found: {lambda_functions_dir}")
            
            # Deploy all functions
            deployment_result = self.lambda_deployer.deploy_all_lambda_functions()
            
            return {
                "status": "success",
                "deployment_result": deployment_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Lambda deployment failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _deploy_step_functions(self) -> Dict[str, Any]:
        """Deploy Step Functions"""
        self.logger.info("Deploying Step Functions...")
        
        try:
            # Import step functions deployment
            infrastructure_path = sdcp_code_dir / 'infrastructure'
            if str(infrastructure_path) not in sys.path:
                sys.path.append(str(infrastructure_path))
            
            from step_functions_definitions import get_enhanced_step_functions_with_integration
            
            roles = {"datascientist_role": self.config['datascientist_role']}
            
            result = get_enhanced_step_functions_with_integration(
                environment=self.environment,
                roles=roles,
                account_id=self.account_id,
                region=self.region,
                data_bucket=self.config['data_bucket'],
                model_bucket=self.config['model_bucket'],
                model_prefix=self.config['model_prefix'],
                registry_prefix=self.config['registry_prefix']
            )
            
            return {
                "status": "success",
                "training_pipeline_arn": result.get('training_pipeline'),
                "enhanced_prediction_pipeline_arn": result.get('enhanced_prediction_pipeline'),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Step Functions deployment failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_and_push_containers(self) -> Dict[str, Any]:
        """Build and push all container images - try CodeBuild first, then local Docker"""
        
        try:
            # Step 1: Create ECR repositories if needed
            self.logger.info(" Creating ECR repositories if needed...")
            for repo_name in self.config['containers']:
                try:
                    self.ecr_client.create_repository(repositoryName=repo_name)
                    self.logger.info(f"   Created ECR repository: {repo_name}")
                except self.ecr_client.exceptions.RepositoryAlreadyExistsException:
                    self.logger.info(f"    ECR repository already exists: {repo_name}")
                except Exception as e:
                    self.logger.warning(f"    Could not create repository {repo_name}: {str(e)}")
            
            self.logger.info(" Building Docker containers locally...")
            return self._build_containers_locally_with_details()     
               
            # # Step 2: Build containers using CodeBuild or local Docker
            # try:
            #     # Try CodeBuild first
            #     self.logger.info("  Attempting CodeBuild approach...")
            #     codebuild_result = self._build_containers_via_codebuild()
                
            #     if codebuild_result:
            #         self.logger.info("   Containers built successfully via CodeBuild")
            #         return {
            #             "status": "success",
            #             "build_method": "codebuild",
            #             "message": "All containers built and pushed via CodeBuild",
            #             "timestamp": datetime.now().isoformat()
            #         }
            #     else:
            #         self.logger.warning("    CodeBuild failed, trying local Docker build...")
            #         return self._build_containers_locally_with_details()
                    
            # except Exception as e:
            #     self.logger.warning(f"    CodeBuild not available: {str(e)}")
            #     self.logger.info("   Falling back to local Docker build...")
            #     return self._build_containers_locally_with_details()
                
        except Exception as e:
            error_msg = f"Container build process failed: {str(e)}"
            self.logger.error(f" {error_msg}")
            return {
                "status": "failed",
                "build_method": "unknown",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_containers_via_codebuild(self) -> bool:
        """Build containers using CodeBuild"""
        
        try:
            # Updated path for sdcp_code structure
            scripts_dir = sdcp_code_dir / 'scripts'
            codebuild_script = scripts_dir / 'build_via_codebuild.py'
            
            if not codebuild_script.exists():
                self.logger.warning(f"    CodeBuild script not found: {codebuild_script}")
                return False
            
            self.logger.info(f"   Running CodeBuild script: {codebuild_script}")
            
            # Run CodeBuild script
            result = subprocess.run([
                'python', str(codebuild_script),
                '--region', self.region,
                '--environment', self.environment
            ], capture_output=True, text=True, cwd=str(sdcp_code_dir.parent), timeout=900)  # 15 minute timeout
            
            if result.returncode == 0:
                self.logger.info("   CodeBuild completed successfully")
                if result.stdout:
                    self.logger.info(f"   CodeBuild output:\n{result.stdout}")
                return True
            else:
                self.logger.warning(f"    CodeBuild failed with return code: {result.returncode}")
                if result.stderr:
                    self.logger.warning(f"   CodeBuild stderr: {result.stderr}")
                if result.stdout:
                    self.logger.warning(f"   CodeBuild stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning("   CodeBuild timed out after 15 minutes")
            return False
        except Exception as e:
            self.logger.warning(f"    CodeBuild exception: {str(e)}")
            return False
    
    def _build_containers_locally_with_details(self) -> Dict[str, Any]:
        """Build containers locally using Docker with detailed tracking"""
        
        self.logger.info(" Building containers locally using Docker...")
        
        try:
            # Step 1: Docker ECR login
            self.logger.info("   Logging into ECR...")
            token_response = self.ecr_client.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            endpoint = token_response['authorizationData'][0]['proxyEndpoint']
            
            # Decode token
            import base64
            username, password = base64.b64decode(token).decode().split(':')
            
            login_result = subprocess.run([
                'docker', 'login', '--username', username, '--password-stdin', endpoint
            ], input=password, text=True, capture_output=True)
            
            if login_result.returncode != 0:
                error_msg = f"Docker ECR login failed: {login_result.stderr}"
                self.logger.error(f"   {error_msg}")
                return {
                    "status": "failed",
                    "build_method": "local_docker",
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            
            self.logger.info("   Successfully logged into ECR")
            
            # Step 2: Build each container
            containers_built = []
            containers_failed = []
            
            # Updated container directories for sdcp_code structure
            container_mapping = {
                'energy-preprocessing': 'preprocessing',
                'energy-training': 'training',
                # 'energy-prediction': 'prediction'  # if it exists
            }
            
            for repo_name in self.config['containers']:
                container_subdir = container_mapping.get(repo_name, repo_name.replace('energy-', ''))
                container_dir = sdcp_code_dir / 'containers' / container_subdir
                
                self.logger.info(f"    Building {repo_name}...")
                self.logger.info(f"     Directory: {container_dir}")
                
                if not container_dir.exists():
                    error_msg = f"Container directory not found: {container_dir}"
                    self.logger.warning(f"      {error_msg}")
                    containers_failed.append({
                        "name": repo_name,
                        "error": error_msg,
                        "directory": str(container_dir)
                    })
                    continue
                
                # Check for Dockerfile
                dockerfile_path = container_dir / 'Dockerfile'
                if not dockerfile_path.exists():
                    error_msg = f"Dockerfile not found in {container_dir}"
                    self.logger.error(f"     {error_msg}")
                    containers_failed.append({
                        "name": repo_name,
                        "error": error_msg,
                        "directory": str(container_dir)
                    })
                    continue
                
                try:
                    # Define image URIs
                    base_image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}:latest"
                    env_image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}:{self.environment}-latest"
                    
                    self.logger.info(f"      Building tags: latest, {self.environment}-latest")
                    
                    # Build Docker image
                    build_result = subprocess.run([
                        'docker', 'build', 
                        '-t', base_image_uri, 
                        '-t', env_image_uri, 
                        str(container_dir)
                    ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
                    
                    if build_result.returncode != 0:
                        error_msg = f"Docker build failed for {repo_name}"
                        self.logger.error(f"     {error_msg}")
                        self.logger.error(f"     Build stderr: {build_result.stderr}")
                        containers_failed.append({
                            "name": repo_name,
                            "error": error_msg,
                            "build_stderr": build_result.stderr,
                            "directory": str(container_dir)
                        })
                        continue
                    
                    self.logger.info(f"     Build successful for {repo_name}")
                    
                    # Push both tags
                    push_success = True
                    pushed_tags = []
                    
                    for push_uri in [base_image_uri, env_image_uri]:
                        self.logger.info(f"     Pushing: {push_uri}")
                        
                        push_result = subprocess.run([
                            'docker', 'push', push_uri
                        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                        
                        if push_result.returncode != 0:
                            error_msg = f"Push failed for {push_uri}"
                            self.logger.error(f"     {error_msg}")
                            self.logger.error(f"     Push stderr: {push_result.stderr}")
                            push_success = False
                            break
                        else:
                            self.logger.info(f"     Push successful: {push_uri}")
                            pushed_tags.append(push_uri)
                    
                    if push_success:
                        self.logger.info(f"   {repo_name} - BUILD AND PUSH COMPLETED")
                        containers_built.append({
                            "name": repo_name,
                            "base_image_uri": base_image_uri,
                            "env_image_uri": env_image_uri,
                            "pushed_tags": pushed_tags,
                            "directory": str(container_dir)
                        })
                    else:
                        containers_failed.append({
                            "name": repo_name,
                            "error": "Push operation failed",
                            "partial_pushes": pushed_tags,
                            "directory": str(container_dir)
                        })
                    
                except subprocess.TimeoutExpired:
                    error_msg = f"Build/push timeout for {repo_name}"
                    self.logger.error(f"     {error_msg}")
                    containers_failed.append({
                        "name": repo_name,
                        "error": error_msg,
                        "directory": str(container_dir)
                    })
                except Exception as e:
                    error_msg = f"Exception during build/push: {str(e)}"
                    self.logger.error(f"     {error_msg}")
                    containers_failed.append({
                        "name": repo_name,
                        "error": error_msg,
                        "directory": str(container_dir)
                    })
            
            # Step 3: Generate summary
            total_containers = len(self.config['containers'])
            successful_builds = len(containers_built)
            failed_builds = len(containers_failed)
            
            self.logger.info(f" LOCAL BUILD SUMMARY:")
            self.logger.info(f"   Total containers: {total_containers}")
            self.logger.info(f"   Successfully built: {successful_builds}")
            self.logger.info(f"   Failed: {failed_builds}")
            
            if containers_built:
                self.logger.info(f"   Successfully built containers:")
                for container in containers_built:
                    self.logger.info(f"     {container['name']}")
            
            if containers_failed:
                self.logger.error(f"   Failed containers:")
                for container in containers_failed:
                    self.logger.error(f"     {container['name']}: {container['error']}")
            
            # Determine status
            if successful_builds == total_containers:
                status = "success"
                self.logger.info("   ALL CONTAINERS BUILT SUCCESSFULLY!")
            elif successful_builds > 0:
                status = "partial"
                self.logger.warning(f"    PARTIAL SUCCESS: {successful_builds}/{total_containers}")
            else:
                status = "failed"
                self.logger.error("   ALL CONTAINER BUILDS FAILED!")
            
            return {
                "status": status,
                "build_method": "local_docker",
                "total_containers": total_containers,
                "containers_built": containers_built,
                "containers_failed": containers_failed,
                "successful_builds": successful_builds,
                "failed_builds": failed_builds,
                "success_rate": f"{successful_builds}/{total_containers}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Local container build failed: {str(e)}"
            self.logger.error(f" {error_msg}")
            return {
                "status": "failed",
                "build_method": "local_docker",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def _setup_eventbridge(self) -> Dict[str, Any]:
        """Setup EventBridge rules"""
        self.logger.info("Setting up EventBridge rules...")
        
        try:
            # Import EventBridge setup
            infrastructure_path = sdcp_code_dir / 'infrastructure'
            if str(infrastructure_path) not in sys.path:
                sys.path.append(str(infrastructure_path))
            
            from step_functions_definitions import create_enhanced_eventbridge_rules

            roles = {"datascientist_role": self.config['datascientist_role']}
            
            # Get state machine ARNs
            state_machines = self.stepfunctions_client.list_state_machines()
            
            enhanced_prediction_arn = None
            training_pipeline_arn = None
            
            for sm in state_machines['stateMachines']:
                if sm['name'] == self.config['enhanced_prediction_state_machine']:
                    enhanced_prediction_arn = sm['stateMachineArn']
                elif sm['name'] == self.config['training_state_machine']:
                    training_pipeline_arn = sm['stateMachineArn']
            
            if not enhanced_prediction_arn:
                raise Exception(f"Enhanced prediction state machine not found")
            
            if not training_pipeline_arn:
                raise Exception(f"Training state machine not found")
            
            # Create EventBridge rules for both pipelines
            state_machine_arns = {
                'enhanced_prediction_pipeline': enhanced_prediction_arn,
                'training_pipeline': training_pipeline_arn
            }
            
            rules_result = create_enhanced_eventbridge_rules(
                self.account_id,
                self.region,
                state_machine_arns,
                roles["datascientist_role"],
                self.environment  # Pass environment for naming
            )
            
            self.logger.info("✓ Enhanced EventBridge rules created successfully")
            
            return {
                "status": "success",
                "rules_created": rules_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"EventBridge setup failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_complete_deployment(self) -> Dict[str, Any]:
        """Validate complete deployment"""
        self.logger.info("Validating complete deployment...")
        
        try:
            validation_results = {
                "lambda_functions": [],
                "step_functions": [],
                "eventbridge_rules": [],
                "containers": []
            }
            
            # Validate Lambda functions
            expected_functions = [
                f"{self.config['lambda_prefix']}-model-registry",
                f"{self.config['lambda_prefix']}-endpoint-manager",
                f"{self.config['lambda_prefix']}-profile-predictor",
                f"{self.config['lambda_prefix']}-prediction-orchestrator",
                f"{self.config['lambda_prefix']}-cost-optimizer"
            ]
            
            for function_name in expected_functions:
                try:
                    self.lambda_client.get_function(FunctionName=function_name)
                    validation_results["lambda_functions"].append({
                        "name": function_name,
                        "status": "EXISTS"
                    })
                except Exception:
                    validation_results["lambda_functions"].append({
                        "name": function_name,
                        "status": "MISSING"
                    })
            
            # Validate Step Functions
            for sf_name in [self.config['training_state_machine'], self.config['enhanced_prediction_state_machine']]:
                try:
                    sf_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{sf_name}"
                    self.stepfunctions_client.describe_state_machine(stateMachineArn=sf_arn)
                    validation_results["step_functions"].append({
                        "name": sf_name,
                        "status": "EXISTS"
                    })
                except Exception:
                    validation_results["step_functions"].append({
                        "name": sf_name,
                        "status": "MISSING"
                    })
            
            # Assess overall validation
            lambda_success = len([f for f in validation_results["lambda_functions"] if f["status"] == "EXISTS"])
            sf_success = len([f for f in validation_results["step_functions"] if f["status"] == "EXISTS"])
            
            overall_status = "SUCCESS" if lambda_success >= 3 and sf_success >= 2 else "PARTIAL"
            
            return {
                "status": overall_status,
                "validation_details": validation_results,
                "summary": {
                    "lambda_functions_deployed": lambda_success,
                    "step_functions_deployed": sf_success
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Deployment validation failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        self.logger.info("Running integration tests...")
        
        try:
            # Basic connectivity tests
            test_results = {
                "aws_connectivity": False,
                "s3_access": False,
                "lambda_accessibility": False,
                "step_functions_accessibility": False
            }
            
            # Test AWS connectivity
            try:
                self.iam_client.get_user()
                test_results["aws_connectivity"] = True
            except Exception:
                pass
            
            # Test S3 access
            try:
                self.s3_client.head_bucket(Bucket=self.config['data_bucket'])
                test_results["s3_access"] = True
            except Exception:
                pass
            
            # Test Lambda accessibility
            try:
                functions = self.lambda_client.list_functions()
                test_results["lambda_accessibility"] = len(functions['Functions']) > 0
            except Exception:
                pass
            
            # Test Step Functions accessibility
            try:
                state_machines = self.stepfunctions_client.list_state_machines()
                test_results["step_functions_accessibility"] = len(state_machines['stateMachines']) > 0
            except Exception:
                pass
            
            # Calculate overall success
            passed_tests = sum(1 for result in test_results.values() if result)
            total_tests = len(test_results)
            success_rate = (passed_tests / total_tests) * 100
            
            return {
                "status": "SUCCESS" if success_rate >= 75 else "PARTIAL",
                "test_results": test_results,
                "summary": {
                    "tests_passed": passed_tests,
                    "total_tests": total_tests,
                    "success_rate": success_rate
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Integration tests failed: {str(e)}")
            return {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_deployment_artifacts(self, deployment_summary: Dict[str, Any]):
        """Generate CI/CD deployment artifacts"""
        self.logger.info("Generating deployment artifacts...")
        
        try:
            # Generate main deployment summary
            summary_filename = f"deployment-summary-{self.environment}-{self.github_run_id or 'local'}.json"
            
            with open(summary_filename, 'w') as f:
                json.dump(deployment_summary, f, indent=2, default=str)
            
            self.logger.info(f"Generated deployment summary: {summary_filename}")
            
            # Generate timing report
            timing_filename = f"deployment-timing-{self.environment}-{self.github_run_id or 'local'}.json"
            
            timing_report = {
                "environment": self.environment,
                "total_deployment_time_minutes": deployment_summary["timing"]["total_deployment_minutes"],
                "step_timings": deployment_summary["timing"],
                "timestamp": datetime.now().isoformat()
            }
            
            with open(timing_filename, 'w') as f:
                json.dump(timing_report, f, indent=2, default=str)
            
            self.logger.info(f"Generated timing report: {timing_filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate deployment artifacts: {str(e)}")
    
    def _assess_overall_success(self, deployment_summary: Dict[str, Any]) -> bool:
        """Assess overall deployment success"""
        
        try:
            # Check if any steps failed
            failed_steps = deployment_summary.get("steps_failed", [])
            if failed_steps:
                return False
            
            # Check validation results
            environment_validation = deployment_summary.get("validation_results", {}).get("environment", {})
            if not environment_validation.get("environment_ready", True):  # Default True if skipped
                return False
            
            deployment_validation = deployment_summary.get("validation_results", {}).get("deployment", {})
            if deployment_validation.get("status") == "FAILED":
                return False
            
            # Check resource deployment results
            resources = deployment_summary.get("resources_deployed", {})
            
            lambda_result = resources.get("lambda_functions", {})
            if lambda_result.get("status") == "failed":
                return False
            
            sf_result = resources.get("step_functions", {})
            if sf_result.get("status") == "failed":
                return False
            
            # If we get here, deployment was successful
            return True
            
        except Exception as e:
            self.logger.error(f"Error assessing overall success: {str(e)}")
            return False
    
    def _generate_failure_report(self, deployment_summary: Dict[str, Any], error: Exception):
        """Generate failure report"""
        failure_filename = f"deployment-failure-{self.environment}-{self.github_run_id or 'local'}.json"
        
        failure_report = {
            **deployment_summary,
            "failure_details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "failed_step": deployment_summary["steps_executed"][-1] if deployment_summary["steps_executed"] else None
            }
        }
        
        with open(failure_filename, 'w') as f:
            json.dump(failure_report, f, indent=2, default=str)
        
        self.logger.error(f"Generated failure report: {failure_filename}")


def main():
    """Main function for unified MLOps deployment"""
    parser = argparse.ArgumentParser(description='Unified MLOps CI/CD Deployment')
    parser.add_argument('--environment', required=True,
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--sagemaker-role-arn', required=True,
                       help='SageMaker role ARN')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--ci-cd-mode', action='store_true',
                       help='Enable CI/CD mode')
    parser.add_argument('--github-run-id', help='GitHub Actions run ID')
    parser.add_argument('--deployment-bucket', help='Override deployment S3 bucket')
    parser.add_argument('--model-bucket', help='Override model S3 bucket')
    parser.add_argument('--skip-environment-validation', action='store_true',
                       help='Skip environment validation (if handled by workflow)')
    
    args = parser.parse_args()
    
    try:
        # Initialize unified deployment
        deployment = UnifiedMLOpsDeployment(
            environment=args.environment,
            sagemaker_role_arn=args.sagemaker_role_arn,
            region=args.region,
            ci_cd_mode=args.ci_cd_mode,
            github_run_id=args.github_run_id,
        )
        
        # Run complete deployment
        result = deployment.deploy_complete_mlops_pipeline(
            skip_environment_validation=args.skip_environment_validation,
            deployment_bucket=args.deployment_bucket,
            model_bucket=args.model_bucket
        )
        
        print(f"Deployment completed successfully for {args.environment} environment")
        print(f"Status: {result['deployment_status']}")
        print(f"Duration: {result['timing']['total_deployment_minutes']:.2f} minutes")
        print(f"Overall Success: {result['overall_success']}")
        
        # Print step summary
        print("\nStep Execution Summary:")
        for step in result.get('steps_executed', []):
            step_name = step.get('step', 'Unknown')
            step_status = step.get('status', 'Unknown')
            print(f"  - {step_name}: {step_status}")
        
        # Print timing breakdown
        print(f"\nTiming Breakdown:")
        for step_name, duration in result.get('timing', {}).items():
            if step_name.startswith('total_'):
                continue
            print(f"  - {step_name}: {duration:.2f} seconds")
        
        return 0 if result['overall_success'] else 1
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
