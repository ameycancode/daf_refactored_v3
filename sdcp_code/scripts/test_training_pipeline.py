#!/usr/bin/env python3
"""
Training Pipeline Test Script
Tests the training Step Functions pipeline and validates results
"""

import os
import sys
import json
import time
import boto3
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

class TrainingPipelineTest:
    def __init__(self, region: str = "us-west-2"):
        """Initialize training pipeline test"""
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

        # Initialize AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)

        self.logger.info(f"Training pipeline test initialized for region: {region}")
        self.logger.info(f"Account ID: {self.account_id}")

    def test_training_pipeline(self, environment: str = "dev", timeout_minutes: int = 45) -> Dict[str, Any]:
        """Test the training pipeline Step Function"""
        try:
            # Determine Step Function name based on environment
            step_function_name = f"energy-forecasting-training-pipeline"
            step_function_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{step_function_name}"
            
            self.logger.info(f"Testing training pipeline: {step_function_name}")
            self.logger.info(f"Step Function ARN: {step_function_arn}")
            
            # Validate Step Function exists
            try:
                self.stepfunctions_client.describe_state_machine(stateMachineArn=step_function_arn)
                self.logger.info("✓ Step Function exists and is accessible")
            except Exception as e:
                raise Exception(f"Step Function not found or not accessible: {str(e)}")
            
            # Prepare execution input
            execution_name = f"training-test-{environment}-{int(time.time())}"
            execution_input = {
                "test_mode": True,
                "environment": environment,
                "all_profiles": True,
                "profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            }

            self.logger.info(f"Starting execution: {execution_name}")
            self.logger.info(f"Execution input: {json.dumps(execution_input, indent=2)}")

            # Start execution
            start_response = self.stepfunctions_client.start_execution(
                stateMachineArn=step_function_arn,
                name=execution_name,
                input=json.dumps(execution_input)
            )
            
            execution_arn = start_response['executionArn']
            self.logger.info(f"✓ Execution started: {execution_arn}")

            # Monitor execution
            result = self._monitor_execution(execution_arn, timeout_minutes)
            
            # Validate results
            validation_result = self._validate_training_results(result, environment)
            
            return {
                "status": "success",
                "execution_arn": execution_arn,
                "execution_name": execution_name,
                "result": result,
                "validation": validation_result,
                "environment": environment
            }
            
        except Exception as e:
            self.logger.error(f"Training pipeline test failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "environment": environment
            }
    
    def _monitor_execution(self, execution_arn: str, timeout_minutes: int) -> Dict[str, Any]:
        """Monitor Step Function execution"""
        self.logger.info(f"Monitoring execution for up to {timeout_minutes} minutes...")
        
        start_time = datetime.now()
        timeout_time = start_time + timedelta(minutes=timeout_minutes)
        
        while datetime.now() < timeout_time:
            try:
                response = self.stepfunctions_client.describe_execution(executionArn=execution_arn)
                status = response['status']
                
                elapsed_time = datetime.now() - start_time
                self.logger.info(f"Execution status: {status} (elapsed: {elapsed_time})")

                if status == 'SUCCEEDED':
                    self.logger.info("✓ Training pipeline execution completed successfully")
                    return {
                        "status": status,
                        "output": json.loads(response.get('output', '{}')),
                        "start_date": response['startDate'].isoformat(),
                        "stop_date": response['stopDate'].isoformat(),
                        "execution_time": str(elapsed_time)
                    }
                elif status in ['FAILED', 'TIMED_OUT', 'ABORTED']:
                    error_message = response.get('error', 'Unknown error')
                    self.logger.error(f"✗ Training pipeline execution failed: {status}")
                    self.logger.error(f"Error: {error_message}")
                    return {
                        "status": status,
                        "error": error_message,
                        "start_date": response['startDate'].isoformat(),
                        "stop_date": response.get('stopDate', datetime.now()).isoformat() if response.get('stopDate') else datetime.now().isoformat(),
                        "execution_time": str(elapsed_time)
                    }
                
                # Wait before next check
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error monitoring execution: {str(e)}")
                time.sleep(30)
                continue
        
        # Timeout reached
        self.logger.warning(f"✗ Execution monitoring timed out after {timeout_minutes} minutes")
        return {
            "status": "TIMEOUT",
            "error": f"Monitoring timed out after {timeout_minutes} minutes",
            "execution_time": str(datetime.now() - start_time)
        }
    
    def _validate_training_results(self, result: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Validate training pipeline results"""
        validation = {
            "execution_successful": False,
            "profiles_trained": [],
            "models_created": [],
            "performance_validation": {},
            "s3_artifacts_check": {},
            "issues": []
        }
        
        try:
            # Check execution status
            if result.get("status") == "SUCCEEDED":
                validation["execution_successful"] = True
                self.logger.info("✓ Training execution completed successfully")

                # Extract output information
                output = result.get("output", {})
                
                # Check for trained profiles
                if "trained_profiles" in output:
                    validation["profiles_trained"] = output["trained_profiles"]
                    self.logger.info(f"✓ Profiles trained: {validation['profiles_trained']}")

                # Check for created models
                if "models_created" in output:
                    validation["models_created"] = output["models_created"]
                    self.logger.info(f"✓ Models created: {validation['models_created']}")

                # Check performance metrics
                if "performance_metrics" in output:
                    validation["performance_validation"] = self._validate_performance_metrics(
                        output["performance_metrics"]
                    )
                
                # Check S3 artifacts
                validation["s3_artifacts_check"] = self._check_s3_artifacts(environment)
                
                # Validate expected profiles were trained
                expected_profiles = ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
                trained_profiles = validation["profiles_trained"]
                
                missing_profiles = set(expected_profiles) - set(trained_profiles)
                if missing_profiles:
                    validation["issues"].append(f"Missing trained profiles: {list(missing_profiles)}")
                    self.logger.warning(f" Missing profiles: {missing_profiles}")
                else:
                    self.logger.info("✓ All expected profiles were trained")

            else:
                validation["execution_successful"] = False
                validation["issues"].append(f"Execution failed with status: {result.get('status')}")
                self.logger.error(f"✗ Training execution failed: {result.get('status')}")
                if result.get("error"):
                    validation["issues"].append(f"Error details: {result.get('error')}")
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Error during validation: {str(e)}")

        return validation
    
    def _validate_performance_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model performance metrics"""
        performance_validation = {
            "profiles_meeting_thresholds": [],
            "profiles_below_thresholds": [],
            "performance_summary": {}
        }
        
        # Performance thresholds
        thresholds = {
            "RNN": {"min_r2": 0.85, "max_mape": 5.0, "max_rmse": 0.1},
            "RN": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12},
            "M": {"min_r2": 0.85, "max_mape": 4.0, "max_rmse": 0.08},
            "S": {"min_r2": 0.82, "max_mape": 5.5, "max_rmse": 0.10},
            "AGR": {"min_r2": 0.80, "max_mape": 7.0, "max_rmse": 0.15},
            "L": {"min_r2": 0.75, "max_mape": 8.0, "max_rmse": 0.20},
            "A6": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12}
        }
        
        for profile, profile_metrics in metrics.items():
            if profile in thresholds:
                threshold = thresholds[profile]
                
                r2_score = profile_metrics.get("r2_score", 0)
                mape = profile_metrics.get("mape", 100)
                rmse = profile_metrics.get("rmse", 999)
                
                meets_threshold = (
                    r2_score >= threshold["min_r2"] and
                    mape <= threshold["max_mape"] and
                    rmse <= threshold["max_rmse"]
                )
                
                performance_validation["performance_summary"][profile] = {
                    "r2_score": r2_score,
                    "mape": mape,
                    "rmse": rmse,
                    "meets_threshold": meets_threshold,
                    "threshold": threshold
                }
                
                if meets_threshold:
                    performance_validation["profiles_meeting_thresholds"].append(profile)
                    self.logger.info(f"✓ {profile}: Performance meets thresholds (R²={r2_score:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.3f})")
                else:
                    performance_validation["profiles_below_thresholds"].append(profile)
                    self.logger.warning(f" {profile}: Performance below thresholds (R²={r2_score:.3f}, MAPE={mape:.2f}%, RMSE={rmse:.3f})")

        return performance_validation
    
    def _check_s3_artifacts(self, environment: str) -> Dict[str, Any]:
        """Check for expected S3 artifacts from training"""
        s3_check = {
            "models_in_s3": [],
            "results_in_s3": [],
            "missing_artifacts": [],
            "s3_accessible": False
        }
        
        try:
            # Check model bucket
            model_bucket = f"sdcp-{environment}-sagemaker-energy-forecasting-models"
            data_bucket = f"sdcp-{environment}-sagemaker-energy-forecasting-data"
            
            # Test S3 access
            try:
                self.s3_client.head_bucket(Bucket=model_bucket)
                s3_check["s3_accessible"] = True
                self.logger.info("✓ S3 model bucket accessible")
            except Exception as e:
                self.logger.warning(f" S3 model bucket not accessible: {str(e)}")
                return s3_check
            
            # Check for model files
            try:
                model_response = self.s3_client.list_objects_v2(
                    Bucket=model_bucket,
                    Prefix="xgboost/"
                )
                
                if "Contents" in model_response:
                    models = [obj["Key"] for obj in model_response["Contents"] if obj["Key"].endswith(".pkl")]
                    s3_check["models_in_s3"] = models
                    self.logger.info(f"✓ Found {len(models)} model files in S3")
                else:
                    self.logger.info(" No model files found in S3 (may be expected for test)")
            except Exception as e:
                self.logger.warning(f" Error checking model files: {str(e)}")

            # Check for training results
            try:
                results_response = self.s3_client.list_objects_v2(
                    Bucket=data_bucket,
                    Prefix="archived_folders/forecasting/data/xgboost/train_results/"
                )
                
                if "Contents" in results_response:
                    results = [obj["Key"] for obj in results_response["Contents"]]
                    s3_check["results_in_s3"] = results
                    self.logger.info(f"✓ Found {len(results)} result files in S3")
                else:
                    self.logger.info(" No training result files found in S3")
            except Exception as e:
                self.logger.warning(f" Error checking result files: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error during S3 artifact check: {str(e)}")
            s3_check["missing_artifacts"].append(f"S3 check failed: {str(e)}")
        
        return s3_check
    
    def get_execution_history(self, execution_arn: str) -> List[Dict[str, Any]]:
        """Get detailed execution history for debugging"""
        try:
            response = self.stepfunctions_client.get_execution_history(
                executionArn=execution_arn,
                maxResults=100,
                reverseOrder=True
            )
            
            return response.get("events", [])
        except Exception as e:
            self.logger.error(f"Error getting execution history: {str(e)}")
            return []

def main():
    """Main function for training pipeline test"""
    parser = argparse.ArgumentParser(description='Test Training Pipeline')
    parser.add_argument('--environment', default='dev',
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--timeout-minutes', type=int, default=45,
                       help='Test timeout in minutes')
    parser.add_argument('--all-profiles', action='store_true',
                       help='Test all 7 profiles')
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = TrainingPipelineTest(region=args.region)
        
        print("="*60)
        print("TRAINING PIPELINE TEST")
        print("="*60)
        print(f"Environment: {args.environment}")
        print(f"Timeout: {args.timeout_minutes} minutes")
        print(f"All Profiles: {args.all_profiles}")
        print("="*60)
        
        # Run test
        result = test.test_training_pipeline(
            environment=args.environment,
            timeout_minutes=args.timeout_minutes
        )
        
        # Print results
        print("="*60)
        print("TEST RESULTS")
        print("="*60)
        
        if result["status"] == "success":
            print("✓ Training pipeline test PASSED")
            
            validation = result["validation"]
            if validation["execution_successful"]:
                print(f"✓ Execution completed successfully")
                print(f"✓ Profiles trained: {len(validation['profiles_trained'])}")
                print(f"✓ Models created: {len(validation['models_created'])}")

                if validation["performance_validation"]:
                    perf = validation["performance_validation"]
                    print(f"✓ Profiles meeting thresholds: {len(perf['profiles_meeting_thresholds'])}")
                    if perf["profiles_below_thresholds"]:
                        print(f" Profiles below thresholds: {perf['profiles_below_thresholds']}")
            
            if validation["issues"]:
                print(" Issues found:")
                for issue in validation["issues"]:
                    print(f"  - {issue}")
            
            # Save results
            result_filename = f"training_test_results_{args.environment}_{int(time.time())}.json"
            with open(result_filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"✓ Results saved to: {result_filename}")

            sys.exit(0)
        else:
            print("✗ Training pipeline test FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
