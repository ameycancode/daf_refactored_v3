# =============================================================================
# ENHANCED TRAINING PIPELINE TESTING - scripts/test_training_pipeline_cicd.py
# =============================================================================
"""
Enhanced Training Pipeline Testing with CI/CD Integration
Extends existing training pipeline tests with CI/CD reporting capabilities
"""

import os
import sys
import json
import argparse
import logging
import time
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import existing test functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from test_training_pipeline import TrainingPipelineTest
except ImportError:
    print("Error: Could not import test_training_pipeline.py")
    print("Make sure the original test script exists in the same directory")
    sys.exit(1)

class CICDTrainingPipelineTest(TrainingPipelineTest):
    """
    CI/CD Enhanced Training Pipeline Test
    Extends the original test with CI/CD-specific features
    """
   
    def __init__(self, environment: str, region: str = "us-west-2",
                 ci_cd_mode: bool = False, github_run_id: Optional[str] = None):
        """Initialize CI/CD enhanced training pipeline test"""
       
        # Initialize parent class
        super().__init__(region=region)
       
        self.environment = environment
        self.ci_cd_mode = ci_cd_mode
        self.github_run_id = github_run_id
        self.test_start_time = datetime.now()
       
        # Environment-specific configuration
        self.env_config = self._get_environment_config()
       
        # CI/CD specific logging setup
        if ci_cd_mode:
            self._setup_cicd_logging()
       
        print(f"CI/CD Enhanced Training Pipeline Test initialized for {environment}")
        print(f"CI/CD Mode: {ci_cd_mode}")
        print(f"GitHub Run ID: {github_run_id}")
   
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        configs = {
            "dev": {
                "step_function_name": "energy-forecasting-training-pipeline",
                "timeout_minutes": 30,
                "expected_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            },
            "preprod": {
                "step_function_name": "energy-forecasting-training-pipeline",
                "timeout_minutes": 45,
                "expected_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            },
            "prod": {
                "step_function_name": "energy-forecasting-training-pipeline",
                "timeout_minutes": 60,
                "expected_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            }
        }
        return configs.get(self.environment, configs["dev"])
   
    def _setup_cicd_logging(self):
        """Setup CI/CD specific logging"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [CI/CD-TEST] - %(levelname)s - %(message)s'
        )
       
        if self.github_run_id:
            log_filename = f"training-test-{self.environment}-{self.github_run_id}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
   
    def test_training_pipeline_cicd(self, timeout_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Test training pipeline with CI/CD enhancements
        """
        test_summary = {
            "environment": self.environment,
            "test_type": "training_pipeline",
            "ci_cd_mode": self.ci_cd_mode,
            "github_run_id": self.github_run_id,
            "test_start_time": self.test_start_time.isoformat(),
            "test_status": "IN_PROGRESS",
            "step_function_execution": {},
            "performance_metrics": {},
            "validation_results": {},
            "error_details": None
        }
       
        try:
            print("="*80)
            print("STARTING CI/CD ENHANCED TRAINING PIPELINE TEST")
            print(f"Environment: {self.environment}")
            print(f"Step Function: {self.env_config['step_function_name']}")
            print("="*80)
           
            # Use provided timeout or environment default
            timeout_mins = timeout_minutes or self.env_config["timeout_minutes"]
           
            # Step 1: Validate Step Function exists
            self._validate_step_function_exists(test_summary)
           
            # Step 2: Start execution
            execution_result = self._start_training_execution(test_summary, timeout_mins)
           
            # Step 3: Monitor execution
            monitoring_result = self._monitor_training_execution(
                execution_result["execution_arn"],
                timeout_mins,
                test_summary
            )

            print(f"Monitoring Result: {monitoring_result}")
           
            # Step 4: Validate results
            validation_result = self._validate_training_results(
                monitoring_result,
                test_summary
            )

            print(f"Validation Result: {validation_result}")
           
            # Step 5: Performance analysis
            performance_result = self._analyze_training_performance(
                monitoring_result,
                test_summary
            )
            print(f"Performance Result: {performance_result}")
           
            test_summary["test_status"] = "SUCCESS"
            test_summary["test_end_time"] = datetime.now().isoformat()
            test_summary["test_duration"] = str(datetime.now() - self.test_start_time)
           
            print("="*80)
            print("CI/CD ENHANCED TRAINING PIPELINE TEST COMPLETED SUCCESSFULLY")
            print(f"Total Duration: {test_summary['test_duration']}")
            print("="*80)

            print(f"Test Summary: {json.dumps(test_summary, indent=2)}")
           
            # Generate CI/CD artifacts
            self._generate_test_artifacts(test_summary)
           
            return test_summary
           
        except Exception as e:
            test_summary["test_status"] = "FAILED"
            test_summary["error_details"] = str(e)
            test_summary["test_end_time"] = datetime.now().isoformat()

            self.logger.error(f"CI/CD Enhanced training pipeline test failed: {str(e)}")
            print(f"CI/CD Enhanced training pipeline test failed: {str(e)}")
           
            # Generate failure report
            self._generate_failure_report(test_summary, e)
           
            raise
   
    def _validate_step_function_exists(self, test_summary: Dict[str, Any]):
        """Validate that the Step Function exists"""
        print("Validating Step Function exists...")
       
        try:
            step_function_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.env_config['step_function_name']}"
           
            response = self.stepfunctions_client.describe_state_machine(
                stateMachineArn=step_function_arn
            )
           
            test_summary["step_function_validation"] = {
                "exists": True,
                "arn": response["stateMachineArn"],
                "status": response["status"],
                "creation_date": response["creationDate"].isoformat()
            }
           
            print(f"✓ Step Function validated: {self.env_config['step_function_name']}")
           
        except Exception as e:
            test_summary["step_function_validation"] = {
                "exists": False,
                "error": str(e)
            }
            raise Exception(f"Step Function validation failed: {str(e)}")
   
    def _start_training_execution(self, test_summary: Dict[str, Any], timeout_minutes: int) -> Dict[str, Any]:
        """Start training pipeline execution"""
        print("Starting training pipeline execution...")
       
        step_function_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.env_config['step_function_name']}"
       
        # Prepare execution input
        # execution_input = {
        #     "test_mode": True,
        #     "environment": self.environment,
        #     "github_run_id": self.github_run_id,
        #     "timeout_minutes": timeout_minutes,
        #     "all_profiles": True
        # }

        execution_input = {
            "PreprocessingJobName": f"test-preprocessing-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingJobName": f"test-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "TrainingDate": datetime.now().strftime('%Y%m%d'),
            "PreprocessingImageUri": f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/energy-preprocessing:latest",
            "TrainingImageUri": f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/energy-training:latest"
        }        
       
        execution_name = f"cicd-training-test-{self.environment}-{int(time.time())}"
        if self.github_run_id:
            execution_name += f"-{self.github_run_id[:8]}"
       
        try:
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=step_function_arn,
                name=execution_name,
                input=json.dumps(execution_input)
            )
           
            execution_result = {
                "execution_arn": response["executionArn"],
                "execution_name": execution_name,
                "start_time": datetime.now().isoformat(),
                "input": execution_input
            }
           
            test_summary["step_function_execution"] = execution_result
           
            print(f"✓ Training execution started: {execution_name}")
            print(f"  Execution ARN: {response['executionArn']}")
           
            return execution_result
           
        except Exception as e:
            raise Exception(f"Failed to start training execution: {str(e)}")
   
    def _monitor_training_execution(self, execution_arn: str, timeout_minutes: int,
                                  test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor training pipeline execution"""
        print(f"Monitoring training execution (timeout: {timeout_minutes} minutes)...")
       
        start_time = datetime.now()
        timeout_time = start_time + timedelta(minutes=timeout_minutes)
       
        monitoring_result = {
            "status": "UNKNOWN",
            "execution_history": [],
            "final_output": None,
            "monitoring_duration": None
        }
       
        try:
            while datetime.now() < timeout_time:
                response = self.stepfunctions_client.describe_execution(
                    executionArn=execution_arn
                )
               
                current_status = response["status"]
                monitoring_result["status"] = current_status
               
                # Log status updates
                if not monitoring_result["execution_history"] or monitoring_result["execution_history"][-1]["status"] != current_status:
                    status_update = {
                        "timestamp": datetime.now().isoformat(),
                        "status": current_status,
                        "elapsed_time": str(datetime.now() - start_time)
                    }
                    monitoring_result["execution_history"].append(status_update)
                    print(f"Training execution status: {current_status} (elapsed: {status_update['elapsed_time']})")
               
                # Check if execution is complete
                if current_status in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
                    if "output" in response:
                        monitoring_result["final_output"] = json.loads(response["output"])
                   
                    monitoring_result["monitoring_duration"] = str(datetime.now() - start_time)
                   
                    print(f"Training execution completed with status: {current_status}")
                    break
               
                # Wait before next check
                time.sleep(30)
            else:
                # Timeout reached
                monitoring_result["status"] = "TIMEOUT"
                monitoring_result["monitoring_duration"] = str(datetime.now() - start_time)
                self.logger.warning(f"Training execution monitoring timed out after {timeout_minutes} minutes")
                print(f"⚠ Training execution monitoring timed out after {timeout_minutes} minutes")
           
            return monitoring_result
           
        except Exception as e:
            monitoring_result["error"] = str(e)
            raise Exception(f"Training execution monitoring failed: {str(e)}")
   
    def _validate_training_results(self, monitoring_result: Dict[str, Any],
                                 test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training pipeline results"""
        print("Validating training pipeline results...")
       
        validation_result = {
            "execution_successful": False,
            "profiles_trained": [],
            "models_created": [],
            "performance_metrics": {},
            "issues_found": []
        }
       
        try:
            # Check execution status
            if monitoring_result["status"] == "SUCCEEDED":
                validation_result["execution_successful"] = True
                print("✓ Training execution completed successfully")
               
                # Validate output structure
                if monitoring_result.get("final_output"):
                    output = monitoring_result["final_output"]
                   
                    # Check for trained profiles
                    if "trained_profiles" in output:
                        validation_result["profiles_trained"] = output["trained_profiles"]
                        print(f"✓ Profiles trained: {len(validation_result['profiles_trained'])}")
                   
                    # Check for created models
                    if "models_created" in output:
                        validation_result["models_created"] = output["models_created"]
                        print(f"✓ Models created: {len(validation_result['models_created'])}")
                   
                    # Extract performance metrics
                    if "performance_metrics" in output:
                        validation_result["performance_metrics"] = output["performance_metrics"]
                        print("✓ Performance metrics available")
               
                # Validate expected profiles were trained
                expected_profiles = set(self.env_config["expected_profiles"])
                actual_profiles = set(validation_result["profiles_trained"])
               
                if not expected_profiles.issubset(actual_profiles):
                    missing_profiles = expected_profiles - actual_profiles
                    validation_result["issues_found"].append(f"Missing profiles: {list(missing_profiles)}")
                    self.logger.warning(f"⚠ Missing profiles: {missing_profiles}")
                    print(f"⚠ Missing profiles: {missing_profiles}")
                else:
                    print("✓ All expected profiles trained")
               
            else:
                validation_result["execution_successful"] = False
                validation_result["issues_found"].append(f"Execution failed with status: {monitoring_result['status']}")
                self.logger.error(f"✗ Training execution failed: {monitoring_result['status']}")
                print(f"✗ Training execution failed: {monitoring_result['status']}")
           
            test_summary["validation_results"] = validation_result
            return validation_result
           
        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["issues_found"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Training result validation failed: {str(e)}")
            print(f"Training result validation failed: {str(e)}")
            return validation_result
   
    def _analyze_training_performance(self, monitoring_result: Dict[str, Any],
                                    test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training performance metrics"""
        print("Analyzing training performance...")
       
        performance_analysis = {
            "execution_time": monitoring_result.get("monitoring_duration", "Unknown"),
            "status": monitoring_result.get("status", "Unknown"),
            "profile_performance": {},
            "overall_assessment": "Unknown"
        }
       
        try:
            if monitoring_result.get("final_output") and "performance_metrics" in monitoring_result["final_output"]:
                metrics = monitoring_result["final_output"]["performance_metrics"]
               
                # Analyze per-profile performance
                for profile in self.env_config["expected_profiles"]:
                    if profile in metrics:
                        profile_metrics = metrics[profile]
                        performance_analysis["profile_performance"][profile] = {
                            "r2_score": profile_metrics.get("r2_score", 0),
                            "mape": profile_metrics.get("mape", 100),
                            "rmse": profile_metrics.get("rmse", 999),
                            "training_time": profile_metrics.get("training_time", "Unknown"),
                            "meets_threshold": self._check_performance_threshold(profile, profile_metrics)
                        }
               
                # Overall assessment
                successful_profiles = sum(1 for p in performance_analysis["profile_performance"].values() if p["meets_threshold"])
                total_profiles = len(self.env_config["expected_profiles"])
               
                if successful_profiles == total_profiles:
                    performance_analysis["overall_assessment"] = "EXCELLENT"
                elif successful_profiles >= total_profiles * 0.8:
                    performance_analysis["overall_assessment"] = "GOOD"
                elif successful_profiles >= total_profiles * 0.6:
                    performance_analysis["overall_assessment"] = "FAIR"
                else:
                    performance_analysis["overall_assessment"] = "POOR"
               
                print(f"Performance assessment: {performance_analysis['overall_assessment']}")
                print(f"Successful profiles: {successful_profiles}/{total_profiles}")
           
            test_summary["performance_metrics"] = performance_analysis
            return performance_analysis
           
        except Exception as e:
            performance_analysis["error"] = str(e)
            self.logger.error(f"Performance analysis failed: {str(e)}")
            print(f"Performance analysis failed: {str(e)}")
            return performance_analysis
   
    def _check_performance_threshold(self, profile: str, metrics: Dict[str, Any]) -> bool:
        """Check if performance metrics meet thresholds"""
        thresholds = {
            "RNN": {"min_r2": 0.85, "max_mape": 5.0, "max_rmse": 0.1},
            "RN": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12},
            "M": {"min_r2": 0.85, "max_mape": 4.0, "max_rmse": 0.08},
            "S": {"min_r2": 0.82, "max_mape": 5.5, "max_rmse": 0.10},
            "AGR": {"min_r2": 0.80, "max_mape": 7.0, "max_rmse": 0.15},
            "L": {"min_r2": 0.75, "max_mape": 8.0, "max_rmse": 0.20},
            "A6": {"min_r2": 0.80, "max_mape": 6.0, "max_rmse": 0.12}
        }
       
        if profile not in thresholds:
            return False
       
        threshold = thresholds[profile]
       
        r2_ok = metrics.get("r2_score", 0) >= threshold["min_r2"]
        mape_ok = metrics.get("mape", 100) <= threshold["max_mape"]
        rmse_ok = metrics.get("rmse", 999) <= threshold["max_rmse"]
       
        return r2_ok and mape_ok and rmse_ok
   
    def _generate_test_artifacts(self, test_summary: Dict[str, Any]):
        """Generate CI/CD test artifacts"""
        print("Generating CI/CD test artifacts...")
       
        # Generate test results JSON
        results_filename = f"training-test-results-{self.environment}-{self.github_run_id or 'local'}.json"
        with open(results_filename, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
       
        print(f"Generated test results: {results_filename}")
       
        # Generate GitHub Actions summary if in CI/CD mode
        if self.ci_cd_mode and os.getenv('GITHUB_STEP_SUMMARY'):
            self._generate_github_test_summary(test_summary)
   
    def _generate_github_test_summary(self, test_summary: Dict[str, Any]):
        """Generate GitHub Actions test summary"""
        summary_file = os.getenv('GITHUB_STEP_SUMMARY')
        if not summary_file:
            return
       
        with open(summary_file, 'a') as f:
            f.write(f"""
## 🎯 Training Pipeline Test Results - {self.environment.upper()}

| Parameter | Value |
|-----------|-------|
| Test Status | {'✅ PASSED' if test_summary['test_status'] == 'SUCCESS' else '❌ FAILED'} |
| Environment | `{self.environment}` |
| Duration | {test_summary.get('test_duration', 'N/A')} |
| Profiles Tested | {len(test_summary.get('validation_results', {}).get('profiles_trained', []))} |

### Performance Analysis
""")
           
            if 'performance_metrics' in test_summary:
                perf = test_summary['performance_metrics']
                f.write(f"- **Overall Assessment**: {perf.get('overall_assessment', 'Unknown')}\n")
                f.write(f"- **Execution Time**: {perf.get('execution_time', 'Unknown')}\n")
               
                if 'profile_performance' in perf:
                    f.write("\n### Profile Performance\n")
                    f.write("| Profile | R² Score | MAPE | RMSE | Threshold Met |\n")
                    f.write("|---------|----------|------|------|---------------|\n")
                   
                    for profile, metrics in perf['profile_performance'].items():
                        threshold_symbol = "✅" if metrics.get('meets_threshold') else "❌"
                        f.write(f"| {profile} | {metrics.get('r2_score', 'N/A'):.3f} | {metrics.get('mape', 'N/A'):.2f}% | {metrics.get('rmse', 'N/A'):.3f} | {threshold_symbol} |\n")
           
            if test_summary['test_status'] != 'SUCCESS':
                f.write(f"\n### Error Details\n```\n{test_summary.get('error_details', 'Unknown error')}\n```\n")
   
    def _generate_failure_report(self, test_summary: Dict[str, Any], error: Exception):
        """Generate failure report for CI/CD"""
        failure_filename = f"training-test-failure-{self.environment}-{self.github_run_id or 'local'}.json"
       
        failure_report = {
            **test_summary,
            "failure_details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "execution_status": test_summary.get("step_function_execution", {}).get("status", "Unknown")
            }
        }
       
        with open(failure_filename, 'w') as f:
            json.dump(failure_report, f, indent=2, default=str)
       
        self.logger.error(f"Generated failure report: {failure_filename}")
        print(f"Generated failure report: {failure_filename}")

def main():
    """Main function for CI/CD enhanced training pipeline test"""
    parser = argparse.ArgumentParser(description='CI/CD Enhanced Training Pipeline Test')
    parser.add_argument('--environment', required=True,
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--ci-cd-mode', action='store_true',
                       help='Enable CI/CD mode')
    parser.add_argument('--github-run-id', help='GitHub Actions run ID')
    parser.add_argument('--timeout-minutes', type=int,
                       help='Test timeout in minutes')
    parser.add_argument('--all-profiles', action='store_true',
                       help='Test all 7 profiles (default behavior)')
   
    args = parser.parse_args()
   
    try:
        # Initialize CI/CD enhanced test
        test = CICDTrainingPipelineTest(
            environment=args.environment,
            region=args.region,
            ci_cd_mode=args.ci_cd_mode,
            github_run_id=args.github_run_id
        )
       
        # Run training pipeline test
        result = test.test_training_pipeline_cicd(
            timeout_minutes=args.timeout_minutes
        )
       
        print(f"Training pipeline test completed for {args.environment} environment")
        print(f"Status: {result['test_status']}")
        print(f"Duration: {result.get('test_duration', 'N/A')}")
       
        if result['test_status'] == 'SUCCESS':
            perf = result.get('performance_metrics', {})
            print(f"Performance: {perf.get('overall_assessment', 'Unknown')}")
            return 0
        else:
            print(f"Error: {result.get('error_details', 'Unknown error')}")
            return 1
       
    except Exception as e:
        print(f"Training pipeline test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
