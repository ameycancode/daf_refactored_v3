# =============================================================================
# ENHANCED PREDICTION PIPELINE TESTING - scripts/test_enhanced_prediction_pipeline_cicd.py
# =============================================================================
"""
Enhanced Prediction Pipeline Testing with CI/CD Integration
Extends existing prediction pipeline tests with CI/CD reporting capabilities
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
    from test_enhanced_prediction_pipeline import EnhancedPredictionPipelineTest
except ImportError:
    print("Error: Could not import test_enhanced_prediction_pipeline.py")
    print("Make sure the original test script exists in the same directory")
    sys.exit(1)

class CICDEnhancedPredictionPipelineTest(EnhancedPredictionPipelineTest):
    """
    CI/CD Enhanced Prediction Pipeline Test
    Extends the original test with CI/CD-specific features
    """
   
    def __init__(self, environment: str, region: str = "us-west-2",
                 ci_cd_mode: bool = False, github_run_id: Optional[str] = None):
        """Initialize CI/CD enhanced prediction pipeline test"""
       
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
       
        print(f"CI/CD Enhanced Prediction Pipeline Test initialized for {environment}")
        print(f"CI/CD Mode: {ci_cd_mode}")
        print(f"GitHub Run ID: {github_run_id}")
   
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        configs = {
            "dev": {
                "step_function_name": "energy-forecasting-enhanced-prediction-pipeline",
                "timeout_minutes": 15,
                "available_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            },
            "preprod": {
                "step_function_name": "energy-forecasting-enhanced-prediction-pipeline",
                "timeout_minutes": 20,
                "available_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            },
            "prod": {
                "step_function_name": "energy-forecasting-enhanced-prediction-pipeline",
                "timeout_minutes": 30,
                "available_profiles": ["RNN", "RN", "M", "S", "AGR", "L", "A6"]
            }
        }
        return configs.get(self.environment, configs["dev"])
   
    def _setup_cicd_logging(self):
        """Setup CI/CD specific logging"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [CI/CD-PRED] - %(levelname)s - %(message)s'
        )
       
        if self.github_run_id:
            log_filename = f"prediction-test-{self.environment}-{self.github_run_id}.log"
            file_handler = logging.FileHandler(log_filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
   
    def test_prediction_pipeline_cicd(self, profiles: str, test_case: str = "default",
                                    timeout_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Test prediction pipeline with CI/CD enhancements
        """
        test_summary = {
            "environment": self.environment,
            "test_type": "prediction_pipeline",
            "test_case": test_case,
            "ci_cd_mode": self.ci_cd_mode,
            "github_run_id": self.github_run_id,
            "test_start_time": self.test_start_time.isoformat(),
            "test_status": "IN_PROGRESS",
            "profiles_requested": [],
            "profiles_tested": [],
            "step_function_execution": {},
            "performance_metrics": {},
            "cost_optimization": {},
            "validation_results": {},
            "error_details": None
        }
       
        try:
            # Parse profiles
            if profiles.lower() == "all":
                requested_profiles = self.env_config["available_profiles"]
            else:
                requested_profiles = [p.strip() for p in profiles.split(",")]
           
            test_summary["profiles_requested"] = requested_profiles
           
            print("="*80)
            print(f"STARTING CI/CD ENHANCED PREDICTION PIPELINE TEST - {test_case}")
            print(f"Environment: {self.environment}")
            print(f"Test Case: {test_case}")
            print(f"Profiles: {requested_profiles}")
            print(f"Step Function: {self.env_config['step_function_name']}")
            print("="*80)
           
            # Use provided timeout or environment default
            timeout_mins = timeout_minutes or self.env_config["timeout_minutes"]
           
            # Step 1: Validate Step Function exists
            self._validate_prediction_step_function(test_summary)
           
            # Step 2: Start execution
            execution_result = self._start_prediction_execution(
                test_summary,
                requested_profiles,
                timeout_mins,
                test_case
            )
            print(f"Execution Result: {execution_result}")
           
            # Step 3: Monitor execution
            monitoring_result = self._monitor_prediction_execution(
                execution_result["execution_arn"],
                timeout_mins,
                test_summary
            )
            print(f"Monitoring Result: {monitoring_result}")

            # Step 4: Validate results
            validation_result = self._validate_prediction_results(
                monitoring_result,
                requested_profiles,
                test_summary
            )
            print(f"Validation Result: {validation_result}")
           
            # Step 5: Analyze cost optimization
            cost_result = self._analyze_cost_optimization(
                monitoring_result,
                test_summary
            )
            print(f"Cost Optimization Result: {cost_result}")
           
            # Step 6: Performance analysis
            performance_result = self._analyze_prediction_performance(
                monitoring_result,
                requested_profiles,
                test_summary
            )
            print(f"Performance Result: {performance_result}")
           
            test_summary["test_status"] = "SUCCESS"
            test_summary["test_end_time"] = datetime.now().isoformat()
            test_summary["test_duration"] = str(datetime.now() - self.test_start_time)
           
            print("="*80)
            print(f"CI/CD ENHANCED PREDICTION PIPELINE TEST COMPLETED - {test_case}")
            print(f"Total Duration: {test_summary['test_duration']}")
            print(f"Profiles Tested: {len(test_summary['profiles_tested'])}")
            print("="*80)
           
            # Generate CI/CD artifacts
            self._generate_prediction_test_artifacts(test_summary)
           
            return test_summary
           
        except Exception as e:
            test_summary["test_status"] = "FAILED"
            test_summary["error_details"] = str(e)
            test_summary["test_end_time"] = datetime.now().isoformat()
           
            self.logger.error(f"CI/CD Enhanced prediction pipeline test failed: {str(e)}")
            print(f"CI/CD Enhanced prediction pipeline test failed: {str(e)}")

            # Generate failure report
            self._generate_prediction_failure_report(test_summary, e)
           
            raise
   
    def _validate_prediction_step_function(self, test_summary: Dict[str, Any]):
        """Validate that the prediction Step Function exists"""
        print("Validating Prediction Step Function exists...")
       
        try:
            step_function_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.env_config['step_function_name']}"
           
            response = self.stepfunctions_client.describe_state_machine(
                stateMachineArn=step_function_arn
            )
           
            test_summary["step_function_validation"] = {
                "exists": True,
                "arn": response["stateMachineArn"],
                "status": response["status"],
                "creation_date": response["creationDate"].isoformat(),
                "definition_hash": response.get("stateMachineRevisionId", "Unknown")
            }
           
            print(f"✓ Prediction Step Function validated: {self.env_config['step_function_name']}")
           
        except Exception as e:
            test_summary["step_function_validation"] = {
                "exists": False,
                "error": str(e)
            }
            raise Exception(f"Prediction Step Function validation failed: {str(e)}")
   
    def _start_prediction_execution(self, test_summary: Dict[str, Any],
                                  profiles: List[str], timeout_minutes: int,
                                  test_case: str) -> Dict[str, Any]:
        """Start prediction pipeline execution"""
        print(f"Starting prediction pipeline execution for {len(profiles)} profiles...")
       
        step_function_arn = f"arn:aws:states:{self.region}:{self.account_id}:stateMachine:{self.env_config['step_function_name']}"
       
        # Prepare execution input
        execution_input = {
            "profiles": profiles,
            "test_mode": True,
            "execution_type": test_case.lower(),
            "environment": self.environment,
            "github_run_id": self.github_run_id,
            # "test_case": test_case,
            "timeout_minutes": timeout_minutes
        }
       
        execution_name = f"cicd-prediction-test-{test_case.lower().replace(' ', '-')}-{self.environment}-{int(time.time())}"
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
                "input": execution_input,
                "profiles_count": len(profiles)
            }
           
            test_summary["step_function_execution"] = execution_result
           
            print(f"✓ Prediction execution started: {execution_name}")
            print(f"  Execution ARN: {response['executionArn']}")
            print(f"  Profiles: {profiles}")
           
            return execution_result
           
        except Exception as e:
            raise Exception(f"Failed to start prediction execution: {str(e)}")
   
    def _monitor_prediction_execution(self, execution_arn: str, timeout_minutes: int,
                                    test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor prediction pipeline execution"""
        print(f"Monitoring prediction execution (timeout: {timeout_minutes} minutes)...")
       
        start_time = datetime.now()
        timeout_time = start_time + timedelta(minutes=timeout_minutes)
       
        monitoring_result = {
            "status": "UNKNOWN",
            "execution_history": [],
            "parallel_executions": [],
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
                    print(f"Prediction execution status: {current_status} (elapsed: {status_update['elapsed_time']})")
               
                # Get execution history for parallel tracking
                if current_status == "RUNNING":
                    try:
                        history_response = self.stepfunctions_client.get_execution_history(
                            executionArn=execution_arn,
                            maxResults=50,
                            reverseOrder=True
                        )
                       
                        # Track parallel executions (Map state iterations)
                        for event in history_response.get("events", []):
                            if event.get("type") == "MapIterationStarted":
                                iteration_info = {
                                    "iteration_index": event.get("mapIterationStartedEventDetails", {}).get("index"),
                                    "timestamp": event.get("timestamp", datetime.now()).isoformat()
                                }
                                if iteration_info not in monitoring_result["parallel_executions"]:
                                    monitoring_result["parallel_executions"].append(iteration_info)
                       
                    except Exception as e:
                        self.logger.debug(f"Could not get execution history: {str(e)}")
                        print(f"Could not get execution history: {str(e)}")
               
                # Check if execution is complete
                if current_status in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
                    if "output" in response:
                        try:
                            monitoring_result["final_output"] = json.loads(response["output"])
                        except json.JSONDecodeError:
                            monitoring_result["final_output"] = response["output"]
                   
                    monitoring_result["monitoring_duration"] = str(datetime.now() - start_time)
                   
                    print(f"Prediction execution completed with status: {current_status}")
                    print(f"Parallel executions tracked: {len(monitoring_result['parallel_executions'])}")
                    break
               
                # Wait before next check
                time.sleep(15)  # More frequent checks for prediction pipeline
            else:
                # Timeout reached
                monitoring_result["status"] = "TIMEOUT"
                monitoring_result["monitoring_duration"] = str(datetime.now() - start_time)
                print(f"Prediction execution monitoring timed out after {timeout_minutes} minutes")

            return monitoring_result
           
        except Exception as e:
            monitoring_result["error"] = str(e)
            raise Exception(f"Prediction execution monitoring failed: {str(e)}")
   
    def _validate_prediction_results(self, monitoring_result: Dict[str, Any],
                                   requested_profiles: List[str],
                                   test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction pipeline results"""
        print("Validating prediction pipeline results...")
       
        validation_result = {
            "execution_successful": False,
            "profiles_completed": [],
            "profiles_failed": [],
            "endpoints_created": [],
            "endpoints_deleted": [],
            "predictions_generated": [],
            "issues_found": []
        }
       
        try:
            # Check execution status
            if monitoring_result["status"] == "SUCCEEDED":
                validation_result["execution_successful"] = True
                print("✓ Prediction execution completed successfully")
               
                # Validate output structure
                if monitoring_result.get("final_output"):
                    output = monitoring_result["final_output"]
                   
                    # Check prediction results
                    if "prediction_results" in output:
                        pred_results = output["prediction_results"]
                        if isinstance(pred_results, list):
                            for result in pred_results:
                                if result.get("status") == "success":
                                    profile = result.get("profile", "Unknown")
                                    validation_result["profiles_completed"].append(profile)
                                    if "prediction_data" in result:
                                        validation_result["predictions_generated"].append(profile)
                                else:
                                    profile = result.get("profile", "Unknown")
                                    validation_result["profiles_failed"].append(profile)
                   
                    # Check endpoint management
                    if "endpoint_results" in output:
                        endpoint_results = output["endpoint_results"]
                        if isinstance(endpoint_results, list):
                            for ep_result in endpoint_results:
                                if ep_result.get("endpoint_created"):
                                    validation_result["endpoints_created"].append(ep_result.get("profile", "Unknown"))
                                if ep_result.get("endpoint_deleted"):
                                    validation_result["endpoints_deleted"].append(ep_result.get("profile", "Unknown"))
                   
                    # Check cleanup results
                    if "cleanup_results" in output:
                        cleanup_results = output["cleanup_results"]
                        if isinstance(cleanup_results, list):
                            for cleanup in cleanup_results:
                                if cleanup.get("cleanup_status") == "success":
                                    validation_result["endpoints_deleted"].append(cleanup.get("profile", "Unknown"))
               
                # Validate all requested profiles were processed
                completed_profiles = set(validation_result["profiles_completed"])
                requested_profiles_set = set(requested_profiles)
               
                if not requested_profiles_set.issubset(completed_profiles):
                    missing_profiles = requested_profiles_set - completed_profiles
                    validation_result["issues_found"].append(f"Missing profiles: {list(missing_profiles)}")
                    self.logger.warning(f"⚠ Missing profiles: {missing_profiles}")
                    print(f"⚠ Missing profiles: {missing_profiles}")
                else:
                    print("✓ All requested profiles completed")
               
                # Validate cost optimization (endpoints cleaned up)
                created_count = len(validation_result["endpoints_created"])
                deleted_count = len(validation_result["endpoints_deleted"])
                if created_count != deleted_count:
                    validation_result["issues_found"].append(f"Endpoint cleanup mismatch: {created_count} created, {deleted_count} deleted")
                    self.logger.warning(f"⚠ Endpoint cleanup mismatch: {created_count} created, {deleted_count} deleted")
                    print(f"⚠ Endpoint cleanup mismatch: {created_count} created, {deleted_count} deleted")
                else:
                    print("✓ Endpoint cost optimization successful")
               
            else:
                validation_result["execution_successful"] = False
                validation_result["issues_found"].append(f"Execution failed with status: {monitoring_result['status']}")
                self.logger.error(f"✗ Prediction execution failed: {monitoring_result['status']}")
                print(f"✗ Prediction execution failed: {monitoring_result['status']}")
           
            test_summary["validation_results"] = validation_result
            test_summary["profiles_tested"] = validation_result["profiles_completed"]
            return validation_result
           
        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["issues_found"].append(f"Validation error: {str(e)}")
            self.logger.error(f"Prediction result validation failed: {str(e)}")
            print(f"Prediction result validation failed: {str(e)}")
            return validation_result
   
    def _analyze_cost_optimization(self, monitoring_result: Dict[str, Any],
                                 test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost optimization effectiveness"""
        print("Analyzing cost optimization...")
       
        cost_analysis = {
            "endpoints_lifecycle": {},
            "cost_savings_achieved": False,
            "cleanup_effectiveness": "Unknown",
            "estimated_savings": "Unknown"
        }
       
        try:
            if monitoring_result.get("final_output"):
                output = monitoring_result["final_output"]
               
                # Analyze endpoint lifecycle
                if "endpoint_results" in output and "cleanup_results" in output:
                    endpoint_results = output["endpoint_results"]
                    cleanup_results = output["cleanup_results"]
                   
                    endpoints_created = len([ep for ep in endpoint_results if ep.get("endpoint_created")])
                    endpoints_deleted = len([cl for cl in cleanup_results if cl.get("cleanup_status") == "success"])
                   
                    cost_analysis["endpoints_lifecycle"] = {
                        "created": endpoints_created,
                        "deleted": endpoints_deleted,
                        "cleanup_rate": (endpoints_deleted / endpoints_created * 100) if endpoints_created > 0 else 0
                    }
                   
                    # Determine cost savings achievement
                    cleanup_rate = cost_analysis["endpoints_lifecycle"]["cleanup_rate"]
                    if cleanup_rate >= 90:
                        cost_analysis["cost_savings_achieved"] = True
                        cost_analysis["cleanup_effectiveness"] = "EXCELLENT"
                        cost_analysis["estimated_savings"] = "95-98%"
                    elif cleanup_rate >= 75:
                        cost_analysis["cost_savings_achieved"] = True
                        cost_analysis["cleanup_effectiveness"] = "GOOD"
                        cost_analysis["estimated_savings"] = "80-95%"
                    elif cleanup_rate >= 50:
                        cost_analysis["cost_savings_achieved"] = True
                        cost_analysis["cleanup_effectiveness"] = "FAIR"
                        cost_analysis["estimated_savings"] = "50-80%"
                    else:
                        cost_analysis["cost_savings_achieved"] = False
                        cost_analysis["cleanup_effectiveness"] = "POOR"
                        cost_analysis["estimated_savings"] = "<50%"
                   
                    print(f"Cost optimization analysis: {cost_analysis['cleanup_effectiveness']}")
                    print(f"Endpoint cleanup rate: {cleanup_rate:.1f}%")
                    print(f"Estimated cost savings: {cost_analysis['estimated_savings']}")
           
            test_summary["cost_optimization"] = cost_analysis
            return cost_analysis
           
        except Exception as e:
            cost_analysis["error"] = str(e)
            self.logger.error(f"Cost optimization analysis failed: {str(e)}")
            print(f"Cost optimization analysis failed: {str(e)}")
            return cost_analysis
   
    def _analyze_prediction_performance(self, monitoring_result: Dict[str, Any],
                                      requested_profiles: List[str],
                                      test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction performance metrics"""
        print("Analyzing prediction performance...")
       
        performance_analysis = {
            "execution_time": monitoring_result.get("monitoring_duration", "Unknown"),
            "status": monitoring_result.get("status", "Unknown"),
            "parallel_efficiency": {},
            "profile_performance": {},
            "overall_assessment": "Unknown"
        }
       
        try:
            # Analyze parallel execution efficiency
            parallel_executions = monitoring_result.get("parallel_executions", [])
            expected_parallel = len(requested_profiles)
            actual_parallel = len(parallel_executions)
           
            performance_analysis["parallel_efficiency"] = {
                "expected_parallel_executions": expected_parallel,
                "actual_parallel_executions": actual_parallel,
                "efficiency_rate": (actual_parallel / expected_parallel * 100) if expected_parallel > 0 else 0
            }
           
            # Analyze per-profile performance
            if monitoring_result.get("final_output") and "prediction_results" in monitoring_result["final_output"]:
                pred_results = monitoring_result["final_output"]["prediction_results"]
               
                if isinstance(pred_results, list):
                    for result in pred_results:
                        profile = result.get("profile", "Unknown")
                        performance_analysis["profile_performance"][profile] = {
                            "status": result.get("status", "Unknown"),
                            "execution_time": result.get("execution_time", "Unknown"),
                            "predictions_count": len(result.get("prediction_data", [])),
                            "endpoint_lifecycle": {
                                "created": result.get("endpoint_created", False),
                                "deleted": result.get("endpoint_deleted", False)
                            }
                        }
           
            # Overall assessment
            successful_profiles = len([p for p in performance_analysis["profile_performance"].values() if p["status"] == "success"])
            total_profiles = len(requested_profiles)
            efficiency_rate = performance_analysis["parallel_efficiency"]["efficiency_rate"]
           
            if successful_profiles == total_profiles and efficiency_rate >= 90:
                performance_analysis["overall_assessment"] = "EXCELLENT"
            elif successful_profiles >= total_profiles * 0.8 and efficiency_rate >= 75:
                performance_analysis["overall_assessment"] = "GOOD"
            elif successful_profiles >= total_profiles * 0.6:
                performance_analysis["overall_assessment"] = "FAIR"
            else:
                performance_analysis["overall_assessment"] = "POOR"
           
            print(f"Performance assessment: {performance_analysis['overall_assessment']}")
            print(f"Successful profiles: {successful_profiles}/{total_profiles}")
            print(f"Parallel efficiency: {efficiency_rate:.1f}%")
           
            test_summary["performance_metrics"] = performance_analysis
            return performance_analysis
           
        except Exception as e:
            performance_analysis["error"] = str(e)
            self.logger.error(f"Performance analysis failed: {str(e)}")
            print(f"Performance analysis failed: {str(e)}")
            return performance_analysis
   
    def _generate_prediction_test_artifacts(self, test_summary: Dict[str, Any]):
        """Generate CI/CD prediction test artifacts"""
        print("Generating CI/CD prediction test artifacts...")
       
        # Generate test results JSON
        test_case_safe = test_summary.get("test_case", "default").lower().replace(" ", "-")
        results_filename = f"prediction-test-results-{test_case_safe}-{self.environment}-{self.github_run_id or 'local'}.json"
        with open(results_filename, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
       
        print(f"Generated prediction test results: {results_filename}")
       
        # Generate GitHub Actions summary if in CI/CD mode
        if self.ci_cd_mode and os.getenv('GITHUB_STEP_SUMMARY'):
            self._generate_github_prediction_summary(test_summary)
   
    def _generate_github_prediction_summary(self, test_summary: Dict[str, Any]):
        """Generate GitHub Actions prediction test summary"""
        summary_file = os.getenv('GITHUB_STEP_SUMMARY')
        if not summary_file:
            return
       
        with open(summary_file, 'a') as f:
            f.write(f"""
## 🔮 Prediction Pipeline Test Results - {test_summary.get('test_case', 'Default')} - {self.environment.upper()}

| Parameter | Value |
|-----------|-------|
| Test Status | {'✅ PASSED' if test_summary['test_status'] == 'SUCCESS' else '❌ FAILED'} |
| Environment | `{self.environment}` |
| Test Case | {test_summary.get('test_case', 'Default')} |
| Duration | {test_summary.get('test_duration', 'N/A')} |
| Profiles Requested | {len(test_summary.get('profiles_requested', []))} |
| Profiles Completed | {len(test_summary.get('profiles_tested', []))} |

### Cost Optimization Results
""")
           
            if 'cost_optimization' in test_summary:
                cost = test_summary['cost_optimization']
                f.write(f"- **Cost Savings Achieved**: {'✅ Yes' if cost.get('cost_savings_achieved') else '❌ No'}\n")
                f.write(f"- **Cleanup Effectiveness**: {cost.get('cleanup_effectiveness', 'Unknown')}\n")
                f.write(f"- **Estimated Savings**: {cost.get('estimated_savings', 'Unknown')}\n")
               
                if 'endpoints_lifecycle' in cost:
                    lifecycle = cost['endpoints_lifecycle']
                    f.write(f"- **Endpoints Created**: {lifecycle.get('created', 0)}\n")
                    f.write(f"- **Endpoints Deleted**: {lifecycle.get('deleted', 0)}\n")
                    f.write(f"- **Cleanup Rate**: {lifecycle.get('cleanup_rate', 0):.1f}%\n")
           
            if 'performance_metrics' in test_summary:
                perf = test_summary['performance_metrics']
                f.write(f"\n### Performance Analysis\n")
                f.write(f"- **Overall Assessment**: {perf.get('overall_assessment', 'Unknown')}\n")
                f.write(f"- **Execution Time**: {perf.get('execution_time', 'Unknown')}\n")
               
                if 'parallel_efficiency' in perf:
                    parallel = perf['parallel_efficiency']
                    f.write(f"- **Parallel Efficiency**: {parallel.get('efficiency_rate', 0):.1f}%\n")
               
                if 'profile_performance' in perf:
                    f.write("\n### Profile Results\n")
                    f.write("| Profile | Status | Predictions | Endpoint Lifecycle |\n")
                    f.write("|---------|--------|-------------|--------------------|\n")
                   
                    for profile, metrics in perf['profile_performance'].items():
                        status_symbol = "✅" if metrics.get('status') == 'success' else "❌"
                        lifecycle = metrics.get('endpoint_lifecycle', {})
                        lifecycle_status = "✅ Created→Deleted" if lifecycle.get('created') and lifecycle.get('deleted') else "❌ Incomplete"
                        f.write(f"| {profile} | {status_symbol} {metrics.get('status', 'Unknown')} | {metrics.get('predictions_count', 0)} | {lifecycle_status} |\n")
           
            if test_summary['test_status'] != 'SUCCESS':
                f.write(f"\n### Error Details\n```\n{test_summary.get('error_details', 'Unknown error')}\n```\n")
   
    def _generate_prediction_failure_report(self, test_summary: Dict[str, Any], error: Exception):
        """Generate failure report for CI/CD"""
        test_case_safe = test_summary.get("test_case", "default").lower().replace(" ", "-")
        failure_filename = f"prediction-test-failure-{test_case_safe}-{self.environment}-{self.github_run_id or 'local'}.json"
       
        failure_report = {
            **test_summary,
            "failure_details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "execution_status": test_summary.get("step_function_execution", {}).get("status", "Unknown"),
                "profiles_requested": test_summary.get("profiles_requested", []),
                "profiles_completed": test_summary.get("profiles_tested", [])
            }
        }
       
        with open(failure_filename, 'w') as f:
            json.dump(failure_report, f, indent=2, default=str)
       
        self.logger.error(f"Generated prediction failure report: {failure_filename}")
        print(f"Generated prediction failure report: {failure_filename}")

def main():
    """Main function for CI/CD enhanced prediction pipeline test"""
    parser = argparse.ArgumentParser(description='CI/CD Enhanced Prediction Pipeline Test')
    parser.add_argument('--environment', required=True,
                       choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--profiles', required=True,
                       help='Comma-separated profiles or "all"')
    parser.add_argument('--ci-cd-mode', action='store_true',
                       help='Enable CI/CD mode')
    parser.add_argument('--github-run-id', help='GitHub Actions run ID')
    parser.add_argument('--timeout-minutes', type=int,
                       help='Test timeout in minutes')
    parser.add_argument('--test-case', default='default',
                       help='Test case name for reporting')
   
    args = parser.parse_args()
   
    try:
        # Initialize CI/CD enhanced test
        test = CICDEnhancedPredictionPipelineTest(
            environment=args.environment,
            region=args.region,
            ci_cd_mode=args.ci_cd_mode,
            github_run_id=args.github_run_id
        )
       
        # Run prediction pipeline test
        result = test.test_prediction_pipeline_cicd(
            profiles=args.profiles,
            test_case=args.test_case,
            timeout_minutes=args.timeout_minutes
        )
       
        print(f"Prediction pipeline test completed for {args.environment} environment")
        print(f"Test Case: {args.test_case}")
        print(f"Status: {result['test_status']}")
        print(f"Duration: {result.get('test_duration', 'N/A')}")
       
        if result['test_status'] == 'SUCCESS':
            profiles_completed = len(result.get('profiles_tested', []))
            profiles_requested = len(result.get('profiles_requested', []))
            print(f"Profiles: {profiles_completed}/{profiles_requested} completed")
           
            cost_opt = result.get('cost_optimization', {})
            if cost_opt.get('cost_savings_achieved'):
                print(f"Cost Optimization: {cost_opt.get('cleanup_effectiveness', 'Unknown')}")
           
            return 0
        else:
            print(f"Error: {result.get('error_details', 'Unknown error')}")
            return 1
       
    except Exception as e:
        print(f"Prediction pipeline test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
