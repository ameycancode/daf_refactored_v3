#!/usr/bin/env python3
"""
Enhanced Prediction Pipeline Testing
Tests the enhanced prediction pipeline with endpoint management and cost optimization
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

class EnhancedPredictionPipelineTest:
    """
    Test class for the enhanced prediction pipeline
    Tests endpoint creation, prediction execution, and cost optimization
    """
    
    def __init__(self, region: str = "us-west-2"):
        self.region = region
        self.account_id = boto3.client('sts').get_caller_identity()['Account']

        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
    
        # Initialize AWS clients
        self.stepfunctions_client = boto3.client('stepfunctions', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
        # Test configuration
        self.step_function_name = "energy-forecasting-enhanced-prediction-pipeline"
        self.step_function_arn = f"arn:aws:states:{region}:{self.account_id}:stateMachine:{self.step_function_name}"
        
        self.logger.info("Enhanced Prediction Pipeline Test initialized")
        self.logger.info(f"Region: {region}, Account: {self.account_id}")
        self.logger.info(f"Step Function: {self.step_function_name}")
    
    def test_prediction_pipeline(self, profiles: List[str], timeout_minutes: int = 20) -> Dict[str, Any]:
        """
        Test the enhanced prediction pipeline with specified profiles
        
        Args:
            profiles: List of profiles to test (e.g., ['RNN', 'RN', 'M'])
            timeout_minutes: Maximum time to wait for completion
            
        Returns:
            Dict containing test results and metrics
        """
        test_start_time = datetime.now()
        
        try:
            self.logger.info("="*60)
            self.logger.info("ENHANCED PREDICTION PIPELINE TEST")
            self.logger.info("="*60)
            self.logger.info(f"Testing profiles: {profiles}")
            self.logger.info(f"Timeout: {timeout_minutes} minutes")
            self.logger.info(f"Test start time: {test_start_time}")
            
            # Step 1: Validate Step Function exists
            self._validate_step_function()
            
            # Step 2: Start execution
            execution_arn, execution_name = self._start_execution(profiles)
            
            # Step 3: Monitor execution
            execution_result = self._monitor_execution(execution_arn, timeout_minutes)
            
            # Step 4: Analyze results
            test_results = self._analyze_results(execution_result, profiles, test_start_time)
            
            self.logger.info("="*60)
            self.logger.info("ENHANCED PREDICTION PIPELINE TEST COMPLETED")
            self.logger.info("="*60)
            self.logger.info(f"Test status: {test_results['status']}")
            self.logger.info(f"Total duration: {test_results['duration']}")
            self.logger.info(f"Profiles tested: {test_results['profiles_completed']}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Prediction pipeline test failed: {str(e)}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'duration': str(datetime.now() - test_start_time),
                'test_start_time': test_start_time.isoformat()
            }
    
    def _validate_step_function(self):
        """Validate that the Step Function exists and is available"""
        try:
            response = self.stepfunctions_client.describe_state_machine(
                stateMachineArn=self.step_function_arn
            )
            
            if response['status'] != 'ACTIVE':
                raise Exception(f"Step Function is not active. Status: {response['status']}")

            self.logger.info(f"✓ Step Function validated: {self.step_function_name}")
            self.logger.info(f"  Status: {response['status']}")
            self.logger.info(f"  Created: {response['creationDate']}")

        except self.stepfunctions_client.exceptions.StateMachineDoesNotExist:
            raise Exception(f"Step Function not found: {self.step_function_name}")
        except Exception as e:
            raise Exception(f"Step Function validation failed: {str(e)}")
    
    def _start_execution(self, profiles: List[str]) -> tuple:
        """Start Step Function execution with specified profiles"""
        execution_input = {
            "profiles": profiles,
            "test_mode": True,
            "timestamp": datetime.now().isoformat()
        }
        
        execution_name = f"prediction-test-{int(time.time())}"
        
        try:
            response = self.stepfunctions_client.start_execution(
                stateMachineArn=self.step_function_arn,
                name=execution_name,
                input=json.dumps(execution_input)
            )
            
            execution_arn = response['executionArn']

            self.logger.info(f"✓ Execution started: {execution_name}")
            self.logger.info(f"  Execution ARN: {execution_arn}")
            self.logger.info(f"  Input profiles: {profiles}")

            return execution_arn, execution_name
            
        except Exception as e:
            raise Exception(f"Failed to start execution: {str(e)}")
    
    def _monitor_execution(self, execution_arn: str, timeout_minutes: int) -> Dict[str, Any]:
        """Monitor Step Function execution until completion or timeout"""
        start_time = datetime.now()
        timeout_time = start_time + timedelta(minutes=timeout_minutes)

        self.logger.info(f"Monitoring execution (timeout: {timeout_minutes} minutes)...")

        status_history = []
        
        while datetime.now() < timeout_time:
            try:
                response = self.stepfunctions_client.describe_execution(
                    executionArn=execution_arn
                )
                
                current_status = response['status']
                
                # Log status changes
                if not status_history or status_history[-1]['status'] != current_status:
                    elapsed = datetime.now() - start_time
                    status_entry = {
                        'status': current_status,
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_time': str(elapsed)
                    }
                    status_history.append(status_entry)

                    self.logger.info(f"Status: {current_status} (elapsed: {elapsed})")

                # Check if execution is complete
                if current_status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                    execution_result = {
                        'status': current_status,
                        'start_time': response['startDate'].isoformat(),
                        'end_time': response.get('stopDate', datetime.now()).isoformat(),
                        'duration': str(datetime.now() - start_time),
                        'status_history': status_history,
                        'output': None,
                        'error': None
                    }
                    
                    if 'output' in response:
                        try:
                            execution_result['output'] = json.loads(response['output'])
                        except json.JSONDecodeError:
                            execution_result['output'] = response['output']
                    
                    if current_status == 'FAILED' and 'error' in response:
                        execution_result['error'] = response.get('error')
                        execution_result['cause'] = response.get('cause')

                    self.logger.info(f"Execution completed with status: {current_status}")
                    return execution_result
                
                # Wait before next check
                time.sleep(15)
                
            except Exception as e:
                self.logger.error(f"Error monitoring execution: {str(e)}")
                time.sleep(15)
                continue
        
        # Timeout reached
        self.logger.warning(f"Execution monitoring timed out after {timeout_minutes} minutes")
        return {
            'status': 'TIMEOUT',
            'duration': str(datetime.now() - start_time),
            'status_history': status_history,
            'error': f'Monitoring timed out after {timeout_minutes} minutes'
        }
    
    def _analyze_results(self, execution_result: Dict[str, Any], 
                        requested_profiles: List[str], 
                        test_start_time: datetime) -> Dict[str, Any]:
        """Analyze execution results and generate test summary"""
        
        analysis = {
            'status': 'UNKNOWN',
            'test_start_time': test_start_time.isoformat(),
            'test_end_time': datetime.now().isoformat(),
            'duration': str(datetime.now() - test_start_time),
            'execution_status': execution_result['status'],
            'profiles_requested': requested_profiles,
            'profiles_completed': [],
            'profiles_failed': [],
            'cost_optimization': {},
            'performance_metrics': {},
            'issues_found': [],
            'execution_details': execution_result
        }
        
        try:
            if execution_result['status'] == 'SUCCEEDED':
                if execution_result.get('output'):
                    self._analyze_successful_execution(execution_result['output'], analysis)
                else:
                    analysis['issues_found'].append("No output data from execution")
                    analysis['status'] = 'PARTIAL_SUCCESS'
            
            elif execution_result['status'] == 'FAILED':
                analysis['status'] = 'FAILED'
                analysis['issues_found'].append(f"Execution failed: {execution_result.get('error', 'Unknown error')}")
            
            elif execution_result['status'] == 'TIMEOUT':
                analysis['status'] = 'TIMEOUT'
                analysis['issues_found'].append("Execution timed out")
            
            else:
                analysis['status'] = 'FAILED'
                analysis['issues_found'].append(f"Unexpected execution status: {execution_result['status']}")
            
            # Final status determination
            if not analysis['issues_found'] and analysis['profiles_completed']:
                analysis['status'] = 'SUCCESS'
            elif analysis['profiles_completed'] and len(analysis['issues_found']) <= 2:
                analysis['status'] = 'PARTIAL_SUCCESS'
            
            # Log summary
            self._log_analysis_summary(analysis)
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {str(e)}")
            analysis['status'] = 'ANALYSIS_ERROR'
            analysis['issues_found'].append(f"Analysis error: {str(e)}")
        
        return analysis
    
    def _analyze_successful_execution(self, output_data: Dict[str, Any], analysis: Dict[str, Any]):
        """Analyze output data from successful execution"""
        
        # Analyze prediction results
        if 'prediction_results' in output_data:
            pred_results = output_data['prediction_results']
            
            for result in pred_results:
                profile = result.get('profile', 'Unknown')
                status = result.get('status', 'Unknown')
                
                if status == 'success':
                    analysis['profiles_completed'].append(profile)
                else:
                    analysis['profiles_failed'].append(profile)
                    analysis['issues_found'].append(f"Profile {profile} failed: {result.get('error', 'Unknown error')}")
        
        # Analyze cost optimization
        if 'cleanup_results' in output_data:
            cleanup_results = output_data['cleanup_results']
            
            endpoints_created = len([r for r in cleanup_results if r.get('endpoint_created')])
            endpoints_deleted = len([r for r in cleanup_results if r.get('cleanup_status') == 'success'])
            
            analysis['cost_optimization'] = {
                'endpoints_created': endpoints_created,
                'endpoints_deleted': endpoints_deleted,
                'cleanup_success_rate': (endpoints_deleted / endpoints_created * 100) if endpoints_created > 0 else 0,
                'cost_savings_achieved': endpoints_deleted >= endpoints_created * 0.8  # 80% cleanup threshold
            }
            
            if not analysis['cost_optimization']['cost_savings_achieved']:
                analysis['issues_found'].append("Cost optimization target not met (< 80% endpoint cleanup)")
        
        # Analyze performance metrics
        execution_time = analysis.get('duration', '0:00:00')
        profiles_completed = len(analysis['profiles_completed'])
        profiles_requested = len(analysis['profiles_requested'])
        
        analysis['performance_metrics'] = {
            'total_execution_time': execution_time,
            'profiles_success_rate': (profiles_completed / profiles_requested * 100) if profiles_requested > 0 else 0,
            'average_time_per_profile': self._calculate_avg_time_per_profile(execution_time, profiles_completed),
            'parallel_efficiency': self._evaluate_parallel_efficiency(execution_time, profiles_completed)
        }
    
    def _calculate_avg_time_per_profile(self, total_time_str: str, profile_count: int) -> str:
        """Calculate average time per profile"""
        if profile_count == 0:
            return "N/A"
        
        try:
            # Parse duration string (format: H:MM:SS.microseconds)
            time_parts = total_time_str.split(':')
            if len(time_parts) >= 3:
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = float(time_parts[2])
                
                total_seconds = hours * 3600 + minutes * 60 + seconds
                avg_seconds = total_seconds / profile_count
                
                avg_minutes = int(avg_seconds // 60)
                avg_secs = int(avg_seconds % 60)
                
                return f"{avg_minutes}:{avg_secs:02d}"
            
        except Exception:
            pass
        
        return "N/A"
    
    def _evaluate_parallel_efficiency(self, total_time_str: str, profile_count: int) -> str:
        """Evaluate parallel execution efficiency"""
        if profile_count <= 1:
            return "N/A"
        
        try:
            # Parse total time
            time_parts = total_time_str.split(':')
            if len(time_parts) >= 3:
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = float(time_parts[2])
                
                total_minutes = hours * 60 + minutes + seconds / 60
                
                # Rough estimate: sequential would take ~profile_count * 10 minutes
                # Parallel should be much faster
                estimated_sequential_time = profile_count * 10
                efficiency = ((estimated_sequential_time - total_minutes) / estimated_sequential_time * 100)
                
                if efficiency > 80:
                    return "EXCELLENT"
                elif efficiency > 60:
                    return "GOOD"
                elif efficiency > 40:
                    return "FAIR"
                else:
                    return "POOR"
            
        except Exception:
            pass
        
        return "UNKNOWN"
    
    def _log_analysis_summary(self, analysis: Dict[str, Any]):
        """Log analysis summary"""
        self.logger.info("="*50)
        self.logger.info("TEST ANALYSIS SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Overall Status: {analysis['status']}")
        self.logger.info(f"Duration: {analysis['duration']}")
        self.logger.info(f"Profiles Requested: {len(analysis['profiles_requested'])}")
        self.logger.info(f"Profiles Completed: {len(analysis['profiles_completed'])}")
        self.logger.info(f"Profiles Failed: {len(analysis['profiles_failed'])}")
        
        if analysis['cost_optimization']:
            cost_opt = analysis['cost_optimization']
            self.logger.info(f"Endpoints Created: {cost_opt.get('endpoints_created', 0)}")
            self.logger.info(f"Endpoints Deleted: {cost_opt.get('endpoints_deleted', 0)}")
            self.logger.info(f"Cleanup Success Rate: {cost_opt.get('cleanup_success_rate', 0):.1f}%")
            self.logger.info(f"Cost Savings Achieved: {cost_opt.get('cost_savings_achieved', False)}")
        
        if analysis['performance_metrics']:
            perf = analysis['performance_metrics']
            self.logger.info(f"Success Rate: {perf.get('profiles_success_rate', 0):.1f}%")
            self.logger.info(f"Parallel Efficiency: {perf.get('parallel_efficiency', 'Unknown')}")
        
        if analysis['issues_found']:
            self.logger.warning("Issues Found:")
            for issue in analysis['issues_found']:
                self.logger.warning(f"  - {issue}")

        self.logger.info("="*50)

def main():
    """Main function for enhanced prediction pipeline testing"""
    parser = argparse.ArgumentParser(description='Enhanced Prediction Pipeline Test')
    parser.add_argument('--profiles', required=True,
                       help='Comma-separated profiles to test or "all"')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--timeout-minutes', type=int, default=20,
                       help='Test timeout in minutes')
    
    args = parser.parse_args()
    
    # Parse profiles
    if args.profiles.lower() == 'all':
        profiles = ['RNN', 'RN', 'M', 'S', 'AGR', 'L', 'A6']
    else:
        profiles = [p.strip() for p in args.profiles.split(',')]
    
    try:
        # Initialize test
        test = EnhancedPredictionPipelineTest(region=args.region)
        
        # Run test
        results = test.test_prediction_pipeline(
            profiles=profiles,
            timeout_minutes=args.timeout_minutes
        )
        
        # Print final results
        print("\n" + "="*60)
        print("ENHANCED PREDICTION PIPELINE TEST RESULTS")
        print("="*60)
        print(f"Status: {results['status']}")
        print(f"Duration: {results['duration']}")
        print(f"Profiles Requested: {len(results.get('profiles_requested', []))}")
        print(f"Profiles Completed: {len(results.get('profiles_completed', []))}")
        
        if results.get('cost_optimization'):
            cost_opt = results['cost_optimization']
            print(f"Cost Optimization: {cost_opt.get('cost_savings_achieved', False)}")
            print(f"Cleanup Rate: {cost_opt.get('cleanup_success_rate', 0):.1f}%")
        
        # Return appropriate exit code
        if results['status'] in ['SUCCESS', 'PARTIAL_SUCCESS']:
            return 0
        else:
            return 1
        
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
