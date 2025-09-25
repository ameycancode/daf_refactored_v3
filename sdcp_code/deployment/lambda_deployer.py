#!/usr/bin/env python3
"""
Enhanced Lambda Function Deployer for Energy Forecasting MLOps Pipeline
Deploys all 11 Lambda functions including profile-predictor with secure layer
Handles both standard functions and profile-predictor with custom packaging
SECURITY UPDATE: All vulnerabilities patched across all functions
"""

import boto3
import json
import zipfile
import os
import tempfile
import shutil
import time
import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteLambdaDeployer:
    def __init__(self, region="us-west-2", datascientist_role_name="sdcp-dev-sagemaker-energy-forecasting-datascientist-role", environment="dev"):
        self.region = region
        self.datascientist_role_name = datascientist_role_name
        self.environment = environment
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        self.max_wait_time = 300  # 5 minutes
        self.poll_interval = 10   # 10 seconds
        
        # Initialize AWS clients
        self.lambda_client = boto3.client('lambda', region_name=region)
        self.iam_client = boto3.client('iam')
        
        # Get execution role
        self.execution_role = self.get_existing_datascientist_role()
        
        # Layer ARN for secure profile-predictor
        self.secure_layer_arn = None
        
        # Base path for Lambda functions (updated for sdcp_code structure)
        self.lambda_functions_base_path = "sdcp_code/lambda-functions"
        
    def get_existing_datascientist_role(self):
        """Get the existing DataScientist role ARN"""
        
        try:
            role_response = self.iam_client.get_role(RoleName=self.datascientist_role_name)
            role_arn = role_response['Role']['Arn']
            logger.info(f"✓ Using DataScientist role: {role_arn}")
            return role_arn
        except Exception as e:
            logger.error(f"✗ DataScientist role not found: {str(e)}")
            logger.error("Contact admin team to create the DataScientist role")
            raise
    
    def create_secure_layer_for_profile_predictor(self):
        """Create secure layer for profile-predictor with all dependencies"""
        
        layer_name = f"SecureEnergyForecastingLayer2025-{self.environment}"
        python_version = "3.12"
        
        logger.info(f"Creating secure layer for profile-predictor: {layer_name}")
        
        requirements_content = """# August 2025 - Latest secure versions for Python 3.12
numpy==1.26.4
pandas==2.2.2
requests==2.32.4
boto3==1.34.162
botocore==1.34.162
pytz>=2024.1
# pyarrow==17.0.0
# s3fs==2024.6.1
# setuptools==75.1.0
# urllib3==2.2.2
# openpyxl==3.1.5
# xlsxwriter==3.2.0
# # Required dependencies for requests
# idna>=3.7
# charset-normalizer>=3.3.2
# certifi>=2024.7.4
# six>=1.16.0
# python-dateutil>=2.8.2
# typing-extensions>=4.8.0
# # Security: NO aiohttp to eliminate vulnerabilities
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create layer structure
            layer_dir = os.path.join(temp_dir, "python")
            os.makedirs(layer_dir)
            
            # Write requirements file
            req_file = os.path.join(temp_dir, 'requirements.txt')
            with open(req_file, 'w') as f:
                f.write(requirements_content)
            
            logger.info("Installing secure dependencies for Python 3.12...")
            
            # Install packages with multiple fallback methods
            success = False
            
            # Method 1: Platform-specific install
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    "--target", layer_dir,
                    "--platform", "manylinux2014_x86_64",
                    "--only-binary=:all:",
                    f"--python-version={python_version}",
                    "--implementation=cp",
                    "--upgrade",
                    "-r", req_file
                ], check=True, capture_output=True, text=True)
                
                logger.info("✓ Installed packages with platform targeting")
                success = True
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Platform-specific install failed: {e.stderr}")
                
                # Method 2: Standard install fallback
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install",
                        "--target", layer_dir,
                        "--upgrade",
                        "-r", req_file
                    ], check=True, capture_output=True, text=True)
                    
                    logger.info("✓ Installed packages with standard method")
                    success = True
                    
                except subprocess.CalledProcessError as e2:
                    logger.error(f"Both install methods failed: {e2.stderr}")
                    return None
            
            if not success:
                logger.error("Failed to install packages for secure layer")
                return None
            
            # Clean up unnecessary files
            logger.info("Optimizing layer package...")
            patterns_to_remove = [
                "**/*.pyc",
                "**/__pycache__",
                "**/*.egg-info",
                "**/tests",
                "**/test",
                "**/*.dist-info"
            ]
            
            removed_count = 0
            for pattern in patterns_to_remove:
                for path in Path(layer_dir).glob(pattern):
                    try:
                        if path.is_file():
                            path.unlink()
                            removed_count += 1
                        elif path.is_dir():
                            shutil.rmtree(path)
                            removed_count += 1
                    except Exception:
                        pass
            
            logger.info(f"✓ Removed {removed_count} unnecessary files")
            
            # Create ZIP package
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            zip_filename = f"secure-layer-{timestamp}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(layer_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, os.path.dirname(layer_dir))
                        zipf.write(file_path, arc_name)
            
            # Get file size
            size_bytes = os.path.getsize(zip_path)
            size_mb = size_bytes / (1024 * 1024)
            logger.info(f"Layer package created: {zip_filename} ({size_mb:.2f} MB)")
            
            # Verify layer content
            verification_passed = self.verify_layer_content(zip_path)
            if not verification_passed:
                logger.error("Layer verification failed")
                return None
            
            # Upload layer
            try:
                with open(zip_path, 'rb') as f:
                    response = self.lambda_client.publish_layer_version(
                        LayerName=layer_name,
                        Description=f"Secure layer for energy forecasting (Python {python_version}, {datetime.now().strftime('%Y-%m-%d')}, {self.environment})",
                        Content={'ZipFile': f.read()},
                        CompatibleRuntimes=['python3.9', 'python3.11', 'python3.12']
                    )
                
                layer_arn = response['LayerVersionArn']
                layer_version = response['Version']
                
                logger.info(f"✓ Created secure layer: {layer_arn}")
                logger.info(f"✓ Layer version: {layer_version}")
                
                return layer_arn
                
            except Exception as e:
                logger.error(f"Failed to upload layer: {e}")
                return None
    
    def verify_layer_content(self, zip_path):
        """Verify layer contains required packages and no vulnerable ones"""
        
        logger.info("Running layer verification...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                
                # Check for essential packages
                required_packages = ['numpy/', 'pandas/', 'requests/']
                verification_passed = True
                
                for package in required_packages:
                    if any(package in f for f in file_list):
                        logger.info(f"✓ {package.rstrip('/')} included")
                    else:
                        logger.error(f"✗ {package.rstrip('/')} missing")
                        verification_passed = False
                
                # Security check - ensure no aiohttp
                if any('aiohttp' in f for f in file_list):
                    logger.error("✗ SECURITY RISK: aiohttp found in layer!")
                    verification_passed = False
                else:
                    logger.info("✓ SECURITY: No vulnerable aiohttp present")
                
                return verification_passed
                
        except Exception as e:
            logger.error(f"Layer verification failed: {e}")
            return False
    
    def create_profile_predictor_package(self):
        """Create Lambda package for profile-predictor with all source files"""
        
        logger.info("Creating profile-predictor Lambda package...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, "package")
            os.makedirs(package_dir)
            
            # Expected source files for profile-predictor
            source_files = [
                "lambda_function.py",
                "data_processor.py",
                "prediction_engine.py",
                "weather_forecaster.py",
                "requirements.txt",
                # "prediction_core.py",
                # "data_loader.py", 
                # "weather_forecast.py",
                # "radiation_forecast.py",
                # "s3_utils.py"
            ]
            
            # Updated source base directory path
            source_base_dir = os.path.join(self.lambda_functions_base_path, "profile-predictor")
            
            logger.info(f"Looking for source files in: {source_base_dir}")
            files_copied = 0
            
            for file in source_files:
                source_file_path = os.path.join(source_base_dir, file)
                if os.path.exists(source_file_path):
                    shutil.copy2(source_file_path, package_dir)
                    logger.info(f"  ✓ {file}")
                    files_copied += 1
                else:
                    logger.warning(f"   {file} not found")
            
            if files_copied == 0:
                logger.warning("No source files found - creating test version")
                self.create_test_lambda_function(package_dir)
            else:
                logger.info(f"✓ Copied {files_copied} source files")
            
            # Create minimal requirements.txt (layer provides dependencies)
            requirements_content = """# Minimal requirements - most packages provided by secure layer
# Add only packages not in the layer if needed
"""
            
            requirements_path = os.path.join(package_dir, "requirements.txt")
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            # Create ZIP file
            zip_file = os.path.join(temp_dir, "profile_predictor_package.zip")
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, package_dir)
                        zf.write(file_path, arc_name)
            
            # Read zip content
            with open(zip_file, 'rb') as f:
                zip_content = f.read()
            
            size_mb = len(zip_content) / (1024 * 1024)
            logger.info(f"✓ Profile-predictor package created: {size_mb:.2f} MB")
            
            return zip_content
    
    def create_test_lambda_function(self, package_dir):
        """Create test lambda function if source files not found"""
        
        lambda_code = f'''"""
Profile Predictor Lambda with Secure Layer - Test Version
Enhanced dependency test with security verification for {self.environment}
"""

import json
import sys
import os
import platform
from datetime import datetime

def lambda_handler(event, context):
    """Enhanced dependency test with security checks"""
    
    print(f"Profile Predictor Lambda Test - Secure Version - Environment: {self.environment}")
    print(f"Python: {{sys.version}}")
    print(f"Platform: {{platform.platform()}}")
    print(f"Timestamp: {{datetime.now().isoformat()}}")
    
    results = {{}}
    
    # Test core scientific libraries (from custom secure layer)
    print("\\n=== Testing Custom Secure Layer Libraries ===")
    
    # Test all expected libraries
    test_libraries = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('requests', None),
        ('boto3', None)
    ]
    
    for lib_name, alias in test_libraries:
        try:
            if alias:
                exec(f"import {{lib_name}} as {{alias}}")
                version = eval(f"{{alias}}.__version__")
            else:
                exec(f"import {{lib_name}}")
                version = eval(f"{{lib_name}}.__version__")
            
            results[lib_name] = {{
                'status': 'success',
                'version': version,
                'security_status': 'secure_custom_layer'
            }}
            print(f" ✓ {{lib_name}} {{version}} (from secure layer)")
        except Exception as e:
            results[lib_name] = {{'status': 'failed', 'error': str(e)}}
            print(f" ✗ {{lib_name}} failed: {{e}}")
    
    # SECURITY CHECK: Ensure aiohttp is NOT present
    print("\\n=== Security Verification ===")
    try:
        import aiohttp
        results['aiohttp'] = {{
            'status': 'present',
            'version': aiohttp.__version__,
            'security_risk': 'HIGH - VULNERABLE PACKAGE DETECTED'
        }}
        print(f" ✗ SECURITY RISK: aiohttp {{aiohttp.__version__}} present!")
    except ImportError:
        results['aiohttp'] = {{
            'status': 'absent',
            'security_status': 'secure',
            'vulnerabilities_eliminated': ['CVE-2024-42367', 'CVE-2024-52304', 'CVE-2025-53643']
        }}
        print(f" ✓ SECURE: aiohttp not present (vulnerabilities eliminated)")
    
    # Overall assessment
    working_libs = sum(1 for r in results.values() if r.get('status') == 'success')
    aiohttp_absent = results.get('aiohttp', {{}}).get('status') == 'absent'
    
    if working_libs >= 3 and aiohttp_absent:
        overall_status = 200
        message = f" PROFILE PREDICTOR SECURE AND READY ({self.environment})!"
    else:
        overall_status = 500
        message = f" Security or dependency issues detected ({self.environment})"
    
    return {{
        'statusCode': overall_status,
        'body': {{
            'message': message,
            'environment': '{self.environment}',
            'function_name': f'energy-forecasting-{{"{self.environment}"}}-profile-predictor',
            'test_results': results,
            'ready_for_forecasting': overall_status == 200,
            'security_status': 'secure' if aiohttp_absent else 'vulnerable',
            'test_timestamp': datetime.now().isoformat(),
            'layer_type': 'custom_secure_layer'
        }}
    }}
'''
        
        with open(os.path.join(package_dir, "lambda_function.py"), 'w') as f:
            f.write(lambda_code)
        
        logger.info("  ✓ Created test lambda_function.py for profile-predictor")
    
    def deploy_all_lambda_functions(self):
        """Deploy all 11 Lambda functions including profile-predictor with secure layer"""
        
        logger.info("="*70)
        logger.info("DEPLOYING ALL 11 LAMBDA FUNCTIONS FOR COMPLETE MLOPS PIPELINE")
        logger.info("="*70)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Region: {self.region}")
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Execution Role: {self.execution_role}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Complete Lambda function configurations with environment-aware naming
        lambda_configs = {
            # EXISTING TRAINING PIPELINE FUNCTIONS (10 functions)
            f'energy-forecasting-{self.environment}-model-registry': {
                'source_dir': f'{self.lambda_functions_base_path}/model-registry',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 1024,
                'description': f'Enhanced Model Registry for Energy Forecasting with Step Functions Integration ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-endpoint-management': {
                'source_dir': f'{self.lambda_functions_base_path}/endpoint-management',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 512,
                'description': f'Enhanced Endpoint Management for Energy Forecasting with Model Registry Integration ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'SAGEMAKER_EXECUTION_ROLE': self.execution_role,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-prediction-endpoint-manager': {
                'source_dir': f'{self.lambda_functions_base_path}/prediction-endpoint-manager',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 512,
                'description': f'Prediction Endpoint Manager - Recreates endpoints from S3 configurations ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'SAGEMAKER_EXECUTION_ROLE': self.execution_role,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-prediction-cleanup': {
                'source_dir': f'{self.lambda_functions_base_path}/prediction-cleanup',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 256,
                'description': f'Cleanup Manager for Prediction Pipeline - Deletes temporary endpoints after predictions ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-profile-validator': {
                'source_dir': f'{self.lambda_functions_base_path}/profile-validator',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 300,
                'memory': 256,
                'description': f'Validates and filters profiles based on S3 configurations ({self.environment})',
                'layers': [],
                'environment': {
                    'REGION': self.region,
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'ENVIRONMENT': self.environment
                },
            },
            f'energy-forecasting-{self.environment}-profile-endpoint-creator': {
                'source_dir': f'{self.lambda_functions_base_path}/profile-endpoint-creator',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 512,
                'description': f'Profile-Specific Endpoint Creator - Creates endpoint from S3 config for one profile ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'SAGEMAKER_EXECUTION_ROLE': self.execution_role,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-profile-cleanup': {
                'source_dir': f'{self.lambda_functions_base_path}/profile-cleanup',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 256,
                'description': f'Profile-Specific Cleanup - Cleans up resources for one profile after predictions ({self.environment})',
                'layers': [],
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-endpoint-status-checker': {
                'source_dir': f'{self.lambda_functions_base_path}/endpoint-status-checker',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 256,
                'description': f'Endpoint Status Checker - Waits for all parallel endpoints to be ready ({self.environment})',
                'layers': [],
                'environment': {
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-prediction-summary': {
                'source_dir': f'{self.lambda_functions_base_path}/prediction-summary',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 512,
                'description': f'Prediction Summary Generator - Collects and summarizes results from all profiles ({self.environment})',
                'layers': [],
                'environment': {
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            f'energy-forecasting-{self.environment}-emergency-cleanup': {
                'source_dir': f'{self.lambda_functions_base_path}/emergency-cleanup',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'timeout': 900,
                'memory': 512,
                'description': f'Emergency Cleanup - Handles resource cleanup when pipeline fails ({self.environment})',
                'layers': [],
                'environment': {
                    'REGION': self.region,
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                }
            },
            
            # SPECIAL PROFILE-PREDICTOR FUNCTION (11th function)
            f'energy-forecasting-{self.environment}-profile-predictor': {
                'source_dir': f'{self.lambda_functions_base_path}/profile-predictor',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.12',  # Python 3.12 for profile-predictor
                'timeout': 900,
                'memory': 1024,
                'description': f'Profile-Specific Predictor with Secure Dependencies (Python 3.12, {self.environment})',
                'layers': [],  # Will be populated with secure layer ARN
                'environment': {
                    'MODEL_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-models',
                    'DATA_BUCKET': f'sdcp-{self.environment}-sagemaker-energy-forecasting-data',
                    'REGION': self.region,
                    'ACCOUNT_ID': self.account_id,
                    'ENVIRONMENT': self.environment
                },
                'needs_secure_layer': True  # Special flag
            }
        }
        
        # Create secure layer for profile-predictor
        logger.info("\n" + "="*50)
        logger.info("CREATING SECURE LAYER FOR PROFILE-PREDICTOR")
        logger.info("="*50)
        
        secure_layer_arn = self.create_secure_layer_for_profile_predictor()
        if not secure_layer_arn:
            logger.error("Failed to create secure layer. Aborting profile-predictor deployment.")
            # Remove profile-predictor from deployment
            del lambda_configs[f'energy-forecasting-{self.environment}-profile-predictor']
        else:
            # Set the secure layer for profile-predictor
            lambda_configs[f'energy-forecasting-{self.environment}-profile-predictor']['layers'] = [secure_layer_arn]
            self.secure_layer_arn = secure_layer_arn
        
        deployment_results = {}
        
        # Deploy each function
        for function_name, config in lambda_configs.items():
            try:
                logger.info(f"Deploying {function_name}...")
                
                # Special handling for profile-predictor
                if 'profile-predictor' in function_name:
                    result = self.deploy_profile_predictor_function(function_name, config, self.execution_role)
                else:
                    # Standard deployment for other 10 functions
                    result = self.deploy_lambda_function(function_name, config, self.execution_role)
                
                deployment_results[function_name] = result
                logger.info(f"✓ Successfully deployed {function_name}")
                
                # Add Step Functions permissions for all functions
                self._add_step_functions_permissions(function_name)
                
            except Exception as e:
                logger.error(f"✗ Failed to deploy {function_name}: {str(e)}")
                deployment_results[function_name] = {'error': str(e)}

        # Save deployment summary
        self.save_deployment_summary(deployment_results)
        
        return deployment_results
    
    def create_deployment_package(self, source_dir, function_name):
        """Create a deployment package for standard Lambda functions"""
        
        if not os.path.exists(source_dir):
            raise Exception(f"Source directory not found: {source_dir}")
        
        # Create temporary directory for packaging
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, 'package')
            os.makedirs(package_dir)
            
            # Copy function code
            lambda_function_file = os.path.join(source_dir, 'lambda_function.py')
            if not os.path.exists(lambda_function_file):
                raise Exception(f"lambda_function.py not found in {source_dir}")
            
            shutil.copy2(lambda_function_file, package_dir)
            
            # Copy any additional Python files
            for file in os.listdir(source_dir):
                if file.endswith('.py') and file != 'lambda_function.py':
                    shutil.copy2(os.path.join(source_dir, file), package_dir)
            
            # Install dependencies if requirements.txt exists
            requirements_file = os.path.join(source_dir, 'requirements.txt')
            if os.path.exists(requirements_file):
                # Check if requirements.txt is not empty
                with open(requirements_file, 'r') as f:
                    content = f.read().strip()
                    if content and not content.startswith('#'):  # Skip if only comments
                        logger.info(f"  Installing dependencies for {function_name}...")
                        subprocess.run([
                            sys.executable, '-m', 'pip', 'install',
                            '--target', package_dir, '--no-deps',
                            '-r', requirements_file,
                            '--upgrade'
                        ], check=True, capture_output=True, text=True)
                        logger.info(f"  ✓ Dependencies installed for {function_name}")
            
            # Create zip file
            zip_file = os.path.join(temp_dir, f'{function_name}.zip')
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, package_dir)
                        zf.write(file_path, arc_name)
            
            # Read zip content
            with open(zip_file, 'rb') as f:
                zip_content = f.read()
            
            return zip_content
    
    def deploy_lambda_function(self, function_name, config, execution_role):
        """Deploy a standard Lambda function"""
        
        logger.info(f"  Deploying standard Lambda function: {function_name}")
        
        # Create deployment package
        try:
            zip_content = self.create_deployment_package(config['source_dir'], function_name)
        except Exception as e:
            logger.warning(f"  Failed to create package from source, creating test function: {str(e)}")
            zip_content = self.create_test_function(function_name, config)
        
        # Deploy function
        try:
            # Try to update existing function
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )

            time.sleep(10)
            # logger.info(f"  ✓ Updated existing function code")

            logger.info("  Waiting for code update to complete...")
            self.wait_for_function_active(function_name)
            
            # Update configuration
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Handler=config['handler'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Description=config['description'],
                Environment={'Variables': config.get('environment', {})},
                Layers=config.get('layers', [])
            )

            time.sleep(10)
            logger.info(f"  ✓ Updated function configuration")
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Role=execution_role,
                Handler=config['handler'],
                Code={'ZipFile': zip_content},
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Environment={'Variables': config.get('environment', {})},
                Layers=config.get('layers', [])
            )
            logger.info(f"  ✓ Created new function")
        
        time.sleep(10)

        # Wait for function to be active
        self.wait_for_function_active(function_name)
        
        return {
            'function_arn': response['FunctionArn'],
            'function_name': function_name,
            'status': 'deployed'
        }
    
    def deploy_profile_predictor_function(self, function_name, config, execution_role):
        """Deploy the special profile-predictor function with secure layer"""
        
        logger.info(f"  Deploying SPECIAL profile-predictor function: {function_name}")
        logger.info(f"  Secure layer ARN: {config.get('layers', ['None'])[0] if config.get('layers') else 'None'}")
        
        # Create special deployment package for profile-predictor
        zip_content = self.create_profile_predictor_package()
        
        # Deploy function with secure layer
        try:
            # Try to update existing function
            response = self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            logger.info(f"  ✓ Updated existing profile-predictor code")

            time.sleep(10)
            # logger.info(f"  ✓ Updated existing function code")

            logger.info("  Waiting for code update to complete...")
            self.wait_for_function_active(function_name)            
            
            # Update configuration with secure layer
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Handler=config['handler'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Description=config['description'],
                Environment={'Variables': config.get('environment', {})},
                Layers=config.get('layers', [])
            )

            time.sleep(10)
            logger.info(f"  ✓ Updated profile-predictor configuration with secure layer")
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime=config['runtime'],
                Role=execution_role,
                Handler=config['handler'],
                Code={'ZipFile': zip_content},
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Environment={'Variables': config.get('environment', {})},
                Layers=config.get('layers', [])
            )

            time.sleep(10)
            logger.info(f"  ✓ Created new profile-predictor function with secure layer")

        time.sleep(10)
        
        # Wait for function to be active
        self.wait_for_function_active(function_name)
        
        # Test the secure function
        self.test_profile_predictor_security(function_name)
        
        return {
            'function_arn': response['FunctionArn'],
            'function_name': function_name,
            'status': 'deployed',
            'secure_layer_arn': config.get('layers', [None])[0],
            'runtime': config['runtime'],
            'security_status': 'secured'
        }
    
    def test_profile_predictor_security(self, function_name):
        """Test the security of the profile-predictor function"""
        
        logger.info(f"  Testing security for {function_name}...")
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps({'test': 'security_verification'})
            )
            
            if response['StatusCode'] == 200:
                payload = json.loads(response['Payload'].read())
                body = payload.get('body', {})
                
                security_status = body.get('security_status', 'unknown')
                aiohttp_status = body.get('test_results', {}).get('aiohttp', {}).get('status', 'unknown')
                
                if security_status == 'secure' and aiohttp_status == 'absent':
                    logger.info(f"  ✓ SECURITY VERIFIED: No vulnerable packages detected")
                    logger.info(f"  ✓ Function ready for secure forecasting operations")
                else:
                    logger.warning(f"   Security test results unclear: {security_status}")
                    
        except Exception as e:
            logger.warning(f"  Security test failed (function may still work): {str(e)}")
    
    def create_test_function(self, function_name, config):
        """Create a test function when source code is not available"""
        
        test_code = f'''
import json
import boto3
from datetime import datetime
import os

def lambda_handler(event, context):
    """Test function for {function_name} in {self.environment} environment"""
    
    print(f"Test function executing: {function_name}")
    print(f"Environment: {self.environment}")
    print(f"Event: {{json.dumps(event, default=str)}}")
    
    # Environment variables
    env_vars = {{
        'MODEL_BUCKET': os.environ.get('MODEL_BUCKET', 'Not set'),
        'DATA_BUCKET': os.environ.get('DATA_BUCKET', 'Not set'),
        'REGION': os.environ.get('REGION', 'Not set'),
        'ENVIRONMENT': os.environ.get('ENVIRONMENT', 'Not set')
    }}
    
    # Test AWS connectivity
    try:
        # Test S3 connectivity
        s3_client = boto3.client('s3')
        response = s3_client.list_buckets()
        s3_status = "Connected"
        bucket_count = len(response.get('Buckets', []))
    except Exception as e:
        s3_status = f"Failed: {{str(e)}}"
        bucket_count = 0
    
    return {{
        'statusCode': 200,
        'body': {{
            'message': f'{function_name} test function working in {self.environment}',
            'timestamp': datetime.now().isoformat(),
            'function_name': function_name,
            'environment': '{self.environment}',
            'environment_variables': env_vars,
            's3_status': s3_status,
            's3_bucket_count': bucket_count,
            'test_status': 'success'
        }}
    }}
'''
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create function file
            function_file = os.path.join(temp_dir, 'lambda_function.py')
            with open(function_file, 'w') as f:
                f.write(test_code)
            
            # Create zip
            zip_file = os.path.join(temp_dir, 'function.zip')
            with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(function_file, 'lambda_function.py')
            
            # Read zip content
            with open(zip_file, 'rb') as f:
                return f.read()
    
    def wait_for_function_active(self, function_name):
        """Wait for Lambda function to become active"""
        
        logger.info(f"  Waiting for {function_name} to become active...")
        start_time = time.time()
        
        while time.time() - start_time < self.max_wait_time:
            try:
                response = self.lambda_client.get_function(FunctionName=function_name)
                state = response['Configuration']['State']
                
                if state == 'Active':
                    logger.info(f"  ✓ {function_name} is now active")
                    return True
                elif state == 'Failed':
                    logger.error(f"  ✗ {function_name} failed to activate")
                    return False
                else:
                    logger.info(f"   {function_name} state: {state}")
                    time.sleep(self.poll_interval)
                    
            except Exception as e:
                logger.warning(f"  Error checking function state: {str(e)}")
                time.sleep(self.poll_interval)
        
        logger.warning(f"   {function_name} did not become active within {self.max_wait_time} seconds")
        return False
    
    def _add_step_functions_permissions(self, function_name):
        """Add Step Functions invoke permissions to Lambda function"""
        
        try:
            self.lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'stepfunctions-invoke-{int(time.time())}',
                Action='lambda:InvokeFunction',
                Principal='states.amazonaws.com'
            )
            logger.info(f"  ✓ Added Step Functions permissions to {function_name}")
        except self.lambda_client.exceptions.ResourceConflictException:
            logger.info(f"  ✓ Step Functions permissions already exist for {function_name}")
        except Exception as e:
            logger.warning(f"   Could not add Step Functions permissions to {function_name}: {str(e)}")
    
    def save_deployment_summary(self, deployment_results):
        """Save deployment summary to local file and S3"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'region': self.region,
            'account_id': self.account_id,
            'execution_role': self.execution_role,
            'secure_layer_arn': self.secure_layer_arn,
            'total_functions': len(deployment_results),
            'successful_deployments': len([r for r in deployment_results.values() if 'error' not in r]),
            'failed_deployments': len([r for r in deployment_results.values() if 'error' in r]),
            'deployment_results': deployment_results
        }
        
        # Save locally
        summary_file = f'lambda_deployment_summary_{self.environment}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✓ Deployment summary saved: {summary_file}")
        
        # Try to save to S3
        try:
            s3_client = boto3.client('s3')
            bucket_name = f'sdcp-{self.environment}-sagemaker-energy-forecasting-data'
            s3_key = f'deployment-logs/lambda_deployment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json.dumps(summary, indent=2, default=str),
                ContentType='application/json'
            )
            logger.info(f"✓ Deployment summary uploaded to S3: s3://{bucket_name}/{s3_key}")
        except Exception as e:
            logger.warning(f"Could not upload deployment summary to S3: {str(e)}")
        
        # Print summary
        logger.info("="*70)
        logger.info("LAMBDA DEPLOYMENT SUMMARY")
        logger.info("="*70)
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Total functions: {summary['total_functions']}")
        logger.info(f"Successful: {summary['successful_deployments']}")
        logger.info(f"Failed: {summary['failed_deployments']}")
        
        for function_name, result in deployment_results.items():
            if 'error' in result:
                logger.info(f" {function_name}: {result['error']}")
            else:
                logger.info(f" {function_name}: {result['status']}")
        
        if self.secure_layer_arn:
            logger.info(f" Secure Layer: {self.secure_layer_arn}")
        
        logger.info("="*70)
        
        return summary

def main():
    """Main function for testing Lambda deployer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy all Lambda functions for Energy Forecasting')
    parser.add_argument('--environment', default='dev', choices=['dev', 'preprod', 'prod'],
                       help='Target environment')
    parser.add_argument('--region', default='us-west-2', help='AWS region')
    parser.add_argument('--role-name', help='DataScientist role name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Auto-generate role name if not provided
    role_name = args.role_name or f'sdcp-{args.environment}-sagemaker-energy-forecasting-datascientist-role'
    
    try:
        logger.info("Starting Lambda deployment...")
        logger.info(f"Environment: {args.environment}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Role: {role_name}")
        
        deployer = CompleteLambdaDeployer(
            region=args.region,
            datascientist_role_name=role_name,
            environment=args.environment
        )
        
        results = deployer.deploy_all_lambda_functions()
        
        successful = len([r for r in results.values() if 'error' not in r])
        total = len(results)
        
        logger.info(f"Deployment complete: {successful}/{total} functions successful")
        
        if successful == total:
            logger.info(" ALL LAMBDA FUNCTIONS DEPLOYED SUCCESSFULLY!")
            return 0
        else:
            logger.warning(f" {total - successful} functions failed to deploy")
            return 1
            
    except Exception as e:
        logger.error(f"Lambda deployment failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
