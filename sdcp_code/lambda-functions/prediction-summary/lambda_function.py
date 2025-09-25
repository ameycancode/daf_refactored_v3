"""
Prediction Summary Lambda Function
Collects and summarizes results from all profile predictions
"""

import json
import boto3
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from io import StringIO, BytesIO

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Generate summary of all profile prediction results
    """
   
    execution_id = context.aws_request_id
   
    try:
        logger.info(f"Starting prediction summary generation [{execution_id}]")
        logger.info(f"Event: {json.dumps(event, default=str)}")
       
        # Extract parameters
        operation = event.get('operation', 'generate_summary')
        prediction_results = event.get('prediction_results', [])
        data_bucket = event.get('data_bucket', 'sdcp-dev-sagemaker-energy-forecasting-data')
        pipeline_execution_id = event.get('execution_id', execution_id)
       
        if operation == 'generate_summary':
            result = generate_prediction_summary(prediction_results, data_bucket, pipeline_execution_id)
        elif operation == 'generate_detailed_report':
            result = generate_detailed_prediction_report(prediction_results, data_bucket, pipeline_execution_id)
        else:
            raise ValueError(f"Unknown operation: {operation}")
       
        return {
            'statusCode': 200,
            'body': result
        }
       
    except Exception as e:
        logger.error(f"Prediction summary generation failed [{execution_id}]: {str(e)}")
        return {
            'statusCode': 500,
            'body': {
                'operation': event.get('operation', 'unknown'),
                'status': 'failed',
                'error': str(e),
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat()
            }
        }

def generate_prediction_summary(prediction_results: List[Dict[str, Any]], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Generate comprehensive summary of all profile predictions
    """
   
    try:
        logger.info(f"Generating prediction summary for {len(prediction_results)} profiles")
       
        # Initialize summary structure
        summary = {
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat(),
            'pipeline_status': 'unknown',
            'profiles_summary': {},
            'overall_metrics': {},
            'output_files': {},
            'errors': []
        }
       
        successful_profiles = []
        failed_profiles = []
        total_predictions = 0
       
        # Process each profile result
        for result in prediction_results:
            if not isinstance(result, dict):
                continue
           
            profile = result.get('profile', 'unknown')
            status = result.get('status', 'unknown')
           
            if status == 'success':
                successful_profiles.append(profile)
               
                # Extract prediction metrics if available
                if 'prediction_results' in result:
                    pred_info = result['prediction_results']
                    predictions_count = pred_info.get('predictions_generated', 0)
                    total_predictions += predictions_count
                   
                    summary['profiles_summary'][profile] = {
                        'status': 'success',
                        'predictions_count': predictions_count,
                        'output_files': pred_info.get('output_files', {}),
                        'visualization_files': pred_info.get('visualization_files', {}),
                        'endpoint_name': pred_info.get('endpoint_name', 'unknown')
                    }
                else:
                    summary['profiles_summary'][profile] = {
                        'status': 'success',
                        'predictions_count': 0,
                        'note': 'No detailed prediction results available'
                    }
               
            else:
                failed_profiles.append(profile)
                error_msg = result.get('error', 'Unknown error')
                summary['profiles_summary'][profile] = {
                    'status': 'failed',
                    'error': error_msg
                }
                summary['errors'].append(f"{profile}: {error_msg}")
       
        # Calculate overall metrics
        total_profiles = len(successful_profiles) + len(failed_profiles)
        success_rate = (len(successful_profiles) / total_profiles * 100) if total_profiles > 0 else 0
       
        summary['overall_metrics'] = {
            'total_profiles_processed': total_profiles,
            'successful_profiles': len(successful_profiles),
            'failed_profiles': len(failed_profiles),
            'success_rate_percent': round(success_rate, 2),
            'total_predictions_generated': total_predictions,
            'successful_profile_list': successful_profiles,
            'failed_profile_list': failed_profiles
        }
       
        # Determine pipeline status
        if len(failed_profiles) == 0:
            summary['pipeline_status'] = 'completed_successfully'
        elif len(successful_profiles) > 0:
            summary['pipeline_status'] = 'completed_with_partial_failures'
        else:
            summary['pipeline_status'] = 'failed'
       
        # Generate and save consolidated summary report
        summary_files = save_summary_to_s3(summary, data_bucket, execution_id)
        summary['output_files']['summary_report'] = summary_files
       
        # Generate consolidated predictions file if we have successful profiles
        if successful_profiles:
            try:
                consolidated_file = create_consolidated_predictions_file(successful_profiles, data_bucket, execution_id)
                if consolidated_file:
                    summary['output_files']['consolidated_predictions'] = consolidated_file
            except Exception as e:
                logger.warning(f"Could not create consolidated predictions file: {str(e)}")
       
        logger.info(f"Summary generated: {len(successful_profiles)}/{total_profiles} profiles successful")
        logger.info(f"Total predictions: {total_predictions}")
       
        return summary
       
    except Exception as e:
        logger.error(f"Failed to generate prediction summary: {str(e)}")
        return {
            'execution_id': execution_id,
            'pipeline_status': 'summary_generation_failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def save_summary_to_s3(summary: Dict[str, Any], data_bucket: str, execution_id: str) -> Dict[str, str]:
    """
    Save summary report to S3
    """
   
    try:
        current_date = datetime.now().strftime('%Y%m%d')
       
        # Save detailed JSON summary
        json_key = f"archived_folders/forecasting/summaries/{current_date}/prediction_summary_{execution_id}.json"
       
        s3_client.put_object(
            Bucket=data_bucket,
            Key=json_key,
            Body=json.dumps(summary, indent=2, default=str),
            ContentType='application/json'
        )
       
        # Generate human-readable summary
        readable_summary = generate_readable_summary(summary)
       
        # Save readable summary
        txt_key = f"archived_folders/forecasting/summaries/{current_date}/prediction_summary_{execution_id}.txt"
       
        s3_client.put_object(
            Bucket=data_bucket,
            Key=txt_key,
            Body=readable_summary,
            ContentType='text/plain'
        )
       
        logger.info(f"Saved summary reports to S3: {json_key}, {txt_key}")
       
        return {
            'json_summary': json_key,
            'readable_summary': txt_key
        }
       
    except Exception as e:
        logger.error(f"Failed to save summary to S3: {str(e)}")
        return {}

def generate_readable_summary(summary: Dict[str, Any]) -> str:
    """
    Generate human-readable summary report
    """
   
    lines = []
    lines.append("="*60)
    lines.append("ENERGY FORECASTING PREDICTION PIPELINE SUMMARY")
    lines.append("="*60)
    lines.append(f"Execution ID: {summary.get('execution_id', 'unknown')}")
    lines.append(f"Timestamp: {summary.get('timestamp', 'unknown')}")
    lines.append(f"Pipeline Status: {summary.get('pipeline_status', 'unknown').upper()}")
    lines.append("")
   
    # Overall metrics
    metrics = summary.get('overall_metrics', {})
    lines.append("OVERALL METRICS:")
    lines.append(f"  Total Profiles Processed: {metrics.get('total_profiles_processed', 0)}")
    lines.append(f"  Successful Profiles: {metrics.get('successful_profiles', 0)}")
    lines.append(f"  Failed Profiles: {metrics.get('failed_profiles', 0)}")
    lines.append(f"  Success Rate: {metrics.get('success_rate_percent', 0):.1f}%")
    lines.append(f"  Total Predictions Generated: {metrics.get('total_predictions_generated', 0)}")
    lines.append("")
   
    # Profile details
    lines.append("PROFILE DETAILS:")
    profiles_summary = summary.get('profiles_summary', {})
   
    for profile, details in profiles_summary.items():
        status = details.get('status', 'unknown')
        lines.append(f"  {profile}: {status.upper()}")
       
        if status == 'success':
            pred_count = details.get('predictions_count', 0)
            endpoint = details.get('endpoint_name', 'unknown')
            lines.append(f"    Predictions: {pred_count}")
            lines.append(f"    Endpoint: {endpoint}")
        elif status == 'failed':
            error = details.get('error', 'Unknown error')
            lines.append(f"    Error: {error}")
        lines.append("")
   
    # Errors summary
    errors = summary.get('errors', [])
    if errors:
        lines.append("ERRORS:")
        for error in errors:
            lines.append(f"  • {error}")
        lines.append("")
   
    # Output files
    output_files = summary.get('output_files', {})
    if output_files:
        lines.append("OUTPUT FILES:")
        for file_type, file_info in output_files.items():
            if isinstance(file_info, dict):
                for key, path in file_info.items():
                    lines.append(f"  {file_type}_{key}: {path}")
            else:
                lines.append(f"  {file_type}: {file_info}")
        lines.append("")
   
    lines.append("="*60)
    lines.append("END OF SUMMARY")
    lines.append("="*60)
   
    return "\n".join(lines)

def create_consolidated_predictions_file(successful_profiles: List[str], data_bucket: str, execution_id: str) -> str:
    """
    Create a consolidated predictions file from all successful profiles
    """
   
    try:
        current_date = datetime.now().strftime('%Y%m%d')
       
        # Collect all prediction files
        all_predictions = []
       
        for profile in successful_profiles:
            # Try to find and load prediction file for this profile
            prediction_key = f"archived_folders/forecasting/data/xgboost/output/{current_date}/{profile}_predictions_{current_date}.csv"
           
            try:
                response = s3_client.get_object(Bucket=data_bucket, Key=prediction_key)
                df = pd.read_csv(BytesIO(response['Body'].read()))
               
                # Add profile identifier
                df['Profile'] = profile
                all_predictions.append(df)
               
                logger.info(f"Loaded {len(df)} predictions from {profile}")
               
            except s3_client.exceptions.NoSuchKey:
                logger.warning(f"Prediction file not found for {profile}: {prediction_key}")
                continue
            except Exception as e:
                logger.warning(f"Could not load predictions for {profile}: {str(e)}")
                continue
       
        if not all_predictions:
            logger.warning("No prediction files found to consolidate")
            return ""
       
        # Consolidate all predictions
        consolidated_df = pd.concat(all_predictions, ignore_index=True)
       
        # Save consolidated file
        consolidated_key = f"archived_folders/forecasting/data/xgboost/output/{current_date}/consolidated_predictions_{execution_id}.csv"
       
        csv_buffer = StringIO()
        consolidated_df.to_csv(csv_buffer, index=False)
       
        s3_client.put_object(
            Bucket=data_bucket,
            Key=consolidated_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
       
        logger.info(f"Created consolidated predictions file: {consolidated_key} ({len(consolidated_df)} total predictions)")
       
        return consolidated_key
       
    except Exception as e:
        logger.error(f"Failed to create consolidated predictions file: {str(e)}")
        return ""

def generate_detailed_prediction_report(prediction_results: List[Dict[str, Any]], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Generate detailed prediction report with analytics
    """
   
    try:
        # Generate basic summary first
        summary = generate_prediction_summary(prediction_results, data_bucket, execution_id)
       
        # Add detailed analytics
        detailed_report = {
            'basic_summary': summary,
            'detailed_analytics': {},
            'recommendations': []
        }
       
        # Analyze prediction patterns if we have successful profiles
        successful_profiles = summary.get('overall_metrics', {}).get('successful_profile_list', [])
       
        if successful_profiles:
            analytics = analyze_prediction_patterns(successful_profiles, data_bucket, execution_id)
            detailed_report['detailed_analytics'] = analytics
           
            # Generate recommendations based on analysis
            recommendations = generate_recommendations(summary, analytics)
            detailed_report['recommendations'] = recommendations
       
        # Save detailed report
        current_date = datetime.now().strftime('%Y%m%d')
        report_key = f"archived_folders/forecasting/reports/{current_date}/detailed_prediction_report_{execution_id}.json"
       
        s3_client.put_object(
            Bucket=data_bucket,
            Key=report_key,
            Body=json.dumps(detailed_report, indent=2, default=str),
            ContentType='application/json'
        )
       
        detailed_report['report_location'] = report_key
       
        return detailed_report
       
    except Exception as e:
        logger.error(f"Failed to generate detailed report: {str(e)}")
        return {
            'execution_id': execution_id,
            'status': 'detailed_report_failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def analyze_prediction_patterns(successful_profiles: List[str], data_bucket: str, execution_id: str) -> Dict[str, Any]:
    """
    Analyze patterns in the prediction data
    """
   
    analytics = {
        'profile_statistics': {},
        'load_distribution': {},
        'temporal_patterns': {}
    }
   
    try:
        current_date = datetime.now().strftime('%Y%m%d')
       
        for profile in successful_profiles:
            prediction_key = f"archived_folders/forecasting/data/xgboost/output/{current_date}/{profile}_predictions_{current_date}.csv"
           
            try:
                response = s3_client.get_object(Bucket=data_bucket, Key=prediction_key)
                df = pd.read_csv(BytesIO(response['Body'].read()))
               
                if 'Predicted_Load' in df.columns:
                    pred_values = df['Predicted_Load'].dropna()
                   
                    analytics['profile_statistics'][profile] = {
                        'count': len(pred_values),
                        'mean': float(pred_values.mean()),
                        'std': float(pred_values.std()),
                        'min': float(pred_values.min()),
                        'max': float(pred_values.max()),
                        'median': float(pred_values.median())
                    }
               
            except Exception as e:
                logger.warning(f"Could not analyze {profile}: {str(e)}")
                continue
       
        return analytics
       
    except Exception as e:
        logger.warning(f"Analytics generation failed: {str(e)}")
        return analytics

def generate_recommendations(summary: Dict[str, Any], analytics: Dict[str, Any]) -> List[str]:
    """
    Generate recommendations based on pipeline results
    """
   
    recommendations = []
   
    try:
        metrics = summary.get('overall_metrics', {})
        success_rate = metrics.get('success_rate_percent', 0)
       
        # Success rate recommendations
        if success_rate == 100:
            recommendations.append("✓ Excellent: All profiles completed successfully")
        elif success_rate >= 80:
            recommendations.append("⚠ Good: Most profiles successful, investigate failed profiles")
        else:
            recommendations.append("⚠ Attention: Low success rate, review pipeline configuration")
       
        # Performance recommendations
        total_predictions = metrics.get('total_predictions_generated', 0)
        if total_predictions > 0:
            recommendations.append(f"✓ Generated {total_predictions} predictions successfully")
       
        # Profile-specific recommendations
        failed_profiles = metrics.get('failed_profile_list', [])
        if failed_profiles:
            recommendations.append(f"⚠ Review failed profiles: {', '.join(failed_profiles)}")
       
        return recommendations
       
    except Exception as e:
        logger.warning(f"Could not generate recommendations: {str(e)}")
        return ["ℹ Recommendations could not be generated"]
