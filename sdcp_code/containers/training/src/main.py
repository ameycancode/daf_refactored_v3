#!/usr/bin/env python3
"""
Refactored Training Container for SageMaker
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import sklearn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go
import plotly.offline as pyo
from datetime import datetime
import pytz
import logging
import json
from time import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append('/opt/ml/processing/code/src')

from config import EnergyForecastingConfig, S3FileManager

class EnergyTrainingPipeline:
    def __init__(self):
        # Initialize configuration
        self.config = EnergyForecastingConfig()
        self.s3_manager = S3FileManager(self.config)
        self.paths = self.config.get_container_paths()
       
        # Pacific timezone
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = self.config.current_date_str
       
        # Training configuration from JSON
        self.training_config = self.config.get_training_config()
       
        logger.info(f"Training pipeline initialized for date: {self.current_date}")
        logger.info(f"Training config: {self.training_config}")
   
    def run_training(self):
        """Main training pipeline matching original train_workflow.py"""
        try:
            logger.info("Starting training pipeline...")
            start_time = datetime.now()
           
            # Step 0: Download training data from S3 to local paths
            logger.info("Step 0: Downloading training data from S3...")
            self._download_training_data_from_s3()
           
            # Step 1: Load training datasets (now from local paths)
            logger.info("Step 1: Loading training datasets...")
            datasets = self._load_training_datasets()
           
            # Step 2: Train models for each dataset
            logger.info("Step 2: Training models...")
            results = self._train_all_models(datasets)
           
            # Step 3: Save training results exactly as original
            logger.info("Step 3: Saving training results...")
            self._save_training_results(results)
           
            # Step 4: Generate training summary
            self._generate_training_summary(results, start_time)
           
            logger.info("Training pipeline completed successfully!")
           
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            self._save_error_log(str(e))
            raise

    def _download_training_data_from_s3(self):
        """Download training data from S3 to local container paths"""
        logger.info("Downloading training data from S3...")
       
        try:
            # Create local directories
            train_dir = os.path.join(self.paths['input_path'], 'train_test_split', 'train')
            test_dir = os.path.join(self.paths['input_path'], 'train_test_split', 'test')
           
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
           
            profiles = self.config.get_profiles()
            download_count = 0
           
            for profile in profiles:
                # Download training file
                train_file = f"{profile}_train_{self.current_date}.csv"
                train_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/train/{train_file}"
                train_local_path = os.path.join(train_dir, train_file)
               
                logger.info(f"Downloading training data for {profile}...")
                logger.info(f"  S3 key: {train_s3_key}")
                logger.info(f"  Local path: {train_local_path}")
               
                if self.s3_manager.download_file(train_s3_key, train_local_path):
                    download_count += 1
                    logger.info(f"  ✅ Successfully downloaded {train_file}")
                else:
                    logger.warning(f"  ⚠️  Failed to download {train_file}")
               
                # Download test file as well (might be needed for validation)
                test_file = f"{profile}_test_{self.current_date}.csv"
                test_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/test/{test_file}"
                test_local_path = os.path.join(test_dir, test_file)
               
                if self.s3_manager.download_file(test_s3_key, test_local_path):
                    logger.info(f"  ✅ Also downloaded {test_file}")
                else:
                    logger.warning(f"  ⚠️  Test file not available: {test_file}")
           
            logger.info(f"Downloaded {download_count}/{len(profiles)} training datasets from S3")
           
            if download_count == 0:
                raise FileNotFoundError("No training datasets could be downloaded from S3")
           
            return download_count > 0
           
        except Exception as e:
            logger.error(f"Failed to download training data from S3: {str(e)}")
            raise

    def _download_additional_s3_data(self):
        """Download any additional data files needed for training"""
        logger.info("Downloading additional S3 data if needed...")
       
        try:
            # Download any lagged data files if they exist
            profiles = self.config.get_profiles()
           
            for profile in profiles:
                # Download lagged data if available
                lagged_file = f"{profile}_lagged_{self.current_date}.csv"
                lagged_s3_key = f"{self.config.config['s3']['processed_data_prefix']}{lagged_file}"
                lagged_local_path = os.path.join(self.paths['input_path'], lagged_file)
               
                if self.s3_manager.download_file(lagged_s3_key, lagged_local_path):
                    logger.info(f"  ✅ Downloaded additional data: {lagged_file}")
                else:
                    logger.info(f"  ℹ️  No additional lagged data: {lagged_file}")
           
            return True
           
        except Exception as e:
            logger.warning(f"Could not download additional S3 data: {str(e)}")
            return False

    def _save_error_log(self, error_message):
        """Save error log to S3"""
        try:
            error_log = {
                'timestamp': datetime.now(self.pacific_tz).isoformat(),
                'error_message': error_message
            }
            error_filename = f"training_error_{self.current_date}.json"
            local_error_path = os.path.join(self.paths['output_path'], error_filename)
            s3_error_key = f"{self.config.config['s3']['train_results_prefix']}{error_filename}"
           
            # Save locally
            with open(local_error_path, 'w') as f:
                json.dump(error_log, f, indent=4)
           
            # Upload to S3
            self.s3_manager.upload_file(local_error_path, s3_error_key)
            logger.info(f"Error log saved to S3: s3://{self.config.model_bucket}/{s3_error_key}")
           
        except Exception as e:
            logger.error(f"Failed to save error log: {str(e)}")
   
    def _load_training_datasets(self):
        """Load training datasets - matches original workflow"""
        datasets = {}
       
        # Look for training files in processed data
        train_dir = os.path.join(self.paths['input_path'], 'train_test_split', 'train')
       
        # Fallback to input_path if train_test_split doesn't exist
        if not os.path.exists(train_dir):
            logger.warning(f"Train directory not found: {train_dir}")
            train_dir = self.paths['input_path']
            logger.info(f"Using fallback directory: {train_dir}")
       
        # List all files in the directory for debugging
        if os.path.exists(train_dir):
            logger.info(f"Files in training directory {train_dir}:")
            for file in os.listdir(train_dir):
                logger.info(f"  📄 {file}")
       
        profiles = self.config.get_profiles()
       
        for profile in profiles:
            train_file = os.path.join(train_dir, f"{profile}_train_{self.current_date}.csv")
           
            logger.info(f"Looking for training file: {train_file}")
           
            if os.path.exists(train_file):
                try:
                    df = pd.read_csv(train_file, parse_dates=['Time'])
                    datasets[profile] = df
                    logger.info(f"✅ Loaded training data for {profile}: {len(df)} records")
                except Exception as e:
                    logger.error(f"❌ Failed to load {train_file}: {str(e)}")
            else:
                logger.warning(f"⚠️  Training file not found: {train_file}")
       
        if not datasets:
            raise FileNotFoundError("No training datasets found after S3 download")

        logger.info(f"Successfully loaded {len(datasets)} training datasets")
        return datasets

    def _train_all_models(self, datasets):
        """Train models for all profiles - matches original workflow"""
        results = {}
       
        for profile, df in datasets.items():
            try:
                logger.info(f"Training model for profile: {profile}")
                start_time = time()
               
                # Prepare data exactly as in original
                X_train, X_test, y_train, y_test, train_cutoff = self._encoding_train_test_split(df)
               
                # Train model exactly as in original
                best_model, grid_search, best_params, training_time = self._train_model(X_train, y_train)
               
                # Evaluate model exactly as in original
                metrics, y_train_pred, y_test_pred = self._evaluate_model(
                    best_model, X_train, y_train, X_test, y_test
                )
               
                # Save model to S3 exactly as in original
                model_saved = self._save_model(best_model, profile, metrics)
               
                # Generate predictions DataFrame exactly as in original
                predictions_df = self._create_predictions_dataframe(
                    X_test, y_test, y_test_pred, profile
                )
               
                # Create visualizations exactly as in original
                plot_files = self._create_visualizations(
                    X_test, y_test, y_test_pred, profile, best_model
                )
               
                results[profile] = {
                    'metrics': metrics,
                    'best_params': best_params,
                    'training_time': training_time,
                    'model_saved': model_saved,
                    'predictions': predictions_df,
                    'plot_files': plot_files,
                    'feature_importance': self._get_feature_importance(best_model)
                }
               
                logger.info(f"Completed training for {profile} in {time() - start_time:.2f} seconds")
               
            except Exception as e:
                logger.error(f"Failed to train model for {profile}: {str(e)}")
                results[profile] = {'error': str(e)}
       
        return results
   
    def _encoding_train_test_split(self, df):
        """Data preparation with robust data cleaning - matches original encoding_train_test_split function"""
        logger.info("Preparing data for training...")
       
        # Create a copy and handle missing values exactly as in original
        df_copy = df.copy()
       
        # Log initial data state
        logger.info(f"Initial data shape: {df_copy.shape}")
        logger.info(f"Missing values before cleaning: {df_copy.isnull().sum().sum()}")
       
        # Check target variable before processing
        if 'Load' in df_copy.columns:
            target_col = 'Load'
        else:
            raise ValueError("Target column 'Load' not found")
       
        # Robust data cleaning for target variable
        logger.info(f"Target variable {target_col} statistics before cleaning:")
        logger.info(f"  NaN values: {df_copy[target_col].isnull().sum()}")
        logger.info(f"  Infinite values: {np.isinf(df_copy[target_col]).sum()}")
        logger.info(f"  Min value: {df_copy[target_col].min()}")
        logger.info(f"  Max value: {df_copy[target_col].max()}")
       
        # Remove rows with NaN or infinite values in target
        initial_rows = len(df_copy)
        df_copy = df_copy[df_copy[target_col].notna()]
        df_copy = df_copy[np.isfinite(df_copy[target_col])]
       
        # Remove extremely large values (potential outliers that could cause XGBoost issues)
        target_mean = df_copy[target_col].mean()
        target_std = df_copy[target_col].std()
       
        # Remove values more than 5 standard deviations from mean
        outlier_threshold = target_mean + 5 * target_std
        df_copy = df_copy[df_copy[target_col] <= outlier_threshold]
        df_copy = df_copy[df_copy[target_col] >= (target_mean - 5 * target_std)]
       
        logger.info(f"Rows removed due to target issues: {initial_rows - len(df_copy)}")
       
        # Clean other critical columns
        critical_columns = ['Load_I_lag_14_days', 'Load_lag_70_days', 'Load', 'Temperature']
        for col in critical_columns:
            if col in df_copy.columns:
                before_count = len(df_copy)
                df_copy = df_copy[df_copy[col].notna()]
                df_copy = df_copy[np.isfinite(df_copy[col])]
                after_count = len(df_copy)
                if before_count != after_count:
                    logger.info(f"Removed {before_count - after_count} rows due to {col} issues")
       
        if df_copy.empty:
            raise ValueError("No valid data after removing missing/infinite values")
       
        logger.info(f"Final data shape after cleaning: {df_copy.shape}")
       
        # Use Count_I for Count exactly as in original
        df_copy['Count'] = df_copy['Count_I']
       
        # Remove unnecessary columns exactly as in original
        columns_to_drop = ['Time', 'Profile', 'Load_I', 'Count_I', 'TradeDate']
        for col in columns_to_drop:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])
       
        # Encode categorical variables exactly as in original
        if 'Weekday' in df_copy.columns:
            weekday_map = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }
            df_copy['Weekday'] = df_copy['Weekday'].map(weekday_map)
            # Handle any unmapped weekday values
            df_copy = df_copy[df_copy['Weekday'].notna()]
       
        if 'Season' in df_copy.columns:
            season_map = {'Summer': 1, 'Winter': 0}
            df_copy['Season'] = df_copy['Season'].map(season_map)
            # Handle any unmapped season values
            df_copy = df_copy[df_copy['Season'].notna()]
       
        # Final check for any remaining NaN or infinite values
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = df_copy[col].isnull().sum()
            inf_count = np.isinf(df_copy[col]).sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Column {col} still has {nan_count} NaN and {inf_count} infinite values")
                # Fill with median for this column
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], df_copy[col].median())
       
        # Split features and target exactly as in original
        X = df_copy.drop(columns=[target_col])
        y = df_copy[target_col]
       
        # Final validation
        if X.empty or y.empty:
            raise ValueError("Features or target is empty after preprocessing")
       
        if y.isnull().any() or np.isinf(y).any():
            raise ValueError("Target variable still contains NaN or infinite values")
       
        # Train-test split by date exactly as in original
        train_cutoff = pd.to_datetime(self.training_config['train_cutoff'])
        train_mask = df['Time'] < train_cutoff
       
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
       
        logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target range: {y.min():.4f} to {y.max():.4f}")
       
        return X_train, X_test, y_train, y_test, train_cutoff
   
    def _train_model(self, X_train, y_train):
        """Train XGBoost model with enhanced error handling"""
        logger.info("Training XGBoost model with GridSearchCV...")
        start_time = time()
       
        # Final validation of training data
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
       
        if y_train.isnull().any():
            raise ValueError("Training target contains NaN values")
       
        if np.isinf(y_train).any():
            raise ValueError("Training target contains infinite values")
       
        logger.info(f"Training data validation passed: {len(X_train)} samples, {len(X_train.columns)} features")
        logger.info(f"Target range: {y_train.min():.4f} to {y_train.max():.4f}")
       
        # XGBoost parameters from JSON configuration
        param_grid = self.training_config['xgboost_params']
       
        # Create XGBoost regressor with additional stability parameters
        xgb_regressor = xgb.XGBRegressor(
            random_state=self.training_config.get('random_state', 42),
            n_jobs=-1,
            tree_method='hist',  # More stable for edge cases
            objective='reg:squarederror',
            eval_metric='rmse'
        )
       
        # Time series cross-validation exactly as in original
        tscv = TimeSeriesSplit(n_splits=self.training_config['cv_splits'])
       
        # Grid search with error handling
        try:
            grid_search = GridSearchCV(
                estimator=xgb_regressor,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                error_score='raise'  # This will help identify specific issues
            )
           
            # Fit model
            grid_search.fit(X_train, y_train)
           
        except Exception as e:
            logger.error(f"GridSearchCV failed: {str(e)}")
            # Try with simpler parameters if grid search fails
            logger.info("Attempting training with simpler parameters...")
           
            # Get simplified parameters from JSON config or use defaults
            simple_params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6
            }

            simple_xgb = xgb.XGBRegressor(
                n_estimators=simple_params.get("n_estimators", 100),
                learning_rate=simple_params.get("learning_rate", 0.1),
                max_depth=simple_params.get("max_depth", 6),
                random_state=self.training_config.get('random_state', 42),
                tree_method='hist',
                objective='reg:squarederror',
                eval_metric='rmse'
            )
           
            simple_xgb.fit(X_train, y_train)
           
            # Create a mock grid_search object for consistency
            class MockGridSearch:
                def __init__(self, estimator):
                    self.best_estimator_ = estimator
                    self.best_params_ = simple_params
           
            grid_search = MockGridSearch(simple_xgb)
       
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        training_time = time() - start_time
       
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        logger.info(f"Best parameters: {best_params}")
       
        return best_model, grid_search, best_params, training_time
   
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model - matches original evaluate_model function"""
        # Generate predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
       
        # Calculate metrics exactly as in original
        metrics = {
            'RMSE_Train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAPE_Train': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'MAPE_Test': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'MSE_Train': mean_squared_error(y_train, y_train_pred),
            'MSE_Test': mean_squared_error(y_test, y_test_pred),
            'MAE_Train': mean_absolute_error(y_train, y_train_pred),
            'MAE_Test': mean_absolute_error(y_test, y_test_pred),
            'R²_Train': r2_score(y_train, y_train_pred),
            'R²_Test': r2_score(y_test, y_test_pred)
        }
       
        # Log metrics exactly as in original
        logger.info("Model evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            if "MAPE" in metric_name:
                logger.info(f"  {metric_name}: {metric_value:.2f}%")
            else:
                logger.info(f"  {metric_name}: {metric_value:.4f}")
       
        return metrics, y_train_pred, y_test_pred
   
    def _save_model(self, model, profile, metrics):
        """Save model to S3 - fixed version with proper error handling"""
        performance_threshold = self.training_config.get('performance_threshold')
       
        if performance_threshold is None or metrics['R²_Test'] >= performance_threshold:
            try:
                # Create model filename exactly as in original
                model_filename = self.config.get_file_path('xgboost_model', profile=profile, date=self.current_date)
               
                # Save model locally first in output directory
                output_model_path = os.path.join(self.paths['output_path'], model_filename)
                os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
                joblib.dump(model, output_model_path)
                logger.info(f"Model saved locally: {output_model_path}")
               
                # Upload to S3 model bucket
                s3_key = f"{self.config.config['s3']['model_prefix']}{model_filename}"
                success = self.s3_manager.upload_file(output_model_path, s3_key, self.config.model_bucket)
               
                if success:
                    logger.info(f"Model uploaded to S3: s3://{self.config.model_bucket}/{s3_key}")
                    return True
                else:
                    logger.error(f"Failed to upload model for {profile} to S3")
                    return False
                   
            except Exception as e:
                logger.error(f"Error saving model for {profile}: {str(e)}")
                return False
        else:
            logger.info(f"Model for {profile} did not meet performance threshold. Not saving.")
            return False
   
    def _create_predictions_dataframe(self, X_test, y_test, y_test_pred, profile):
        """Create predictions DataFrame exactly as in original"""
        predictions_df = pd.DataFrame()
       
        # Add time components if available
        if 'Year' in X_test.columns:
            predictions_df['Year'] = X_test['Year'].values
        if 'Month' in X_test.columns:
            predictions_df['Month'] = X_test['Month'].values
        if 'Day' in X_test.columns:
            predictions_df['Day'] = X_test['Day'].values
        if 'Hour' in X_test.columns:
            predictions_df['Hour'] = X_test['Hour'].values
       
        predictions_df['True_Load'] = y_test.values
        predictions_df['Predicted_Load'] = y_test_pred
        predictions_df['Profile'] = profile
       
        return predictions_df
   
    def _create_visualizations(self, X_test, y_test, y_test_pred, profile, model):
        """Create visualizations exactly as in original"""
        plot_files = []
       
        try:
            # Create actual vs predicted plot exactly as in original
            fig = go.Figure()
           
            fig.add_trace(go.Scatter(
                x=list(range(len(y_test))),
                y=y_test.values,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
           
            fig.add_trace(go.Scatter(
                x=list(range(len(y_test_pred))),
                y=y_test_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
           
            fig.update_layout(
                title=f'{profile} - Actual vs Predicted Load',
                xaxis_title='Time Index',
                yaxis_title='Load',
                template='plotly_white'
            )
           
            # Save plot locally and upload to S3
            plot_filename = f"{profile}_actual_vs_predicted_{self.current_date}.html"
            local_plot_path = os.path.join(self.paths['output_path'], plot_filename)
            fig.write_html(local_plot_path)
           
            # Upload to train_results directory exactly as in original
            s3_key = f"{self.config.config['s3']['train_results_prefix']}{plot_filename}"
            self.s3_manager.upload_file(local_plot_path, s3_key)
           
            plot_files.append(plot_filename)
           
            # Create feature importance plot if possible
            if hasattr(model, 'feature_importances_'):
                importance_plot = self._plot_feature_importance(model, X_test.columns, profile)
                if importance_plot:
                    plot_files.append(importance_plot)
           
        except Exception as e:
            logger.error(f"Failed to create visualizations for {profile}: {str(e)}")
       
        return plot_files
   
    def _plot_feature_importance(self, model, feature_names, profile):
        """Create feature importance plot exactly as in original"""
        try:
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
           
            fig = go.Figure(go.Bar(
                x=feature_importance_df['importance'],
                y=feature_importance_df['feature'],
                orientation='h'
            ))
           
            fig.update_layout(
                title=f'{profile} - Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features',
                template='plotly_white'
            )
           
            # Save plot locally and upload to S3
            plot_filename = f"{profile}_feature_importance_{self.current_date}.html"
            local_plot_path = os.path.join(self.paths['output_path'], plot_filename)
            fig.write_html(local_plot_path)
           
            # Upload to train_results directory
            s3_key = f"{self.config.config['s3']['train_results_prefix']}{plot_filename}"
            self.s3_manager.upload_file(local_plot_path, s3_key)
           
            return plot_filename
           
        except Exception as e:
            logger.error(f"Failed to create feature importance plot for {profile}: {str(e)}")
            return None
   
    def _get_feature_importance(self, model):
        """Get feature importance exactly as in original"""
        try:
            if hasattr(model, 'get_booster'):
                importance = model.get_booster().get_score(importance_type='weight')
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_importance)
            elif hasattr(model, 'feature_importances_'):
                return dict(zip(model.feature_names_in_, model.feature_importances_))
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return {}
   
    def _save_training_results(self, results):
        """Save training results exactly as in original"""
        for profile, result in results.items():
            if 'error' in result:
                continue
           
            try:
                # Save predictions exactly as in original
                predictions_filename = self.config.get_file_path('profile_predictions', profile=profile, date=self.current_date)
                predictions_local = os.path.join(self.paths['output_path'], predictions_filename)
                predictions_s3_key = f"{self.config.config['s3']['train_results_prefix']}{predictions_filename}"
               
                self.s3_manager.save_and_upload_dataframe(
                    result['predictions'], predictions_local, predictions_s3_key
                )
               
                # Save metrics and training details exactly as in original
                metrics_data = {
                    'metrics': result['metrics'],
                    'best_params': result['best_params'],
                    'training_time': result['training_time'],
                    'model_saved': result['model_saved'],
                    'feature_importance': result['feature_importance'],
                    'timestamp': datetime.now(self.pacific_tz).isoformat()
                }
               
                metrics_filename = f"{profile}_metrics_{self.current_date}.json"
                metrics_local = os.path.join(self.paths['output_path'], metrics_filename)
                metrics_s3_key = f"{self.config.config['s3']['train_results_prefix']}{metrics_filename}"
               
                self.s3_manager.save_and_upload_file(metrics_data, metrics_local, metrics_s3_key)
               
                logger.info(f"Saved training results for {profile}")
               
            except Exception as e:
                logger.error(f"Failed to save results for {profile}: {str(e)}")
   
    def _generate_training_summary(self, results, start_time):
        """Generate training summary exactly as in original"""
        from config import ENVIRONMENT, DATA_BUCKET, MODEL_BUCKET
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
       
        summary = {
            'timestamp': end_time.isoformat(),
            'training_date': self.current_date,
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'total_profiles': len(results),
            'successful_profiles': len([r for r in results.values() if 'error' not in r]),
            'failed_profiles': len([r for r in results.values() if 'error' in r]),
            'training_parameters': self.training_config,
            'profile_results': {},
            'environment': ENVIRONMENT,
            'configuration_used': {
                'data_bucket': DATA_BUCKET,
                'model_bucket': MODEL_BUCKET,
                'xgboost_params': self.training_config['xgboost_params'],
                'cv_splits': self.training_config['cv_splits'],
                'train_cutoff': self.training_config['train_cutoff']
            }
        }
       
        # Add per-profile summary exactly as in original
        for profile, result in results.items():
            if 'error' in result:
                summary['profile_results'][profile] = {
                    'status': 'failed',
                    'error': result['error']
                }
            else:
                summary['profile_results'][profile] = {
                    'status': 'success',
                    'test_rmse': result['metrics']['RMSE_Test'],
                    'test_mape': result['metrics']['MAPE_Test'],
                    'test_r2': result['metrics']['R²_Test'],
                    'training_time_minutes': result['training_time'] / 60,
                    'model_saved': result['model_saved'],
                    'best_params': result['best_params']
                }
       
        # Save summary locally and upload to S3 exactly as in original
        summary_filename = f"training_summary_{self.current_date}.json"
        summary_local = os.path.join(self.paths['output_path'], summary_filename)
        summary_s3_key = f"{self.config.config['s3']['train_results_prefix']}{summary_filename}"
       
        self.s3_manager.save_and_upload_file(summary, summary_local, summary_s3_key)
       
        # Print summary to logs exactly as in original
        logger.info("="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total profiles processed: {summary['total_profiles']}")
        logger.info(f"Successful: {summary['successful_profiles']}")
        logger.info(f"Failed: {summary['failed_profiles']}")
        logger.info(f"Total training time: {summary['total_time_minutes']:.2f} minutes")
       
        for profile, profile_result in summary['profile_results'].items():
            if profile_result['status'] == 'success':
                logger.info(f"{profile}: RMSE={profile_result['test_rmse']:.4f}, "
                           f"MAPE={profile_result['test_mape']:.2f}%, "
                           f"R²={profile_result['test_r2']:.4f}")
            else:
                logger.info(f"{profile}: FAILED - {profile_result['error']}")
       
        return summary
   
    def _save_error_log(self, error_message):
        """Save error log with environment information"""
        from config import ENVIRONMENT, DATA_BUCKET, MODEL_BUCKET
       
        error_log = {
            'timestamp': datetime.now(self.pacific_tz).isoformat(),
            'current_date': self.current_date,
            'error': error_message,
            'status': 'failed',
            'xgboost_version': xgb.__version__,
            'environment': ENVIRONMENT,
            'configuration_used': {
                'data_bucket': DATA_BUCKET,
                'model_bucket': MODEL_BUCKET,
                'training_config': self.training_config
            }
        }
       
        # Save locally and upload to S3
        local_file = os.path.join(self.paths['output_path'], 'error_log.json')
        s3_key = f"{self.config.config['s3']['train_results_prefix']}error_log_{self.current_date}.json"
       
        self.s3_manager.save_and_upload_file(error_log, local_file, s3_key)

def main():
    """Main entry point for training container"""
    try:
        # Log environment information
        from config import ENVIRONMENT, DEBUG_MODE, DATA_BUCKET, MODEL_BUCKET
        logger.info("="*60)
        logger.info("ENERGY FORECASTING TRAINING PIPELINE")
        logger.info("="*60)
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Debug Mode: {DEBUG_MODE}")
        logger.info(f"Data Bucket: {DATA_BUCKET}")
        logger.info(f"Model Bucket: {MODEL_BUCKET}")
        logger.info(f"XGBoost version: {xgb.__version__}")
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
        logger.info("="*60)
       
        pipeline = EnergyTrainingPipeline()
        pipeline.run_training()
       
        logger.info("Training pipeline completed successfully!")
        sys.exit(0)
       
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.error("Configuration details:")
        try:
            from config import ENV_CONFIG
            logger.error(f"Loaded config keys: {list(ENV_CONFIG.keys())}")
            logger.error(f"Environment: {ENV_CONFIG.get('ENVIRONMENT', 'unknown')}")
            logger.error(f"Training config: {ENV_CONFIG.get('TRAINING_CONFIG', {})}")
        except Exception as config_error:
            logger.error(f"Could not load config details: {config_error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
