#!/usr/bin/env python3
"""
Refactored Preprocessing Container for SageMaker
Enhanced with Redshift integration and improved data processing
"""

import os
import gc
import psutil
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append('/opt/ml/processing/code/src')

from config import EnergyForecastingConfig, S3FileManager, RedshiftDataManager, MemoryOptimizedRedshiftDataManager, MemoryOptimizedEnergyForecastingConfig


class EnergyPreprocessingPipeline:
    def __init__(self):
        # Initialize configuration
        self.config = EnergyForecastingConfig()
        self.s3_manager = S3FileManager(self.config)
        self.redshift_manager = RedshiftDataManager(self.config)
        self.paths = self.config.get_container_paths()
       
        # Pacific timezone
        self.pacific_tz = pytz.timezone("America/Los_Angeles")
        self.current_date = self.config.current_date_str
       
        logger.info(f"Preprocessing pipeline initialized for date: {self.current_date}")
        logger.info(f"Redshift enabled: {self.config.is_redshift_enabled()}")
   
    def run_preprocessing(self):
        """Main preprocessing pipeline with Redshift integration"""
        try:
            logger.info("Starting preprocessing pipeline...")
            start_time = datetime.now()
           
            # Step 1: Process Load Data (enhanced with Redshift support)
            logger.info("Step 1: Processing Load Data...")
            df_load = self._process_load_data()
           
            # Step 2: Process Temperature Data
            logger.info("Step 2: Processing Temperature Data...")
            df_temperature = self._process_temperature_data()
           
            # Step 3: Merge Load and Temperature
            logger.info("Step 3: Merging Load and Temperature Data...")
            df_merged = self._merge_load_temperature(df_load, df_temperature)
           
            # Step 4: Generate Data Profiles
            logger.info("Step 4: Generating Data by Profile...")
            profile_dfs = self._generate_data_profile(df_merged)
           
            # Step 5: Create Lag Features
            logger.info("Step 5: Creating Lag Features...")
            lagged_dfs = self._save_lagged_profiles(profile_dfs)
           
            # Step 6: Replace Count Data
            logger.info("Step 6: Replacing Count Data...")
            replaced_dfs = self._replace_count_i(lagged_dfs)
           
            # Step 7: Add Radiation for RN
            logger.info("Step 7: Adding Radiation Data for RN...")
            final_dfs = self._add_radiation_to_df_RN(replaced_dfs)
           
            # Step 8: Train-Test Split
            logger.info("Step 8: Performing Train-Test Split...")
            self._train_test_split(final_dfs)
           
            # Step 9: Generate summary
            self._save_processing_summary(start_time)
           
            logger.info("Preprocessing pipeline completed successfully!")
           
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            self._save_error_log(str(e))
            raise
   
    def _process_load_data(self):
        """Enhanced load data processing with Redshift integration and CSV fallback"""
        start_time = time.time()
        
        try:
            if self.config.is_redshift_enabled():
                logger.info("Using Redshift data source...")
                df = self._process_load_data_from_redshift()
            else:
                logger.info("Using CSV data source...")
                df = self._process_load_data_from_csv()
                
            # Save processed data to S3
            s3_key = f"{self.config.config['s3']['processed_data_prefix']}hourly_data/Hourly_Load_Data_{self.current_date}.csv"
            self.s3_manager.upload_dataframe(df, s3_key)
            
            end_time = time.time()
            logger.info(f"Load data processing completed in {(end_time - start_time)/60:.2f} minutes")
            return df
            
        except Exception as e:
            logger.error(f"Load data processing failed: {str(e)}")
            # Try fallback if Redshift was enabled
            if self.config.is_redshift_enabled():
                logger.warning("Redshift failed, attempting CSV fallback...")
                try:
                    df = self._process_load_data_from_csv()
                    logger.info("Successfully processed data using CSV fallback")
                    return df
                except Exception as csv_error:
                    logger.error(f"CSV fallback also failed: {str(csv_error)}")
            raise
    
    def _process_load_data_from_redshift(self):
        """Process load data from Redshift with enhanced data handling"""
        logger.info("Reading SQMD data from Redshift...")
        
        # Query data from Redshift
        df = self.redshift_manager.query_sqmd_data()
        
        if df.empty:
            raise ValueError("No SQMD data retrieved from Redshift")
        
        logger.info(f"Successfully read {len(df)} rows of SQMD data from Redshift")
        
        # Process datetime columns with robust handling
        df = self._process_datetime_columns(df)
        
        # Create profile and NEM classifications
        df['RateGroup'] = df['rategroup'].astype(str)
        df['NEM'] = df['RateGroup'].apply(lambda x: 'NEM' if x.startswith(('NEM', 'SBP')) else 'Non_NEM')
        df['Profile'] = df.apply(lambda row: row['loadprofile'] + '_' + row['NEM'] if row['loadprofile'] == 'RES' else row['loadprofile'], axis=1)
        
        # Select and rename columns
        df = df[['TradeDateTime', 'tradedate', 'tradetime', 'Profile', 'lossadjustedload', 'metercount', 'submission']].copy()
        df.columns = ['TradeDateTime', 'TradeDate', 'TradeTime', 'Profile', 'LossAdjustedLoad', 'MeterCount', 'Submission']
        
        # Enhanced numeric data cleaning
        df = self._clean_numeric_data(df)
        
        # Process Final and Initial submissions
        df_final, df_initial = self._separate_submissions(df)
        
        # Aggregate hourly data
        df_hour_final = self._aggregate_hourly_data(df_final, 'final')
        df_hour_initial = self._aggregate_hourly_data(df_initial, 'initial')
        
        # Calculate load per meter
        df_hour_final['Load_Per_Meter'] = self._safe_division(df_hour_final['LoadHour'], df_hour_final['Count'])
        df_hour_initial['Load_Per_Meter'] = self._safe_division(df_hour_initial['LoadHour'], df_hour_initial['Count'])
        
        # Rename initial columns
        df_hour_initial = df_hour_initial.rename(columns={
            'LoadHour': 'LoadHour_I',
            'Count': 'Count_I',
            'Load_Per_Meter': 'Load_Per_Meter_I'
        })
        
        # Merge final and initial data
        df_merged = pd.merge(df_hour_final, df_hour_initial, on=['TradeDateTime', 'Profile'], how='right')
        df_processed = df_merged[['TradeDateTime', 'Profile', 'Count', 'Load_Per_Meter', 'Count_I', 'Load_Per_Meter_I']].copy()
        
        # Extend dataset and add features
        df_extended = self._extend_dataset(df_processed)
        df_final = self._add_date_features(df_extended)
        
        logger.info(f"Processed Redshift data: {len(df_final)} records")
        return df_final
    
    def _process_load_data_from_csv(self):
        """Process load data from CSV file (original implementation)"""
        input_file = os.path.join(self.paths['input_path'], self.config.get_file_path('load_data'))
       
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Load data file not found: {input_file}")
       
        logger.info(f"Loading load data from: {input_file}")
        df = pd.read_csv(input_file)
       
        # Process data exactly as in original
        df['TradeDateTime'] = pd.to_datetime(df['TradeDate'] + ' ' + df['TradeTime'], format='%Y-%m-%d %H:%M:%S')
        df['RateGroup'] = df['RateGroup'].astype(str)
        df['NEM'] = df['RateGroup'].apply(lambda x: 'NEM' if x.startswith(('NEM', 'SBP')) else 'Non_NEM')
        df['Profile'] = df.apply(lambda row: row['LoadProfile'] + '_' + row['NEM'] if row['LoadProfile'] == 'RES' else row['LoadProfile'], axis=1)
        df = df[['TradeDateTime', 'TradeDate', 'TradeTime', 'Profile', 'LossAdjustedLoad', 'MeterCount', 'Submission']].copy()
       
        # Filter submissions
        df_final = df[df['Submission'] == 'Final']
        df_initial = df[df['Submission'] == 'Initial']
       
        # Group by hourly - Final
        df_hour_final = df_final.groupby(['TradeDateTime', 'Profile']).agg(
            LoadHour=('LossAdjustedLoad', 'sum'),
            Count=('MeterCount', 'sum')
        ).reset_index()
       
        # Group by hourly - Initial
        df_hour_initial = df_initial.groupby(['TradeDateTime', 'Profile']).agg(
            LoadHour=('LossAdjustedLoad', 'sum'),
            Count=('MeterCount', 'sum')
        ).reset_index()
       
        # Calculate Load_Per_Meter
        df_hour_final['Load_Per_Meter'] = df_hour_final['LoadHour'] / df_hour_final['Count']
        df_hour_initial['Load_Per_Meter'] = df_hour_initial['LoadHour'] / df_hour_initial['Count']
       
        # Rename initial columns
        df_hour_initial.rename(columns={
            'LoadHour': 'LoadHour_I',
            'Count': 'Count_I',
            'Load_Per_Meter': 'Load_Per_Meter_I'
        }, inplace=True)
       
        # Merge
        df_merged = pd.merge(df_hour_final, df_hour_initial, on=['TradeDateTime', 'Profile'], how='right')
        df = df_merged[['TradeDateTime', 'Profile', 'Count', 'Load_Per_Meter', 'Count_I', 'Load_Per_Meter_I']].copy()
       
        # Extend dataset and add features
        df_extended = self._extend_dataset(df)
        df_final = self._add_date_features(df_extended)
       
        logger.info(f"Processed CSV data: {len(df_final)} records")
        return df_final
    
    def _process_datetime_columns(self, df):
        """Robust datetime processing for Redshift data"""
        try:
            # Try the expected format first
            df['TradeDateTime'] = pd.to_datetime(df['tradedate'] + ' ' + df['tradetime'], format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            logger.info("Standard time format failed, trying to handle Redshift time format...")
            logger.info(f"Sample tradedate: {df['tradedate'].iloc[0] if not df.empty else 'N/A'}")
            logger.info(f"Sample tradetime: {df['tradetime'].iloc[0] if not df.empty else 'N/A'}")
           
            # Handle case where tradetime might be just hours
            def create_datetime(row):
                date_str = str(row['tradedate'])
                time_str = str(row['tradetime'])
               
                # If time is just hours, add minutes and seconds
                if len(time_str) <= 2:
                    time_str = f"{time_str.zfill(2)}:00:00"
                elif ':' not in time_str:
                    time_str = f"{time_str.zfill(2)}:00:00"
               
                datetime_str = f"{date_str} {time_str}"
                return pd.to_datetime(datetime_str, errors='coerce')
           
            df['TradeDateTime'] = df.apply(create_datetime, axis=1)
           
            # Check for any NaT values
            nat_count = df['TradeDateTime'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Warning: {nat_count} rows have invalid datetime values")
                df = df.dropna(subset=['TradeDateTime'])
                logger.info(f"Removed {nat_count} rows with invalid datetime, remaining: {len(df)} rows")
        
        return df
    
    def _clean_numeric_data(self, df):
        """Enhanced numeric data cleaning"""
        logger.info("Converting numeric columns to proper types...")
        logger.info(f"LossAdjustedLoad dtype before: {df['lossadjustedload'].dtype}")
        logger.info(f"MeterCount dtype before: {df['metercount'].dtype}")
       
        # Convert to numeric, handling any non-numeric values
        df['lossadjustedload'] = pd.to_numeric(df['lossadjustedload'], errors='coerce')
        df['metercount'] = pd.to_numeric(df['metercount'], errors='coerce')
       
        # Remove rows with invalid numeric values
        invalid_load = df['lossadjustedload'].isna().sum()
        invalid_meter = df['metercount'].isna().sum()
       
        if invalid_load > 0 or invalid_meter > 0:
            logger.warning(f"Found {invalid_load} invalid LossAdjustedLoad values and {invalid_meter} invalid MeterCount values")
            df = df.dropna(subset=['lossadjustedload', 'metercount'])
            logger.info(f"Removed invalid rows, remaining: {len(df)} rows")
       
        logger.info(f"LossAdjustedLoad dtype after: {df['lossadjustedload'].dtype}")
        logger.info(f"MeterCount dtype after: {df['metercount'].dtype}")
        
        return df
    
    def _separate_submissions(self, df):
        """Separate Final and Initial submissions"""
        df_final = df[df['Submission'] == 'Final']
        df_initial = df[df['Submission'] == 'Initial']
        
        logger.info(f"Final submissions: {len(df_final)} rows")
        logger.info(f"Initial submissions: {len(df_initial)} rows")
        
        return df_final, df_initial
    
    def _aggregate_hourly_data(self, df, submission_type):
        """Aggregate data by hour with enhanced error handling"""
        if df.empty:
            logger.warning(f"No data to aggregate for {submission_type} submissions")
            return pd.DataFrame(columns=['TradeDateTime', 'Profile', 'LoadHour', 'Count'])
        
        # Ensure numeric columns are properly typed before aggregation
        df['LossAdjustedLoad'] = pd.to_numeric(df['LossAdjustedLoad'], errors='coerce')
        df['MeterCount'] = pd.to_numeric(df['MeterCount'], errors='coerce')
        
        # Group by hourly
        df_hour = df.groupby(['TradeDateTime', 'Profile']).agg(
            LoadHour=('LossAdjustedLoad', 'sum'),
            Count=('MeterCount', 'sum')
        ).reset_index()
        
        # Ensure aggregated columns are numeric
        df_hour['LoadHour'] = pd.to_numeric(df_hour['LoadHour'], errors='coerce')
        df_hour['Count'] = pd.to_numeric(df_hour['Count'], errors='coerce')
        
        logger.info(f"Aggregated {submission_type} data: {len(df_hour)} hourly records")
        return df_hour
    
    def _safe_division(self, numerator, denominator):
        """Safe division with error handling"""
        try:
            result = numerator.astype(float) / denominator.astype(float)
            # Handle division by zero or infinity
            result = result.replace([np.inf, -np.inf], np.nan)
            return result
        except Exception as e:
            logger.error(f"Error during division operations: {e}")
            # Force conversion and try again
            result = pd.to_numeric(numerator, errors='coerce') / pd.to_numeric(denominator, errors='coerce')
            result = result.replace([np.inf, -np.inf], np.nan)
            return result
    
    def _extend_dataset(self, df):
        """Extend dataset with future dates (40 days)"""
        max_date = df['TradeDateTime'].max()
        extended_dates = pd.date_range(start=max_date + pd.Timedelta(hours=1), periods=40 * 24, freq='h')
        profiles = df['Profile'].unique()
        extended_df = pd.DataFrame({'TradeDateTime': extended_dates}).merge(
            pd.DataFrame(profiles, columns=['Profile']), how='cross'
        )
        
        df_extended = pd.concat([df, extended_df], ignore_index=True)
        logger.info(f"Extended dataset: {len(df_extended)} records (added {len(extended_df)} future records)")
        
        return df_extended
    
    def _add_date_features(self, df):
        """Add date-related features"""
        df['Year'] = df['TradeDateTime'].dt.year
        df['Month'] = df['TradeDateTime'].dt.month
        df['Day'] = df['TradeDateTime'].dt.day
        df['Hour'] = df['TradeDateTime'].dt.hour
        df['Weekday'] = df['TradeDateTime'].dt.day_name()
        df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [1, 2, 3, 4, 5, 11, 12] else 'Summer')
        
        # Add holidays
        holidays = self.config.get_data_processing_config()['holidays']
        df['TradeDate'] = df['TradeDateTime'].dt.date.astype(str)
        df['Holiday'] = df['TradeDate'].isin(holidays).astype(int)
        df['Workday'] = df.apply(lambda x: 0 if (x['Holiday'] == 1 or x['Weekday'] in ['Saturday', 'Sunday']) else 1, axis=1)
        
        logger.info("Added date features and holiday information")
        return df
   
    def _process_temperature_data(self):
        """Process temperature data"""
        input_file = os.path.join(self.paths['input_path'], self.config.get_file_path('temperature_data'))
       
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Temperature data file not found: {input_file}")
       
        logger.info(f"Loading temperature data from: {input_file}")
        df_t = pd.read_csv(input_file)
        df_t = df_t[['DATE', 'HourlyDryBulbTemperature']]
        df_t = df_t.rename(columns={'HourlyDryBulbTemperature': 'Temperature'})
       
        df_t['DATE'] = pd.to_datetime(df_t['DATE'])
        df_t['TradeDate'] = df_t['DATE'].dt.date.astype(str)
        df_t['Hour'] = df_t['DATE'].dt.hour
       
        # Clean temperature data
        df_t['Temperature'] = df_t['Temperature'].astype(str).str.replace('s', '', regex=False)
        df_t['Temperature'] = pd.to_numeric(df_t['Temperature'], errors='coerce')
       
        # Group by TradeDate and Hour
        df_t = df_t.groupby(['TradeDate', 'Hour'])['Temperature'].mean().reset_index()
       
        # Create full range
        full_range = pd.MultiIndex.from_product([df_t['TradeDate'].unique(), range(24)], names=['TradeDate', 'Hour'])
        df_t = df_t.set_index(['TradeDate', 'Hour']).reindex(full_range).reset_index()
       
        # Interpolate missing values
        df_t['Temperature_Fill'] = df_t['Temperature'].interpolate(method='linear')
        df_t['TradeDateTime'] = pd.to_datetime(df_t['TradeDate'] + ' ' + df_t['Hour'].astype(str).str.zfill(2) + ':00:00')
       
        logger.info(f"Processed temperature data: {len(df_t)} records")
        return df_t
   
    def _merge_load_temperature(self, df_load, df_temperature):
        """Merge load and temperature data"""
        # Merge data
        df = pd.merge(df_load, df_temperature, on=['TradeDate', 'Hour'], how='left')
       
        # Select and rename columns
        df = df[['TradeDateTime_y', 'Profile', 'Load_Per_Meter', 'Count', 'Load_Per_Meter_I',
                 'Count_I', 'Year', 'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday',
                 'Workday', 'TradeDate', 'Temperature_Fill']]
       
        df = df.rename(columns={
            'TradeDateTime_y': 'Time',
            'Load_Per_Meter': 'Load',
            'Load_Per_Meter_I': 'Load_I',
            'Temperature_Fill': 'Temperature'
        })
       
        # Fill null Time values
        df['Time'] = pd.to_datetime(df['TradeDate'] + ' ' + df['Hour'].astype(str).str.zfill(2) + ':00:00')
       
        logger.info(f"Merged data: {len(df)} records")
        return df
   
    def _generate_data_profile(self, df):
        """Generate data profiles"""
        # Replace null values
        df['Count'] = df['Count'].fillna(df['Count_I'])
        df['Load'] = df['Load'].fillna(df['Load_I'])
       
        # Create profile DataFrames exactly as in original
        profile_mapping = self.config.get_data_processing_config()['profile_mappings']
       
        profile_dfs = {}
        for profile_name, profile_code in profile_mapping.items():
            profile_dfs[profile_code] = df[df['Profile'] == profile_name].copy()
            logger.info(f"Generated profile {profile_code}: {len(profile_dfs[profile_code])} records")
       
        return profile_dfs
   
    def _save_lagged_profiles(self, profile_dfs):
        """Create lag features"""
        profile_start_dates = self.config.get_data_processing_config()['profile_start_dates']
        lag_config = self.config.get_data_processing_config()['lag_features']
       
        lagged_dfs = {}
       
        for profile, df in profile_dfs.items():
            logger.info(f"Creating lag features for {profile}")
           
            df_copy = df.copy()
           
            # Create lag features exactly as in original
            shift_hours_14 = lag_config['load_i_lag_days'] * 24
            df_copy['Load_I_lag_14_days'] = df_copy['Load_I'].shift(shift_hours_14)
           
            shift_hours_70 = lag_config['load_lag_days'] * 24
            df_copy['Load_lag_70_days'] = df_copy['Load'].shift(shift_hours_70)
           
            # Filter by start date
            start_date = pd.to_datetime(profile_start_dates[profile])
            filtered_df = df_copy[df_copy['Time'] >= start_date]
           
            # Save locally and upload to S3
            local_file = os.path.join(self.paths['output_path'], 'processed', f"{profile}_lagged_{self.current_date}.csv")
            s3_key = self.config.get_full_s3_key('processed_data', 'profile_lagged', profile=profile, date=self.current_date)
           
            self.s3_manager.save_and_upload_dataframe(filtered_df, local_file, s3_key)
           
            lagged_dfs[profile] = filtered_df
            logger.info(f"Saved lagged profile {profile}: {len(filtered_df)} records")
       
        return lagged_dfs
   
    def _replace_count_i(self, lagged_dfs):
        """Replace Count_I"""
        updated_dfs = {}
       
        for profile, df in lagged_dfs.items():
            logger.info(f"Replacing Count_I for {profile}")
           
            df_copy = df.copy()
           
            # Find last day with non-null Count_I exactly as in original
            last_non_null_date = df_copy[df_copy['Count_I'].notnull()]['Time'].dt.date.max()
           
            if pd.isna(last_non_null_date):
                logger.warning(f"No non-null Count_I values found in {profile}")
                updated_dfs[profile] = df_copy
                continue
           
            # Calculate mean for last day
            last_day_data = df_copy[df_copy['Time'].dt.date == last_non_null_date]
            count_mean = last_day_data['Count_I'].mean()
           
            # Replace Count_I
            df_copy['Count_I'] = df_copy['Count_I'].fillna(count_mean)
           
            updated_dfs[profile] = df_copy
            logger.info(f"Replaced Count_I for {profile} with mean: {count_mean}")
       
        return updated_dfs
   
    def _add_radiation_to_df_RN(self, updated_dfs):
        """Add radiation data to RN profile"""
        radiation_file = os.path.join(self.paths['input_path'], self.config.get_file_path('radiation_data'))
       
        if not os.path.exists(radiation_file):
            logger.warning(f"Radiation file not found: {radiation_file}")
            return updated_dfs
       
        logger.info(f"Loading radiation data from: {radiation_file}")
        df_meteo = pd.read_csv(radiation_file)
       
        # Process radiation data exactly as in original
        df_meteo.columns = df_meteo.columns.str.replace(r"\s*\(.*?\)", "", regex=True)
        df_meteo['time'] = pd.to_datetime(df_meteo['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_meteo['time'] = df_meteo['time'].str.replace("T", " ", regex=False)
        df_meteo['time'] = pd.to_datetime(df_meteo['time'])
        df_meteo = df_meteo[['time', 'shortwave_radiation']]
       
        # Merge with RN profile
        if 'df_RN' in updated_dfs:
            df_RN = updated_dfs['df_RN']
            updated_df_RN = df_RN.merge(df_meteo, left_on='Time', right_on='time', how='left')
            updated_df_RN.drop(columns=['time'], inplace=True)
           
            # Define output columns exactly as in original
            output_columns = [
                'Time', 'Profile', 'Load', 'Count', 'Load_I', 'Count_I', 'Year',
                'Month', 'Day', 'Hour', 'Weekday', 'Season', 'Holiday', 'Workday',
                'TradeDate', 'Temperature', 'Load_I_lag_14_days', 'Load_lag_70_days',
                'shortwave_radiation'
            ]
           
            updated_df_RN = updated_df_RN[output_columns]
            updated_dfs['df_RN'] = updated_df_RN
           
            logger.info(f"Added radiation data to df_RN: {len(updated_df_RN)} records")
       
        return updated_dfs
   
    def _train_test_split(self, final_dfs):
        """Perform train-test split"""
        split_date = pd.to_datetime(self.config.get_data_processing_config()['split_date'])
       
        for profile, df in final_dfs.items():
            logger.info(f"Splitting data for {profile}")
           
            # Split data
            train_set = df[df['Time'] < split_date]
            test_set = df[df['Time'] >= split_date]
           
            # Add suffix for RN profile exactly as in original
            suffix = "_r" if profile == "df_RN" else ""
           
            # Save train set to processed/train_test_split/train/
            train_local = os.path.join(
                self.paths['output_path'], 'processed', 'train_test_split', 'train',
                f"{profile}_train_{self.current_date}.csv"
            )
            train_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/train/{profile}_train_{self.current_date}.csv"
            self.s3_manager.save_and_upload_dataframe(train_set, train_local, train_s3_key)
           
            # Save test set to processed/train_test_split/test/
            test_local = os.path.join(
                self.paths['output_path'], 'processed', 'train_test_split', 'test',
                f"{profile}_test_{self.current_date}{suffix}.csv"
            )
            test_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/test/{profile}_test_{self.current_date}{suffix}.csv"
            self.s3_manager.save_and_upload_dataframe(test_set, test_local, test_s3_key)
           
            # Save test set to input directory for prediction container
            input_local = os.path.join(
                self.paths['output_path'], 'input',
                f"{profile}_test_{self.current_date}{suffix}.csv"
            )
            input_s3_key = f"{self.config.config['s3']['input_data_prefix']}{profile}_test_{self.current_date}{suffix}.csv"
            self.s3_manager.save_and_upload_dataframe(test_set, input_local, input_s3_key)
           
            logger.info(f"Split {profile}: {len(train_set)} train, {len(test_set)} test records")
   
    def _save_processing_summary(self, start_time):
        """Save processing summary with enhanced information"""
        from config import ENVIRONMENT, DATA_BUCKET, REDSHIFT_CLUSTER_IDENTIFIER
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
       
        summary = {
            "timestamp": end_time.isoformat(),
            "processing_time_seconds": processing_time,
            "processing_time_minutes": processing_time / 60,
            "split_date": self.config.get_data_processing_config()['split_date'],
            "current_date": self.current_date,
            "profiles_processed": self.config.get_profiles(),
            "data_source": "redshift" if self.config.is_redshift_enabled() else "csv",
            "redshift_enabled": self.config.is_redshift_enabled(),
            "data_reading_period_days": self.config.get_data_reading_period_days(),
            "status": "completed",
            "environment": ENVIRONMENT,
            "configuration_used": {
                "data_bucket": DATA_BUCKET,
                "model_bucket": self.config.model_bucket,
                "redshift_cluster": REDSHIFT_CLUSTER_IDENTIFIER,
                "lag_features": self.config.get_data_processing_config()['lag_features'],
                "redshift_config": self.config.get_redshift_config() if self.config.is_redshift_enabled() else None
            }
        }
       
        # Save locally and upload to S3
        local_file = os.path.join(self.paths['output_path'], "processing_summary.json")
        s3_key = f"{self.config.config['s3']['processed_data_prefix']}processing_summary_{self.current_date}.json"
       
        self.s3_manager.save_and_upload_file(summary, local_file, s3_key)
       
        logger.info(f"Processing completed in {processing_time/60:.2f} minutes")
   
    def _save_error_log(self, error_message):
        """Save error log with enhanced information"""
        from config import ENVIRONMENT, REDSHIFT_CLUSTER_IDENTIFIER
        
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "current_date": self.current_date,
            "error": error_message,
            "data_source": "redshift" if self.config.is_redshift_enabled() else "csv",
            "redshift_enabled": self.config.is_redshift_enabled(),
            "environment": ENVIRONMENT,
            "redshift_cluster": REDSHIFT_CLUSTER_IDENTIFIER,
            "status": "failed"
        }
       
        # Save locally and upload to S3
        local_file = os.path.join(self.paths['output_path'], "error_log.json")
        s3_key = f"{self.config.config['s3']['processed_data_prefix']}error_log_{self.current_date}.json"
       
        self.s3_manager.save_and_upload_file(error_log, local_file, s3_key)


class MemoryOptimizedEnergyPreprocessingPipeline(EnergyPreprocessingPipeline):
    """Memory-optimized preprocessing pipeline for large datasets"""
    
    def __init__(self):
        super().__init__()
        
        # Use memory-optimized configuration and managers
        if os.getenv('MEMORY_OPTIMIZATION') == '1':
            logger.info("Initializing memory-optimized pipeline...")
            self.config = MemoryOptimizedEnergyForecastingConfig()
            self.redshift_manager = MemoryOptimizedRedshiftDataManager(self.config)
            self.memory_optimization = True
        else:
            self.memory_optimization = False
        
        # Memory monitoring
        self._log_memory_status("Pipeline initialization")
    
    def _process_load_data_from_redshift(self):
        """Memory-optimized Redshift data processing"""
        logger.info("Using memory-optimized Redshift data processing...")
        
        try:
            # Check initial memory
            self._log_memory_status("Before data query")
            
            if self.memory_optimization:
                # Use chunked processing for large datasets
                df = self.redshift_manager.query_sqmd_data_chunked()
            else:
                # Use regular processing
                df = self.redshift_manager.query_sqmd_data()
            
            self._log_memory_status("After data query")
            
            if df.empty:
                raise ValueError("No SQMD data retrieved from Redshift")
            
            logger.info(f"Successfully processed {len(df):,} rows of SQMD data from Redshift")
 
            self._log_memory_status("After data processing")
            
            return df
            
        except Exception as e:
            logger.error(f"Memory-optimized Redshift processing failed: {str(e)}")
            # Clean up memory on error
            gc.collect()
            raise
    
    def _log_memory_status(self, stage):
        """Log current memory usage"""
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            available_gb = memory_info.available / (1024**3)
            used_gb = memory_info.used / (1024**3)
            total_gb = memory_info.total / (1024**3)
            
            logger.info(f"Memory Status - {stage}:")
            logger.info(f"  Used: {used_gb:.1f} GB / {total_gb:.1f} GB ({memory_percent:.1f}%)")
            logger.info(f"  Available: {available_gb:.1f} GB")
            
            # Warn if memory usage is high
            if memory_percent > 80:
                logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                
        except Exception as e:
            logger.warning(f"Could not get memory status: {str(e)}")
    
    def _train_test_split(self, final_dfs):
        """Memory-optimized train-test split"""
        split_date = pd.to_datetime(self.config.get_data_processing_config()['split_date'])
        
        for profile, df in final_dfs.items():
            logger.info(f"Splitting data for {profile}")
            self._log_memory_status(f"Before splitting {profile}")
            
            # Split data
            train_set = df[df['Time'] < split_date].copy()
            test_set = df[df['Time'] >= split_date].copy()
            
            # Clear original dataframe from memory immediately
            del df
            gc.collect()
            
            # Save and upload files
            suffix = "_r" if profile == "df_RN" else ""
            
            # Save train set
            train_local = os.path.join(
                self.paths['output_path'], 'processed', 'train_test_split', 'train',
                f"{profile}_train_{self.current_date}.csv"
            )
            train_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/train/{profile}_train_{self.current_date}.csv"
            self.s3_manager.save_and_upload_dataframe(train_set, train_local, train_s3_key)
            
            # Clear train set from memory
            del train_set
            gc.collect()
            
            # Save test set to processed directory
            test_local = os.path.join(
                self.paths['output_path'], 'processed', 'train_test_split', 'test',
                f"{profile}_test_{self.current_date}{suffix}.csv"
            )
            test_s3_key = f"{self.config.config['s3']['processed_data_prefix']}train_test_split/test/{profile}_test_{self.current_date}{suffix}.csv"
            self.s3_manager.save_and_upload_dataframe(test_set, test_local, test_s3_key)
            
            # Save test set to input directory (for prediction)
            input_local = os.path.join(
                self.paths['output_path'], 'input',
                f"{profile}_test_{self.current_date}{suffix}.csv"
            )
            input_s3_key = f"{self.config.config['s3']['input_data_prefix']}{profile}_test_{self.current_date}{suffix}.csv"
            self.s3_manager.save_and_upload_dataframe(test_set, input_local, input_s3_key)
            
            logger.info(f"Split {profile}: {len(test_set)} test records")
            
            # Clear test set from memory
            del test_set
            gc.collect()
            
            self._log_memory_status(f"After processing {profile}")


def main():
    """Main entry point for preprocessing container"""
    try:
        # Log environment information
        from config import ENVIRONMENT, DEBUG_MODE, DATA_BUCKET, REDSHIFT_CLUSTER_IDENTIFIER
        logger.info("="*60)
        logger.info("ENERGY FORECASTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Debug Mode: {DEBUG_MODE}")
        logger.info(f"Data Bucket: {DATA_BUCKET}")
        logger.info(f"Redshift Cluster: {REDSHIFT_CLUSTER_IDENTIFIER}")
        logger.info("="*60)
        
        # Use memory-optimized pipeline if environment variable is set
        if os.getenv('MEMORY_OPTIMIZATION') == '1':
            logger.info("Using memory-optimized processing pipeline")
            pipeline = MemoryOptimizedEnergyPreprocessingPipeline()
        else:
            logger.info("Using standard processing pipeline")
            pipeline = EnergyPreprocessingPipeline()
 
        pipeline.run_preprocessing()
        logger.info("Preprocessing pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        logger.error("Configuration details:")
        try:
            from config import ENV_CONFIG
            logger.error(f"Loaded config keys: {list(ENV_CONFIG.keys())}")
            logger.error(f"Environment: {ENV_CONFIG.get('ENVIRONMENT', 'unknown')}")
        except Exception as config_error:
            logger.error(f"Could not load config details: {config_error}")
        sys.exit(1)


if __name__ == "__main__":
    main()