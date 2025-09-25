#!/usr/bin/env python3
"""
Redshift Infrastructure Setup Script for Energy Load Forecasting Pipeline
Creates forecasting table and materialized view for SDCP energy forecasting
Uses Redshift Data API to avoid GitHub Actions connection issues
"""
import boto3
import time
import os
import json
import traceback
from datetime import datetime

def setup_energy_forecasting_redshift():
    """Setup Redshift table and materialized view for energy forecasting"""
    
    # Get configuration from environment variables
    try:
        cluster_identifier = os.environ['REDSHIFT_CLUSTER_IDENTIFIER']
        database = os.environ['REDSHIFT_DATABASE']
        db_user = os.environ['REDSHIFT_DB_USER']
        region = os.environ['REDSHIFT_REGION']
        
        # Schema and table configuration
        forecasting_schema = os.environ['REDSHIFT_FORECASTING_SCHEMA']
        forecasting_table = os.environ['REDSHIFT_FORECASTING_TABLE']
        bi_schema = os.environ['REDSHIFT_BI_SCHEMA']
        materialized_view = os.environ['REDSHIFT_MATERIALIZED_VIEW']
        input_schema = os.environ['REDSHIFT_INPUT_SCHEMA']
        input_table = os.environ['REDSHIFT_INPUT_TABLE']
        
        env_name = os.environ['ENVIRONMENT']
        
    except KeyError as e:
        print(f" Missing required environment variable: {e}")
        return 'failed'
    
    print(f" Setting up Redshift infrastructure for {env_name} environment")
    print(f"   Cluster: {cluster_identifier}")
    print(f"   Database: {database}")
    print(f"   Forecasting Schema: {forecasting_schema}")
    print(f"   BI Schema: {bi_schema}")
    print(f"   Table: {forecasting_schema}.{forecasting_table}")
    print(f"   Materialized View: {bi_schema}.{materialized_view}")
    
    try:
        # Step 1: Verify cluster exists and get basic info
        print('Step 1: Verifying Redshift cluster...')
        try:
            verify_cluster_exists(cluster_identifier, region)
        except Exception as e:
            print(f" CRITICAL: Cluster verification failed: {str(e)}")
            return 'failed'
        
        # Step 2: Verify schemas exist
        print('Step 2: Verifying schemas exist...')
        try:
            verify_schemas_exist(cluster_identifier, database, db_user, region,
                               forecasting_schema, bi_schema, input_schema)
        except Exception as e:
            print(f" CRITICAL: Schema verification failed: {str(e)}")
            return 'failed'
        
        # Step 3: Create forecasting table
        print('Step 3: Creating forecasting table...')
        try:
            table_already_existed = create_forecasting_table(cluster_identifier, database, db_user, region,
                                   forecasting_schema, forecasting_table)
        except Exception as e:
            print(f" CRITICAL: Forecasting table creation failed: {str(e)}")
            return 'failed'
        
        # Step 3.1: Verify table was created
        if not table_already_existed:
            print('Step 3.1: Verifying table creation...')
            try:
                verify_table_creation(cluster_identifier, database, db_user, region,
                                    forecasting_schema, forecasting_table)
            except Exception as e:
                print(f" CRITICAL: Table verification failed: {str(e)}")
                return 'failed'
        else:
            print(' Step 3.1: Skipping table verification since it already existed')

        # Step 4: Create materialized view
        print('Step 4: Creating materialized view...')
        try:
            view_already_existed = create_materialized_view(cluster_identifier, database, db_user, region,
                                   bi_schema, materialized_view, forecasting_schema, 
                                   forecasting_table, input_schema, input_table)
        except Exception as e:
            print(f" CRITICAL: Materialized view creation failed: {str(e)}")
            return 'failed'
        
        # Step 4.1: Verify materialized view was created
        if not view_already_existed:
            print('Step 4.1: Verifying materialized view creation...')
            try:
                verify_view_creation(cluster_identifier, database, db_user, region,
                                bi_schema, materialized_view)
            except Exception as e:
                print(f" CRITICAL: Materialized view verification failed: {str(e)}")
                return 'failed'
        else:
            print(' Step 4.1: Skipping materialized view verification since it already existed')

        # Step 5: Save configuration for reference
        print('Step 5: Saving configuration...')
        try:
            save_configuration(cluster_identifier, database, forecasting_schema, 
                             forecasting_table, bi_schema, materialized_view, env_name)
        except Exception as e:
            print(f" Configuration save failed (non-critical): {str(e)}")
        
        print(' Energy forecasting Redshift infrastructure setup completed successfully!')
        return 'success'
    
    except Exception as e:
        print(f" Error setting up energy forecasting infrastructure: {str(e)}")
        traceback.print_exc()
        return 'failed'

def verify_cluster_exists(cluster_identifier, region):
    """Verify cluster exists and get basic info"""
    try:
        redshift_client = boto3.client('redshift', region_name=region)
        
        cluster_response = redshift_client.describe_clusters(
            ClusterIdentifier=cluster_identifier
        )
        cluster = cluster_response['Clusters'][0]
        
        print(f"   Cluster found: {cluster_identifier}")
        print(f"   Status: {cluster['ClusterStatus']}")
        print(f"   Endpoint: {cluster['Endpoint']['Address']}:{cluster['Endpoint']['Port']}")
        
        if cluster['ClusterStatus'] != 'available':
            raise Exception(f"Cluster is not available. Status: {cluster['ClusterStatus']}")
        
        return cluster
        
    except Exception as e:
        print(f"   Error verifying cluster: {str(e)}")
        raise

def verify_schemas_exist(cluster_identifier, database, db_user, region,
                        forecasting_schema, bi_schema, input_schema):
    """Verify that all required schemas exist"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
        
        # Check all required schemas
        schema_check_sql = f"""
        SELECT nspname as schema_name
        FROM pg_namespace
        WHERE nspname IN ('{forecasting_schema}', '{bi_schema}', '{input_schema}')
        ORDER BY nspname
        """
        
        print(f"   Checking schemas: {forecasting_schema}, {bi_schema}, {input_schema}")
        
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=schema_check_sql
        )
        
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'schema verification')
        
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        
        found_schemas = []
        for row in result_response.get('Records', []):
            if row and len(row) > 0 and 'stringValue' in row[0]:
                found_schemas.append(row[0]['stringValue'])
        
        print(f"   Found schemas: {found_schemas}")
        
        # Check if all required schemas exist
        required_schemas = [forecasting_schema, bi_schema, input_schema]
        missing_schemas = [schema for schema in required_schemas if schema not in found_schemas]
        
        if missing_schemas:
            print(f"   Missing required schemas: {missing_schemas}")
            
            # Show available schemas for debugging
            all_schemas_sql = """
            SELECT nspname 
            FROM pg_namespace 
            WHERE nspname NOT LIKE 'pg_%' AND nspname != 'information_schema' 
            ORDER BY nspname
            """
            
            all_schemas_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=all_schemas_sql
            )
            
            all_schemas_query_id = all_schemas_response['Id']
            wait_for_query_completion(redshift_data_client, all_schemas_query_id, 'list all schemas')
            
            all_schemas_result = redshift_data_client.get_statement_result(Id=all_schemas_query_id)
            
            available_schemas = []
            for row in all_schemas_result.get('Records', []):
                if row and len(row) > 0 and 'stringValue' in row[0]:
                    available_schemas.append(row[0]['stringValue'])
            
            print(f"   Available schemas in cluster: {available_schemas}")
            raise Exception(f"Missing required schemas: {missing_schemas}. Available: {available_schemas}")
        
        print(f"   All required schemas exist: {required_schemas}")
        
    except Exception as e:
        print(f"   Schema verification failed: {str(e)}")
        raise

def create_forecasting_table(cluster_identifier, database, db_user, region,
                           forecasting_schema, forecasting_table):
    """Create forecasting table using Redshift Data API"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
        
        # Check if table already exists
        check_table_sql = f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = '{forecasting_schema}'
        AND table_name = '{forecasting_table}'
        """
        
        print(f"   Checking if table {forecasting_schema}.{forecasting_table} exists...")
        
        check_response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=check_table_sql
        )
        
        check_query_id = check_response['Id']
        wait_for_query_completion(redshift_data_client, check_query_id, 'table existence check')
        
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=check_query_id)
        table_exists = int(result_response['Records'][0][0]['longValue']) > 0
        
        if table_exists:
            print(f"   Table {forecasting_schema}.{forecasting_table} already exists")
            
            # Verify table structure
            verify_table_sql = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = '{forecasting_schema}'
            AND table_name = '{forecasting_table}'
            ORDER BY ordinal_position
            """
            
            verify_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=verify_table_sql
            )
            
            verify_query_id = verify_response['Id']
            wait_for_query_completion(redshift_data_client, verify_query_id, 'table structure verification')
            
            structure_result = redshift_data_client.get_statement_result(Id=verify_query_id)
            columns = structure_result['Records']
            
            print(f"   Table has {len(columns)} columns:")
            for col in columns:
                col_name = col[0]['stringValue']
                col_type = col[1]['stringValue']
                max_len = col[2].get('stringValue', '') if col[2].get('stringValue') else ''
                length_info = f"({max_len})" if max_len else ""
                print(f"      - {col_name}: {col_type}{length_info}")
            
            return True  # Indicate that the table already existed
        else:
            # Create new table
            print(f"   Creating table {forecasting_schema}.{forecasting_table}...")
            
            create_table_sql = f"""
            CREATE TABLE {forecasting_schema}.{forecasting_table} (
                tradedatetime timestamp without time zone ENCODE raw,
                predicted_load numeric(18, 6) ENCODE az64,
                count numeric(18, 0) ENCODE az64,
                load_all numeric(18, 6) ENCODE az64,
                profile character varying(256) ENCODE lzo,
                segment character varying(256) ENCODE lzo,
                created_at timestamp without time zone DEFAULT ('now'::text)::timestamp with time zone ENCODE az64
            ) 
            DISTSTYLE AUTO
            SORTKEY (tradedatetime);
            """
            
            create_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=create_table_sql
            )
            
            create_query_id = create_response['Id']
            wait_for_query_completion(redshift_data_client, create_query_id, 'table creation')
            
            print(f"   Table {forecasting_schema}.{forecasting_table} created successfully")
            
            return False  # Indicate that the table was newly created
        
    except Exception as e:
        print(f"   Error with forecasting table: {str(e)}")
        raise

def verify_table_creation(cluster_identifier, database, db_user, region,
                         forecasting_schema, forecasting_table):
    """Explicitly verify that the table was actually created"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
        
        verify_sql = f"""
        SELECT
            table_schema,
            table_name,
            table_type
        FROM information_schema.tables
        WHERE table_schema = '{forecasting_schema}'
        AND table_name = '{forecasting_table}'
        """
        
        print(f"   Verifying table {forecasting_schema}.{forecasting_table}...")
        
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=verify_sql
        )
        
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'table verification')
        
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        records = result_response.get('Records', [])
        
        if records:
            table_info = records[0]
            found_schema = table_info[0]['stringValue']
            found_table = table_info[1]['stringValue']
            found_type = table_info[2]['stringValue']
            
            print(f"   VERIFIED: Table found!")
            print(f"      Schema: {found_schema}")
            print(f"      Table: {found_table}")
            print(f"      Type: {found_type}")
            
        else:
            print(f"   Table {forecasting_schema}.{forecasting_table} NOT FOUND after creation!")
            raise Exception(f"Table verification failed - table not found after creation")
        
    except Exception as e:
        print(f"   Table verification failed: {str(e)}")
        raise

def create_materialized_view(cluster_identifier, database, db_user, region,
                           bi_schema, materialized_view, forecasting_schema, 
                           forecasting_table, input_schema, input_table):
    """Create materialized view using Redshift Data API"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
        
        # Check if view already exists
        check_view_sql = f"""
        SELECT COUNT(*)
        FROM pg_class c
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = '{bi_schema}'
        AND c.relname = '{materialized_view}'
        AND c.relkind IN ('m', 'v')
        """
        
        print(f"   Checking if materialized view {bi_schema}.{materialized_view} exists...")
        
        check_response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=check_view_sql
        )
        
        check_query_id = check_response['Id']
        wait_for_query_completion(redshift_data_client, check_query_id, 'view existence check')
        
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=check_query_id)
        view_exists = int(result_response['Records'][0][0]['longValue']) > 0
        
        if view_exists:
            print(f"   Materialized view {bi_schema}.{materialized_view} already exists")
            print(f"   Refreshing materialized view to ensure latest data...")
            
            # Refresh the materialized view
            refresh_sql = f"REFRESH MATERIALIZED VIEW {bi_schema}.{materialized_view};"
            
            refresh_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=refresh_sql
            )
            
            refresh_query_id = refresh_response['Id']
            wait_for_query_completion(redshift_data_client, refresh_query_id, 'materialized view refresh', max_wait_seconds=600)
            
            print(f"   Materialized view refreshed successfully")

            return True  # Indicate that the view already existed
            
        else:
            # Create new materialized view
            print(f"   Creating materialized view {bi_schema}.{materialized_view}...")
            
            create_view_sql = get_materialized_view_sql(bi_schema, materialized_view, 
                                                       forecasting_schema, forecasting_table,
                                                       input_schema, input_table)
            
            create_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=create_view_sql
            )
            
            create_query_id = create_response['Id']
            wait_for_query_completion(redshift_data_client, create_query_id, 'materialized view creation', max_wait_seconds=600)
            
            print(f"   Materialized view {bi_schema}.{materialized_view} created successfully")
            
            return False  # Indicate that the view was newly created
        
    except Exception as e:
        print(f"   Error with materialized view: {str(e)}")
        raise

def get_materialized_view_sql(bi_schema, materialized_view, forecasting_schema, 
                             forecasting_table, input_schema, input_table):
    """Generate the SQL for creating the materialized view"""
    return f"""
    CREATE MATERIALIZED VIEW {bi_schema}.{materialized_view} AS
    WITH latest_predictions AS (
        -- Deduplicate predictions - get latest for each date/hour/profile/segment
        SELECT 
            tradedatetime as forecast_datetime,
            profile as load_profile,
            segment as load_segment,
            load_all as predicted_lossadjustedload,
            ROW_NUMBER() OVER (
                PARTITION BY 
                    CAST(tradedatetime AS DATE),
                    EXTRACT(HOUR FROM tradedatetime),
                    profile,
                    segment
                ORDER BY created_at DESC  -- Latest by datetime
            ) as rn
        FROM {forecasting_schema}.{forecasting_table}
    ),
    hourly_data AS (
        SELECT 
            a.tradedatelocal as trade_date,
            CAST(a.tradehourstartlocal AS INTEGER) as trade_hour,
            a.loadprofile,
            
            CASE 
                WHEN a.loadprofile = 'RES' THEN 
                    CASE 
                        WHEN a.rategroup LIKE '%NEM%' OR a.rategroup LIKE '%SBP%' THEN 'SOLAR'
                        ELSE 'NONSOLAR'
                    END
                ELSE 'ALL'  -- For all non-RES profiles, segment is ALL
            END as segment,
            
            a.submission,
            
            -- Time dimensions
            EXTRACT(YEAR FROM a.tradedatelocal) as year,
            EXTRACT(MONTH FROM a.tradedatelocal) as month,
            EXTRACT(DAY FROM a.tradedatelocal) as day,
            CASE 
                WHEN EXTRACT(DOW FROM a.tradedatelocal) IN (0, 6) THEN 'Weekend'
                ELSE 'Weekday'
            END as day_type,
            CASE EXTRACT(DOW FROM a.tradedatelocal)
                WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday'
                WHEN 3 THEN 'Wednesday' WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday'
                WHEN 6 THEN 'Saturday'
            END as dow,
            
            -- Time-specific columns based on hour
            CASE 
                WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 10 AND 17 THEN 'Solar Hours (10am-6pm)'
                ELSE 'Non-Solar Hours'
            END as solar_hours,
            
            CASE 
                WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 16 AND 20 THEN 'On Peak (4-9pm)'
                WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 0 AND 5 THEN 'Super Off Peak (12-6am)'
                ELSE 'Off Peak (Other)'
            END as tou_period,
            
            SUM(a.lossadjustedload) as lossadjustedload,
            
            COALESCE(
                CASE 
                    WHEN a.loadprofile = 'RES' THEN
                        -- For RES profile, match segment exactly (SOLAR/NONSOLAR)
                        CASE 
                            WHEN (a.rategroup LIKE '%NEM%' OR a.rategroup LIKE '%SBP%') AND p.load_segment = 'SOLAR' THEN p.predicted_lossadjustedload
                            WHEN (a.rategroup NOT LIKE '%NEM%' AND a.rategroup NOT LIKE '%SBP%') AND p.load_segment = 'NONSOLAR' THEN p.predicted_lossadjustedload
                            ELSE NULL
                        END
                    ELSE
                        -- For all other profiles, use ALL segment predictions
                        CASE WHEN p.load_segment = 'ALL' THEN p.predicted_lossadjustedload ELSE NULL END
                END, 
                0
            ) as demand_bid_output
            
        FROM {input_schema}.{input_table} a
        LEFT JOIN latest_predictions p  -- Use deduplicated predictions
            ON a.tradedatelocal = CAST(p.forecast_datetime AS DATE)
            AND CAST(a.tradehourstartlocal AS INTEGER) = EXTRACT(HOUR FROM p.forecast_datetime)
            AND a.loadprofile = p.load_profile
            AND (
                -- For RES profile, match on SOLAR/NONSOLAR segment
                (a.loadprofile = 'RES' AND 
                 (((a.rategroup LIKE '%NEM%' OR a.rategroup LIKE '%SBP%') AND p.load_segment = 'SOLAR') OR 
                  ((a.rategroup NOT LIKE '%NEM%' AND a.rategroup NOT LIKE '%SBP%') AND p.load_segment = 'NONSOLAR'))
                )
                OR
                -- For all other profiles, match on ALL segment
                (a.loadprofile != 'RES' AND p.load_segment = 'ALL')
            )
            AND p.rn = 1  -- Only get the latest prediction
        WHERE a.submission IN ('Final', 'Initial')
            AND a.tradedatelocal > '2024-11-05'
        GROUP BY 
            a.tradedatelocal, 
            CAST(a.tradehourstartlocal AS INTEGER), 
            a.loadprofile,
            CASE 
                WHEN a.loadprofile = 'RES' THEN 
                    CASE 
                        WHEN a.rategroup LIKE '%NEM%' OR a.rategroup LIKE '%SBP%' THEN 'SOLAR'
                        ELSE 'NONSOLAR'
                    END
                ELSE 'ALL'
            END,
            a.submission,
            EXTRACT(YEAR FROM a.tradedatelocal), 
            EXTRACT(MONTH FROM a.tradedatelocal), 
            EXTRACT(DAY FROM a.tradedatelocal),
            CASE WHEN EXTRACT(DOW FROM a.tradedatelocal) IN (0, 6) THEN 'Weekend' ELSE 'Weekday' END,
            CASE EXTRACT(DOW FROM a.tradedatelocal) 
                WHEN 0 THEN 'Sunday' WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday' 
                WHEN 3 THEN 'Wednesday' WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' 
                WHEN 6 THEN 'Saturday' 
            END,
            CASE WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 10 AND 17 THEN 'Solar Hours (10am-6pm)' ELSE 'Non-Solar Hours' END,
            CASE WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 16 AND 20 THEN 'On Peak (4-9pm)' 
                 WHEN CAST(a.tradehourstartlocal AS INTEGER) BETWEEN 0 AND 5 THEN 'Super Off Peak (12-6am)' 
                 ELSE 'Off Peak (Other)' 
            END,
            COALESCE(
                CASE 
                    WHEN a.loadprofile = 'RES' THEN
                        CASE 
                            WHEN (a.rategroup LIKE '%NEM%' OR a.rategroup LIKE '%SBP%') AND p.load_segment = 'SOLAR' THEN p.predicted_lossadjustedload
                            WHEN (a.rategroup NOT LIKE '%NEM%' AND a.rategroup NOT LIKE '%SBP%') AND p.load_segment = 'NONSOLAR' THEN p.predicted_lossadjustedload
                            ELSE NULL
                        END
                    ELSE
                        CASE WHEN p.load_segment = 'ALL' THEN p.predicted_lossadjustedload ELSE NULL END
                END, 
                0
            )
    ),

    hourly_combined AS (
        -- Combine INITIAL and FINAL data for each hour
        SELECT 
            trade_date, trade_hour, loadprofile, segment,
            year, month, day, day_type, dow, tou_period, solar_hours,
            
            -- ASQMD (from FINAL submission)
            MAX(CASE WHEN submission = 'Final' THEN lossadjustedload END) as asqmd,
            -- ESQMD (from INITIAL submission)  
            MAX(CASE WHEN submission = 'Initial' THEN lossadjustedload END) as esqmd,
            -- Demand Bid Output (same for both)
            MAX(demand_bid_output) as demand_bid_output
            
        FROM hourly_data
        GROUP BY trade_date, trade_hour, loadprofile, segment, year, month, day, day_type, dow, tou_period, solar_hours
    )
    -- Final SELECT: Hourly-level data with individual error calculations
    SELECT 
        trade_date as "Flowday",  -- Same as trade_date
        CAST(trade_hour AS VARCHAR) || ':00' as "Time",  -- Hour format: "0:00", "1:00", etc.
        day_type as "DayType",
        dow as "DOW",
        segment as "Segment",
        loadprofile as "CustomerProfile",
        solar_hours as "SolarHours",
        tou_period as "TOUPeriod",
        
        -- Load values
        asqmd/1000 as "ASQMD",
        asqmd/1000 as "ASQMDNet",  -- Same as ASQMD
        esqmd/1000 as "ESQMD", 
        esqmd/1000 as "ESQMDNet",  -- Same as ESQMD
        demand_bid_output/1000 as "DemandBidOutput",
        
        -- Individual hourly error calculations for PowerBI
        CASE 
            WHEN asqmd IS NULL OR asqmd = 0 THEN NULL
            WHEN demand_bid_output = 0 THEN 0
            ELSE ABS(demand_bid_output - asqmd)
        END as "AE",  -- Absolute Error
        
        CASE 
            WHEN asqmd IS NULL OR asqmd = 0 THEN NULL
            WHEN demand_bid_output = 0 THEN 0
            ELSE ABS(demand_bid_output - asqmd) / ABS(asqmd)
        END as "APE",  -- Absolute Percentage Error
        
        CASE 
            WHEN asqmd IS NULL OR asqmd = 0 THEN NULL
            WHEN demand_bid_output = 0 THEN 0
            ELSE POWER(demand_bid_output - asqmd, 2)
        END as "SE",  -- Squared Error
        
        -- Time and dimension columns
        year as "Year",
        month as "Month", 
        day as "Day",
        trade_hour as "Hour",
        CAST(trade_date AS VARCHAR) || ' ' || CAST(trade_hour AS VARCHAR) || ':00:00' as "TradeDateTime",  -- DateTime string
        
        -- Additional columns
        'SDCP' as "Developer",
        CURRENT_TIMESTAMP as "LastUpdated"
    FROM hourly_combined
    WHERE asqmd IS NOT NULL  -- Only include records with valid actual data
    """

def verify_view_creation(cluster_identifier, database, db_user, region,
                        bi_schema, materialized_view):
    """Verify that the materialized view was actually created using multiple approaches"""
    try:
        redshift_data_client = boto3.client('redshift-data', region_name=region)
       
        # Method 1: Use pg_class with relkind 'm' for materialized views
        verify_sql_method1 = f"""
        SELECT
            n.nspname as schema_name,
            c.relname as view_name,
            c.relkind as object_type
        FROM pg_class c
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = '{bi_schema}'
        AND c.relname = '{materialized_view}'
        AND c.relkind IN ('m', 'v')
        """
       
        print(f"   Verifying materialized view {bi_schema}.{materialized_view} using pg_class...")
       
        response = redshift_data_client.execute_statement(
            ClusterIdentifier=cluster_identifier,
            Database=database,
            DbUser=db_user,
            Sql=verify_sql_method1
        )
       
        query_id = response['Id']
        wait_for_query_completion(redshift_data_client, query_id, 'view verification (method 1)')
       
        # Get results
        result_response = redshift_data_client.get_statement_result(Id=query_id)
        records = result_response.get('Records', [])
       
        if records:
            view_info = records[0]
            found_schema = view_info[0]['stringValue']
            found_view = view_info[1]['stringValue']
            object_type = view_info[2]['stringValue']
           
            print(f"   VERIFIED: Materialized view found using pg_class!")
            print(f"      Schema: {found_schema}")
            print(f"      View: {found_view}")
            # print(f"      Object Type: {object_type} (m = materialized view)")
           
        else:
            print(f"   Method 1 failed, trying alternative method...")
           
            # Method 2: Try using information_schema.tables (some versions include materialized views here)
            verify_sql_method2 = f"""
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = '{bi_schema}'
            AND table_name = '{materialized_view}'
            """
           
            print(f"   Trying verification using information_schema.tables...")
           
            response2 = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=verify_sql_method2
            )
           
            query_id2 = response2['Id']
            wait_for_query_completion(redshift_data_client, query_id2, 'view verification (method 2)')
           
            result_response2 = redshift_data_client.get_statement_result(Id=query_id2)
            records2 = result_response2.get('Records', [])
           
            if records2:
                view_info2 = records2[0]
                found_schema2 = view_info2[0]['stringValue']
                found_view2 = view_info2[1]['stringValue']
                table_type2 = view_info2[2]['stringValue']
               
                print(f"   VERIFIED: Materialized view found using information_schema!")
                print(f"      Schema: {found_schema2}")
                print(f"      View: {found_view2}")
                print(f"      Table Type: {table_type2}")
               
            else:
                print(f"   Method 2 also failed, trying direct query method...")
               
                # Method 3: Try to query the view directly (ultimate test)
                test_query_sql = f"""
                SELECT COUNT(*) as row_count
                FROM {bi_schema}.{materialized_view}
                LIMIT 1
                """
               
                print(f"   Trying direct query to verify view exists and is accessible...")
               
                response3 = redshift_data_client.execute_statement(
                    ClusterIdentifier=cluster_identifier,
                    Database=database,
                    DbUser=db_user,
                    Sql=test_query_sql
                )
               
                query_id3 = response3['Id']
                wait_for_query_completion(redshift_data_client, query_id3, 'direct query test')
               
                result_response3 = redshift_data_client.get_statement_result(Id=query_id3)
               
                if result_response3.get('Records'):
                    row_count = int(result_response3['Records'][0][0]['longValue'])
                    print(f"   VERIFIED: Materialized view is accessible and contains {row_count:,} rows")
                else:
                    print(f"   Direct query failed - materialized view may not exist or be accessible")
                    raise Exception(f"Materialized view verification failed - view not accessible")
       
        # If we get here, at least one method succeeded
        # Get final row count for confirmation
        try:
            count_sql = f"SELECT COUNT(*) FROM {bi_schema}.{materialized_view}"
           
            count_response = redshift_data_client.execute_statement(
                ClusterIdentifier=cluster_identifier,
                Database=database,
                DbUser=db_user,
                Sql=count_sql
            )
           
            count_query_id = count_response['Id']
            wait_for_query_completion(redshift_data_client, count_query_id, 'final row count check')
           
            count_result = redshift_data_client.get_statement_result(Id=count_query_id)
            final_row_count = int(count_result['Records'][0][0]['longValue'])
           
            print(f"   Final Row Count: {final_row_count:,}")
           
            if final_row_count == 0:
                print(f"   Warning: Materialized view exists but contains no data")
            else:
                print(f"   Materialized view contains data and is ready for use")
               
        except Exception as count_error:
            print(f"   Could not get row count (non-critical): {str(count_error)}")
       
    except Exception as e:
        print(f"   Materialized view verification failed: {str(e)}")
        raise

def wait_for_query_completion(redshift_data_client, query_id, operation_name, max_wait_seconds=300):
    """Wait for Redshift Data API query to complete with better error handling"""
    print(f"   Waiting for {operation_name} to complete (Query ID: {query_id})")
    
    waited = 0
    while waited < max_wait_seconds:
        try:
            status_response = redshift_data_client.describe_statement(Id=query_id)
            status = status_response['Status']
            
            if status == 'FINISHED':
                print(f"   {operation_name} completed successfully")
                return
            elif status == 'FAILED':
                error_msg = status_response.get('Error', 'Unknown error')
                print(f"   {operation_name} failed: {error_msg}")
                
                # Try to get more details
                if 'QueryString' in status_response:
                    query_string = status_response['QueryString']
                    print(f"   Failed query: {query_string[:200]}...")
                
                raise Exception(f'{operation_name} failed: {error_msg}')
            elif status == 'ABORTED':
                raise Exception(f'{operation_name} was aborted')
            
            # Still running, wait a bit more
            time.sleep(5)
            waited += 5
            
            if waited % 30 == 0:  # Log progress every 30 seconds
                print(f"   Still waiting for {operation_name}... ({waited}s elapsed, status: {status})")
        
        except Exception as e:
            if 'failed:' in str(e) or 'aborted' in str(e):
                raise  # Re-raise operation failures
            else:
                print(f"   Error checking status: {str(e)}")
                time.sleep(5)
                waited += 5
                continue
    
    raise Exception(f'{operation_name} timed out after {max_wait_seconds} seconds')

def save_configuration(cluster_identifier, database, forecasting_schema, 
                      forecasting_table, bi_schema, materialized_view, env_name):
    """Save configuration for reference"""
    try:
        config = {
            'infrastructure_type': 'redshift_energy_forecasting',
            'cluster_identifier': cluster_identifier,
            'database': database,
            'environment': env_name,
            'forecasting': {
                'schema': forecasting_schema,
                'table': forecasting_table,
                'full_table_name': f'{forecasting_schema}.{forecasting_table}'
            },
            'bi': {
                'schema': bi_schema,
                'materialized_view': materialized_view,
                'full_view_name': f'{bi_schema}.{materialized_view}'
            },
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'uses_data_api': True,
                'schemas_pre_exist': True,
                'supports_energy_forecasting': True,
                'materialized_view_enabled': True,
                'error_calculations_included': True
            }
        }
        
        # Save to local file (will be uploaded as artifact)
        config_filename = f'redshift-setup-{env_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(config_filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   Configuration saved to {config_filename}")
        print(f"   Setup summary:")
        print(f"      Database: {database}")
        print(f"      Forecasting table: {forecasting_schema}.{forecasting_table}")
        print(f"      Materialized view: {bi_schema}.{materialized_view}")
        
    except Exception as e:
        print(f"   Configuration save failed: {str(e)}")
        raise

if __name__ == '__main__':
    print("=== Energy Forecasting Redshift Infrastructure Setup ===")
    print("Setting up forecasting table and materialized view for SDCP energy forecasting")
    
    # Run the setup
    try:
        result = setup_energy_forecasting_redshift()
        print(f'Setup result: {result}')
        
        if result == 'success':
            print(' SUCCESS: Energy forecasting infrastructure setup completed')
            exit(0)
        else:
            print(' FAILURE: Energy forecasting infrastructure setup failed')
            exit(1)
        
    except Exception as e:
        print(f' FATAL ERROR: {str(e)}')
        traceback.print_exc()
        exit(1)
