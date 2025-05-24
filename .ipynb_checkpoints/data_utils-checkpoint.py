"""
Reddit Data Utilities

This module provides functions for loading, joining, and processing Reddit data
based on configuration settings.
"""

import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime


def load_and_join_data(config_path="data_config.yaml"):
    """
    Load and join Reddit data based on configuration
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with loaded and joined data
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    data = {}
    for data_type, sources in config['data_paths'].items():
        data[data_type] = {}
        for source_name, path in sources.items():
            if os.path.exists(path):
                data[data_type][source_name] = pd.read_parquet(path)
                print(f"Loaded {data_type} data for r/{source_name}: {len(data[data_type][source_name])} records")
            else:
                print(f"Warning: File not found - {path}")
                data[data_type][source_name] = pd.DataFrame()
    
    # Perform joins
    joined_data = {}
    
    # Join posts and comments
    if 'posts_to_comments' in config['join_keys']:
        join_config = config['join_keys']['posts_to_comments']
        
        for source_name in data['posts'].keys():
            if source_name in data['comments']:
                posts_df = data['posts'][source_name]
                comments_df = data['comments'][source_name]
                
                if len(posts_df) == 0 or len(comments_df) == 0:
                    print(f"Skipping join for r/{source_name} - missing data")
                    continue
                
                # Convert timestamps to datetime
                if 'created_utc' in posts_df.columns:
                    posts_df['date'] = pd.to_datetime(posts_df['created_utc'], unit='s').dt.date
                
                if 'created_utc' in comments_df.columns:
                    comments_df['date'] = pd.to_datetime(comments_df['created_utc'], unit='s').dt.date
                
                # Apply any transformations
                if 'right_transform' in join_config and join_config['right_transform']:
                    transform_func = eval(join_config['right_transform'])
                    comments_df[join_config['right_key']] = transform_func(comments_df[join_config['right_key']])
                
                # Perform the join
                joined_df = posts_df.merge(
                    comments_df,
                    left_on=join_config['left_key'],
                    right_on=join_config['right_key'],
                    how='left',
                    suffixes=('_post', '_comment')
                )
                
                print(f"Joined data for r/{source_name}: {len(joined_df)} records")
                joined_data[f"{source_name}_posts_comments"] = joined_df
    
    # Perform aggregations
    aggregated_data = {}
    if 'aggregation_rules' in config:
        for agg_name, agg_config in config['aggregation_rules'].items():
            for source_name, joined_df in joined_data.items():
                # Skip if no data
                if len(joined_df) == 0:
                    continue
                    
                # Group by specified columns
                group_cols = [col for col in agg_config['group_by'] if col in joined_df.columns]
                if not group_cols:
                    print(f"Warning: No valid group columns for {agg_name} in {source_name}")
                    continue
                    
                grouped = joined_df.groupby(group_cols)
                
                # Apply metrics
                agg_dict = {}
                for metric in agg_config['metrics']:
                    if metric['column'] in joined_df.columns:
                        agg_dict[metric['name']] = pd.NamedAgg(
                            column=metric['column'],
                            aggfunc=metric['function']
                        )
                
                if agg_dict:
                    aggregated_df = grouped.agg(**agg_dict).reset_index()
                    print(f"Created aggregation {agg_name} for {source_name}: {len(aggregated_df)} records")
                    aggregated_data[f"{source_name}_{agg_name}"] = aggregated_df
    
    # Add all data to result
    result = {
        'raw': data,
        'joined': joined_data,
        'aggregated': aggregated_data
    }
    
    return result


def prepare_sentiment_features(data):
    """
    Prepare sentiment features for modeling
    
    Parameters:
    -----------
    data : dict
        Dictionary of DataFrames from load_and_join_data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with sentiment features ready for modeling
    """
    # Find daily sentiment aggregations
    sentiment_dfs = []
    
    for key, df in data['aggregated'].items():
        if 'daily_sentiment' in key and 'compound_sentiment_mean' in df.columns:
            # Add source information
            source = key.split('_')[0]
            df['source'] = source
            sentiment_dfs.append(df)
    
    if not sentiment_dfs:
        print("No sentiment data found")
        return pd.DataFrame()
    
    # Combine all sentiment data
    combined_sentiment = pd.concat(sentiment_dfs, ignore_index=True)
    
    # Add any additional feature engineering here
    
    return combined_sentiment


def save_processed_data(data, output_dir="data"):
    """
    Save processed data to CSV files
    
    Parameters:
    -----------
    data : dict
        Dictionary of DataFrames from load_and_join_data
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save joined data
    for key, df in data['joined'].items():
        if len(df) > 0:
            output_path = os.path.join(output_dir, f"{key}_joined.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved joined data to {output_path}")
    
    # Save aggregated data
    for key, df in data['aggregated'].items():
        if len(df) > 0:
            output_path = os.path.join(output_dir, f"{key}_aggregated.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved aggregated data to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Loading and processing Reddit data...")
    data = load_and_join_data()
    
    # Prepare features
    sentiment_features = prepare_sentiment_features(data)
    if len(sentiment_features) > 0:
        print(f"Prepared sentiment features: {len(sentiment_features)} records")
        
        # Save to CSV
        sentiment_features.to_csv("data/sentiment_features.csv", index=False)
        print("Saved sentiment features to data/sentiment_features.csv")
    
    # Save all processed data
    save_processed_data(data)
    
    print("Data processing complete!")
