"""
Feature Engineering for CS2 Win Probability Model
This module creates the features used for model training and prediction.
"""
import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features for the win probability model from a snapshot DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing snapshot data. 
                           Must include 'cts_alive', 'ts_alive', 'time_left', 
                           and 'bomb_planted'.

    Returns:
        pd.DataFrame: The DataFrame with added feature columns.
    """
    # Convert map_name to categorical type for XGBoost compatibility
    if 'map_name' in df.columns:
        df['map_name'] = df['map_name'].astype('category')
    
    # Player advantage
    df['player_advantage'] = df['cts_alive'] - df['ts_alive']
    
    # Ratio of alive players
    # Added a small epsilon to avoid division by zero if both teams have 0 players
    df['ct_alive_ratio'] = df['cts_alive'] / (df['cts_alive'] + df['ts_alive'] + 1e-8)
    
    # Unambiguous time features
    # Time left in the round (0 if bomb is planted)
    df['round_time_left'] = df.apply(lambda row: row['time_left'] if not row['bomb_planted'] else 0, axis=1)
    # Time left on the bomb timer (0 if not planted)
    df['bomb_time_left'] = df.apply(lambda row: row['time_left'] if row['bomb_planted'] else 0, axis=1)
    
    # Contextual time pressure features (normalized 0-1)
    # Pressure on CTs from the bomb
    df['time_pressure_ct'] = df.apply(
        lambda row: (40.0 - row['time_left']) / 40.0 if row['bomb_planted'] else 0, 
        axis=1
    )
    # Pressure on Ts from the round timer
    df['time_pressure_t'] = df.apply(
        lambda row: (115.0 - row['time_left']) / 115.0 if not row['bomb_planted'] else 0, 
        axis=1
    )
    
    return df
