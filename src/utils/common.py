"""
Simple common utilities for CS2 demo analysis
"""

from pathlib import Path
import pandas as pd
import polars as pl


def get_project_root():
    """Get the project root directory - replaces Path(__file__).parent.parent.parent pattern"""
    # Find the project root by looking for key files
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:
        if (current_path / 'config.py').exists() and (current_path / 'main.py').exists():
            return current_path
        current_path = current_path.parent
    
    # Fallback to the common pattern
    return Path(__file__).parent.parent.parent


def ensure_pandas(df):
    """Convert polars DataFrame to pandas if needed"""
    if df is None:
        return None
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def ensure_dir(file_path):
    """Create parent directories for a file path - replaces mkdir pattern"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
