"""
Simple Directory Demo Processor
Processes all demo files in a directory and extracts snapshots
"""

import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.snapshot_extractor import extract_snapshots_to_json


def process_all_demos_in_directory(demo_directory, output_file="all_snapshots.json"):
    """
    Simple function to process all .dem files in a directory
    
    Args:
        demo_directory: Path to directory containing .dem files  
        output_file: Output JSON file for all snapshots
    """
    
    demo_path = Path(demo_directory)
    
    # Find all .dem files
    demo_files = list(demo_path.glob("**/*.dem"))
    # Sort by modification time, newest first
    demo_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not demo_files:
        print(f"No .dem files found in {demo_directory}")
        return
        
    print(f"Found {len(demo_files)} demo files (sorted by newest first)")

    # convert Path objects to strings
    demo_files = [str(p) for p in demo_files]
    extract_snapshots_to_json(demo_files, output_file)

    
    print(f"All demos processed. Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage - change this path to your demo directory
    demo_dir = "G://CS2/demos"  # Change this to your demo directory
    process_all_demos_in_directory(demo_dir, "all_snapshots.json")
