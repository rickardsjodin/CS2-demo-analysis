"""
Simple Directory Demo Processor
Processes all demo files in a directory and extracts snapshots
"""

import sys
import os
import json
from pathlib import Path
from tqdm import tqdm


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
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
    
    if not demo_files:
        print(f"No .dem files found in {demo_directory}")
        return
        
    print(f"Found {len(demo_files)} demo files")
    
    # Process first demo to create the file
    first_demo = demo_files[0]
    print(f"Processing: {first_demo.name}")
    extract_snapshots_to_json(str(first_demo), output_file)
    
    # Process remaining demos in append mode with progress bar
    for demo_file in tqdm(demo_files[1:], desc="Processing demos", unit="demo"):
        try:
            extract_snapshots_to_json(str(demo_file), output_file, append_mode=True)
        except Exception as e:
            tqdm.write(f"Error processing {demo_file.name}: {e}")
            continue
    
    print(f"All demos processed. Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage - change this path to your demo directory
    demo_dir = "F://CS2/demos"  # Change this to your demo directory
    process_all_demos_in_directory(demo_dir, "all_snapshots.json")
