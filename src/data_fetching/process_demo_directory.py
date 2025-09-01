"""
Simple Directory Demo Processor
Processes all demo files in a directory and extracts snapshots
"""

import os
import json
from pathlib import Path
from ..core.snapshot_extractor import extract_snapshots_to_json


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
    
    # Process remaining demos in append mode
    for demo_file in demo_files[1:]:
        print(f"Processing: {demo_file.name}")
        try:
            extract_snapshots_to_json(str(demo_file), output_file, append_mode=True)
        except Exception as e:
            print(f"Error processing {demo_file.name}: {e}")
            continue
    
    print(f"All demos processed. Results saved to {output_file}")


if __name__ == "__main__":
    # Example usage - change this path to your demo directory
    demo_dir = "F://CS2/demos"  # Change this to your demo directory
    process_all_demos_in_directory(demo_dir, "all_snapshots.json")
