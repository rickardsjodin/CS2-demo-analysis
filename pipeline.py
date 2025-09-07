#!/usr/bin/env python3
"""
Simple script to run the CS2 demo pipeline for ML training data preparation.

This script provides an easy way to download, process, and cache CS2 demos
while managing disk space efficiently. Focused on preparing training data
for machine learning models.

Examples:
    # Process up to 5 demos from IEM Cologne 2025
    python pipeline.py --event 7907 --max-demos 5
    
    # List all processed demos
    python pipeline.py --list
    
    # Load a specific demo for ML training
    python pipeline.py --load 98668
    
    # Get all training data
    python pipeline.py --training-data
"""

import sys
from pathlib import Path

# Add project root to path so we can import our modules
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import and run the pipeline
from src.workflows.demo_pipeline import main

if __name__ == "__main__":
    main()
