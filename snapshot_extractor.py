"""
CS2 Snapshot Extractor - Production Version
Minimal, concise code for extracting tick-based snapshots for ML training
"""

import json
import pandas as pd
import polars as pl
from cache_utils import load_demo


def extract_snapshots_to_json(demo_file: str, output_file: str = "snapshots.json", tick_interval=200):
    """Extract snapshots with ticks_left and save to JSON"""
    
    # Load demo
    demo = load_demo(demo_file)
    
    # Get rounds data
    rounds = demo.rounds
    if isinstance(rounds, pl.DataFrame):
        rounds = rounds.to_pandas()
    
    # Get kills data
    kills = demo.kills
    if isinstance(kills, pl.DataFrame):
        kills = kills.to_pandas()
    
    snapshots = []
    
    # Process each round
    for _, round_row in rounds.iterrows():
        freeze_end = round_row['freeze_end']
        end_tick = round_row['end']
        
        current_tick = freeze_end
        while current_tick < end_tick:
            ticks_left = end_tick - current_tick
            
            snapshot = {
                "ticks_left": ticks_left
            }
            
            snapshots.append(snapshot)
            current_tick += tick_interval
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(snapshots, f, indent=2)
    
    print(f"✅ Saved {len(snapshots)} snapshots to {output_file}")



if __name__ == "__main__":
    extract_snapshots_to_json("the-mongolz-vs-vitality-m1-mirage.dem")
