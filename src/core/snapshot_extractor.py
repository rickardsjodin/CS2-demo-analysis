"""
CS2 Snapshot Extractor - Production Version
Minimal, concise code for extracting tick-based snapshots for ML training
"""

import json
import os
import pickle
import pandas as pd
import polars as pl
from tqdm import tqdm
try:
    from ..utils.cache_utils import CACHE_DIR
except ImportError:
    # Handle case when running directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.cache_utils import CACHE_DIR

ROUND_TIME = 115  
BOMB_TIME = 40   

def extract_snapshots_to_json(parsed_demo_file: str, output_file: str = "snapshots.json", tick_rate=64, append_mode=False):
    """Extract snapshots with time_left and alive counts at every kill event
    
    Args:
        demo_file: Path to the demo file
        output_file: Output JSON file path
        tick_rate: Game tick rate  
        append_mode: If True, append to existing file instead of overwriting
    """
    
    with open(parsed_demo_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Create a minimal demo-like object with the cached data
    class CachedDemo:
        def __init__(self, data):
            self.kills = data.get('kills')
            self.rounds = data.get('rounds')
            self.damages = data.get('damages')
            self.smokes = data.get('smokes')
            self.flashes = data.get('flashes')
            self.grenades = data.get('grenades')
            self.bomb = data.get('bomb')
            self.ticks = data.get('ticks')
    
    demo = CachedDemo(cache_data)

    
    # Get rounds data
    rounds = demo.rounds
    if isinstance(rounds, pl.DataFrame):
        rounds = rounds.to_pandas()
    
    # Get kills data to calculate alive counts
    kills = demo.kills
    if isinstance(kills, pl.DataFrame):
        kills = kills.to_pandas()
    
    # Get bomb events to track plants
    bomb_events = demo.bomb
    if isinstance(bomb_events, pl.DataFrame):
        bomb_events = bomb_events.to_pandas()
    
    
    snapshots = []
    
    # Process each round
    for _, round_row in rounds.iterrows():
        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        end_tick = round_row['end']
        winner = round_row['winner']
        plant_tick = round_row['bomb_plant']
        
        # Get kills for this round
        round_kills = kills[kills['round_num'] == round_num]
        
        # Take snapshot at round start (freeze end)
        snapshot_ticks = [freeze_end]
        
        # Add all kill event ticks for this round
        kill_ticks = round_kills['tick'].tolist()
        snapshot_ticks.extend(kill_ticks)
        
        # Remove duplicates and sort
        snapshot_ticks = sorted(set(snapshot_ticks))
        
        # Process each snapshot tick
        for current_tick in snapshot_ticks:
            # Skip if tick is after round end
            if current_tick >= end_tick:
                continue
                
            # Calculate time remaining based on game timers 
            round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
            
            # If bomb is planted, calculate bomb timer
            if plant_tick is not None and current_tick >= plant_tick:
                ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
                bomb_planted = True
            else:
                ticks_left = round_ticks_left
                bomb_planted = False

            if ticks_left <= 0:
                continue
            
            # Count deaths up to current tick
            deaths_so_far = round_kills[round_kills['tick'] <= current_tick]
            ct_deaths = len(deaths_so_far[deaths_so_far['victim_side'] == 'ct'])
            t_deaths = len(deaths_so_far[deaths_so_far['victim_side'] == 't'])
            
            # Calculate alive players (start with 5 per team)
            cts_alive = 5 - ct_deaths
            ts_alive = 5 - t_deaths
            
            snapshot = {
                "source": f"Round {round_num} in {parsed_demo_file}",
                "time_left": ticks_left / tick_rate, 
                "cts_alive": cts_alive,
                "ts_alive": ts_alive,
                "bomb_planted": bomb_planted,
                "winner": winner
            }
            
            snapshots.append(snapshot)
    
    # Save to JSON
    if append_mode and os.path.exists(output_file):
        # Load existing snapshots
        try:
            with open(output_file, 'r') as f:
                existing_snapshots = json.load(f)
            snapshots = existing_snapshots + snapshots
        except (json.JSONDecodeError, FileNotFoundError):
            # If file doesn't exist or is invalid, just use new snapshots
            pass
    
    with open(output_file, 'w') as f:
        json.dump(snapshots, f, indent=2)
    
    print(f"✅ Saved {len(snapshots)} snapshots to {output_file}")
    

if __name__ == "__main__":

    demo_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR)]
    output_file = "all_snapshots.json"

    first_demo = demo_files[0]
    print(f"Processing: {first_demo}")
    extract_snapshots_to_json(str(first_demo), output_file)
    
    # Process remaining demos in append mode with progress bar
    for demo_file in tqdm(demo_files[1:], desc="Processing demos", unit="demo"):
        try:
            extract_snapshots_to_json(str(demo_file), output_file, append_mode=True)
        except Exception as e:
            tqdm.write(f"Error processing {demo_file}: {e}")
            continue
    
    print(f"All demos processed. Results saved to {output_file}")
