"""
CS2 Snapshot Extractor - Production Version
Minimal, concise code for extracting tick-based snapshots for ML training
"""

import json
import os
import pandas as pd
import polars as pl
from cache_utils import load_demo

ROUND_TIME = 115  
BOMB_TIME = 40   

def extract_snapshots_to_json(demo_file: str, output_file: str = "snapshots.json", tick_interval=200, tick_rate=64, append_mode=False):
    """Extract snapshots with time_left and alive counts from kills data
    
    Args:
        demo_file: Path to the demo file
        output_file: Output JSON file path
        tick_interval: Tick interval for snapshots
        tick_rate: Game tick rate  
        append_mode: If True, append to existing file instead of overwriting
    """
    
    # Load demo
    demo = load_demo(demo_file, use_cache=True)
    
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

        current_tick = freeze_end
        while current_tick < end_tick:
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
                current_tick += tick_interval
                continue
            
            # Count deaths up to current tick
            deaths_so_far = round_kills[round_kills['tick'] <= current_tick]
            ct_deaths = len(deaths_so_far[deaths_so_far['victim_side'] == 'ct'])
            t_deaths = len(deaths_so_far[deaths_so_far['victim_side'] == 't'])
            
            # Calculate alive players (start with 5 per team)
            cts_alive = 5 - ct_deaths
            ts_alive = 5 - t_deaths
            
            snapshot = {
                "source": f"Round {round_num} in {demo_file}",
                "time_left": ticks_left / tick_rate, 
                "cts_alive": cts_alive,
                "ts_alive": ts_alive,
                "bomb_planted": bomb_planted,
                "winner": winner
            }
            
            snapshots.append(snapshot)
            current_tick += tick_interval
    
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
    extract_snapshots_to_json("F:\\CS2\\demos\\8039\\TYLOO_vs._Vitality_at_Esports_World_Cup_2025__Inferno_Nuke_Overpass__demo_99354_tyloo-vs-vitality-m2-nuke.dem")
