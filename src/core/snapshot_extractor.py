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

def extract_snapshots_from_demo(parsed_demo_file: str, tick_rate=64):
    """Extracts snapshots from a single parsed demo file."""
    
    with open(parsed_demo_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    class CachedDemo:
        def __init__(self, data):
            self.kills = data.get('kills')
            self.rounds = data.get('rounds')
            self.ticks = data.get('ticks')
    
    demo = CachedDemo(cache_data)

    rounds = demo.rounds
    if isinstance(rounds, pd.DataFrame):
        rounds = pl.from_pandas(rounds)
    
    kills = demo.kills
    if isinstance(kills, pd.DataFrame):
        kills = pl.from_pandas(kills)
        
    ticks_details = demo.ticks
    if isinstance(ticks_details, pd.DataFrame):
        ticks_details = pl.from_pandas(ticks_details)

    weapon_tiers = {
        "Karambit": 0, "Butterfly Knife": 0, "M9 Bayonet": 0, "Skeleton Knife": 0, 
        "Stiletto Knife": 0, "Nomad Knife": 0, "Knife": 0,
        "Glock-18": 1, "USP-S": 1, "P2000": 1, "P250": 1, "Tec-9": 1, "CZ75-Auto": 1, 
        "Five-SeveN": 1, "Desert Eagle": 2, "R8 Revolver": 2,
        "Nova": 3, "XM1014": 3, "Sawed-Off": 3, "MAG-7": 3,
        "MAC-10": 4, "MP9": 4, "MP7": 4, "MP5-SD": 4, "UMP-45": 4, "P90": 4, "PP-Bizon": 4,
        "Galil AR": 5, "FAMAS": 5, "AK-47": 6, "M4A4": 6, "M4A1-S": 6, "SG 553": 6, "AUG": 6,
        "SSG 08": 7, "AWP": 8, "G3SG1": 7, "SCAR-20": 7,
        "M249": 5, "Negev": 5
    }
    
    grenade_types = [
        "High Explosive Grenade", "Flashbang", "Smoke Grenade", "Molotov", 
        "Incendiary Grenade", "Decoy Grenade", "C4 Explosive"
    ]

    snapshots = []
    
    for round_row in rounds.iter_rows(named=True):
        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        end_tick = round_row['end']
        winner = round_row['winner']
        plant_tick = round_row['bomb_plant']
        
        if freeze_end is None or end_tick is None:
            tqdm.write(f"Skipping round {round_num} in {os.path.basename(parsed_demo_file)} due to missing start/end ticks.")
            continue

        round_kills = kills.filter(pl.col('round_num') == round_num)
        
        snapshot_ticks = [freeze_end]
        snapshot_ticks.extend(round_kills['tick'].to_list())
        snapshot_ticks = sorted(set(snapshot_ticks))
        
        for current_tick in snapshot_ticks:
            if current_tick >= end_tick:
                continue
                
            round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
            
            if plant_tick is not None and current_tick >= plant_tick:
                ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
                bomb_planted = True
            else:
                ticks_left = round_ticks_left
                bomb_planted = False

            if ticks_left <= 0:
                continue
            
            deaths_so_far = round_kills.filter(pl.col('tick') <= current_tick)
            ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
            t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height
            
            cts_alive = 5 - ct_deaths
            ts_alive = 5 - t_deaths

            current_details = ticks_details.filter(pl.col('tick') == current_tick)

            hp_t = current_details.filter(pl.col('side') == 't')['health'].sum()
            hp_ct = current_details.filter(pl.col('side') == 'ct')['health'].sum()

            ct_main_weapons = 0
            t_main_weapons = 0
            ct_grenades = 0
            t_grenades = 0

            if 'inventory' in current_details.columns:
                dead_player_steamids = deaths_so_far['victim_steamid'].unique().to_list()
                alive_players = current_details.filter(~pl.col('steamid').is_in(dead_player_steamids))

                for player_row in alive_players.iter_rows(named=True):
                    inventory = player_row['inventory']
                    player_side = player_row['side']
                    
                    best_weapon_tier = 0
                    if inventory:
                        for item_name in inventory:
                            if item_name in weapon_tiers:
                                tier = weapon_tiers[item_name]
                                if tier > best_weapon_tier:
                                    best_weapon_tier = tier
                    
                    if best_weapon_tier >= 5:
                        if player_side == 'ct':
                            ct_main_weapons += 1
                        else:
                            t_main_weapons += 1
                    
                    if inventory:
                        grenade_count = sum(1 for item in inventory if item in grenade_types)
                        if player_side == 'ct':
                            ct_grenades += grenade_count
                        else:
                            t_grenades += grenade_count
            
            snapshot = {
                "source": f"Round {round_num} in {os.path.basename(parsed_demo_file)}",
                "time_left": ticks_left / tick_rate, 
                "cts_alive": cts_alive,
                "ts_alive": ts_alive,
                "hp_t": int(hp_t) if hp_t is not None else 0,
                "hp_ct": int(hp_ct) if hp_ct is not None else 0,
                "bomb_planted": bomb_planted,
                "ct_main_weapons": ct_main_weapons,
                "t_main_weapons": t_main_weapons,
                "ct_grenades": ct_grenades,
                "t_grenades": t_grenades,
                "winner": winner
            }
            
            snapshots.append(snapshot)
            
    return snapshots

def extract_snapshots_to_json(demo_files: list, output_file: str = "snapshots.json", tick_rate=64):
    """Extract snapshots from multiple demo files and save to a single JSON file."""
    
    all_snapshots = []
    
    for demo_file in tqdm(demo_files, desc="Processing demos", unit="demo"):
        try:
            snapshots = extract_snapshots_from_demo(str(demo_file), tick_rate)
            all_snapshots.extend(snapshots)
        except Exception as e:
            tqdm.write(f"Error processing {demo_file}: {e}")
            continue

    with open(output_file, 'w') as f:
        json.dump(all_snapshots, f, indent=2)
    
    print(f"✅ Saved {len(all_snapshots)} snapshots to {output_file}")


if __name__ == "__main__":

    demo_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
    output_file = "all_snapshots.json"

    if demo_files:
        extract_snapshots_to_json(demo_files, output_file)
    else:
        print("No processed demo files found in cache.")
