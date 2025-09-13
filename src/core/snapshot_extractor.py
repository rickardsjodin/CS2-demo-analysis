"""
CS2 Snapshot Extractor - Production Version
Refactored for improved readability, scalability, and maintainability.
Extracts tick-based snapshots from parsed CS2 demo files for machine learning training.
"""

import json
import os
import pickle
import polars as pl
from tqdm import tqdm
from typing import List, Dict, Any, Optional

try:
    from ..utils.cache_utils import CACHE_DIR
    from .constants import ROUND_TIME, BOMB_TIME, WEAPON_TIERS, GRENADE_TYPES
except ImportError:
    # Handle case when running directly for testing or standalone execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.cache_utils import CACHE_DIR
    from core.constants import ROUND_TIME, BOMB_TIME, WEAPON_TIERS, GRENADE_TYPES

def load_demo_data(parsed_demo_file: str) -> Optional[Dict[str, pl.DataFrame]]:
    """Loads and prepares data from a parsed demo file."""
    try:
        with open(parsed_demo_file, 'rb') as f:
            cache_data = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        tqdm.write(f"Error loading {parsed_demo_file}: {e}")
        return None

    data = {}
    for name in ['kills', 'rounds', 'ticks']:
        df = cache_data.get(name)
        if df is None:
            tqdm.write(f"Warning: Missing '{name}' data in {parsed_demo_file}")
            return None
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        data[name] = df
    return data

def calculate_player_stats(alive_players: pl.DataFrame) -> Dict[str, int]:
    """Calculates equipment and stats for alive players at a specific tick."""
    stats = {
        "ct_main_weapons": 0, "t_main_weapons": 0,
        "ct_grenades": 0, "t_grenades": 0,
        "ct_helmets": 0, "t_helmets": 0,
        "ct_armor": 0, "t_armor": 0,
        "defusers": 0
    }

    for player_row in alive_players.iter_rows(named=True):
        inventory = player_row.get('inventory', [])
        player_side = player_row['side']
        
        best_weapon_tier = 0
        if inventory:
            for item_name in inventory:
                tier = WEAPON_TIERS.get(item_name)
                if tier is not None:
                    if tier > best_weapon_tier:
                        best_weapon_tier = tier
                elif (item_name not in GRENADE_TYPES) and ("Knife" not in item_name):
                    tqdm.write(f"Warning: Unknown weapon '{item_name}' not in WEAPON_TIERS.")
        
        if best_weapon_tier >= 5:
            if player_side == 'ct':
                stats["ct_main_weapons"] += 1
            else:
                stats["t_main_weapons"] += 1
        
        grenade_count = sum(1 for item in inventory if item in GRENADE_TYPES)
        armor = player_row.get('armor', 0) or 0

        if player_side == 'ct':
            stats["ct_grenades"] += grenade_count
            stats["ct_helmets"] += 1 if player_row.get('has_helmet') else 0
            stats["ct_armor"] += armor
        else:
            stats["t_grenades"] += grenade_count
            stats["t_helmets"] += 1 if player_row.get('has_helmet') else 0
            stats["t_armor"] += armor

        stats["defusers"] += 1 if player_row.get('has_defuser') else 0
        
    return stats

def create_snapshot(
    current_tick: int, round_row: Dict, demo_data: Dict, tick_rate: int, file_name: str
) -> Optional[Dict[str, Any]]:
    """Creates a single snapshot for a given tick."""
    
    freeze_end = round_row['freeze_end']
    end_tick = round_row['end']
    plant_tick = round_row['bomb_plant']
    
    if current_tick >= end_tick:
        return None

    round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
    
    if plant_tick is not None and current_tick >= plant_tick:
        ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
        bomb_planted = True
    else:
        ticks_left = round_ticks_left
        bomb_planted = False

    if ticks_left <= 0:
        return None

    round_kills = demo_data['kills'].filter(pl.col('round_num') == round_row['round_num'])
    deaths_so_far = round_kills.filter(pl.col('tick') <= current_tick)
    
    ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
    t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height
    
    current_details = demo_data['ticks'].filter(pl.col('tick') == current_tick)

    hp_t = current_details.filter(pl.col('side') == 't')['health'].sum()
    hp_ct = current_details.filter(pl.col('side') == 'ct')['health'].sum()

    player_stats = {}
    if 'inventory' in current_details.columns:
        dead_player_steamids = deaths_so_far['victim_steamid'].unique().to_list()
        alive_players = current_details.filter(~pl.col('steamid').is_in(dead_player_steamids))
        player_stats = calculate_player_stats(alive_players)

    return {
        "source": f"Round {round_row['round_num']} in {file_name}",
        "time_left": ticks_left / tick_rate,
        "cts_alive": 5 - ct_deaths,
        "ts_alive": 5 - t_deaths,
        "hp_ct": int(hp_ct) if hp_ct is not None else 0,
        "hp_t": int(hp_t) if hp_t is not None else 0,
        "bomb_planted": bomb_planted,
        "winner": round_row['winner'],
        **player_stats
    }

def extract_snapshots_from_demo(parsed_demo_file: str, tick_rate=64) -> List[Dict[str, Any]]:
    """Extracts snapshots from a single parsed demo file."""
    
    demo_data = load_demo_data(parsed_demo_file)
    if not demo_data:
        return []

    snapshots = []
    file_name = os.path.basename(parsed_demo_file)
    
    for round_row in demo_data['rounds'].iter_rows(named=True):
        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        
        if freeze_end is None or round_row['end'] is None:
            tqdm.write(f"Skipping round {round_num} in {file_name} due to missing start/end ticks.")
            continue

        round_kills = demo_data['kills'].filter(pl.col('round_num') == round_num)
        
        snapshot_ticks = sorted(set([freeze_end] + round_kills['tick'].to_list()))
        
        for current_tick in snapshot_ticks:
            snapshot = create_snapshot(current_tick, round_row, demo_data, tick_rate, file_name)
            if snapshot:
                snapshots.append(snapshot)
            
    return snapshots

def extract_snapshots_to_json(demo_files: List[str], output_file: str = "snapshots.json", tick_rate=64):
    """Extract snapshots from multiple demo files and save to a single JSON file."""
    
    all_snapshots = []
    
    for demo_file in tqdm(demo_files, desc="Processing demos", unit="demo"):
        try:
            snapshots = extract_snapshots_from_demo(demo_file, tick_rate)
            all_snapshots.extend(snapshots)
        except Exception as e:
            tqdm.write(f"Error processing {demo_file}: {e}")

    try:
        with open(output_file, 'w') as f:
            json.dump(all_snapshots, f, indent=2)
        print(f"✅ Saved {len(all_snapshots)} snapshots to {output_file}")
    except IOError as e:
        print(f"❌ Error saving snapshots to {output_file}: {e}")


def main():
    """Main function to run the snapshot extraction process."""
    
    cache_path = CACHE_DIR
    
    if not os.path.exists(cache_path):
        print(f"Cache directory not found at {cache_path}")
        return

    demo_files = [os.path.join(cache_path, f) for f in os.listdir(cache_path) if f.endswith('.pkl')]
    
    if not demo_files:
        print("No processed demo files found in cache.")
        return
        
    # It's good practice to specify the output directory
    output_dir = "data/datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_snapshots.json")

    extract_snapshots_to_json(demo_files, output_file)


if __name__ == "__main__":
    main()
