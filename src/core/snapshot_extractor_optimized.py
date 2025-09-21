"""
CS2 Snapshot Extractor - Storage Optimized Version
Uses Parquet format instead of JSON for better compression and performance.
Maintains all the same data structure and information as the original extractor.
"""

import os
import pickle
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import json

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.cache_utils import get_cache_filename, load_demo

try:
    from ..utils.cache_utils import CACHE_DIR
    from .constants import ROUND_TIME, BOMB_TIME, WEAPON_TIERS, MOLOTOV_NADE, HE_NADE, SMOKE_NADE, FLASH_NADE, GRENADE_AND_BOMB_TYPES
except ImportError:
    # Handle case when running directly for testing or standalone execution
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.cache_utils import CACHE_DIR
    from core.constants import ROUND_TIME, BOMB_TIME, WEAPON_TIERS, MOLOTOV_NADE, HE_NADE, SMOKE_NADE, FLASH_NADE, GRENADE_AND_BOMB_TYPES

def load_demo_data(parsed_demo_file: str) -> Optional[Dict[str, pl.DataFrame]]:
    """Loads and prepares data from a parsed demo file."""
    if '.dem' in parsed_demo_file:
        load_demo(parsed_demo_file)
        parsed_demo_file = get_cache_filename(parsed_demo_file)
        
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

def create_snapshot(
    current_tick: int, round_row: Dict, demo_data: Dict, tick_rate: int, file_name: str
) -> Optional[Dict[str, Any]]:
    """Creates a single snapshot for a given tick."""
    
    freeze_end = round_row['freeze_end']
    end_tick = round_row['end']
    plant_tick = round_row['bomb_plant']


    if freeze_end is None and round_row['round_num'] == 1:
        freeze_end = 200
    
    # Check for missing required data
    if freeze_end is None or end_tick is None:
        return None
    

    round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
    
    if plant_tick is not None and current_tick >= plant_tick:
        ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
        bomb_planted = True
    else:
        ticks_left = round_ticks_left
        bomb_planted = False

    round_kills = demo_data['kills'].filter(pl.col('round_num') == round_row['round_num'])
    deaths_so_far = round_kills.filter(pl.col('tick') <= current_tick)
    
    ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
    t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height
    
    current_details = demo_data['ticks'].filter(pl.col('tick') == current_tick)

    hp_t = current_details.filter(pl.col('side') == 't')['health'].sum()
    hp_ct = current_details.filter(pl.col('side') == 'ct')['health'].sum()

    dead_player_steamids = deaths_so_far['victim_steamid'].unique().to_list()
    alive_players = current_details.filter(~pl.col('steamid').is_in(dead_player_steamids))

    alive_player_info = []
    for player_row in alive_players.iter_rows(named=True):
        alive_player_info.append(player_row)

    return {
        "source": f"Round {round_row['round_num']} in {file_name}",
        "time_left": ticks_left / tick_rate,
        "cts_alive": 5 - ct_deaths,
        "ts_alive": 5 - t_deaths,
        "hp_ct": int(hp_ct) if hp_ct is not None else 0,
        "hp_t": int(hp_t) if hp_t is not None else 0,
        "bomb_planted": bomb_planted,
        "winner": round_row['winner'],
        "after_end": current_tick > end_tick,
        "alive_player_details": alive_player_info
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

        if freeze_end is None and round_num == 1:
            freeze_end = 200
        
        if freeze_end is None or round_row['end'] is None:
            tqdm.write(f"Skipping round {round_num} in {file_name} due to missing start/end ticks.")
            continue

        round_kills = demo_data['kills'].filter(pl.col('round_num') == round_num)
        
        snapshot_ticks = sorted(set([freeze_end] + round_kills['tick'].to_list()))
        
        for current_tick in snapshot_ticks:
            snapshot = create_snapshot(current_tick, round_row, demo_data, tick_rate, file_name)
            if snapshot:
                snapshots.append(snapshot)
            else:
                tqdm.write(f"Snapshot is None in {file_name}")
            
    return snapshots

def normalize_snapshots_for_parquet(snapshots: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Converts nested snapshot data into a flat structure suitable for Parquet storage.
    One row per snapshot with all 10 player slots (filled with None if not used).
    """
    
    normalized_data = []
    
    for snapshot in snapshots:
        # Start with base snapshot info
        record = {
            'source': snapshot['source'],
            'time_left': snapshot['time_left'],
            'cts_alive': snapshot['cts_alive'],
            'ts_alive': snapshot['ts_alive'],
            'hp_ct': snapshot['hp_ct'],
            'hp_t': snapshot['hp_t'],
            'bomb_planted': snapshot['bomb_planted'],
            'winner': snapshot['winner'],
            'after_end': snapshot.get('after_end', False),
        }
        
        # Add all 10 player slots (CS2 max 10 players)
        for i in range(10):
            if i < len(snapshot['alive_player_details']):
                player = snapshot['alive_player_details'][i]
                record.update({
                    f'player_{i}_inventory': json.dumps(player.get('inventory', [])),
                    f'player_{i}_health': player.get('health'),
                    f'player_{i}_has_defuser': player.get('has_defuser'),
                    f'player_{i}_has_helmet': player.get('has_helmet'),
                    f'player_{i}_armor': player.get('armor'),
                    f'player_{i}_place': player.get('place'),
                    f'player_{i}_side': player.get('side'),
                    f'player_{i}_X': player.get('X'),
                    f'player_{i}_Y': player.get('Y'),
                    f'player_{i}_Z': player.get('Z'),
                    f'player_{i}_tick': player.get('tick'),
                    f'player_{i}_steamid': player.get('steamid'),
                    f'player_{i}_name': player.get('name'),
                    f'player_{i}_round_num': player.get('round_num'),
                })
            else:
                # Fill empty slots with None to maintain consistent schema
                record.update({
                    f'player_{i}_inventory': None,
                    f'player_{i}_health': None,
                    f'player_{i}_has_defuser': None,
                    f'player_{i}_has_helmet': None,
                    f'player_{i}_armor': None,
                    f'player_{i}_place': None,
                    f'player_{i}_side': None,
                    f'player_{i}_X': None,
                    f'player_{i}_Y': None,
                    f'player_{i}_Z': None,
                    f'player_{i}_tick': None,
                    f'player_{i}_steamid': None,
                    f'player_{i}_name': None,
                    f'player_{i}_round_num': None,
                })
        
        normalized_data.append(record)
    
    return pl.DataFrame(normalized_data)

def extract_snapshots_to_parquet(demo_files: List[str], output_file: str = "snapshots.parquet", tick_rate=64):
    """Extract snapshots from multiple demo files and save to a Parquet file."""
    
    all_snapshots = []
    
    for demo_file in tqdm(demo_files, desc="Processing demos", unit="demo"):
        try:
            snapshots = extract_snapshots_from_demo(demo_file, tick_rate)
            all_snapshots.extend(snapshots)
        except Exception as e:
            tqdm.write(f"Error processing {demo_file}: {e}")

    if not all_snapshots:
        print("No snapshots extracted.")
        return

    try:
        # Convert to DataFrame for Parquet storage
        df = normalize_snapshots_for_parquet(all_snapshots)
        
        # Save to Parquet with compression
        df.write_parquet(
            output_file,
            compression="snappy",  # Good balance of speed and compression
            use_pyarrow=True
        )
        
        print(f"✅ Saved {len(all_snapshots)} snapshots to {output_file}")
        
        # Show file size comparison info
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"📁 File size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving snapshots to {output_file}: {e}")

def load_snapshots_from_parquet(parquet_file: str) -> List[Dict[str, Any]]:
    """
    Load snapshots from Parquet file and convert back to original nested structure.
    This function provides compatibility with the original JSON format.
    """
    
    try:
        df = pl.read_parquet(parquet_file)
        
        snapshots = []
        
        # Each row represents a complete snapshot
        for row in df.iter_rows(named=True):
            snapshot = {
                'source': row['source'],
                'time_left': row['time_left'],
                'cts_alive': row['cts_alive'],
                'ts_alive': row['ts_alive'],
                'hp_ct': row['hp_ct'],
                'hp_t': row['hp_t'],
                'bomb_planted': row['bomb_planted'],
                'winner': row['winner'],
                'alive_player_details': []
            }
            
            # Add after_end if it exists
            if 'after_end' in row:
                snapshot['after_end'] = row['after_end']
            
            # Check all 10 player slots for data
            for i in range(10):
                inventory_key = f'player_{i}_inventory'
                if inventory_key in row and row[inventory_key] is not None:
                    player = {
                        'inventory': json.loads(row[f'player_{i}_inventory']),
                        'health': row[f'player_{i}_health'],
                        'has_defuser': row[f'player_{i}_has_defuser'],
                        'has_helmet': row[f'player_{i}_has_helmet'],
                        'armor': row[f'player_{i}_armor'],
                        'place': row[f'player_{i}_place'],
                        'side': row[f'player_{i}_side'],
                        'X': row[f'player_{i}_X'],
                        'Y': row[f'player_{i}_Y'],
                        'Z': row[f'player_{i}_Z'],
                        'tick': row[f'player_{i}_tick'],
                        'steamid': row[f'player_{i}_steamid'],
                        'name': row[f'player_{i}_name'],
                        'round_num': row[f'player_{i}_round_num'],
                    }
                    snapshot['alive_player_details'].append(player)
            
            snapshots.append(snapshot)
        
        return snapshots
        
    except Exception as e:
        print(f"❌ Error loading snapshots from {parquet_file}: {e}")
        return []

def convert_json_to_parquet(json_file: str, parquet_file: str):
    """Convert existing JSON snapshot file to Parquet format."""
    
    try:
        with open(json_file, 'r') as f:
            snapshots = json.load(f)
        
        df = normalize_snapshots_for_parquet(snapshots)
        df.write_parquet(parquet_file, compression="snappy", use_pyarrow=True)
        
        # Show file size comparison
        json_size = os.path.getsize(json_file) / (1024 * 1024)
        parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)
        compression_ratio = json_size / parquet_size if parquet_size > 0 else 0
        
        print(f"✅ Converted {json_file} to {parquet_file}")
        print(f"📁 JSON size: {json_size:.2f} MB")
        print(f"📁 Parquet size: {parquet_size:.2f} MB")
        print(f"🗜️ Compression ratio: {compression_ratio:.1f}x smaller")
        
    except Exception as e:
        print(f"❌ Error converting {json_file} to Parquet: {e}")

def main():
    """Main function to run the optimized snapshot extraction process."""
    
    cache_path = CACHE_DIR
    
    if not os.path.exists(cache_path):
        print(f"Cache directory not found at {cache_path}")
        return

    demo_files = [os.path.join(cache_path, f) for f in os.listdir(cache_path) if f.endswith('.pkl')]
    if not demo_files:
        print("No processed demo files found in cache.")
        return
        
    # Set up output directory
    output_dir = "data/datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_snapshots_optimized.parquet")

    # Extract snapshots to Parquet format
    extract_snapshots_to_parquet(demo_files, output_file)
    
    # Optional: Convert existing JSON file to Parquet for comparison
    json_file = os.path.join(output_dir, "snapshots.json")
    if os.path.exists(json_file):
        parquet_file = os.path.join(output_dir, "snapshots_converted.parquet")
        convert_json_to_parquet(json_file, parquet_file)

if __name__ == "__main__":
    main()