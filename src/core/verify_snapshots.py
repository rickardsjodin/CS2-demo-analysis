"""
Standalone script to verify and inspect Parquet snapshot files.
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List
import polars as pl

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the snapshot loading function

def load_snapshots_from_parquet(parquet_file: str) -> List[Dict[str, Any]]:
    """
    Load snapshots from Parquet file and convert back to original nested structure.
    This function provides compatibility with the original JSON format.
    """
    
    try:
        df = pl.read_parquet(parquet_file)
        
        snapshots = []
        
        # Group by snapshot (using source and time_left as unique identifiers)
        grouped = df.group_by(['source', 'time_left'], maintain_order=True)
        
        for group_key, group_df in grouped:
            # Get the first row for base snapshot info
            first_row = group_df.row(0, named=True)
            
            # Reconstruct the snapshot
            snapshot = {
                'source': first_row['source'],
                'time_left': first_row['time_left'],
                'cts_alive': first_row['cts_alive'],
                'ts_alive': first_row['ts_alive'],
                'hp_ct': first_row['hp_ct'],
                'hp_t': first_row['hp_t'],
                'bomb_planted': first_row['bomb_planted'],
                'winner': first_row['winner'],
                'alive_player_details': []
            }
            
            # Reconstruct player data
            for row in group_df.iter_rows(named=True):
                # Find player data in this row (look for player_0_, player_1_, etc.)
                for key in row.keys():
                    if key.startswith('player_') and key.endswith('_inventory'):
                        player_idx = key.split('_')[1]
                        
                        # Check if this player has data
                        inventory_json = row[f'player_{player_idx}_inventory']
                        if inventory_json:
                            player = {
                                'inventory': json.loads(inventory_json),
                                'health': row[f'player_{player_idx}_health'],
                                'has_defuser': row[f'player_{player_idx}_has_defuser'],
                                'has_helmet': row[f'player_{player_idx}_has_helmet'],
                                'armor': row[f'player_{player_idx}_armor'],
                                'place': row[f'player_{player_idx}_place'],
                                'side': row[f'player_{player_idx}_side'],
                                'X': row[f'player_{player_idx}_X'],
                                'Y': row[f'player_{player_idx}_Y'],
                                'Z': row[f'player_{player_idx}_Z'],
                                'tick': row[f'player_{player_idx}_tick'],
                                'steamid': row[f'player_{player_idx}_steamid'],
                                'name': row[f'player_{player_idx}_name'],
                                'round_num': row[f'player_{player_idx}_round_num'],
                            }
                            snapshot['alive_player_details'].append(player)
            
            snapshots.append(snapshot)
        
        return snapshots
        
    except Exception as e:
        print(f"❌ Error loading snapshots from {parquet_file}: {e}")
        return []

def verify_parquet_content(parquet_file: str, show_sample: bool = True, compare_with_json: str = None):
    """
    Verify the content of a Parquet snapshot file and optionally compare with JSON.
    """
    try:
        # Load Parquet file
        df = pl.read_parquet(parquet_file)
        
        print(f"📊 Parquet File Analysis: {parquet_file}")
        print(f"├─ Total rows: {df.height:,}")
        print(f"├─ Total columns: {df.width}")
        print(f"├─ File size: {os.path.getsize(parquet_file) / (1024*1024):.2f} MB")
        
        # Show column info
        print("\n📋 Column Information:")
        print(df.schema)
        
        # Show basic stats
        print("\n📈 Basic Statistics:")
        base_cols = ['cts_alive', 'ts_alive', 'hp_ct', 'hp_t', 'time_left']
        for col in base_cols:
            if col in df.columns:
                col_data = df[col]
                min_val = col_data.min()
                max_val = col_data.max()
                mean_val = col_data.mean()
                print(f"├─ {col}: min={min_val}, max={max_val}, mean={mean_val:.2f}")
        
        # Count unique sources (demos)
        unique_sources = df['source'].n_unique()
        print(f"├─ Unique demo sources: {unique_sources}")
        
        # Show sample data
        if show_sample:
            print("\n📄 Sample Data (first 3 rows):")
            sample_cols = ['source', 'time_left', 'cts_alive', 'ts_alive', 'bomb_planted', 'winner']
            print(df.select(sample_cols).head(3))
        
        # Test conversion back to original format
        print("\n🔄 Testing conversion back to original format...")
        snapshots = load_snapshots_from_parquet(parquet_file)
        print(f"├─ Successfully converted {len(snapshots)} snapshots")
        
        if snapshots:
            sample_snapshot = snapshots[0]
            print(f"├─ First snapshot has {len(sample_snapshot['alive_player_details'])} alive players")
            print(f"├─ Sample player inventory: {sample_snapshot['alive_player_details'][0]['inventory']}")
        
        # Compare with JSON if provided
        if compare_with_json and os.path.exists(compare_with_json):
            print(f"\n🔍 Comparing with JSON file: {compare_with_json}")
            
            with open(compare_with_json, 'r') as f:
                json_snapshots = json.load(f)
            
            json_size = os.path.getsize(compare_with_json) / (1024*1024)
            parquet_size = os.path.getsize(parquet_file) / (1024*1024)
            compression_ratio = json_size / parquet_size if parquet_size > 0 else 0
            
            print(f"├─ JSON snapshots: {len(json_snapshots)}")
            print(f"├─ Parquet snapshots: {len(snapshots)}")
            print(f"├─ JSON size: {json_size:.2f} MB")
            print(f"├─ Parquet size: {parquet_size:.2f} MB")
            print(f"└─ Compression: {compression_ratio:.1f}x smaller")
            
            # Quick data integrity check
            if len(json_snapshots) == len(snapshots):
                print("✅ Snapshot count matches!")
                
                # Compare first snapshot structure
                if snapshots and json_snapshots:
                    parquet_keys = set(snapshots[0].keys())
                    json_keys = set(json_snapshots[0].keys())
                    if parquet_keys == json_keys:
                        print("✅ Snapshot structure matches!")
                    else:
                        print(f"⚠️ Structure mismatch: {parquet_keys.symmetric_difference(json_keys)}")
            else:
                print(f"⚠️ Snapshot count mismatch!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying {parquet_file}: {e}")
        return False

def inspect_specific_snapshot(parquet_file: str, snapshot_index: int = 0):
    """
    Inspect a specific snapshot in detail.
    """
    try:
        snapshots = load_snapshots_from_parquet(parquet_file)
        
        if not snapshots or snapshot_index >= len(snapshots):
            print(f"❌ Invalid snapshot index {snapshot_index}. Available: 0-{len(snapshots)-1}")
            return
        
        snapshot = snapshots[snapshot_index]
        
        print(f"🔍 Detailed Snapshot #{snapshot_index}")
        print(f"├─ Source: {snapshot['source']}")
        print(f"├─ Time left: {snapshot['time_left']:.2f}s")
        print(f"├─ Score: CT:{snapshot['cts_alive']} vs T:{snapshot['ts_alive']}")
        print(f"├─ Health: CT:{snapshot['hp_ct']} vs T:{snapshot['hp_t']}")
        print(f"├─ Bomb planted: {snapshot['bomb_planted']}")
        print(f"├─ Winner: {snapshot['winner']}")
        print(f"└─ Alive players: {len(snapshot['alive_player_details'])}")
        
        print("\n👥 Player Details:")
        for i, player in enumerate(snapshot['alive_player_details']):
            print(f"  Player {i+1}: {player['name']} ({player['side']})")
            print(f"  ├─ Health: {player['health']}, Armor: {player['armor']}")
            print(f"  ├─ Position: ({player['X']:.1f}, {player['Y']:.1f}, {player['Z']:.1f})")
            print(f"  ├─ Location: {player['place']}")
            print(f"  └─ Inventory: {', '.join(player['inventory'])}")
            print()
        
    except Exception as e:
        print(f"❌ Error inspecting snapshot: {e}")

def main():
    """Interactive verification of snapshot files."""
    
    # Default paths
    data_dir = "data/datasets"
    parquet_file = os.path.join(data_dir, "all_snapshots_optimized.parquet")
    json_file = os.path.join(data_dir, "snapshots.json")
    
    if not os.path.exists(parquet_file):
        print(f"❌ Parquet file not found: {parquet_file}")
        return
    
    print("🔍 CS2 Snapshot Verification Tool")
    print("=" * 50)
    
    # Basic verification
    print("\n1️⃣ Basic File Verification:")
    success = verify_parquet_content(
        parquet_file, 
        show_sample=True, 
        compare_with_json=json_file if os.path.exists(json_file) else None
    )
    
    if not success:
        return
    
    # Interactive inspection
    while True:
        print("\n" + "=" * 50)
        print("What would you like to do?")
        print("1. Inspect specific snapshot")
        print("2. Re-run verification")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            try:
                index = int(input("Enter snapshot index (0-based): "))
                inspect_specific_snapshot(parquet_file, index)
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "2":
            verify_parquet_content(parquet_file, show_sample=True, compare_with_json=json_file if os.path.exists(json_file) else None)
        
        elif choice == "3":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()