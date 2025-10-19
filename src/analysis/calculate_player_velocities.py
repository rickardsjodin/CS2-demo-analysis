"""
Calculate player velocities for each round in a CS2 demo file.
Minimal script with minimal error handling.
"""

import os
from pathlib import Path
from awpy import Demo
import polars as pl
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import hashlib

from src.core.snapshot_extractor_optimized import sort_demo_paths_by_dir_number_desc
from src.utils.cache_utils import load_demo

# File to store player velocity data
PLAYER_VELOCITY_DATA_FILE = Path("player_velocity_data.json")

# Top teams to prioritize (based on HLTV rankings)
TOP_TEAMS = [
    'vitality','spirit','falcons',   'mongolz', 'mouz', 
    'furia', 'g2', 'aurora', 'natus-vincere', 'navi', '3dmax', 
    'faze', 'astralis', 'liquid', 'heroic', 'complexity', 
    'eternal-fire', 'saw', 'virtus-pro', 'big'
]

START_OFFSET_SECONDS = 20

def get_team_priority(demo_file):
    """Get priority score for a demo based on team names in filename."""
    filename_lower = Path(demo_file).name.lower()
    for i, team in enumerate(TOP_TEAMS):
        if team in filename_lower:
            return i  # Lower number = higher priority
    return len(TOP_TEAMS)  # Lowest priority for non-top teams


def calculate_velocity(x_vel, y_vel, z_vel):
    """Calculate velocity magnitude from velocity components."""
    return math.sqrt(x_vel**2 + y_vel**2 + z_vel**2)


def get_demo_hash(demo_file):
    """Get a unique hash for a demo file based on its path and modification time."""
    demo_path = Path(demo_file)
    mtime = demo_path.stat().st_mtime
    size = demo_path.stat().st_size
    hash_input = f"{demo_path.name}_{mtime}_{size}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def load_player_velocity_data():
    """Load existing player velocity data from file."""
    if PLAYER_VELOCITY_DATA_FILE.exists():
        try:
            with open(PLAYER_VELOCITY_DATA_FILE, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded existing player velocity data ({len(data.get('player_info', {}))} players, {len(data.get('processed_demos', []))} demos)")
            return data
        except Exception as e:
            print(f"⚠️ Error loading player velocity data: {e}")
            return {'player_info': {}, 'processed_demos': []}
    return {'player_info': {}, 'processed_demos': []}


def save_player_velocity_data(data):
    """Save player velocity data to file."""
    try:
        with open(PLAYER_VELOCITY_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved player velocity data to {PLAYER_VELOCITY_DATA_FILE}")
    except Exception as e:
        print(f"❌ Error saving player velocity data: {e}")


def merge_player_data(existing_data, new_data):
    """Merge new player data into existing data."""
    for steam_id, new_info in new_data.items():
        if steam_id not in existing_data:
            existing_data[steam_id] = {
                'name': new_info['name'],
                'total_vel_xy': 0,
                'total_time_alive': 0,
                'round_count': 0
            }
        
        existing_data[steam_id]['name'] = new_info['name']  # Update name in case it changed
        existing_data[steam_id]['total_vel_xy'] += new_info['total_vel_xy']
        existing_data[steam_id]['total_time_alive'] += new_info['total_time_alive']
        existing_data[steam_id]['round_count'] += new_info['round_count']
    
    return existing_data


def calculate_velocities_by_round(demo_file, tick_rate=64):
    """
    Open a demo and calculate velocities for each player in each round.
    Velocity is calculated by comparing positions between consecutive ticks.
    
    Args:
        demo_file: Path to the .dem file
        tick_rate: Ticks per second (default 64)
        
    Returns:
        Tuple of (round_velocities dict, rounds_df, ticks_df with velocities)
    """
    # Parse the demo
    print(f"Parsing demo: {demo_file}")
    dem = load_demo(demo_file, use_cache=True)
    
    # Get ticks, rounds, and kills data
    ticks_df = dem['ticks']
    rounds_df = dem['rounds']
    kills_df = dem['kills']
    
    print(f"Total ticks: {len(ticks_df)}")
    print(f"Total rounds: {len(rounds_df)}")
    print(f"Total kills: {len(kills_df)}")
    
    # Sort by player name and tick to ensure correct ordering
    ticks_df = ticks_df.sort(['name', 'tick'])
    
    # Mark death ticks for each player in each round
    # Create a death lookup: for each (round, player), get the tick they died
    deaths = kills_df.select([
        pl.col('round_num'),
        pl.col('victim_name').alias('name'),
        pl.col('tick').alias('death_tick')
    ])
    
    # Join death information to ticks
    ticks_df = ticks_df.join(
        deaths,
        on=['round_num', 'name'],
        how='left'
    )
    
    # Mark ticks as alive (before death) or dead (at or after death)
    ticks_df = ticks_df.with_columns([
        pl.when(pl.col('death_tick').is_null())
        .then(True)
        .when(pl.col('tick') < pl.col('death_tick'))
        .then(True)
        .otherwise(False)
        .alias('is_alive')
    ])
    
    # Calculate velocity by comparing consecutive ticks for each player
    ticks_df = ticks_df.with_columns([
        pl.col('X').shift(1).over('name').alias('X_prev'),
        pl.col('Y').shift(1).over('name').alias('Y_prev'),
        pl.col('Z').shift(1).over('name').alias('Z_prev'),
        pl.col('tick').shift(1).over('name').alias('tick_prev'),
    ])
    
    ticks_df = ticks_df.with_columns([
        ((pl.col('X') - pl.col('X_prev'))**2 + 
         (pl.col('Y') - pl.col('Y_prev'))**2 + 
         (pl.col('Z') - pl.col('Z_prev'))**2).sqrt().alias('distance'),
        ((pl.col('X') - pl.col('X_prev'))**2 + 
         (pl.col('Y') - pl.col('Y_prev'))**2).sqrt().alias('distance_xy'),
        (pl.col('tick') - pl.col('tick_prev')).alias('tick_diff')
    ])
    
    ticks_df = ticks_df.with_columns([
        (pl.col('distance') / (pl.col('tick_diff') / tick_rate)).alias('velocity'),
        (pl.col('distance_xy') / (pl.col('tick_diff') / tick_rate)).alias('velocity_xy')
    ])
    
    # Filter out invalid velocities (inf, -inf from division by zero when tick_diff is 0)
    ticks_df = ticks_df.with_columns([
        pl.when((pl.col('velocity').is_finite()) & (pl.col('velocity_xy').is_finite()))
        .then(pl.col('velocity'))
        .otherwise(None)
        .alias('velocity'),
        pl.when((pl.col('velocity').is_finite()) & (pl.col('velocity_xy').is_finite()))
        .then(pl.col('velocity_xy'))
        .otherwise(None)
        .alias('velocity_xy')
    ])
    
    round_velocities = {}
    
    # Process each round
    for round_row in rounds_df.iter_rows(named=True):
        round_num = round_row['round_num']
        start_tick = round_row['freeze_end']
        if start_tick is None:
            continue
        
        start_tick = start_tick + int(START_OFFSET_SECONDS * tick_rate) # Start a while into the round

        end_tick = round_row['end']
        
        # Filter ticks for this round - only include ticks where player is alive
        round_ticks = ticks_df.filter(
            (pl.col('tick') >= start_tick) & 
            (pl.col('tick') <= end_tick) &
            (pl.col('velocity').is_not_null()) &
            (pl.col('is_alive') == True)
        )
        
        # Group by player and calculate statistics
        player_velocities = round_ticks.group_by('name').agg([
            pl.col('velocity').mean().alias('avg_velocity'),
            pl.col('velocity').max().alias('max_velocity'),
            pl.col('velocity').min().alias('min_velocity'),
            pl.col('velocity_xy').mean().alias('avg_velocity_xy'),
            pl.col('velocity_xy').max().alias('max_velocity_xy'),
            pl.col('velocity_xy').min().alias('min_velocity_xy'),
            pl.col('velocity').count().alias('num_ticks'),
            ((pl.col('tick').max() - pl.col('tick').min()) / tick_rate).alias('time_alive')
        ])
        
        round_velocities[round_num] = player_velocities
    
    return round_velocities, rounds_df, ticks_df


def plot_round_velocities(rounds_df, ticks_df, round_num):
    """Plot velocity over time for all players in a round."""
    round_data = rounds_df.filter(pl.col('round_num') == round_num)
    start_tick = round_data['freeze_end'][0]
    end_tick = round_data['end'][0]
    
    round_ticks = ticks_df.filter(
        (pl.col('tick') >= start_tick) & 
        (pl.col('tick') <= end_tick) &
        (pl.col('velocity').is_not_null()) &
        (pl.col('is_alive') == True)
    )
    
    # Get unique players
    players = round_ticks['name'].unique().to_list()
    
    plt.figure(figsize=(14, 8))
    
    for player in players:
        player_data = round_ticks.filter(pl.col('name') == player).to_pandas()
        plt.plot(player_data['tick'], player_data['velocity_xy'], label=player, alpha=0.7)
    
    plt.xlabel('Tick')
    plt.ylabel('Velocity (units/second)')
    plt.title(f'Player Velocities - Round {round_num}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    dem_dir = Path("G:\\CS2\\demos")
    
    if not os.path.exists(dem_dir):
        print(f"Demo directory not found at {dem_dir}")

    # Load existing data
    velocity_data = load_player_velocity_data()
    player_info_by_id = velocity_data['player_info']
    processed_demos = set(velocity_data['processed_demos'])
    
    # Recursively collect all .dem files in cache_path and its subfolders
    demo_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dem_dir)
        for file in files
        if file.endswith('.dem')
    ]
    # Sort by directory number first, then by team priority

    with open('player_velocity_data_v1.json', 'r') as f:
        data_v1 = json.load(f)

    data_v1_hashes = data_v1['processed_demos']
    
    demo_files = sort_demo_paths_by_dir_number_desc(demo_files)
    # demo_files = sorted(demo_files, key=lambda x: (get_team_priority(x), -os.path.getmtime(x)))[:80]

    
    # Filter out already processed demos
    demos_to_process = []
    for demo_file in demo_files:
        demo_hash = get_demo_hash(demo_file)
        if demo_hash not in processed_demos and demo_hash in data_v1_hashes:
            demos_to_process.append(demo_file)
    
    print(f"📊 Total demos: {len(demo_files)}")
    print(f"✅ Already processed: {len(demo_files) - len(demos_to_process)}")
    print(f"🆕 To process: {len(demos_to_process)}")
    print(f"\n⚠️  Press Ctrl+C at any time to pause and save progress\n")

    new_player_data = {} # Temporary storage for new player data

    try:
        for demo_file in tqdm(demos_to_process, desc="Processing demos"):
            demo_hash = get_demo_hash(demo_file)
            
            try:
                # Calculate velocities for all rounds
                velocities, rounds_df, ticks_df = calculate_velocities_by_round(demo_file)
                
                # Aggregate player stats across all rounds in this demo
                for round_num, player_data in velocities.items():
                
                    for row in player_data.iter_rows(named=True):
                        player_name = row['name']
                        avg_vel_xy = row['avg_velocity_xy']
                        time_alive = row['time_alive']
                        
                        # Skip if velocity is NaN or null
                        if avg_vel_xy is None or (isinstance(avg_vel_xy, float) and math.isnan(avg_vel_xy)):
                            continue
                        
                        # Get steam_id for this player from ticks_df
                        player_ticks = ticks_df.filter(pl.col('name') == player_name)
                        if len(player_ticks) > 0:
                            steam_id = str(player_ticks['steamid'][0])  # Convert to string for JSON
                            
                            # Initialize player entry if not exists
                            if steam_id not in new_player_data:
                                new_player_data[steam_id] = {
                                    'name': player_name,
                                    'total_vel_xy': 0,
                                    'total_time_alive': 0,
                                    'round_count': 0
                                }
                            
                            # Accumulate statistics
                            new_player_data[steam_id]['total_vel_xy'] += avg_vel_xy
                            new_player_data[steam_id]['total_time_alive'] += time_alive
                            new_player_data[steam_id]['round_count'] += 1
            except Exception as e:
                print("Skipping demo: " + str(e))
                continue

            # Mark demo as processed
            processed_demos.add(demo_hash)
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Paused by user! Saving progress...")
    
    # Merge new data with existing data
    player_info_by_id = merge_player_data(player_info_by_id, new_player_data)
    
    # Save updated data
    velocity_data = {
        'player_info': player_info_by_id,
        'processed_demos': list(processed_demos)
    }
    save_player_velocity_data(velocity_data)
    
    # Calculate final averages for display
    display_data = {}
    for steam_id, info in player_info_by_id.items():
        display_data[steam_id] = {
            'name': info['name'],
            'avg_vel_xy': info['total_vel_xy'] / info['round_count'],
            'avg_time_alive': info['total_time_alive'] / info['round_count'],
            'round_count': info['round_count']
        }
    
    # Print results
    print("\n=== Player Statistics Across All Demos ===")
    for steam_id, info in sorted(display_data.items(), key=lambda x: x[1]['round_count'], reverse=True)[:20]:
        print(f"{info['name']} (Steam ID: {steam_id}):")
        print(f"  Avg XY Velocity: {info['avg_vel_xy']:.2f} units/s")
        print(f"  Avg Time Alive: {info['avg_time_alive']:.2f} seconds")
        print(f"  Round Samples: {info['round_count']}")
    
    # Get top 10 players by round count
    top_10_players = sorted(display_data.items(), key=lambda x: x[1]['round_count'], reverse=True)[:80]
    
    # Create scatter plot
    plt.figure(figsize=(14, 10))
    
    for steam_id, info in top_10_players:
        if info['name'] in ['Ex3rcice', 'MUTiRiS']:
            continue
        plt.scatter(info['avg_vel_xy'], info['avg_time_alive'], 
                   s=10, alpha=0.6)
        # Add text label next to each point
        plt.text(info['avg_vel_xy'] + 0.2, info['avg_time_alive'] - 0.1, 
                info['name'], fontsize=8, alpha=0.8)
    
    plt.xlabel('Average XY Velocity (units/second)')
    plt.ylabel('Average Time Alive (seconds)')
    plt.title('Top 20 Players: XY Velocity vs Time Alive\n(Bubble size = number of round samples)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
