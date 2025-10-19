"""
Calculate player velocities for each round in a CS2 demo file.
Minimal script with minimal error handling.
"""

import os
from pathlib import Path
import sys
from awpy import Demo
import polars as pl
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import hashlib

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
                't_side': {
                    '5v5': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    '4v4': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    '3v3': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    'all': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0}
                },
                'ct_side': {
                    '5v5': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    '4v4': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    '3v3': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                    'all': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0}
                }
            }
        
        existing_data[steam_id]['name'] = new_info['name']  # Update name in case it changed
        
        # Merge T side stats
        for player_count in ['5v5', '4v4', '3v3', 'all']:
            existing_data[steam_id]['t_side'][player_count]['total_vel_xy'] += new_info['t_side'][player_count]['total_vel_xy']
            existing_data[steam_id]['t_side'][player_count]['total_time_alive'] += new_info['t_side'][player_count]['total_time_alive']
            existing_data[steam_id]['t_side'][player_count]['round_count'] += new_info['t_side'][player_count]['round_count']
        
        # Merge CT side stats
        for player_count in ['5v5', '4v4', '3v3', 'all']:
            existing_data[steam_id]['ct_side'][player_count]['total_vel_xy'] += new_info['ct_side'][player_count]['total_vel_xy']
            existing_data[steam_id]['ct_side'][player_count]['total_time_alive'] += new_info['ct_side'][player_count]['total_time_alive']
            existing_data[steam_id]['ct_side'][player_count]['round_count'] += new_info['ct_side'][player_count]['round_count']
    
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
    
    # Count alive players per side at each tick for player count categories
    player_counts = ticks_df.filter(pl.col('is_alive') == True).group_by(['tick', 'side']).agg([
        pl.col('name').n_unique().alias('player_count')
    ])
    
    # Join player counts back to ticks
    ticks_df = ticks_df.join(
        player_counts,
        on=['tick', 'side'],
        how='left'
    )
    
    # Categorize player count (5v5, 4v4, 3v3, or other)
    ticks_df = ticks_df.with_columns([
        pl.when(pl.col('player_count') == 5)
        .then(pl.lit('5v5'))
        .when(pl.col('player_count') == 4)
        .then(pl.lit('4v4'))
        .when(pl.col('player_count') == 3)
        .then(pl.lit('3v3'))
        .otherwise(pl.lit('other'))
        .alias('player_count_category')
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
        
        # Group by player, side, and player count category
        player_velocities = round_ticks.group_by(['name', 'side', 'player_count_category']).agg([
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

    # with open('player_velocity_data_v1.json', 'r') as f:
    #     data_v1 = json.load(f)

    # data_v1_hashes = data_v1['processed_demos']
    
    demo_files = sort_demo_paths_by_dir_number_desc(demo_files)[:3000]
    demo_files = sorted(demo_files, key=lambda x: (get_team_priority(x), -os.path.getmtime(x)))[:1]

    
    # Filter out already processed demos
    demos_to_process = []
    for demo_file in demo_files:
        demo_hash = get_demo_hash(demo_file)
        if demo_hash not in processed_demos:
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
                        player_side = row['side']  # Get the side (T or CT)
                        player_count_cat = row['player_count_category']  # Get player count (5v5, 4v4, 3v3, other)
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
                                    't_side': {
                                        '5v5': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        '4v4': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        '3v3': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        'all': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0}
                                    },
                                    'ct_side': {
                                        '5v5': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        '4v4': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        '3v3': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0},
                                        'all': {'total_vel_xy': 0, 'total_time_alive': 0, 'round_count': 0}
                                    }
                                }
                            
                            # Determine which side stats to update
                            side_key = 't_side' if player_side.lower() == 't' else 'ct_side'
                            
                            # Always accumulate in 'all' category (includes any player count)
                            new_player_data[steam_id][side_key]['all']['total_vel_xy'] += avg_vel_xy
                            new_player_data[steam_id][side_key]['all']['total_time_alive'] += time_alive
                            new_player_data[steam_id][side_key]['all']['round_count'] += 1
                            
                            # Also accumulate in specific player count category if it's 5v5, 4v4, or 3v3
                            if player_count_cat in ['5v5', '4v4', '3v3']:
                                new_player_data[steam_id][side_key][player_count_cat]['total_vel_xy'] += avg_vel_xy
                                new_player_data[steam_id][side_key][player_count_cat]['total_time_alive'] += time_alive
                                new_player_data[steam_id][side_key][player_count_cat]['round_count'] += 1
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
        t_rounds_all = info['t_side']['all']['round_count']
        ct_rounds_all = info['ct_side']['all']['round_count']
        total_rounds = t_rounds_all + ct_rounds_all
        
        if total_rounds == 0:
            continue
            
        display_data[steam_id] = {
            'name': info['name'],
            'round_count': total_rounds,
            't_side': {},
            'ct_side': {},
            'combined': {}
        }
        
        # Process each player count category for T side
        for pc in ['5v5', '4v4', '3v3', 'all']:
            rounds = info['t_side'][pc]['round_count']
            display_data[steam_id]['t_side'][pc] = {
                'avg_vel_xy': info['t_side'][pc]['total_vel_xy'] / rounds if rounds > 0 else 0,
                'avg_time_alive': info['t_side'][pc]['total_time_alive'] / rounds if rounds > 0 else 0,
                'round_count': rounds
            }
        
        # Process each player count category for CT side
        for pc in ['5v5', '4v4', '3v3', 'all']:
            rounds = info['ct_side'][pc]['round_count']
            display_data[steam_id]['ct_side'][pc] = {
                'avg_vel_xy': info['ct_side'][pc]['total_vel_xy'] / rounds if rounds > 0 else 0,
                'avg_time_alive': info['ct_side'][pc]['total_time_alive'] / rounds if rounds > 0 else 0,
                'round_count': rounds
            }
        
        # Combined stats (all player counts)
        display_data[steam_id]['combined'] = {
            'avg_vel_xy': (info['t_side']['all']['total_vel_xy'] + info['ct_side']['all']['total_vel_xy']) / total_rounds,
            'avg_time_alive': (info['t_side']['all']['total_time_alive'] + info['ct_side']['all']['total_time_alive']) / total_rounds
        }
    
    # Print results
    print("\n=== Player Statistics Across All Demos ===")
    for steam_id, info in sorted(display_data.items(), key=lambda x: x[1]['round_count'], reverse=True)[:20]:
        print(f"\n{info['name']} (Steam ID: {steam_id}):")
        print(f"  Total Rounds: {info['round_count']}")
        
        if info['t_side']['all']['round_count'] > 0:
            print(f"\n  T-Side (Total: {info['t_side']['all']['round_count']} rounds):")
            print(f"    Overall: {info['t_side']['all']['avg_vel_xy']:.2f} units/s, {info['t_side']['all']['avg_time_alive']:.2f}s alive")
            
            for pc in ['5v5', '4v4', '3v3']:
                if info['t_side'][pc]['round_count'] > 0:
                    print(f"    {pc} ({info['t_side'][pc]['round_count']} rounds): " + 
                          f"{info['t_side'][pc]['avg_vel_xy']:.2f} units/s, " +
                          f"{info['t_side'][pc]['avg_time_alive']:.2f}s alive")
        
        if info['ct_side']['all']['round_count'] > 0:
            print(f"\n  CT-Side (Total: {info['ct_side']['all']['round_count']} rounds):")
            print(f"    Overall: {info['ct_side']['all']['avg_vel_xy']:.2f} units/s, {info['ct_side']['all']['avg_time_alive']:.2f}s alive")
            
            for pc in ['5v5', '4v4', '3v3']:
                if info['ct_side'][pc]['round_count'] > 0:
                    print(f"    {pc} ({info['ct_side'][pc]['round_count']} rounds): " + 
                          f"{info['ct_side'][pc]['avg_vel_xy']:.2f} units/s, " +
                          f"{info['ct_side'][pc]['avg_time_alive']:.2f}s alive")
        
        print(f"\n  Combined: {info['combined']['avg_vel_xy']:.2f} units/s, {info['combined']['avg_time_alive']:.2f}s alive")
    
    # Get top players by round count
    top_players = sorted(display_data.items(), key=lambda x: x[1]['round_count'], reverse=True)[:80]
    
    # Filter players with sufficient data
    players_to_plot = [(steam_id, info) for steam_id, info in top_players 
                       if info['name'] not in ['Ex3rcice', 'MUTiRiS']]
    
    # Define colors for each scenario
    scenario_colors = {
        '5v5': '#2E86AB',  # Blue
        '4v4': '#A23B72',  # Purple
        '3v3': '#F18F01',  # Orange
        'all': '#90A959'   # Green
    }
    
    # Create figure with 2x2 subplots (T-side and CT-side, with scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    # Collect all data points to determine global axis ranges
    all_vel_xy = []
    all_time_alive = []
    
    for steam_id, info in players_to_plot:
        for side in ['t_side', 'ct_side']:
            for scenario in ['5v5', '4v4', '3v3', 'all']:
                if info[side][scenario]['round_count'] > 5:
                    all_vel_xy.append(info[side][scenario]['avg_vel_xy'])
                    all_time_alive.append(info[side][scenario]['avg_time_alive'])
    
    # Calculate global axis limits with some padding
    x_min, x_max = min(all_vel_xy), max(all_vel_xy)
    y_min, y_max = min(all_time_alive), max(all_time_alive)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_lim = (x_min - x_padding, x_max + x_padding)
    y_lim = (y_min - y_padding, y_max + y_padding)
    
    # T-Side by scenario
    for scenario in ['5v5', '4v4', '3v3']:
        scenario_plotted = False
        for steam_id, info in players_to_plot:
            if info['t_side'][scenario]['round_count'] > 5:  # Only show if enough samples
                ax = axes[0, 0]
                ax.scatter(info['t_side'][scenario]['avg_vel_xy'], 
                          info['t_side'][scenario]['avg_time_alive'],
                          s=15, alpha=0.6, color=scenario_colors[scenario], 
                          label=scenario if not scenario_plotted else "")
                scenario_plotted = True
    
    axes[0, 0].set_xlabel('Average XY Velocity (units/second)', fontsize=11)
    axes[0, 0].set_ylabel('Average Time Alive (seconds)', fontsize=11)
    axes[0, 0].set_title('T-Side: Velocity vs Time Alive by Scenario', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_xlim(x_lim)
    axes[0, 0].set_ylim(y_lim)
    
    # CT-Side by scenario
    for scenario in ['5v5', '4v4', '3v3']:
        scenario_plotted = False
        for steam_id, info in players_to_plot:
            if info['ct_side'][scenario]['round_count'] > 5:  # Only show if enough samples
                ax = axes[0, 1]
                ax.scatter(info['ct_side'][scenario]['avg_vel_xy'], 
                          info['ct_side'][scenario]['avg_time_alive'],
                          s=15, alpha=0.6, color=scenario_colors[scenario],
                          label=scenario if not scenario_plotted else "")
                scenario_plotted = True
    
    axes[0, 1].set_xlabel('Average XY Velocity (units/second)', fontsize=11)
    axes[0, 1].set_ylabel('Average Time Alive (seconds)', fontsize=11)
    axes[0, 1].set_title('CT-Side: Velocity vs Time Alive by Scenario', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].set_xlim(x_lim)
    axes[0, 1].set_ylim(y_lim)
    
    # T-Side overall with player names
    for steam_id, info in players_to_plot[:30]:  # Top 30 players
        if info['t_side']['all']['round_count'] > 10:
            ax = axes[1, 0]
            ax.scatter(info['t_side']['all']['avg_vel_xy'], 
                      info['t_side']['all']['avg_time_alive'],
                      s=20, alpha=0.7, color=scenario_colors['all'])
            ax.text(info['t_side']['all']['avg_vel_xy'] + 0.3, 
                   info['t_side']['all']['avg_time_alive'] - 0.2,
                   info['name'], fontsize=7, alpha=0.8)
    
    axes[1, 0].set_xlabel('Average XY Velocity (units/second)', fontsize=11)
    axes[1, 0].set_ylabel('Average Time Alive (seconds)', fontsize=11)
    axes[1, 0].set_title('T-Side: Overall (All Scenarios) - Top 30 Players', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(x_lim)
    axes[1, 0].set_ylim(y_lim)
    
    # CT-Side overall with player names
    for steam_id, info in players_to_plot[:30]:  # Top 30 players
        if info['ct_side']['all']['round_count'] > 10:
            ax = axes[1, 1]
            ax.scatter(info['ct_side']['all']['avg_vel_xy'], 
                      info['ct_side']['all']['avg_time_alive'],
                      s=20, alpha=0.7, color='#C73E1D')  # Red for CT
            ax.text(info['ct_side']['all']['avg_vel_xy'] + 0.3, 
                   info['ct_side']['all']['avg_time_alive'] - 0.2,
                   info['name'], fontsize=7, alpha=0.8)
    
    axes[1, 1].set_xlabel('Average XY Velocity (units/second)', fontsize=11)
    axes[1, 1].set_ylabel('Average Time Alive (seconds)', fontsize=11)
    axes[1, 1].set_title('CT-Side: Overall (All Scenarios) - Top 30 Players', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(x_lim)
    axes[1, 1].set_ylim(y_lim)
    
    plt.tight_layout()
    plt.show()
