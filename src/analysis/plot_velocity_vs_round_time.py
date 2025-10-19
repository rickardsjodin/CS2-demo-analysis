"""
Plot average player velocity vs round time.
Shows how velocity changes throughout a round on average across multiple demos.
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
import numpy as np

from src.core.snapshot_extractor_optimized import sort_demo_paths_by_dir_number_desc
from src.utils.cache_utils import load_demo

# File to store velocity vs time data
VELOCITY_TIME_DATA_FILE = Path("velocity_vs_time_data.json")

# Top teams to prioritize (based on HLTV rankings)
TOP_TEAMS = [
    'spirit','vitality','falcons', 'mongolz', 'mouz', 
    'furia', 'g2', 'aurora', 'natus-vincere', 'navi', '3dmax', 
    'faze', 'astralis', 'liquid', 'heroic', 'complexity', 
    'eternal-fire', 'saw', 'virtus-pro', 'big'
]

# Configuration
START_OFFSET_SECONDS = 0
TIME_BIN_SIZE = 1.0  # seconds - bin size for averaging velocities
MAX_ROUND_TIME = 120  # seconds - maximum round time to consider

def get_team_priority(demo_file):
    """Get priority score for a demo based on team names in filename."""
    filename_lower = Path(demo_file).name.lower()
    for i, team in enumerate(TOP_TEAMS):
        if team in filename_lower:
            return i  # Lower number = higher priority
    return len(TOP_TEAMS)  # Lowest priority for non-top teams


def get_demo_hash(demo_file):
    """Get a unique hash for a demo file based on its path and modification time."""
    demo_path = Path(demo_file)
    mtime = demo_path.stat().st_mtime
    size = demo_path.stat().st_size
    hash_input = f"{demo_path.name}_{mtime}_{size}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def load_velocity_time_data():
    """Load existing velocity vs time data from file."""
    if VELOCITY_TIME_DATA_FILE.exists():
        try:
            with open(VELOCITY_TIME_DATA_FILE, 'r') as f:
                data = json.load(f)
            print(f"✅ Loaded existing velocity-time data ({len(data.get('processed_demos', []))} demos)")
            return data
        except Exception as e:
            print(f"⚠️ Error loading velocity-time data: {e}")
            return {'time_bins': {}, 'processed_demos': []}
    return {'time_bins': {}, 'processed_demos': []}


def save_velocity_time_data(data):
    """Save velocity vs time data to file."""
    try:
        with open(VELOCITY_TIME_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved velocity-time data to {VELOCITY_TIME_DATA_FILE}")
    except Exception as e:
        print(f"❌ Error saving velocity-time data: {e}")


def merge_time_bin_data(existing_bins, new_bins):
    """Merge new time bin data into existing data."""
    for time_bin, new_values in new_bins.items():
        if time_bin not in existing_bins:
            existing_bins[time_bin] = {
                'velocities': [],
                'count': 0
            }
        existing_bins[time_bin]['velocities'].extend(new_values['velocities'])
        existing_bins[time_bin]['count'] += new_values['count']
    
    return existing_bins


def calculate_velocity_vs_time(demo_file, tick_rate=64):
    """
    Calculate velocity vs time for all rounds in a demo.
    Groups velocities by time bins from round start.
    
    Args:
        demo_file: Path to the .dem file
        tick_rate: Ticks per second (default 64)
        
    Returns:
        Dictionary mapping time bins to lists of velocities
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
    
    # Sort by player name and tick to ensure correct ordering
    ticks_df = ticks_df.sort(['name', 'tick'])
    
    # Mark death ticks for each player in each round
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
    
    # Filter out invalid velocities
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
    
    time_bins = {}
    
    # Process each round
    for round_row in rounds_df.iter_rows(named=True):
        round_num = round_row['round_num']
        start_tick = round_row['freeze_end']
        if start_tick is None:
            continue
        
        start_tick = start_tick + int(START_OFFSET_SECONDS * tick_rate)
        end_tick = round_row['end']
        
        # Filter ticks for this round - only include ticks where player is alive
        round_ticks = ticks_df.filter(
            (pl.col('tick') >= start_tick) & 
            (pl.col('tick') <= end_tick) &
            (pl.col('velocity').is_not_null()) &
            (pl.col('is_alive') == True)
        )
        
        # Calculate time from round start for each tick
        round_ticks = round_ticks.with_columns([
            ((pl.col('tick') - start_tick) / tick_rate).alias('time_from_start')
        ])
        
        # Bin the time and group velocities
        round_ticks = round_ticks.with_columns([
            (pl.col('time_from_start') / TIME_BIN_SIZE).floor().alias('time_bin')
        ])
        
        # Filter to only include reasonable round times
        round_ticks = round_ticks.filter(
            pl.col('time_from_start') <= MAX_ROUND_TIME
        )
        
        # Group by time bin and collect all velocities
        for row in round_ticks.iter_rows(named=True):
            time_bin = str(int(row['time_bin']))  # Convert to string for JSON
            velocity_xy = row['velocity_xy']
            
            if velocity_xy is None or (isinstance(velocity_xy, float) and math.isnan(velocity_xy)):
                continue
            
            if time_bin not in time_bins:
                time_bins[time_bin] = {
                    'velocities': [],
                    'count': 0
                }
            
            time_bins[time_bin]['velocities'].append(velocity_xy)
            time_bins[time_bin]['count'] += 1
    
    return time_bins


def plot_velocity_vs_time(time_bins_data):
    """Plot average velocity vs round time."""
    # Convert time bins to sorted list
    time_bins = sorted([(int(k), v) for k, v in time_bins_data.items()])
    
    times = []
    avg_velocities = []
    std_velocities = []
    counts = []
    
    for time_bin, data in time_bins:
        if data['count'] > 0 and len(data['velocities']) > 0:
            # Convert time bin to actual time (center of bin)
            time = (time_bin + 0.5) * TIME_BIN_SIZE
            times.append(time)
            
            velocities = np.array(data['velocities'])
            avg_velocities.append(np.mean(velocities))
            std_velocities.append(np.std(velocities))
            counts.append(data['count'])
    
    times = np.array(times)
    avg_velocities = np.array(avg_velocities)
    std_velocities = np.array(std_velocities)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Main plot with shaded standard deviation
    ax1.plot(times, avg_velocities, 'b-', linewidth=2, label='Average Velocity')
    ax1.fill_between(times, 
                      avg_velocities - std_velocities, 
                      avg_velocities + std_velocities, 
                      alpha=0.3, color='blue', label='±1 Std Dev')
    
    ax1.set_xlabel('Time from Round Start (seconds)', fontsize=12)
    ax1.set_ylabel('Average XY Velocity (units/second)', fontsize=12)
    ax1.set_title('Player Velocity vs Round Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Sample count plot
    ax2.bar(times, counts, width=TIME_BIN_SIZE * 0.8, alpha=0.6, color='green')
    ax2.set_xlabel('Time from Round Start (seconds)', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Number of Velocity Samples per Time Bin', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('velocity_vs_round_time.png', dpi=300, bbox_inches='tight')
    print("📊 Saved plot to velocity_vs_round_time.png")
    plt.show()
    
    # Print statistics
    print("\n=== Velocity Statistics by Time ===")
    print(f"{'Time (s)':<10} {'Avg Vel':<12} {'Std Dev':<12} {'Samples':<10}")
    print("-" * 50)
    for i in range(len(times)):
        print(f"{times[i]:<10.1f} {avg_velocities[i]:<12.2f} {std_velocities[i]:<12.2f} {counts[i]:<10}")


if __name__ == '__main__':
    dem_dir = Path("G:\\CS2\\demos")
    
    if not os.path.exists(dem_dir):
        print(f"Demo directory not found at {dem_dir}")
        exit(1)
    
    # Load existing data
    velocity_data = load_velocity_time_data()
    time_bins_data = velocity_data['time_bins']
    processed_demos = set(velocity_data['processed_demos'])
    
    # Recursively collect all .dem files
    demo_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(dem_dir)
        for file in files
        if file.endswith('.dem')
    ]
    
    # Sort by directory number first, then by team priority
    demo_files = sort_demo_paths_by_dir_number_desc(demo_files)
    demo_files = sorted(demo_files, key=lambda x: (get_team_priority(x), -os.path.getmtime(x)))[:20]
    
    # Filter out already processed demos
    demos_to_process = []
    for demo_file in demo_files:
        demo_hash = get_demo_hash(demo_file)
        if demo_hash not in processed_demos:
            demos_to_process.append(demo_file)
    
    print(f"📊 Total demos: {len(demo_files)}")
    print(f"✅ Already processed: {len(demo_files) - len(demos_to_process)}")
    print(f"🆕 To process: {len(demos_to_process)}")
    
    new_time_bins = {}  # Temporary storage for new time bin data
    
    for demo_file in tqdm(demos_to_process, desc="Processing demos"):
        demo_hash = get_demo_hash(demo_file)
        
        try:
            # Calculate velocity vs time for this demo
            demo_time_bins = calculate_velocity_vs_time(demo_file)
            
            # Merge with new_time_bins
            new_time_bins = merge_time_bin_data(new_time_bins, demo_time_bins)
            
        except Exception as e:
            print(f"\n⚠️ Error processing {demo_file}: {e}")
            continue
        
        # Mark demo as processed
        processed_demos.add(demo_hash)
    
    # Merge new data with existing data
    time_bins_data = merge_time_bin_data(time_bins_data, new_time_bins)
    
    # Save updated data
    velocity_data = {
        'time_bins': time_bins_data,
        'processed_demos': list(processed_demos)
    }
    save_velocity_time_data(velocity_data)
    
    # Create the plot
    if time_bins_data:
        plot_velocity_vs_time(time_bins_data)
    else:
        print("⚠️ No data to plot!")
