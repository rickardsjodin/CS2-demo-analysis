"""
Simple and clean main script for CS2 demo analysis
"""

from awpy import Demo
from matplotlib import pyplot as plt
import pandas as pd
import polars as pl

# Import our custom modules
from analysis import get_player_kill_death_analysis, create_probability_scenarios_table
from plotting import plot_kill_death_analysis, plot_positive_negative_impact, plot_impact_difference_per_round, plot_individual_impacts, compare_individual_impacts
from formatting import display_available_players, display_player_header, format_player_analysis, display_summary_stats
from cache_utils import load_demo, clear_cache


# ================================
# CONFIGURATION - EDIT THESE VALUES
# ================================
DEMO_FILE = "vitality-vs-gamer-legion-m1-train.dem"
PLAYER_TO_ANALYZE = "Kursy"

# Optional: Set these for player comparison (leave as None to skip comparison)
COMPARE_PLAYER1 = "REZ"  # e.g., "REZ"
COMPARE_PLAYER2 = "Kursy"  # e.g., "Kursy"

# Cache settings
USE_CACHE = True  # Set to False to disable caching
CLEAR_CACHE_ON_START = False  # Set to True to clear cache before running


def main():
    """Main function to run the CS2 demo analysis"""
    
    # Clear cache if requested
    if CLEAR_CACHE_ON_START:
        clear_cache()
    
    # Load demo (from cache if available)
    dem = load_demo(DEMO_FILE, use_cache=USE_CACHE)
    if dem is None:
        return
    
    # Display available players
    all_players = display_available_players(dem)
    if not all_players:
        print("❌ No players found in demo")
        return
    
    print()
    
    if PLAYER_TO_ANALYZE not in all_players:
        print(f"❌ Player '{PLAYER_TO_ANALYZE}' not found in demo")
        print(f"Available players: {', '.join(all_players)}")
        return
    
    # Display player analysis
    display_player_header(PLAYER_TO_ANALYZE)
    
    # Generate the analysis
    stats = get_player_kill_death_analysis(dem, PLAYER_TO_ANALYZE, debug=False)
    
    if stats is not None and not stats.empty:
        # Display formatted analysis
        format_player_analysis(stats, PLAYER_TO_ANALYZE)
        display_summary_stats(stats)
        
        # Generate plots
        print(f"\n📊 Generating visualizations for {PLAYER_TO_ANALYZE}...")
        plot_kill_death_analysis(stats, PLAYER_TO_ANALYZE)
        plot_positive_negative_impact(stats, PLAYER_TO_ANALYZE)
        plot_impact_difference_per_round(stats, PLAYER_TO_ANALYZE)
        plot_individual_impacts(dem, PLAYER_TO_ANALYZE)
        
    else:
        print(f"❌ No data found for player: {PLAYER_TO_ANALYZE}")
        print("Available players listed above ↑")
    
    # Display probability scenarios table
    print("\n" + "="*80)
    create_probability_scenarios_table()
    
    # Optional player comparison
    if COMPARE_PLAYER1 and COMPARE_PLAYER2:
        print(f"\n🔥 Comparing {COMPARE_PLAYER1} vs {COMPARE_PLAYER2}...")
        compare_players(dem, COMPARE_PLAYER1, COMPARE_PLAYER2)


def compare_players(dem, player1, player2):
    """Compare two players' individual impacts"""
    print("\n" + "="*80)
    print(f"INDIVIDUAL IMPACT COMPARISON: {player1.upper()} vs {player2.upper()}")
    print("="*80)
    
    compare_individual_impacts(dem, player1, player2)


if __name__ == "__main__":
    main()
    plt.show()
