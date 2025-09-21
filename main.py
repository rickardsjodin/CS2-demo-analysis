"""
Simple and clean main script for CS2 demo analysis
"""

from pathlib import Path
import sys
from matplotlib import pyplot as plt
import config


# Import our custom modules
from src.core.analysis import get_player_kill_death_analysis
from src.utils.plotting import compare_individual_impacts_vertical, plot_kill_death_analysis, plot_positive_negative_impact, plot_impact_difference_per_round, plot_individual_impacts_by_round, compare_individual_impacts, plot_all_players_stats_table
from src.utils.formatting import display_available_players, display_player_header, format_player_analysis, display_summary_stats
from src.utils.cache_utils import load_demo, clear_cache


# ================================
# CONFIGURATION - EDIT config.py OR THESE VALUES
# ================================
# You can edit config.py for project-wide settings, or override them here:

DEMO_FILE = str("demos/the-mongolz-vs-vitality-m1-mirage.dem")  # or set your own path
PLAYER_TO_ANALYZE = "NiKo"  # or set your own player name

player1 = "mezii"
player2 = "bLitz"

# Cache settings
USE_CACHE = config.USE_CACHE
CLEAR_CACHE_ON_START = config.CLEAR_CACHE_ON_START


def main():
    """Main function to run the CS2 demo analysis"""
    
    # Clear cache if requested
    if CLEAR_CACHE_ON_START:
        clear_cache()
    
    # Load demo (from cache if available)
    dem = load_demo(DEMO_FILE, use_cache=USE_CACHE)
    if dem is None:
        return
    
    # # Display available players
    all_players = display_available_players(dem)
    # if not all_players:
    #     print("❌ No players found in demo")
    #     return
    
    # print()
    
    # if PLAYER_TO_ANALYZE not in all_players:
    #     print(f"❌ Player '{PLAYER_TO_ANALYZE}' not found in demo")
    #     print(f"Available players: {', '.join(all_players)}")
    #     return
    
    # Display player analysis
    display_player_header(PLAYER_TO_ANALYZE)
    
    

    # Display all players stats table
    print("\n" + "="*80)
    print("📊 GENERATING ALL PLAYERS STATISTICS TABLE")
    print("="*80)
    # plot_all_players_stats_table(dem)
    
    # Display probability scenarios table
    print("\n" + "="*80)
    # create_probability_scenarios_table()
    
    """Compare two players' individual impacts"""
    print("\n" + "="*80)
    print(f"INDIVIDUAL IMPACT COMPARISON: {player1.upper()} vs {player2.upper()}")
    print("="*80)
    
    compare_individual_impacts_vertical(dem, player1, player2)



if __name__ == "__main__":
    main()
    plt.show()
