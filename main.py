"""
Simple and clean main script for CS2 demo analysis
"""

from matplotlib import pyplot as plt
import config

# Import our custom modules
from src.core.analysis import get_player_kill_death_analysis, create_probability_scenarios_table
from src.utils.plotting import compare_individual_impacts_vertical, plot_kill_death_analysis, plot_positive_negative_impact, plot_impact_difference_per_round, plot_individual_impacts_by_round, compare_individual_impacts, plot_all_players_stats_table
from src.utils.formatting import display_available_players, display_player_header, format_player_analysis, display_summary_stats
from src.utils.cache_utils import load_demo, clear_cache


# ================================
# CONFIGURATION - EDIT config.py OR THESE VALUES
# ================================
# You can edit config.py for project-wide settings, or override them here:

DEMO_FILE = str("G:\\CS2\\demos\\fissure-playground-2-falcons-vs-virtuspro-bo3-qivzNI2LmnWi0RrHw-7sxj\\falcons-vs-virtus-pro-m1-mirage.dem")  # or set your own path
PLAYER_TO_ANALYZE = "NiKo"  # or set your own player name

# Optional: Set these for player comparison (leave as None to skip comparison)
COMPARE_PLAYER1 = "NiKo"  # e.g., "Kursy"
COMPARE_PLAYER2 = "m0NESY"  # e.g., "REZ"

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
        # plot_kill_death_analysis(stats, PLAYER_TO_ANALYZE)
        # plot_positive_negative_impact(stats, PLAYER_TO_ANALYZE)
        # plot_impact_difference_per_round(stats, PLAYER_TO_ANALYZE)
        # plot_individual_impacts_by_round(dem, PLAYER_TO_ANALYZE)
        
    else:
        print(f"❌ No data found for player: {PLAYER_TO_ANALYZE}")
        print("Available players listed above ↑")
    
    # Display all players stats table
    print("\n" + "="*80)
    print("📊 GENERATING ALL PLAYERS STATISTICS TABLE")
    print("="*80)
    # plot_all_players_stats_table(dem)
    
    # Display probability scenarios table
    print("\n" + "="*80)
    # create_probability_scenarios_table()
    
    # Optional player comparison
    if COMPARE_PLAYER1 and COMPARE_PLAYER2:
        print(f"\n🔥 Comparing {COMPARE_PLAYER1} vs {COMPARE_PLAYER2}...")
        compare_players(dem, COMPARE_PLAYER1, COMPARE_PLAYER2)


def compare_players(dem, player1, player2):
    """Compare two players' individual impacts"""
    print("\n" + "="*80)
    print(f"INDIVIDUAL IMPACT COMPARISON: {player1.upper()} vs {player2.upper()}")
    print("="*80)
    
    compare_individual_impacts_vertical(dem, player1, player2)


if __name__ == "__main__":
    main()
    plt.show()
