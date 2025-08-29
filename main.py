"""
Main CS2 Demo Analysis Script
Simplified main file that imports functions from separate modules.
"""

#%%
# Auto-reload setup for Jupyter/IPython environments
# In Jupyter, run these commands manually or uncomment them:
# %load_ext autoreload
# %autoreload 2

#%%
from awpy import Demo
import pandas as pd
import polars as pl
import importlib

# Import our custom modules
from analysis import get_player_kill_death_analysis, create_probability_scenarios_table
from plotting import plot_kill_death_analysis, plot_positive_negative_impact, compare_players_impact, plot_impact_difference_per_round, plot_individual_impacts, compare_individual_impacts

#%%
# Force reload modules (run this cell when you make changes to the modules)
import analysis
import plotting
importlib.reload(analysis)
importlib.reload(plotting)
print("✅ Modules reloaded successfully!")

#%%
# Parse the demo file
dem = Demo("the-mongolz-vs-vitality-m1-mirage.dem")
dem.parse()

#%%
# Display available players
print("📋 AVAILABLE PLAYERS:")
print("="*50)

# Get unique player names from kills data
if dem.kills is not None and len(dem.kills) > 0:
    kills_df = dem.kills.to_pandas() if hasattr(dem.kills, 'to_pandas') else dem.kills
    
    # Get unique attackers and victims
    attackers = set(kills_df['attacker_name'].dropna().unique())
    victims = set(kills_df['victim_name'].dropna().unique())
    all_players = sorted(attackers.union(victims))
    
    for i, player in enumerate(all_players, 1):
        print(f"{i:2d}. {player}")
else:
    print("❌ No player data found")
    exit()

print()

#%%
# MAIN ANALYSIS - Change this to analyze different players
player_name = "mezii"  # Change this to any player name from the list above

print(f"╔{'═' * 68}╗")
print(f"║ {player_name.upper()} - KILL/DEATH ANALYSIS{' ' * (67 - len(player_name) - 25)}║")
print(f"╚{'═' * 68}╝")
print()

# Generate the analysis
stats = get_player_kill_death_analysis(dem, player_name, debug=False)
if stats is not None and not stats.empty:
    print("\nRound-by-Round Analysis:")
    print("─" * 80)
    
    # Custom formatting for better readability
    print(f"{'Round':<8} {'Side':<4} {'Kills':<45} {'Deaths':<45} {'Impact':<10}")
    print("─" * 115)
    
    for _, row in stats.iterrows():
        round_num = f"{row['Round']:<8}"
        side = f"{row['Side']:<4}"
        kills = f"{row['Kills']:<45}"
        deaths = f"{row['Deaths']:<45}"
        impact = f"{row['Impact']:<10}"
        print(f"{round_num} {side} {kills} {deaths} {impact}")
    
    print("─" * 115)
    
    # Summary statistics - extract impact from the DataFrame
    total_kills = 0
    total_deaths = 0
    total_impact = 0
    
    for _, row in stats.iterrows():
        # Count kills and deaths (exclude '-' entries)
        kills_count = len([k for k in row['Kills'].split(' | ') if k != '-' and k.strip()])
        deaths_count = len([d for d in row['Deaths'].split(' | ') if d != '-' and d.strip()])
        
        total_kills += kills_count
        total_deaths += deaths_count
        
        # Extract impact value (remove '+' and convert to float)
        impact_str = row['Impact'].replace('+', '')
        try:
            total_impact += float(impact_str)
        except:
            pass
    
    print(f"\nSUMMARY:")
    print(f"Total Kills: {total_kills}")
    print(f"Total Deaths: {total_deaths}")
    print(f"K/D Ratio: {total_kills/max(total_deaths, 1):.2f}")
    print(f"Total Impact: {total_impact:+.1f} points")
    
    # Create the bar plot
    print(f"\n📊 Generating bar plot for {player_name}...")
    plot_kill_death_analysis(stats, player_name)
    
    # Create the positive/negative impact analysis
    print(f"\n📈 Generating positive/negative impact analysis for {player_name}...")
    plot_positive_negative_impact(stats, player_name)
    
    # Create the impact difference per round plot
    print(f"\n🎯 Generating impact difference per round analysis for {player_name}...")
    plot_impact_difference_per_round(stats, player_name)
    
    # Create the individual impacts timeline plot
    print(f"\n⚡ Generating individual kill/death impacts timeline for {player_name}...")
    plot_individual_impacts(dem, player_name)

else:
    print(f"❌ No data found for player: {player_name}")
    print("Available players listed above ↑")

# %%
# Display probability scenarios table
print("\n" + "="*80)
create_probability_scenarios_table()

# %%
# Compare two players (example: mezii vs another player)
print("\n" + "="*80)
print("PLAYER COMPARISON")
print("="*80)

# You can change these player names to compare different players
player1 = "mezii"
player2 = "bLitz"  # Change this to another player from the available list

compare_players_impact(dem, player1, player2)

# %%
# Compare individual impacts between two players
print("\n" + "="*80)
print("INDIVIDUAL IMPACT COMPARISON")
print("="*80)

compare_individual_impacts(dem, player1, player2)

# %%
