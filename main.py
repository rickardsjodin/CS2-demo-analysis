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
from plotting import plot_kill_death_analysis, plot_positive_negative_impact, plot_impact_difference_per_round, plot_individual_impacts, compare_individual_impacts

#%%
# Force reload modules (run this cell when you make changes to the modules)
import analysis
import plotting
importlib.reload(analysis)
importlib.reload(plotting)
print("✅ Modules reloaded successfully!")

#%%
# Parse the demo file
dem = Demo("vitality-vs-gamer-legion-m1-train.dem")
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
player_name = "Kursy"  # Change this to any player name from the list above

print(f"╔{'═' * 68}╗")
print(f"║ {player_name.upper()} - KILL/DEATH ANALYSIS{' ' * (67 - len(player_name) - 25)}║")
print(f"╚{'═' * 68}╝")
print()

# Generate the analysis
stats = get_player_kill_death_analysis(dem, player_name, debug=False)
if stats is not None and not stats.empty:
    print("\n" + "="*80)
    print(f"📋 ROUND-BY-ROUND ANALYSIS: {player_name.upper()}")
    print("="*80)
    
    for _, row in stats.iterrows():
        round_num = str(row['Round']).replace('💣', '').strip()
        has_bomb = '💣' in str(row['Round'])
        side = str(row['Side'])
        kills_raw = str(row['Kills'])
        deaths_raw = str(row['Deaths'])
        impact_raw = str(row['Impact'])
        
        # Side indicator
        side_icon = "🔵CT" if side == "CT" else "🟠T"
        bomb_icon = "💣" if has_bomb else ""
        
        # Clean impact display
        impact_val = impact_raw.strip()
        if impact_val.startswith('+'):
            impact_display = f"🟢{impact_val}"
        elif impact_val.startswith('-'):
            impact_display = f"🔴{impact_val}"
        else:
            impact_display = f"⚪{impact_val}"
        
        # Round header
        print(f"\nRound {round_num} {bomb_icon} │ {side_icon} │ {impact_display}")
        print("─" * 50)
        
        # Process kills - each on its own line
        if kills_raw != '-':
            print("  KILLS:")
            kills = [k.strip() for k in kills_raw.split(' | ') if k.strip()]
            for i, kill in enumerate(kills, 1):
                if '(' in kill and ')' in kill:
                    # Extract game state and impact
                    main_part = kill[:kill.rfind('(')].strip()
                    impact_part = kill[kill.rfind('('):].strip()
                    
                    # Remove weapon info and clean up
                    if ' vs ' in main_part:
                        # Remove weapon part after "vs"
                        game_state = main_part.split(' vs ')[0].replace(' v ', 'v')
                    else:
                        game_state = main_part.replace(' v ', 'v')
                    
                    print(f"    {i}. {game_state} {impact_part}")
                else:
                    print(f"    {i}. {kill}")
        
        # Process deaths - each on its own line  
        if deaths_raw != '-':
            print("  DEATHS:")
            deaths = [d.strip() for d in deaths_raw.split(' | ') if d.strip()]
            for i, death in enumerate(deaths, 1):
                if '(' in death and ')' in death:
                    # Extract game state and impact
                    main_part = death[:death.rfind('(')].strip()
                    impact_part = death[death.rfind('('):].strip()
                    
                    # Remove weapon info and clean up
                    if ' vs ' in main_part:
                        # Remove weapon part after "vs"
                        game_state = main_part.split(' vs ')[0].replace(' v ', 'v')
                    else:
                        game_state = main_part.replace(' v ', 'v')
                    
                    print(f"    {i}. {game_state} {impact_part}")
                else:
                    print(f"    {i}. {death}")
    
    print("\n" + "="*80)
    
    # Enhanced summary statistics section
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
    
    # Calculate summary statistics
    total_kills = len([k for k in stats['Kills'] if k != '-'])
    total_deaths = len([d for d in stats['Deaths'] if d != '-'])
    total_impact = sum(float(imp.replace('+', '').replace('⚪', '').replace('🟢', '').replace('🔴', '').strip()) 
                      for imp in stats['Impact'] if imp not in ['-', '⚪-'])
    
    print(f"\n📊 SUMMARY")
    print("─" * 30)
    print(f"Kills:  {total_kills}")
    print(f"Deaths: {total_deaths}")
    print(f"K/D:    {total_kills/max(total_deaths, 1):.2f}")
    print(f"Impact: {total_impact:+.1f}")
    print("─" * 30)
    
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
# Compare two players (example: Kursy vs another player)
print("\n" + "="*80)
print("PLAYER COMPARISON")
print("="*80)

# You can change these player names to compare different players
player1 = "REZ"
player2 = "Kursy"  # Change this to another player from the available list

# %%
# Compare individual impacts between two players
print("\n" + "="*80)
print("INDIVIDUAL IMPACT COMPARISON")
print("="*80)

compare_individual_impacts(dem, player1, player2)

# %%
