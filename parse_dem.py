#%%
from awpy import Demo
import pandas as pd
import polars as pl
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

#%%
# Parse the demo file
dem = Demo("the-mongolz-vs-vitality-m1-mirage.dem")
dem.parse()

#%%
import awpy
import pandas as pd

def get_win_probability(ct_alive, t_alive, post_plant=False):
    if ct_alive == 0:
        return 0.0  # T wins, no CTs left
    if t_alive == 0:
        return 1.0  # CT wins
    
    # Probability lookup table based on provided data
    # Format: win_probabilities[t_alive][ct_alive] = T win probability
    win_probabilities = {
        5: {5: 0.488, 4: 0.702, 3: 0.890, 2: 0.981, 1: 0.999},
        4: {5: 0.315, 4: 0.525, 3: 0.769, 2: 0.942, 1: 0.996},
        3: {5: 0.143, 4: 0.300, 3: 0.546, 2: 0.819, 1: 0.974},
        2: {5: 0.032, 4: 0.099, 3: 0.258, 2: 0.547, 1: 0.870},
        1: {5: 0.002, 4: 0.009, 3: 0.045, 2: 0.189, 1: 0.551}
    }
    
    # Get base T win probability
    if t_alive in win_probabilities and ct_alive in win_probabilities[t_alive]:
        t_win_prob = win_probabilities[t_alive][ct_alive]
    else:
        # Fallback for edge cases
        ratio = ct_alive / (ct_alive + t_alive)
        t_win_prob = 1 - (ratio * 0.8 + 0.1)
    
    # Convert to CT win probability
    ct_win_prob = 1 - t_win_prob
    
    # Adjust for post-plant scenarios (T gets significant advantage)
    if post_plant:
        # Post-plant adjustment should be proportional to base probability
        # Don't let it become identical to 0 CT scenario
        if ct_alive > 0:
            # T gets advantage but CT still has some chance proportional to player count
            post_plant_boost = 0.15 + (0.10 * (t_alive / ct_alive))  # 15-25% boost
            t_win_prob = min(0.90, t_win_prob + post_plant_boost)
        ct_win_prob = 1 - t_win_prob
    
    return ct_win_prob

def calculate_impact_score(ct_before, t_before, ct_after, t_after, is_post_plant=False):
    """
    Calculate the impact score of a kill based on how it changes round win probability.
    
    Args:
        ct_before: CT players alive before the kill
        t_before: T players alive before the kill  
        ct_after: CT players alive after the kill
        t_after: T players alive after the kill
        is_post_plant: Whether bomb is planted
    
    Returns:
        Impact score (higher = more impactful)
    """
    
    # Calculate win probabilities before and after
    prob_before = get_win_probability(ct_before, t_before, is_post_plant)
    prob_after = get_win_probability(ct_after, t_after, is_post_plant)
    
    # Impact is the change in win probability (0-100 scale)
    impact = abs(prob_after - prob_before) * 100
    
    return round(impact, 1)

#%%
def get_player_kill_death_analysis(demo, player_name, debug=False):
    """
    Calculate proper game state by tracking alive players throughout each round.
    Includes bomb plant detection and shows all rounds even if empty.
    
    Args:
        demo: Parsed demo object
        player_name: Name of the player to analyze
        debug: Whether to show debug information
    
    Returns:
        DataFrame with round-by-round kill/death analysis including game states and bomb plants
    """
    from collections import defaultdict
    
    kills_df = demo.kills
    
    if kills_df is None:
        return None
    
    # Convert to pandas if it's a Polars DataFrame
    if hasattr(kills_df, 'to_pandas'):
        kills_df = kills_df.to_pandas()
    
    # Get bomb events if available
    bomb_events = None
    if hasattr(demo, 'bomb') and demo.bomb is not None:
        bomb_events = demo.bomb
        if hasattr(bomb_events, 'to_pandas'):
            bomb_events = bomb_events.to_pandas()
    
    # Debug: Let's see what bomb data we have
    if debug:
        print("Sample kill data:")
        print(kills_df[['attacker_name', 'victim_name', 'attacker_side', 'victim_side', 'round_num']].head(10))
        print("\nChecking for bomb events...")
        if bomb_events is not None:
            print(f"Bomb events shape: {bomb_events.shape}")
            print("Bomb events columns:", list(bomb_events.columns))
            print("Sample bomb events:")
            print(bomb_events.head())
        else:
            print("No bomb events found in demo.bomb")
            # Try other possible bomb-related attributes
            bomb_attrs = [attr for attr in dir(demo) if 'bomb' in attr.lower()]
            print(f"Bomb-related attributes: {bomb_attrs}")
        print()
    
    # Get all rounds that had any activity
    all_rounds = set(kills_df['round_num'].unique())
    if bomb_events is not None:
        all_rounds.update(bomb_events['round_num'].unique())
    
    # Initialize round stats for ALL rounds (including empty ones)
    round_stats = {}
    for round_num in range(1, max(all_rounds) + 1):
        round_stats[round_num] = {
            'kills': [], 
            'deaths': [], 
            'bomb_planted': False, 
            'plant_tick': None,
            'kill_impacts': [],
            'death_impacts': []
        }
    
    # Process bomb events first to get plant information
    if bomb_events is not None and not bomb_events.empty:
        if debug:
            print(f"Debug: Found {len(bomb_events)} bomb events")
            print("Bomb event columns:", bomb_events.columns.tolist())
        
        for _, bomb_event in bomb_events.iterrows():
            round_num = bomb_event.get('round_num', 0)
            if round_num in round_stats:
                # Check for different possible bomb event indicators
                event_columns = bomb_event.index.tolist()
                event_values = bomb_event.values
                
                # Look for plant indicators in various ways - BE MORE STRICT
                is_plant = False
                plant_tick = None
                
                # Method 1: Check for exact 'plant' event type
                if 'event_type' in event_columns:
                    event_type = str(bomb_event['event_type']).lower()
                    if event_type == 'plant' or event_type == 'bomb_plant':
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant via event_type in round {round_num}")
                
                # Method 2: Check for bombsite column with actual plant event
                if 'site' in event_columns and 'event_type' in event_columns:
                    if 'plant' in str(bomb_event['event_type']).lower():
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant via bombsite+event_type in round {round_num}")
                
                # Method 3: Only if we have a specific bomb plant indicator
                for col, val in zip(event_columns, event_values):
                    if isinstance(val, str) and val.lower() in ['plant', 'bomb_plant', 'c4_plant']:
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant in round {round_num}, column {col}, value: {val}")
                        break
                
                if is_plant:
                    round_stats[round_num]['bomb_planted'] = True
                    round_stats[round_num]['plant_tick'] = plant_tick
    else:
        if debug:
            print("Debug: No bomb events found or bomb_events is None")
    
    # Process each round
    for round_num in sorted(round_stats.keys()):
        if round_num == 0:
            continue
            
        round_kills = kills_df[kills_df['round_num'] == round_num].sort_values('tick')
        
        # Start with standard 5v5
        ct_alive = 5
        t_alive = 5
        
        # Process kills in chronological order
        for _, kill in round_kills.iterrows():
            killer = kill.get('attacker_name', '')
            victim = kill.get('victim_name', '')
            weapon = kill.get('weapon', 'Unknown')
            victim_side = kill.get('victim_side', '')
            kill_tick = kill.get('tick', 0)
            
            # Check if this kill happened after bomb plant
            is_post_plant = (round_stats[round_num]['bomb_planted'] and 
                           round_stats[round_num]['plant_tick'] and 
                           kill_tick > round_stats[round_num]['plant_tick'])
            
            # Determine weapon category
            weapon_category = "rifle"
            if weapon in ['glock', 'usp_silencer', 'p2000', 'deagle', 'elite', 'fiveseven', 'cz75a', 'tec9', 'p250', 'revolver']:
                weapon_category = "pistol"
            elif weapon in ['ak47', 'm4a1', 'm4a1_s', 'aug', 'sg556', 'famas', 'galilar']:
                weapon_category = "rifle"
            elif weapon in ['awp', 'ssg08', 'scar20', 'g3sg1']:
                weapon_category = "sniper"
            
            # Current game state before this kill
            game_state = f"{ct_alive} v {t_alive}"
            if is_post_plant:
                game_state += " post plant"
            
            # Calculate impact score for this kill
            ct_after = ct_alive - 1 if victim_side == 'ct' else ct_alive
            t_after = t_alive - 1 if victim_side == 't' else t_alive
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record events for our target player
            if killer == player_name:
                weapon_info = f" ({weapon_category})" if weapon_category != 'rifle' else ""
                round_stats[round_num]['kills'].append(f"{game_state}{weapon_info}")
                round_stats[round_num]['kill_impacts'].append(impact)
            
            if victim == player_name:
                weapon_info = f" vs {weapon_category}" if weapon_category != 'rifle' else ""
                round_stats[round_num]['deaths'].append(f"{game_state}{weapon_info}")
                round_stats[round_num]['death_impacts'].append(-impact)  # Negative for deaths
            
            # Update alive counts after the kill
            if victim_side == 'ct':
                ct_alive = max(0, ct_alive - 1)
            elif victim_side == 't':
                t_alive = max(0, t_alive - 1)
    
    # Format results - NOW INCLUDING ALL ROUNDS
    results = []
    for round_num in sorted(round_stats.keys()):
        if round_num == 0:
            continue
            
        side = "CT" if round_num <= 12 else "T"
        if round_num > 24:  # Overtime
            side = "CT" if ((round_num - 25) // 6) % 2 == 0 else "T"
        
        # Format kills and deaths with impact scores
        kills_list = round_stats[round_num]['kills']
        deaths_list = round_stats[round_num]['deaths']
        kill_impacts = round_stats[round_num]['kill_impacts']
        death_impacts = round_stats[round_num]['death_impacts']
        
        # Add impact scores to the display
        kills_with_impact = []
        for i, kill in enumerate(kills_list):
            impact = kill_impacts[i] if i < len(kill_impacts) else 0
            # Truncate long descriptions for better formatting
            kill_display = kill[:30] + "..." if len(kill) > 33 else kill
            kills_with_impact.append(f"{kill_display} (+{impact})")
        
        deaths_with_impact = []
        for i, death in enumerate(deaths_list):
            impact = abs(death_impacts[i]) if i < len(death_impacts) else 0
            # Truncate long descriptions for better formatting
            death_display = death[:30] + "..." if len(death) > 33 else death
            deaths_with_impact.append(f"{death_display} (-{impact})")
        
        # Join multiple events with proper formatting
        kills_text = ' | '.join(kills_with_impact) if kills_with_impact else '-'
        deaths_text = ' | '.join(deaths_with_impact) if deaths_with_impact else '-'
        
        # Calculate round impact
        round_impact = sum(kill_impacts) + sum(death_impacts)
        
        # Add bomb plant indicator if relevant
        round_info = ""
        if round_stats[round_num]['bomb_planted']:
            round_info = " 💣"
        
        results.append({
            'Round': f"{round_num}{round_info}",
            'Side': side,
            'Kills': kills_text,
            'Deaths': deaths_text,
            'Impact': f"{round_impact:+.1f}"
        })
    
    return pd.DataFrame(results)

#%%
def plot_kill_death_analysis(stats_df, player_name):
    """
    Create bar plots for kill/death analysis showing:
    1. Kills and Deaths per round
    2. Impact scores per round
    """
    if stats_df is None or stats_df.empty:
        print("No data to plot")
        return
    
    # Extract data from the DataFrame
    rounds = []
    kills_count = []
    deaths_count = []
    impacts = []
    sides = []
    bomb_plants = []
    
    for _, row in stats_df.iterrows():
        round_str = str(row['Round'])
        bomb_planted = '💣' in round_str
        round_num = int(round_str.replace('💣', '').strip())
        
        rounds.append(round_num)
        sides.append(row['Side'])
        bomb_plants.append(bomb_planted)
        
        # Count kills and deaths (exclude '-' entries)
        kills = [k for k in str(row['Kills']).split(' | ') if k != '-' and k.strip()]
        deaths = [d for d in str(row['Deaths']).split(' | ') if d != '-' and d.strip()]
        
        kills_count.append(len(kills))
        deaths_count.append(len(deaths))
        
        # Extract impact value
        impact_str = str(row['Impact']).replace('+', '')
        try:
            impacts.append(float(impact_str))
        except:
            impacts.append(0.0)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Kills and Deaths per round
    x = np.arange(len(rounds))
    width = 0.35
    
    # Color code by side
    kill_colors = ['lightblue' if side == 'CT' else 'orange' for side in sides]
    death_colors = ['darkblue' if side == 'CT' else 'red' for side in sides]
    
    bars1 = ax1.bar(x - width/2, kills_count, width, label='Kills', color=kill_colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, deaths_count, width, label='Deaths', color=death_colors, alpha=0.8)
    
    # Add bomb plant indicators
    for i, (bomb, kill, death) in enumerate(zip(bomb_plants, kills_count, deaths_count)):
        if bomb:
            max_height = max(kill, death) + 0.5
            ax1.text(i, max_height, 'BOMB', ha='center', va='bottom', fontsize=10, 
                    color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{player_name} - Kills and Deaths per Round')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add side transition line
    for i in range(len(rounds)-1):
        if sides[i] != sides[i+1]:
            ax1.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 2: Impact scores per round
    impact_colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in impacts]
    bars3 = ax2.bar(x, impacts, color=impact_colors, alpha=0.7)
    
    # Add bomb plant indicators
    for i, bomb in enumerate(bomb_plants):
        if bomb:
            height = impacts[i] + (5 if impacts[i] >= 0 else -5)
            ax2.text(i, height, 'BOMB', ha='center', va='center', fontsize=10, 
                    color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Impact Score')
    ax2.set_title(f'{player_name} - Impact Score per Round')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add side transition line
    for i in range(len(rounds)-1):
        if sides[i] != sides[i+1]:
            ax2.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add side labels
    ct_rounds = [i for i, s in enumerate(sides) if s == 'CT']
    t_rounds = [i for i, s in enumerate(sides) if s == 'T']
    
    if ct_rounds:
        ax1.text(np.mean(ct_rounds), max(max(kills_count), max(deaths_count)) * 0.9, 
                'CT SIDE', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    if t_rounds:
        ax1.text(np.mean(t_rounds), max(max(kills_count), max(deaths_count)) * 0.9, 
                'T SIDE', ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    total_kills = sum(kills_count)
    total_deaths = sum(deaths_count)
    total_impact = sum(impacts)
    
    print(f"\n📊 PLOT SUMMARY:")
    print(f"Total Kills: {total_kills}")
    print(f"Total Deaths: {total_deaths}")
    print(f"Total Impact: {total_impact:+.1f} points")
    print(f"Average Impact per Round: {total_impact/len(rounds):+.1f}")

#%%
def plot_positive_negative_impact(stats_df, player_name):
    """
    Create specialized plots showing positive vs negative impact breakdown
    """
    if stats_df is None or stats_df.empty:
        print("No data to plot")
        return
    
    # Extract data from the DataFrame
    rounds = []
    positive_impacts = []
    negative_impacts = []
    sides = []
    bomb_plants = []
    kill_counts = []
    death_counts = []
    
    for _, row in stats_df.iterrows():
        round_str = str(row['Round'])
        bomb_planted = '💣' in round_str
        round_num = int(round_str.replace('💣', '').strip())
        
        rounds.append(round_num)
        sides.append(row['Side'])
        bomb_plants.append(bomb_planted)
        
        # Count kills and deaths
        kills = [k for k in str(row['Kills']).split(' | ') if k != '-' and k.strip()]
        deaths = [d for d in str(row['Deaths']).split(' | ') if d != '-' and d.strip()]
        kill_counts.append(len(kills))
        death_counts.append(len(deaths))
        
        # Extract impact value and separate positive/negative
        impact_str = str(row['Impact']).replace('+', '')
        try:
            impact = float(impact_str)
            positive_impacts.append(max(0, impact))
            negative_impacts.append(min(0, impact))
        except:
            positive_impacts.append(0.0)
            negative_impacts.append(0.0)
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    x = np.arange(len(rounds))
    
    # Plot 1: Positive vs Negative Impact Stacked
    pos_bars = ax1.bar(x, positive_impacts, label='Positive Impact', color='green', alpha=0.7)
    neg_bars = ax1.bar(x, negative_impacts, label='Negative Impact', color='red', alpha=0.7)
    
    # Add bomb indicators
    for i, bomb in enumerate(bomb_plants):
        if bomb:
            max_height = max(positive_impacts[i], abs(negative_impacts[i])) + 5
            ax1.text(i, max_height, 'BOMB', ha='center', va='center', fontsize=8, 
                    color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Impact Score')
    ax1.set_title(f'{player_name} - Positive vs Negative Impact per Round')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 2: Impact Distribution (Histogram)
    all_impacts = [p + n for p, n in zip(positive_impacts, negative_impacts)]
    pos_only = [p for p in positive_impacts if p > 0]
    neg_only = [n for n in negative_impacts if n < 0]
    
    ax2.hist(pos_only, bins=10, alpha=0.7, color='green', label=f'Positive ({len(pos_only)} rounds)', edgecolor='black')
    ax2.hist(neg_only, bins=10, alpha=0.7, color='red', label=f'Negative ({len(neg_only)} rounds)', edgecolor='black')
    ax2.set_xlabel('Impact Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{player_name} - Impact Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Impact over Time
    cumulative_impact = np.cumsum(all_impacts)
    colors = ['green' if val >= 0 else 'red' for val in cumulative_impact]
    
    ax3.plot(x, cumulative_impact, marker='o', linewidth=2, markersize=4, color='navy')
    ax3.fill_between(x, 0, cumulative_impact, alpha=0.3, color='lightblue')
    
    # Color points based on positive/negative
    for i, (xi, yi) in enumerate(zip(x, cumulative_impact)):
        color = 'green' if all_impacts[i] > 0 else 'red' if all_impacts[i] < 0 else 'gray'
        ax3.scatter(xi, yi, color=color, s=30, zorder=5)
    
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative Impact')
    ax3.set_title(f'{player_name} - Cumulative Impact Progression')
    ax3.set_xticks(x)
    ax3.set_xticklabels(rounds)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Plot 4: Impact vs Performance (K-D) correlation
    kd_diff = [k - d for k, d in zip(kill_counts, death_counts)]
    
    # Separate into positive and negative impact rounds
    pos_kd = [kd for kd, imp in zip(kd_diff, all_impacts) if imp > 0]
    pos_imp = [imp for imp in all_impacts if imp > 0]
    neg_kd = [kd for kd, imp in zip(kd_diff, all_impacts) if imp < 0]
    neg_imp = [imp for imp in all_impacts if imp < 0]
    neutral_kd = [kd for kd, imp in zip(kd_diff, all_impacts) if imp == 0]
    
    if pos_kd and pos_imp:
        ax4.scatter(pos_kd, pos_imp, color='green', alpha=0.7, s=50, label='Positive Impact Rounds')
    if neg_kd and neg_imp:
        ax4.scatter(neg_kd, neg_imp, color='red', alpha=0.7, s=50, label='Negative Impact Rounds')
    if neutral_kd:
        ax4.scatter(neutral_kd, [0]*len(neutral_kd), color='gray', alpha=0.7, s=50, label='Neutral Rounds')
    
    ax4.set_xlabel('Kill-Death Difference')
    ax4.set_ylabel('Impact Score')
    ax4.set_title(f'{player_name} - K-D vs Impact Correlation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add side transition lines to relevant plots
    for ax in [ax1, ax3]:
        for i in range(len(rounds)-1):
            if sides[i] != sides[i+1]:
                ax.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed impact analysis
    total_positive = sum(pos_only) if pos_only else 0
    total_negative = sum(neg_only) if neg_only else 0
    positive_rounds = len(pos_only)
    negative_rounds = len(neg_only)
    neutral_rounds = len(rounds) - positive_rounds - negative_rounds
    
    print(f"\n📊 DETAILED IMPACT ANALYSIS for {player_name}:")
    print(f"{'Metric':<25} {'Value':<15} {'Percentage':<15}")
    print("─" * 55)
    print(f"{'Positive Impact Rounds:':<25} {positive_rounds:<15} {positive_rounds/len(rounds)*100:.1f}%")
    print(f"{'Negative Impact Rounds:':<25} {negative_rounds:<15} {negative_rounds/len(rounds)*100:.1f}%")
    print(f"{'Neutral Rounds:':<25} {neutral_rounds:<15} {neutral_rounds/len(rounds)*100:.1f}%")
    print(f"{'Total Positive Impact:':<25} {total_positive:<15.1f}")
    print(f"{'Total Negative Impact:':<25} {total_negative:<15.1f}")
    print(f"{'Net Impact:':<25} {total_positive + total_negative:<15.1f}")
    
    if positive_rounds > 0:
        print(f"{'Avg Positive Impact:':<25} {total_positive/positive_rounds:<15.1f}")
    if negative_rounds > 0:
        print(f"{'Avg Negative Impact:':<25} {total_negative/negative_rounds:<15.1f}")
    
    # Find best and worst rounds
    if all_impacts:
        best_round_idx = all_impacts.index(max(all_impacts))
        worst_round_idx = all_impacts.index(min(all_impacts))
        
        print(f"\n🎯 KEY ROUNDS:")
        print(f"Best Round: {rounds[best_round_idx]} (Impact: +{all_impacts[best_round_idx]:.1f})")
        print(f"Worst Round: {rounds[worst_round_idx]} (Impact: {all_impacts[worst_round_idx]:+.1f})")

#%%
def compare_players_impact(dem, player1_name, player2_name):
    """
    Compare two players' impact scores side by side
    """
    print(f"\n🔍 Comparing {player1_name} vs {player2_name}")
    
    # Get stats for both players
    stats1 = get_player_kill_death_analysis(dem, player1_name, debug=False)
    stats2 = get_player_kill_death_analysis(dem, player2_name, debug=False)
    
    if stats1 is None or stats1.empty or stats2 is None or stats2.empty:
        print("❌ Could not get data for one or both players")
        return
    
    # Extract data for both players
    def extract_player_data(stats_df, player_name):
        rounds = []
        impacts = []
        sides = []
        bomb_plants = []
        kills_count = []
        deaths_count = []
        
        for _, row in stats_df.iterrows():
            round_str = str(row['Round'])
            bomb_planted = '💣' in round_str
            round_num = int(round_str.replace('💣', '').strip())
            
            rounds.append(round_num)
            sides.append(row['Side'])
            bomb_plants.append(bomb_planted)
            
            # Count kills and deaths
            kills = [k for k in str(row['Kills']).split(' | ') if k != '-' and k.strip()]
            deaths = [d for d in str(row['Deaths']).split(' | ') if d != '-' and d.strip()]
            
            kills_count.append(len(kills))
            deaths_count.append(len(deaths))
            
            # Extract impact value
            impact_str = str(row['Impact']).replace('+', '')
            try:
                impacts.append(float(impact_str))
            except:
                impacts.append(0.0)
        
        return {
            'rounds': rounds,
            'impacts': impacts,
            'sides': sides,
            'bomb_plants': bomb_plants,
            'kills_count': kills_count,
            'deaths_count': deaths_count,
            'total_impact': sum(impacts),
            'total_kills': sum(kills_count),
            'total_deaths': sum(deaths_count)
        }
    
    data1 = extract_player_data(stats1, player1_name)
    data2 = extract_player_data(stats2, player2_name)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Plot 1: Impact scores comparison
    x = np.arange(len(data1['rounds']))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, data1['impacts'], width, label=f'{player1_name}', 
                   color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, data2['impacts'], width, label=f'{player2_name}', 
                   color='orangered', alpha=0.8)
    
    # Add bomb plant indicators
    for i, (bomb1, bomb2) in enumerate(zip(data1['bomb_plants'], data2['bomb_plants'])):
        if bomb1 or bomb2:
            max_impact = max(abs(data1['impacts'][i]), abs(data2['impacts'][i]))
            height = max_impact + 10
            ax1.text(i, height, 'BOMB', ha='center', va='center', fontsize=8, 
                    color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Impact Score')
    ax1.set_title(f'Impact Score Comparison: {player1_name} vs {player2_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(data1['rounds'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add side transition lines
    for i in range(len(data1['rounds'])-1):
        if data1['sides'][i] != data1['sides'][i+1]:
            ax1.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Plot 2: Kills and Deaths comparison
    kills1 = data1['kills_count']
    deaths1 = data1['deaths_count']
    kills2 = data2['kills_count']
    deaths2 = data2['deaths_count']
    
    # Calculate KD difference per round
    kd_diff1 = [k - d for k, d in zip(kills1, deaths1)]
    kd_diff2 = [k - d for k, d in zip(kills2, deaths2)]
    
    bars3 = ax2.bar(x - width/2, kd_diff1, width, label=f'{player1_name} (K-D)', 
                   color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, kd_diff2, width, label=f'{player2_name} (K-D)', 
                   color='orangered', alpha=0.8)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Kill-Death Difference')
    ax2.set_title(f'Kill-Death Difference per Round: {player1_name} vs {player2_name}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(data1['rounds'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add side transition lines
    for i in range(len(data1['rounds'])-1):
        if data1['sides'][i] != data1['sides'][i+1]:
            ax2.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add side labels
    ct_rounds = [i for i, s in enumerate(data1['sides']) if s == 'CT']
    t_rounds = [i for i, s in enumerate(data1['sides']) if s == 'T']
    
    if ct_rounds:
        ax1.text(np.mean(ct_rounds), ax1.get_ylim()[1] * 0.9, 
                'CT SIDE', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    if t_rounds:
        ax1.text(np.mean(t_rounds), ax1.get_ylim()[1] * 0.9, 
                'T SIDE', ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"{'Metric':<20} {player1_name:<15} {player2_name:<15} {'Difference':<15}")
    print("─" * 70)
    print(f"{'Total Impact:':<20} {data1['total_impact']:+8.1f}      {data2['total_impact']:+8.1f}      {data1['total_impact']-data2['total_impact']:+8.1f}")
    print(f"{'Total Kills:':<20} {data1['total_kills']:8}      {data2['total_kills']:8}      {data1['total_kills']-data2['total_kills']:+8}")
    print(f"{'Total Deaths:':<20} {data1['total_deaths']:8}      {data2['total_deaths']:8}      {data1['total_deaths']-data2['total_deaths']:+8}")
    
    kd1 = data1['total_kills'] / max(data1['total_deaths'], 1)
    kd2 = data2['total_kills'] / max(data2['total_deaths'], 1)
    print(f"{'K/D Ratio:':<20} {kd1:8.2f}      {kd2:8.2f}      {kd1-kd2:+8.2f}")
    
    avg_impact1 = data1['total_impact'] / len(data1['rounds'])
    avg_impact2 = data2['total_impact'] / len(data2['rounds'])
    print(f"{'Avg Impact/Round:':<20} {avg_impact1:+8.1f}      {avg_impact2:+8.1f}      {avg_impact1-avg_impact2:+8.1f}")

#%%
def create_probability_scenarios_table():
    """
    Create a table showing win probabilities for different scenarios.
    This helps understand how the impact scoring system works.
    """
    print("╔" + "═" * 78 + "╗")
    print("║" + " CS2 WIN PROBABILITY SCENARIOS ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Use the main get_win_probability function from calculate_impact_score
    
    scenarios = [
        # Normal scenarios
        (5, 5, False, "Full teams"),
        (5, 4, False, "CT numerical advantage"),
        (4, 5, False, "T numerical advantage"),
        (4, 4, False, "Even teams"),
        (3, 3, False, "Mid-round even"),
        (2, 2, False, "Late round even"),
        (1, 1, False, "1v1 clutch"),
        
        # Lopsided scenarios
        (5, 1, False, "CT overwhelming advantage"),
        (1, 5, False, "T overwhelming advantage"),
        (3, 1, False, "CT strong advantage"),
        (1, 3, False, "T strong advantage"),
        (2, 1, False, "CT advantage"),
        (1, 2, False, "T advantage"),
        
        # Post-plant scenarios
        (5, 5, True, "Full teams, bomb planted"),
        (4, 4, True, "Even teams, bomb planted"),
        (3, 3, True, "Mid-round even, bomb planted"),
        (2, 2, True, "Late round even, bomb planted"),
        (1, 1, True, "1v1 clutch, bomb planted"),
        (3, 1, True, "CT defending bomb site"),
        (1, 3, True, "T protecting bomb"),
        (2, 1, True, "CT retake scenario"),
        (1, 2, True, "T post-plant advantage"),
    ]
    
    print(f"{'Scenario':<25} {'CT':<3} {'T':<3} {'Post-Plant':<11} {'CT Win %':<8} {'Impact Notes'}")
    print("─" * 85)
    
    for ct, t, post_plant, description in scenarios:
        prob = get_win_probability(ct, t, post_plant)
        post_plant_str = "Yes" if post_plant else "No"
        
        # Calculate what impact a kill would have
        # Example: if CT kills T player
        if t > 0:
            prob_after_ct_kill = get_win_probability(ct, t-1, post_plant)
            impact_ct_kill = abs(prob_after_ct_kill - prob) * 100
        else:
            impact_ct_kill = 0
            
        impact_note = f"CT 1-kill impact: ~{impact_ct_kill:.1f}"
        
        print(f"{description:<25} {ct:<3} {t:<3} {post_plant_str:<11} {prob*100:>6.1f}%  {impact_note}")
    
    print("─" * 85)
    print("\nKey Insights:")
    print("• Higher impact when teams are evenly matched (5v5, 4v4)")
    print("• Lower impact in lopsided situations (5v1, 1v5)")
    print("• Post-plant scenarios favor Terrorists")
    print("• 'CT 1-kill impact' = impact of ONE CT kill on win probability")
    print("• Impact = |Win% After - Win% Before| × 100")
    print()

#%%
# Main execution - analyze player performance
player_name = "mezii"  # Change this to the player you want to analyze

# Get available player names first
print("Available players in demo:")
available_players = set()
if hasattr(dem, 'kills') and dem.kills is not None:
    kills_data = dem.kills
    if hasattr(kills_data, 'to_pandas'):
        kills_data = kills_data.to_pandas()
    
    for _, kill in kills_data.iterrows():
        if kill.get('attacker_name'):
            available_players.add(kill.get('attacker_name'))
        if kill.get('victim_name'):
            available_players.add(kill.get('victim_name'))

for player in sorted(available_players):
    print(f"• {player}")
print()

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
player2 = "bLitz"  

compare_players_impact(dem, player1, player2)

# %%
