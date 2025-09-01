"""
Plotting functions for CS2 demo analysis visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration - easily adjustable axis limits
DEFAULT_AXIS_LIMIT = 70  # Default minimum axis limit for impact plots


def plot_kill_death_analysis(stats_df, player_name):
    """
    Create bar plots for kill/death analysis showing:
    1. Kills and Deaths per round
    2. Impact scores per round
    """
    if stats_df is None or len(stats_df) == 0:
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
    ax2.set_ylabel('Round swing')
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
    
    # Print summary statistics
    total_kills = sum(kills_count)
    total_deaths = sum(deaths_count)
    total_impact = sum(impacts)
    
    print(f"\n📊 PLOT SUMMARY:")
    print(f"Total Kills: {total_kills}")
    print(f"Total Deaths: {total_deaths}")
    print(f"Total Impact: {total_impact:+.1f} points")
    print(f"Average Impact per Round: {total_impact/len(rounds):+.1f}")


def plot_positive_negative_impact(stats_df, player_name):
    """
    Create specialized plots showing positive vs negative impact breakdown
    """
    if stats_df is None or len(stats_df) == 0:
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
    ax1.set_ylabel('Round swing')
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
    ax2.set_xlabel('Round swing')
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
    ax4.set_ylabel('Round swing')
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


def plot_impact_difference_per_round(stats_df, player_name):
    """
    Create a specialized plot showing impact difference for each round.
    Displays positive and negative impact clearly with detailed breakdown.
    """
    if stats_df is None or len(stats_df) == 0:
        print("No data to plot")
        return
    
    # Extract data from the DataFrame
    rounds = []
    impacts = []
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
        
        # Extract impact value
        impact_str = str(row['Impact']).replace('+', '')
        try:
            impacts.append(float(impact_str))
        except:
            impacts.append(0.0)
    
    # Create the main figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
    
    x = np.arange(len(rounds))
    
    # Plot 1: Impact Difference - Waterfall Style
    positive_impacts = [max(0, imp) for imp in impacts]
    negative_impacts = [min(0, imp) for imp in impacts]
    
    # Create waterfall-style bars
    pos_bars = ax1.bar(x, positive_impacts, color='green', alpha=0.8, label='Positive Impact')
    neg_bars = ax1.bar(x, negative_impacts, color='red', alpha=0.8, label='Negative Impact')
    
    # Add value labels on bars
    for i, (pos, neg) in enumerate(zip(positive_impacts, negative_impacts)):
        if pos > 0:
            ax1.text(i, pos + 1, f'+{pos:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        if neg < 0:
            ax1.text(i, neg - 1, f'{neg:.1f}', ha='center', va='top', fontweight='bold', fontsize=9)
    
    # Add bomb plant indicators
    for i, bomb in enumerate(bomb_plants):
        if bomb:
            max_height = max(abs(impacts[i]), 5) + 8
            ax1.text(i, max_height, '💣', ha='center', va='center', fontsize=14)
    
    # Add side transitions and labels
    for i in range(len(rounds)-1):
        if sides[i] != sides[i+1]:
            ax1.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add side background colors
    ct_indices = [i for i, s in enumerate(sides) if s == 'CT']
    t_indices = [i for i, s in enumerate(sides) if s == 'T']
    
    if ct_indices:
        ax1.axvspan(min(ct_indices)-0.5, max(ct_indices)+0.5, alpha=0.1, color='blue', label='CT Side')
    if t_indices:
        ax1.axvspan(min(t_indices)-0.5, max(t_indices)+0.5, alpha=0.1, color='orange', label='T Side')
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Round swing')
    ax1.set_title(f'{player_name} - Impact Difference per Round (Positive vs Negative)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rounds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Plot 2: Cumulative Impact Progression
    cumulative_impact = np.cumsum(impacts)
    
    # Color the line based on current cumulative value
    colors = ['green' if val >= 0 else 'red' for val in cumulative_impact]
    
    # Plot the line
    ax2.plot(x, cumulative_impact, linewidth=3, color='navy', marker='o', markersize=6)
    
    # Fill area based on positive/negative
    ax2.fill_between(x, 0, cumulative_impact, 
                     where=[cum >= 0 for cum in cumulative_impact], 
                     color='green', alpha=0.3, interpolate=True, label='Positive Total')
    ax2.fill_between(x, 0, cumulative_impact, 
                     where=[cum < 0 for cum in cumulative_impact], 
                     color='red', alpha=0.3, interpolate=True, label='Negative Total')
    
    # Add value labels at key points
    for i in range(0, len(x), max(1, len(x)//10)):  # Show every nth point
        ax2.text(i, cumulative_impact[i] + (2 if cumulative_impact[i] >= 0 else -2), 
                f'{cumulative_impact[i]:+.1f}', ha='center', 
                va='bottom' if cumulative_impact[i] >= 0 else 'top', fontsize=8)
    
    # Add side transitions
    for i in range(len(rounds)-1):
        if sides[i] != sides[i+1]:
            ax2.axvline(x=i+0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Cumulative Impact')
    ax2.set_title(f'{player_name} - Cumulative Impact Progression')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rounds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    # Plot 3: Impact Distribution and Statistics
    # Create histogram of impact values
    pos_impacts_only = [imp for imp in impacts if imp > 0]
    neg_impacts_only = [imp for imp in impacts if imp < 0]
    neutral_rounds = len([imp for imp in impacts if imp == 0])
    
    # Create bins for histogram
    all_impacts = [imp for imp in impacts if imp != 0]
    if all_impacts:
        bins = np.linspace(min(all_impacts), max(all_impacts), 15)
        
        ax3.hist(pos_impacts_only, bins=bins[bins >= 0], alpha=0.7, color='green', 
                label=f'Positive Rounds ({len(pos_impacts_only)})', edgecolor='black')
        ax3.hist(neg_impacts_only, bins=bins[bins <= 0], alpha=0.7, color='red', 
                label=f'Negative Rounds ({len(neg_impacts_only)})', edgecolor='black')
    
    # Add statistics text box
    total_positive = sum(pos_impacts_only) if pos_impacts_only else 0
    total_negative = sum(neg_impacts_only) if neg_impacts_only else 0
    net_impact = total_positive + total_negative
    
    stats_text = f"""IMPACT STATISTICS:
Positive Rounds: {len(pos_impacts_only)} ({len(pos_impacts_only)/len(rounds)*100:.1f}%)
Negative Rounds: {len(neg_impacts_only)} ({len(neg_impacts_only)/len(rounds)*100:.1f}%)
Neutral Rounds: {neutral_rounds} ({neutral_rounds/len(rounds)*100:.1f}%)

Total Positive: +{total_positive:.1f}
Total Negative: {total_negative:.1f}
Net Impact: {net_impact:+.1f}

Avg Per Round: {net_impact/len(rounds):+.1f}"""
    
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax3.set_xlabel('Round swing')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{player_name} - Impact Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    
    plt.tight_layout()
    
    # Print detailed round-by-round analysis
    print(f"\n📊 DETAILED ROUND IMPACT ANALYSIS for {player_name}")
    print("=" * 80)
    
    # Find most impactful rounds
    if impacts:
        best_round_idx = impacts.index(max(impacts))
        worst_round_idx = impacts.index(min(impacts))
        
        print(f"\n🎯 EXTREME ROUNDS:")
        print(f"Best Round:  #{rounds[best_round_idx]} (Impact: +{impacts[best_round_idx]:.1f}) - {sides[best_round_idx]} side")
        print(f"Worst Round: #{rounds[worst_round_idx]} (Impact: {impacts[worst_round_idx]:+.1f}) - {sides[worst_round_idx]} side")
    
    # Analyze by side
    ct_impacts = [impacts[i] for i, side in enumerate(sides) if side == 'CT']
    t_impacts = [impacts[i] for i, side in enumerate(sides) if side == 'T']
    
    if ct_impacts and t_impacts:
        print(f"\n⚔️  SIDE PERFORMANCE:")
        print(f"CT Side: {sum(ct_impacts):+.1f} total, {sum(ct_impacts)/len(ct_impacts):+.1f} avg per round")
        print(f"T Side:  {sum(t_impacts):+.1f} total, {sum(t_impacts)/len(t_impacts):+.1f} avg per round")
    
    # Analyze bomb round performance
    bomb_round_impacts = [impacts[i] for i, bomb in enumerate(bomb_plants) if bomb]
    regular_round_impacts = [impacts[i] for i, bomb in enumerate(bomb_plants) if not bomb]
    
    if bomb_round_impacts:
        print(f"\n💣 BOMB ROUND ANALYSIS:")
        print(f"Bomb Rounds: {sum(bomb_round_impacts):+.1f} total, {sum(bomb_round_impacts)/len(bomb_round_impacts):+.1f} avg per round")
        if regular_round_impacts:
            print(f"Regular Rounds: {sum(regular_round_impacts):+.1f} total, {sum(regular_round_impacts)/len(regular_round_impacts):+.1f} avg per round")
    
    print(f"\n📈 IMPACT TREND:")
    if len(impacts) >= 3:
        early_rounds = impacts[:len(impacts)//3]
        mid_rounds = impacts[len(impacts)//3:2*len(impacts)//3]
        late_rounds = impacts[2*len(impacts)//3:]
        
        print(f"Early Game: {sum(early_rounds)/len(early_rounds):+.1f} avg per round")
        print(f"Mid Game:   {sum(mid_rounds)/len(mid_rounds):+.1f} avg per round") 
        print(f"Late Game:  {sum(late_rounds)/len(late_rounds):+.1f} avg per round")


def plot_individual_impacts_by_round(dem, player_name):
    """
    Plot individual impacts grouped by round with clear round numbers.
    Shows each kill/death as a separate bar within each round.
    """
    from ..core.analysis import get_win_probability, calculate_impact_score
    from collections import defaultdict
    
    kills_df = dem.kills
    bomb_events = dem.bomb
    
    if kills_df is None or len(kills_df) == 0:
        print("❌ No kill data found in demo")
        return
    
    # Convert to pandas if needed
    if hasattr(kills_df, 'to_pandas'):
        kills_df = kills_df.to_pandas()
    if bomb_events is not None and hasattr(bomb_events, 'to_pandas'):
        bomb_events = bomb_events.to_pandas()
    
    # Get bomb plant information
    bomb_plants = {}
    if bomb_events is not None and len(bomb_events) > 0:
        for _, bomb_event in bomb_events.iterrows():
            round_num = bomb_event.get('round_num', 0)
            event_columns = bomb_event.index.tolist()
            
            # Check for plant events
            is_plant = False
            plant_tick = None
            
            if 'event_type' in event_columns:
                event_type = str(bomb_event['event_type']).lower()
                if event_type == 'plant' or event_type == 'bomb_plant':
                    is_plant = True
                    plant_tick = bomb_event.get('tick', 0)
            
            if is_plant:
                bomb_plants[round_num] = plant_tick
    
    # Extract individual impacts by round
    round_impacts = defaultdict(list)
    
    # Get all rounds with activity
    all_rounds = sorted(set(kills_df['round_num'].unique()))
    
    for round_num in all_rounds:
        if round_num == 0:
            continue
            
        round_kills = kills_df[kills_df['round_num'] == round_num].sort_values('tick')
        
        # Track alive players for this round
        ct_alive = 5
        t_alive = 5
        
        # Check if bomb was planted in this round
        plant_tick = bomb_plants.get(round_num, None)
        
        for _, kill in round_kills.iterrows():
            killer = kill.get('attacker_name', '')
            victim = kill.get('victim_name', '')
            victim_side = kill.get('victim_side', '')
            kill_tick = kill.get('tick', 0)
            
            # Check if this kill happened after bomb plant
            is_post_plant = (plant_tick is not None and kill_tick > plant_tick)
            
            # Current game state before this kill
            game_state = ""
            
            # Calculate impact score for this kill
            ct_after = ct_alive - 1 if victim_side == 'ct' else ct_alive
            t_after = t_alive - 1 if victim_side == 't' else t_alive
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record events for our target player
            if killer == player_name:
                attacker_side = kill.get('attacker_side', '')
                if attacker_side.lower() == 'ct':
                    game_state = f"{ct_alive}v{t_alive}"
                else:
                    game_state = f"{t_alive}v{ct_alive}"
                if is_post_plant:
                    game_state += " post-plant"

                round_impacts[round_num].append({
                    'type': 'kill',
                    'impact': impact,
                    'game_state': game_state,
                    'opponent': victim,
                    'post_plant': is_post_plant
                })
            
            if victim == player_name:
                if victim_side.lower() == 'ct':
                    game_state = f"{ct_alive}v{t_alive}"
                else:
                    game_state = f"{t_alive}v{ct_alive}"
                if is_post_plant:
                    game_state += " post-plant"

                round_impacts[round_num].append({
                    'type': 'death',
                    'impact': -impact,  # Negative for deaths
                    'game_state': game_state,
                    'opponent': killer,
                    'post_plant': is_post_plant
                })
            
            # Update alive counts after the kill
            if victim_side == 'ct':
                ct_alive = max(0, ct_alive - 1)
            elif victim_side == 't':
                t_alive = max(0, t_alive - 1)
    
    if not round_impacts:
        print(f"❌ No individual impacts found for player: {player_name}")
        return
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Prepare data for plotting
    sorted_rounds = sorted(round_impacts.keys())
    x_positions = []
    impact_values = []
    colors = []
    round_labels = []
    
    x_pos = 0
    round_x_centers = []
    
    for round_num in sorted_rounds:
        impacts = round_impacts[round_num]
        round_start = x_pos
        
        for i, event in enumerate(impacts):
            x_positions.append(x_pos)
            impact_values.append(event['impact'])
            colors.append('green' if event['impact'] > 0 else 'red')
            round_labels.append(f"R{round_num}")
            x_pos += 1
        
        # Calculate center position for round label
        round_center = (round_start + x_pos - 1) / 2
        round_x_centers.append((round_center, round_num, len(impacts)))
        
        # Add space between rounds
        x_pos += 1
    
    # Create the bar plot
    bars = ax.bar(x_positions, impact_values, color=colors, alpha=0.7, width=0.8)
    
    # Add impact value and game state labels on bars
    event_index = 0
    for round_num in sorted_rounds:
        impacts = round_impacts[round_num]
        for event in impacts:
            pos = x_positions[event_index]
            impact = impact_values[event_index]
            game_state = event['game_state']
            
            # Simplify post-plant notation
            if 'post-plant' in game_state:
                game_state_short = game_state.replace(' post-plant', ' (post)')
            else:
                game_state_short = game_state
            
            # Combined label: impact value + game state
            combined_label = f'{impact:+.1f}\n{game_state_short}'
            
            ax.text(pos, impact + (2.5 if impact > 0 else -2.5), combined_label, 
                    ha='center', va='bottom' if impact > 0 else 'top', 
                    fontsize=8, fontweight='bold', color='black')
            
            event_index += 1
    
    # Add delta impact labels at the bottom of each round
    x_pos = 0
    for round_num in sorted_rounds:
        impacts = round_impacts[round_num]
        if impacts:
            # Calculate total impact for this round
            round_total = sum(event['impact'] for event in impacts)
            
            # Position at the true center of the round's width
            round_start_x = x_pos
            round_end_x = x_pos + len(impacts) - 1
            round_center = (round_start_x + round_end_x) / 2
            
            # Add delta label at the bottom
            ax.text(round_start_x, -60, f'Δ{round_total:+.1f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold', 
                    color='darkgreen' if round_total > 0 else 'darkred')
        
        x_pos += len(impacts) + 1
    
    # Add vertical separators between rounds
    separator_positions = []
    x_pos = 0
    for round_num in sorted_rounds:
        impacts = round_impacts[round_num]
        x_pos += len(impacts)
        if round_num != sorted_rounds[-1]:  # Don't add separator after last round
            separator_positions.append(x_pos + 0.5)
        x_pos += 1
    
    for sep_pos in separator_positions:
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Individual Events by Round')
    ax.set_ylabel('Round swing')
    ax.set_title(f'{player_name} - Individual Kill/Death Impacts by Round')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Set symmetric y-axis with minimum span of -DEFAULT_AXIS_LIMIT to +DEFAULT_AXIS_LIMIT
    if impact_values:
        max_positive = max([val for val in impact_values if val > 0] + [0])
        max_negative = abs(min([val for val in impact_values if val < 0] + [0]))
        max_abs = max(max_positive, max_negative, DEFAULT_AXIS_LIMIT)  # Ensure minimum of DEFAULT_AXIS_LIMIT
        ax.set_ylim(-max_abs, max_abs)
    else:
        ax.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
    
    # Add round labels at the top - now that y-limits are set
    for center_x, round_num, event_count in round_x_centers:
        ax.text(center_x, ax.get_ylim()[1] * 0.95, f'{round_num}', 
                ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Remove x-axis tick labels since we have round labels
    ax.set_xticks([])
    
    # Adjust layout to accommodate top labels - set top margin and then tight layout
    plt.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Print round-by-round statistics
    print(f"\n🎯 ROUND-BY-ROUND IMPACT ANALYSIS for {player_name}")
    print("=" * 80)
    
    total_rounds_with_activity = len(round_impacts)
    total_events = sum(len(impacts) for impacts in round_impacts.values())
    total_impact = sum(event['impact'] for impacts in round_impacts.values() for event in impacts)
    
    print(f"Rounds with Activity: {total_rounds_with_activity}")
    print(f"Total Events: {total_events}")
    print(f"Net Impact: {total_impact:+.1f}")
    print(f"Average Impact per Event: {total_impact/total_events:+.1f}")
    print()
    
    # Show details for each round
    for round_num in sorted_rounds:
        impacts = round_impacts[round_num]
        round_total = sum(event['impact'] for event in impacts)
        kills = [e for e in impacts if e['type'] == 'kill']
        deaths = [e for e in impacts if e['type'] == 'death']
        
        print(f"Round {round_num:2d}: {round_total:+6.1f} total "
              f"({len(kills)} kills, {len(deaths)} deaths)")
        
        for event in impacts:
            symbol = "🎯" if event['type'] == 'kill' else "💀"
            post_plant = " [POST-PLANT]" if event['post_plant'] else ""
            print(f"  {symbol} {event['type'].upper():5s} vs {event['opponent']:<15s} "
                  f"({event['game_state']:<12s}){post_plant} = {event['impact']:+5.1f}")
        print()
    
    return round_impacts


def compare_individual_impacts(dem, player1_name, player2_name):
    """
    Compare individual kill/death impacts between two players using the same plot style.
    Shows detailed round-by-round comparison with individual events.
    """
    print(f"\n🔍 Comparing Individual Impacts: {player1_name} vs {player2_name}")
    
    # Get individual impacts for both players 
    impacts1 = get_individual_impacts_data(dem, player1_name)
    impacts2 = get_individual_impacts_data(dem, player2_name)
    
    if not impacts1 or not impacts2:
        print("❌ Could not get individual impact data for one or both players")
        return
    
    print(f"✅ Found {len(impacts1)} events for {player1_name}")
    print(f"✅ Found {len(impacts2)} events for {player2_name}")
    
    # Create side-by-side plots using the same style as plot_individual_impacts_by_round
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    def create_individual_impact_plot(impacts, ax, player_name, set_ylim=True):
        """Create the same individual impact plot as plot_individual_impacts_by_round"""
        # Group impacts by round
        round_impacts = defaultdict(list)
        for event in impacts:
            round_impacts[event['round']].append(event)
        
        # Prepare data for plotting
        sorted_rounds = sorted(round_impacts.keys())
        x_positions = []
        impact_values = []
        colors = []
        round_labels = []
        
        x_pos = 0
        round_x_centers = []
        
        for round_num in sorted_rounds:
            impacts_in_round = round_impacts[round_num]
            round_start = x_pos
            
            for i, event in enumerate(impacts_in_round):
                x_positions.append(x_pos)
                impact_values.append(event['impact'])
                colors.append('green' if event['impact'] > 0 else 'red')
                round_labels.append(f"R{round_num}")
                x_pos += 1
            
            # Calculate center position for round label
            round_center = (round_start + x_pos - 1) / 2
            round_x_centers.append((round_center, round_num, len(impacts_in_round)))
            
            # Add space between rounds
            x_pos += 1
        
        # Create the bar plot
        bars = ax.bar(x_positions, impact_values, color=colors, alpha=0.7, width=0.8)
        
        # Only set y-limits if requested (for single player mode)
        if set_ylim and impact_values:
            max_positive = max([val for val in impact_values if val > 0] + [0])
            max_negative = abs(min([val for val in impact_values if val < 0] + [0]))
            max_abs = max(max_positive, max_negative, DEFAULT_AXIS_LIMIT)  # Minimum of DEFAULT_AXIS_LIMIT, but allow growth
            ax.set_ylim(-max_abs, max_abs)
        elif set_ylim:
            ax.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
        
        # Add impact value and game state labels on bars
        event_index = 0
        for round_num in sorted_rounds:
            impacts_in_round = round_impacts[round_num]
            for event in impacts_in_round:
                pos = x_positions[event_index]
                impact = impact_values[event_index]
                game_state = event['game_state']
                
                # Simplify post-plant notation
                if 'post-plant' in game_state:
                    game_state_short = game_state.replace(' post-plant', ' (post)')
                else:
                    game_state_short = game_state
                
                # Combined label: impact value + game state
                combined_label = f'{impact:+.1f}\n{game_state_short}'
                
                ax.text(pos, impact + (2.2 if impact > 0 else -2.2), combined_label, 
                        ha='center', va='bottom' if impact > 0 else 'top', 
                        fontsize=7, fontweight='bold', color='black')
                
                event_index += 1
        
        # Add delta impact labels at the bottom of each round
        x_pos = 0
        for round_num in sorted_rounds:
            impacts_in_round = round_impacts[round_num]
            if impacts_in_round:
                # Calculate total impact for this round
                round_total = sum(event['impact'] for event in impacts_in_round)
                
                # Position at the true center of the round's width
                round_start_x = x_pos
                round_end_x = x_pos + len(impacts_in_round) - 1
                round_center = (round_start_x + round_end_x) / 2
                
                # Add delta label at the bottom
                ax.text(round_center - 2, -55, f'Δ{round_total:+.1f}', 
                        ha='center', va='center', fontsize=11, fontweight='bold', 
                        color='darkgreen' if round_total > 0 else 'darkred')
            
            x_pos += len(impacts_in_round) + 1
        
        # Add vertical separators between rounds
        separator_positions = []
        x_pos = 0
        for round_num in sorted_rounds:
            impacts_in_round = round_impacts[round_num]
            x_pos += len(impacts_in_round)
            if round_num != sorted_rounds[-1]:  # Don't add separator after last round
                separator_positions.append(x_pos + 0.5)
            x_pos += 1
        
        for sep_pos in separator_positions:
            ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Individual Events by Round')
        ax.set_ylabel('Round swing')
        ax.set_title(f'{player_name} - Individual Kill/Death Impacts by Round')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Remove x-axis tick labels since we have round labels
        ax.set_xticks([])
        
        return impact_values, round_x_centers
    
    # Create plots for both players (don't set y-limits yet)
    impact_values1, round_centers1 = create_individual_impact_plot(impacts1, ax1, player1_name, set_ylim=False)
    impact_values2, round_centers2 = create_individual_impact_plot(impacts2, ax2, player2_name, set_ylim=False)
    
    # Now set common symmetric y-axis for both plots based on all data
    all_impact_values = impact_values1 + impact_values2
    if all_impact_values:
        max_positive = max([val for val in all_impact_values if val > 0] + [0])
        max_negative = abs(min([val for val in all_impact_values if val < 0] + [0]))
        max_abs = max(max_positive, max_negative, DEFAULT_AXIS_LIMIT)  # Ensure minimum of DEFAULT_AXIS_LIMIT, but allow growth
        
        print(f"DEBUG: Max positive: {max_positive}, Max negative: {max_negative}, Max abs: {max_abs}")
        
        # Apply the same scale to both plots
        ax1.set_ylim(-max_abs, max_abs)
        ax2.set_ylim(-max_abs, max_abs)
        
        # Now add round labels at the top with the correct y-limits
        for round_centers, ax in [(round_centers1, ax1), (round_centers2, ax2)]:
            for center_x, round_num, event_count in round_centers:
                ax.text(center_x, ax.get_ylim()[1] * 0.95, f'{round_num}', 
                        ha='center', va='top', fontsize=10, fontweight='bold')
    else:
        ax1.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
        ax2.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
    
    # Adjust layout to accommodate top labels
    plt.subplots_adjust(top=0.88)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    
    # Print comparison statistics
    print(f"\n📊 COMPARISON STATISTICS")
    print("=" * 60)
    
    total_impact1 = sum(event['impact'] for event in impacts1)
    total_impact2 = sum(event['impact'] for event in impacts2)
    
    print(f"{player1_name}:")
    print(f"  Total Events: {len(impacts1)}")
    print(f"  Net Impact: {total_impact1:+.1f}")
    print(f"  Avg Impact/Event: {total_impact1/len(impacts1):+.1f}")
    
    print(f"\n{player2_name}:")
    print(f"  Total Events: {len(impacts2)}")
    print(f"  Net Impact: {total_impact2:+.1f}")
    print(f"  Avg Impact/Event: {total_impact2/len(impacts2):+.1f}")
    
    print(f"\n🏆 Winner: {player1_name if total_impact1 > total_impact2 else player2_name}")
    print(f"Impact Difference: {abs(total_impact1 - total_impact2):.1f}")
    

def compare_individual_impacts_vertical(dem, player1_name, player2_name):
    """
    Compare individual kill/death impacts between two players using vertical layout.
    Shows detailed round-by-round comparison with individual events stacked vertically.
    Rounds are aligned between both players for easy comparison.
    """
    print(f"\n🔍 Comparing Individual Impacts (Vertical): {player1_name} vs {player2_name}")
    
    # Get individual impacts for both players 
    impacts1 = get_individual_impacts_data(dem, player1_name)
    impacts2 = get_individual_impacts_data(dem, player2_name)
    
    if not impacts1 or not impacts2:
        print("❌ Could not get individual impact data for one or both players")
        return
    
    print(f"✅ Found {len(impacts1)} events for {player1_name}")
    print(f"✅ Found {len(impacts2)} events for {player2_name}")
    
    # Get all rounds present in both players' data for alignment
    all_rounds = set()
    for event in impacts1:
        all_rounds.add(event['round'])
    for event in impacts2:
        all_rounds.add(event['round'])
    all_rounds = sorted(list(all_rounds))
    
    print(f"🎯 Aligning {len(all_rounds)} rounds: {all_rounds}")
    
    # Create vertically stacked plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    def create_aligned_impact_plot(impacts, ax, player_name, all_rounds, set_ylim=True):
        """Create impact plot with aligned round positions"""
        # Group impacts by round
        round_impacts = defaultdict(list)
        for event in impacts:
            round_impacts[event['round']].append(event)
        
        # Calculate maximum events per round to determine spacing
        max_events_per_round = 4#max(len(round_impacts.get(r, [])) for r in all_rounds)
        if max_events_per_round == 0:
            max_events_per_round = 1
        
        # Prepare data for plotting with aligned positions
        x_positions = []
        impact_values = []
        colors = []
        round_x_centers = []
        
        x_pos = 0
        round_spacing = max_events_per_round + 2  # Fixed spacing per round
        
        for round_num in all_rounds:
            impacts_in_round = round_impacts.get(round_num, [])
            round_start = x_pos
            
            # Position events within the round's allocated space
            if impacts_in_round:
                event_spacing = max_events_per_round / len(impacts_in_round)
                for i, event in enumerate(impacts_in_round):
                    event_x_pos = x_pos + i * event_spacing
                    x_positions.append(event_x_pos)
                    impact_values.append(event['impact'])
                    colors.append('green' if event['impact'] > 0 else 'red')
                
                # Center position for round label
                round_center = x_pos + (len(impacts_in_round) - 1) * event_spacing / 2
            else:
                # No events for this round, still reserve space
                round_center = x_pos + max_events_per_round / 2
            
            round_x_centers.append((round_center, round_num, len(impacts_in_round)))
            x_pos += round_spacing
        
        # Create the bar plot
        if x_positions and impact_values:
            bars = ax.bar(x_positions, impact_values, color=colors, alpha=0.7, width=0.8)
            
            # Add impact value and game state labels on bars
            event_index = 0
            for round_num in all_rounds:
                impacts_in_round = round_impacts.get(round_num, [])
                for event in impacts_in_round:
                    pos = x_positions[event_index]
                    impact = impact_values[event_index]
                    game_state = event['game_state']
                    
                    # Simplify post-plant notation
                    if 'post-plant' in game_state:
                        game_state_short = game_state.replace(' post-plant', ' (post)')
                    else:
                        game_state_short = game_state
                    
                    # Combined label: impact value + game state
                    combined_label = f'{impact:+.1f}\n{game_state_short}'
                    
                    ax.text(pos, impact + (2.0 if impact > 0 else -2.0), combined_label, 
                            ha='center', va='bottom' if impact > 0 else 'top', 
                            fontsize=6, fontweight='bold', color='black')
                    
                    event_index += 1
        
        # Add delta impact labels at the bottom of each round
        for i, round_num in enumerate(all_rounds):
            impacts_in_round = round_impacts.get(round_num, [])
            if impacts_in_round:
                # Calculate total impact for this round
                round_total = sum(event['impact'] for event in impacts_in_round)
                
                # Position at the center of the round
                round_center = i * round_spacing + round_spacing / 2
                
                # Add delta label at the bottom
                ax.text(round_center - 2, -55, f'Δ{round_total:+.1f}', 
                        ha='center', va='center', fontsize=9, fontweight='bold', 
                        color='darkgreen' if round_total > 0 else 'darkred')
        
        # Add vertical separators between rounds
        separator_x = round_spacing / 2
        for i in range(len(all_rounds) - 1):
            ax.axvline(x=separator_x + i * round_spacing, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Only set y-limits if requested (for single player mode)
        if set_ylim and impact_values:
            max_positive = max([val for val in impact_values if val > 0] + [0])
            max_negative = abs(min([val for val in impact_values if val < 0] + [0]))
            max_abs = max(max_positive, max_negative, DEFAULT_AXIS_LIMIT)  # Minimum of DEFAULT_AXIS_LIMIT, but allow growth
            ax.set_ylim(-max_abs, max_abs)
        elif set_ylim:
            ax.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
        
        ax.set_ylabel('Round swing')
        ax.set_title(f'{player_name}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        # Set x-axis limits to show all rounds
        if all_rounds:
            ax.set_xlim(-1, len(all_rounds) * round_spacing - round_spacing/2)
        
        # Remove x-axis tick labels since we have round labels
        ax.set_xticks([])
        
        return impact_values, round_x_centers
    
    # Create plots for both players with aligned rounds (don't set y-limits yet)
    impact_values1, round_centers1 = create_aligned_impact_plot(impacts1, ax1, player1_name, all_rounds, set_ylim=False)
    impact_values2, round_centers2 = create_aligned_impact_plot(impacts2, ax2, player2_name, all_rounds, set_ylim=False)
    
    # Now set common symmetric y-axis for both plots based on all data
    all_impact_values = impact_values1 + impact_values2
    if all_impact_values:
        max_positive = max([val for val in all_impact_values if val > 0] + [0])
        max_negative = abs(min([val for val in all_impact_values if val < 0] + [0]))
        max_abs = max(max_positive, max_negative, DEFAULT_AXIS_LIMIT)  # Ensure minimum of DEFAULT_AXIS_LIMIT, but allow growth
        
        print(f"DEBUG: Max positive: {max_positive}, Max negative: {max_negative}, Max abs: {max_abs}")
        
        # Apply the same scale to both plots
        ax1.set_ylim(-max_abs, max_abs)
        ax2.set_ylim(-max_abs, max_abs)
        
        # Now add round labels at the top with the correct y-limits
        for round_centers, ax in [(round_centers1, ax1), (round_centers2, ax2)]:
            for center_x, round_num, event_count in round_centers:
                ax.text(center_x, ax.get_ylim()[1] * 0.95, f'{round_num}', 
                        ha='center', va='top', fontsize=9, fontweight='bold')
    else:
        ax1.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
        ax2.set_ylim(-DEFAULT_AXIS_LIMIT, DEFAULT_AXIS_LIMIT)
    
    # Adjust layout to accommodate top labels for vertical layout
    plt.subplots_adjust(top=0.93, hspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Print comparison statistics
    print(f"\n📊 COMPARISON STATISTICS (Vertical Layout - Aligned)")
    print("=" * 60)
    
    total_impact1 = sum(event['impact'] for event in impacts1)
    total_impact2 = sum(event['impact'] for event in impacts2)
    
    print(f"{player1_name}:")
    print(f"  Total Events: {len(impacts1)}")
    print(f"  Net Impact: {total_impact1:+.1f}")
    print(f"  Avg Impact/Event: {total_impact1/len(impacts1):+.1f}")
    
    print(f"\n{player2_name}:")
    print(f"  Total Events: {len(impacts2)}")
    print(f"  Net Impact: {total_impact2:+.1f}")
    print(f"  Avg Impact/Event: {total_impact2/len(impacts2):+.1f}")
    
    print(f"\n🏆 Winner: {player1_name if total_impact1 > total_impact2 else player2_name}")
    print(f"Impact Difference: {abs(total_impact1 - total_impact2):.1f}")
    




def get_individual_impacts_data(dem, player_name):
    """
    Extract individual impact data for a player without plotting.
    Helper function for comparison analysis.
    """
    from ..core.win_probability import get_win_probability, calculate_impact_score
    from collections import defaultdict
    
    kills_df = dem.kills
    bomb_events = dem.bomb
    
    if kills_df is None or len(kills_df) == 0:
        return []
    
    # Convert to pandas if needed
    if hasattr(kills_df, 'to_pandas'):
        kills_df = kills_df.to_pandas()
    if bomb_events is not None and hasattr(bomb_events, 'to_pandas'):
        bomb_events = bomb_events.to_pandas()
    
    # Get bomb plant information
    bomb_plants = {}
    if bomb_events is not None and len(bomb_events) > 0:
        for _, bomb_event in bomb_events.iterrows():
            round_num = bomb_event.get('round_num', 0)
            event_columns = bomb_event.index.tolist()
            event_values = bomb_event.values
            
            # Check for plant events
            is_plant = False
            plant_tick = None
            
            if 'event_type' in event_columns:
                event_type = str(bomb_event['event_type']).lower()
                if event_type == 'plant' or event_type == 'bomb_plant':
                    is_plant = True
                    plant_tick = bomb_event.get('tick', 0)
            
            if is_plant:
                bomb_plants[round_num] = plant_tick
    
    # Extract individual impacts
    individual_impacts = []
    event_counter = 0
    
    # Get all rounds with activity
    all_rounds = sorted(set(kills_df['round_num'].unique()))
    
    for round_num in all_rounds:
        if round_num == 0:
            continue
            
        round_kills = kills_df[kills_df['round_num'] == round_num].sort_values('tick')
        
        # Track alive players for this round
        ct_alive = 5
        t_alive = 5
        
        # Check if bomb was planted in this round
        plant_tick = bomb_plants.get(round_num, None)
        
        for _, kill in round_kills.iterrows():
            killer = kill.get('attacker_name', '')
            victim = kill.get('victim_name', '')
            weapon = kill.get('weapon', 'Unknown')
            victim_side = kill.get('victim_side', '')
            kill_tick = kill.get('tick', 0)
            
            # Check if this kill happened after bomb plant
            is_post_plant = (plant_tick is not None and kill_tick > plant_tick)
            
            # Determine weapon category
            weapon_category = "rifle"
            if weapon in ['glock', 'usp_silencer', 'p2000', 'deagle', 'elite', 'fiveseven', 'cz75a', 'tec9', 'p250', 'revolver']:
                weapon_category = "pistol"
            elif weapon in ['ak47', 'm4a1', 'm4a1_s', 'aug', 'sg556', 'famas', 'galilar']:
                weapon_category = "rifle"
            elif weapon in ['awp', 'ssg08', 'scar20', 'g3sg1']:
                weapon_category = "sniper"
            
            # Current game state before this kill
            game_state = ""
            
            # Calculate impact score for this kill
            ct_after = ct_alive - 1 if victim_side == 'ct' else ct_alive
            t_after = t_alive - 1 if victim_side == 't' else t_alive
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record events for our target player
            if killer == player_name:
                attacker_side = kill.get('attacker_side', '')
                if attacker_side.lower() == 'ct':
                    game_state = f"{ct_alive}v{t_alive}"
                else:
                    game_state = f"{t_alive}v{ct_alive}"
                if is_post_plant:
                    game_state += " post-plant"

                individual_impacts.append({
                    'event_id': event_counter,
                    'round': round_num,
                    'type': 'kill',
                    'impact': impact,
                    'game_state': game_state,
                    'weapon': weapon_category,
                    'victim': victim,
                    'post_plant': is_post_plant,
                    'tick': kill_tick
                })
                event_counter += 1
            
            if victim == player_name:
                if round_num == 5:
                    pass
                if victim_side.lower() == 'ct':
                    game_state = f"{ct_alive}v{t_alive}"
                else:
                    game_state = f"{t_alive}v{ct_alive}"
                if is_post_plant:
                    game_state += " post-plant"

                individual_impacts.append({
                    'event_id': event_counter,
                    'round': round_num,
                    'type': 'death',
                    'impact': -impact,  # Negative for deaths
                    'game_state': game_state,
                    'weapon': weapon_category,
                    'killer': killer,
                    'post_plant': is_post_plant,
                    'tick': kill_tick
                })
                event_counter += 1
            
            # Update alive counts after the kill
            if victim_side == 'ct':
                ct_alive = max(0, ct_alive - 1)
            elif victim_side == 't':
                t_alive = max(0, t_alive - 1)
    
    return individual_impacts


def plot_all_players_stats_table(dem):
    """
    Create a table showing kills, deaths, and impact for all players in the match.
    """
    from ..core.analysis import get_player_kill_death_analysis
    from .formatting import display_available_players
    import matplotlib.pyplot as plt
    
    # Get list of all players
    print("📊 Generating stats table for all players...")
    all_players = display_available_players(dem)
    
    if not all_players:
        print("❌ No players found in demo")
        return
    
    # Collect stats for all players
    player_stats = []
    
    for player in all_players:
        print(f"Processing {player}...")
        stats_df = get_player_kill_death_analysis(dem, player)
        
        if stats_df is not None and len(stats_df) > 0:
            # Calculate totals
            total_kills = 0
            total_deaths = 0
            total_impact = 0.0
            
            for _, row in stats_df.iterrows():
                # Count kills
                kills_raw = str(row['Kills'])
                if kills_raw != '-':
                    kills = [k for k in kills_raw.split(' | ') if k.strip() and k != '-']
                    total_kills += len(kills)
                
                # Count deaths  
                deaths_raw = str(row['Deaths'])
                if deaths_raw != '-':
                    deaths = [d for d in deaths_raw.split(' | ') if d.strip() and d != '-']
                    total_deaths += len(deaths)
                
                # Sum impact
                impact_str = str(row['Impact']).replace('+', '').strip()
                try:
                    total_impact += float(impact_str)
                except:
                    pass
            
            # Calculate K/D ratio
            kd_ratio = total_kills / max(total_deaths, 1)
            
            player_stats.append({
                'Player': player,
                'Kills': total_kills,
                'Deaths': total_deaths,
                'K/D': kd_ratio,
                'Impact': total_impact
            })
    
    if not player_stats:
        print("❌ No valid player stats found")
        return
    
    # Sort by impact (highest first)
    player_stats.sort(key=lambda x: x['Impact'], reverse=True)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Rank', 'Player', 'Kills', 'Deaths', 'K/D', 'Impact']
    table_data = []
    
    for i, stats in enumerate(player_stats, 1):
        table_data.append([
            f"{i}",
            stats['Player'],
            f"{stats['Kills']}",
            f"{stats['Deaths']}",
            f"{stats['K/D']:.2f}",
            f"{stats['Impact']:+.1f}"
        ])
    
    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on performance
    for i, stats in enumerate(player_stats, 1):
        impact = stats['Impact']
        if impact > 0:
            color = '#E8F5E8'  # Light green for positive impact
        elif impact < 0:
            color = '#FFE8E8'  # Light red for negative impact
        else:
            color = '#F0F0F0'  # Light gray for neutral
            
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    # Highlight best performer
    if len(player_stats) > 0:
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('#FFD700')  # Gold for #1
            
    # Add title
    plt.suptitle('🏆 CS2 Match Statistics - All Players', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add subtitle with match summary
    total_rounds = max([int(str(row['Round']).replace('💣', '').strip()) 
                       for player in all_players[:1]  # Just check first player
                       for _, row in get_player_kill_death_analysis(dem, player).iterrows()
                       if get_player_kill_death_analysis(dem, player) is not None] or [0])
    
    plt.figtext(0.5, 0.02, 
                f"Match Summary: {len(player_stats)} players • {total_rounds} rounds",
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Print text summary as well
    print("\n" + "="*80)
    print("🏆 MATCH STATISTICS - ALL PLAYERS")
    print("="*80)
    print(f"{'Rank':<4} {'Player':<20} {'Kills':<6} {'Deaths':<6} {'K/D':<6} {'Impact':<8}")
    print("-" * 80)
    
    for i, stats in enumerate(player_stats, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{rank_emoji:<4} {stats['Player']:<20} {stats['Kills']:<6} {stats['Deaths']:<6} {stats['K/D']:<6.2f} {stats['Impact']:<+8.1f}")
    
    print("-" * 80)
    
    # Calculate team stats if possible (assume first 5 vs last 5 for now)
    if len(player_stats) >= 10:
        team1_impact = sum(p['Impact'] for p in player_stats[:5])
        team2_impact = sum(p['Impact'] for p in player_stats[5:10])
        print(f"\n🔵 Team 1 Total Impact: {team1_impact:+.1f}")
        print(f"🟠 Team 2 Total Impact: {team2_impact:+.1f}")
        winner = "Team 1" if team1_impact > team2_impact else "Team 2"
        print(f"🏆 Impact Winner: {winner}")
    
