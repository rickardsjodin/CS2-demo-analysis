"""
Plotting functions for CS2 demo analysis visualization.
"""

import matplotlib.pyplot as plt
import numpy as np


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
    ax1.set_ylabel('Impact Score')
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
    
    ax3.set_xlabel('Impact Score')
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


def plot_individual_impacts(dem, player_name):
    """
    Plot each individual kill/death impact as separate data points.
    Shows multiple impacts within the same round as separate events.
    """
    from analysis import get_win_probability, calculate_impact_score
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
            game_state = f"{ct_alive}v{t_alive}"
            if is_post_plant:
                game_state += " post-plant"
            
            # Calculate impact score for this kill
            ct_after = ct_alive - 1 if victim_side == 'ct' else ct_alive
            t_after = t_alive - 1 if victim_side == 't' else t_alive
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record events for our target player
            if killer == player_name:
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
    
    if not individual_impacts:
        print(f"❌ No individual impacts found for player: {player_name}")
        return
    
    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14))
    
    # Prepare data for plotting
    event_ids = [event['event_id'] for event in individual_impacts]
    impacts = [event['impact'] for event in individual_impacts]
    rounds = [event['round'] for event in individual_impacts]
    types = [event['type'] for event in individual_impacts]
    game_states = [event['game_state'] for event in individual_impacts]
    
    # Plot 1: Individual Impact Timeline
    kill_indices = [i for i, t in enumerate(types) if t == 'kill']
    death_indices = [i for i, t in enumerate(types) if t == 'death']
    
    # Plot kills and deaths separately
    if kill_indices:
        kill_events = [event_ids[i] for i in kill_indices]
        kill_impacts = [impacts[i] for i in kill_indices]
        ax1.scatter(kill_events, kill_impacts, color='green', s=100, alpha=0.8, 
                   label=f'Kills ({len(kill_indices)})', marker='^', edgecolors='black')
    
    if death_indices:
        death_events = [event_ids[i] for i in death_indices]
        death_impacts = [impacts[i] for i in death_indices]
        ax1.scatter(death_events, death_impacts, color='red', s=100, alpha=0.8, 
                   label=f'Deaths ({len(death_indices)})', marker='v', edgecolors='black')
    
    # Add round labels
    round_changes = []
    current_round = rounds[0] if rounds else 1
    for i, round_num in enumerate(rounds):
        if round_num != current_round:
            round_changes.append(i)
            current_round = round_num
    
    # Add vertical lines for round changes
    for change_idx in round_changes:
        ax1.axvline(x=event_ids[change_idx], color='gray', linestyle='--', alpha=0.7)
    
    # Add round number annotations
    for i in range(0, len(event_ids), max(1, len(event_ids)//15)):  # Show every nth event
        ax1.annotate(f'R{rounds[i]}', (event_ids[i], impacts[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Event Sequence')
    ax1.set_ylabel('Impact Score')
    ax1.set_title(f'{player_name} - Individual Kill/Death Impacts Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Plot 2: Impact by Round (grouped)
    round_data = defaultdict(list)
    for event in individual_impacts:
        round_data[event['round']].append(event['impact'])
    
    sorted_rounds = sorted(round_data.keys())
    round_positions = []
    round_impacts_flat = []
    round_labels = []
    
    pos = 0
    for round_num in sorted_rounds:
        round_impacts = round_data[round_num]
        for i, impact in enumerate(round_impacts):
            round_positions.append(pos + i * 0.3)
            round_impacts_flat.append(impact)
            round_labels.append(round_num)
        pos += len(round_impacts) + 1
    
    # Color by positive/negative
    colors = ['green' if imp > 0 else 'red' for imp in round_impacts_flat]
    ax2.bar(round_positions, round_impacts_flat, color=colors, alpha=0.7, width=0.25)
    
    # Add value labels on bars
    for pos, impact in zip(round_positions, round_impacts_flat):
        ax2.text(pos, impact + (0.5 if impact > 0 else -0.5), f'{impact:+.1f}', 
                ha='center', va='bottom' if impact > 0 else 'top', fontsize=8, rotation=90)
    
    ax2.set_xlabel('Events Grouped by Round')
    ax2.set_ylabel('Impact Score')
    ax2.set_title(f'{player_name} - Individual Impacts Grouped by Round')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Plot 3: Impact Distribution Analysis
    positive_impacts = [imp for imp in impacts if imp > 0]
    negative_impacts = [imp for imp in impacts if imp < 0]
    
    # Create histogram
    if positive_impacts:
        ax3.hist(positive_impacts, bins=15, alpha=0.7, color='green', 
                label=f'Positive Impacts ({len(positive_impacts)})', edgecolor='black')
    if negative_impacts:
        ax3.hist(negative_impacts, bins=15, alpha=0.7, color='red', 
                label=f'Negative Impacts ({len(negative_impacts)})', edgecolor='black')
    
    ax3.set_xlabel('Impact Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{player_name} - Individual Impact Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    
    plt.tight_layout()
    
    # Print detailed statistics
    print(f"\n📊 INDIVIDUAL IMPACT ANALYSIS for {player_name}")
    print("=" * 70)
    
    total_events = len(individual_impacts)
    kill_events = len([e for e in individual_impacts if e['type'] == 'kill'])
    death_events = len([e for e in individual_impacts if e['type'] == 'death'])
    
    print(f"Total Individual Events: {total_events}")
    print(f"• Kills: {kill_events}")
    print(f"• Deaths: {death_events}")
    
    if positive_impacts:
        print(f"\nPositive Impacts: {len(positive_impacts)}")
        print(f"• Highest: +{max(positive_impacts):.1f}")
        print(f"• Average: +{sum(positive_impacts)/len(positive_impacts):.1f}")
        print(f"• Total: +{sum(positive_impacts):.1f}")
    
    if negative_impacts:
        print(f"\nNegative Impacts: {len(negative_impacts)}")
        print(f"• Worst: {min(negative_impacts):.1f}")
        print(f"• Average: {sum(negative_impacts)/len(negative_impacts):.1f}")
        print(f"• Total: {sum(negative_impacts):.1f}")
    
    total_impact = sum(impacts)
    print(f"\nNet Impact: {total_impact:+.1f}")
    print(f"Impact per Event: {total_impact/total_events:+.1f}")
    
    # Find highest impact events
    if impacts:
        max_impact = max(impacts)
        min_impact = min(impacts)
        
        best_event = next(e for e in individual_impacts if e['impact'] == max_impact)
        worst_event = next(e for e in individual_impacts if e['impact'] == min_impact)
        
        print(f"\n🎯 EXTREME EVENTS:")
        print(f"Best Event: {best_event['type']} in round {best_event['round']} "
              f"({best_event['game_state']}) = +{best_event['impact']:.1f}")
        print(f"Worst Event: {worst_event['type']} in round {worst_event['round']} "
              f"({worst_event['game_state']}) = {worst_event['impact']:+.1f}")
    
    # Analyze by game state
    game_state_impacts = defaultdict(list)
    for event in individual_impacts:
        game_state_impacts[event['game_state']].append(event['impact'])
    
    print(f"\n🎮 IMPACT BY GAME STATE:")
    for state, state_impacts in sorted(game_state_impacts.items()):
        avg_impact = sum(state_impacts) / len(state_impacts)
        print(f"• {state}: {avg_impact:+.1f} avg ({len(state_impacts)} events)")
    
    return individual_impacts


def compare_individual_impacts(dem, player1_name, player2_name):
    """
    Compare individual kill/death impacts between two players side by side.
    Shows round-by-round comparison with detailed analysis.
    """
    print(f"\n🔍 Comparing Individual Impacts: {player1_name} vs {player2_name}")
    
    # Get individual impacts for both players (but don't show the plots)
    print(f"\n📊 Analyzing {player1_name}...")
    from analysis import get_win_probability, calculate_impact_score
    from collections import defaultdict
    
    # Get impacts for both players without showing individual plots
    impacts1 = get_individual_impacts_data(dem, player1_name)
    impacts2 = get_individual_impacts_data(dem, player2_name)
    
    if not impacts1 or not impacts2:
        print("❌ Could not get individual impact data for one or both players")
        return
    
    print(f"✅ Found {len(impacts1)} events for {player1_name}")
    print(f"✅ Found {len(impacts2)} events for {player2_name}")
    
    # Create comparison visualization - now only 2 rows
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1 & 2: Round-by-round comparison with common y-axis
    def create_round_comparison(impacts, ax, player_name):
        """Create round-by-round impact visualization"""
        round_data = {}
        for event in impacts:
            round_num = event['round']
            if round_num not in round_data:
                round_data[round_num] = []
            round_data[round_num].append(event['impact'])
        
        rounds = sorted(round_data.keys())
        round_totals = [sum(round_data[r]) for r in rounds]
        round_counts = [len(round_data[r]) for r in rounds]
        
        # Create bars with colors based on positive/negative
        colors = ['green' if total > 0 else 'red' if total < 0 else 'gray' for total in round_totals]
        bars = ax.bar(rounds, round_totals, color=colors, alpha=0.7)
        
        # Add event count labels on bars
        for i, (round_num, total, count) in enumerate(zip(rounds, round_totals, round_counts)):
            height = total + (1 if total > 0 else -1)
            ax.text(round_num, height, f'{count}e', ha='center', 
                   va='bottom' if total > 0 else 'top', fontsize=8)
        
        ax.set_title(f'{player_name} - Impact by Round (with event counts)')
        ax.set_xlabel('Round')
        ax.set_ylabel('Total Round Impact')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        
        return round_data, rounds, round_totals
    
    round_data1, rounds1, totals1 = create_round_comparison(impacts1, ax1, player1_name)
    round_data2, rounds2, totals2 = create_round_comparison(impacts2, ax2, player2_name)
    
    # Set common y-axis for both round plots
    all_totals = totals1 + totals2
    if all_totals:
        y_min = min(all_totals) - 5
        y_max = max(all_totals) + 5
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
    
    # Plot 3: Direct statistical comparison
    def create_stats_comparison(impacts1, impacts2, ax):
        """Create side-by-side statistical comparison"""
        # Extract data
        pos1 = [imp['impact'] for imp in impacts1 if imp['impact'] > 0]
        neg1 = [imp['impact'] for imp in impacts1 if imp['impact'] < 0]
        pos2 = [imp['impact'] for imp in impacts2 if imp['impact'] > 0]
        neg2 = [imp['impact'] for imp in impacts2 if imp['impact'] < 0]
        
        # Create comparison bars
        categories = ['Positive\nEvents', 'Negative\nEvents', 'Total\nEvents', 'Net\nImpact']
        
        player1_stats = [
            len(pos1),
            len(neg1), 
            len(impacts1),
            sum(imp['impact'] for imp in impacts1)
        ]
        
        player2_stats = [
            len(pos2),
            len(neg2),
            len(impacts2), 
            sum(imp['impact'] for imp in impacts2)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, player1_stats, width, label=player1_name, 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, player2_stats, width, label=player2_name, 
                      color='orangered', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}' if isinstance(height, float) else f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Statistical Comparison')
        ax.set_ylabel('Count / Impact Value')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    create_stats_comparison(impacts1, impacts2, ax3)
    
    # Plot 4: Head-to-head impact distribution
    all_impacts1 = [imp['impact'] for imp in impacts1]
    all_impacts2 = [imp['impact'] for imp in impacts2]
    
    # Create overlapping histograms
    bins = np.linspace(min(min(all_impacts1), min(all_impacts2)), 
                      max(max(all_impacts1), max(all_impacts2)), 20)
    
    ax4.hist(all_impacts1, bins=bins, alpha=0.6, color='steelblue', 
            label=f'{player1_name} ({len(all_impacts1)} events)', edgecolor='black')
    ax4.hist(all_impacts2, bins=bins, alpha=0.6, color='orangered', 
            label=f'{player2_name} ({len(all_impacts2)} events)', edgecolor='black')
    
    ax4.set_title('Impact Distribution Comparison')
    ax4.set_xlabel('Impact Score')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    
    plt.tight_layout()
    
    # Print detailed comparison
    print(f"\n📊 DETAILED INDIVIDUAL IMPACT COMPARISON")
    print("=" * 80)
    
    # Calculate comprehensive stats
    def calculate_player_stats(impacts, player_name):
        all_impacts = [imp['impact'] for imp in impacts]
        positive_impacts = [imp for imp in all_impacts if imp > 0]
        negative_impacts = [imp for imp in all_impacts if imp < 0]
        kill_impacts = [imp['impact'] for imp in impacts if imp['type'] == 'kill']
        death_impacts = [imp['impact'] for imp in impacts if imp['type'] == 'death']
        
        return {
            'name': player_name,
            'total_events': len(impacts),
            'kill_events': len(kill_impacts),
            'death_events': len(death_impacts),
            'positive_events': len(positive_impacts),
            'negative_events': len(negative_impacts),
            'total_impact': sum(all_impacts),
            'total_positive': sum(positive_impacts) if positive_impacts else 0,
            'total_negative': sum(negative_impacts) if negative_impacts else 0,
            'avg_impact': sum(all_impacts) / len(all_impacts) if all_impacts else 0,
            'avg_positive': sum(positive_impacts) / len(positive_impacts) if positive_impacts else 0,
            'avg_negative': sum(negative_impacts) / len(negative_impacts) if negative_impacts else 0,
            'max_impact': max(all_impacts) if all_impacts else 0,
            'min_impact': min(all_impacts) if all_impacts else 0,
        }
    
    stats1 = calculate_player_stats(impacts1, player1_name)
    stats2 = calculate_player_stats(impacts2, player2_name)
    
    # Print comparison table
    print(f"{'Metric':<25} {player1_name:<15} {player2_name:<15} {'Difference':<15}")
    print("─" * 75)
    print(f"{'Total Events:':<25} {stats1['total_events']:<15} {stats2['total_events']:<15} {stats1['total_events']-stats2['total_events']:+<15}")
    print(f"{'Kill Events:':<25} {stats1['kill_events']:<15} {stats2['kill_events']:<15} {stats1['kill_events']-stats2['kill_events']:+<15}")
    print(f"{'Death Events:':<25} {stats1['death_events']:<15} {stats2['death_events']:<15} {stats1['death_events']-stats2['death_events']:+<15}")
    print(f"{'Positive Events:':<25} {stats1['positive_events']:<15} {stats2['positive_events']:<15} {stats1['positive_events']-stats2['positive_events']:+<15}")
    print(f"{'Negative Events:':<25} {stats1['negative_events']:<15} {stats2['negative_events']:<15} {stats1['negative_events']-stats2['negative_events']:+<15}")
    print()
    print(f"{'Total Impact:':<25} {stats1['total_impact']:+<15.1f} {stats2['total_impact']:+<15.1f} {stats1['total_impact']-stats2['total_impact']:+<15.1f}")
    print(f"{'Total Positive:':<25} {stats1['total_positive']:+<15.1f} {stats2['total_positive']:+<15.1f} {stats1['total_positive']-stats2['total_positive']:+<15.1f}")
    print(f"{'Total Negative:':<25} {stats1['total_negative']:+<15.1f} {stats2['total_negative']:+<15.1f} {stats1['total_negative']-stats2['total_negative']:+<15.1f}")
    print()
    print(f"{'Avg Impact/Event:':<25} {stats1['avg_impact']:+<15.1f} {stats2['avg_impact']:+<15.1f} {stats1['avg_impact']-stats2['avg_impact']:+<15.1f}")
    print(f"{'Avg Positive Impact:':<25} {stats1['avg_positive']:+<15.1f} {stats2['avg_positive']:+<15.1f} {stats1['avg_positive']-stats2['avg_positive']:+<15.1f}")
    print(f"{'Avg Negative Impact:':<25} {stats1['avg_negative']:+<15.1f} {stats2['avg_negative']:+<15.1f} {stats1['avg_negative']-stats2['avg_negative']:+<15.1f}")
    print()
    print(f"{'Highest Impact:':<25} {stats1['max_impact']:+<15.1f} {stats2['max_impact']:+<15.1f} {stats1['max_impact']-stats2['max_impact']:+<15.1f}")
    print(f"{'Lowest Impact:':<25} {stats1['min_impact']:+<15.1f} {stats2['min_impact']:+<15.1f} {stats1['min_impact']-stats2['min_impact']:+<15.1f}")
    
    # Find head-to-head moments (same rounds)
    rounds1_set = set(imp['round'] for imp in impacts1)
    rounds2_set = set(imp['round'] for imp in impacts2)
    common_rounds = rounds1_set.intersection(rounds2_set)
    
    if common_rounds:
        print(f"\n⚔️ HEAD-TO-HEAD ROUNDS ({len(common_rounds)} rounds with both players active):")
        
        h2h_stats1 = {'events': 0, 'impact': 0}
        h2h_stats2 = {'events': 0, 'impact': 0}
        
        for round_num in sorted(common_rounds):
            round_impacts1 = [imp for imp in impacts1 if imp['round'] == round_num]
            round_impacts2 = [imp for imp in impacts2 if imp['round'] == round_num]
            
            total1 = sum(imp['impact'] for imp in round_impacts1)
            total2 = sum(imp['impact'] for imp in round_impacts2)
            
            h2h_stats1['events'] += len(round_impacts1)
            h2h_stats1['impact'] += total1
            h2h_stats2['events'] += len(round_impacts2)
            h2h_stats2['impact'] += total2
            
            winner = player1_name if total1 > total2 else player2_name if total2 > total1 else "Tie"
            print(f"Round {round_num}: {player1_name} {total1:+.1f} vs {player2_name} {total2:+.1f} → {winner}")
        
        print(f"\nHead-to-Head Summary:")
        print(f"{player1_name}: {h2h_stats1['impact']:+.1f} impact ({h2h_stats1['events']} events)")
        print(f"{player2_name}: {h2h_stats2['impact']:+.1f} impact ({h2h_stats2['events']} events)")
        
        h2h_winner = player1_name if h2h_stats1['impact'] > h2h_stats2['impact'] else player2_name
        print(f"Head-to-Head Winner: {h2h_winner}")
    
    return stats1, stats2


def get_individual_impacts_data(dem, player_name):
    """
    Extract individual impact data for a player without plotting.
    Helper function for comparison analysis.
    """
    from analysis import get_win_probability, calculate_impact_score
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
            game_state = f"{ct_alive}v{t_alive}"
            if is_post_plant:
                game_state += " post-plant"
            
            # Calculate impact score for this kill
            ct_after = ct_alive - 1 if victim_side == 'ct' else ct_alive
            t_after = t_alive - 1 if victim_side == 't' else t_alive
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record events for our target player
            if killer == player_name:
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
