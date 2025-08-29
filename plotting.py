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


def compare_players_impact(dem, player1_name, player2_name, get_player_analysis_func):
    """
    Compare two players' impact scores side by side
    """
    print(f"\n🔍 Comparing {player1_name} vs {player2_name}")
    
    # Get stats for both players
    stats1 = get_player_analysis_func(dem, player1_name, debug=False)
    stats2 = get_player_analysis_func(dem, player2_name, debug=False)
    
    if stats1 is None or len(stats1) == 0 or stats2 is None or len(stats2) == 0:
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
