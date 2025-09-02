"""
Dataset Statistics and Validation Script for CS2 Snapshots
Computes comprehensive statistics to validate dataset quality and analyze win probabilities
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def clean_dataset(df):
    """Clean the dataset by removing invalid entries"""
    print("🧹 Cleaning dataset...")
    
    original_count = len(df)
    
    # Remove entries with invalid player counts (should be 0-5)
    df = df[(df['cts_alive'] >= 0) & (df['cts_alive'] <= 5)]
    df = df[(df['ts_alive'] >= 0) & (df['ts_alive'] <= 5)]
    
    # Remove entries with invalid time values (should be 0-115 for normal rounds)
    df = df[(df['time_left'] >= 0) & (df['time_left'] <= 115)]  
    
    # Remove entries with missing winner data
    df = df[df['winner'].notna()]
    df = df[df['winner'].isin(['ct', 't'])]
    
    cleaned_count = len(df)
    removed_count = original_count - cleaned_count
    
    if removed_count > 0:
        print(f"🗑️  Removed {removed_count:,} invalid entries ({removed_count/original_count*100:.1f}%)")
        print(f"✅ Cleaned dataset: {cleaned_count:,} valid snapshots remaining")
    else:
        print("✅ No invalid entries found")
    
    return df

def load_dataset(file_path):
    """Load the JSON dataset"""
    print(f"📂 Loading dataset from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✅ Loaded {len(df):,} snapshots")
        return df
    
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None

def categorize_time_phase(time_left):
    """Categorize time into early, middle, and late round phases"""
    if time_left > 75:
        return "early"
    elif time_left > 35:
        return "middle"
    else:
        return "late"

def compute_basic_statistics(df):
    """Compute basic dataset statistics"""
    print("\n📊 BASIC DATASET STATISTICS")
    print("=" * 50)
    
    # Total snapshots
    total_snapshots = len(df)
    print(f"Total snapshots: {total_snapshots:,}")
    
    # Unique sources (rounds)
    unique_rounds = df['source'].nunique()
    print(f"Unique rounds: {unique_rounds:,}")
    
    # Average snapshots per round
    avg_snapshots_per_round = total_snapshots / unique_rounds
    print(f"Average snapshots per round: {avg_snapshots_per_round:.1f}")
    
    # Time range
    print(f"Time left range: {df['time_left'].min():.1f}s - {df['time_left'].max():.1f}s")
    
    # Player count ranges
    print(f"CTs alive range: {df['cts_alive'].min()} - {df['cts_alive'].max()}")
    print(f"Ts alive range: {df['ts_alive'].min()} - {df['ts_alive'].max()}")
    
    # Bomb planted distribution
    bomb_planted_count = df['bomb_planted'].sum()
    bomb_planted_pct = bomb_planted_count / total_snapshots * 100
    print(f"Bomb planted snapshots: {bomb_planted_count:,} ({bomb_planted_pct:.1f}%)")
    
    # Winner distribution
    winner_counts = df['winner'].value_counts()
    print(f"\nWinner distribution:")
    for winner, count in winner_counts.items():
        pct = count / total_snapshots * 100
        print(f"  {winner.upper()}: {count:,} ({pct:.1f}%)")
    
    return {
        'total_snapshots': total_snapshots,
        'unique_rounds': unique_rounds,
        'avg_snapshots_per_round': avg_snapshots_per_round,
        'bomb_planted_pct': bomb_planted_pct,
        'winner_distribution': winner_counts
    }

def analyze_time_phases(df):
    """Analyze the three time phases"""
    print("\n⏰ TIME PHASE ANALYSIS")
    print("=" * 50)
    
    # Add time phase column
    df['time_phase'] = df['time_left'].apply(categorize_time_phase)
    
    # Phase distribution
    phase_counts = df['time_phase'].value_counts()
    total = len(df)
    
    print("Time phase distribution:")
    for phase in ['early', 'middle', 'late']:
        count = phase_counts.get(phase, 0)
        pct = count / total * 100
        print(f"  {phase.capitalize()} (>75s, 35-75s, <35s): {count:,} ({pct:.1f}%)")
    
    # Win rates by phase
    print("\nWin rates by time phase:")
    for phase in ['early', 'middle', 'late']:
        phase_data = df[df['time_phase'] == phase]
        if len(phase_data) > 0:
            ct_wins = (phase_data['winner'] == 'ct').sum()
            t_wins = (phase_data['winner'] == 't').sum()
            total_phase = len(phase_data)
            
            ct_win_rate = ct_wins / total_phase * 100
            t_win_rate = t_wins / total_phase * 100
            
            print(f"  {phase.capitalize()}: CT {ct_win_rate:.1f}% | T {t_win_rate:.1f}%")
    
    return df

def compute_scenario_probabilities(df):
    """Compute win probabilities for different scenarios"""
    print("\n🎯 SCENARIO WIN PROBABILITIES")
    print("=" * 50)
    
    scenarios_stats = defaultdict(lambda: {'total': 0, 'ct_wins': 0, 't_wins': 0})
    
    # Group by key scenario variables
    for _, row in df.iterrows():
        # Create scenario key
        scenario_key = (
            row['time_phase'],
            row['cts_alive'],
            row['ts_alive'],
            row['bomb_planted']
        )
        
        scenarios_stats[scenario_key]['total'] += 1
        if row['winner'] == 'ct':
            scenarios_stats[scenario_key]['ct_wins'] += 1
        else:
            scenarios_stats[scenario_key]['t_wins'] += 1
    
    # Convert to DataFrame for easier analysis
    scenario_data = []
    for (time_phase, cts, ts, bomb), stats in scenarios_stats.items():
        if stats['total'] >= 10:  # Only include scenarios with sufficient data
            ct_win_rate = stats['ct_wins'] / stats['total']
            t_win_rate = stats['t_wins'] / stats['total']
            
            scenario_data.append({
                'time_phase': time_phase,
                'cts_alive': cts,
                'ts_alive': ts,
                'bomb_planted': bomb,
                'total_samples': stats['total'],
                'ct_win_rate': ct_win_rate,
                't_win_rate': t_win_rate
            })
    
    scenarios_df = pd.DataFrame(scenario_data)
    print(f"Analyzed {len(scenarios_df)} scenarios with ≥10 samples each")
    
    return scenarios_df

def create_player_count_heatmap(df, time_phase=None, bomb_planted=None):
    """Create heatmap of win probabilities by player counts"""
    
    # Filter data if specified
    filtered_df = df.copy()
    if time_phase:
        filtered_df = filtered_df[filtered_df['time_phase'] == time_phase]
    if bomb_planted is not None:
        filtered_df = filtered_df[filtered_df['bomb_planted'] == bomb_planted]
    
    # Create pivot table
    win_rates = []
    player_counts = range(5, 0, -1)  # Reverse order: 5, 4, 3, 2, 1
    
    heatmap_data = np.zeros((5, 5))
    sample_counts = np.zeros((5, 5))
    
    for ct_idx, ct_count in enumerate(player_counts):
        for t_idx, t_count in enumerate(player_counts):
            scenario_data = filtered_df[
                (filtered_df['cts_alive'] == ct_count) & 
                (filtered_df['ts_alive'] == t_count)
            ]
            
            if len(scenario_data) > 0:
                t_win_rate = (scenario_data['winner'] == 't').mean()
                heatmap_data[t_idx, ct_idx] = t_win_rate * 100
                sample_counts[t_idx, ct_idx] = len(scenario_data)
            else:
                heatmap_data[t_idx, ct_idx] = np.nan
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Custom colormap (red to green)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=100)
    
    # Create mask for cells with insufficient data
    mask = sample_counts < 10
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=50,
        vmin=0,
        vmax=100,
        mask=mask,
        cbar_kws={'label': 'T Win Probability (%)'},
        linewidths=0.5,
        linecolor='white'
    )
    
    # Create title
    title_parts = []
    if time_phase:
        title_parts.append(f"{time_phase.capitalize()} Round")
    if bomb_planted is not None:
        title_parts.append("Post-Plant" if bomb_planted else "Pre-Plant")
    
    title = "T Win Probability - " + " ".join(title_parts) if title_parts else "T Win Probability"
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    # Custom labels
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    
    # Save with descriptive filename
    filename_parts = ['dataset_stats']
    if time_phase:
        filename_parts.append(time_phase)
    if bomb_planted is not None:
        filename_parts.append('postplant' if bomb_planted else 'preplant')
    filename_parts.append('heatmap.png')
    
    filename = f"../../outputs/visualizations/{'_'.join(filename_parts)}"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Heatmap saved: {filename}")
    
    return heatmap_data, sample_counts

def analyze_bomb_impact(df):
    """Analyze the impact of bomb planting on win probabilities"""
    print("\n💣 BOMB PLANTING IMPACT ANALYSIS")
    print("=" * 50)
    
    # Overall impact
    no_bomb_data = df[df['bomb_planted'] == False]
    bomb_data = df[df['bomb_planted'] == True]
    
    no_bomb_t_rate = (no_bomb_data['winner'] == 't').mean() * 100
    bomb_t_rate = (bomb_data['winner'] == 't').mean() * 100
    
    print(f"T win rate without bomb: {no_bomb_t_rate:.1f}%")
    print(f"T win rate with bomb: {bomb_t_rate:.1f}%")
    print(f"Bomb impact: +{bomb_t_rate - no_bomb_t_rate:.1f} percentage points for T")
    
    # Impact by time phase
    print("\nBomb impact by time phase:")
    for phase in ['early', 'middle', 'late']:
        phase_no_bomb = df[(df['time_phase'] == phase) & (df['bomb_planted'] == False)]
        phase_bomb = df[(df['time_phase'] == phase) & (df['bomb_planted'] == True)]
        
        if len(phase_no_bomb) > 0 and len(phase_bomb) > 0:
            no_bomb_rate = (phase_no_bomb['winner'] == 't').mean() * 100
            bomb_rate = (phase_bomb['winner'] == 't').mean() * 100
            impact = bomb_rate - no_bomb_rate
            
            print(f"  {phase.capitalize()}: {no_bomb_rate:.1f}% → {bomb_rate:.1f}% (+{impact:.1f}pp)")

def analyze_player_advantage(df):
    """Analyze player count advantages"""
    print("\n👥 PLAYER COUNT ADVANTAGE ANALYSIS")
    print("=" * 50)
    
    # Calculate player advantage for each snapshot
    df['player_advantage'] = df['cts_alive'] - df['ts_alive']
    
    # Group by advantage
    advantage_analysis = []
    for advantage in range(-4, 5):  # -4 to +4 player advantage
        adv_data = df[df['player_advantage'] == advantage]
        if len(adv_data) > 0:
            ct_win_rate = (adv_data['winner'] == 'ct').mean() * 100
            sample_count = len(adv_data)
            
            advantage_analysis.append({
                'advantage': advantage,
                'description': f"CT +{advantage}" if advantage > 0 else f"T +{abs(advantage)}" if advantage < 0 else "Equal",
                'ct_win_rate': ct_win_rate,
                'sample_count': sample_count
            })
    
    print("Player advantage impact on CT win rate:")
    print("Advantage    Description      CT Win %    Samples")
    print("-" * 50)
    
    for analysis in advantage_analysis:
        print(f"{analysis['advantage']:>4}        {analysis['description']:<12}    {analysis['ct_win_rate']:>6.1f}%    {analysis['sample_count']:>7,}")

def create_time_distribution_plot(df):
    """Create plots showing time distribution and win rates over time"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time distribution
    plt.subplot(2, 2, 1)
    plt.hist(df['time_left'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Time Left (seconds)')
    plt.ylabel('Number of Snapshots')
    plt.title('Distribution of Snapshots by Time Left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Win rate over time
    plt.subplot(2, 2, 2)
    
    # Bin time into intervals for smoother visualization
    time_bins = np.arange(0, 116, 5)
    df['time_bin'] = pd.cut(df['time_left'], bins=time_bins)
    
    win_rate_by_time = df.groupby('time_bin')['winner'].apply(lambda x: (x == 't').mean() * 100)
    
    # Get bin centers for plotting
    bin_centers = [(interval.left + interval.right) / 2 for interval in win_rate_by_time.index]
    
    plt.plot(bin_centers, win_rate_by_time.values, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Time Left (seconds)')
    plt.ylabel('T Win Rate (%)')
    plt.title('T Win Rate by Time Left')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Plot 3: Player count distribution
    plt.subplot(2, 2, 3)
    
    player_combinations = df.groupby(['cts_alive', 'ts_alive']).size().reset_index(name='count')
    
    # Create a simple bar plot of most common scenarios
    top_scenarios = player_combinations.nlargest(10, 'count')
    scenario_labels = [f"{row['cts_alive']}v{row['ts_alive']}" for _, row in top_scenarios.iterrows()]
    
    plt.bar(range(len(top_scenarios)), top_scenarios['count'])
    plt.xlabel('Player Scenarios')
    plt.ylabel('Number of Snapshots')
    plt.title('Top 10 Most Common Player Scenarios')
    plt.xticks(range(len(top_scenarios)), scenario_labels, rotation=45)
    
    # Plot 4: Phase distribution
    plt.subplot(2, 2, 4)
    
    phase_counts = df['time_phase'].value_counts()
    phase_winner = df.groupby(['time_phase', 'winner']).size().unstack(fill_value=0)
    
    phase_winner.plot(kind='bar', ax=plt.gca(), color=['orange', 'blue'])
    plt.xlabel('Time Phase')
    plt.ylabel('Number of Snapshots')
    plt.title('Snapshots and Winners by Time Phase')
    plt.legend(['CT Wins', 'T Wins'])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../../outputs/visualizations/dataset_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Time analysis plots saved: ../../outputs/visualizations/dataset_time_analysis.png")

def generate_comprehensive_report(df, scenarios_df, basic_stats):
    """Generate a comprehensive text report"""
    
    report_lines = []
    report_lines.append("CS2 DATASET STATISTICS AND VALIDATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Basic statistics
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 20)
    report_lines.append(f"Total snapshots: {basic_stats['total_snapshots']:,}")
    report_lines.append(f"Unique rounds: {basic_stats['unique_rounds']:,}")
    report_lines.append(f"Average snapshots per round: {basic_stats['avg_snapshots_per_round']:.1f}")
    report_lines.append(f"Bomb planted scenarios: {basic_stats['bomb_planted_pct']:.1f}%")
    report_lines.append("")
    
    # Time phase breakdown
    report_lines.append("TIME PHASE BREAKDOWN")
    report_lines.append("-" * 20)
    phase_counts = df['time_phase'].value_counts()
    total = len(df)
    
    for phase in ['early', 'middle', 'late']:
        count = phase_counts.get(phase, 0)
        pct = count / total * 100
        phase_data = df[df['time_phase'] == phase]
        ct_win_rate = (phase_data['winner'] == 'ct').mean() * 100 if len(phase_data) > 0 else 0
        
        report_lines.append(f"{phase.capitalize()} phase: {count:,} snapshots ({pct:.1f}%) - CT win rate: {ct_win_rate:.1f}%")
    
    report_lines.append("")
    
    # Top scenarios by volume
    report_lines.append("TOP 20 SCENARIOS BY VOLUME")
    report_lines.append("-" * 30)
    top_scenarios = scenarios_df.nlargest(20, 'total_samples')
    
    report_lines.append("Phase    Players  Bomb   Samples   CT Win%   T Win%")
    report_lines.append("-" * 55)
    
    for _, scenario in top_scenarios.iterrows():
        bomb_status = "Yes" if scenario['bomb_planted'] else "No"
        report_lines.append(
            f"{scenario['time_phase']:<8} {scenario['cts_alive']}v{scenario['ts_alive']:<6} "
            f"{bomb_status:<5} {scenario['total_samples']:>7,} {scenario['ct_win_rate']*100:>8.1f}% {scenario['t_win_rate']*100:>8.1f}%"
        )
    
    report_lines.append("")
    
    # Data quality checks
    report_lines.append("DATA QUALITY CHECKS")
    report_lines.append("-" * 20)
    
    # Check for impossible scenarios
    impossible_scenarios = 0
    quality_issues = []
    
    # Check player counts
    invalid_players = df[(df['cts_alive'] < 0) | (df['cts_alive'] > 5) | 
                        (df['ts_alive'] < 0) | (df['ts_alive'] > 5)]
    if len(invalid_players) > 0:
        quality_issues.append(f"Invalid player counts: {len(invalid_players)} snapshots")
    
    # Check time values
    invalid_time = df[(df['time_left'] < 0) | (df['time_left'] > 115)]
    if len(invalid_time) > 0:
        quality_issues.append(f"Invalid time values: {len(invalid_time)} snapshots")
    
    # Check for missing winners
    missing_winners = df[df['winner'].isna()]
    if len(missing_winners) > 0:
        quality_issues.append(f"Missing winner data: {len(missing_winners)} snapshots")
    
    if quality_issues:
        for issue in quality_issues:
            report_lines.append(f"⚠️  {issue}")
    else:
        report_lines.append("✅ No data quality issues detected")
    
    report_lines.append("")
    
    # Dataset coverage analysis
    report_lines.append("DATASET COVERAGE ANALYSIS")
    report_lines.append("-" * 25)
    
    # Coverage by scenario type
    total_possible_scenarios = 3 * 5 * 5 * 2  # 3 phases * 5 CT * 5 T * 2 bomb states = 150
    covered_scenarios = len(scenarios_df)
    coverage_pct = covered_scenarios / total_possible_scenarios * 100
    
    report_lines.append(f"Scenario coverage: {covered_scenarios}/{total_possible_scenarios} ({coverage_pct:.1f}%)")
    
    # Scenarios with insufficient data
    low_sample_scenarios = scenarios_df[scenarios_df['total_samples'] < 50]
    report_lines.append(f"Scenarios with <50 samples: {len(low_sample_scenarios)}")
    
    high_sample_scenarios = scenarios_df[scenarios_df['total_samples'] >= 100]
    report_lines.append(f"Scenarios with ≥100 samples: {len(high_sample_scenarios)}")
    
    # Save report
    report_text = '\n'.join(report_lines)
    
    with open('../../outputs/reports/dataset_validation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("📋 Comprehensive report saved: ../../outputs/reports/dataset_validation_report.txt")
    
    return report_text

def main():
    """Main function to run the complete dataset analysis"""
    
    print("🔍 CS2 Dataset Statistics and Validation")
    print("=" * 60)
    
    # Create output directories
    Path('../../outputs/visualizations').mkdir(parents=True, exist_ok=True)
    Path('../../outputs/reports').mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = '../../data/datasets/all_snapshots.json'
    df = load_dataset(dataset_path)
    
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Clean the dataset
    df = clean_dataset(df)
    
    # Add time phase categorization
    df = analyze_time_phases(df)
    
    # Compute basic statistics
    basic_stats = compute_basic_statistics(df)
    
    # Analyze scenario probabilities
    scenarios_df = compute_scenario_probabilities(df)
    
    # Save scenario analysis
    scenarios_df.to_csv('../../outputs/reports/scenario_probabilities.csv', index=False, encoding='utf-8')
    print("💾 Scenario probabilities saved: ../../outputs/reports/scenario_probabilities.csv")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    # Overall heatmap
    create_player_count_heatmap(df)
    
    # Phase-specific heatmaps
    for phase in ['early', 'middle', 'late']:
        create_player_count_heatmap(df, time_phase=phase)
    
    # Bomb-specific heatmaps
    create_player_count_heatmap(df, bomb_planted=False)
    create_player_count_heatmap(df, bomb_planted=True)
    
    # Time analysis plots
    create_time_distribution_plot(df)
    
    # Additional analyses
    analyze_bomb_impact(df)
    analyze_player_advantage(df)
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    generate_comprehensive_report(df, scenarios_df, basic_stats)
    
    print("\n✅ Dataset analysis complete!")
    print("📊 Check ../../outputs/visualizations/ for charts")
    print("📋 Check ../../outputs/reports/ for detailed reports")
    print("\n🎯 Key findings:")
    print(f"   • {basic_stats['total_snapshots']:,} total snapshots from {basic_stats['unique_rounds']:,} rounds")
    print(f"   • {len(scenarios_df)} distinct scenarios with sufficient data (≥10 samples)")
    print(f"   • Bomb planted in {basic_stats['bomb_planted_pct']:.1f}% of snapshots")
    
    # Show winner distribution
    for winner, count in basic_stats['winner_distribution'].items():
        pct = count / basic_stats['total_snapshots'] * 100
        print(f"   • {winner.upper()} wins: {pct:.1f}%")

if __name__ == "__main__":
    main()
