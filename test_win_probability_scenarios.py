"""
Test scenarios for CS2 win probability model validation
Creates a comprehensive table to sanity check model results against expected outcomes
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from win_probability import get_win_probability
from train_win_probability_model import predict_win_probability

def load_trained_model(model_file='ct_win_probability_model.pkl'):
    """Load the trained model for predictions"""
    try:
        return joblib.load(model_file)
    except FileNotFoundError:
        print(f"❌ Model file {model_file} not found. Please train the model first.")
        return None

def create_test_scenarios():
    """Create comprehensive test scenarios covering various game states"""
    
    scenarios = []
    
    # Standard scenarios - no bomb planted
    time_values = [115, 90, 60, 30, 15, 5]  # Different time points
    
    for time_left in time_values:
        for ct_alive in range(1, 6):
            for t_alive in range(1, 6):
                scenarios.append({
                    'scenario': f'{ct_alive}v{t_alive}_T{time_left}s',
                    'time_left': time_left,
                    'ct_alive': ct_alive,
                    't_alive': t_alive,
                    'bomb_planted': False,
                    'description': f'{ct_alive} CT vs {t_alive} T, {time_left}s left, no bomb'
                })
    
    # Post-plant scenarios
    bomb_time_values = [40, 30, 20, 10, 5, 2]  # Bomb timer values
    
    for time_left in bomb_time_values:
        for ct_alive in range(1, 6):
            for t_alive in range(1, 6):
                scenarios.append({
                    'scenario': f'{ct_alive}v{t_alive}_bomb_{time_left}s',
                    'time_left': time_left,
                    'ct_alive': ct_alive,
                    't_alive': t_alive,
                    'bomb_planted': True,
                    'description': f'{ct_alive} CT vs {t_alive} T, bomb planted, {time_left}s left'
                })
    
    return scenarios

def run_model_predictions(scenarios, model_data):
    """Run predictions using the trained ML model"""
    
    if model_data is None:
        return [0.5] * len(scenarios)  # Default if no model
    
    predictions = []
    
    for scenario in scenarios:
        try:
            prob = predict_win_probability(
                scenario['time_left'],
                scenario['ct_alive'],
                scenario['t_alive'],
                scenario['bomb_planted']
            )
            predictions.append(prob)
        except Exception as e:
            print(f"Error predicting {scenario['scenario']}: {e}")
            predictions.append(0.5)  # Default value
    
    return predictions

def run_baseline_predictions(scenarios):
    """Run predictions using the baseline win probability function"""
    
    predictions = []
    
    for scenario in scenarios:
        prob = get_win_probability(
            scenario['ct_alive'],
            scenario['t_alive'],
            scenario['bomb_planted']
        )
        predictions.append(prob)
    
    return predictions

def create_comparison_table(scenarios, ml_predictions, baseline_predictions):
    """Create a comprehensive comparison table"""
    
    df = pd.DataFrame({
        'scenario': [s['scenario'] for s in scenarios],
        'time_left': [s['time_left'] for s in scenarios],
        'ct_alive': [s['ct_alive'] for s in scenarios],
        't_alive': [s['t_alive'] for s in scenarios],
        'bomb_planted': [s['bomb_planted'] for s in scenarios],
        'description': [s['description'] for s in scenarios],
        'ml_prediction': ml_predictions,
        'baseline_prediction': baseline_predictions,
        'difference': [abs(ml - bl) for ml, bl in zip(ml_predictions, baseline_predictions)]
    })
    
    return df

def create_heatmap_table(df, prediction_type='ml_prediction'):
    """Create a heatmap table similar to the one in the attachment"""
    
    # Filter for no bomb scenarios first (like the original table)
    no_bomb_df = df[df['bomb_planted'] == False].copy()
    
    # Create pivot table for 5x5 player scenarios
    player_scenarios = no_bomb_df[
        (no_bomb_df['ct_alive'] <= 5) & 
        (no_bomb_df['t_alive'] <= 5) &
        (no_bomb_df['time_left'] == 60)  # Use mid-round timing
    ].copy()
    
    if len(player_scenarios) == 0:
        print("❌ No valid scenarios found for heatmap")
        return
    
    # Create pivot table
    heatmap_data = player_scenarios.pivot_table(
        values=prediction_type,
        index='t_alive',
        columns='ct_alive',
        aggfunc='mean'
    )
    
    # Convert to T win percentages (1 - CT win probability)
    heatmap_data = (1 - heatmap_data) * 100
    
    # Reverse both axes to have 5v5 at top left and 1v1 at bottom right
    heatmap_data = heatmap_data.iloc[::-1, ::-1]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap (red to green like the original)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    n_bins = 100
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'T Win Probability (%)'},
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title(f'T Win Probability Heatmap - {prediction_type.replace("_", " ").title()}\n(60 seconds remaining, no bomb planted)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    # Customize the labels - reverse order for both axes to match 5v5 top-left layout
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    
    filename = f't_win_probability_heatmap_{prediction_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Heatmap saved as '{filename}'")
    
    return heatmap_data

def create_bomb_scenario_heatmap(df):
    """Create heatmap for post-plant scenarios"""
    
    bomb_df = df[df['bomb_planted'] == True].copy()
    
    # Use 10 second bomb timer scenarios
    bomb_scenarios = bomb_df[
        (bomb_df['ct_alive'] <= 5) & 
        (bomb_df['t_alive'] <= 5) &
        (bomb_df['time_left'] == 10)
    ].copy()
    
    if len(bomb_scenarios) == 0:
        print("❌ No valid bomb scenarios found")
        return
    
    # Create pivot table
    heatmap_data = bomb_scenarios.pivot_table(
        values='ml_prediction',
        index='t_alive',
        columns='ct_alive',
        aggfunc='mean'
    )
    
    # Convert to T win percentages and reverse both axes
    heatmap_data = (1 - heatmap_data) * 100
    heatmap_data = heatmap_data.iloc[::-1, ::-1]
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=100)
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'T Win Probability (%)'},
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title('T Win Probability - Post-Plant Scenarios\n(10 seconds on bomb timer)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('t_win_probability_heatmap_bomb.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Bomb scenario heatmap saved as 't_win_probability_heatmap_bomb.png'")

def create_reference_table(df, prediction_type='ml_prediction'):
    """Create a table that exactly matches the reference image format"""
    
    # Filter for no bomb scenarios with 60 seconds remaining
    no_bomb_df = df[
        (df['bomb_planted'] == False) & 
        (df['time_left'] == 60)
    ].copy()
    
    # Create the exact table structure from the reference image
    print(f"\n📊 T Win % Table - {prediction_type.replace('_', ' ').title()}")
    print("=" * 50)
    
    # Header
    header = "T Win%".ljust(8) + "".join([f"{i} CT".rjust(10) for i in range(5, 0, -1)])
    print(header)
    print("-" * 58)
    
    # Data rows (5T to 1T, top to bottom)
    for t_players in range(5, 0, -1):
        row = f"{t_players} T".ljust(8)
        
        for ct_players in range(5, 0, -1):
            # Find the scenario
            scenario_data = no_bomb_df[
                (no_bomb_df['ct_alive'] == ct_players) & 
                (no_bomb_df['t_alive'] == t_players)
            ]
            
            if len(scenario_data) > 0:
                # Convert to T win percentage
                if prediction_type == 'ml_prediction':
                    t_win_prob = (1 - scenario_data[prediction_type].iloc[0]) * 100
                else:
                    t_win_prob = (1 - scenario_data[prediction_type].iloc[0]) * 100
                row += f"{t_win_prob:.1f}%".rjust(10)
            else:
                row += "N/A".rjust(10)
        
        print(row)
    
    print("\n")

def create_reference_table(df, prediction_type='ml_prediction'):
    """Create a table that exactly matches the reference image format"""
    
    # Filter for no bomb scenarios with 60 seconds remaining
    no_bomb_df = df[
        (df['bomb_planted'] == False) & 
        (df['time_left'] == 60)
    ].copy()
    
    # Create the exact table structure from the reference image
    print(f"\n📊 T Win % Table - {prediction_type.replace('_', ' ').title()}")
    print("=" * 50)
    
    # Header
    header = "T Win%".ljust(8) + "".join([f"{i} CT".rjust(10) for i in range(5, 0, -1)])
    print(header)
    print("-" * 58)
    
    # Data rows (5T to 1T, top to bottom)
    for t_players in range(5, 0, -1):
        row = f"{t_players} T".ljust(8)
        
        for ct_players in range(5, 0, -1):
            # Find the scenario
            scenario_data = no_bomb_df[
                (no_bomb_df['ct_alive'] == ct_players) & 
                (no_bomb_df['t_alive'] == t_players)
            ]
            
            if len(scenario_data) > 0:
                # Convert to T win percentage
                if prediction_type == 'ml_prediction':
                    t_win_prob = (1 - scenario_data[prediction_type].iloc[0]) * 100
                else:
                    t_win_prob = (1 - scenario_data[prediction_type].iloc[0]) * 100
                row += f"{t_win_prob:.1f}%".rjust(10)
            else:
                row += "N/A".rjust(10)
        
        print(row)
    
    print("\n")

def analyze_prediction_differences(df):
    """Analyze differences between ML and baseline predictions"""
    
    print("\n🔍 Prediction Analysis:")
    print(f"📊 Total scenarios tested: {len(df)}")
    print(f"📈 Average ML prediction: {df['ml_prediction'].mean():.1%}")
    print(f"📉 Average baseline prediction: {df['baseline_prediction'].mean():.1%}")
    print(f"🔄 Average difference: {df['difference'].mean():.3f}")
    print(f"📏 Max difference: {df['difference'].max():.3f}")
    
    # Find scenarios with largest differences
    print(f"\n🚨 Top 5 scenarios with largest differences:")
    top_diff = df.nlargest(5, 'difference')[['scenario', 'description', 'ml_prediction', 'baseline_prediction', 'difference']]
    
    for _, row in top_diff.iterrows():
        print(f"  {row['scenario']}: ML={row['ml_prediction']:.1%}, Baseline={row['baseline_prediction']:.1%}, Diff={row['difference']:.3f}")
    
    # Analyze by bomb status
    print(f"\n💣 Analysis by bomb status:")
    for bomb_status in [False, True]:
        subset = df[df['bomb_planted'] == bomb_status]
        bomb_label = "Post-plant" if bomb_status else "No bomb"
        print(f"  {bomb_label}: {len(subset)} scenarios, avg diff: {subset['difference'].mean():.3f}")

def create_summary_statistics_table(df):
    """Create a summary statistics table"""
    
    # Group by player counts
    summary_stats = []
    
    for ct in range(1, 6):
        for t in range(1, 6):
            scenario_data = df[(df['ct_alive'] == ct) & (df['t_alive'] == t)]
            
            if len(scenario_data) > 0:
                summary_stats.append({
                    'ct_alive': ct,
                    't_alive': t,
                    'scenario_count': len(scenario_data),
                    'ml_avg': scenario_data['ml_prediction'].mean(),
                    'baseline_avg': scenario_data['baseline_prediction'].mean(),
                    'avg_difference': scenario_data['difference'].mean(),
                    'max_difference': scenario_data['difference'].max()
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    print("\n📋 Summary Statistics by Player Count:")
    print(summary_df.round(3))
    
    return summary_df

def create_detailed_test_report(scenarios, ml_predictions, baseline_predictions):
    """Create a detailed test report with specific edge cases"""
    
    print("\n🧪 Detailed Test Scenarios:")
    print("=" * 80)
    
    # Test specific scenarios that should have predictable outcomes
    test_cases = [
        # Should heavily favor CT
        {'time': 90, 'ct': 5, 't': 1, 'bomb': False, 'expected': 'High CT win (>90%)'},
        {'time': 60, 'ct': 4, 't': 1, 'bomb': False, 'expected': 'High CT win (>85%)'},
        
        # Should heavily favor T
        {'time': 30, 'ct': 1, 't': 5, 'bomb': False, 'expected': 'High T win (<15%)'},
        {'time': 15, 'ct': 1, 't': 4, 'bomb': False, 'expected': 'High T win (<20%)'},
        
        # Close scenarios
        {'time': 60, 'ct': 3, 't': 3, 'bomb': False, 'expected': 'Close (40-60%)'},
        {'time': 45, 'ct': 2, 't': 2, 'bomb': False, 'expected': 'Close (40-60%)'},
        
        # Post-plant scenarios (should favor T)
        {'time': 20, 'ct': 3, 't': 3, 'bomb': True, 'expected': 'T advantage (<45%)'},
        {'time': 10, 'ct': 2, 't': 2, 'bomb': True, 'expected': 'Strong T advantage (<40%)'},
        {'time': 5, 'ct': 1, 't': 1, 'bomb': True, 'expected': 'Very strong T advantage (<35%)'},
    ]
    
    for test_case in test_cases:
        # Find matching scenario
        matching_scenarios = [
            (i, s) for i, s in enumerate(scenarios) 
            if s['time_left'] == test_case['time'] and 
               s['ct_alive'] == test_case['ct'] and 
               s['t_alive'] == test_case['t'] and 
               s['bomb_planted'] == test_case['bomb']
        ]
        
        if matching_scenarios:
            idx, scenario = matching_scenarios[0]
            ml_pred = ml_predictions[idx]
            baseline_pred = baseline_predictions[idx]
            
            print(f"\n🎯 {scenario['description']}")
            print(f"   Expected: {test_case['expected']}")
            print(f"   ML Model: {ml_pred:.1%}")
            print(f"   Baseline: {baseline_pred:.1%}")
            print(f"   Difference: {abs(ml_pred - baseline_pred):.3f}")
            
            # Validate expectations
            if 'High CT win' in test_case['expected']:
                if ml_pred > 0.85:
                    print("   ✅ ML model meets expectation")
                else:
                    print("   ⚠️ ML model below expectation")
            elif 'High T win' in test_case['expected']:
                if ml_pred < 0.20:
                    print("   ✅ ML model meets expectation")
                else:
                    print("   ⚠️ ML model above expectation")
            elif 'Close' in test_case['expected']:
                if 0.4 <= ml_pred <= 0.6:
                    print("   ✅ ML model meets expectation")
                else:
                    print("   ⚠️ ML model outside expected range")

def main():
    """Main function to run all tests and create visualizations"""
    
    print("🧪 CS2 Win Probability Model Test Scenarios")
    print("=" * 60)
    
    # Load the trained model
    print("📦 Loading trained model...")
    model_data = load_trained_model()
    
    # Create test scenarios
    print("🔧 Creating test scenarios...")
    scenarios = create_test_scenarios()
    print(f"✅ Created {len(scenarios)} test scenarios")
    
    # Run predictions
    print("🔮 Running ML model predictions...")
    ml_predictions = run_model_predictions(scenarios, model_data)
    
    print("📊 Running baseline predictions...")
    baseline_predictions = run_baseline_predictions(scenarios)
    
    # Create comparison table
    print("📋 Creating comparison table...")
    df = create_comparison_table(scenarios, ml_predictions, baseline_predictions)
    
    # Save detailed results
    df.to_csv('test_scenario_results.csv', index=False)
    print("💾 Detailed results saved to 'test_scenario_results.csv'")
    
    # Analyze differences
    analyze_prediction_differences(df)
    
    # Create reference tables that match the original format
    print("\n🎯 Creating reference tables...")
    create_reference_table(df, 'ml_prediction')
    create_reference_table(df, 'baseline_prediction')
    
    # Create summary statistics
    summary_df = create_summary_statistics_table(df)
    summary_df.to_csv('summary_statistics.csv', index=False)
    print("💾 Summary statistics saved to 'summary_statistics.csv'")
    
    # Create heatmaps
    print("\n🎨 Creating visualization heatmaps...")
    
    # ML model heatmap
    ml_heatmap = create_heatmap_table(df, 'ml_prediction')
    
    # Baseline heatmap
    baseline_heatmap = create_heatmap_table(df, 'baseline_prediction')
    
    # Bomb scenario heatmap
    create_bomb_scenario_heatmap(df)
    
    # Create detailed test report
    create_detailed_test_report(scenarios, ml_predictions, baseline_predictions)
    
    print("\n✅ Test scenarios complete!")
    print("📊 Check the generated PNG files for visual comparisons")
    print("📋 Check the CSV files for detailed numerical results")

if __name__ == "__main__":
    main()
