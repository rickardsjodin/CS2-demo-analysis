"""
Test scenarios for CS2 win probability model validation
Creates a comprehensive table to sanity check model results against expected outcomes
"""

import sys
import os
from pathlib import Path
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.core.win_probability import get_win_probability
from train_win_probability_model import predict_win_probability

def load_dataset_baseline(dataset_path='../../data/datasets/all_snapshots.json'):
    """Load the actual dataset to use as baseline for comparison"""
    print("📂 Loading dataset for baseline comparison...")
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Clean the data (same cleaning as in dataset_statistics.py)
        print("🧹 Cleaning dataset...")
        initial_count = len(df)
        
        # Remove invalid entries
        df = df[
            (df['cts_alive'] >= 0) & (df['cts_alive'] <= 5) &
            (df['ts_alive'] >= 0) & (df['ts_alive'] <= 5) &
            (df['time_left'] >= 0) & (df['time_left'] <= 115) &
            (df['winner'].isin(['ct', 't']))
        ].copy()
        
        removed_count = initial_count - len(df)
        print(f"🗑️  Removed {removed_count} invalid entries ({removed_count/initial_count*100:.1f}%)")
        print(f"✅ Clean dataset: {len(df):,} valid snapshots")
        
        # Add time phase categorization with time ranges for better matching
        def categorize_time_phase(time_left):
            if time_left > 75:
                return "early"
            elif time_left >= 50 and time_left <= 70:  # Range for 60-second scenarios
                return "middle_60s"
            elif time_left > 35:
                return "middle"
            else:
                return "late"
        
        df['time_phase'] = df['time_left'].apply(categorize_time_phase)
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Using fallback theoretical baseline...")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print("Using fallback theoretical baseline...")
        return None

def compute_dataset_probabilities(dataset_df):
    """Compute win probabilities from the actual dataset"""
    if dataset_df is None:
        return {}
    
    print("📊 Computing win probabilities from dataset...")
    
    # Group by scenario characteristics and compute win rates
    scenario_probs = {}
    
    for _, group in dataset_df.groupby(['time_phase', 'cts_alive', 'ts_alive', 'bomb_planted']):
        if len(group) >= 5:  # Only use scenarios with at least 5 samples
            ct_wins = (group['winner'] == 'ct').sum()
            total = len(group)
            ct_win_rate = ct_wins / total
            
            # Create scenario key
            time_phase = group['time_phase'].iloc[0]
            cts_alive = group['cts_alive'].iloc[0]
            ts_alive = group['ts_alive'].iloc[0]
            bomb_planted = group['bomb_planted'].iloc[0]
            
            scenario_key = (time_phase, cts_alive, ts_alive, bomb_planted)
            scenario_probs[scenario_key] = {
                'ct_win_rate': ct_win_rate,
                'sample_count': total
            }
    
    print(f"✅ Computed probabilities for {len(scenario_probs)} scenarios")
    return scenario_probs

def get_dataset_baseline_prediction(time_left, ct_alive, t_alive, bomb_planted, dataset_probs):
    """Get baseline prediction from dataset statistics"""
    
    # Categorize time phase with specific handling for 60-second scenarios
    if time_left > 75:
        time_phase = "early"
    elif time_left == 60:  # Exact match for our test scenarios
        time_phase = "middle_60s"
    elif time_left >= 50 and time_left <= 70:
        time_phase = "middle_60s"
    elif time_left > 35:
        time_phase = "middle"
    else:
        time_phase = "late"
    
    # Look for exact match first
    scenario_key = (time_phase, ct_alive, t_alive, bomb_planted)
    
    if scenario_key in dataset_probs:
        return dataset_probs[scenario_key]['ct_win_rate']
    
    # If no exact match for middle_60s, try regular middle
    if time_phase == "middle_60s":
        fallback_key = ("middle", ct_alive, t_alive, bomb_planted)
        if fallback_key in dataset_probs:
            return dataset_probs[fallback_key]['ct_win_rate']
    
    # If no exact match, try without bomb status
    for bomb_status in [not bomb_planted, bomb_planted]:
        fallback_key = (time_phase, ct_alive, t_alive, bomb_status)
        if fallback_key in dataset_probs:
            # Apply bomb modifier if needed
            base_prob = dataset_probs[fallback_key]['ct_win_rate']
            if bomb_planted and not bomb_status:
                # Reduce CT win rate for bomb scenarios
                base_prob = max(0.0, base_prob - 0.25)  # Bomb gives T +25% advantage
            elif not bomb_planted and bomb_status:
                # Increase CT win rate for no-bomb scenarios
                base_prob = min(1.0, base_prob + 0.25)
            return base_prob
    
    # If still no match, try different time phases
    phase_fallbacks = ["middle_60s", "middle", "early", "late"] if time_phase == "middle_60s" else ["early", "middle", "middle_60s", "late"]
    for phase in phase_fallbacks:
        if phase == time_phase:  # Skip the original phase we already tried
            continue
        fallback_key = (phase, ct_alive, t_alive, bomb_planted)
        if fallback_key in dataset_probs:
            return dataset_probs[fallback_key]['ct_win_rate']
    
    # Final fallback - use theoretical baseline
    return get_win_probability(ct_alive, t_alive, bomb_planted)

def load_single_trained_model(model_file='../../data/models/ct_win_probability_model.pkl'):
    """Load a single trained model for predictions"""
    try:
        return joblib.load(model_file)
    except FileNotFoundError:
        print(f"❌ Model file {model_file} not found. Please train the model first.")
        return None

def load_all_trained_models():
    """Load all trained models for comparison"""
    
    print("📦 Loading all trained models for comparison...")
    
    # Load model summary
    try:
        with open('../../data/models/model_summary.json', 'r') as f:
            model_summary = json.load(f)
        print(f"✅ Found {model_summary['total_models']} models")
    except FileNotFoundError:
        print("❌ Model summary not found. Using default model only.")
        return {'default': load_single_trained_model()}
    
    models = {}
    
    # Load individual models
    for model_name, model_info in model_summary['models'].items():
        # Skip XGBoost for now due to calibration issues
        if model_name == 'xgboost':
            print(f"⚠️  Skipping {model_name} due to calibration issues")
            continue
            
        try:
            model_path = model_info['filename']
            model_data = joblib.load(model_path)
            models[model_name] = {
                'data': model_data,
                'performance': {
                    'accuracy': model_info['accuracy'],
                    'auc': model_info['auc'],
                    'log_loss': model_info['log_loss'],
                    'rank': model_info['rank']
                }
            }
            print(f"✅ Loaded {model_name} (Rank #{model_info['rank']}, AUC: {model_info['auc']:.3f})")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    
    # Load ensemble model if available
    try:
        ensemble_data = joblib.load('../../data/models/ct_win_probability_ensemble.pkl')
        models['ensemble'] = {
            'data': ensemble_data,
            'performance': {
                'accuracy': ensemble_data['accuracy'],
                'auc': ensemble_data['auc'],
                'log_loss': ensemble_data['log_loss'],
                'rank': 0  # Special rank for ensemble
            }
        }
        print(f"✅ Loaded ensemble model (AUC: {ensemble_data['auc']:.3f})")
    except FileNotFoundError:
        print("ℹ️  No ensemble model found")
    except Exception as e:
        print(f"❌ Failed to load ensemble: {e}")
    
    return models

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

def predict_with_model(model_data, time_left, cts_alive, ts_alive, bomb_planted):
    """Make prediction with a specific model"""
    
    # Create feature vector
    round_time_left = time_left if not bomb_planted else 0
    bomb_time_left = time_left if bomb_planted else 0
    
    feature_data = {
        'round_time_left': round_time_left,
        'bomb_time_left': bomb_time_left,
        'cts_alive': cts_alive, 
        'ts_alive': ts_alive, 
        'bomb_planted': bomb_planted,
    }
    
    # Create DataFrame with proper feature names
    feature_columns = model_data['feature_columns']
    X = pd.DataFrame([feature_data], columns=feature_columns)
    
    # Scale if needed
    scaler = model_data.get('scaler')
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict probability - handle both calibrated and original models
    model = model_data['model']
    if model is not None:
        ct_win_prob = model.predict_proba(X)[0, 1]
    else:
        # Fallback to original model if calibrated model is None
        original_model = model_data.get('original_model')
        if original_model is not None:
            ct_win_prob = original_model.predict_proba(X)[0, 1]
        else:
            raise ValueError("Both model and original_model are None")
    
    return ct_win_prob

def predict_with_ensemble(models, time_left, cts_alive, ts_alive, bomb_planted):
    """Make prediction with ensemble of all models"""
    
    predictions = []
    
    for model_name, model_info in models.items():
        if model_name == 'ensemble':  # Skip ensemble to avoid recursion
            continue
        if model_name == 'xgboost':  # Skip XGBoost due to issues
            continue
        
        try:
            pred = predict_with_model(model_info['data'], time_left, cts_alive, ts_alive, bomb_planted)
            predictions.append(pred)
        except Exception as e:
            print(f"⚠️ Error with {model_name}: {e}")
    
    if predictions:
        return np.mean(predictions)
    else:
        return 0.5  # Default fallback

def run_all_model_predictions(scenarios, models):
    """Run predictions using all loaded models"""
    
    results = {}
    model_names = list(models.keys())
    
    print(f"🔮 Running predictions with {len(models)} models...")
    
    for model_name, model_info in models.items():
        # Skip XGBoost due to calibration issues
        if model_name == 'xgboost':
            print(f"   Skipping {model_name} (calibration issues)")
            continue
            
        print(f"   Running {model_name}...")
        predictions = []
        
        for scenario in scenarios:
            try:
                if model_name == 'ensemble':
                    # Use ensemble prediction
                    pred = predict_with_ensemble(models, 
                                               scenario['time_left'],
                                               scenario['ct_alive'], 
                                               scenario['t_alive'],
                                               scenario['bomb_planted'])
                else:
                    # Use individual model
                    pred = predict_with_model(model_info['data'],
                                            scenario['time_left'],
                                            scenario['ct_alive'],
                                            scenario['t_alive'], 
                                            scenario['bomb_planted'])
                
                predictions.append(pred)
                
            except Exception as e:
                print(f"❌ Error predicting {scenario['scenario']} with {model_name}: {e}")
                predictions.append(0.5)  # Default value
        
        results[model_name] = predictions
    
    return results

def run_baseline_predictions(scenarios, dataset_probs=None):
    """Run predictions using dataset-based baseline or theoretical fallback"""
    
    predictions = []
    
    for scenario in scenarios:
        if dataset_probs is not None:
            # Use dataset-based prediction
            prob = get_dataset_baseline_prediction(
                scenario['time_left'],
                scenario['ct_alive'],
                scenario['t_alive'],
                scenario['bomb_planted'],
                dataset_probs
            )
        else:
            # Fallback to theoretical baseline
            prob = get_win_probability(
                scenario['ct_alive'],
                scenario['t_alive'],
                scenario['bomb_planted']
            )
        
        predictions.append(prob)
    
    return predictions

def create_multi_model_comparison_table(scenarios, all_model_predictions, baseline_predictions, baseline_type="Dataset"):
    """Create a comprehensive comparison table with all models"""
    
    # Base scenario information
    base_data = {
        'scenario': [s['scenario'] for s in scenarios],
        'time_left': [s['time_left'] for s in scenarios],
        'ct_alive': [s['ct_alive'] for s in scenarios],
        't_alive': [s['t_alive'] for s in scenarios],
        'bomb_planted': [s['bomb_planted'] for s in scenarios],
        'description': [s['description'] for s in scenarios],
        'baseline_prediction': baseline_predictions,
        'baseline_type': baseline_type
    }
    
    # Add predictions for each model
    for model_name, predictions in all_model_predictions.items():
        base_data[f'{model_name}_prediction'] = predictions
        base_data[f'{model_name}_diff_from_baseline'] = [abs(ml - bl) for ml, bl in zip(predictions, baseline_predictions)]
    
    df = pd.DataFrame(base_data)
    
    # Add ensemble statistics
    model_columns = [f'{name}_prediction' for name in all_model_predictions.keys() if name != 'ensemble']
    if len(model_columns) > 1:
        df['model_avg'] = df[model_columns].mean(axis=1)
        df['model_std'] = df[model_columns].std(axis=1)
        df['model_range'] = df[model_columns].max(axis=1) - df[model_columns].min(axis=1)
    
    return df

def create_heatmap_table(df, prediction_type='ml_prediction', model_name='ML Model'):
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
    
    plt.title(f'T Win Probability Heatmap - {model_name.replace("_", " ").title()}\n(60 seconds remaining, no bomb planted)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    # Customize the labels - reverse order for both axes to match 5v5 top-left layout
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'../../outputs/visualizations/t_win_probability_heatmap_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Heatmap for {model_name} saved as '{filename}'")
    
    return heatmap_data

def create_bomb_scenario_heatmap(df, prediction_type='ml_prediction', model_name='ML Model'):
    """Create heatmap for post-plant scenarios"""
    
    bomb_df = df[df['bomb_planted'] == True].copy()
    
    # Use 10 second bomb timer scenarios
    bomb_scenarios = bomb_df[
        (bomb_df['ct_alive'] <= 5) & 
        (bomb_df['t_alive'] <= 5) &
        (bomb_df['time_left'] == 10)
    ].copy()
    
    if len(bomb_scenarios) == 0:
        print(f"❌ No valid bomb scenarios found for {model_name}")
        return
    
    # Create pivot table
    heatmap_data = bomb_scenarios.pivot_table(
        values=prediction_type,
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
    
    plt.title(f'T Win Probability - Post-Plant Scenarios - {model_name.replace("_", " ").title()}\n(10 seconds on bomb timer)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    filename = f'../../outputs/visualizations/t_win_probability_heatmap_bomb_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Bomb scenario heatmap for {model_name} saved as '{filename}'")

def create_difference_heatmap(df, prediction_type='ml_prediction', model_name='ML Model'):
    """Create a heatmap showing differences from baseline"""
    
    # Filter for no bomb scenarios first
    no_bomb_df = df[df['bomb_planted'] == False].copy()
    
    # Create pivot table for 5x5 player scenarios
    player_scenarios = no_bomb_df[
        (no_bomb_df['ct_alive'] <= 5) & 
        (no_bomb_df['t_alive'] <= 5) &
        (no_bomb_df['time_left'] == 60)  # Use mid-round timing
    ].copy()
    
    if len(player_scenarios) == 0:
        print(f"❌ No valid scenarios found for difference heatmap for {model_name}")
        return
    
    # Calculate differences (T win perspective)
    player_scenarios['model_t_win'] = (1 - player_scenarios[prediction_type]) * 100
    player_scenarios['baseline_t_win'] = (1 - player_scenarios['baseline_prediction']) * 100
    player_scenarios['difference'] = player_scenarios['model_t_win'] - player_scenarios['baseline_t_win']
    
    # Create pivot table for differences
    heatmap_data = player_scenarios.pivot_table(
        values='difference',
        index='t_alive',
        columns='ct_alive',
        aggfunc='mean'
    )
    
    # Reverse both axes to have 5v5 at top left and 1v1 at bottom right
    heatmap_data = heatmap_data.iloc[::-1, ::-1]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create custom diverging colormap (red for negative, blue for positive)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
    n_bins = 100
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
    
    # Use consistent scale from -50 to +50 for all models
    vmin, vmax = -50, 50
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Difference in T Win Probability (%)'},
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.title(f'Difference from Baseline - {model_name.replace("_", " ").title()}\n(60 seconds remaining, no bomb planted)\nScale: -50% to +50% | Red = Model lower, Blue = Model higher', 
              fontsize=14, fontweight='bold')
    plt.xlabel('CT Players Alive', fontsize=12, fontweight='bold')
    plt.ylabel('T Players Alive', fontsize=12, fontweight='bold')
    
    # Customize the labels - reverse order for both axes to match 5v5 top-left layout
    plt.gca().set_xticklabels([f'{i} CT' for i in range(5, 0, -1)], fontweight='bold')
    plt.gca().set_yticklabels([f'{i} T' for i in range(5, 0, -1)], fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'../../outputs/visualizations/difference_from_baseline_heatmap_{model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Difference heatmap for {model_name} saved as '{filename}'")
    
    return heatmap_data

def create_reference_table(df, prediction_type='ml_prediction', model_name=None):
    """Create a table that exactly matches the reference image format"""
    
    # Filter for no bomb scenarios with 60 seconds remaining
    no_bomb_df = df[
        (df['bomb_planted'] == False) & 
        (df['time_left'] == 60)
    ].copy()
    
    # Use model_name if provided, otherwise derive from prediction_type
    display_name = model_name.replace('_', ' ').title() if model_name else prediction_type.replace('_', ' ').title()
    
    # Create the exact table structure from the reference image
    print(f"\n📊 T Win % Table - {display_name}")
    print("=" * 60)
    
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
                t_win_prob = (1 - scenario_data[prediction_type].iloc[0]) * 100
                row += f"{t_win_prob:.1f}%".rjust(10)
            else:
                row += "N/A".rjust(10)
        
        print(row)
    
    print("\n")

def create_difference_table(df, prediction_type='ml_prediction', model_name=None):
    """Create a table showing the difference from baseline predictions"""
    
    # Filter for no bomb scenarios with 60 seconds remaining
    no_bomb_df = df[
        (df['bomb_planted'] == False) & 
        (df['time_left'] == 60)
    ].copy()
    
    # Use model_name if provided, otherwise derive from prediction_type
    display_name = model_name.replace('_', ' ').title() if model_name else prediction_type.replace('_', ' ').title()
    
    # Create the difference table
    print(f"\n📊 Difference from Baseline Table - {display_name}")
    print("=" * 70)
    print("(Positive values = Model predicts higher T win rate than baseline)")
    print("(Negative values = Model predicts lower T win rate than baseline)")
    print("=" * 70)
    
    # Header
    header = "Diff%".ljust(8) + "".join([f"{i} CT".rjust(10) for i in range(5, 0, -1)])
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
                # Calculate difference (T win perspective)
                model_t_win = (1 - scenario_data[prediction_type].iloc[0]) * 100
                baseline_t_win = (1 - scenario_data['baseline_prediction'].iloc[0]) * 100
                diff = model_t_win - baseline_t_win
                
                # Format with sign
                if diff >= 0:
                    row += f"+{diff:.1f}%".rjust(10)
                else:
                    row += f"{diff:.1f}%".rjust(10)
            else:
                row += "N/A".rjust(10)
        
        print(row)
    
    print("\n")

def analyze_prediction_differences(df, dataset_probs=None):
    """Analyze differences between ML and baseline predictions"""
    
    baseline_type = df['baseline_type'].iloc[0] if 'baseline_type' in df.columns else "Theoretical"
    
    print(f"\n🔍 Prediction Analysis (vs {baseline_type} Baseline):")
    print("=" * 60)
    print(f"📊 Total scenarios tested: {len(df)}")
    print(f"📈 Average ML prediction: {df['ml_prediction'].mean():.1%}")
    print(f"📉 Average baseline prediction: {df['baseline_prediction'].mean():.1%}")
    print(f"🔄 Average difference: {df['difference'].mean():.3f}")
    print(f"📏 Max difference: {df['difference'].max():.3f}")
    print(f"📐 Std dev of differences: {df['difference'].std():.3f}")
    
    # Calculate correlation
    correlation = df['ml_prediction'].corr(df['baseline_prediction'])
    print(f"🔗 Correlation: {correlation:.3f}")
    
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
        if len(subset) > 0:
            print(f"  {bomb_label}: {len(subset)} scenarios, avg diff: {subset['difference'].mean():.3f}")
    
    # If using dataset baseline, show coverage statistics
    if dataset_probs is not None:
        print(f"\n📊 Dataset Baseline Coverage:")
        print(f"   Total scenarios in dataset: {len(dataset_probs)}")
        
        # Count how many test scenarios had exact matches
        exact_matches = 0
        for _, row in df.iterrows():
            time_phase = "early" if row['time_left'] > 75 else "middle" if row['time_left'] > 35 else "late"
            scenario_key = (time_phase, row['ct_alive'], row['t_alive'], row['bomb_planted'])
            if scenario_key in dataset_probs:
                exact_matches += 1
        
        exact_match_pct = exact_matches / len(df) * 100
        print(f"   Exact scenario matches: {exact_matches}/{len(df)} ({exact_match_pct:.1f}%)")
    
    return {
        'avg_difference': df['difference'].mean(),
        'max_difference': df['difference'].max(),
        'correlation': correlation,
        'total_scenarios': len(df)
    }

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
    
    print("🧪 CS2 Win Probability Model Test Scenarios - All Models Comparison")
    print("=" * 80)
    
    # Load the dataset for baseline comparison
    print("📊 Loading dataset for baseline comparison...")
    dataset_df = load_dataset_baseline()
    dataset_probs = compute_dataset_probabilities(dataset_df) if dataset_df is not None else None
    
    baseline_type = "Dataset-based" if dataset_probs else "Theoretical"
    print(f"✅ Using {baseline_type} baseline for comparison")
    
    # Load all trained models
    print("\n📦 Loading all trained models...")
    all_models = load_all_trained_models()
    
    if not all_models:
        print("❌ No models found. Please train models first.")
        return
    
    # Create model performance summary
    print(f"\n📋 Model Performance Summary:")
    print("=" * 70)
    print(f"{'Model':<20} {'Rank':<6} {'AUC':<8} {'Accuracy':<10} {'Log Loss':<10}")
    print("=" * 70)
    
    # Sort by rank
    sorted_models = sorted(all_models.items(), 
                          key=lambda x: x[1]['performance']['rank'] if x[1]['performance']['rank'] > 0 else 999)
    
    for model_name, model_info in sorted_models:
        perf = model_info['performance']
        rank_str = str(perf['rank']) if perf['rank'] > 0 else 'Ens'
        
        print(f"{model_name.replace('_', ' ').title():<20} "
              f"{rank_str:<6} "
              f"{perf['auc']:.3f}    "
              f"{perf['accuracy']:.3f}    "
              f"{perf['log_loss']:.3f}")
    
    # Create test scenarios
    print("\n🔧 Creating test scenarios...")
    scenarios = create_test_scenarios()
    print(f"✅ Created {len(scenarios)} test scenarios")
    
    # Run predictions with all models
    all_model_predictions = run_all_model_predictions(scenarios, all_models)
    
    print(f"📊 Running {baseline_type.lower()} baseline predictions...")
    baseline_predictions = run_baseline_predictions(scenarios, dataset_probs)
    
    # Create comparison table
    print("📋 Creating comprehensive comparison table...")
    df = create_multi_model_comparison_table(scenarios, all_model_predictions, baseline_predictions, baseline_type)
    
    # Save detailed results
    df.to_csv('../../outputs/reports/all_models_test_scenario_results.csv', index=False, encoding='utf-8')
    print("💾 Detailed results saved to '../../outputs/reports/all_models_test_scenario_results.csv'")
    
    # Analyze differences for each model
    model_names = list(all_models.keys())
    print(f"\n🔍 Model Performance Analysis (vs {baseline_type} Baseline):")
    print("=" * 80)
    
    model_stats = {}
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        diff_col = f'{model_name}_diff_from_baseline'
        
        if pred_col in df.columns:
            model_predictions = df[pred_col]
            differences = df[diff_col]
            correlation = model_predictions.corr(df['baseline_prediction'])
            
            model_stats[model_name] = {
                'avg_prediction': model_predictions.mean(),
                'avg_difference': differences.mean(),
                'max_difference': differences.max(),
                'correlation': correlation
            }
            
            print(f"\n📊 {model_name.replace('_', ' ').title()}:")
            print(f"   Average prediction: {model_predictions.mean():.1%}")
            print(f"   Average difference from baseline: {differences.mean():.3f}")
            print(f"   Max difference: {differences.max():.3f}")
            print(f"   Correlation with baseline: {correlation:.3f}")
    
    # Model agreement analysis
    print(f"\n🤝 Model Agreement Analysis:")
    print("=" * 50)
    
    model_pred_columns = [f'{name}_prediction' for name in model_names]
    if len(model_pred_columns) > 1:
        # Calculate correlations between models
        correlation_matrix = df[model_pred_columns].corr()
        print("\n📊 Model Correlation Matrix:")
        print(correlation_matrix.round(3))
        
        # Find scenarios with highest disagreement
        if 'model_range' in df.columns:
            print("\n🚨 Top 5 scenarios with highest model disagreement:")
            top_disagreement = df.nlargest(5, 'model_range')
            
            for _, row in top_disagreement.iterrows():
                print(f"\n  📍 {row['description']}:")
                print(f"      Range: {row['model_range']:.3f} (std: {row.get('model_std', 0):.3f})")
                for model_name in model_names:
                    pred_col = f'{model_name}_prediction'
                    if pred_col in row:
                        print(f"      {model_name}: {row[pred_col]:.1%}")
    
    # Create reference tables for each model
    print("\n🎯 Creating reference tables for each model...")
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        if pred_col in df.columns:
            create_reference_table(df, pred_col, model_name)
    
    # Create baseline reference table
    create_reference_table(df, 'baseline_prediction', 'baseline')
    
    # Create difference tables for each model
    print("\n📊 Creating difference from baseline tables for each model...")
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        if pred_col in df.columns:
            create_difference_table(df, pred_col, model_name)
    
    # Create heatmaps for each model
    print("\n🎨 Creating visualization heatmaps for all models...")
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        if pred_col in df.columns:
            try:
                create_heatmap_table(df, pred_col, model_name)
            except Exception as e:
                print(f"❌ Error creating heatmap for {model_name}: {e}")
    
    # Create baseline heatmap
    try:
        create_heatmap_table(df, 'baseline_prediction', 'baseline')
    except Exception as e:
        print(f"❌ Error creating baseline heatmap: {e}")
    
    # Create difference heatmaps for each model
    print("\n🔍 Creating difference from baseline heatmaps...")
    for model_name in model_names:
        pred_col = f'{model_name}_prediction'
        if pred_col in df.columns:
            try:
                create_difference_heatmap(df, pred_col, model_name)
            except Exception as e:
                print(f"❌ Error creating difference heatmap for {model_name}: {e}")
    
    # Bomb scenario heatmaps for top models
    top_models = sorted(model_names, key=lambda x: all_models[x]['performance']['rank'] if all_models[x]['performance']['rank'] > 0 else 999)[:3]
    for model_name in top_models:
        pred_col = f'{model_name}_prediction'
        if pred_col in df.columns:
            try:
                create_bomb_scenario_heatmap(df, pred_col, model_name)
            except Exception as e:
                print(f"❌ Error creating bomb heatmap for {model_name}: {e}")
    
    # Save model comparison summary
    summary_data = []
    for model_name, stats in model_stats.items():
        summary_data.append({
            'model_name': model_name,
            'rank': all_models[model_name]['performance']['rank'],
            'auc': all_models[model_name]['performance']['auc'],
            'accuracy': all_models[model_name]['performance']['accuracy'],
            'avg_prediction': stats['avg_prediction'],
            'avg_difference_from_baseline': stats['avg_difference'],
            'max_difference_from_baseline': stats['max_difference'],
            'correlation_with_baseline': stats['correlation']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('../../outputs/reports/all_models_performance_summary.csv', index=False, encoding='utf-8')
    print("💾 Model performance summary saved to '../../outputs/reports/all_models_performance_summary.csv'")
    
    # Overall summary
    best_correlation = max(model_stats.values(), key=lambda x: x['correlation'])
    best_model_name = [k for k, v in model_stats.items() if v == best_correlation][0]
    
    print(f"\n🎯 OVERALL MODEL COMPARISON SUMMARY:")
    print("=" * 50)
    print(f"Models tested: {len(model_names)}")
    print(f"Scenarios tested: {len(scenarios)}")
    print(f"Baseline type: {baseline_type}")
    print(f"Best correlation with baseline: {best_correlation['correlation']:.3f} ({best_model_name})")
    print(f"Average model agreement (std): {df.get('model_std', pd.Series([0])).mean():.3f}")
    
    print("\n✅ All models test scenarios complete!")
    print("📊 Check the generated PNG files for visual comparisons")
    print("📋 Check the CSV files for detailed numerical results")
    print("🎯 All models have been compared across scenarios")

if __name__ == "__main__":
    main()
