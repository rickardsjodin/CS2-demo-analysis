"""
Evaluate Binning Approach for CT Win Probability Prediction
Compares binning-based prediction against ML models using same data split
Uses HLTV features only
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.train_win_probability_model import load_and_prepare_data
from src.ml import feature_sets
import joblib
import pickle
from functools import lru_cache

# Binning configuration for HLTV features
BIN_SIZES = {
    # Player count - exact match
    'cts_alive': 0,
    'ts_alive': 0,
    # Categorical features - exact match
    'bomb_planted': 0,
    'map_name': 0,
    "ct_avg_gear": 0,
    "t_avg_gear": 0,
}

# Global cache for binning predictions
_prediction_cache = {}


def create_cache_key(test_sample, feature_columns, bin_sizes):
    """
    Create a hashable cache key from test sample
    
    Args:
        test_sample: Single row from test set
        feature_columns: List of features to use
        bin_sizes: Dict mapping feature names to bin sizes
    
    Returns:
        Tuple that can be used as dictionary key
    """
    # Only include features that are actually used in binning
    key_parts = []
    for feature in sorted(feature_columns):
        bin_size = bin_sizes.get(feature, 0)
        if bin_size < 0:
            continue
        
        value = test_sample[feature]
        
        # For categorical or exact match, use the value directly
        if bin_size == 0:
            key_parts.append((feature, value))
        else:
            # For range matching, bin the value to reduce cache misses
            # Round to nearest bin edge
            binned_value = round(value / bin_size) * bin_size
            key_parts.append((feature, binned_value))
    
    return tuple(key_parts)


def apply_binning_mask(test_sample, train_df, feature_columns, bin_sizes):
    """
    Apply binning mask to filter training data based on test sample
    
    Args:
        test_sample: Single row from test set
        train_df: Training dataframe
        feature_columns: List of features to use for binning
        bin_sizes: Dict mapping feature names to bin sizes
    
    Returns:
        Filtered dataframe matching the binning criteria
    """
    mask = pd.Series(True, index=train_df.index)
    
    for feature in feature_columns:
        value = test_sample[feature]
        bin_size = bin_sizes.get(feature, 0)
        
        # Skip features not in bin configuration
        if bin_size < 0:
            continue
        
        # Exact match for categorical or zero bin size
        if bin_size == 0:
            mask &= train_df[feature] == value
        else:
            # Range matching for numerical features
            mask &= train_df[feature].between(value - bin_size, value + bin_size)
    
    return train_df[mask]


def predict_with_binning(test_sample, train_df, feature_columns, bin_sizes, min_samples=10, use_cache=True):
    """
    Predict CT win probability for a test sample using binning approach
    If sample size is too small, retries with map_name constraint removed
    
    Args:
        test_sample: Single row from test set
        train_df: Training dataframe with features and 'ct_wins' target
        feature_columns: List of features to use for binning
        bin_sizes: Dict mapping feature names to bin sizes
        min_samples: Minimum samples needed for prediction (otherwise return 0.5)
        use_cache: Whether to use prediction cache
    
    Returns:
        Predicted CT win probability
    """
    # Check cache first
    if use_cache:
        cache_key = create_cache_key(test_sample, feature_columns, bin_sizes)
        if cache_key in _prediction_cache:
            return _prediction_cache[cache_key]
    
    # Try with full bin configuration
    matching_samples = apply_binning_mask(test_sample, train_df, feature_columns, bin_sizes)
    n_samples = len(matching_samples)
    
    # If not enough samples and map_name was used, retry without map constraint
    if n_samples < min_samples and 'map_name' in bin_sizes and bin_sizes['map_name'] == 0:
        bin_sizes_relaxed = bin_sizes.copy()
        bin_sizes_relaxed['map_name'] = -1
        matching_samples = apply_binning_mask(test_sample, train_df, feature_columns, bin_sizes_relaxed)
        n_samples = len(matching_samples)
    
    # Return default probability if still not enough samples
    if n_samples < 3:
        probability = 0.5
    else:
        # Calculate CT win rate
        ct_wins = matching_samples['ct_wins'].sum()
        probability = ct_wins / n_samples
    
    # Store in cache
    if use_cache:
        _prediction_cache[cache_key] = probability
    
    return probability


def evaluate_binning_approach(use_cache=True):
    """Evaluate binning approach using same data split as model training"""
    
    # Check for cached results
    cache_path = project_root / "cache" / "binning_results.pkl"
    
    if use_cache and cache_path.exists():
        print("=" * 80)
        print("LOADING CACHED BINNING RESULTS")
        print("=" * 80)
        print(f"\n📂 Loading from {cache_path}")
        
        with open(cache_path, 'rb') as f:
            cached_results = pickle.load(f)
        
        print("✅ Loaded cached results!")
        
        # Print cached results
        print("\n" + "=" * 80)
        print("RESULTS (from cache)")
        print("=" * 80)
        print(f"\n{'Metric':<20} {'Value':<10}")
        print("-" * 30)
        print(f"{'Accuracy':<20} {cached_results['accuracy']:.4f}")
        print(f"{'AUC':<20} {cached_results['auc']:.4f}")
        print(f"{'Log Loss':<20} {cached_results['log_loss']:.4f}")
        
        print(f"\n{'Statistic':<30} {'Value':<10}")
        print("-" * 40)
        print(f"{'Test samples':<30} {len(cached_results['y_test'])}")
        print(f"{'CT wins (actual)':<30} {cached_results['y_test'].sum()}")
        print(f"{'CT wins (predicted)':<30} {cached_results['predictions'].sum()}")
        print(f"{'Default predictions (0.5)':<30} {(cached_results['predictions_proba'] == 0.5).sum()}")
        print(f"{'Min probability':<30} {cached_results['predictions_proba'].min():.4f}")
        print(f"{'Max probability':<30} {cached_results['predictions_proba'].max():.4f}")
        print(f"{'Mean probability':<30} {cached_results['predictions_proba'].mean():.4f}")
        
        return cached_results
    
    print("=" * 80)
    print("BINNING APPROACH EVALUATION (HLTV Features)")
    print("=" * 80)
    
    # Load data with same preprocessing as models
    print("\n📊 Loading and preparing data...")
    X, y, feature_columns, df = load_and_prepare_data(data_file=None, check_data=False)
    
    # Use HLTV feature set only
    hltv_features = feature_sets.HLTV_FEATURES
    X_subset = X[hltv_features]
    
    print(f"✅ Loaded {len(X_subset)} samples with {len(hltv_features)} features")
    print(f"   Features: {', '.join(hltv_features)}")
    
    # Split data with same random state as model training
    print("\n🔀 Splitting data (80/20 train/test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create training dataframe with target
    train_df = X_train.copy()
    train_df['ct_wins'] = y_train
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Make predictions for test set
    print("\n🔮 Making predictions using binning approach...")
    
    # Clear cache before starting
    global _prediction_cache
    _prediction_cache.clear()
    
    predictions_proba = []
    
    for idx, row in tqdm(X_test.iterrows(), total=len(X_test), desc="Predicting"):
        prob = predict_with_binning(row, train_df, hltv_features, BIN_SIZES)
        predictions_proba.append(prob)
    
    predictions_proba = np.array(predictions_proba)
    predictions = (predictions_proba > 0.5).astype(int)
    
    # Report cache statistics
    cache_hits = len(_prediction_cache)
    cache_hit_rate = cache_hits / len(X_test) * 100
    duplicate_scenarios = len(X_test) - cache_hits
    
    print(f"\n💾 Cache statistics:")
    print(f"   Unique scenarios: {cache_hits}")
    print(f"   Duplicate scenarios: {duplicate_scenarios} ({(100 - cache_hit_rate):.1f}%)")
    print(f"   Cache efficiency: {cache_hit_rate:.1f}% of test set are unique")
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions_proba)
    logloss = log_loss(y_test, predictions_proba)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<20} {accuracy:.4f}")
    print(f"{'AUC':<20} {auc:.4f}")
    print(f"{'Log Loss':<20} {logloss:.4f}")
    
    # Additional statistics
    print(f"\n{'Statistic':<30} {'Value':<10}")
    print("-" * 40)
    print(f"{'Test samples':<30} {len(y_test)}")
    print(f"{'CT wins (actual)':<30} {y_test.sum()}")
    print(f"{'CT wins (predicted)':<30} {predictions.sum()}")
    print(f"{'Default predictions (0.5)':<30} {(predictions_proba == 0.5).sum()}")
    print(f"{'Min probability':<30} {predictions_proba.min():.4f}")
    print(f"{'Max probability':<30} {predictions_proba.max():.4f}")
    print(f"{'Mean probability':<30} {predictions_proba.mean():.4f}")
    
    print("\n✅ Evaluation complete!")
    
    # Prepare results
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': logloss,
        'predictions': predictions,
        'predictions_proba': predictions_proba,
        'y_test': y_test,
        'X_test': X_test
    }
    
    # Save results to cache
    cache_path = project_root / "cache" / "binning_results.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving results to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print("✅ Results cached for future runs!")
    
    return results


def compare_with_xgboost(binning_results):
    """Compare binning approach with XGBoost model on same test set"""
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH XGBOOST MODEL")
    print("=" * 80)
    
    # Load XGBoost model
    model_path = project_root / "data" / "models" / "ct_win_probability_xgboost_hltv.pkl"
    
    if not model_path.exists():
        print("⚠️ XGBoost HLTV model not found. Please train it first.")
        return
    
    print(f"\n📂 Loading XGBoost model from {model_path.name}...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    
    # Get predictions from XGBoost
    X_test = binning_results['X_test']
    y_test = binning_results['y_test']
    
    print("🔮 Making XGBoost predictions...")
    xgb_proba = model.predict_proba(X_test)[:, 1]
    xgb_predictions = (xgb_proba > 0.5).astype(int)
    
    # Calculate XGBoost metrics
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    xgb_logloss = log_loss(y_test, xgb_proba)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("METRIC COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Binning':<12} {'XGBoost':<12} {'Difference':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {binning_results['accuracy']:.4f}      {xgb_accuracy:.4f}      {xgb_accuracy - binning_results['accuracy']:+.4f}")
    print(f"{'AUC':<20} {binning_results['auc']:.4f}      {xgb_auc:.4f}      {xgb_auc - binning_results['auc']:+.4f}")
    print(f"{'Log Loss':<20} {binning_results['log_loss']:.4f}      {xgb_logloss:.4f}      {xgb_logloss - binning_results['log_loss']:+.4f}")
    
    # Analyze disagreements
    binning_proba = binning_results['predictions_proba']
    binning_pred = binning_results['predictions']
    
    # Find where models disagree
    disagreements = binning_pred != xgb_predictions
    n_disagreements = disagreements.sum()
    
    print("\n" + "=" * 80)
    print("DISAGREEMENT ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Total disagreements: {n_disagreements} ({n_disagreements / len(y_test) * 100:.2f}%)")
    
    if n_disagreements > 0:
        # Who was right more often?
        binning_correct = binning_pred == y_test
        xgb_correct = xgb_predictions == y_test
        
        binning_right_xgb_wrong = disagreements & binning_correct
        xgb_right_binning_wrong = disagreements & xgb_correct
        both_wrong = disagreements & ~binning_correct & ~xgb_correct
        
        # Sanity check
        total_check = binning_right_xgb_wrong.sum() + xgb_right_binning_wrong.sum() + both_wrong.sum()
        if total_check != n_disagreements:
            print(f"⚠️ WARNING: Disagreement categories don't add up! {total_check} != {n_disagreements}")
        
        print(f"\n   Binning correct, XGBoost wrong: {binning_right_xgb_wrong.sum()} ({binning_right_xgb_wrong.sum() / n_disagreements * 100:.1f}% of disagreements)")
        print(f"   XGBoost correct, Binning wrong:  {xgb_right_binning_wrong.sum()} ({xgb_right_binning_wrong.sum() / n_disagreements * 100:.1f}% of disagreements)")
        print(f"   Both wrong:                      {both_wrong.sum()} ({both_wrong.sum() / n_disagreements * 100:.1f}% of disagreements)")
        
        # Debug: Check some disagreement cases
        if both_wrong.sum() == 0:
            print("\n   🔍 Verification: When models disagree...")
            sample_indices = np.where(disagreements)[0][:5]
            for idx in sample_indices:
                print(f"      Sample {idx}: True={y_test.iloc[idx]}, Binning={binning_pred[idx]}, XGBoost={xgb_predictions[idx]}")
            print("   Note: When predictions disagree (0 vs 1), exactly one must be correct!")
        
        # Analyze disagreements by scenario
        print("\n" + "=" * 80)
        print("DISAGREEMENTS BY SCENARIO")
        print("=" * 80)
        
        X_test_reset = X_test.reset_index(drop=True)
        disagreement_data = X_test_reset[disagreements].copy()
        disagreement_data['y_true'] = y_test.values[disagreements]
        disagreement_data['binning_pred'] = binning_pred[disagreements]
        disagreement_data['xgb_pred'] = xgb_predictions[disagreements]
        disagreement_data['binning_proba'] = binning_proba[disagreements]
        disagreement_data['xgb_proba'] = xgb_proba[disagreements]
        disagreement_data['binning_correct'] = binning_correct.values[disagreements]
        disagreement_data['xgb_correct'] = xgb_correct.values[disagreements]
        
        # Analyze by player count
        print("\n📊 Disagreements by Player Count:")
        for ct_alive in sorted(disagreement_data['cts_alive'].unique()):
            for t_alive in sorted(disagreement_data['ts_alive'].unique()):
                mask = (disagreement_data['cts_alive'] == ct_alive) & (disagreement_data['ts_alive'] == t_alive)
                count = mask.sum()
                if count > 0:
                    binning_win_rate = disagreement_data[mask]['binning_correct'].mean()
                    xgb_win_rate = disagreement_data[mask]['xgb_correct'].mean()
                    print(f"   {ct_alive}v{t_alive}: {count:4} disagreements | Binning: {binning_win_rate:.1%} | XGBoost: {xgb_win_rate:.1%}")
        
        # Analyze by bomb status
        print("\n💣 Disagreements by Bomb Status:")
        for bomb_planted in sorted(disagreement_data['bomb_planted'].unique()):
            mask = disagreement_data['bomb_planted'] == bomb_planted
            count = mask.sum()
            binning_win_rate = disagreement_data[mask]['binning_correct'].mean()
            xgb_win_rate = disagreement_data[mask]['xgb_correct'].mean()
            status = "Planted" if bomb_planted else "Not planted"
            print(f"   {status}: {count:4} disagreements | Binning: {binning_win_rate:.1%} | XGBoost: {xgb_win_rate:.1%}")
        
        # Analyze by confidence difference
        print("\n🎯 Disagreements by Confidence:")
        disagreement_data['confidence_diff'] = abs(disagreement_data['binning_proba'] - disagreement_data['xgb_proba'])
        
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            mask = disagreement_data['confidence_diff'] > threshold
            count = mask.sum()
            if count > 0:
                binning_win_rate = disagreement_data[mask]['binning_correct'].mean()
                xgb_win_rate = disagreement_data[mask]['xgb_correct'].mean()
                print(f"   Confidence diff > {threshold:.1f}: {count:4} cases | Binning: {binning_win_rate:.1%} | XGBoost: {xgb_win_rate:.1%}")
        
        # Show some specific examples
        print("\n" + "=" * 80)
        print("EXAMPLE DISAGREEMENTS (Top 10 by confidence difference)")
        print("=" * 80)
        
        top_disagreements = disagreement_data.nlargest(10, 'confidence_diff')
        
        # Get feature columns from test set
        feature_cols = [col for col in X_test.columns if col in top_disagreements.columns]
        
        # Print detailed table with all features
        print("\nDetailed Feature Values:")
        print("-" * 120)
        
        for idx, (_, row) in enumerate(top_disagreements.iterrows(), 1):
            print(f"\n{idx}. Disagreement #{idx}:")
            print(f"   True outcome: {'CT wins' if row['y_true'] else 'T wins'}")
            print(f"   Binning: {row['binning_proba']:.3f} → {'CT' if row['binning_pred'] else 'T'} {'✓' if row['binning_correct'] else '✗'}")
            print(f"   XGBoost: {row['xgb_proba']:.3f} → {'CT' if row['xgb_pred'] else 'T'} {'✓' if row['xgb_correct'] else '✗'}")
            print(f"   Confidence diff: {row['confidence_diff']:.3f}")
            print(f"   Features:")
            
            # Print all feature values
            for feature in feature_cols:
                value = row[feature]
                # Format based on type
                if isinstance(value, (int, np.integer)):
                    print(f"      {feature:<20}: {int(value)}")
                elif isinstance(value, (float, np.floating)):
                    print(f"      {feature:<20}: {value:.2f}")
                elif isinstance(value, bool):
                    print(f"      {feature:<20}: {value}")
                else:
                    print(f"      {feature:<20}: {value}")
            
            print()  # Blank line between examples


if __name__ == "__main__":
    # Set to False to force recalculation
    results = evaluate_binning_approach(use_cache=True)
    compare_with_xgboost(results)
