import os
import json
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

def clean_dataset(df):
    """Clean the dataset by removing invalid entries"""
    print("🧹 Cleaning dataset...")
    
    original_count = len(df)
    
    # Identify entries with invalid player counts (should be 0-5)
    invalid_players = (df['cts_alive'] < 0) | (df['cts_alive'] > 5) | (df['ts_alive'] < 0) | (df['ts_alive'] > 5)
    
    # Identify entries with invalid time values (should be 0-115 for normal rounds)
    invalid_time = (df['time_left'] < 0) | (df['time_left'] > 115)
    
    # Identify entries with missing winner data
    invalid_winner = df['winner'].isna() | ~df['winner'].isin(['ct', 't'])
    
    # Find all invalid entries
    invalid_mask = invalid_players | invalid_time | invalid_winner
    
    if invalid_mask.any():
        # Get unique sources of invalid entries
        invalid_sources = df[invalid_mask]['source'].unique()
        print(f"🚨 Found invalid entries from {len(invalid_sources)} sources")
        
        # Remove all snapshots from sources that have any invalid entries
        df = df[~df['source'].isin(invalid_sources)]
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        
        print(f"🗑️  Removed {removed_count:,} snapshots from {len(invalid_sources)} sources with invalid data ({removed_count/original_count*100:.1f}%)")
        print(f"✅ Cleaned dataset: {cleaned_count:,} valid snapshots remaining")
    else:
        print("✅ No invalid entries found")
    
    return df


def filter_scenarios(df, time_min=None, time_max=None, bomb_planted=None):
    """Simple filter function - set parameters to None to skip that filter"""
    original_count = len(df)
    
    if time_min is not None:
        df = df[df['time_left'] >= time_min]
        print(f"🔍 Time >= {time_min}s: {len(df):,} snapshots remaining")
    
    if time_max is not None:
        df = df[df['time_left'] <= time_max]
        print(f"🔍 Time <= {time_max}s: {len(df):,} snapshots remaining")
    
    if bomb_planted is not None:
        df = df[df['bomb_planted'] == bomb_planted]
        status = "planted" if bomb_planted else "not planted"
        print(f"🔍 Bomb {status}: {len(df):,} snapshots remaining")
    
    removed = original_count - len(df)
    if removed > 0:
        print(f"📊 Filtered out {removed:,} snapshots ({removed/original_count*100:.1f}%)")
    
    return df


def load_and_prepare_data(json_file=None, range=[0, 1]):
    """Load snapshots and prepare features for ML training"""
    
    if json_file is None:
        # Construct path relative to the project root, regardless of where the script is run from
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir)) # up from src/ml
        json_file = os.path.join(project_dir, 'data', 'datasets', 'all_snapshots.json')

    print(f"📊 Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        snapshots = json.load(f)

    total_snapshots = len(snapshots)
    print(f"✅ Loaded {total_snapshots} snapshots")
    
    # Slice snapshots based on range
    start_idx = int(total_snapshots * range[0])
    end_idx = int(total_snapshots * range[1])
    snapshots = snapshots[start_idx:end_idx]
    print(f"📊 Using range [{range[0]:.1%} - {range[1]:.1%}]: {len(snapshots)} snapshots (indices {start_idx} to {end_idx})")
    
    df = pd.DataFrame(snapshots)

    # Apply scenario filters here - modify these values as needed
    # df = filter_scenarios(df, time_min=60, time_max=100, bomb_planted=False)

    clean_dataset(df)
    
    # Create target variable
    df['ct_wins'] = (df['winner'] == 'ct').astype(int)
    
    # Feature engineering
    df['round_time_left'] = df.apply(lambda row: row['time_left'] if not row['bomb_planted'] else 115, axis=1)
    df['bomb_time_left'] = df.apply(lambda row: row['time_left'] if row['bomb_planted'] else 100, axis=1)
    
    # Select features
    feature_columns = [
        'round_time_left',
        'bomb_time_left',
        'cts_alive', 
        'ts_alive',
        'bomb_planted',
    ]
    
    X = df[feature_columns]
    y = df['ct_wins']
    
    print(f"🎯 Target distribution: CT wins: {y.mean():.1%}")
    
    return X, y, feature_columns, json_file

def train_xgboost_minimal(X, y, feature_columns):
    """Train a minimal XGBoost model."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")

    print("\n🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    # Calibrate XGBoost
    print("   🎯 Calibrating XGBoost probabilities...")
    # xgb_calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, xgb_pred)
    auc = roc_auc_score(y_test, xgb_pred_proba)
    logloss = log_loss(y_test, xgb_pred_proba)
    
    print("\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   AUC:      {auc:.3f}")
    print(f"   Log Loss: {logloss:.3f}")
    
    # Save the model
    model_data = {
        'model': xgb_model,
        'scaler': None, # No scaler used in this minimal setup
        'feature_columns': feature_columns,
        'model_type': 'xgboost_minimal',
        'accuracy': accuracy,
        'auc': auc,
        'log_loss': logloss,
        'is_calibrated': False # Calibration is not performed in this script
    }
    
    # Save the model with proper path construction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_dir, 'data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, 'ct_win_probability_xgboost_minimal.pkl')
    
    joblib.dump(model_data, model_filename)
    print(f"\n💾 Saved minimal XGBoost model as '{model_filename}'")
    
    return xgb_model

def test_model_against_summary(model):
    import os
    import pandas as pd
    import numpy as np
    
    # Load baseline scenario probabilities from CSV
    baseline_path = os.path.join(os.path.dirname(__file__), "../../outputs/reports/scenario_probabilities.csv")
    baseline_df = pd.read_csv(baseline_path)
    
    print("Comparing model predictions with baseline scenario probabilities:")
    print("=" * 80)
    
    # Prepare scenarios for model input
    scenarios = []
    baseline_probs = []
    
    for _, row in baseline_df.iterrows():
        # Create feature vector for this scenario
        # We need to map the CSV columns to our model features
        time_phase = row['time_phase']
        
        # Estimate time_left based on phase (these are rough estimates)
        if time_phase == 'early':
            round_time_left = 90  # Early round, plenty of time
            bomb_time_left = 115    # No bomb planted in early rounds typically
        elif time_phase == 'middle':
            round_time_left = 45  # Mid round
            bomb_time_left = 115    # Will be set below if bomb is planted
        else:  # late
            round_time_left = 15  # Late in round
            bomb_time_left = 115    # Will be set below if bomb is planted
        
        # Adjust for bomb planted scenarios
        if row['bomb_planted']:
            bomb_time_left = round_time_left if round_time_left > 0 else 20  # Bomb timer
            round_time_left = 115  # No regular round time when bomb is planted
        
        scenario_features = {
            'round_time_left': round_time_left,
            'bomb_time_left': bomb_time_left,
            'cts_alive': row['cts_alive'],
            'ts_alive': row['ts_alive'],
            'bomb_planted': int(row['bomb_planted'])
        }
        
        scenarios.append(scenario_features)
        baseline_probs.append(row['ct_win_rate'])
    
    # Convert scenarios to DataFrame for model prediction
    scenarios_df = pd.DataFrame(scenarios)
    
    # Get model predictions for all scenarios
    model_predictions = model.predict_proba(scenarios_df)[:, 1]  # Get probability of CT win (class 1)
    
    # Calculate overall statistics
    total_score = 0.0
    absolute_errors = []
    
    print(f"{'Scenario':<25} {'Baseline':<10} {'Model':<10} {'Diff':<10} {'Sample Size'}")
    print("-" * 80)
    
    # Compare each scenario
    for i, (_, row) in enumerate(baseline_df.iterrows()):
        baseline_prob = baseline_probs[i]
        model_pred = model_predictions[i]
        diff = abs(model_pred - baseline_prob)
        
        absolute_errors.append(diff)
        total_score += diff
        
        # Create scenario description
        scenario_desc = f"{row['time_phase']} {row['cts_alive']}v{row['ts_alive']}"
        if row['bomb_planted']:
            scenario_desc += " (bomb)"
        
        print(f"{scenario_desc:<25} {baseline_prob:<10.3f} {model_pred:<10.3f} {diff:<10.3f} {row['total_samples']}")
    
    # Calculate statistics
    mean_absolute_error = np.mean(absolute_errors)
    median_absolute_error = np.median(absolute_errors)
    max_error = np.max(absolute_errors)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY STATISTICS:")
    print(f"Total scenarios evaluated: {len(baseline_df)}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")
    print(f"Median Absolute Error: {median_absolute_error:.4f}")
    print(f"Maximum Error: {max_error:.4f}")
    print(f"Total Score (sum of absolute differences): {total_score:.3f}")
    
    # Performance grading based on mean absolute error
    if mean_absolute_error < 0.05:
        grade = "Excellent"
    elif mean_absolute_error < 0.1:
        grade = "Very Good"
    elif mean_absolute_error < 0.15:
        grade = "Good"
    elif mean_absolute_error < 0.25:
        grade = "Fair"
    else:
        grade = "Needs Improvement"
    
    print(f"Performance Grade: {grade}")
    
    # Find scenarios with largest errors for analysis
    error_indices = np.argsort(absolute_errors)[-5:]  # Top 5 worst predictions
    print(f"\nTop 5 scenarios with largest prediction errors:")
    for idx in reversed(error_indices):
        row = baseline_df.iloc[idx]
        scenario_desc = f"{row['time_phase']} {row['cts_alive']}v{row['ts_alive']}"
        if row['bomb_planted']:
            scenario_desc += " (bomb)"
        print(f"  {scenario_desc}: Baseline={baseline_probs[idx]:.3f}, Model={model_predictions[idx]:.3f}, Error={absolute_errors[idx]:.3f}")
    
    return mean_absolute_error, scenarios_df, model_predictions, baseline_probs

def main():
    """Main training pipeline"""
    print("🎮 Minimal CS2 Win Probability Model Training (XGBoost)")
    print("=" * 60)
    
    # Define the path here to use in error messages
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_file_path = os.path.join(project_dir, 'data', 'datasets', 'all_snapshots.json')
    
    try:
        # Load and prepare data
        X, y, feature_columns, _ = load_and_prepare_data(data_file_path, range=[0,1])
        
        # Train and save the model
        model = train_xgboost_minimal(X, y, feature_columns)
        
        test_model_against_summary(model)
        
        print("\n✅ Minimal training complete!")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Data file not found.")
        print(f"   Please ensure '{data_file_path}' exists.")
        print("   You might need to run the data processing script first.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
