"""
CS2 Win Probability ML Model Training
Trains a model to predict CT team win probability using game state snapshots
"""

# ============================================================================
# MODEL SELECTION - Edit this to choose which models to train
# ============================================================================
TRAIN_MODELS = {
    'random_forest': True,      # Fast, good baseline
    'xgboost': True,           # High performance (needs: pip install xgboost)
    'lightgbm': True,          # Fast, good performance (needs: pip install lightgbm)
    'logistic_regression': True, # Very fast, interpretable
    'neural_network': False,    # Slower, can overfit
    'ensemble': False           # Combines all models (automatic if >1 model)
}

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Handle imports for both module and script execution
try:
    from ..utils.common import get_project_root, ensure_dir
    PROJECT_ROOT = get_project_root()
except ImportError:
    # Fallback when running as script
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    def ensure_dir(file_path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

# Try to import advanced models, fallback gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

def load_and_prepare_data(json_file=None):
    """Load snapshots and prepare features for ML training"""
    
    if json_file is None:
        json_file = PROJECT_ROOT / "data" / "datasets" / "all_snapshots3.json"
    
    print(f"📊 Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        snapshots = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(snapshots)
    
    print(f"✅ Loaded {len(df)} snapshots")
    print(f"📈 Data shape: {df.shape}")
    print(f"🏷️ Columns: {list(df.columns)}")
    
    # Create target variable (1 if CT wins, 0 if T wins)
    df['ct_wins'] = (df['winner'] == 'ct').astype(int)
    
    # Feature engineering - Fix the time_left ambiguity
    df['player_advantage'] = df['cts_alive'] - df['ts_alive']
    df['ct_alive_ratio'] = df['cts_alive'] / (df['cts_alive'] + df['ts_alive'] + 1e-8)  # Avoid division by zero
    
    # Create separate, unambiguous time features
    df['round_time_left'] = df.apply(lambda row: row['time_left'] if not row['bomb_planted'] else 0, axis=1)
    df['bomb_time_left'] = df.apply(lambda row: row['time_left'] if row['bomb_planted'] else 0, axis=1)
    
    # Create contextual time features that always have consistent meaning
    df['time_pressure_ct'] = df.apply(lambda row: 
        0 if not row['bomb_planted'] else (40.0 - row['time_left']) / 40.0, axis=1)
    df['time_pressure_t'] = df.apply(lambda row: 
        (115.0 - row['time_left']) / 115.0 if not row['bomb_planted'] else 0, axis=1)
    
    # Select features for training
    feature_columns = [
        'round_time_left',      # Time left in round (0 if bomb planted)
        'bomb_time_left',       # Time left on bomb (0 if not planted)
        # 'time_pressure_ct',     # Higher = more pressure on CT (0-1 scale)
        # 'time_pressure_t',      # Higher = more pressure on T (0-1 scale)
        'cts_alive', 
        'ts_alive',
        'bomb_planted',
        # 'player_advantage',
        # 'ct_alive_ratio',
    ]
    
    X = df[feature_columns]
    y = df['ct_wins']
    
    print(f"🎯 Target distribution:")
    print(f"   CT wins: {y.sum()} ({y.mean():.1%})")
    print(f"   T wins:  {len(y) - y.sum()} ({1 - y.mean():.1%})")
    
    return X, y, feature_columns, df

def train_models(X, y):
    """Train selected models based on TRAIN_MODELS configuration"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Training set: {len(X_train)} samples")
    print(f"🧪 Test set: {len(X_test)} samples")
    
    # Scale features (needed for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    # Check which models are selected for training
    selected_models = [name for name, selected in TRAIN_MODELS.items() if selected and name != 'ensemble']
    
    print(f"\n🎯 Selected models to train: {', '.join(selected_models)}")
    if TRAIN_MODELS.get('ensemble', False):
        print("   📊 Ensemble model will be created from trained models")
    
    if not selected_models:
        print("⚠️ No models selected for training! Please set at least one model to True in TRAIN_MODELS")
        return {}, X_test, y_test
    
    # 1. Random Forest (if selected)
    if TRAIN_MODELS.get('random_forest', False):
        print("\n🌲 Training High-Complexity Random Forest...")
        # Parameters tuned to reduce underfitting - more complex model
        rf_model = RandomForestClassifier(
            n_estimators=50,         # Many more trees for better learning
            max_depth=5,            # No depth limit - let trees grow fully
            min_samples_split=2,       # Allow very granular splits
            min_samples_leaf=1,        # Allow single-sample leaf nodes
            max_features=0.8,          # Use 80% of features for better pattern capture
            random_state=41, 
            class_weight='balanced',   # Handle class imbalance
            bootstrap=False,            # Enable bootstrap sampling
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)
        
        # CALIBRATION ANALYSIS - Compare before and after calibration
        print("   📊 Analyzing calibration effectiveness...")
        
        
        # Calibrate Random Forest
        print("   🎯 Calibrating Random Forest probabilities...")
        rf_calibrated = rf_model # CalibratedClassifierCV(rf_model, method='isotonic', cv=5)
        rf_calibrated.fit(X_train, y_train)
        
        # Get calibrated probabilities
        rf_pred = rf_calibrated.predict(X_test)
        rf_pred_proba = rf_calibrated.predict_proba(X_test)[:, 1]
        
        # Calibration quality metrics
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curves for both versions
        fraction_pos_cal, mean_pred_cal = calibration_curve(y_test, rf_pred_proba, n_bins=10)
        
        # Calculate Brier Score (lower is better)
        from sklearn.metrics import brier_score_loss
        brier_calibrated = brier_score_loss(y_test, rf_pred_proba)
        
        # Calculate reliability (how close predicted probabilities are to actual frequencies)
        def calculate_reliability(y_true, y_prob, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            reliability = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    reliability += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return reliability
        
        reliability_cal = calculate_reliability(y_test, rf_pred_proba)
        
        models['random_forest'] = {
            'model': rf_calibrated,
            'original_model': rf_model,
            'scaler': None,
            'predictions': rf_pred,
            'probabilities': rf_pred_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'auc': roc_auc_score(y_test, rf_pred_proba),
            'log_loss': log_loss(y_test, rf_pred_proba),
            'brier_score_calibrated': brier_calibrated,
            'reliability_calibrated': reliability_cal
        }    # 2. XGBoost (if available and selected)
    if TRAIN_MODELS.get('xgboost', False):
        if XGBOOST_AVAILABLE:
            print("\n🚀 Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            xgb_model.fit(X_train, y_train)
            # Calibrate XGBoost
            print("   🎯 Calibrating XGBoost probabilities...")
            xgb_calibrated = xgb_model# CalibratedClassifierCV(xgb_model, method='sigmoid', cv=5)
            xgb_calibrated.fit(X_train, y_train)
            
            xgb_pred = xgb_calibrated.predict(X_test)
            xgb_pred_proba = xgb_calibrated.predict_proba(X_test)[:, 1]
            
            models['xgboost'] = {
                'model': xgb_calibrated,
                'original_model': xgb_model,
                'scaler': None,
                'predictions': xgb_pred,
                'probabilities': xgb_pred_proba,
                'accuracy': accuracy_score(y_test, xgb_pred),
                'auc': roc_auc_score(y_test, xgb_pred_proba),
                'log_loss': log_loss(y_test, xgb_pred_proba)
            }
        else:
            print("\n⚠️ XGBoost selected but not available. Install with: pip install xgboost")

    # 3. LightGBM (if available and selected)
    if TRAIN_MODELS.get('lightgbm', False):
        if LIGHTGBM_AVAILABLE:
            print("\n⚡ Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )

            lgb_model.fit(X_train, y_train)
            
            # Calibrate LightGBM
            print("   🎯 Calibrating LightGBM probabilities...")
            lgb_calibrated = lgb_model# CalibratedClassifierCV(lgb_model, method='sigmoid', cv=5)
            lgb_calibrated.fit(X_train, y_train)
            
            lgb_pred = lgb_calibrated.predict(X_test)
            lgb_pred_proba = lgb_calibrated.predict_proba(X_test)[:, 1]
            
            models['lightgbm'] = {
                'model': lgb_calibrated,
                'original_model': lgb_model,
                'scaler': None,
                'predictions': lgb_pred,
                'probabilities': lgb_pred_proba,
                'accuracy': accuracy_score(y_test, lgb_pred),
                'auc': roc_auc_score(y_test, lgb_pred_proba),
                'log_loss': log_loss(y_test, lgb_pred_proba)
            }
        else:
            print("\n⚠️ LightGBM selected but not available. Install with: pip install lightgbm")
    
    # 4. Logistic Regression (if selected)
    if TRAIN_MODELS.get('logistic_regression', False):
        print("\n📊 Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train_scaled, y_train)
        
        # Calibrate Logistic Regression
        print("   🎯 Calibrating Logistic Regression probabilities...")
        lr_calibrated = lr_model# CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
        lr_calibrated.fit(X_train_scaled, y_train)
        
        lr_pred = lr_calibrated.predict(X_test_scaled)
        lr_pred_proba = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
        
        models['logistic_regression'] = {
            'model': lr_calibrated,
            'original_model': lr_model,
            'scaler': scaler,
            'predictions': lr_pred,
            'probabilities': lr_pred_proba,
            'accuracy': accuracy_score(y_test, lr_pred),
            'auc': roc_auc_score(y_test, lr_pred_proba),
            'log_loss': log_loss(y_test, lr_pred_proba)
        }
    
    # 5. Neural Network (MLP) (if selected)
    if TRAIN_MODELS.get('neural_network', False):
        print("\n🧠 Training Neural Network (MLP)...")
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=500,
            alpha=0.01
        )
        mlp_model.fit(X_train_scaled, y_train)
        
        # Calibrate Neural Network
        print("   🎯 Calibrating Neural Network probabilities...")
        mlp_calibrated =  CalibratedClassifierCV(mlp_model, method='sigmoid', cv=5)
        mlp_calibrated.fit(X_train_scaled, y_train)
        
        mlp_pred = mlp_calibrated.predict(X_test_scaled)
        mlp_pred_proba = mlp_calibrated.predict_proba(X_test_scaled)[:, 1]
        
        models['neural_network'] = {
            'model': mlp_calibrated,
            'original_model': mlp_model,
            'scaler': scaler,
            'predictions': mlp_pred,
            'probabilities': mlp_pred_proba,
            'accuracy': accuracy_score(y_test, mlp_pred),
            'auc': roc_auc_score(y_test, mlp_pred_proba),
            'log_loss': log_loss(y_test, mlp_pred_proba)
        }
    
    # 6. Ensemble Model (if selected and multiple models are trained)
    if TRAIN_MODELS.get('ensemble', False) and len(models) > 1:
        print("\n🏆 Training Ensemble Model with all trained models...")
        
        # Create ensemble predictions by averaging probabilities
        ensemble_proba = np.zeros(len(y_test))
        
        for name, model_info in models.items():
            if model_info['scaler'] is not None:
                # Use scaled features
                model_proba = model_info['model'].predict_proba(X_test_scaled)[:, 1]
            else:
                # Use unscaled features
                model_proba = model_info['model'].predict_proba(X_test)[:, 1]
            ensemble_proba += model_proba
        
        # Average the probabilities
        ensemble_proba /= len(models)
        ensemble_predictions = (ensemble_proba > 0.5).astype(int)
        
        # Store component model names (exclude ensemble itself)
        component_models = list(models.keys())
        
        models['ensemble'] = {
            'model': None,  # Custom ensemble logic
            'original_model': None,
            'scaler': None,
            'predictions': ensemble_predictions,
            'probabilities': ensemble_proba,
            'accuracy': accuracy_score(y_test, ensemble_predictions),
            'auc': roc_auc_score(y_test, ensemble_proba),
            'log_loss': log_loss(y_test, ensemble_proba),
            'component_models': component_models
        }
    elif TRAIN_MODELS.get('ensemble', False) and len(models) <= 1:
        print("\n⚠️ Ensemble selected but only one model trained. Ensemble needs at least 2 models.")
    
    # Print comprehensive results
    if not models:
        print("\n❌ No models were successfully trained!")
        return models, X_test, y_test
    
    print(f"\n📊 Model Performance Comparison ({len(models)} models trained):")
    print("   All models calibrated with CalibratedClassifierCV for better probability estimates")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<10} {'AUC':<8} {'Log Loss':<10}")
    print("-" * 80)
    
    for name, model_info in models.items():
        print(f"{name.replace('_', ' ').title():<20} "
              f"{model_info['accuracy']:.3f}    "
              f"{model_info['auc']:.3f}  "
              f"{model_info['log_loss']:.3f}")
    
    # Detailed classification reports
    print(f"\n📋 Detailed Classification Reports:")
    
    for name, model_info in models.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(classification_report(y_test, model_info['predictions'], 
                                  target_names=['T wins', 'CT wins']))
    
    return models, X_test, y_test

def analyze_feature_importance(models, feature_columns):
    """Analyze and visualize feature importance from multiple models"""
    
    # Collect feature importances from different models
    importance_data = {}
    
    # Random Forest
    if 'random_forest' in models:
        # Use original model for feature importance since calibrated wrapper doesn't have this attribute
        original_rf = models['random_forest']['original_model']
        importance_data['Random Forest'] = original_rf.feature_importances_
    
    # XGBoost
    if 'xgboost' in models and XGBOOST_AVAILABLE:
        # Use original model for feature importance
        original_xgb = models['xgboost']['original_model']
        importance_data['XGBoost'] = original_xgb.feature_importances_
    
    # LightGBM
    if 'lightgbm' in models and LIGHTGBM_AVAILABLE:
        # Use original model for feature importance
        original_lgb = models['lightgbm']['original_model']
        importance_data['LightGBM'] = original_lgb.feature_importances_
    
    # Logistic Regression (use absolute coefficients)
    if 'logistic_regression' in models:
        # Use original model for coefficients
        original_lr = models['logistic_regression']['original_model']
        lr_coef = np.abs(original_lr.coef_[0])
        # Normalize to 0-1 scale for comparison
        importance_data['Logistic Regression'] = lr_coef / lr_coef.sum()
    
    # Neural Network - doesn't have easily interpretable feature importance
    # We could implement permutation importance but that's computationally expensive
    
    # Create visualization
    n_models = len(importance_data)
    
    if n_models == 0:
        print("No models available for feature importance analysis.")
        return
    
    # Calculate subplot layout
    if n_models == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axes = [ax]
    else:
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        
        # Ensure axes is always a list
        if n_models == 2 and rows == 1:
            axes = list(axes)
        elif rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
    
    # Plot feature importance for each model
    for idx, (model_name, importances) in enumerate(importance_data.items()):
        ax = axes[idx]
        
        feature_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        bars = ax.barh(feature_df['feature'], feature_df['importance'])
        ax.set_title(f'{model_name} - Feature Importance')
        ax.set_xlabel('Importance')
        
        # Add value labels on bars
        for bar, value in zip(bars, feature_df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    # Hide empty subplots if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "visualizations" / "feature_importance.png"
    ensure_dir(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Feature importance plot saved as '{output_path}'")
    
    # Print feature importance summary
    print("\n🎯 Feature Importance Summary:")
    for model_name, importances in importance_data.items():
        print(f"\n{model_name} - Top 3 Features:")
        feature_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for i, (feature, importance) in enumerate(feature_df.head(3)[['feature', 'importance']].values):
            print(f"  {i+1}. {feature}: {importance:.3f}")
    
    # Calculate average importance across models (where available)
    if len(importance_data) > 1:
        avg_importance = np.mean([imp for imp in importance_data.values()], axis=0)
        feature_df = pd.DataFrame({
            'feature': feature_columns,
            'avg_importance': avg_importance
        }).sort_values('avg_importance', ascending=False)
        
        print(f"\n🏆 Overall Top 3 Features (averaged across models):")
        for i, (feature, importance) in enumerate(feature_df.head(3)[['feature', 'avg_importance']].values):
            print(f"  {i+1}. {feature}: {importance:.3f}")

def visualize_calibration(models, y_test):
    """Create calibration plots to show how well probabilities are calibrated"""
    
    # Check if we have calibration data
    models_with_calibration = {name: info for name, info in models.items() 
                              if 'uncalibrated_probabilities' in info}
    
    if not models_with_calibration:
        print("No calibration data available for visualization")
        return
    
    n_models = len(models_with_calibration)
    
    # Create subplots for calibration curves
    fig, axes = plt.subplots(1, min(n_models, 3), figsize=(5 * min(n_models, 3), 5))
    
    if n_models == 1:
        axes = [axes]
    elif n_models > 3:
        # Only show first 3 models to keep plot readable
        models_with_calibration = dict(list(models_with_calibration.items())[:3])
        n_models = 3
    
    from sklearn.calibration import calibration_curve
    
    for i, (name, model_info) in enumerate(models_with_calibration.items()):
        ax = axes[i] if n_models > 1 else axes[0]
        
        # Get both calibrated and uncalibrated probabilities
        y_prob_uncal = model_info['uncalibrated_probabilities']
        y_prob_cal = model_info['probabilities']
        
        # Calculate calibration curves
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(y_test, y_prob_uncal, n_bins=10)
        fraction_pos_cal, mean_pred_cal = calibration_curve(y_test, y_prob_cal, n_bins=10)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Plot uncalibrated and calibrated curves
        ax.plot(mean_pred_uncal, fraction_pos_uncal, 'o-', 
                label=f'Uncalibrated (Brier: {model_info["brier_score_uncalibrated"]:.3f})', 
                color='red', alpha=0.7)
        ax.plot(mean_pred_cal, fraction_pos_cal, 's-', 
                label=f'Calibrated (Brier: {model_info["brier_score_calibrated"]:.3f})', 
                color='blue')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{name.replace("_", " ").title()} - Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "visualizations" / "calibration_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Calibration curves saved as '{output_path}'")
    
    # Create histogram of probability distributions
    fig, axes = plt.subplots(2, min(n_models, 3), figsize=(5 * min(n_models, 3), 8))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (name, model_info) in enumerate(models_with_calibration.items()):
        col = i if n_models > 1 else 0
        
        # Uncalibrated probabilities histogram
        axes[0, col].hist(model_info['uncalibrated_probabilities'], bins=20, alpha=0.7, 
                         color='red', density=True, label='Uncalibrated')
        axes[0, col].set_title(f'{name.replace("_", " ").title()} - Uncalibrated Probabilities')
        axes[0, col].set_xlabel('Predicted Probability')
        axes[0, col].set_ylabel('Density')
        axes[0, col].set_xlim([0, 1])
        
        # Calibrated probabilities histogram
        axes[1, col].hist(model_info['probabilities'], bins=20, alpha=0.7, 
                         color='blue', density=True, label='Calibrated')
        axes[1, col].set_title(f'{name.replace("_", " ").title()} - Calibrated Probabilities')
        axes[1, col].set_xlabel('Predicted Probability')
        axes[1, col].set_ylabel('Density')
        axes[1, col].set_xlim([0, 1])
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "visualizations" / "probability_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Probability distributions saved as '{output_path}'")

def visualize_predictions(models, X_test, y_test):
    """Create visualizations of model predictions"""
    
    n_models = len(models)
    
    # Calculate optimal subplot layout
    if n_models == 1:
        rows, cols = 1, 1
    elif n_models == 2:
        rows, cols = 1, 2
    elif n_models <= 4:
        rows, cols = 2, 2
    elif n_models <= 6:
        rows, cols = 2, 3
    elif n_models <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3  # Maximum layout for many models
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_models == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    model_items = list(models.items())
    
    for i, (name, model_info) in enumerate(model_items):
        if i >= rows * cols:  # Skip if we have too many models
            break
            
        row = i // cols
        col = i % cols
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, model_info['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['T wins', 'CT wins'], 
                   yticklabels=['T wins', 'CT wins'], ax=axes[row, col])
        axes[row, col].set_title(f'{name.replace("_", " ").title()} - Confusion Matrix')
        axes[row, col].set_ylabel('True Label')
        axes[row, col].set_xlabel('Predicted Label')
    
    # Hide unused subplots
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "visualizations" / "model_performance.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"📊 Model performance plot saved as '{output_path}'")

def save_all_models(models, feature_columns):
    """Save all trained models individually and as a collection"""
    
    print(f"\n💾 Saving all {len(models)} trained models...")
    
    # Sort models by AUC score for ranking
    sorted_models = sorted(models.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    print(f"\n� Model Performance Ranking (by AUC):")
    for i, (name, info) in enumerate(sorted_models):
        print(f"  {i+1}. {name.replace('_', ' ').title()}: "
              f"AUC={info['auc']:.3f}, Acc={info['accuracy']:.3f}, LogLoss={info['log_loss']:.3f}")
    
    # Save each model individually
    saved_models = {}
    for name, model_info in models.items():
        # Skip ensemble for individual saving since it has custom logic
        if name == 'ensemble':
            continue
            
        model_data = {
            'model': model_info['model'],
            'original_model': model_info.get('original_model'),
            'scaler': model_info['scaler'],
            'feature_columns': feature_columns,
            'model_type': name,
            'accuracy': model_info['accuracy'],
            'auc': model_info['auc'],
            'log_loss': model_info['log_loss'],
            'is_calibrated': True
        }
        
        # Save individual model
        model_filename = PROJECT_ROOT / "data" / "models" / f"ct_win_probability_{name}.pkl"
        model_filename.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_filename)
        print(f"✅ Saved {name} model as '{model_filename}'")
        saved_models[name] = model_filename
    
    # Save ensemble model separately if it exists
    if 'ensemble' in models:
        ensemble_data = {
            'component_models': saved_models,  # Reference to individual model files
            'feature_columns': feature_columns,
            'model_type': 'ensemble',
            'accuracy': models['ensemble']['accuracy'],
            'auc': models['ensemble']['auc'],
            'log_loss': models['ensemble']['log_loss'],
            'component_model_names': models['ensemble']['component_models']
        }
        
        ensemble_filename = PROJECT_ROOT / "data" / "models" / "ct_win_probability_ensemble.pkl"
        ensemble_filename.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ensemble_data, ensemble_filename)
        print(f"✅ Saved ensemble model as '{ensemble_filename}'")
        saved_models['ensemble'] = ensemble_filename
    
    # Save the best model as the default
    best_model_name, best_model_info = sorted_models[0]
    if best_model_name != 'ensemble':
        best_model_data = {
            'model': best_model_info['model'],
            'original_model': best_model_info.get('original_model'),
            'scaler': best_model_info['scaler'],
            'feature_columns': feature_columns,
            'model_type': best_model_name,
            'accuracy': best_model_info['accuracy'],
            'auc': best_model_info['auc'],
            'log_loss': best_model_info['log_loss'],
            'is_calibrated': True
        }
        
        # Add ensemble component info if applicable
        if 'component_models' in best_model_info:
            best_model_data['component_models'] = best_model_info['component_models']
        
        default_model_path = PROJECT_ROOT / "data" / "models" / "ct_win_probability_model.pkl"
        default_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model_data, default_model_path)
        print(f"✅ Best model ({best_model_name}) saved as default '{default_model_path}'")
    
    # Save model summary
    model_summary = {
        'models': {name: {
            'filename': str(saved_models[name]),  # Convert Path to string for JSON serialization
            'accuracy': models[name]['accuracy'],
            'auc': models[name]['auc'],
            'log_loss': models[name]['log_loss'],
            'rank': i+1
        } for i, (name, _) in enumerate(sorted_models) if name in saved_models},
        'best_model': sorted_models[0][0],
        'feature_columns': feature_columns,
        'total_models': len(models)
    }
    
    import json
    summary_path = PROJECT_ROOT / "data" / "models" / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(model_summary, f, indent=2)
    print(f"✅ Model summary saved as '{summary_path}'")
    
    return saved_models, sorted_models[0]

def predict_win_probability(time_left, cts_alive, ts_alive, bomb_planted, model_file=None):
    """Use trained model to predict CT win probability for a given game state"""
    
    if model_file is None:
        model_file = PROJECT_ROOT / "data" / "models" / "ct_win_probability_model.pkl"
    
    # Load model
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Create feature vector with unambiguous time features
    player_advantage = cts_alive - ts_alive
    ct_alive_ratio = cts_alive / (cts_alive + ts_alive + 1e-8)
    
    # Separate time features - no more ambiguity!
    round_time_left = time_left if not bomb_planted else 0
    bomb_time_left = time_left if bomb_planted else 0
    
    # Contextual time pressure features (0-1 scale, consistent meaning)
    if not bomb_planted:
        time_pressure_ct = 0#time_left / 115.0  # Higher = more time for CT
        time_pressure_t = (115.0 - time_left) / 115.0  # Higher = more pressure on T
    else:
        time_pressure_ct = (40.0 - time_left) / 40.0  # Higher = bomb about to explode (bad for CT)
        time_pressure_t = 0#time_left / 40.0  # Higher = more time for T to defend bomb
    
    # Create DataFrame with proper feature names to avoid sklearn warning
    feature_data = {
        'round_time_left': round_time_left,
        'bomb_time_left': bomb_time_left,
        # 'time_pressure_ct': time_pressure_ct,
        # 'time_pressure_t': time_pressure_t,
        'cts_alive': cts_alive, 
        'ts_alive': ts_alive, 
        'bomb_planted': bomb_planted,
        # 'player_advantage': player_advantage, 
        # 'ct_alive_ratio': ct_alive_ratio, 
    }
    
    X = pd.DataFrame([feature_data], columns=feature_columns)
    
    # Scale if needed
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict probability
    ct_win_prob = model.predict_proba(X)[0, 1]
    
    return ct_win_prob

def main():
    """Main training pipeline"""
    
    try:
        print("🎮 CS2 Win Probability Model Training")
        print("🎯 All models will be calibrated with CalibratedClassifierCV for better probability estimates")
        print("=" * 80)
        
        # Display selected models
        selected_models = [name for name, selected in TRAIN_MODELS.items() if selected]
        print(f"📋 Selected models for training: {', '.join(selected_models)}")
        
        if not selected_models:
            print("❌ No models selected! Please configure MODELS_TO_TRAIN in the script.")
            return
        
        print("=" * 80)
        
        # Load and prepare data
        X, y, feature_columns, df = load_and_prepare_data()
        
        # Train models
        models, X_test, y_test = train_models(X, y)
        
        if not models:
            print("❌ No models were successfully trained!")
            return
        
        # Analyze feature importance
        analyze_feature_importance(models, feature_columns)
        
        # Visualize calibration effectiveness
        visualize_calibration(models, y_test)
        
        # Visualize results
        visualize_predictions(models, X_test, y_test)
        
        # Save all models
        saved_models, (best_model_name, best_model_info) = save_all_models(models, feature_columns)
        
        # Test prediction function
        print("\n🧪 Testing prediction function:")
        test_scenarios = [
            (60, 5, 5, False, "Equal teams, mid-round"),
            (30, 3, 5, False, "CT disadvantage"),
            (20, 5, 2, True, "CT advantage with bomb planted"),
            (5, 1, 1, True, "1v1 clutch with bomb planted"),
            (45, 4, 4, False, "Equal teams, early round"),
            (10, 2, 3, True, "CT slight disadvantage, bomb planted"),
            (90, 5, 3, False, "CT advantage, plenty of time"),
            (15, 1, 3, False, "CT major disadvantage, no bomb"),
            (3, 2, 1, True, "CT advantage, very low time, bomb planted"),
            (75, 5, 1, False, "CT major advantage, early round"),
            (25, 3, 3, True, "Equal teams, bomb planted, mid-time"),
            (8, 1, 2, False, "1v2 clutch for CT, no bomb"),
            (35, 4, 5, False, "T slight advantage, mid-round"),
            (12, 3, 1, True, "CT advantage, bomb planted, low time"),
            (50, 2, 4, False, "T major advantage, no bomb")
        ]
        
        for time_left, cts, ts, bomb, description in test_scenarios:
            prob = predict_win_probability(time_left, cts, ts, bomb)
            print(f"  {description}: {prob:.1%} CT win probability")
        
        print(f"\n✅ Training complete! Best model: {best_model_name}")
        print(f"📊 Total models trained: {len(models)}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
