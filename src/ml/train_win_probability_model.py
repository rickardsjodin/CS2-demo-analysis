"""
CS2 Win Probability ML Model Training
Trains a model to predict CT team win probability using game state snapshots
"""

# ============================================================================
# MODEL SELECTION - Edit this to choose which models to train
# ============================================================================
TRAIN_MODELS = {
    'random_forest': False,          # Fast, good baseline
    'xgboost_hltv': True,           # HLTV-style minimal features
    'xgboost_minimal': True,        # Core game state features
    'xgboost_player_features': True, # Complete feature set with player data
    'xgboost_all': True,            # All available features
    'lightgbm': False,              # Fast, good performance (needs: pip install lightgbm)
    'logistic_regression': False,   # Very fast, interpretable
    'neural_network': False,        # Slower, can overfit
    'ensemble': False               # Combines all models (automatic if >1 model)
}

import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import polars as pl
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

from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.constants import FLASH_NADE, GRENADE_AND_BOMB_TYPES, HE_NADE, MOLOTOV_NADE, SMOKE_NADE, WEAPON_TIERS

warnings.filterwarnings('ignore')

# Handle imports for both module and script execution
try:
    from ..utils.common import get_project_root, ensure_dir
    from .feature_engineering import create_features
    from . import feature_sets
    PROJECT_ROOT = get_project_root()
except (ImportError, ModuleNotFoundError):
    # Fallback when running as script
    from pathlib import Path
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.append(str(PROJECT_ROOT))
    from src.ml.feature_engineering import create_features
    from src.ml import feature_sets
    from feature_sets import FEATURE_SET
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

def comprehensive_data_check(df):
    """Comprehensive check of data quality for ML training"""
    print("\n🔍 Comprehensive Data Quality Check:")
    print("=" * 60)
    
    # 1. Check data types
    print("\n1. Data Types Analysis:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # 2. Check for object columns that should be boolean
    print("\n2. Potential Boolean Columns (stored as object):")
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5:  # Likely categorical/boolean
                print(f"   {col}: {unique_vals}")
    
    # 3. Check for missing values
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"   {col}: {count} missing ({count/len(df)*100:.1f}%)")
    else:
        print("   ✅ No missing values found")
    
    # 4. Check for infinite values
    print("\n4. Infinite Values:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = []
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            print(f"   {col}: {inf_count} infinite values")
            inf_cols.append(col)
    
    if not inf_cols:
        print("   ✅ No infinite values found")
    
    return {
        'object_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_columns': missing.index.tolist() if len(missing) > 0 else [],
        'infinite_columns': inf_cols
    }

def load_and_prepare_data(data_file=None, feature_set=None):
    """Load snapshots and prepare features for ML training"""
    
    if data_file is None:
        data_file = PROJECT_ROOT / "data" / "datasets" / "all_snapshots.parquet"
    
    if ".json" in str(data_file):
        print("⚠️ JSON format detected. Consider converting to parquet for better performance.")
        with open(data_file, 'r') as f:
            data = json.load(f)
    else:
        # Load from parquet with proper data type handling
        data = load_snapshots_from_parquet(data_file)

    # Convert to DataFrame - data types should now be correct from source
    df = pd.DataFrame(data)
    
    print(f"✅ Loaded {len(df)} snapshots")
    print(f"📈 Data shape: {df.shape}")
    
    # Create target variable (1 if CT wins, 0 if T wins)
    df['ct_wins'] = (df['winner'] == 'ct').astype(int)
    
    # Engineer features using the centralized function
    df = create_features(df)
    
    # Use provided feature set or default
    if feature_set is None:
        try:
            feature_columns = feature_sets.ALL_FEATURES
            print(f"🧬 Using default feature set with {len(feature_columns)} features.")
        except AttributeError:
            print(f"⚠️ Feature set not found in feature_sets.py. Using minimal features.")
            feature_columns = feature_sets.MINIMAL_FEATURES
    else:
        feature_columns = feature_set
        print(f"🧬 Using custom feature set with {len(feature_columns)} features.")

    X = df[feature_columns]
    y = df['ct_wins']
    
    # Minimal validation - data should be clean from source processing
    print("🔧 Final data validation...")
    
    # Quick check for any remaining issues
    null_columns = []
    for col in X.columns:
        if X[col].isnull().any():
            null_count = X[col].isnull().sum()
            null_columns.append(f"{col}: {null_count}")
            
            # Fill remaining nulls with appropriate defaults
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(0)
            elif X[col].dtype == 'bool':
                X[col] = X[col].fillna(False)
            elif X[col].dtype == 'object':
                X[col] = X[col].fillna('unknown')
    
    if null_columns:
        print(f"   ⚠️ Fixed remaining null values in: {', '.join(null_columns)}")
    else:
        print("   ✅ No null values found - data clean from source!")
    
    print(f"🎯 Target distribution:")
    print(f"   CT wins: {y.sum()} ({y.mean():.1%})")
    print(f"   T wins:  {len(y) - y.sum()} ({1 - y.mean():.1%})")
    
    # Final data quality check
    data_issues = comprehensive_data_check(X)
    
    return X, y, feature_columns, df

def calculate_player_stats(alive_players: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculates equipment and stats for alive players at a specific tick."""
    stats = {
        "ct_main_weapons": 0, "t_main_weapons": 0,
        "ct_helmets": 0, "t_helmets": 0,
        "ct_armor": 0, "t_armor": 0,
        "defusers": 0,
        "ct_smokes": 0, "ct_flashes": 0, "ct_he_nades" : 0, "ct_molotovs": 0,
        "t_smokes": 0, "t_flashes": 0, "t_he_nades" : 0, "t_molotovs": 0
    }

    for player_row in alive_players:
        inventory = player_row.get('inventory', [])
        player_side = player_row['side']
        
        best_weapon_tier = player_row['best_weapon_tier']

        if best_weapon_tier >= 5:
            if player_side == 'ct':
                stats["ct_main_weapons"] += 1
            else:
                stats["t_main_weapons"] += 1
        
        smoke_count = sum(1 for item in inventory if item == SMOKE_NADE)
        molotov_count = sum(1 for item in inventory if item in MOLOTOV_NADE)
        flash_count = sum(1 for item in inventory if item == FLASH_NADE)
        he_count = sum(1 for item in inventory if item == HE_NADE)

        armor = player_row.get('armor', 0) or 0
        has_armor = armor > 0

        if player_side == 'ct':
            stats["ct_smokes"] += smoke_count
            stats["ct_flashes"] += flash_count
            stats["ct_he_nades"] += he_count
            stats["ct_molotovs"] += molotov_count
            stats["ct_helmets"] += 1 if player_row.get('has_helmet') else 0
            stats["ct_armor"] += has_armor
        else:
            stats["t_smokes"] += smoke_count
            stats["t_flashes"] += flash_count
            stats["t_he_nades"] += he_count
            stats["t_molotovs"] += molotov_count
            stats["t_helmets"] += 1 if player_row.get('has_helmet') else 0
            stats["t_armor"] += has_armor

        stats["defusers"] += 1 if player_row.get('has_defuser') else 0
        
    return stats

def load_snapshots_from_parquet(parquet_file: str) -> List[Dict[str, Any]]:
    """
    Load snapshots from Parquet file and convert back to original nested structure.
    This function provides compatibility with the original JSON format.
    Ensures proper data types are maintained for ML training.
    """
    
    df = pl.read_parquet(parquet_file)

    snapshots = []
    player_base_keys = [f'player_{i}_' for i in range(10)]

    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        # Convert row to mutable dict to avoid Polars read-only issues
        row_dict = dict(row)
        
        alive_player_info = []
        for player_base_key in player_base_keys:
            inventory_list = row_dict[player_base_key + 'inventory']

            # Handle missing/dead players with proper data types
            if inventory_list is None or inventory_list == '[]':
                row_dict[player_base_key + 'best_weapon_tier'] = 0
                row_dict[player_base_key + 'health'] = 0
                row_dict[player_base_key + 'has_defuser'] = False  # Ensure bool type
                row_dict[player_base_key + 'has_helmet'] = False   # Ensure bool type
                row_dict[player_base_key + 'armor'] = 0
                row_dict[player_base_key + 'side'] = -1
                continue

            inventory_list = json.loads(inventory_list)

            best_weapon_tier = 0
            for item_name in inventory_list:
                tier = WEAPON_TIERS.get(item_name)
                if tier is not None:
                    if tier > best_weapon_tier:
                        best_weapon_tier = tier

            if row_dict[player_base_key + 'has_defuser'] is None:
                pass

            row_dict[player_base_key + 'side'] = 0 if row_dict[player_base_key + 'side'] == 'ct' else 1
            row_dict[player_base_key + 'best_weapon_tier'] = best_weapon_tier
            row_dict[player_base_key + 'has_defuser'] = bool(row_dict[player_base_key + 'has_defuser'])
            row_dict[player_base_key + 'has_helmet'] = bool(row_dict[player_base_key + 'has_helmet'])

            player_info = {}
            armor = row_dict[player_base_key + 'armor']

            player_info['best_weapon_tier'] = best_weapon_tier
            player_info['has_defuser'] = row_dict[player_base_key + 'has_defuser']
            player_info['has_helmet'] = row_dict[player_base_key + 'has_helmet']
            player_info['health'] = row_dict[player_base_key + 'health']
            player_info['side'] = row_dict[player_base_key + 'side']
            player_info['armor'] = 1 if armor > 0 else 0
            
            alive_player_info.append(player_info)
        
        player_summaries = calculate_player_stats(alive_player_info) 
    
        snapshots.append({
            **row_dict,
            **player_summaries
        })
    
    return snapshots
    

def train_models():
    """Train selected models based on TRAIN_MODELS configuration"""
    
    print(f"\n🔄 Preparing to train models...")
    
    # Check which models are selected for training
    selected_models = [name for name, selected in TRAIN_MODELS.items() if selected and name != 'ensemble']
    
    print(f"\n🎯 Selected models to train: {', '.join(selected_models)}")
    if TRAIN_MODELS.get('ensemble', False):
        print("   📊 Ensemble model will be created from trained models")
    
    if not selected_models:
        print("⚠️ No models selected for training! Please set at least one model to True in TRAIN_MODELS")
        return {}, None, None
    
    # Load base dataset ONCE with full feature engineering
    print(f"\n📊 Loading and preparing base dataset...")
    # json_file = PROJECT_ROOT / "data" / "datasets" / "all_snapshots.json"
    X_full, y, all_feature_columns, base_df = load_and_prepare_data(data_file=None)
    
    print(f"✅ Base dataset loaded: {len(X_full)} samples with {len(all_feature_columns)} total features")
    
    models = {}
    
    # Group XGBoost models to train them efficiently
    xgboost_models = [name for name in selected_models if name.startswith('xgboost_')]
    other_models = [name for name in selected_models if not name.startswith('xgboost_')]
    
    if xgboost_models and XGBOOST_AVAILABLE:
        print(f"\n🚀 Training {len(xgboost_models)} XGBoost variants...")
        for model_name in xgboost_models:
            config = feature_sets.get_xgboost_config(model_name)
            features = config['features']
            hyperparams = config['hyperparams']
            
            print(f"\n  🔧 Training {model_name}:")
            print(f"     Features: {len(features)} ({config['description']})")
            print(f"     Hyperparams: {hyperparams}")
            
            # Slice the specific features for this model from the full dataset
            try:
                X_model = X_full[features]
                feature_columns = features
                print(f"     ✅ Selected {len(features)} features from base dataset")
            except KeyError as e:
                missing_features = [f for f in features if f not in X_full.columns]
                print(f"     ❌ Missing features for {model_name}: {missing_features}")
                print(f"     Available features: {list(X_full.columns)}")
                continue
            
            # Split data with consistent random state for fair comparison
            X_train, X_test, y_train, y_test = train_test_split(
                X_model, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"     Training set: {len(X_train)} samples")
            print(f"     Test set: {len(X_test)} samples")
            
            # Train XGBoost with model-specific hyperparameters
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                **hyperparams
            )
            
            xgb_model.fit(X_train, y_train)
            
            # Make predictions
            xgb_pred = xgb_model.predict(X_test)
            xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Store model results
            models[model_name] = {
                'model': xgb_model,
                'original_model': xgb_model,
                'scaler': None,
                'predictions': xgb_pred,
                'probabilities': xgb_pred_proba,
                'accuracy': accuracy_score(y_test, xgb_pred),
                'auc': roc_auc_score(y_test, xgb_pred_proba),
                'log_loss': log_loss(y_test, xgb_pred_proba),
                'feature_columns': feature_columns,
                'X_test': X_test,
                'y_test': y_test,
                'config': config
            }
            
            print(f"     ✅ {model_name} - AUC: {models[model_name]['auc']:.3f}, Acc: {models[model_name]['accuracy']:.3f}")
    
    elif xgboost_models and not XGBOOST_AVAILABLE:
        print("\n⚠️ XGBoost models selected but XGBoost not available. Install with: pip install xgboost")
    
    # Train other model types
    if other_models:
        print(f"\n🎲 Training other model types: {', '.join(other_models)}")
        
        # Use the full feature set from the base dataset for non-XGBoost models
        X_other = X_full  # Use all available features for non-XGBoost models
        feature_columns = all_feature_columns
        
        print(f"     Using {len(feature_columns)} features for other models")
        
        # Split data for non-XGBoost models (same random state for consistency)
        X_train, X_test, y_train, y_test = train_test_split(
            X_other, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (needed for some models)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest (if selected)
        if 'random_forest' in other_models:
            print("\n🌲 Training High-Complexity Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=41, 
                class_weight='balanced',
                bootstrap=False,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)
            rf_calibrated = rf_model
            
            rf_pred = rf_calibrated.predict(X_test)
            rf_pred_proba = rf_calibrated.predict_proba(X_test)[:, 1]
            
            models['random_forest'] = {
                'model': rf_calibrated,
                'original_model': rf_model,
                'scaler': None,
                'predictions': rf_pred,
                'probabilities': rf_pred_proba,
                'accuracy': accuracy_score(y_test, rf_pred),
                'auc': roc_auc_score(y_test, rf_pred_proba),
                'log_loss': log_loss(y_test, rf_pred_proba),
                'feature_columns': feature_columns,
                'X_test': X_test,
                'y_test': y_test
            }
        
        # 2. LightGBM (if available and selected)
        if 'lightgbm' in other_models:
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
                lgb_calibrated = lgb_model
                
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
                    'log_loss': log_loss(y_test, lgb_pred_proba),
                    'feature_columns': feature_columns,
                    'X_test': X_test,
                    'y_test': y_test
                }
            else:
                print("\n⚠️ LightGBM selected but not available. Install with: pip install lightgbm")
        
        # 3. Logistic Regression (if selected)
        if 'logistic_regression' in other_models:
            print("\n📊 Training Logistic Regression...")
            lr_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            lr_model.fit(X_train_scaled, y_train)
            lr_calibrated = lr_model
            
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
                'log_loss': log_loss(y_test, lr_pred_proba),
                'feature_columns': feature_columns,
                'X_test': X_test,
                'y_test': y_test
            }
        
        # 4. Neural Network (MLP) (if selected)
        if 'neural_network' in other_models:
            print("\n🧠 Training Neural Network (MLP)...")
            mlp_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=500,
                alpha=0.01
            )
            mlp_model.fit(X_train_scaled, y_train)
            mlp_calibrated = CalibratedClassifierCV(mlp_model, method='sigmoid', cv=5)
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
                'log_loss': log_loss(y_test, mlp_pred_proba),
                'feature_columns': feature_columns,
                'X_test': X_test,
                'y_test': y_test
            }
    
    # 5. Ensemble Model (if selected and multiple models are trained)
    if TRAIN_MODELS.get('ensemble', False) and len(models) > 1:
        print("\n🏆 Training Ensemble Model with all trained models...")
        
        # For ensemble, we'll use a consistent test set based on the full feature set
        # Create a test set from the full dataset for fair ensemble evaluation
        X_ensemble, _, y_ensemble, _ = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create ensemble predictions by averaging probabilities
        ensemble_proba = np.zeros(len(y_ensemble))
        ensemble_count = 0
        
        print(f"     Creating ensemble from {len(models)} models...")
        
        for name, model_info in models.items():
            try:
                # Get the features this model needs
                model_features = model_info['feature_columns']
                X_model_test = X_ensemble[model_features]
                
                if model_info['scaler'] is not None:
                    # Scale the features if needed
                    X_model_test_scaled = model_info['scaler'].transform(X_model_test)
                    model_proba = model_info['model'].predict_proba(X_model_test_scaled)[:, 1]
                else:
                    # Use unscaled features
                    model_proba = model_info['model'].predict_proba(X_model_test)[:, 1]
                
                ensemble_proba += model_proba
                ensemble_count += 1
                print(f"       ✅ Added {name} to ensemble")
                
            except Exception as e:
                print(f"       ⚠️ Could not add {name} to ensemble: {e}")
        
        if ensemble_count > 0:
            # Average the probabilities
            ensemble_proba /= ensemble_count
            ensemble_predictions = (ensemble_proba > 0.5).astype(int)
            
            # Store component model names
            component_models = list(models.keys())
            
            models['ensemble'] = {
                'model': None,  # Custom ensemble logic
                'original_model': None,
                'scaler': None,
                'predictions': ensemble_predictions,
                'probabilities': ensemble_proba,
                'accuracy': accuracy_score(y_ensemble, ensemble_predictions),
                'auc': roc_auc_score(y_ensemble, ensemble_proba),
                'log_loss': log_loss(y_ensemble, ensemble_proba),
                'component_models': component_models,
                'feature_columns': all_feature_columns,  # Use full features for reference
                'X_test': X_ensemble,
                'y_test': y_ensemble
            }
            
            print(f"     ✅ Ensemble created with {ensemble_count} models")
        else:
            print("⚠️ Could not create ensemble - no compatible models found")
    elif TRAIN_MODELS.get('ensemble', False) and len(models) <= 1:
        print("\n⚠️ Ensemble selected but only one model trained. Ensemble needs at least 2 models.")
    
    # Print comprehensive results
    if not models:
        print("\n❌ No models were successfully trained!")
        return models, None, None
    
    print(f"\n📊 Model Performance Comparison ({len(models)} models trained):")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<8} {'Log Loss':<10} {'Features':<10}")
    print("-" * 80)
    
    for name, model_info in models.items():
        feature_count = len(model_info['feature_columns']) if model_info['feature_columns'] else 0
        print(f"{name.replace('_', ' ').title():<25} "
              f"{model_info['accuracy']:.3f}    "
              f"{model_info['auc']:.3f}  "
              f"{model_info['log_loss']:.3f}    "
              f"{feature_count:<10}")
    
    # Detailed classification reports for best models
    print(f"\n📋 Detailed Classification Reports (Top 3 Models):")
    
    # Sort by AUC and show top 3
    sorted_models = sorted(models.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for name, model_info in sorted_models[:3]:
        print(f"\n{name.replace('_', ' ').title()}:")
        print(classification_report(model_info['y_test'], model_info['predictions'], 
                                  target_names=['T wins', 'CT wins']))
    
    # Return the best model's test data for further analysis
    best_model = sorted_models[0][1]
    return models, best_model['X_test'], best_model['y_test']

def analyze_feature_importance(models, default_feature_columns):
    """Analyze and visualize feature importance from multiple models"""
    
    # Collect feature importances from different models
    importance_data = {}
    
    # Process all models, including multiple XGBoost variants
    for model_name, model_info in models.items():
        if model_name == 'ensemble':
            continue  # Skip ensemble for feature importance
        
        feature_columns = model_info.get('feature_columns', default_feature_columns)
        
        if model_name.startswith('xgboost_') and XGBOOST_AVAILABLE:
            # Use original model for feature importance
            original_model = model_info['original_model']
            importance_data[model_name] = {
                'importances': original_model.feature_importances_,
                'features': feature_columns
            }
        elif model_name == 'random_forest' and 'random_forest' in models:
            # Use original model for feature importance
            original_rf = model_info['original_model']
            importance_data[model_name] = {
                'importances': original_rf.feature_importances_,
                'features': feature_columns
            }
        elif model_name == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # Use original model for feature importance
            original_lgb = model_info['original_model']
            importance_data[model_name] = {
                'importances': original_lgb.feature_importances_,
                'features': feature_columns
            }
        elif model_name == 'logistic_regression':
            # Use original model for coefficients
            original_lr = model_info['original_model']
            lr_coef = np.abs(original_lr.coef_[0])
            # Normalize to 0-1 scale for comparison
            importance_data[model_name] = {
                'importances': lr_coef / lr_coef.sum(),
                'features': feature_columns
            }
    
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
        cols = min(3, n_models)  # Max 3 columns
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
    for idx, (model_name, model_data) in enumerate(importance_data.items()):
        ax = axes[idx]
        
        importances = model_data['importances']
        features = model_data['features']
        
        # Create DataFrame for this model's features
        feature_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Take top 15 features to keep plot readable
        if len(feature_df) > 15:
            feature_df = feature_df.tail(15)
        
        bars = ax.barh(feature_df['feature'], feature_df['importance'])
        ax.set_title(f'{model_name.replace("_", " ").title()} - Feature Importance\n({len(features)} total features)')
        ax.set_xlabel('Importance')
        
        # Add value labels on bars
        for bar, value in zip(bars, feature_df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=8)
    
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
    for model_name, model_data in importance_data.items():
        importances = model_data['importances']
        features = model_data['features']
        
        print(f"\n{model_name.replace('_', ' ').title()} - Top 5 Features ({len(features)} total):")
        
        feature_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        for i, (feature, importance) in enumerate(feature_df.head(5)[['feature', 'importance']].values):
            print(f"  {i+1}. {feature}: {importance:.3f}")
    
    # Compare XGBoost models specifically
    xgb_models = {k: v for k, v in importance_data.items() if k.startswith('xgboost_')}
    if len(xgb_models) > 1:
        print(f"\n🚀 XGBoost Models Feature Comparison:")
        
        # Find common features across XGBoost models
        all_features = set()
        for model_data in xgb_models.values():
            all_features.update(model_data['features'])
        
        # Create comparison for top features
        print(f"   Total unique features across XGBoost models: {len(all_features)}")
        
        # Show how each model performs with different feature counts
        print(f"   Model complexity comparison:")
        for model_name in sorted(xgb_models.keys()):
            model_info = models[model_name]
            feature_count = len(xgb_models[model_name]['features'])
            print(f"     {model_name}: {feature_count} features, AUC={model_info['auc']:.3f}")
    
    # Calculate average importance across models (where features overlap)
    if len(importance_data) > 1:
        # Find features that appear in multiple models
        feature_counts = {}
        feature_sums = {}
        
        for model_name, model_data in importance_data.items():
            for feature, importance in zip(model_data['features'], model_data['importances']):
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                    feature_sums[feature] = 0
                feature_counts[feature] += 1
                feature_sums[feature] += importance
        
        # Calculate averages for features that appear in multiple models
        common_features = {feature: feature_sums[feature] / feature_counts[feature] 
                          for feature, count in feature_counts.items() if count > 1}
        
        if common_features:
            print(f"\n🏆 Top 5 Features (averaged across {len(importance_data)} models):")
            sorted_common = sorted(common_features.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, avg_importance) in enumerate(sorted_common[:5]):
                model_count = feature_counts[feature]
                print(f"  {i+1}. {feature}: {avg_importance:.3f} (appears in {model_count} models)")

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

def load_existing_model_summary(summary_path):
    """Load existing model summary if it exists, otherwise return empty structure"""
    import json
    
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                existing_summary = json.load(f)
            
            # Handle migration from old format with global feature_columns
            if 'feature_columns' in existing_summary and existing_summary['feature_columns']:
                global_features = existing_summary['feature_columns']
                # Add feature_columns to models that don't have them
                for model_name, model_data in existing_summary.get('models', {}).items():
                    if 'feature_columns' not in model_data:
                        model_data['feature_columns'] = global_features
                        print(f"📋 Migrated feature_columns for existing model: {model_name}")
                # Remove global feature_columns after migration
                del existing_summary['feature_columns']
            
            print(f"📋 Loaded existing model summary with {len(existing_summary.get('models', {}))} models")
            return existing_summary
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Error reading existing model summary: {e}")
            print("🔄 Will create new summary")
    else:
        print("📝 No existing model summary found, creating new one")
    
    return {
        'models': {},
        'best_model': None,
        'total_models': 0
    }

def merge_model_summaries(existing_summary, new_models, sorted_models, saved_models):
    """Merge existing model summary with newly trained models"""
    import os
    
    # Start with existing models
    merged_models = existing_summary.get('models', {}).copy()
    
    # Update or add newly trained models
    for i, (name, model_info) in enumerate(sorted_models):
        if name in saved_models:
            # Check if model file actually exists
            model_path = saved_models[name]
            if os.path.exists(model_path):
                # Use model-specific feature columns
                model_feature_columns = model_info.get('feature_columns', [])
                
                merged_models[name] = {
                    'filename': str(model_path),
                    'accuracy': new_models[name]['accuracy'],
                    'auc': new_models[name]['auc'],
                    'log_loss': new_models[name]['log_loss'],
                    'feature_columns': model_feature_columns,
                    'rank': i+1  # Will be recalculated later
                }
                
                # Add config info if this is an XGBoost model
                if 'config' in model_info:
                    merged_models[name]['config'] = model_info['config']
                
                print(f"✅ Updated model entry for '{name}' with {len(model_feature_columns)} features")
            else:
                print(f"⚠️  Model file not found for '{name}': {model_path}")
    
    # Re-rank all models by AUC (including existing ones)
    # First, try to load performance metrics for existing models that weren't retrained
    for model_name, model_data in merged_models.items():
        if model_name not in new_models:
            # This is an existing model that wasn't retrained
            # Keep its existing metrics for ranking
            print(f"📋 Preserving existing model: {model_name}")
    
    # Sort all models by AUC for proper ranking
    sorted_all_models = sorted(merged_models.items(), 
                              key=lambda x: x[1]['auc'], reverse=True)
    
    # Update ranks
    for i, (name, model_data) in enumerate(sorted_all_models):
        merged_models[name]['rank'] = i + 1
    
    # Determine best model (highest AUC)
    best_model = sorted_all_models[0][0] if sorted_all_models else None
    
    return {
        'models': merged_models,
        'best_model': best_model,
        'total_models': len(merged_models)
    }

def save_all_models(models, default_feature_columns):
    """Save all trained models individually and as a collection"""
    
    print(f"\n💾 Saving all {len(models)} trained models...")
    
    # Sort models by AUC score for ranking
    sorted_models = sorted(models.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    print(f"\n🏆 Model Performance Ranking (by AUC):")
    for i, (name, info) in enumerate(sorted_models):
        feature_count = len(info['feature_columns']) if info['feature_columns'] else 0
        print(f"  {i+1}. {name.replace('_', ' ').title()}: "
              f"AUC={info['auc']:.3f}, Acc={info['accuracy']:.3f}, LogLoss={info['log_loss']:.3f}, Features={feature_count}")
    
    # Save each model individually
    saved_models = {}
    for name, model_info in models.items():
        # Skip ensemble for individual saving since it has custom logic
        if name == 'ensemble':
            continue
            
        # Use model-specific feature columns if available
        model_feature_columns = model_info.get('feature_columns', default_feature_columns)
        
        model_data = {
            'model': model_info['model'],
            'original_model': model_info.get('original_model'),
            'scaler': model_info['scaler'],
            'feature_columns': model_feature_columns,
            'model_type': name,
            'accuracy': model_info['accuracy'],
            'auc': model_info['auc'],
            'log_loss': model_info['log_loss'],
            'is_calibrated': True
        }
        
        # Add XGBoost configuration if available
        if 'config' in model_info:
            model_data['config'] = model_info['config']
        
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
            'feature_columns': models['ensemble']['feature_columns'],
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
        best_model_feature_columns = best_model_info.get('feature_columns', default_feature_columns)
        
        best_model_data = {
            'model': best_model_info['model'],
            'original_model': best_model_info.get('original_model'),
            'scaler': best_model_info['scaler'],
            'feature_columns': best_model_feature_columns,
            'model_type': best_model_name,
            'accuracy': best_model_info['accuracy'],
            'auc': best_model_info['auc'],
            'log_loss': best_model_info['log_loss'],
            'is_calibrated': True
        }
        
        # Add XGBoost configuration if available
        if 'config' in best_model_info:
            best_model_data['config'] = best_model_info['config']
        
        # Add ensemble component info if applicable
        if 'component_models' in best_model_info:
            best_model_data['component_models'] = best_model_info['component_models']
        
        default_model_path = PROJECT_ROOT / "data" / "models" / "ct_win_probability_model.pkl"
        default_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model_data, default_model_path)
        print(f"✅ Best model ({best_model_name}) saved as default '{default_model_path}'")
    
    # Load existing model summary and merge with new models
    summary_path = PROJECT_ROOT / "data" / "models" / "model_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    existing_summary = load_existing_model_summary(summary_path)
    model_summary = merge_model_summaries(existing_summary, models, sorted_models, saved_models)
    
    import json
    with open(summary_path, 'w') as f:
        json.dump(model_summary, f, indent=2)
    print(f"✅ Model summary saved as '{summary_path}'")
    print(f"📊 Total models in summary: {model_summary['total_models']}")
    print(f"🏆 Best model: {model_summary['best_model']}")
    
    return saved_models, sorted_models[0]


def main():
    """Main training pipeline"""
    
    try:
        print("🎮 CS2 Win Probability Model Training")
        print("🎯 Support for multiple XGBoost models with different feature sets")
        print("=" * 80)
        
        # Display selected models
        selected_models = [name for name, selected in TRAIN_MODELS.items() if selected]
        print(f"📋 Selected models for training: {', '.join(selected_models)}")
        
        if not selected_models:
            print("❌ No models selected! Please configure TRAIN_MODELS in the script.")
            return
        
        # Show XGBoost configurations
        xgboost_models = [name for name in selected_models if name.startswith('xgboost_')]
        if xgboost_models:
            print(f"\n🚀 XGBoost Models Configuration:")
            for model_name in xgboost_models:
                config = feature_sets.get_xgboost_config(model_name)
                print(f"   {model_name}: {len(config['features'])} features - {config['description']}")
        
        print("=" * 80)
        
        # Train models
        models, X_test, y_test = train_models()
        
        if not models:
            print("❌ No models were successfully trained!")
            return
        
        # Use the first model's feature columns for analysis (or could be best model's)
        first_model = list(models.values())[0]
        feature_columns = first_model['feature_columns']
        
        # Analyze feature importance
        analyze_feature_importance(models, feature_columns)
        
        # Visualize calibration effectiveness (if data available)
        visualize_calibration(models, y_test)
        
        # Visualize results
        visualize_predictions(models, X_test, y_test)
        
        # Save all models
        saved_models, (best_model_name, best_model_info) = save_all_models(models, feature_columns)
        
        print(f"\n✅ Training complete! Best model: {best_model_name}")
        print(f"📊 Total models trained: {len(models)}")
        
        # Show final summary of XGBoost models
        xgb_models = {k: v for k, v in models.items() if k.startswith('xgboost_')}
        if xgb_models:
            print(f"\n🚀 XGBoost Models Summary:")
            for name, info in sorted(xgb_models.items(), key=lambda x: x[1]['auc'], reverse=True):
                feature_count = len(info['feature_columns'])
                print(f"   {name}: AUC={info['auc']:.3f}, Features={feature_count}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
