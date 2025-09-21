"""
Defines feature sets for the CS2 win probability model.
This allows for easy switching between different feature combinations for training and prediction.
Supports multiple XGBoost models with different feature sets.
"""

# Core HLTV-style features (minimal but effective)
HLTV_FEATURES = [
    'cts_alive',
    'ts_alive',
    "ct_main_weapons",
    "t_main_weapons",
    'bomb_planted',
]

# Essential game state features
MINIMAL_FEATURES = [
    'round_time_left',
    'bomb_time_left',
    'cts_alive',
    'ts_alive',
    'bomb_planted',
    'hp_t',
    'hp_ct',
    "ct_main_weapons",
    "t_main_weapons",
    "defusers",
    "ct_armor",
    "t_armor",
    "ct_helmets",
    "t_helmets",
    "ct_smokes",
    "ct_flashes",
    "ct_he_nades",
    "ct_molotovs",
    "t_smokes",
    "t_flashes",
    "t_he_nades",
    "t_molotovs",
]

# Individual player features
player_base_keys = [f'player_{i}_' for i in range(10)]

BASE_PLAYER_FEATURES = [
    'best_weapon_tier',
    'health',
    'has_defuser',
    'has_helmet',
    'armor',
    'side',
]

# Generate all player-specific feature columns
all_player_keys = []
for player_base_key in player_base_keys:
    for feature in BASE_PLAYER_FEATURES:
        all_player_keys.append(player_base_key + feature)

# Combined feature sets
PLAYER_FEATURES = MINIMAL_FEATURES + all_player_keys
ALL_FEATURES = PLAYER_FEATURES  # Currently the most comprehensive set

# XGBoost model configurations with their respective feature sets
XGBOOST_CONFIGS = {
    'xgboost_hltv': {
        'features': HLTV_FEATURES,
        'description': 'HLTV-style minimal features for fast prediction',
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    'xgboost_minimal': {
        'features': MINIMAL_FEATURES,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
    'xgboost_player_features': {
        'features': PLAYER_FEATURES,
        'description': 'Complete feature set with individual player data',
        'hyperparams': {
            'n_estimators': 400,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
        }
    },
    'xgboost_all': {
        'features': ALL_FEATURES,
        'description': 'All available features (same as player_features currently)',
        'hyperparams': {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    }
}

# Default feature set for backward compatibility
FEATURE_SET = MINIMAL_FEATURES# PLAYER_FEATURES

def get_xgboost_config(model_name):
    """Get configuration for a specific XGBoost model variant"""
    return XGBOOST_CONFIGS.get(model_name, XGBOOST_CONFIGS['xgboost_all'])

def get_available_xgboost_models():
    """Get list of available XGBoost model configurations"""
    return list(XGBOOST_CONFIGS.keys())

def get_feature_set_info():
    """Get information about all available feature sets"""
    info = {
        'hltv': {
            'features': HLTV_FEATURES,
            'count': len(HLTV_FEATURES),
            'description': 'HLTV-style minimal features'
        },
        'minimal': {
            'features': MINIMAL_FEATURES,
            'count': len(MINIMAL_FEATURES),
            'description': 'Core game state features'
        },
        'player_features': {
            'features': PLAYER_FEATURES,
            'count': len(PLAYER_FEATURES),
            'description': 'Complete feature set with individual player data'
        },
        'all_features': {
            'features': ALL_FEATURES,
            'count': len(ALL_FEATURES),
            'description': 'All available features'
        }
    }
    return info

