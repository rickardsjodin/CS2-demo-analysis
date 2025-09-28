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

HLTV_WITH_TIME = HLTV_FEATURES + [
    'round_time_left',
    'bomb_time_left',
]

LVL1 = HLTV_WITH_TIME + [
    "ct_armor",
    "t_armor",
    "ct_helmets",
    "t_helmets",
]

# Essential game state features
LVL2 = LVL1 + [
    'hp_t',
    'hp_ct',
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
PLAYER_FEATURES = LVL2 + all_player_keys
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
    'xgboost_hltv_time': {
        'features': HLTV_WITH_TIME,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    },
}


def get_xgboost_config(model_name):
    """Get configuration for a specific XGBoost model variant"""
    return XGBOOST_CONFIGS.get(model_name, None)

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
        'hltv_time': {
            'features': HLTV_WITH_TIME,
            'count': len(HLTV_WITH_TIME),
            'description': 'Core game state features'
        },
    }
    return info

