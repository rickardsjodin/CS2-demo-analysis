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
    'map_name'
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

# Random Forest model configurations with their respective feature sets
RANDOM_FOREST_CONFIGS = {
    'random_forest_hltv': {
        'features': HLTV_FEATURES,
        'description': 'HLTV-style minimal features for fast prediction',
        'hyperparams': {
            'n_estimators': 50,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.8,
            'class_weight': 'balanced',
        }
    },
    'random_forest_hltv_time': {
        'features': HLTV_WITH_TIME,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 7,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 0.8,
            'class_weight': 'balanced',
        }
    },
    'random_forest_all': {
        'features': ALL_FEATURES,
        'description': 'Full feature set including individual player data',
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 0.8,
            'class_weight': 'balanced',
        }
    },
}

# LightGBM model configurations with their respective feature sets
LIGHTGBM_CONFIGS = {
    'lightgbm_hltv': {
        'features': HLTV_FEATURES,
        'description': 'HLTV-style minimal features for fast prediction',
        'hyperparams': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 15,
            'min_child_samples': 20,
        }
    },
    'lightgbm_hltv_time': {
        'features': HLTV_WITH_TIME,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 31,
            'min_child_samples': 20,
        }
    },
    'lightgbm_all': {
        'features': ALL_FEATURES,
        'description': 'Full feature set including individual player data',
        'hyperparams': {
            'n_estimators': 200,
            'max_depth': -1,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 63,
            'min_child_samples': 20,
        }
    },
}

# Logistic Regression model configurations with their respective feature sets
LOGISTIC_REGRESSION_CONFIGS = {
    'logistic_regression_hltv': {
        'features': HLTV_FEATURES,
        'description': 'HLTV-style minimal features for fast prediction',
        'hyperparams': {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
        }
    },
    'logistic_regression_hltv_time': {
        'features': HLTV_WITH_TIME,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'C': 0.1,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced',
        }
    },
    'logistic_regression_all': {
        'features': ALL_FEATURES,
        'description': 'Full feature set including individual player data',
        'hyperparams': {
            'C': 0.01,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 2000,
            'class_weight': 'balanced',
        }
    },
}

# Neural Network model configurations with their respective feature sets
NEURAL_NETWORK_CONFIGS = {
    'neural_network_hltv': {
        'features': HLTV_FEATURES,
        'description': 'HLTV-style minimal features for fast prediction',
        'hyperparams': {
            'hidden_layer_sizes': (30,),
            'activation': 'relu',
            'alpha': 0.01,
            'learning_rate_init': 0.01,
            'max_iter': 1000,
        }
    },
    'neural_network_hltv_time': {
        'features': HLTV_WITH_TIME,
        'description': 'Core game state features without individual player data',
        'hyperparams': {
            'hidden_layer_sizes': (50,),
            'activation': 'relu',
            'alpha': 0.001,
            'learning_rate_init': 0.01,
            'max_iter': 1000,
        }
    },
    'neural_network_all': {
        'features': LVL2,
        'description': 'Full feature set including individual player data',
        'hyperparams': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'alpha': 1e-5,
            'learning_rate_init': 0.001,
            'max_iter': 1000,
        }
    },
}

# Hyperparameter search spaces for different model types and tuning intensities
HYPERPARAMETER_GRIDS = {
    'xgboost_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3]
    },
    'xgboost_thorough': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    },
    'lightgbm_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_samples': [10, 20],
        'num_leaves': [31, 63]
    },
    'lightgbm_thorough': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [5, 10, 20, 30],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'num_leaves': [15, 31, 63, 127]
    },
    'random_forest_quick': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.8],
        'bootstrap': [True, False]
    },
    'random_forest_thorough': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    },
    'logistic_regression_quick': {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000],
        'class_weight': ['balanced', None]
    },
    'logistic_regression_thorough': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [500, 1000, 2000],
        'class_weight': ['balanced', None],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # For elasticnet
    },
    'neural_network_quick': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [1000]
    },
    'neural_network_thorough': {
        'hidden_layer_sizes': [
            (50,), (100,), (200,),
            (50, 50), (100, 50), (100, 100),
            (50, 30, 20), (100, 50, 25), (200, 100, 50)
        ],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [500, 1000, 2000]
    }
}


def get_xgboost_config(model_name):
    """Get configuration for a specific XGBoost model variant"""
    return XGBOOST_CONFIGS.get(model_name, None)

def get_available_xgboost_models():
    """Get list of available XGBoost model configurations"""
    return list(XGBOOST_CONFIGS.keys())

def get_random_forest_config(model_name):
    """Get configuration for a specific Random Forest model variant"""
    return RANDOM_FOREST_CONFIGS.get(model_name, None)

def get_available_random_forest_models():
    """Get list of available Random Forest model configurations"""
    return list(RANDOM_FOREST_CONFIGS.keys())

def get_lightgbm_config(model_name):
    """Get configuration for a specific LightGBM model variant"""
    return LIGHTGBM_CONFIGS.get(model_name, None)

def get_available_lightgbm_models():
    """Get list of available LightGBM model configurations"""
    return list(LIGHTGBM_CONFIGS.keys())

def get_logistic_regression_config(model_name):
    """Get configuration for a specific Logistic Regression model variant"""
    return LOGISTIC_REGRESSION_CONFIGS.get(model_name, None)

def get_available_logistic_regression_models():
    """Get list of available Logistic Regression model configurations"""
    return list(LOGISTIC_REGRESSION_CONFIGS.keys())

def get_neural_network_config(model_name):
    """Get configuration for a specific Neural Network model variant"""
    return NEURAL_NETWORK_CONFIGS.get(model_name, None)

def get_available_neural_network_models():
    """Get list of available Neural Network model configurations"""
    return list(NEURAL_NETWORK_CONFIGS.keys())

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

def get_hyperparameter_grid(model_type, intensity='quick'):
    """
    Get hyperparameter grid for a specific model type and intensity level.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', etc.)
        intensity: 'quick' or 'thorough'
        
    Returns:
        Parameter grid dictionary or None if not found
    """
    # Map XGBoost variants to generic xgboost
    if model_type.startswith('xgboost'):
        base_model_type = 'xgboost'
    else:
        base_model_type = model_type
    
    grid_key = f"{base_model_type}_{intensity}"
    return HYPERPARAMETER_GRIDS.get(grid_key)

def get_available_tuning_models():
    """Get list of models that support hyperparameter tuning"""
    models = set()
    for key in HYPERPARAMETER_GRIDS.keys():
        model_name = key.rsplit('_', 1)[0]  # Remove '_quick' or '_thorough'
        models.add(model_name)
    return sorted(list(models))

def estimate_tuning_time(model_type, intensity='quick', search_method='random', n_iter=50):
    """
    Estimate the time required for hyperparameter tuning.
    
    Args:
        model_type: Type of model
        intensity: 'quick' or 'thorough'
        search_method: 'grid' or 'random'
        n_iter: Number of iterations for random search
        
    Returns:
        Estimated time information
    """
    param_grid = get_hyperparameter_grid(model_type, intensity)
    if not param_grid:
        return "Unknown model type"
    
    if search_method == 'grid':
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        combinations_to_test = total_combinations
    else:
        combinations_to_test = min(n_iter, 
                                 sum(len(values) for values in param_grid.values()))
    
    # Rough time estimates per combination (in seconds)
    time_per_combination = {
        'xgboost': 30,
        'xgboost_hltv': 30,
        'xgboost_hltv_time': 30,
        'lightgbm': 20,
        'random_forest': 15,
        'logistic_regression': 5,
        'neural_network': 60
    }
    
    # Handle XGBoost variants by checking if model starts with 'xgboost'
    if model_type.startswith('xgboost'):
        base_time = 30
    else:
        base_time = time_per_combination.get(model_type, 30)
    total_seconds = combinations_to_test * base_time * 5  # 5-fold CV
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    return {
        'combinations': combinations_to_test,
        'estimated_seconds': total_seconds,
        'estimated_time': f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m",
        'search_method': search_method
    }

