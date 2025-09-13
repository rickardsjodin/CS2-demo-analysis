"""
Defines feature sets for the CS2 win probability model.
This allows for easy switching between different feature combinations for training and prediction.
"""

# A minimal set of core features
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
    "ct_grenades",
    "t_grenades",
    "ct_helmets",
    "t_helmets",
    "ct_armor",
    "t_armor",
    "defusers",
]

# Features including player advantage and ratios
EXTENDED_FEATURES = MINIMAL_FEATURES + [
    'player_advantage',
    'ct_alive_ratio',
]

# Features including time pressure
TIME_PRESSURE_FEATURES = MINIMAL_FEATURES + [
    'time_pressure_ct',
    'time_pressure_t',
]

# All engineered features combined
ALL_FEATURES = [
    'round_time_left',
    'bomb_time_left',
    'cts_alive',
    'ts_alive',
    'bomb_planted',
    'player_advantage',
    'ct_alive_ratio',
    'time_pressure_ct',
    'time_pressure_t',
]

FEATURE_SET = MINIMAL_FEATURES

# Default feature set to be used if not specified otherwise
DEFAULT_FEATURES = EXTENDED_FEATURES
