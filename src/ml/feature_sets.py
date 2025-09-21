"""
Defines feature sets for the CS2 win probability model.
This allows for easy switching between different feature combinations for training and prediction.
"""

HLTV_FEATURES = [
    'cts_alive',
    'ts_alive',
    "ct_main_weapons",
    "t_main_weapons",
    'bomb_planted',
]

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

player_base_keys = [f'player_{i}_' for i in range(10)]

BASE_PLAYER_FEATURES = [
    'best_weapon_tier',
    'health',
    'has_defuser',
    'has_helmet',
    'armor',
    'side',
]

all_player_keys = []
for player_base_key in player_base_keys:
    for feature in BASE_PLAYER_FEATURES:
        all_player_keys.append(player_base_key + feature)

PLAYER_FEATURES = MINIMAL_FEATURES + all_player_keys

FEATURE_SET = PLAYER_FEATURES #MINIMAL_FEATURES

