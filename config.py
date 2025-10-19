"""
Configuration settings for CS2 demo analysis
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DEMOS_DIR = PROJECT_ROOT / "demos"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Default demo file (update this to your demo file)
DEFAULT_DEMO_FILE = DEMOS_DIR / "vitality-vs-gamer-legion-m1-train.dem"

# Analysis settings
DEFAULT_PLAYER = "REZ"  # Change this to analyze a different player by default
USE_CACHE = True
CLEAR_CACHE_ON_START = False

# Model paths
MODELS_DIR = DATA_DIR / "models"
CT_WIN_MODEL = MODELS_DIR / "ct_win_probability_model.pkl"
IMPROVED_CT_MODEL = MODELS_DIR / "improved_ct_win_probability_model.pkl"
SIMPLE_CT_MODEL = MODELS_DIR / "simple_ct_win_model.pkl"

# Output settings
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Data processing settings
TICK_RATE = 64
ROUND_TIME = 115
BOMB_TIME = 40

# CS2 Map names - centralized for consistent categorical encoding
# Order matters! This must be used consistently during training and prediction
CS2_MAPS = [
    'de_dust2', 
    'de_nuke', 
    'de_overpass', 
    'de_inferno',
    'de_train', 
    'de_mirage', 
    'de_ancient', 
    'de_anubis', 
    'de_vertigo'
]

# Gear categories - centralized for consistent categorical encoding
# Order matters! This must be used consistently during training and prediction
GEAR_CATEGORY_NAMES = [
    'starter_pistol',    # 0: $0-800
    'upgraded_pistol',   # 1: $800-1500
    'smg_shotgun',       # 2: $1500-2700
    'tier2_rifle',       # 3: $2700-3500
    'tier1_rifle',       # 4: $3500-4500
    'sniper'             # 5: $4500+
]


# Create directories if they don't exist
for directory in [DATA_DIR, DEMOS_DIR, OUTPUTS_DIR, CACHE_DIR, MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)
