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

# Create directories if they don't exist
for directory in [DATA_DIR, DEMOS_DIR, OUTPUTS_DIR, CACHE_DIR, MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)
