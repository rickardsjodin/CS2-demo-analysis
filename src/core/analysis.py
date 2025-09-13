"""
Core analysis functions for CS2 demo processing.
"""

import pandas as pd
import polars as pl

from tqdm import tqdm

from config import BOMB_TIME, ROUND_TIME
from src.ml.test_win_probability_scenarios import load_all_trained_models
from src.ml.feature_engineering import create_features
from .snapshot_extractor import create_snapshot 

def predict(model_data, snapshot_data):
    df = pd.DataFrame([snapshot_data])
    try:
        df = create_features(df)
    except:
        print()
    
    # Ensure all required feature columns are present
    feature_columns = model_data['feature_columns']
    X = df[feature_columns]
    
    # Predict probability - handle both calibrated and original models
    model = model_data['model']
    ct_win_prob = model.predict_proba(X)[0, 1]
    return ct_win_prob



def get_player_kill_death_analysis(dem_file, player_name, debug=False):
    """
    Analyze kills and deaths for a specific player with impact scoring.
    
    Args:
        dem: Parsed demo object
        player_name: Name of player to analyze
        debug: Enable debug output
    
    Returns:
        DataFrame with round-by-round analysis
    """
    all_models = load_all_trained_models()
    pred_model = all_models['xgboost']['data']

    dem = {}
    for name in ['kills', 'rounds', 'ticks']:
        df = getattr(dem_file, name, None)
        if df is None:
            tqdm.write(f"Warning: Missing '{name}' in dem")
            return None
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        dem[name] = df
    results = []
    for round_row in dem['rounds'].iter_rows(named=True):
        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        plant_tick = round_row['bomb_plant']

        if freeze_end is None and round_num == 1:
            round_row['freeze_end'] = 200
            freeze_end = 200
        
        if freeze_end is None or round_row['end'] is None:
            tqdm.write(f"Skipping round {round_num} due to missing start/end ticks.")
            continue

        round_kills = dem['kills'].filter(pl.col('round_num') == round_num)

        relevant_kills = round_kills.filter(
            (pl.col("attacker_name") == player_name) | (pl.col("victim_name") == player_name)
        )
        
        snapshot_ticks = sorted(set(relevant_kills['tick'].to_list()))
        round_impact = 0.0

        for current_tick in snapshot_ticks:
            snapshot_before = create_snapshot(current_tick - 50, round_row, dem, 64, player_name)
            snapshot_after = create_snapshot(current_tick, round_row, dem, 64, player_name)

            ct_win_before = predict(pred_model, snapshot_before)
            ct_win_after = predict(pred_model, snapshot_after)

            kill_row = relevant_kills.filter(pl.col("tick") == current_tick)
            was_ct = 'ct' in kill_row['attacker_side'].to_list()
            was_kill = player_name in kill_row['attacker_name'].to_list()

            impact = (ct_win_after - ct_win_before) * 100

            round_impact += abs(impact) if was_kill else -abs(impact)
        
            # Summarize kills/deaths for this round
            deaths_so_far = round_kills.filter(pl.col('tick') < current_tick)
            
            ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
            t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height

            if plant_tick is not None and current_tick >= plant_tick:
                bomb_planted = True
            else:
                bomb_planted = False

            results.append({
                'round': int(round_num),
                'side': 'ct' if was_ct else 't',
                'kills': '',
                'deaths': '',
                'impact': float(round(round_impact, 1)),
                'event_id': 0,
                'event_round': round_num,
                'event_type': 0,
                'event_impact': impact,
                'game_state': f'{5 -ct_deaths}v{5 -t_deaths}',
                'weapon': '',
                'victim': '',
                'post_plant': bomb_planted,
                'tick': current_tick 
            })
    
    return results
