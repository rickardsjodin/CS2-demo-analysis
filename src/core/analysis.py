"""
Core analysis functions for CS2 demo processing.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional
import pandas as pd
import polars as pl

from tqdm import tqdm


from config import BOMB_TIME, ROUND_TIME
from src.core.constants import FLASH_NADE, GRENADE_AND_BOMB_TYPES, HE_NADE, MOLOTOV_NADE, SMOKE_NADE, WEAPON_TIERS
from src.ml.feature_engineering import create_features



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

def calculate_player_stats(alive_players: pl.DataFrame) -> Dict[str, int]:
    """Calculates equipment and stats for alive players at a specific tick."""
    stats = {
        "ct_main_weapons": 0, "t_main_weapons": 0,
        "ct_helmets": 0, "t_helmets": 0,
        "ct_armor": 0, "t_armor": 0,
        "defusers": 0,
        "ct_smokes": 0, "ct_flashes": 0, "ct_he_nades" : 0, "ct_molotovs": 0,
        "t_smokes": 0, "t_flashes": 0, "t_he_nades" : 0, "t_molotovs": 0
    }

    for player_row in alive_players.iter_rows(named=True):
        inventory = player_row.get('inventory', [])
        player_side = player_row['side']
        
        best_weapon_tier = 0
        if inventory:
            for item_name in inventory:
                tier = WEAPON_TIERS.get(item_name)
                if tier is not None:
                    if tier > best_weapon_tier:
                        best_weapon_tier = tier
                elif (item_name not in GRENADE_AND_BOMB_TYPES) and ("Knife" not in item_name):
                    tqdm.write(f"Warning: Unknown weapon '{item_name}' not in WEAPON_TIERS.")
        
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

def create_snapshot(
    current_tick: int, round_row: Dict, demo_data: Dict, tick_rate: int, file_name: str
) -> Optional[Dict[str, Any]]:
    """Creates a single snapshot for a given tick."""
    
    freeze_end = round_row['freeze_end']
    end_tick = round_row['end']
    plant_tick = round_row['bomb_plant']
    
    if current_tick > end_tick:
        return None

    round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
    
    if plant_tick is not None and current_tick >= plant_tick:
        ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
        bomb_planted = True
    else:
        ticks_left = round_ticks_left
        bomb_planted = False

    if ticks_left <= 0:
        return None

    round_kills = demo_data['kills'].filter(pl.col('round_num') == round_row['round_num'])
    deaths_so_far = round_kills.filter(pl.col('tick') <= current_tick)
    
    ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
    t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height
    
    current_details = demo_data['ticks'].filter(pl.col('tick') == current_tick)

    hp_t = current_details.filter(pl.col('side') == 't')['health'].sum()
    hp_ct = current_details.filter(pl.col('side') == 'ct')['health'].sum()

    player_stats = {}
    if 'inventory' in current_details.columns:
        dead_player_steamids = deaths_so_far['victim_steamid'].unique().to_list()
        alive_players = current_details.filter(~pl.col('steamid').is_in(dead_player_steamids))
        player_stats = calculate_player_stats(alive_players)

    return {
        "source": f"Round {round_row['round_num']} in {file_name}",
        "time_left": ticks_left / tick_rate,
        "cts_alive": 5 - ct_deaths,
        "ts_alive": 5 - t_deaths,
        "hp_ct": int(hp_ct) if hp_ct is not None else 0,
        "hp_t": int(hp_t) if hp_t is not None else 0,
        "bomb_planted": bomb_planted,
        "winner": round_row['winner'],
        **player_stats
    }


def calculate_round_probabilities_all_ticks(
    dem_file, 
    round_num: int, 
    pred_model, 
    tick_rate: int = 64,
    file_name: str = "demo"
) -> list:
    """
    Calculate CT win probability at every single tick of a specified round.
    
    Args:
        dem_file: Parsed demo object with kills, rounds, and ticks data
        round_num: The round number to analyze
        pred_model: The prediction model with 'model' and 'feature_columns'
        tick_rate: Tick rate of the demo (default: 64)
        file_name: Name of the demo file for reference
    
    Returns:
        List of dictionaries containing tick number, time, probability, and game state info
    """
    # Prepare demo data
    dem = {}
    for name in ['kills', 'rounds', 'ticks']:
        df = getattr(dem_file, name, None)
        if df is None:
            tqdm.write(f"Warning: Missing '{name}' in dem")
            return []
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        dem[name] = df
    
    # Get the specific round
    round_data = dem['rounds'].filter(pl.col('round_num') == round_num)
    if round_data.height == 0:
        tqdm.write(f"Error: Round {round_num} not found")
        return []
    
    round_row = round_data.row(0, named=True)
    freeze_end = round_row['freeze_end']
    end_tick = round_row['end']
    
    if freeze_end is None:
        if round_num == 1:
            freeze_end = 200
        else:
            tqdm.write(f"Error: Round {round_num} has no freeze_end tick")
            return []
    
    if end_tick is None:
        tqdm.write(f"Error: Round {round_num} has no end tick")
        return []
    
    # Calculate probabilities for every tick from freeze_end to end_tick
    tick_probabilities = []
    
    for current_tick in tqdm(range(freeze_end, end_tick + 1), 
                            desc=f"Calculating probabilities for round {round_num}",
                            leave=False):
        snapshot = create_snapshot(current_tick, round_row, dem, tick_rate, file_name)
        
        if snapshot is None:
            continue
        
        try:
            ct_win_prob = predict(pred_model, snapshot)
            
            tick_probabilities.append({
                'tick': current_tick,
                'time_elapsed': (current_tick - freeze_end) / tick_rate,
                'time_left': snapshot['time_left'],
                'ct_win_probability': float(ct_win_prob),
                't_win_probability': float(1 - ct_win_prob),
                'cts_alive': snapshot['cts_alive'],
                'ts_alive': snapshot['ts_alive'],
                'bomb_planted': snapshot['bomb_planted'],
                'hp_ct': snapshot['hp_ct'],
                'hp_t': snapshot['hp_t']
            })
        except Exception as e:
            tqdm.write(f"Warning: Failed to predict at tick {current_tick}: {e}")
            continue
    
    return tick_probabilities


def get_player_kill_death_analysis(dem_file, player_name, pred_model, debug=False):
    """
    Analyze kills and deaths for a specific player with impact scoring.

    Args:
        dem_file: Parsed demo object
        player_name: Name of player to analyze
        debug: Enable debug output

    Returns:
        DataFrame with round-by-round analysis
    """

    dem = {}
    for name in ['kills', 'rounds', 'ticks', 'damages']:
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
        round_end = round_row['end']
        round_winner = round_row['winner']

        if freeze_end is None and round_num == 1:
            round_row['freeze_end'] = 200
            freeze_end = 200

        side = dem['ticks'].filter((pl.col('name') == player_name) & (pl.col('tick') >= freeze_end))['side']
        if len(side) == 0:
            tqdm.write(f"Skipping round {round_num} due to missing side info.")
            continue

        player_side = side[0]

        if freeze_end is None or round_row['end'] is None:
            tqdm.write(f"Skipping round {round_num} due to missing start/end ticks.")
            continue

        round_kills = dem['kills'].filter(pl.col('round_num') == round_num)
        round_damages = dem['damages'].filter(pl.col('round_num') == round_num)

        relevant_damages = round_damages.filter(
            (pl.col("attacker_name") == player_name)
        )
        damaged_victims = relevant_damages['victim_name'].to_list()

        relevant_kills = round_kills.filter(
            (pl.col("attacker_name") == player_name) | 
            (pl.col("victim_name") == player_name) | 
            (pl.col('victim_name').is_in(damaged_victims)) |
            (pl.col('assister_name') == player_name)
        )

        snapshot_ticks = sorted(set(relevant_kills['tick'].to_list()))

        for current_tick in snapshot_ticks:
            snapshot_before = create_snapshot(current_tick - 1, round_row, dem, 64, player_name)
            snapshot_after = create_snapshot(current_tick, round_row, dem, 64, player_name)

            if snapshot_before is None:
                continue

            round_winning_kill = current_tick + 1 >= round_end

            ct_win_before = predict(pred_model, snapshot_before)
            ct_win_after = predict(pred_model, snapshot_after) if not round_winning_kill else (1 if round_winner == 'ct' else 0)

            kill_row = relevant_kills.filter(pl.col("tick") == current_tick)
            attacker_was_ct = 'ct' in kill_row['attacker_side'].to_list()
            attacker = kill_row['attacker_name'].to_list()[0]

            was_kill = player_name in kill_row['attacker_name'].to_list()
            was_flash_assist = player_name in kill_row['assister_name'].to_list() and kill_row['assistedflash'][0]
            was_assist = not was_kill and player_name not in kill_row['victim_name'].to_list() or was_flash_assist
            was_death = not was_kill and not was_assist


            assist_damage = relevant_damages.filter(pl.col('victim_name') == kill_row['victim_name'])['dmg_health'].sum()
            kill_had_an_assist = kill_row['assister_name'].to_list()[0] != None

            impact = (ct_win_after - ct_win_before) * 100

            assist_weight = 0.5
            assist_cred = (assist_damage / 100) * assist_weight if not was_flash_assist else assist_weight
            kill_cred = 1.0 if not kill_had_an_assist else (1.0 - assist_cred)

            if was_kill:
                event_impact = abs(impact) * kill_cred
                assist_str = ' kill' +  (' (shared)' if kill_had_an_assist else '')
            elif was_assist:
                event_impact = abs(impact) * assist_cred
                assist_str = ' assist' + (" (f)" if was_flash_assist else "")
            else:
                assist_str = ' death'
                event_impact = -abs(impact)
                
            # Summarize kills/deaths for this round
            deaths_so_far = round_kills.filter(pl.col('tick') < current_tick)

            ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
            t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height

            if plant_tick is not None and current_tick >= plant_tick:
                bomb_planted = True
            else:
                bomb_planted = False

            plant_str = ' (pp)' if bomb_planted else ''

            results.append({
                'round': int(round_num),
                'side': player_side,
                'kills': '',
                'deaths': '',
                'impact': float(round(event_impact, 1)),
                'event_id': 0,
                'event_round': round_num,
                'event_type': 0,
                'event_impact': event_impact,
                'game_state': (f'{5 - t_deaths}v{5 - ct_deaths}'  if player_side == 't' else  f'{5 - ct_deaths}v{5 - t_deaths}')  + assist_str + plant_str,
                'pre_win': float(ct_win_before if player_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if player_side == 'ct' else (1-ct_win_after)),
                'weapon': '',
                'victim': '',
                'post_plant': bomb_planted,
                'tick': current_tick
            })

            if was_death:
                # Was death traded within 5 seconds?
                tick_in_5_sec = current_tick + (64 * 5)
                attacker_death = round_kills.filter(
                        (pl.col("victim_name") == attacker) & 
                        (pl.col('tick') < tick_in_5_sec)
                    )
                if attacker_death.height > 0:
                    trade_tick = attacker_death['tick'][0]
                    trade_snapshot_before = create_snapshot(trade_tick - 1, round_row, dem, 64, player_name)
                    trade_snapshot_after = create_snapshot(trade_tick, round_row, dem, 64, player_name)
                    trade_ct_win_before = predict(pred_model, trade_snapshot_before)
                    trade_ct_win_after = predict(pred_model, trade_snapshot_after)
                    trade_impact = abs((trade_ct_win_after - trade_ct_win_before)) * 100
                    trade_weight = 0.5
                    trade_cred = trade_impact * trade_weight

                    if plant_tick is not None and trade_tick >= plant_tick:
                        bomb_planted = True
                    else:
                        bomb_planted = False

                    results.append({
                        'round': int(round_num),
                        'side': player_side,
                        'kills': '',
                        'deaths': '',
                        'impact': float(round(trade_cred, 1)),
                        'event_id': 0,
                        'event_round': round_num,
                        'event_type': 0,
                        'event_impact': trade_cred,
                        'game_state': (f'{5 - t_deaths}v{5 - ct_deaths}'  if player_side == 't' else  f'{5 - ct_deaths}v{5 - t_deaths}') + " traded" + plant_str,
                        'pre_win': float(trade_ct_win_before if player_side == 'ct' else (1-trade_ct_win_before)),
                        'post_win': float(trade_ct_win_after if player_side == 'ct' else (1-trade_ct_win_after)),
                        'weapon': '',
                        'victim': '',
                        'post_plant': bomb_planted,
                        'tick': trade_tick
                    })


        did_player_survive_round = round_kills.filter(pl.col("victim_name") == player_name).height == 0
        did_player_lose_round = player_side != round_winner

        # Handle saving
        if did_player_lose_round and did_player_survive_round:
            current_tick = round_kills[-1]['tick'][0]
            snapshot = create_snapshot(current_tick, round_row, dem, 64, player_name)

            if snapshot is None:
                continue

            ct_win_prob = predict(pred_model, snapshot)

            deaths_so_far = round_kills.filter(pl.col('tick') <= current_tick)
            amount_of_players_left = 5 - deaths_so_far.filter(pl.col('victim_side') == player_side).height

            impact = 100 - ct_win_prob * 100 if round_winner == 'ct' else ct_win_prob * 100 
            event_impact = -abs(impact) / amount_of_players_left

            if plant_tick is not None and current_tick >= plant_tick:
                bomb_planted = True
            else:
                bomb_planted = False

            results.append({
                'round': int(round_num),
                'side': player_side,
                'kills': '',
                'deaths': '',
                'impact': float(round(event_impact, 1)),
                'event_id': 0,
                'event_round': round_num,
                'event_type': 0,
                'event_impact': event_impact,
                'game_state': f'{amount_of_players_left} saving',
                'pre_win': float(ct_win_before if player_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if player_side == 'ct' else (1-ct_win_after)),
                'weapon': '',
                'victim': '',
                'post_plant': bomb_planted,
                'tick': current_tick
            })



    return results


