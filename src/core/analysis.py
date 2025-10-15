"""
Core analysis functions for CS2 demo processing.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional
import joblib
import pandas as pd
import polars as pl

from tqdm import tqdm


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import BOMB_TIME, MODELS_DIR, ROUND_TIME, TICK_RATE
from src.core.constants import FLASH_NADE, GRENADE_AND_BOMB_TYPES, HE_NADE, MOLOTOV_NADE, SMOKE_NADE, WEAPON_TIERS
from src.ml.feature_engineering import create_features
from src.utils.cache_utils import load_demo



def predict(model_data, snapshot_data):
    df = pd.DataFrame([snapshot_data])
    try:
        df = create_features(df)
    except Exception as e:
        print(f"❌ Error in create_features: {e}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame data: {df.to_dict('records')[0]}")
        raise

    # Ensure all required feature columns are present
    feature_columns = model_data['feature_columns']
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError(f"Missing required feature columns: {missing_cols}")
    
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
    current_tick: int, round_row: Dict, demo_data: Dict, tick_rate: int, file_name: str = ""
) -> Optional[Dict[str, Any]]:
    """Creates a single snapshot for a given tick."""
    
    freeze_end = round_row['freeze_end']
    end_tick = round_row['end']
    plant_tick = round_row['bomb_plant']
    
    if current_tick > end_tick:
        if file_name:  # Only log if we have context
            tqdm.write(f"⚠️ Tick {current_tick} > end_tick {end_tick} in {file_name}")
        return None

    round_ticks_left = max(0, (freeze_end + ROUND_TIME * tick_rate) - current_tick)
    
    if plant_tick is not None and current_tick >= plant_tick:
        ticks_left = max(0, (plant_tick + BOMB_TIME * tick_rate) - current_tick)
        bomb_planted = True
    else:
        ticks_left = round_ticks_left
        bomb_planted = False

    if ticks_left <= 0:
        if file_name:  # Only log if we have context
            tqdm.write(f"⚠️ ticks_left <= 0 ({ticks_left}) at tick {current_tick} in {file_name}")
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
    tick_step: int = 10,
    file_name: str = "demo"
) -> list:
    """
    Calculate CT win probability at every single tick of a specified round.
    
    Args:
        dem_file: Parsed demo object with kills, rounds, and ticks data
        round_num: The round number to analyze
        pred_model: The prediction model with 'model' and 'feature_columns'
        tick_rate: Tick rate of the demo (default: 64)
        tick_step: Step size for sampling ticks (default: 10, meaning every 10th tick)
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
    
    for current_tick in tqdm(range(freeze_end, end_tick + 1, tick_step), 
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
            snapshot_before = create_snapshot(current_tick - 1, round_row, dem, TICK_RATE)
            snapshot_after = create_snapshot(current_tick, round_row, dem, TICK_RATE)

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


                    deaths_so_far = round_kills.filter(pl.col('tick') < trade_tick)

                    ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
                    t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height

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


def get_all_player_names(dem):
    kills_df = dem['kills']
    
    # Get unique attackers and victims (Polars syntax)
    attackers = set(kills_df['attacker_name'].drop_nulls().unique().to_list())
    victims = set(kills_df['victim_name'].drop_nulls().unique().to_list())
    all_players = sorted(attackers.union(victims))
    
    return all_players

def get_player_and_round_num_to_side_map(dem):
    """
    Create a mapping of (player_name, round_num) to side ('ct' or 't').
    
    Args:
        dem: Dictionary containing demo data with 'ticks' and 'rounds' DataFrames
        
    Returns:
        Dictionary mapping (player_name, round_num) tuples to side strings
    """
    player_side_map = {}  # map[(player_name, round_num)] = side
    
    # Get all rounds
    rounds_df = dem['rounds']
    ticks_df = dem['ticks']
    
    # For each round, get player sides from the tick data
    for round_row in rounds_df.iter_rows(named=True):
        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        
        # Use freeze_end as reference tick, or default to 200 for round 1
        if freeze_end is None:
            if round_num == 1:
                freeze_end = 200
            else:
                continue  # Skip rounds without freeze_end
        
        # Get all player data at or after freeze_end for this round
        round_ticks = ticks_df.filter(
            (pl.col('tick') >= freeze_end) & 
            (pl.col('tick') <= round_row['end']) if round_row['end'] is not None 
            else pl.col('tick') >= freeze_end
        )
        
        # Get unique player-side combinations for this round
        player_sides = round_ticks.select(['name', 'side']).unique()
        
        for player_row in player_sides.iter_rows(named=True):
            player_name = player_row['name']
            side = player_row['side']
            if player_name is not None and side is not None:
                player_side_map[(player_name, round_num)] = side
    
    return player_side_map

def get_kill_death_analysis(dem_file, pred_model, debug=False):

    TRADE_LIMIT_SEC = 5
    TRADE_KILL_FACTOR = 0.6
    FLASH_ASSIST_FACTOR = 0.4
    DAMAGE_ASSIST_FACTOR = 0.6

    dem = {}
    for name in ['kills', 'rounds', 'ticks', 'damages']:
        df = getattr(dem_file, name, None)
        if df is None:
            tqdm.write(f"Warning: Missing '{name}' in dem")
            return None
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        dem[name] = df
    
    all_players = get_all_player_names(dem)

    player_side_map = get_player_and_round_num_to_side_map(dem)
    player_analysis = {player_name: [] for player_name in all_players}

    for round_row in dem['rounds'].iter_rows(named=True):
        round_player_analysis = {player_name: [] for player_name in all_players}

        round_num = round_row['round_num']
        freeze_end = round_row['freeze_end']
        plant_tick = round_row['bomb_plant']
        round_end = round_row['end']
        round_winner = round_row['winner']

        if freeze_end is None and round_num == 1:
            round_row['freeze_end'] = 200
            freeze_end = 200

        if freeze_end is None or round_row['end'] is None:
            tqdm.write(f"Skipping round {round_num} due to missing start/end ticks.")
            continue

        round_kills = dem['kills'].filter((pl.col('round_num') == round_num) & (pl.col('tick') <= round_end))
        round_damages = dem['damages'].filter((pl.col('round_num') == round_num) & (pl.col('tick') <= round_end))

        cts_alive = 5
        ts_alive = 5
        
        for kill_row in round_kills.iter_rows(named=True):
            tick = kill_row['tick']
            attacker = kill_row['attacker_name']
            attacker_side = kill_row['attacker_side']
            victim_side = kill_row['victim_side']

            if attacker_side == victim_side:
                cts_alive -= 1 if victim_side == 'ct' else 0
                ts_alive -= 1 if victim_side == 't' else 0
                continue # Don't care about TK's

            victim = kill_row['victim_name']
            flash_assist = kill_row['assistedflash']
            flash_assister = kill_row['assister_name']
            flash_assister_side = kill_row['assister_side']
            bomb_planted = plant_tick is not None and tick >= plant_tick

            snapshot_before = create_snapshot(tick - 1, round_row, dem, TICK_RATE)
            snapshot_after = create_snapshot(tick, round_row, dem, TICK_RATE)
            
            if snapshot_before is None or snapshot_after is None:
                continue

            ct_win_before = predict(pred_model, snapshot_before)
            ct_win_after = predict(pred_model, snapshot_after)

            ct_delta = ct_win_after - ct_win_before

            attacker_delta = ct_delta if attacker_side == 'ct' else -ct_delta
            victim_delta = ct_delta if victim_side == 'ct' else -ct_delta

            was_trade_kill = round_kills.filter( 
                (pl.col('tick') < tick) & 
                (pl.col('tick') > tick - TRADE_LIMIT_SEC * TICK_RATE) &
                (pl.col('attacker_name') == victim)
            ).height > 0

            # FLASH ASSISTS
            if flash_assist:
                assister_credit = attacker_delta * FLASH_ASSIST_FACTOR
                attacker_delta -= assister_credit # Remove the given credit from attacker pie

                round_player_analysis[flash_assister].append({
                    'round': int(round_num),
                    'side': flash_assister_side,
                    'impact': float(round(assister_credit * 100, 2)),
                    'event_round': round_num,
                    'event_type': 'flash_assist',
                    'game_state': f'{cts_alive}v{ts_alive}' if flash_assister_side == 'ct' else f'{ts_alive}v{cts_alive}',
                    'pre_win': float(ct_win_before if flash_assister_side == 'ct' else (1-ct_win_before)),
                    'post_win': float(ct_win_after if flash_assister_side == 'ct' else (1-ct_win_after)),
                    'post_plant': bomb_planted,
                    'trade': was_trade_kill,
                    'tick': tick
                })
                 
            # DAMAGE CONTRIBUTION
            player_damage_contribution = {}
            for damages in round_damages.filter(
                (pl.col('victim_name') == victim) & 
                (pl.col('attacker_name') != attacker)
            ).iter_rows(named=True):
                dmg_attacker_name = damages['attacker_name']
                dmg_attacker_side = damages['attacker_side']
                dmg_victim_side = damages['victim_side']
                if dmg_attacker_side == dmg_victim_side: # Don't care about TK's
                    continue
                dmg = damages['dmg_health']
                if dmg_attacker_name not in player_damage_contribution:
                    player_damage_contribution[dmg_attacker_name] = {
                        'dmg': dmg,
                        'attacker_side': dmg_attacker_side
                    }
                else:
                    player_damage_contribution[dmg_attacker_name]['dmg'] += dmg

            for damage_assister in player_damage_contribution.keys():
                damage_on_victim = player_damage_contribution[damage_assister]['dmg']
                damage_assister_side = player_damage_contribution[damage_assister]['attacker_side']

                damage_assister_credit = attacker_delta * (damage_on_victim / 100) * DAMAGE_ASSIST_FACTOR

                attacker_delta -= damage_assister_credit

                round_player_analysis[damage_assister].append({
                    'round': int(round_num),
                    'side': damage_assister_side,
                    'impact': float(round(damage_assister_credit * 100, 2)),
                    'event_round': round_num,
                    'event_type': 'assist',
                    'game_state': f'{cts_alive}v{ts_alive}' if damage_assister_side == 'ct' else f'{ts_alive}v{cts_alive}',
                    'pre_win': float(ct_win_before if damage_assister_side == 'ct' else (1-ct_win_before)),
                    'post_win': float(ct_win_after if damage_assister_side == 'ct' else (1-ct_win_after)),
                    'post_plant': bomb_planted,
                    'trade': was_trade_kill,
                    'tick': tick
                })


            # KILLS
            trade_factor = TRADE_KILL_FACTOR if was_trade_kill else 1

            attacker_credit = attacker_delta * trade_factor 

            round_player_analysis[attacker].append({
                'round': int(round_num),
                'side': attacker_side,
                'impact': float(round(attacker_credit * 100, 2)),
                'event_round': round_num,
                'event_type': 'kill',
                'game_state': f'{cts_alive}v{ts_alive}' if attacker_side == 'ct' else f'{ts_alive}v{cts_alive}',
                'pre_win': float(ct_win_before if attacker_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if attacker_side == 'ct' else (1-ct_win_after)),
                'post_plant': bomb_planted,
                'trade': was_trade_kill,
                'tick': tick
            })

            will_be_traded = round_kills.filter(
                (pl.col('tick') > tick) & 
                (pl.col('tick') < tick + TRADE_LIMIT_SEC * TICK_RATE) &
                (pl.col('victim_name') == attacker)
            ).height > 0

            victim_trade_factor = 1 - TRADE_KILL_FACTOR if will_be_traded else 1
            victim_credit = victim_delta * victim_trade_factor 

            round_player_analysis[victim].append({
                'round': int(round_num),
                'side': victim_side,
                'impact': float(round(victim_credit * 100, 2)),
                'event_round': round_num,
                'event_type': 'death' + (' (t)' if will_be_traded else ''),
                'game_state': f'{cts_alive}v{ts_alive}' if victim_side == 'ct' else f'{ts_alive}v{cts_alive}',
                'pre_win': float(ct_win_before if victim_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if victim_side == 'ct' else (1-ct_win_after)),
                'post_plant': bomb_planted,
                'trade': will_be_traded,
                'tick': tick
            })

            cts_alive -= 1 if victim_side == 'ct' else 0
            ts_alive -= 1 if victim_side == 't' else 0

        # BOMB PLANT
        if plant_tick is not None:
            snapshot_before = create_snapshot(plant_tick - 1, round_row, dem, TICK_RATE)
            snapshot_after = create_snapshot(plant_tick, round_row, dem, TICK_RATE)

            if snapshot_before is not None and snapshot_after is not None:
                ct_win_pre_plant = predict(pred_model, snapshot_before)
                ct_win_post_plant = predict(pred_model, snapshot_after)

                ts_that_have_contributed = set()
                # Check which Ts have contributed
                for player in round_player_analysis.keys():
                    contributions = round_player_analysis[player]
                    if any(item['impact'] > 0 and item['side'] == 't' for item in contributions):
                        ts_that_have_contributed.add(player)
                
                deaths_so_far = round_kills.filter(pl.col('tick') <= plant_tick)
                ct_deaths = deaths_so_far.filter(pl.col('victim_side') == 'ct').height
                t_deaths = deaths_so_far.filter(pl.col('victim_side') == 't').height
                cts_alive = 5 - ct_deaths
                ts_alive = 5 - t_deaths

                if len(ts_that_have_contributed) > 0:
                    for player in ts_that_have_contributed:
                        credit = -(ct_win_post_plant - ct_win_pre_plant) / len(ts_that_have_contributed)
                        round_player_analysis[player].append({
                            'round': int(round_num),
                            'side': 't',
                            'impact': float(round(credit * 100, 2)),
                            'event_round': round_num,
                            'event_type': 'bomb_plant',
                            'game_state': f'{ts_alive}v{cts_alive}',
                            'pre_win': (1-ct_win_pre_plant),
                            'post_win': (1-ct_win_post_plant),
                            'post_plant': True,
                            'trade': False,
                            'tick': plant_tick
                        })

            # TODO: Discredit all CT's?

        # SAVING / ROUND END
        snapshot_before_end = create_snapshot(round_end - 100, round_row, dem, TICK_RATE)
        
        if snapshot_before_end is not None and not 'killed' in round_row['reason']:
            ct_win_pre_end = predict(pred_model, snapshot_before_end)
            ct_win_after_end = 1 if round_winner == 'ct' else 0

            end_delta = ct_win_after_end - ct_win_pre_end

            dead_players = round_kills['victim_name'].to_list()
            alive_players = [name for name in all_players if name not in dead_players]

            ct_players = []
            t_players = []
            for player in alive_players:
                if player_side_map[(player, round_num)] == 'ct':
                    ct_players.append(player)
                else:
                    t_players.append(player)

            cts_alive = len(ct_players)
            ts_alive = len(t_players)

            if cts_alive == 0 or ts_alive == 0:
                pass
            
            for ct_player in ct_players:
                end_credit = end_delta / len(ct_players)
                round_player_analysis[ct_player].append({
                    'round': int(round_num),
                    'side': 'ct',
                    'impact': float(round(end_credit * 100, 2)),
                    'event_round': round_num,
                    'event_type': round_row['reason'],
                    'game_state': f'{cts_alive}v{ts_alive}',
                    'pre_win': ct_win_pre_end,
                    'post_win': ct_win_after_end,
                    'post_plant': plant_tick is not None,
                    'trade': False,
                    'tick': round_end
                })

            for t_player in t_players:
                end_credit = -end_delta / len(t_players)
                round_player_analysis[t_player].append({
                    'round': int(round_num),
                    'side': 't',
                    'impact': float(round(end_credit * 100, 2)),
                    'event_round': round_num,
                    'event_type': round_row['reason'],
                    'game_state': f'{ts_alive}v{cts_alive}',
                    'pre_win': 1 -ct_win_pre_end,
                    'post_win': 1 - ct_win_after_end,
                    'post_plant': plant_tick is not None,
                    'trade': False,
                    'tick': round_end
                })




        for player in round_player_analysis.keys():
            player_analysis[player].extend(
                sorted(
                    round_player_analysis[player],
                    key=lambda item: item['tick']
                )
            )

    return player_analysis


if __name__ == "__main__":
    model_path = MODELS_DIR / f"ct_win_probability_xgboost_hltv.pkl"
    pred_model = joblib.load(model_path)


    DEMO_FILE = str("demos/the-mongolz-vs-vitality-m1-mirage.dem")  # or set your own path
    dem = load_demo(DEMO_FILE, use_cache=True)
    
    get_kill_death_analysis(dem, pred_model)
    
    