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
    TRADE_VICTIM_FACTOR = 0.3
    FLASH_ASSIST_FACTOR = 0.3
    DAMAGE_ASSIST_FACTOR = 0.5

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

            previous_victims = (
                round_kills
                .filter(
                    (pl.col('tick') < tick) &
                    (pl.col('tick') > tick - TRADE_LIMIT_SEC * TICK_RATE) &
                    (pl.col('attacker_name') == victim) &
                    (pl.col('victim_side') != victim_side)
                )
                .sort('tick')  # ascending order by tick
            )

            latest_traded_victim = previous_victims[-1]['victim_name'][0] if previous_victims.height > 0 else None

            was_trade_kill = previous_victims.height > 0

            if was_trade_kill and attacker == 'apEX':
                pass

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

            total_assist_credit = 0
            for damage_assister in player_damage_contribution.keys():
                damage_on_victim = player_damage_contribution[damage_assister]['dmg']
                damage_assister_side = player_damage_contribution[damage_assister]['attacker_side']

                damage_assister_credit = attacker_delta * (damage_on_victim / 100) * DAMAGE_ASSIST_FACTOR

                attacker_delta -= damage_assister_credit
                total_assist_credit += damage_assister_credit

                round_player_analysis[damage_assister].append({
                    'round': int(round_num),
                    'side': damage_assister_side,
                    'impact': float(round(damage_assister_credit * 100, 2)),
                    'event_round': round_num,
                    'event_type': 'assist ' + (str(round(damage_on_victim)) + 'hp'),
                    'game_state': f'{cts_alive}v{ts_alive}' if damage_assister_side == 'ct' else f'{ts_alive}v{cts_alive}',
                    'pre_win': float(ct_win_before if damage_assister_side == 'ct' else (1-ct_win_before)),
                    'post_win': float(ct_win_after if damage_assister_side == 'ct' else (1-ct_win_after)),
                    'post_plant': bomb_planted,
                    'trade': was_trade_kill,
                    'tick': tick
                })

            # TRADE
            if was_trade_kill and latest_traded_victim is not None:
                trade_victim_credit = TRADE_VICTIM_FACTOR * attacker_delta
                attacker_delta -= trade_victim_credit
                round_player_analysis[latest_traded_victim].append({
                    'round': int(round_num),
                    'side': attacker_side,
                    'impact': float(round(trade_victim_credit* 100, 2)),
                    'event_round': round_num,
                    'event_type': 'traded',
                    'game_state': f'{cts_alive}v{ts_alive}' if attacker_side == 'ct' else f'{ts_alive}v{cts_alive}',
                    'pre_win': float(ct_win_before if attacker_side == 'ct' else (1-ct_win_before)),
                    'post_win': float(ct_win_after if attacker_side == 'ct' else (1-ct_win_after)),
                    'post_plant': bomb_planted,
                    'trade': True,
                    'tick': tick
                })

            attacker_credit = attacker_delta
            # KILLS
            round_player_analysis[attacker].append({
                'round': int(round_num),
                'side': attacker_side,
                'impact': float(round(attacker_credit * 100, 2)),
                'event_round': round_num,
                'event_type': 'kill' + (' (t)' if was_trade_kill else '') + (' (a) ' + str(round(total_assist_credit, 1)) if total_assist_credit > 0 else '') + (' (fa)' if flash_assist else ''),
                'game_state': f'{cts_alive}v{ts_alive}' if attacker_side == 'ct' else f'{ts_alive}v{cts_alive}',
                'pre_win': float(ct_win_before if attacker_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if attacker_side == 'ct' else (1-ct_win_after)),
                'post_plant': bomb_planted,
                'trade': was_trade_kill,
                'tick': tick
            })

            victim_credit = victim_delta 

            round_player_analysis[victim].append({
                'round': int(round_num),
                'side': victim_side,
                'impact': float(round(victim_credit * 100, 2)),
                'event_round': round_num,
                'event_type': 'death',
                'game_state': f'{cts_alive}v{ts_alive}' if victim_side == 'ct' else f'{ts_alive}v{cts_alive}',
                'pre_win': float(ct_win_before if victim_side == 'ct' else (1-ct_win_before)),
                'post_win': float(ct_win_after if victim_side == 'ct' else (1-ct_win_after)),
                'post_plant': bomb_planted,
                'trade': False,
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
    
    