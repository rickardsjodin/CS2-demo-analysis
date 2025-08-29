"""
Core analysis functions for CS2 demo processing.
"""

import pandas as pd
import polars as pl
from collections import defaultdict
from win_probability import get_win_probability, calculate_impact_score


def get_player_kill_death_analysis(dem, player_name, debug=False):
    """
    Analyze kills and deaths for a specific player with impact scoring.
    
    Args:
        dem: Parsed demo object
        player_name: Name of player to analyze
        debug: Enable debug output
    
    Returns:
        DataFrame with round-by-round analysis
    """
    # Get game data
    kills = dem.kills
    bomb_events = dem.bomb  # Fix: use dem.bomb instead of dem.bomb_events
    
    if kills is None or len(kills) == 0:
        print("❌ No kill data found in demo")
        return None
    
    # Convert Polars to pandas if needed
    if isinstance(kills, pl.DataFrame):
        kills = kills.to_pandas()
    
    if bomb_events is not None and isinstance(bomb_events, pl.DataFrame):
        bomb_events = bomb_events.to_pandas()
    
    # Initialize round statistics
    round_stats = defaultdict(lambda: {
        'kills': [], 
        'deaths': [], 
        'bomb_planted': False, 
        'plant_tick': None,
        'kill_impacts': [],
        'death_impacts': []
    })
    
    # Get all rounds that exist in the kill data
    all_rounds = sorted(kills['round_num'].unique())
    for round_num in all_rounds:
        if round_num > 0:  # Skip round 0 (warmup)
            round_stats[round_num] = {
                'kills': [], 
                'deaths': [], 
                'bomb_planted': False, 
                'plant_tick': None,
                'kill_impacts': [],
                'death_impacts': []
            }
    
    # Process bomb events first to get plant information
    if bomb_events is not None and len(bomb_events) > 0:
        if debug:
            print(f"Debug: Found {len(bomb_events)} bomb events")
            print("Bomb event columns:", bomb_events.columns.tolist())
        
        for _, bomb_event in bomb_events.iterrows():
            round_num = bomb_event.get('round_num', 0)
            if round_num in round_stats:
                # Check for different possible bomb event indicators
                event_columns = bomb_event.index.tolist()
                event_values = bomb_event.values
                
                # Look for plant indicators in various ways - BE MORE STRICT
                is_plant = False
                plant_tick = None
                
                # Method 1: Check for exact 'plant' event type
                if 'event_type' in event_columns:
                    event_type = str(bomb_event['event_type']).lower()
                    if event_type == 'plant' or event_type == 'bomb_plant':
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant via event_type in round {round_num}")
                
                # Method 2: Check for bombsite column with actual plant event
                if 'site' in event_columns and 'event_type' in event_columns:
                    if 'plant' in str(bomb_event['event_type']).lower():
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant via bombsite+event_type in round {round_num}")
                
                # Method 3: Only if we have a specific bomb plant indicator
                for col, val in zip(event_columns, event_values):
                    if isinstance(val, str) and val.lower() in ['plant', 'bomb_plant', 'c4_plant']:
                        is_plant = True
                        plant_tick = bomb_event.get('tick', 0)
                        if debug:
                            print(f"Debug: Found plant in round {round_num}, column {col}, value: {val}")
                        break
                
                if is_plant:
                    round_stats[round_num]['bomb_planted'] = True
                    round_stats[round_num]['plant_tick'] = plant_tick
    else:
        if debug:
            print("Debug: No bomb events found or bomb_events is None")
    
    # Process kills and deaths with game state tracking
    for round_num in sorted(round_stats.keys()):
        if round_num == 0:
            continue
            
        round_kills = kills[kills['round_num'] == round_num]
        
        # Track alive players during the round (start with 5v5)
        ct_alive = 5
        t_alive = 5
        
        for _, kill in round_kills.iterrows():
            attacker_name = kill.get('attacker_name', '')
            victim_name = kill.get('victim_name', '')
            attacker_side = kill.get('attacker_side', '')
            victim_side = kill.get('victim_side', '')
            
            # Check if bomb is planted at time of this kill
            is_post_plant = (round_stats[round_num]['bomb_planted'] and 
                           kill.get('tick', 0) >= round_stats[round_num].get('plant_tick', 0))
            
            # Calculate impact before updating counts
            if victim_side == 'ct':
                ct_after = max(0, ct_alive - 1)
                t_after = t_alive
            elif victim_side == 't':
                ct_after = ct_alive
                t_after = max(0, t_alive - 1)
            else:
                ct_after = ct_alive
                t_after = t_alive
            
            impact = calculate_impact_score(ct_alive, t_alive, ct_after, t_after, is_post_plant)
            
            # Record kills for our target player
            if attacker_name == player_name:
                game_state = f"{ct_alive} v {t_alive}"
                if is_post_plant:
                    game_state += " post plant"
                
                weapon = kill.get('weapon', 'unknown')
                kill_description = f"{game_state} vs {weapon}"
                
                round_stats[round_num]['kills'].append(kill_description)
                round_stats[round_num]['kill_impacts'].append(impact)
            
            # Record deaths for our target player
            if victim_name == player_name:
                game_state = f"{ct_alive} v {t_alive}"
                if is_post_plant:
                    game_state += " post plant"
                
                weapon = kill.get('weapon', 'unknown')
                death_description = f"{game_state} vs {weapon}"
                
                round_stats[round_num]['deaths'].append(death_description)
                round_stats[round_num]['death_impacts'].append(-impact)  # Negative for deaths
            
            # Update alive counts after the kill
            if victim_side == 'ct':
                ct_alive = max(0, ct_alive - 1)
            elif victim_side == 't':
                t_alive = max(0, t_alive - 1)
    
    # Format results - NOW INCLUDING ALL ROUNDS
    results = []
    for round_num in sorted(round_stats.keys()):
        if round_num == 0:
            continue
            
        side = "CT" if round_num <= 12 else "T"
        if round_num > 24:  # Overtime
            side = "CT" if ((round_num - 25) // 6) % 2 == 0 else "T"
        
        # Format kills and deaths with impact scores
        kills_list = round_stats[round_num]['kills']
        deaths_list = round_stats[round_num]['deaths']
        kill_impacts = round_stats[round_num]['kill_impacts']
        death_impacts = round_stats[round_num]['death_impacts']
        
        # Add impact scores to the display
        kills_with_impact = []
        for i, kill in enumerate(kills_list):
            impact = kill_impacts[i] if i < len(kill_impacts) else 0
            # Truncate long descriptions for better formatting
            kill_display = kill[:30] + "..." if len(kill) > 33 else kill
            kills_with_impact.append(f"{kill_display} (+{impact})")
        
        deaths_with_impact = []
        for i, death in enumerate(deaths_list):
            impact = abs(death_impacts[i]) if i < len(death_impacts) else 0
            # Truncate long descriptions for better formatting
            death_display = death[:30] + "..." if len(death) > 33 else death
            deaths_with_impact.append(f"{death_display} (-{impact})")
        
        # Join multiple events with proper formatting
        kills_text = ' | '.join(kills_with_impact) if kills_with_impact else '-'
        deaths_text = ' | '.join(deaths_with_impact) if deaths_with_impact else '-'
        
        # Calculate round impact
        round_impact = sum(kill_impacts) + sum(death_impacts)
        
        # Add bomb plant indicator if relevant
        round_info = ""
        if round_stats[round_num]['bomb_planted']:
            round_info = " 💣"
        
        results.append({
            'Round': f"{round_num}{round_info}",
            'Side': side,
            'Kills': kills_text,
            'Deaths': deaths_text,
            'Impact': f"{round_impact:+.1f}"
        })
    
    return pd.DataFrame(results)


def create_probability_scenarios_table():
    """
    Create a table showing win probabilities for different scenarios.
    This helps understand how the impact scoring system works.
    """
    print("╔" + "═" * 78 + "╗")
    print("║" + " CS2 WIN PROBABILITY SCENARIOS ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    print(f"{'Scenario':<25} {'CT':<3} {'T':<3} {'Post-Plant':<11} {'CT Win%':<8} {'Notes'}")
    print("─" * 85)
    
    scenarios = [
        (5, 5, False, "Balanced"),
        (5, 4, False, "CT slight advantage"),
        (4, 5, False, "T slight advantage"), 
        (5, 3, False, "CT advantage"),
        (3, 5, False, "T advantage"),
        (5, 2, False, "CT big advantage"),
        (2, 5, False, "T big advantage"),
        (5, 1, False, "CT overwhelming advantage"),
        (1, 5, False, "T overwhelming advantage"),
        (1, 1, False, "1v1 clutch"),
        (2, 2, False, "2v2"),
        (3, 3, False, "3v3"),
        (5, 5, True, "Balanced post-plant"),
        (5, 4, True, "CT slight adv post-plant"),
        (4, 5, True, "T slight adv post-plant"),
        (3, 2, True, "CT adv post-plant"),
        (2, 3, True, "T adv post-plant"),
        (1, 1, True, "1v1 post-plant clutch"),
    ]
    
    for ct, t, post_plant, description in scenarios:
        prob = get_win_probability(ct, t, post_plant)
        post_plant_str = "Yes" if post_plant else "No"
        
        # Calculate impact of a CT getting one kill
        if t > 0:
            prob_after_ct_kill = get_win_probability(ct, t-1, post_plant)
            impact_ct_kill = abs(prob_after_ct_kill - prob) * 100
        else:
            impact_ct_kill = 0
            
        impact_note = f"CT 1-kill impact: ~{impact_ct_kill:.1f}"
        
        print(f"{description:<25} {ct:<3} {t:<3} {post_plant_str:<11} {prob*100:>6.1f}%  {impact_note}")
    
    print("─" * 85)
    print("\nKey Insights:")
    print("• Higher impact when teams are evenly matched (5v5, 4v4)")
    print("• Lower impact in lopsided situations (5v1, 1v5)")
    print("• Post-plant scenarios favor Terrorists")
    print("• 'CT 1-kill impact' = impact of ONE CT kill on win probability")
    print("• Impact = |Win% After - Win% Before| × 100")
    print()
