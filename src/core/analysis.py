"""
Core analysis functions for CS2 demo processing.
"""

import pandas as pd
import polars as pl
from collections import defaultdict
from .win_probability import get_win_probability, calculate_impact_score
from ..utils.common import ensure_pandas


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
    kills = ensure_pandas(kills)
    bomb_events = ensure_pandas(bomb_events)
    
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
                if attacker_side == 'ct':
                    game_state = f"{ct_alive} v {t_alive}"
                else:
                    game_state = f"{t_alive} v {ct_alive}"
                
                if is_post_plant:
                    game_state += " post plant"
                
                weapon = kill.get('weapon', 'unknown')
                kill_description = f"{game_state} vs {weapon}"
                
                round_stats[round_num]['kills'].append(kill_description)
                round_stats[round_num]['kill_impacts'].append(impact)
            
            # Record deaths for our target player
            if victim_name == player_name:
                if victim_side == 'ct':
                    game_state = f"{ct_alive} v {t_alive}"
                else:
                    game_state = f"{t_alive} v {ct_alive}"

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
    Create a comprehensive table showing win probabilities for different scenarios.
    This helps understand how the impact scoring system works in various game states.
    """
    print("╔" + "═" * 120 + "╗")
    print("║" + " CS2 WIN PROBABILITY SCENARIOS - COMPREHENSIVE ANALYSIS ".center(120) + "║")
    print("╚" + "═" * 120 + "╝")
    print()
    
    def print_section_header(title):
        print(f"\n{'═' * 15} {title.upper()} {'═' * (105 - len(title))}")
        print(f"{'Scenario':<30} {'CT':<3} {'T':<3} {'Plant':<6} {'CT Win%':<8} {'CT Kill Δ':<10} {'T Kill Δ':<10} {'Context'}")
        print("─" * 120)
    
    def calculate_impacts(ct, t, post_plant):
        """Calculate impact of both CT and T getting a kill"""
        base_prob = get_win_probability(ct, t, post_plant)
        
        # CT gets a kill (T loses a player)
        ct_kill_impact = 0
        if t > 0:
            prob_after_ct_kill = get_win_probability(ct, t-1, post_plant)
            ct_kill_impact = abs(prob_after_ct_kill - base_prob) * 100
        
        # T gets a kill (CT loses a player)
        t_kill_impact = 0
        if ct > 0:
            prob_after_t_kill = get_win_probability(ct-1, t, post_plant)
            t_kill_impact = abs(prob_after_t_kill - base_prob) * 100
            
        return base_prob, ct_kill_impact, t_kill_impact
    
    def print_scenario(ct, t, post_plant, description, context=""):
        prob, ct_impact, t_impact = calculate_impacts(ct, t, post_plant)
        plant_str = "Yes" if post_plant else "No"
        
        print(f"{description:<30} {ct:<3} {t:<3} {plant_str:<6} {prob*100:>6.1f}%  "
              f"{ct_impact:>7.1f}     {t_impact:>7.1f}     {context}")
    
    # ROUND START SCENARIOS
    print_section_header("ROUND START SCENARIOS")
    print_scenario(5, 5, False, "Perfect Balance", "Standard round start")
    print_scenario(5, 4, False, "CT Slight Edge", "T lost someone early")
    print_scenario(4, 5, False, "T Slight Edge", "CT lost someone early")
    print_scenario(5, 3, False, "CT Moderate Advantage", "T lost 2 early")
    print_scenario(3, 5, False, "T Moderate Advantage", "CT lost 2 early")
    print_scenario(5, 2, False, "CT Strong Advantage", "T disaster")
    print_scenario(2, 5, False, "T Strong Advantage", "CT disaster")
    print_scenario(5, 1, False, "CT Overwhelming", "T lone survivor")
    print_scenario(1, 5, False, "T Overwhelming", "CT lone survivor")
    
    # MID-ROUND SCENARIOS
    print_section_header("MID-ROUND BALANCED SCENARIOS")
    print_scenario(4, 4, False, "Even Mid-Round", "Standard mid-round state")
    print_scenario(3, 3, False, "Close Mid-Round", "High stakes remaining")
    print_scenario(2, 2, False, "Tense Late Round", "Every kill crucial")
    print_scenario(4, 3, False, "CT Edge Mid-Round", "Slight CT advantage")
    print_scenario(3, 4, False, "T Edge Mid-Round", "Slight T advantage")
    print_scenario(3, 2, False, "CT Late Advantage", "CT likely to win")
    print_scenario(2, 3, False, "T Late Advantage", "T likely to win")
    
    # CLUTCH SCENARIOS
    print_section_header("CLUTCH SCENARIOS (1vX)")
    print_scenario(1, 1, False, "1v1 Clutch", "Ultimate skill battle")
    print_scenario(1, 2, False, "1v2 CT Clutch", "Hero CT play needed")
    print_scenario(2, 1, False, "1v2 T Clutch", "Hero T play needed")
    print_scenario(1, 3, False, "1v3 CT Clutch", "Nearly impossible CT")
    print_scenario(3, 1, False, "1v3 T Clutch", "Nearly impossible T")
    print_scenario(1, 4, False, "1v4 CT Clutch", "Miracle needed")
    print_scenario(4, 1, False, "1v4 T Clutch", "Miracle needed")
    print_scenario(1, 5, False, "1v5 CT Ace", "Legendary play")
    print_scenario(5, 1, False, "1v5 T Ace", "Legendary play")
    
    # PRE-PLANT VS POST-PLANT COMPARISON
    print_section_header("PRE-PLANT vs POST-PLANT COMPARISON")
    comparison_scenarios = [
        (5, 5, "Full Teams"),
        (4, 4, "Even Mid-Round"),
        (3, 3, "Close Battle"),
        (2, 2, "Late Round"),
        (1, 1, "1v1 Clutch"),
        (3, 2, "CT Advantage"),
        (2, 3, "T Advantage"),
        (4, 2, "CT Strong Position"),
        (2, 4, "T Strong Position"),
    ]
    
    for ct, t, desc in comparison_scenarios:
        pre_prob, pre_ct_impact, pre_t_impact = calculate_impacts(ct, t, False)
        post_prob, post_ct_impact, post_t_impact = calculate_impacts(ct, t, True)
        
        print(f"{desc + ' (Pre-Plant)':<30} {ct:<3} {t:<3} {'No':<6} {pre_prob*100:>6.1f}%  "
              f"{pre_ct_impact:>7.1f}     {pre_t_impact:>7.1f}     Plant changes everything")
        print(f"{desc + ' (Post-Plant)':<30} {ct:<3} {t:<3} {'Yes':<6} {post_prob*100:>6.1f}%  "
              f"{post_ct_impact:>7.1f}     {post_t_impact:>7.1f}     T favored after plant")
        print("─" * 120)
    
    # EXTREME SCENARIOS
    print_section_header("EXTREME & EDGE CASE SCENARIOS")
    print_scenario(5, 0, False, "T Eliminated", "Round over - CT wins")
    print_scenario(0, 5, False, "CT Eliminated", "Round over - T wins")
    print_scenario(4, 1, True, "CT Retake 4v1", "Post-plant retake")
    print_scenario(1, 4, True, "T Defend 1v4", "Post-plant defense")
    print_scenario(3, 1, True, "CT Retake 3v1", "Favored retake")
    print_scenario(1, 3, True, "T Defend 1v3", "Difficult defense")
    print_scenario(2, 1, True, "CT Retake 2v1", "Standard retake")
    print_scenario(1, 2, True, "T Defend 1v2", "Standard defense")
    
    # HIGH IMPACT SCENARIOS
    print_section_header("HIGHEST IMPACT KILL SCENARIOS")
    high_impact_scenarios = []
    
    # Find scenarios with highest impact kills
    test_scenarios = [
        (5, 5, False), (4, 4, False), (3, 3, False), (2, 2, False), (1, 1, False),
        (5, 4, False), (4, 5, False), (5, 3, False), (3, 5, False),
        (5, 5, True), (4, 4, True), (3, 3, True), (2, 2, True), (1, 1, True),
        (4, 3, True), (3, 4, True), (3, 2, True), (2, 3, True)
    ]
    
    for ct, t, post_plant in test_scenarios:
        _, ct_impact, t_impact = calculate_impacts(ct, t, post_plant)
        max_impact = max(ct_impact, t_impact)
        if max_impact > 15:  # Only show high-impact scenarios
            high_impact_scenarios.append((ct, t, post_plant, max_impact))
    
    # Sort by impact and show top scenarios
    high_impact_scenarios.sort(key=lambda x: x[3], reverse=True)
    
    for ct, t, post_plant, max_impact in high_impact_scenarios[:8]:
        _, ct_impact, t_impact = calculate_impacts(ct, t, post_plant)
        plant_context = "post-plant" if post_plant else "pre-plant"
        impact_source = "CT kill" if ct_impact > t_impact else "T kill"
        
        print_scenario(ct, t, post_plant, f"High Impact {ct}v{t}", 
                      f"{max_impact:.1f} impact from {impact_source}")
    
    # COMPREHENSIVE ANALYSIS SUMMARY
    print("\n" + "═" * 120)
    print("🎯 COMPREHENSIVE IMPACT ANALYSIS INSIGHTS")
    print("═" * 120)
    
    print("\n📊 IMPACT RANGES BY SCENARIO TYPE:")
    print("• Balanced scenarios (5v5, 4v4, 3v3): 15-25 impact per kill - HIGHEST")
    print("• Slight advantages (5v4, 4v5): 10-20 impact per kill - HIGH")
    print("• Moderate advantages (5v3, 3v5): 5-15 impact per kill - MEDIUM")
    print("• Strong advantages (5v2, 2v5): 2-8 impact per kill - LOW")
    print("• Overwhelming scenarios (5v1, 1v5): 0-3 impact per kill - MINIMAL")
    print("• Clutch scenarios (1v1, 1v2): 15-35 impact per kill - EXTREME")
    
    print("\n💣 POST-PLANT EFFECTS:")
    print("• Generally increases T win probability by 15-25%")
    print("• Most dramatic in balanced scenarios (5v5 → T favored)")
    print("• CT retakes become much more valuable")
    print("• Time pressure amplifies every decision")
    
    print("\n🎮 TACTICAL IMPLICATIONS:")
    print("• Most valuable kills happen in balanced mid-round situations")
    print("• Entry frags (5v5 → 5v4) have massive impact")
    print("• Clutch situations create the highest individual impact")
    print("• Post-plant kills favor the defending team more")
    print("• Eco rounds (uneven starts) have lower individual kill impact")
    
    print("\n🔥 HIGHEST VALUE SITUATIONS:")
    print("• 1v1 post-plant clutches: 35-45 impact")
    print("• Entry kills in 5v5: 20-25 impact")
    print("• Trade kills in 4v4: 18-23 impact")
    print("• Clutch entries (1v2 → 1v1): 25-35 impact")
    print("• Post-plant retake kills: 20-30 impact")
    
    print("\n⚡ MATHEMATICAL NOTES:")
    print("• Impact = |Win% After Kill - Win% Before Kill| × 100")
    print("• Values represent probability shift in percentage points")
    print("• Higher impact = more influential kill for round outcome")
    print("• Negative values for deaths (impact taken away from team)")
    print("• Post-plant scenarios use adjusted probability tables")
    print()
