"""
Core analysis functions for CS2 demo processing.
"""

import pandas as pd
import polars as pl
import os
from collections import defaultdict
from .win_probability import get_win_probability, calculate_impact_score
from ..utils.common import ensure_pandas
from .snapshot_extractor import load_demo_data, calculate_player_stats


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
    # Support two input types:
    # - `dem` is a path to a parsed-demo pickle file (string)
    # - `dem` is an in-memory parsed demo object with attributes `kills` and `bomb`
    kills = None
    bomb_events = None

    if isinstance(dem, str) and os.path.exists(dem):
        # Load structured demo data from the cache/pickle using the shared loader
        demo_data = load_demo_data(dem)
        if demo_data is None:
            print(f"❌ Could not load demo data from {dem}")
            return None

        # Extract kills as a pandas DataFrame
        kills = demo_data['kills']

        # Construct a minimal bomb_events DataFrame from the rounds table (plant ticks)
        rounds_df = demo_data['rounds']
        # Convert to pandas for the rest of the codepath
        rounds_pd = ensure_pandas(rounds_df)
        plant_rows = rounds_pd[rounds_pd['bomb_plant'].notnull()][['round_num', 'bomb_plant']]
        if not plant_rows.empty:
            bomb_events = pd.DataFrame({
                'round_num': plant_rows['round_num'].tolist(),
                'tick': plant_rows['bomb_plant'].tolist(),
                'event_type': ['plant'] * len(plant_rows)
            })
        else:
            bomb_events = pd.DataFrame(columns=['round_num', 'tick', 'event_type'])
    else:
        # Assume an in-memory parsed demo object
        kills = getattr(dem, 'kills', None)
        # support both dem.bomb and dem.bomb_events depending on caller
        bomb_events = getattr(dem, 'bomb', None)
        if bomb_events is None:
            bomb_events = getattr(dem, 'bomb_events', None)
    
    if kills is None or len(kills) == 0:
        print("❌ No kill data found in demo")
        return None
    
    # Convert Polars to pandas if needed
    kills = ensure_pandas(kills)
    # bomb_events may be None or already pandas; ensure it's a pandas DataFrame
    if bomb_events is None:
        bomb_events = pd.DataFrame(columns=['round_num', 'tick', 'event_type'])
    else:
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
    
    # Build a robust rounds/kills/ticks view using snapshot_extractor loader when possible
    demo_data = None
    # If we were passed a filepath, load structured demo data
    if isinstance(dem, str):
        demo_data = load_demo_data(dem)
        if demo_data is None:
            print(f"❌ Could not load demo data from {dem}")
            return None
        # Convert kills to pandas for iteration
        kills = ensure_pandas(demo_data['kills'])
        ticks = demo_data.get('ticks')
        rounds_table = demo_data.get('rounds')
        if ticks is not None:
            ticks = ensure_pandas(ticks)
        if rounds_table is not None:
            rounds_table = ensure_pandas(rounds_table)
        # Populate round_stats with plant info from rounds table
        if rounds_table is not None and 'bomb_plant' in rounds_table.columns:
            for _, rr in rounds_table.iterrows():
                rn = rr.get('round_num')
                if rn in round_stats:
                    plant_tick = rr.get('bomb_plant')
                    if pd.notna(plant_tick):
                        round_stats[rn]['bomb_planted'] = True
                        round_stats[rn]['plant_tick'] = int(plant_tick)
    else:
        # keep existing in-memory behaviour
        if debug:
            print("Using in-memory demo object for analysis")
    
    # Process kills and deaths with updated game-state tracking using demo_data when available
    for round_num in sorted(round_stats.keys()):
        if round_num == 0:
            continue

        # select round kills robustly
        try:
            round_kills = kills[kills['round_num'] == round_num]
        except Exception:
            round_kills = pd.DataFrame()

        # Track alive players during the round (start with 5v5)
        ct_alive = 5
        t_alive = 5

        # Pre-compute snapshot tick for inventory/player stats: prefer plant_tick then freeze_end then round start
        snapshot_tick = None
        if round_stats[round_num].get('plant_tick'):
            snapshot_tick = round_stats[round_num]['plant_tick']
        else:
            # try to pull freeze_end from rounds_table if available
            if isinstance(demo_data, dict) and demo_data.get('rounds') is not None:
                rounds_pd = ensure_pandas(demo_data['rounds'])
                rr = rounds_pd[rounds_pd['round_num'] == round_num]
                if not rr.empty and 'freeze_end' in rr.columns:
                    snapshot_tick = int(rr.iloc[0]['freeze_end'])

        # If we have tick-level data and a snapshot tick, compute inventory stats for alive players
        if snapshot_tick is not None and isinstance(demo_data, dict) and demo_data.get('ticks') is not None:
            ticks_pd = ensure_pandas(demo_data['ticks'])
            current_details = ticks_pd[ticks_pd['tick'] == snapshot_tick]
            # Exclude players already dead earlier in the round
            if not current_details.empty:
                # compute dead players up to snapshot tick from kills
                deaths_so_far = kills[(kills['round_num'] == round_num) & (kills['tick'] <= snapshot_tick)]
                dead_player_steamids = deaths_so_far['victim_steamid'].unique().tolist() if not deaths_so_far.empty else []
                alive_players = current_details[~current_details['steamid'].isin(dead_player_steamids)]
                # convert alive_players back to polars for calculate_player_stats which expects a pl.DataFrame
                try:
                    alive_pl = pl.from_pandas(alive_players)
                    player_stats = calculate_player_stats(alive_pl)
                    round_stats[round_num].update(player_stats)
                except Exception:
                    # if conversion fails, skip player stats
                    if debug:
                        print(f"Debug: Could not compute player stats for round {round_num}")

        for _, kill in round_kills.iterrows():
            attacker_name = kill.get('attacker_name', '')
            victim_name = kill.get('victim_name', '')
            attacker_side = kill.get('attacker_side', '')
            victim_side = kill.get('victim_side', '')

            # Check if bomb is planted at time of this kill
            is_post_plant = False
            plant_tick = round_stats[round_num].get('plant_tick')
            if plant_tick and pd.notna(kill.get('tick', None)):
                is_post_plant = int(kill.get('tick', 0)) >= int(plant_tick)

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
