"""
Formatting utilities for displaying CS2 analysis results
"""

def format_player_analysis(stats, player_name):
    """Format and display the round-by-round player analysis in a clean format"""
    
    print("\n" + "="*80)
    print(f"📋 ROUND-BY-ROUND ANALYSIS: {player_name.upper()}")
    print("="*80)
    
    for _, row in stats.iterrows():
        round_num = str(row['Round']).replace('💣', '').strip()
        has_bomb = '💣' in str(row['Round'])
        side = str(row['Side'])
        kills_raw = str(row['Kills'])
        deaths_raw = str(row['Deaths'])
        impact_raw = str(row['Impact'])
        
        # Side indicator
        side_icon = "🔵CT" if side == "CT" else "🟠T"
        bomb_icon = "💣" if has_bomb else ""
        
        # Clean impact display
        impact_val = impact_raw.strip()
        if impact_val.startswith('+'):
            impact_display = f"🟢{impact_val}"
        elif impact_val.startswith('-'):
            impact_display = f"🔴{impact_val}"
        else:
            impact_display = f"⚪{impact_val}"
        
        # Round header
        print(f"\nRound {round_num} {bomb_icon} │ {side_icon} │ {impact_display}")
        print("─" * 50)
        
        # Process kills - each on its own line
        if kills_raw != '-':
            print("  KILLS:")
            kills = [k.strip() for k in kills_raw.split(' | ') if k.strip()]
            for i, kill in enumerate(kills, 1):
                formatted_kill = _clean_event_text(kill)
                print(f"    {i}. {formatted_kill}")
        
        # Process deaths - each on its own line  
        if deaths_raw != '-':
            print("  DEATHS:")
            deaths = [d.strip() for d in deaths_raw.split(' | ') if d.strip()]
            for i, death in enumerate(deaths, 1):
                formatted_death = _clean_event_text(death)
                print(f"    {i}. {formatted_death}")
    
    print("\n" + "="*80)


def _clean_event_text(event_text):
    """Clean up event text by removing weapons and formatting game state"""
    if '(' in event_text and ')' in event_text:
        # Extract game state and impact
        main_part = event_text[:event_text.rfind('(')].strip()
        impact_part = event_text[event_text.rfind('('):].strip()
        
        # Remove weapon info and clean up
        if ' vs ' in main_part:
            # Remove weapon part after "vs"
            game_state = main_part.split(' vs ')[0].replace(' v ', 'v')
        else:
            game_state = main_part.replace(' v ', 'v')
        
        return f"{game_state} {impact_part}"
    else:
        return event_text


def display_summary_stats(stats):
    """Display clean summary statistics"""
    
    # Calculate summary statistics
    total_kills = len([k for k in stats['Kills'] if k != '-'])
    total_deaths = len([d for d in stats['Deaths'] if d != '-'])
    total_impact = sum(float(imp.replace('+', '').replace('⚪', '').replace('🟢', '').replace('🔴', '').strip()) 
                      for imp in stats['Impact'] if imp not in ['-', '⚪-'])
    
    print(f"\n📊 SUMMARY")
    print("─" * 30)
    print(f"Kills:  {total_kills}")
    print(f"Deaths: {total_deaths}")
    print(f"K/D:    {total_kills/max(total_deaths, 1):.2f}")
    print(f"Impact: {total_impact:+.1f}")
    print("─" * 30)


def display_available_players(dem):
    """Display list of available players from the demo"""
    
    print("📋 AVAILABLE PLAYERS:")
    print("="*50)
    
    # Get unique player names from kills data
    if dem.kills is not None and len(dem.kills) > 0:
        kills_df = dem.kills.to_pandas() if hasattr(dem.kills, 'to_pandas') else dem.kills
        
        # Get unique attackers and victims
        attackers = set(kills_df['attacker_name'].dropna().unique())
        victims = set(kills_df['victim_name'].dropna().unique())
        all_players = sorted(attackers.union(victims))
        
        for i, player in enumerate(all_players, 1):
            print(f"{i:2d}. {player}")
        
        return all_players
    else:
        print("❌ No player data found")
        return []


def display_player_header(player_name):
    """Display clean header for player analysis"""
    print(f"╔{'═' * 68}╗")
    print(f"║ {player_name.upper()} - KILL/DEATH ANALYSIS{' ' * (67 - len(player_name) - 25)}║")
    print(f"╚{'═' * 68}╝")
    print()
