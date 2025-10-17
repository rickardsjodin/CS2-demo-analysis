#!/usr/bin/env python3
"""
Analyze 13-0 scoreboard data using pandas DataFrames.

This script loads the raw scoreboard JSON data and converts it into
structured DataFrames for analysis. Creates separate DataFrames for:
- T side players (winners)
- CT side players (losers)
- Combined data with all players
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


# Input file
INPUT_JSON = "13_0_raw_scoreboard_data.json"


def load_scoreboard_data(json_file: Path) -> Dict[str, Any]:
    """Load the scoreboard data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_player_dataframe(scoreboards: List[Dict[str, Any]], side: str = "both") -> pd.DataFrame:
    """
    Create a DataFrame from scoreboard data.
    
    Args:
        scoreboards: List of scoreboard dictionaries
        side: "T", "CT", or "both" - which side to include
    
    Returns:
        DataFrame with all player stats
    """
    rows = []
    
    for sb in scoreboards:
        match_id = sb["match_id"]
        match_url = sb["match_url"]
        map_name = sb["map_name"]
        event = sb["event"]
        date = sb["date"]
        
        # Process winner (13-0 team)
        winner = sb["winner"]
        if side == "both" or side == winner["side"]:
            headers = winner["table_headers"]
            for row_data in winner["table_rows"]:
                player_dict = {
                    "match_id": match_id,
                    "match_url": match_url,
                    "map": map_name,
                    "event": event,
                    "date": date,
                    "team": winner["team_name"],
                    "side": winner["side"],
                    "result": "won",
                    "rounds_won": 13,
                    "rounds_lost": 0
                }
                
                # Add all columns from the table
                for i, header in enumerate(headers):
                    if i < len(row_data):
                        player_dict[header] = row_data[i]
                
                rows.append(player_dict)
        
        # Process loser (0-13 team)
        loser = sb["loser"]
        if side == "both" or side == loser["side"]:
            headers = loser["table_headers"]
            for row_data in loser["table_rows"]:
                player_dict = {
                    "match_id": match_id,
                    "match_url": match_url,
                    "map": map_name,
                    "event": event,
                    "date": date,
                    "team": loser["team_name"],
                    "side": loser["side"],
                    "result": "lost",
                    "rounds_won": 0,
                    "rounds_lost": 13
                }
                
                # Add all columns from the table
                for i, header in enumerate(headers):
                    if i < len(row_data):
                        player_dict[header] = row_data[i]
                
                rows.append(player_dict)
    
    return pd.DataFrame(rows)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to proper types.
    
    Common columns: K, D, +/-, ADR, KAST, Rating
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Define columns that should be numeric
    numeric_columns = {
        'K': 'int',
        'D': 'int',
        '+/-': 'int',
        'ADR': 'float',
        'KAST': 'float',
        'Rating': 'float',
        'HS': 'float',  # Headshot percentage
        'FK': 'int',  # First kills
        'FD': 'int',  # First deaths
    }
    
    for col, dtype in numeric_columns.items():
        if col in df.columns:
            # Clean the column (remove %, etc.)
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            
            # Convert to numeric
            if dtype == 'int':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def print_dataframe_summary(df: pd.DataFrame, title: str):
    """Print a summary of the DataFrame."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)


def analyze_t_vs_ct_performance(df: pd.DataFrame):
    """Analyze T side vs CT side performance in 13-0 matches."""
    print(f"\n{'='*80}")
    print("T Side vs CT Side Analysis (13-0 Matches)")
    print(f"{'='*80}")
    
    # Group by side and result
    if 'side' in df.columns and 'result' in df.columns:
        print("\nDistribution by Side and Result:")
        print(df.groupby(['side', 'result']).size())
        
        # T side stats
        t_side = df[df['side'] == 'T']
        ct_side = df[df['side'] == 'CT']
        
        print(f"\n{'─'*80}")
        print("T Side Players (in 13-0 matches):")
        print(f"{'─'*80}")
        print(f"Total players: {len(t_side)}")
        print(f"Winners: {len(t_side[t_side['result'] == 'won'])}")
        print(f"Losers: {len(t_side[t_side['result'] == 'lost'])}")
        
        if 'Rating' in df.columns:
            print(f"\nAverage Rating:")
            print(f"  T side winners: {t_side[t_side['result'] == 'won']['Rating'].mean():.3f}")
            print(f"  T side losers:  {t_side[t_side['result'] == 'lost']['Rating'].mean():.3f}")
        
        print(f"\n{'─'*80}")
        print("CT Side Players (in 13-0 matches):")
        print(f"{'─'*80}")
        print(f"Total players: {len(ct_side)}")
        print(f"Winners: {len(ct_side[ct_side['result'] == 'won'])}")
        print(f"Losers: {len(ct_side[ct_side['result'] == 'lost'])}")
        
        if 'Rating' in df.columns:
            print(f"\nAverage Rating:")
            print(f"  CT side winners: {ct_side[ct_side['result'] == 'won']['Rating'].mean():.3f}")
            print(f"  CT side losers:  {ct_side[ct_side['result'] == 'lost']['Rating'].mean():.3f}")


def analyze_top_performers(df: pd.DataFrame, n: int = 10):
    """Find top performers in 13-0 matches."""
    print(f"\n{'='*80}")
    print(f"Top {n} Performers in 13-0 Matches")
    print(f"{'='*80}")
    
    if 'Rating' not in df.columns or 'Player' not in df.columns:
        print("Missing required columns (Rating, Player)")
        return
    
    # Top performers overall
    print(f"\nTop {n} by Rating (all players):")
    top_players = df.nlargest(n, 'Rating')[['Player', 'team', 'map', 'side', 'result', 'Rating', 'K', 'D', 'ADR']]
    print(top_players.to_string(index=False))
    
    # Top performers on winning side
    winners = df[df['result'] == 'won']
    if not winners.empty:
        print(f"\nTop {n} Winners by Rating:")
        top_winners = winners.nlargest(n, 'Rating')[['Player', 'team', 'map', 'side', 'Rating', 'K', 'D', 'ADR']]
        print(top_winners.to_string(index=False))
    
    # Top performers on losing side
    losers = df[df['result'] == 'lost']
    if not losers.empty:
        print(f"\nTop {n} Losers by Rating (despite 0-13 loss):")
        top_losers = losers.nlargest(n, 'Rating')[['Player', 'team', 'map', 'side', 'Rating', 'K', 'D', 'ADR']]
        print(top_losers.to_string(index=False))


def analyze_by_map(df: pd.DataFrame):
    """Analyze 13-0 matches by map."""
    print(f"\n{'='*80}")
    print("Analysis by Map")
    print(f"{'='*80}")
    
    if 'map' not in df.columns:
        print("Missing 'map' column")
        return
    
    print("\nMaps where 13-0 occurred:")
    map_counts = df.groupby('map').size() / 10  # Divide by 10 (5 players per team × 2 teams)
    print(map_counts.astype(int))
    
    if 'side' in df.columns and 'result' in df.columns:
        print("\n13-0 Wins by Side per Map:")
        wins_by_map = df[df['result'] == 'won'].groupby(['map', 'side']).size().unstack(fill_value=0)
        print(wins_by_map)


def export_dataframes_to_csv(df_all: pd.DataFrame, df_t: pd.DataFrame, df_ct: pd.DataFrame):
    """Export DataFrames to CSV files."""
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export all data
    csv_all = output_dir / "13_0_all_players.csv"
    df_all.to_csv(csv_all, index=False)
    print(f"\n💾 Saved all players to: {csv_all}")
    
    # Export T side data
    csv_t = output_dir / "13_0_t_side_players.csv"
    df_t.to_csv(csv_t, index=False)
    print(f"💾 Saved T side players to: {csv_t}")
    
    # Export CT side data
    csv_ct = output_dir / "13_0_ct_side_players.csv"
    df_ct.to_csv(csv_ct, index=False)
    print(f"💾 Saved CT side players to: {csv_ct}")


def main():
    """Main function."""
    print("="*80)
    print("📊 13-0 Scoreboard Data Analysis")
    print("="*80)
    
    # Load data
    input_path = Path(INPUT_JSON)
    if not input_path.exists():
        print(f"❌ Error: Input file '{INPUT_JSON}' not found.")
        print(f"   Please run parse_13_0_raw_tables.py first.")
        return 1
    
    print(f"\n📖 Loading data from {INPUT_JSON}...")
    data = load_scoreboard_data(input_path)
    
    scoreboards = data.get("scoreboards", [])
    if not scoreboards:
        print("❌ No scoreboard data found in file.")
        return 1
    
    print(f"✅ Loaded {len(scoreboards)} scoreboard(s)")
    
    # Create DataFrames
    print("\n🔄 Creating DataFrames...")
    
    # All players
    df_all = create_player_dataframe(scoreboards, side="both")
    df_all = clean_numeric_columns(df_all)
    print_dataframe_summary(df_all, "All Players DataFrame")
    
    # T side only
    df_t = create_player_dataframe(scoreboards, side="T")
    df_t = clean_numeric_columns(df_t)
    print_dataframe_summary(df_t, "T Side Players DataFrame")
    
    # CT side only
    df_ct = create_player_dataframe(scoreboards, side="CT")
    df_ct = clean_numeric_columns(df_ct)
    print_dataframe_summary(df_ct, "CT Side Players DataFrame")
    
    # Perform analyses
    analyze_t_vs_ct_performance(df_all)
    analyze_top_performers(df_all, n=10)
    analyze_by_map(df_all)
    
    # Export to CSV
    export_dataframes_to_csv(df_all, df_t, df_ct)
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
