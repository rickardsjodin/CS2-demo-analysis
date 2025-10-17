#!/usr/bin/env python3
"""
Parse raw scoreboard tables from 13-0 matches.

Extracts the complete table data for both teams (T side and CT side)
without trying to interpret individual columns. Captures all player
rows and all columns as-is.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
HLTV = "https://www.hltv.org"
UA = "CS2-research/0.1 (contact: rickard.sjodin5@gmail.com; research only)"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 6
BASE_BACKOFF_DELAY = 0.5
MAX_BACKOFF_DELAY = 4.0
DELAY_BETWEEN_REQUESTS = 0.8

# Input/Output files
INPUT_JSON = "13_0_matches.json"
OUTPUT_JSON = "13_0_raw_scoreboard_data.json"


def sleep_backoff(try_idx: int) -> None:
    """Exponential backoff sleep."""
    time.sleep(min(MAX_BACKOFF_DELAY, BASE_BACKOFF_DELAY * (2 ** try_idx)))


def http_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """HTTP GET with retries and backoff."""
    for i in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code in (429, 503, 502, 500):
                sleep_backoff(i)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            sleep_backoff(i)
    raise RuntimeError(f"Failed GET after retries: {url}")


def parse_table_headers(table) -> List[str]:
    """Extract column headers from a table."""
    headers = []
    
    # Try to find header row
    header_row = table.select_one("thead tr") or table.select_one("tr")
    
    if header_row:
        # Get all th or td elements in header
        header_cells = header_row.select("th") or header_row.select("td")
        for cell in header_cells:
            headers.append(cell.get_text(strip=True))
    
    return headers


def parse_table_rows(table) -> List[List[str]]:
    """Extract all data rows from a table."""
    rows_data = []
    
    # Get all rows (skip header if in thead)
    tbody = table.select_one("tbody")
    if tbody:
        rows = tbody.select("tr")
    else:
        rows = table.select("tr")
    
    for row in rows:
        cells = row.select("td")
        if not cells:
            # Might be a header row, skip it
            continue
        
        row_data = []
        for cell in cells:
            # Get text content
            text = cell.get_text(strip=True)
            row_data.append(text)
        
        # Only include non-empty rows with data
        if row_data and any(row_data):
            rows_data.append(row_data)
    
    return rows_data


def determine_side_from_half_scores(match_url: str, session: requests.Session, 
                                   winner_team: str, map_name: str) -> tuple[str, str]:
    """
    Determine which side each team played (T or CT).
    
    Returns:
        (winner_side, loser_side) - tuple of "T" or "CT"
    """
    try:
        resp = http_get(session, match_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find the specific map holder for this map
        map_holders = soup.select(".mapholder")
        target_map_holder = None
        
        for holder in map_holders:
            map_name_elem = holder.select_one(".mapname")
            if map_name_elem and map_name.lower() in map_name_elem.get_text(strip=True).lower():
                target_map_holder = holder
                break
        
        if not target_map_holder:
            return "T", "CT"
        
        # Get team names
        team_elements = soup.select(".teamName")
        if len(team_elements) < 2:
            return "T", "CT"
        
        team1_name = team_elements[0].get_text(strip=True)
        winner_is_team1 = winner_team == team1_name
        
        # Look at round history to determine starting side
        round_history_teams = target_map_holder.select(".round-history-team")
        
        if len(round_history_teams) >= 2:
            team1_rounds = round_history_teams[0].select(".round-history-outcome")
            
            # Count wins in first half (rounds 0-11)
            team1_first_half_wins = sum(1 for i in range(min(12, len(team1_rounds))) 
                                       if 'won' in team1_rounds[i].get('class', []))
            
            if winner_is_team1:
                if team1_first_half_wins >= 12:
                    return "T", "CT"
                else:
                    return "CT", "T"
            else:
                team2_first_half_wins = 12 - team1_first_half_wins
                if team2_first_half_wins >= 12:
                    return "T", "CT"
                else:
                    return "CT", "T"
        
        return "T", "CT"
    
    except Exception as e:
        print(f"    Warning: Error determining sides: {e}")
        return "T", "CT"


def parse_scoreboard_for_map(match_url: str, map_name: str, winner_team: str, 
                            loser_team: str, session: requests.Session) -> Optional[Dict[str, Any]]:
    """
    Parse the raw scoreboard tables for a specific map.
    
    Returns:
        Dictionary with raw table data for both teams
    """
    try:
        # Fetch match page to find stats link
        resp = http_get(session, match_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find the stats link for this map
        map_holders = soup.select(".mapholder, .standard-box, .maps")
        stats_url = None
        
        for holder in map_holders:
            map_name_elem = holder.select_one(".mapname")
            if not map_name_elem:
                continue
                
            holder_map_name = map_name_elem.get_text(strip=True)
            
            if map_name.lower() in holder_map_name.lower():
                # Look for stats link
                stats_link = (
                    holder.select_one("a[href*='/stats/matches/']") or
                    holder.select_one("a.stats-link")
                )
                
                if stats_link:
                    href = stats_link.get("href")
                    if href:
                        stats_url = HLTV + href if not href.startswith("http") else href
                        break
        
        # If not found in map holders, try all stats links
        if not stats_url:
            all_stats_links = soup.select("a[href*='/stats/matches/']")
            for link in all_stats_links:
                href = link.get("href", "")
                if map_name.lower().replace(" ", "-") in href.lower():
                    stats_url = HLTV + href if not href.startswith("http") else href
                    break
            
            # Last resort: use first stats link if only one map
            if not stats_url and len(all_stats_links) == 1:
                href = all_stats_links[0].get("href", "")
                stats_url = HLTV + href if not href.startswith("http") else href
        
        if not stats_url:
            print(f"  ✗ Could not find stats URL for {map_name}")
            return None
        
        print(f"  ✓ Stats URL: {stats_url}")
        
        # Fetch the stats page
        time.sleep(DELAY_BETWEEN_REQUESTS)
        stats_resp = http_get(session, stats_url)
        stats_soup = BeautifulSoup(stats_resp.text, "html.parser")
        
        # Find all tables on the page
        all_tables = stats_soup.select("table")
        print(f"  Found {len(all_tables)} total tables")
        
        # Look for the main stats tables (usually have class 'table' or 'stats-table')
        stats_tables = (
            stats_soup.select("table.table.totalstats") or
            stats_soup.select("table.stats-table") or
            stats_soup.select("table[class*='total']") or
            all_tables
        )
        
        print(f"  Found {len(stats_tables)} stats tables")
        
        if len(stats_tables) < 2:
            print(f"  ✗ Not enough tables found")
            return None
        
        # Get team names from the stats page
        team_name_elements = stats_soup.select(".team-name, .teamName")
        team1_name_on_page = team_name_elements[0].get_text(strip=True) if len(team_name_elements) > 0 else winner_team
        team2_name_on_page = team_name_elements[1].get_text(strip=True) if len(team_name_elements) > 1 else loser_team
        
        print(f"  Team 1 on stats page: {team1_name_on_page}")
        print(f"  Team 2 on stats page: {team2_name_on_page}")
        
        # Parse first table (team1)
        table1_headers = parse_table_headers(stats_tables[0])
        table1_rows = parse_table_rows(stats_tables[0])
        
        print(f"  Table 1: {len(table1_headers)} columns, {len(table1_rows)} rows")
        print(f"    Headers: {table1_headers}")
        if table1_rows:
            print(f"    Sample row: {table1_rows[0]}")
        
        # Parse second table (team2)
        table2_headers = parse_table_headers(stats_tables[1])
        table2_rows = parse_table_rows(stats_tables[1])
        
        print(f"  Table 2: {len(table2_headers)} columns, {len(table2_rows)} rows")
        print(f"    Headers: {table2_headers}")
        if table2_rows:
            print(f"    Sample row: {table2_rows[0]}")
        
        # Determine which side each team played
        winner_side, loser_side = determine_side_from_half_scores(match_url, session, winner_team, map_name)
        
        # Determine which table is winner/loser
        if winner_team == team1_name_on_page or winner_team in team1_name_on_page:
            winner_table_headers = table1_headers
            winner_table_rows = table1_rows
            winner_table_team_name = team1_name_on_page
            loser_table_headers = table2_headers
            loser_table_rows = table2_rows
            loser_table_team_name = team2_name_on_page
        else:
            winner_table_headers = table2_headers
            winner_table_rows = table2_rows
            winner_table_team_name = team2_name_on_page
            loser_table_headers = table1_headers
            loser_table_rows = table1_rows
            loser_table_team_name = team1_name_on_page
        
        return {
            "winner": {
                "team_name": winner_team,
                "team_name_on_stats_page": winner_table_team_name,
                "side": winner_side,
                "rounds_won": 13,
                "table_headers": winner_table_headers,
                "table_rows": winner_table_rows
            },
            "loser": {
                "team_name": loser_team,
                "team_name_on_stats_page": loser_table_team_name,
                "side": loser_side,
                "rounds_won": 0,
                "table_headers": loser_table_headers,
                "table_rows": loser_table_rows
            }
        }
    
    except Exception as e:
        print(f"  ✗ Error parsing scoreboard: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_13_0_matches(json_file: Path) -> List[Dict]:
    """Load the 13-0 matches from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('matches', [])


def parse_all_scoreboards(matches: List[Dict], session: requests.Session) -> List[Dict[str, Any]]:
    """Parse scoreboards for all 13-0 matches."""
    scoreboards = []
    
    print(f"\n📊 Parsing scoreboards for {len(matches)} matches...\n")
    
    with tqdm(matches, desc="Processing matches", unit="match") as pbar:
        for match_data in pbar:
            match_id = match_data['match_id']
            match_url = match_data['match_url']
            event = match_data['event']
            date = match_data['date']
            
            # Process each 13-0 map in this match
            for map_info in match_data.get('maps', []):
                map_name = map_info['map_name']
                winner = map_info['winner']
                
                # Determine loser team
                team1 = match_data['team1']
                team2 = match_data['team2']
                loser = team2 if winner == team1 else team1
                
                print(f"\n{'='*80}")
                print(f"Match {match_id}: {winner} vs {loser} - {map_name}")
                print(f"{'='*80}")
                
                # Parse scoreboard
                result = parse_scoreboard_for_map(match_url, map_name, winner, loser, session)
                
                if result:
                    scoreboard_data = {
                        "match_id": match_id,
                        "match_url": match_url,
                        "map_name": map_name,
                        "event": event,
                        "date": date,
                        "winner": result["winner"],
                        "loser": result["loser"]
                    }
                    scoreboards.append(scoreboard_data)
                    print(f"  ✓ Successfully parsed scoreboard")
                else:
                    print(f"  ✗ Failed to parse scoreboard")
    
    return scoreboards


def save_scoreboards_to_json(scoreboards: List[Dict[str, Any]], output_file: Path):
    """Save scoreboard data to JSON file."""
    data = {
        "total_maps": len(scoreboards),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scoreboards": scoreboards
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Scoreboard data saved to {output_file}")


def print_summary(scoreboards: List[Dict[str, Any]]):
    """Print a summary of the parsed scoreboards."""
    if not scoreboards:
        print("\n❌ No scoreboard data could be parsed.")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 Parsed {len(scoreboards)} 13-0 map scoreboards")
    print(f"{'='*80}\n")
    
    # Count T side vs CT side wins
    t_side_wins = sum(1 for sb in scoreboards if sb["winner"]["side"] == "T")
    ct_side_wins = sum(1 for sb in scoreboards if sb["winner"]["side"] == "CT")
    
    print(f"T side 13-0 wins: {t_side_wins}")
    print(f"CT side 13-0 wins: {ct_side_wins}")
    print()


def main():
    """Main function."""
    print("="*80)
    print("📊 13-0 Match Raw Scoreboard Parser")
    print("="*80)
    
    # Load matches
    input_path = Path(INPUT_JSON)
    if not input_path.exists():
        print(f"❌ Error: Input file '{INPUT_JSON}' not found.")
        print(f"   Please run find_13_0_matches.py first to generate the match data.")
        return 1
    
    print(f"\n📖 Loading matches from {INPUT_JSON}...")
    matches = load_13_0_matches(input_path)
    
    if not matches:
        print("❌ No matches found in input file.")
        return 1
    
    print(f"✅ Loaded {len(matches)} matches")
    
    # Create session
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en"})
    
    # Parse scoreboards
    try:
        scoreboards = parse_all_scoreboards(matches, session)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display and save results
    print_summary(scoreboards)
    
    if scoreboards:
        output_path = Path(OUTPUT_JSON)
        save_scoreboards_to_json(scoreboards, output_path)
    
    return 0 if scoreboards else 1


if __name__ == "__main__":
    sys.exit(main())
