#!/usr/bin/env python3
"""
Find HLTV matches with 13-0 map scores.

Searches through HLTV match results and identifies maps where one team
won with a perfect 13-0 final score (won all rounds without the opponent scoring).

Configuration:
  All settings are configured at the top of this file. Edit the configuration
  variables and then run the script:
  
  python find_13_0_matches.py

Configuration Options:
  - SEARCH_MODE: "event" or "date_range"
  - EVENT_ID: HLTV event ID (when using "event" mode)
  - FROM_DATE/TO_DATE: Date range (when using "date_range" mode)
  - MAX_MATCHES: Maximum number of matches to analyze
  - SAVE_TO_JSON: Whether to save results to JSON
  - OUTPUT_FILE: Output filename for JSON results
"""

from __future__ import annotations
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HLTV = "https://www.hltv.org"
RESULTS_URL = HLTV + "/results"
UA = "CS2-research/0.1 (contact: rickard.sjodin5@gmail.com; research only)"

# --- Configuration -----------------------------------------------------------

# HTTP request settings
REQUEST_TIMEOUT = 20  # seconds
MAX_RETRIES = 6
BASE_BACKOFF_DELAY = 0.5  # seconds
MAX_BACKOFF_DELAY = 4.0  # seconds

# Rate limiting (be polite to HLTV)
DELAY_BETWEEN_PAGES = 0.8  # seconds
DELAY_BETWEEN_MATCHES = 0.5  # seconds

# Pagination settings
RESULTS_PER_PAGE = 100  # HLTV typically uses 100 results per page
MAX_PAGES_TO_FETCH = 10  # Maximum number of pages to fetch (safety limit)
# Note: 50 pages × 100 matches/page = 5000 matches maximum

# CS2 game format (MR12)
ROUNDS_PER_REGULATION_HALF = 12  # Rounds per half in regulation
ROUNDS_PER_OT_HALF = 6  # Rounds per half in overtime
TOTAL_REGULATION_ROUNDS = 24  # Total rounds in regulation (2 halves)

# Score thresholds
PERFECT_HALF_SCORE = 13  # What constitutes a "perfect" half (13-0)
ZERO_ROUNDS = 0

# --- Search Configuration (EDIT THESE TO RUN THE SCRIPT) --------------------

# Search mode: "event" or "date_range"
SEARCH_MODE = "date_range"  # Options: "event", "date_range"

# Event search settings (used when SEARCH_MODE = "event")
EVENT_ID = 7907  # HLTV event ID to search

# Date range search settings (used when SEARCH_MODE = "date_range")
FROM_DATE = "2024-12-01"  # Start date (YYYY-MM-DD) or None for no start limit
TO_DATE = "2024-12-31"    # End date (YYYY-MM-DD) or None for no end limit

# General search settings
MAX_MATCHES = None  # Maximum number of matches to analyze (None for unlimited)

# Output settings
SAVE_TO_JSON = True  # Whether to save results to JSON file
OUTPUT_FILE = "13_0_matches.json"  # Output filename (relative or absolute path)

# --- Data structures ---------------------------------------------------------

@dataclass
class HalfScore:
    """Represents the score of one half."""
    team1_score: int
    team2_score: int
    
    def is_13_0(self) -> bool:
        """Check if this is a 13-0 half."""
        return (self.team1_score == PERFECT_HALF_SCORE and self.team2_score == ZERO_ROUNDS) or \
               (self.team1_score == ZERO_ROUNDS and self.team2_score == PERFECT_HALF_SCORE)
    
    def get_winner(self) -> Optional[int]:
        """Return 1 if team1 won 13-0, 2 if team2 won 13-0, None otherwise."""
        if self.team1_score == PERFECT_HALF_SCORE and self.team2_score == ZERO_ROUNDS:
            return 1
        elif self.team2_score == PERFECT_HALF_SCORE and self.team1_score == ZERO_ROUNDS:
            return 2
        return None

@dataclass
class MapScore:
    """Represents scores for a single map."""
    map_name: str
    team1_total: int
    team2_total: int
    first_half: Optional[HalfScore] = None
    second_half: Optional[HalfScore] = None
    overtime_halves: List[HalfScore] = None
    
    def __post_init__(self):
        if self.overtime_halves is None:
            self.overtime_halves = []
    
    def is_13_0_map(self) -> bool:
        """Check if this map ended with a 13-0 final score."""
        return (self.team1_total == PERFECT_HALF_SCORE and self.team2_total == ZERO_ROUNDS) or \
               (self.team1_total == ZERO_ROUNDS and self.team2_total == PERFECT_HALF_SCORE)
    
    def get_map_winner(self) -> Optional[int]:
        """Return 1 if team1 won 13-0, 2 if team2 won 13-0, None otherwise."""
        if self.team1_total == PERFECT_HALF_SCORE and self.team2_total == ZERO_ROUNDS:
            return 1
        elif self.team2_total == PERFECT_HALF_SCORE and self.team1_total == ZERO_ROUNDS:
            return 2
        return None
    
@dataclass
class Match:
    """Represents a CS2 match with 13-0 map scores."""
    match_id: str
    match_url: str
    team1: str
    team2: str
    date: str
    event: str
    maps: List[MapScore]
    
    def has_13_0_map(self) -> bool:
        """Check if any map in this match ended with a 13-0 final score."""
        return any(map_score.is_13_0_map() for map_score in self.maps)
    
    def get_13_0_summary(self) -> str:
        """Return a human-readable summary of 13-0 maps in this match."""
        lines = [f"{self.team1} vs {self.team2} - {self.event} ({self.date})"]
        lines.append(f"  URL: {self.match_url}")
        
        for map_score in self.maps:
            if map_score.is_13_0_map():
                winner = map_score.get_map_winner()
                winning_team = self.team1 if winner == 1 else self.team2
                lines.append(f"  ⭐ Map: {map_score.map_name} - {winning_team} won 13-0")
        
        return "\n".join(lines)

# --- HTTP helpers ------------------------------------------------------------

def sleep_backoff(try_idx: int) -> None:
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

# --- Scraping functions ------------------------------------------------------

def list_match_urls_for_event(session: requests.Session, event_id: int, 
                               max_matches: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Get list of (match_url, match_id) for an event.
    
    Returns:
        List of tuples (full_match_url, match_id)
    """
    matches = []
    seen_ids = set()
    offset = 0
    pages_fetched = 0
    
    pbar = tqdm(desc=f"Fetching matches for event {event_id}", unit="page", leave=False)
    
    while True:
        # Check if we've hit the page limit
        if pages_fetched >= MAX_PAGES_TO_FETCH:
            pbar.set_postfix_str(f"Reached page limit ({MAX_PAGES_TO_FETCH} pages)")
            break
        
        params = {"event": str(event_id), "offset": str(offset)} if offset else {"event": str(event_id)}
        pbar.set_postfix({"offset": offset, "matches": len(matches), "pages": pages_fetched})
        
        resp = http_get(session, RESULTS_URL, params=params)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find match links
        match_links = soup.select(".result-con a.a-reset[href^='/matches/']")
        
        new_matches = 0
        for link in match_links:
            href = link.get("href", "")
            match_id = href.split("/")[2] if len(href.split("/")) > 2 else ""
            
            if match_id and match_id not in seen_ids:
                seen_ids.add(match_id)
                matches.append((HLTV + href, match_id))
                new_matches += 1
        
        pbar.update(1)
        pages_fetched += 1
        
        if new_matches == 0:
            break
        
        if max_matches and len(matches) >= max_matches:
            matches = matches[:max_matches]
            break
        
        offset += RESULTS_PER_PAGE
        time.sleep(DELAY_BETWEEN_PAGES)
    
    pbar.close()
    return matches

def parse_match_date_from_timestamp(timestamp_ms: int) -> datetime:
    """Convert HLTV timestamp (milliseconds) to datetime object."""
    return datetime.fromtimestamp(timestamp_ms / 1000.0)

def is_date_in_range(match_date: datetime, start_date: Optional[str], end_date: Optional[str]) -> bool:
    """Check if a match date falls within the specified range."""
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        if match_date < start:
            return False
    
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        # Add one day to include matches on end_date
        end = end.replace(hour=23, minute=59, second=59)
        if match_date > end:
            return False
    
    return True

def list_match_urls_for_date_range(session: requests.Session, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   max_matches: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Get list of (match_url, match_id) for a date range.
    
    HLTV doesn't support date filtering in the URL, so we fetch results by offset
    and filter by date based on the data-zonedgrouping-entry-unix attribute.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_matches: Maximum number of matches to fetch
    
    Returns:
        List of tuples (full_match_url, match_id)
    """
    matches = []
    seen_ids = set()
    offset = 0
    pages_fetched = 0
    outside_range_count = 0
    max_outside_range = 200  # Stop if we see 200 consecutive matches outside range
    
    pbar = tqdm(desc=f"Fetching matches", unit="page", leave=False)
    
    while True:
        # Check if we've hit the page limit
        if pages_fetched >= MAX_PAGES_TO_FETCH:
            pbar.set_postfix_str(f"Reached page limit ({MAX_PAGES_TO_FETCH} pages)")
            break
        
        # HLTV only accepts offset parameter, no date filters
        params = {"offset": str(offset)} if offset else {}
        
        pbar.set_postfix({"offset": offset, "matches": len(matches), "outside_range": outside_range_count, "pages": pages_fetched})
        
        resp = http_get(session, RESULTS_URL, params=params)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Find result containers - they have timestamp data
        result_containers = soup.select(".result-con")
        
        if not result_containers:
            break
        
        new_matches = 0
        page_has_matches_in_range = False
        pages_fetched += 1
        
        for result_con in result_containers:
            # Extract timestamp from data attribute
            timestamp_elem = result_con.select_one("[data-zonedgrouping-entry-unix]")
            
            # Get match link
            match_link = result_con.select_one('a.a-reset[href^="/matches/"]')
            if not match_link:
                continue
            
            href = match_link.get("href", "")
            match_id = href.split("/")[2] if len(href.split("/")) > 2 else ""
            
            if not match_id or match_id in seen_ids:
                continue
            
            # Check date range if we have timestamp
            if timestamp_elem and (start_date or end_date):
                try:
                    timestamp = int(timestamp_elem.get("data-zonedgrouping-entry-unix", "0"))
                    match_date = parse_match_date_from_timestamp(timestamp)
                    
                    if not is_date_in_range(match_date, start_date, end_date):
                        outside_range_count += 1
                        # If we have a start date and match is before it, we've gone too far back
                        if start_date and match_date < datetime.strptime(start_date, "%Y-%m-%d"):
                            # We're past the start date, likely no more matches to find
                            if outside_range_count >= max_outside_range:
                                pbar.set_postfix_str("Reached start of date range")
                                pbar.close()
                                return matches
                        continue
                    else:
                        outside_range_count = 0  # Reset counter when we find a match in range
                        page_has_matches_in_range = True
                except (ValueError, TypeError):
                    # If we can't parse the date, include the match anyway
                    pass
            
            seen_ids.add(match_id)
            matches.append((HLTV + href, match_id))
            new_matches += 1
        
        pbar.update(1)
        
        # Stop if we haven't found any new matches
        if new_matches == 0:
            break
        
        # Stop if we've hit the max matches limit
        if max_matches and len(matches) >= max_matches:
            matches = matches[:max_matches]
            break
        
        # If using date filtering and we've gone too far outside range, stop
        if (start_date or end_date) and outside_range_count >= max_outside_range:
            pbar.set_postfix_str("No more matches in date range")
            break
        
        offset += RESULTS_PER_PAGE
        time.sleep(DELAY_BETWEEN_PAGES)
    
    pbar.close()
    return matches

def parse_match_page(session: requests.Session, match_url: str, match_id: str) -> Optional[Match]:
    """
    Parse a match page and extract scores including half scores.
    
    Returns:
        Match object if successfully parsed, None otherwise
    """
    try:
        resp = http_get(session, match_url)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract team names
        team_elements = soup.select(".teamName")
        if len(team_elements) < 2:
            return None
        team1 = team_elements[0].get_text(strip=True)
        team2 = team_elements[1].get_text(strip=True)
        
        # Extract date
        date_elem = soup.select_one(".date")
        date = date_elem.get_text(strip=True) if date_elem else "Unknown"
        
        # Extract event name
        event_elem = soup.select_one(".event a")
        event = event_elem.get_text(strip=True) if event_elem else "Unknown"
        
        # Extract map scores
        maps = []
        map_holders = soup.select(".mapholder")
        
        for map_holder in map_holders:
            # Get map name
            map_name_elem = map_holder.select_one(".mapname")
            if not map_name_elem:
                continue
            map_name = map_name_elem.get_text(strip=True)
            
            # Get total scores
            results_elem = map_holder.select_one(".results")
            if not results_elem:
                continue
            
            # Parse total scores (format: "13" and "7" in separate spans)
            score_spans = results_elem.select(".results-team-score")
            if len(score_spans) < 2:
                continue
            
            try:
                team1_total = int(score_spans[0].get_text(strip=True))
                team2_total = int(score_spans[1].get_text(strip=True))
            except (ValueError, IndexError):
                continue
            
            # Try to extract half scores from the detailed stats
            # HLTV shows half scores in the round history section
            first_half = None
            second_half = None
            overtime_halves = []
            
            # Look for round history divs
            round_history = map_holder.select(".round-history-team")
            if len(round_history) >= 2:
                # Parse round-by-round data
                team1_rounds = round_history[0].select(".round-history-outcome")
                team2_rounds = round_history[1].select(".round-history-outcome")
                
                # Calculate half scores (MR12 format: first 12 rounds per team = regulation)
                # In CS2 MR12: rounds 1-12 are first half, 13-24 are second half
                total_rounds = max(len(team1_rounds), len(team2_rounds))
                
                if total_rounds >= ROUNDS_PER_REGULATION_HALF:
                    # First half (rounds 1-12)
                    team1_first_half = sum(1 for i in range(min(ROUNDS_PER_REGULATION_HALF, len(team1_rounds))) 
                                          if 'won' in team1_rounds[i].get('class', []))
                    team2_first_half = ROUNDS_PER_REGULATION_HALF - team1_first_half  # Assuming all rounds played
                    first_half = HalfScore(team1_first_half, team2_first_half)
                
                if total_rounds >= TOTAL_REGULATION_ROUNDS:
                    # Second half (rounds 13-24)
                    team1_second_half = sum(1 for i in range(ROUNDS_PER_REGULATION_HALF, min(TOTAL_REGULATION_ROUNDS, len(team1_rounds))) 
                                           if 'won' in team1_rounds[i].get('class', []))
                    team2_second_half = min(ROUNDS_PER_REGULATION_HALF, total_rounds - ROUNDS_PER_REGULATION_HALF) - team1_second_half
                    second_half = HalfScore(team1_second_half, team2_second_half)
                
                # Overtime halves (every 6 rounds after round 24)
                if total_rounds > TOTAL_REGULATION_ROUNDS:
                    ot_start = TOTAL_REGULATION_ROUNDS
                    while ot_start < total_rounds:
                        ot_end = min(ot_start + ROUNDS_PER_OT_HALF, total_rounds)
                        team1_ot = sum(1 for i in range(ot_start, min(ot_end, len(team1_rounds))) 
                                      if 'won' in team1_rounds[i].get('class', []))
                        team2_ot = (ot_end - ot_start) - team1_ot
                        overtime_halves.append(HalfScore(team1_ot, team2_ot))
                        ot_start = ot_end
            
            map_score = MapScore(
                map_name=map_name,
                team1_total=team1_total,
                team2_total=team2_total,
                first_half=first_half,
                second_half=second_half,
                overtime_halves=overtime_halves
            )
            maps.append(map_score)
        
        if not maps:
            return None
        
        return Match(
            match_id=match_id,
            match_url=match_url,
            team1=team1,
            team2=team2,
            date=date,
            event=event,
            maps=maps
        )
    
    except Exception as e:
        print(f"Error parsing match {match_url}: {e}", file=sys.stderr)
        return None

def find_13_0_matches_in_event(session: requests.Session, event_id: int, 
                                max_matches: Optional[int] = None) -> List[Match]:
    """
    Find all matches with 13-0 map scores in a specific event.
    
    Returns:
        List of Match objects that contain at least one map with 13-0 final score
    """
    print(f"🔍 Searching event {event_id} for 13-0 map scores...")
    
    # Get match URLs
    match_urls = list_match_urls_for_event(session, event_id, max_matches)
    if not match_urls:
        print("No matches found for this event.")
        return []
    
    print(f"📊 Found {len(match_urls)} matches to analyze")
    
    # Parse each match and check for 13-0 map scores
    matches_with_13_0 = []
    
    with tqdm(match_urls, desc="Analyzing matches", unit="match") as pbar:
        for match_url, match_id in pbar:
            match = parse_match_page(session, match_url, match_id)
            time.sleep(DELAY_BETWEEN_MATCHES)  # Be polite to HLTV
            
            if match and match.has_13_0_map():
                matches_with_13_0.append(match)
                pbar.set_postfix_str(f"Found {len(matches_with_13_0)} 13-0 maps")
    
    return matches_with_13_0

def find_13_0_matches_in_date_range(session: requests.Session, 
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    max_matches: Optional[int] = None) -> List[Match]:
    """
    Find all matches with 13-0 map scores in a date range.
    
    Returns:
        List of Match objects that contain at least one 13-0 map score
    """
    date_str = f"{start_date or 'beginning'} to {end_date or 'today'}"
    print(f"🔍 Searching matches from {date_str} for 13-0 scores...")
    
    # Get match URLs
    match_urls = list_match_urls_for_date_range(session, start_date, end_date, max_matches)
    if not match_urls:
        print("No matches found for this date range.")
        return []
    
    print(f"📊 Found {len(match_urls)} matches to analyze")
    
    # Parse each match and check for 13-0 map scores
    matches_with_13_0 = []
    
    with tqdm(match_urls, desc="Analyzing matches", unit="match") as pbar:
        for match_url, match_id in pbar:
            match = parse_match_page(session, match_url, match_id)
            time.sleep(DELAY_BETWEEN_MATCHES)  # Be polite to HLTV
            
            if match and match.has_13_0_map():
                matches_with_13_0.append(match)
                pbar.set_postfix_str(f"Found {len(matches_with_13_0)} 13-0 maps")
    
    return matches_with_13_0

# --- Output functions --------------------------------------------------------

def save_results_to_json(matches: List[Match], output_file: Path):
    """Save match results to JSON file."""
    data = {
        "total_matches_with_13_0_maps": len(matches),
        "total_13_0_maps": sum(sum(1 for map_score in match.maps if map_score.is_13_0_map()) for match in matches),
        "generated_at": datetime.now().isoformat(),
        "matches": []
    }
    
    for match in matches:
        match_data = {
            "match_id": match.match_id,
            "match_url": match.match_url,
            "team1": match.team1,
            "team2": match.team2,
            "date": match.date,
            "event": match.event,
            "maps": []
        }
        
        for map_score in match.maps:
            if map_score.is_13_0_map():
                winner = map_score.get_map_winner()
                winning_team = match.team1 if winner == 1 else match.team2
                
                map_data = {
                    "map_name": map_score.map_name,
                    "final_score": f"{map_score.team1_total}-{map_score.team2_total}",
                    "winner": winning_team
                }
                
                match_data["maps"].append(map_data)
        
        data["matches"].append(match_data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to {output_file}")

def print_results_summary(matches: List[Match]):
    """Print a summary of matches with 13-0 map scores."""
    if not matches:
        print("\n❌ No matches with 13-0 map scores found.")
        return
    
    # Count total 13-0 maps
    total_13_0_maps = sum(sum(1 for map_score in match.maps if map_score.is_13_0_map()) for match in matches)
    
    print(f"\n{'='*80}")
    print(f"🎯 Found {len(matches)} matches containing {total_13_0_maps} maps with 13-0 scores!")
    print(f"{'='*80}\n")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match.get_13_0_summary()}")
        print()

# --- Main --------------------------------------------------------------------

def main():
    """
    Main function - reads configuration from top of file and runs the search.
    
    To use this script:
    1. Edit the configuration variables at the top of the file
    2. Run: python find_13_0_matches.py
    """
    print("="*80)
    print("🔍 HLTV 13-0 Match Finder")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Search mode: {SEARCH_MODE}")
    
    if SEARCH_MODE == "event":
        print(f"   Event ID: {EVENT_ID}")
    elif SEARCH_MODE == "date_range":
        print(f"   Date range: {FROM_DATE or 'beginning'} to {TO_DATE or 'today'}")
    
    print(f"   Max matches: {MAX_MATCHES or 'unlimited'}")
    print(f"   Save to JSON: {SAVE_TO_JSON}")
    if SAVE_TO_JSON:
        print(f"   Output file: {OUTPUT_FILE}")
    print()
    
    # Validate configuration
    if SEARCH_MODE not in ["event", "date_range"]:
        print("❌ Error: SEARCH_MODE must be either 'event' or 'date_range'")
        print("   Please edit the configuration at the top of the file.")
        return 1
    
    if SEARCH_MODE == "event" and not EVENT_ID:
        print("❌ Error: EVENT_ID must be set when SEARCH_MODE is 'event'")
        print("   Please edit the configuration at the top of the file.")
        return 1
    
    # Create session
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en"})
    
    # Find matches based on search mode
    matches_with_13_0 = []
    
    try:
        if SEARCH_MODE == "event":
            matches_with_13_0 = find_13_0_matches_in_event(session, EVENT_ID, MAX_MATCHES)
        elif SEARCH_MODE == "date_range":
            matches_with_13_0 = find_13_0_matches_in_date_range(
                session, FROM_DATE, TO_DATE, MAX_MATCHES
            )
    except KeyboardInterrupt:
        print("\n\n⚠️  Search interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during search: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display results
    print_results_summary(matches_with_13_0)
    
    # Save to file if configured
    if SAVE_TO_JSON and matches_with_13_0:
        output_path = Path(OUTPUT_FILE)
        save_results_to_json(matches_with_13_0, output_path)
    
    # Return exit code based on results
    return 0 if matches_with_13_0 else 1

if __name__ == "__main__":
    sys.exit(main())
