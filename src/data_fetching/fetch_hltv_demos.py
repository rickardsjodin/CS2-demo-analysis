#!/usr/bin/env python3
"""
Fetch HLTV pro match demos by event.

- Enumerates finished matches from https://www.hltv.org/results?event=<EVENT_ID>&offset=<...>
- Opens each match page and extracts one or more demo download links (/download/demo/<id>)
- Streams the resulting .zip to disk with a descriptive filename
- Polite: identifies as a simple research UA, rate-limits, retries, and de-duplicates

Usage:
  python fetch_hltv_demos.py --event 7907 --out ./demos --concurrency 4 --max 50
"""

from __future__ import annotations
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HLTV = "https://www.hltv.org"
RESULTS_URL = HLTV + "/results"
MATCH_PATH_RE = re.compile(r"^/matches/\d+/")
DEMO_PATH_RE = re.compile(r"^/download/demo/\d+")
UA = "CS2-training/0.1 (contact: rickard.sjodin5@gmail.com; experimental only)"

# --- tiny helpers ------------------------------------------------------------

def sleep_backoff(try_idx: int) -> None:
    # 0.5s, 1s, 2s, 4s ... (cap)
    time.sleep(min(4.0, 0.5 * (2 ** try_idx)))

def http_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    for i in range(6):
        try:
            resp = session.get(url, timeout=20, **kwargs)
            if resp.status_code in (429, 503, 502, 500):
                sleep_backoff(i)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            sleep_backoff(i)
    raise RuntimeError(f"Failed GET after retries: {url}")

def slugify(text: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", text.strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_")

@dataclass(frozen=True)
class MatchRef:
    url: str  # absolute
    label: str

@dataclass
class DemoRef:
    demo_url: str     # absolute /download/demo/<id>
    demo_id: str      # numeric string
    match_title: str  # e.g., "MOUZ_vs_G2_BLAST_Open_London_2025"
    maps_hint: str    # "Mirage_Nuke" if we can find it, else ""

# --- scraping ----------------------------------------------------------------

def list_match_pages_for_event(session: requests.Session, event_id: int, max_matches: Optional[int]) -> List[MatchRef]:
    """Paginate results?event=<id> with offset=0,100,200,... collect finished match page URLs."""
    out: List[MatchRef] = []
    seen_hrefs = set()
    offset = 0
    pbar = tqdm(desc="Enumerating matches", unit="page", leave=False)
    while True:
        params = {"event": str(event_id), "offset": str(offset)} if offset else {"event": str(event_id)}
        pbar.set_postfix({"url": f"{RESULTS_URL}?event={event_id}&offset={offset}"})
        resp = http_get(session, RESULTS_URL, params=params)
        soup = BeautifulSoup(resp.text, "html.parser")

        # HLTV results pages have two layouts:
        # 1. Event page: `div.result-con > a.a-reset`
        # 2. Generic results: `div.result-con > table` with a `td.event` column.
        # When we paginate past the end of an event's results, HLTV returns the generic layout.
        # We need to handle both and filter matches for our event ID on generic pages.
        hrefs_on_page = []
        result_containers = soup.select(".result-con")
        is_generic_page = soup.select_one(".result-con td.event") is not None

        if is_generic_page:
            for rcon in result_containers:
                # Ensure this match is for the event we're scraping
                if rcon.select_one(f'td.event a[href*="/events/{event_id}/"]'):
                    match_link = rcon.select_one('a.a-reset[href^="/matches/"]')
                    if match_link:
                        hrefs_on_page.append(match_link["href"])
        else:
            # This is an event-specific page, all matches are for our event.
            for rcon in result_containers:
                match_link = rcon.select_one('a.a-reset[href^="/matches/"]')
                if match_link:
                    hrefs_on_page.append(match_link["href"])

        new_hrefs = []
        for h in hrefs_on_page:
            if h not in seen_hrefs:
                seen_hrefs.add(h)
                new_hrefs.append(h)

        if new_hrefs:
            pbar.set_postfix_str(f"found {len(new_hrefs)} new matches on page")

        # convert to absolute URLs
        for h in new_hrefs:
            out.append(MatchRef(url=HLTV + h, label=h))

        pbar.update(1)
        # Stop if no new matches found on this page
        if not new_hrefs:
            break
        offset += 100
        if max_matches and len(out) >= max_matches:
            out = out[:max_matches]
            break
        # Be polite between pages
        time.sleep(0.8)
    pbar.close()
    return out

def parse_match_for_demos(session: requests.Session, match_url: str) -> DemoRef | None:
    """Open a finished match page and extract the first /download/demo/<id> (covers whole match)."""
    resp = http_get(session, match_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title pieces we can use for filenames (teams, event, date)
    title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else "match"
    title_slug = slugify(title.replace(" | HLTV.org", ""))

    # Optional: map names present in the 'Rewatch' section
    maps_hint = "_".join(sorted({img.get("alt", "") for img in soup.select('img[alt]') if img.get("alt","") in {
        "Mirage","Nuke","Inferno","Overpass","Ancient","Vertigo","Anubis","Dust2","Train","Tuscan","Cobblestone"
    }})) or ""

    # Demo link: "Click here if your download does not start" points to /download/demo/<id>
    a = soup.find("a", href=DEMO_PATH_RE)
    if not a:
        return None
    demo_path = a["href"]
    demo_id = demo_path.rsplit("/", 1)[-1]
    return DemoRef(demo_url=HLTV + demo_path, demo_id=demo_id, match_title=title_slug, maps_hint=maps_hint)

def choose_filename(demo: DemoRef, out_dir: Path, content_disp: Optional[str]) -> Path:
    # Try to honor server-suggested filename if present
    if content_disp and "filename=" in content_disp:
        fname = content_disp.split("filename=", 1)[-1].strip().strip('"')
        fname = slugify(fname)
    else:
        base = demo.match_title
        if demo.maps_hint:
            base += f"__{demo.maps_hint}"
        base += f"__demo_{demo.demo_id}.zip"
        fname = base
    return out_dir / fname

def download_demo(session: requests.Session, demo: DemoRef, out_dir: Path, overwrite: bool=False) -> Tuple[Path, str]:
    """Follow the /download/demo/<id> redirect and stream to file."""
    # First request usually lands on a meta page that 302s to static storage
    with session.get(demo.demo_url, stream=True, allow_redirects=True, timeout=30) as r:
        r.raise_for_status()
        cd = r.headers.get("Content-Disposition", "")
        target = choose_filename(demo, out_dir, cd)

        if target.exists() and not overwrite:
            return target, "exists"

        tmp = target.with_suffix(target.suffix + ".part")
        hasher = hashlib.sha256()
        size = 0
        chunk = 1 << 20  # 1 MiB
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f, tqdm(total=int(r.headers.get("Content-Length") or 0),
                                        unit="B", unit_scale=True, desc=target.name, leave=False) as bar:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if not chunk_bytes:
                    continue
                f.write(chunk_bytes)
                hasher.update(chunk_bytes)
                size += len(chunk_bytes)
                bar.update(len(chunk_bytes))
        os.replace(tmp, target)
    return target, hasher.hexdigest()

# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Download HLTV demos for a given event ID.")
    ap.add_argument("--event", type=int, required=True, help="HLTV event id (see /events/<id>/...)")
    ap.add_argument("--out", type=Path, default=Path("F://CS2/demos_zipped"), help="Output directory")
    ap.add_argument("--concurrency", type=int, default=1, help="Parallel downloads (be polite; 2–6 is fine)")
    ap.add_argument("--max", type=int, default=None, help="Max matches to process (useful for testing)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files if they exist")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en"})

    # Step 1: discover match pages for this event
    matches = list_match_pages_for_event(session, args.event, args.max)
    if not matches:
        print("No matches found (event may be ongoing or has no results yet).")
        return

    # Step 2: extract demo links
    demos: List[DemoRef] = []
    seen_demo_ids: Set[str] = set()
    with tqdm(matches, desc="Scanning matches", unit="match") as pbar:
        for m in pbar:
            pbar.set_postfix_str(m.label)
            d = parse_match_for_demos(session, m.url)
            time.sleep(0.5)  # polite
            if d and d.demo_id not in seen_demo_ids:
                demos.append(d)
                seen_demo_ids.add(d.demo_id)

    if not demos:
        print("No demo downloads found yet for this event (try again later).")
        return

    # Step 3: download in parallel
    event_out_dir = args.out / str(args.event)
    event_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(demos)} demo zip(s) to {event_out_dir.resolve()}")

    def _worker(dr: DemoRef):
        try:
            return download_demo(session, dr, event_out_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"Error downloading {dr.demo_url}: {e}", file=sys.stderr)
            return None

    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        for res in tqdm(ex.map(_worker, demos), total=len(demos), desc="Downloads"):
            pass

    print("Done.")

if __name__ == "__main__":
    sys.exit(main())
