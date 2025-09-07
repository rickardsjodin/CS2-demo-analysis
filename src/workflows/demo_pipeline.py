#!/usr/bin/env python3
"""
Disk-space efficient CS2 demo processing pipeline.

This module provides a complete workflow for:
1. Downloading demos from HLTV
2. Extracting them from archives
3. Parsing and analyzing them
4. Caching the results
5. Cleaning up temporary files to save disk space
6. Tracking what's been processed to avoid duplicates

Usage:
    python -m src.workflows.demo_pipeline --event 7907 --max-demos 10
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from tqdm import tqdm
# Removed concurrent processing imports for simplicity

# Import our existing modules
from src.data_fetching.fetch_hltv_demos import (
    list_match_pages_for_event, 
    parse_match_for_demos, 
    download_demo,
    DemoRef,
    UA
)
from src.data_fetching.unzip_demos import extract_with_7zip, get_archive_type
from src.utils.cache_utils import load_demo, save_demo_to_cache, get_cache_filename, CACHE_DIR
from src.utils.common import ensure_pandas
import config


@dataclass
class ProcessedDemo:
    """Track information about a processed demo"""
    demo_id: str
    match_title: str
    maps_hint: str
    download_url: str
    cache_file: str
    processed_at: str
    file_size_mb: float
    players: List[str]


class DemoPipeline:
    """Manages the complete demo processing pipeline with disk space optimization"""
    
    def __init__(self, temp_dir: Optional[Path] = None, keep_demos: bool = False):
        """
        Initialize the pipeline.
        
        Args:
            temp_dir: Directory for temporary files (default: system temp)
            keep_demos: Whether to keep the original .dem files after processing
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "cs2_pipeline"
        self.keep_demos = keep_demos
        self.processed_demos_file = config.PROJECT_ROOT / "processed_demos.json"
        self.processed_demos = self._load_processed_demos()
        
        # Create temp directory if it doesn't exist
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup session for downloads
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept-Language": "en"})
    
    def _load_processed_demos(self) -> Dict[str, ProcessedDemo]:
        """Load the registry of already processed demos"""
        if self.processed_demos_file.exists():
            try:
                with open(self.processed_demos_file, 'r') as f:
                    data = json.load(f)
                return {k: ProcessedDemo(**v) for k, v in data.items()}
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                print(f"⚠️ Error loading processed demos registry: {e}")
                print("🔄 Starting with empty registry")
        return {}
    
    def _save_processed_demos(self):
        """Save the registry of processed demos"""
        try:
            data = {k: asdict(v) for k, v in self.processed_demos.items()}
            with open(self.processed_demos_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving processed demos registry: {e}")
    
    def is_demo_processed(self, demo_id: str) -> bool:
        """Check if a demo has already been processed"""
        if demo_id in self.processed_demos:
            # Also verify the cache file still exists
            cache_file = Path(self.processed_demos[demo_id].cache_file)
            if cache_file.exists():
                return True
            else:
                # Cache file was deleted, remove from registry
                print(f"🗑️ Cache file missing for {demo_id}, will reprocess")
                del self.processed_demos[demo_id]
                self._save_processed_demos()
        return False
    
    def get_demo_refs_for_event(self, event_id: int, max_demos: Optional[int] = None) -> List[DemoRef]:
        """Get demo references for an event, filtering out already processed ones"""
        print(f"🔍 Discovering matches for event {event_id}...")
        
        # Get all matches for the event
        matches = list_match_pages_for_event(self.session, event_id, max_demos)
        if not matches:
            print(f"❌ No matches found for event {event_id}")
            return []
        
        print(f"📋 Found {len(matches)} matches")
        
        # Extract demo references
        demos: List[DemoRef] = []
        seen_demo_ids: Set[str] = set()
        
        with tqdm(matches, desc="Scanning for demos", unit="match") as pbar:
            for match in pbar:
                pbar.set_postfix_str(match.label)
                try:
                    demo_ref = parse_match_for_demos(self.session, match.url)
                    if demo_ref and demo_ref.demo_id not in seen_demo_ids:
                        # Check if already processed
                        if not self.is_demo_processed(demo_ref.demo_id):
                            demos.append(demo_ref)
                        else:
                            print(f"⏭️ Skipping already processed demo: {demo_ref.demo_id}")
                        seen_demo_ids.add(demo_ref.demo_id)
                    time.sleep(0.5)  # Be polite to HLTV
                except Exception as e:
                    print(f"⚠️ Error processing match {match.url}: {e}")
                    continue
        
        print(f"🎯 Found {len(demos)} new demos to process")
        return demos
    
    def download_and_extract_demo(self, demo_ref: DemoRef) -> Optional[Path]:
        """Download and extract a demo, returning the path to the .dem file"""
        temp_archive_dir = self.temp_dir / "archives"
        temp_demo_dir = self.temp_dir / "demos"
        
        temp_archive_dir.mkdir(exist_ok=True)
        temp_demo_dir.mkdir(exist_ok=True)
        
        try:
            # Download the archive
            print(f"⬇️ Downloading {demo_ref.match_title}...")
            archive_path, _ = download_demo(self.session, demo_ref, temp_archive_dir, overwrite=True)
            
            # Determine archive type and extract
            print(f"📦 Extracting {archive_path.name}...")
            archive_type = get_archive_type(archive_path)
            
            if archive_type == 'unknown':
                print(f"❌ Unknown archive type for {archive_path.name}")
                return None
            
            # Extract using the appropriate method
            dem_files_data = []
            
            if archive_type in ['zip', '7z', 'rar']:
                try:
                    if archive_type == 'zip':
                        import zipfile
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            dem_files = [f for f in zip_ref.namelist() if f.endswith('.dem')]
                            for dem_file in dem_files:
                                dem_content = zip_ref.read(dem_file)
                                dem_files_data.append((Path(dem_file).name, dem_content))
                    
                    elif archive_type == '7z':
                        import py7zr
                        with py7zr.SevenZipFile(archive_path, 'r') as archive:
                            all_files = archive.getnames()
                            dem_files = [f for f in all_files if f.endswith('.dem')]
                            if dem_files:
                                bio_dict = archive.read(targets=dem_files)
                                for dem_file in dem_files:
                                    if dem_file in bio_dict:
                                        dem_content = bio_dict[dem_file].read()
                                        dem_files_data.append((Path(dem_file).name, dem_content))
                    
                    elif archive_type == 'rar':
                        # Use 7zip for RAR files
                        dem_files_data = extract_with_7zip(archive_path)
                
                except Exception as e:
                    print(f"❌ Error extracting {archive_path.name}: {e}")
                    return None
            
            if not dem_files_data:
                print(f"❌ No .dem files found in {archive_path.name}")
                return None
            
            # Save the first (usually only) .dem file
            dem_filename, dem_content = dem_files_data[0]
            demo_path = temp_demo_dir / f"{demo_ref.match_title}_{demo_ref.demo_id}.dem"
            
            with open(demo_path, 'wb') as f:
                f.write(dem_content)
            
            print(f"✅ Extracted to {demo_path.name}")
            
            # Clean up the archive immediately to save space
            try:
                archive_path.unlink()
                print(f"🗑️ Removed archive {archive_path.name}")
            except Exception as e:
                print(f"⚠️ Could not remove archive: {e}")
            
            return demo_path
        
        except Exception as e:
            print(f"❌ Error downloading/extracting {demo_ref.match_title}: {e}")
            return None
    
    def process_demo(self, demo_path: Path, demo_ref: DemoRef) -> bool:
        """Process a demo file: parse, analyze, cache, and optionally clean up"""
        try:
            print(f"🎮 Processing {demo_path.name}...")
            
            # Load and parse the demo
            demo = load_demo(str(demo_path), use_cache=False)  # Force parsing since we just extracted it
            
            if demo is None:
                print(f"❌ Failed to parse {demo_path.name}")
                return False
            
            # Get basic info about the demo
            players = []
            if hasattr(demo, 'kills') and demo.kills is not None:
                try:
                    # Convert Polars to pandas if needed
                    kills_df = ensure_pandas(demo.kills)
                    
                    if len(kills_df) > 0:
                        attackers = kills_df['attacker_name'].dropna().tolist()
                        victims = kills_df['victim_name'].dropna().tolist()
                        players = sorted(set(attackers + victims))
                        
                except Exception as e:
                    print(f"⚠️ Could not extract players: {e}")
                    # Try alternative column names as fallback
                    try:
                        if hasattr(demo.kills, 'columns'):
                            print(f"Available columns: {list(demo.kills.columns)}")
                    except:
                        pass
                    players = []
            
            # Save to cache
            save_demo_to_cache(demo, str(demo_path))
            
            # Get cache file info
            cache_file = get_cache_filename(str(demo_path))
            if cache_file and cache_file.exists():
                cache_size_mb = cache_file.stat().st_size / 1024 / 1024
                
                # Record the processed demo
                processed_demo = ProcessedDemo(
                    demo_id=demo_ref.demo_id,
                    match_title=demo_ref.match_title,
                    maps_hint=demo_ref.maps_hint,
                    download_url=demo_ref.demo_url,
                    cache_file=str(cache_file),
                    processed_at=datetime.now().isoformat(),
                    file_size_mb=cache_size_mb,
                    players=players
                )
                
                self.processed_demos[demo_ref.demo_id] = processed_demo
                self._save_processed_demos()
                
                print(f"✅ Cached {demo_path.name} ({cache_size_mb:.1f} MB)")
                print(f"👥 Players: {', '.join(players[:5])}{'...' if len(players) > 5 else ''}")
                
                # Clean up the demo file unless we want to keep it
                if not self.keep_demos:
                    try:
                        demo_path.unlink()
                        print(f"🗑️ Removed demo file {demo_path.name}")
                    except Exception as e:
                        print(f"⚠️ Could not remove demo file: {e}")
                
                return True
            else:
                print(f"❌ Failed to create cache file for {demo_path.name}")
                return False
        
        except Exception as e:
            print(f"❌ Error processing {demo_path.name}: {e}")
            return False
    
    def download_and_extract_demo(self, demo_ref: DemoRef) -> Optional[Path]:
        """Download and extract a demo, returning the path to the .dem file"""
        temp_archive_dir = self.temp_dir / "archives"
        temp_demo_dir = self.temp_dir / "demos"
        
        temp_archive_dir.mkdir(exist_ok=True)
        temp_demo_dir.mkdir(exist_ok=True)
        
        try:
            # Download the archive
            print(f"⬇️ Downloading {demo_ref.match_title}...")
            archive_path, _ = download_demo(self.session, demo_ref, temp_archive_dir, overwrite=True)
            
            # Determine archive type and extract
            print(f"📦 Extracting {archive_path.name}...")
            archive_type = get_archive_type(archive_path)
            
            if archive_type == 'unknown':
                print(f"❌ Unknown archive type for {archive_path.name}")
                return None
            
            # Extract using the appropriate method
            dem_files_data = []
            
            if archive_type in ['zip', '7z', 'rar']:
                try:
                    if archive_type == 'zip':
                        import zipfile
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            dem_files = [f for f in zip_ref.namelist() if f.endswith('.dem')]
                            for dem_file in dem_files:
                                dem_content = zip_ref.read(dem_file)
                                dem_files_data.append((Path(dem_file).name, dem_content))
                    
                    elif archive_type == '7z':
                        import py7zr
                        with py7zr.SevenZipFile(archive_path, 'r') as archive:
                            all_files = archive.getnames()
                            dem_files = [f for f in all_files if f.endswith('.dem')]
                            if dem_files:
                                bio_dict = archive.read(targets=dem_files)
                                for dem_file in dem_files:
                                    if dem_file in bio_dict:
                                        dem_content = bio_dict[dem_file].read()
                                        dem_files_data.append((Path(dem_file).name, dem_content))
                    
                    elif archive_type == 'rar':
                        # Use 7zip for RAR files
                        dem_files_data = extract_with_7zip(archive_path)
                
                except Exception as e:
                    print(f"❌ Error extracting {archive_path.name}: {e}")
                    return None
            
            if not dem_files_data:
                print(f"❌ No .dem files found in {archive_path.name}")
                return None
            
            # Save the first (usually only) .dem file
            dem_filename, dem_content = dem_files_data[0]
            demo_path = temp_demo_dir / f"{demo_ref.match_title}_{demo_ref.demo_id}.dem"
            
            with open(demo_path, 'wb') as f:
                f.write(dem_content)
            
            # Verify the file was written successfully
            if not demo_path.exists() or demo_path.stat().st_size == 0:
                print(f"❌ Failed to write demo file")
                return None
            
            print(f"✅ Extracted ({demo_path.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Clean up the archive immediately to save space
            try:
                archive_path.unlink()
                print(f"🗑️ Removed archive {archive_path.name}")
            except Exception as e:
                print(f"⚠️ Could not remove archive: {e}")
            
            return demo_path
        
        except Exception as e:
            print(f"❌ Error downloading/extracting {demo_ref.match_title}: {e}")
            return None
    
    def process_event(self, event_id: int, max_demos: Optional[int] = None) -> int:
        """Process all demos for an event sequentially, returning the number successfully processed"""
        print(f"🚀 Starting demo pipeline for event {event_id}")
        print(f"📁 Cache directory: {CACHE_DIR}")
        print(f"🗂️ Temp directory: {self.temp_dir}")
        print(f"💾 Keep demos: {self.keep_demos}")
        print()
        
        # Get demo references
        demo_refs = self.get_demo_refs_for_event(event_id, max_demos)
        
        if not demo_refs:
            print("❌ No new demos to process")
            return 0
        
        print(f"📋 Processing {len(demo_refs)} demos sequentially...")
        print()
        
        # Process demos one by one sequentially
        processed_count = 0
        failed_count = 0
        
        with tqdm(total=len(demo_refs), desc="Processing demos", unit="demo") as pbar:
            for i, demo_ref in enumerate(demo_refs):
                try:
                    print(f"\n🎮 [{i+1}/{len(demo_refs)}] Processing {demo_ref.match_title}")
                    
                    # Download and extract the demo
                    demo_path = self.download_and_extract_demo(demo_ref)
                    if demo_path is None:
                        failed_count += 1
                        pbar.set_postfix_str(f"✅ {processed_count}/{len(demo_refs)} success, ❌ {failed_count} failed")
                        pbar.update(1)
                        continue
                    
                    # Process the demo (parse, cache, cleanup)
                    success = self.process_demo(demo_path, demo_ref)
                    if success:
                        processed_count += 1
                        pbar.set_postfix_str(f"✅ {processed_count}/{len(demo_refs)} success, ❌ {failed_count} failed")
                    else:
                        failed_count += 1
                        pbar.set_postfix_str(f"✅ {processed_count}/{len(demo_refs)} success, ❌ {failed_count} failed")
                        
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Exception processing {demo_ref.match_title}: {e}")
                    pbar.set_postfix_str(f"✅ {processed_count}/{len(demo_refs)} success, ❌ {failed_count} failed")
                
                pbar.update(1)
        
        print(f"\n🎉 Event {event_id} complete!")
        print(f"📊 Results: {processed_count}/{len(demo_refs)} demos processed successfully")
        if processed_count < len(demo_refs):
            failed_count = len(demo_refs) - processed_count
            success_rate = (processed_count / len(demo_refs)) * 100
            print(f"⚠️ {failed_count} demos failed (success rate: {success_rate:.1f}%)")
        
        return processed_count
    
    def process_events(self, event_ids: List[int], max_demos: Optional[int] = None) -> Dict[int, int]:
        """
        Process demos from multiple events.
        
        Args:
            event_ids: List of HLTV event IDs to process
            max_demos: Maximum demos per event (applies to each event individually)
            
        Returns:
            Dictionary mapping event_id to number of successfully processed demos
        """
        print(f"🚀 Starting multi-event demo pipeline")
        print(f"📅 Events to process: {event_ids}")
        print(f"📊 Max demos per event: {max_demos or 'unlimited'}")
        print()
        
        results = {}
        total_processed = 0
        
        for i, event_id in enumerate(event_ids, 1):
            print(f"\n{'='*60}")
            print(f"📊 Event {i}/{len(event_ids)}: {event_id}")
            print(f"{'='*60}")
            
            try:
                processed_count = self.process_event(event_id, max_demos)
                results[event_id] = processed_count
                total_processed += processed_count
                
                print(f"✅ Event {event_id}: {processed_count} demos processed")
                
            except Exception as e:
                print(f"❌ Error processing event {event_id}: {e}")
                results[event_id] = 0
        
        print(f"\n{'='*60}")
        print(f"🎉 Multi-event pipeline complete!")
        print(f"📊 Total demos processed: {total_processed}")
        print(f"📈 Results by event:")
        for event_id, count in results.items():
            print(f"   Event {event_id}: {count} demos")
        print(f"{'='*60}")
        
        # Clean up temp directory after all events
        self._cleanup_temp_dir()
        
        return results
    
    def _cleanup_temp_dir(self):
        """Clean up the temporary directory"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean up temp directory: {e}")
    
    def list_processed_demos(self):
        """List all processed demos with their information"""
        if not self.processed_demos:
            print("ℹ️ No processed demos found")
            return
        
        print(f"📊 Processed Demos ({len(self.processed_demos)} total):")
        print("="*80)
        
        total_size = 0
        for demo_id, demo_info in self.processed_demos.items():
            print(f"🎮 {demo_info.match_title}")
            print(f"   ID: {demo_id}")
            print(f"   Maps: {demo_info.maps_hint}")
            print(f"   Cache: {demo_info.file_size_mb:.1f} MB")
            print(f"   Players: {len(demo_info.players)}")
            print(f"   Processed: {demo_info.processed_at}")
            
            # Check if cache file still exists
            cache_path = Path(demo_info.cache_file)
            if cache_path.exists():
                total_size += demo_info.file_size_mb
                print(f"   Status: ✅ Available")
            else:
                print(f"   Status: ❌ Cache missing")
            print()
        
        print(f"💾 Total cache size: {total_size:.1f} MB")
    
    def analyze_cached_demo(self, demo_id: str, player_name: str):
        """Load a cached demo and return basic info for ML training data"""
        if demo_id not in self.processed_demos:
            print(f"❌ Demo {demo_id} not found in processed demos")
            return None
        
        demo_info = self.processed_demos[demo_id]
        cache_file = Path(demo_info.cache_file)
        
        if not cache_file.exists():
            print(f"❌ Cache file missing for demo {demo_id}")
            return None
        
        print(f"🎮 Loading {demo_info.match_title} for ML training data")
        
        # Load the cached demo
        # Create a dummy path that will trigger cache loading
        dummy_path = str(cache_file).replace('.pkl', '.dem')
        demo = load_demo(dummy_path, use_cache=True)
        
        if demo is None:
            print(f"❌ Failed to load cached demo")
            return None
        
        print(f"✅ Demo loaded successfully for ML training data")
        print(f"📊 Available data: kills, rounds, damages, grenades, etc.")
        print(f"👥 Players: {', '.join(demo_info.players)}")
        
        return demo
    
    def get_training_data(self, demo_ids: List[str] = None) -> List[dict]:
        """
        Get training data from cached demos.
        
        Args:
            demo_ids: List of specific demo IDs to load. If None, loads all cached demos.
            
        Returns:
            List of demo data dictionaries ready for ML training
        """
        if demo_ids is None:
            demo_ids = list(self.processed_demos.keys())
        
        training_data = []
        
        for demo_id in demo_ids:
            if demo_id not in self.processed_demos:
                print(f"⚠️ Demo {demo_id} not found in processed demos")
                continue
            
            demo_info = self.processed_demos[demo_id]
            cache_file = Path(demo_info.cache_file)
            
            if not cache_file.exists():
                print(f"⚠️ Cache file missing for demo {demo_id}")
                continue
            
            # Create a dummy path that will trigger cache loading
            dummy_path = str(cache_file).replace('.pkl', '.dem')
            demo = load_demo(dummy_path, use_cache=True)
            
            if demo is None:
                print(f"⚠️ Failed to load demo {demo_id}")
                continue
            
            # Package the demo data for ML training
            demo_data = {
                'demo_id': demo_id,
                'match_title': demo_info.match_title,
                'maps_hint': demo_info.maps_hint,
                'players': demo_info.players,
                'processed_at': demo_info.processed_at,
                'demo_object': demo,  # The actual parsed demo object
                'kills': demo.kills if hasattr(demo, 'kills') else None,
                'rounds': demo.rounds if hasattr(demo, 'rounds') else None,
                'damages': demo.damages if hasattr(demo, 'damages') else None,
                'smokes': demo.smokes if hasattr(demo, 'smokes') else None,
                'flashes': demo.flashes if hasattr(demo, 'flashes') else None,
                'grenades': demo.grenades if hasattr(demo, 'grenades') else None,
                'bomb': demo.bomb if hasattr(demo, 'bomb') else None,
                'ticks': demo.ticks if hasattr(demo, 'ticks') else None
            }
            
            training_data.append(demo_data)
            print(f"✅ Loaded training data for {demo_info.match_title}")
        
        print(f"🤖 Training data ready: {len(training_data)} demos loaded")
        return training_data


def main():
    """Command line interface for the demo pipeline"""
    parser = argparse.ArgumentParser(description="CS2 Demo Processing Pipeline")
    
    # Event specification - either single event or multiple events
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event", type=int, help="Single HLTV event ID")
    group.add_argument("--events", type=int, nargs='+', help="Multiple HLTV event IDs (space-separated)")
    
    parser.add_argument("--max-demos", type=int, help="Maximum number of demos per event")
    parser.add_argument("--temp-dir", type=Path, help="Temporary directory for downloads")
    parser.add_argument("--keep-demos", action="store_true", help="Keep original .dem files after processing")

    parser.add_argument("--list", action="store_true", help="List processed demos and exit")
    parser.add_argument("--load", help="Load a specific demo for ML training (provide demo_id)")
    parser.add_argument("--training-data", action="store_true", help="Get all training data from cached demos")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DemoPipeline(temp_dir=args.temp_dir, keep_demos=args.keep_demos)
    
    if args.list:
        pipeline.list_processed_demos()
        return
    
    if args.load:
        demo = pipeline.analyze_cached_demo(args.load, "")
        if demo:
            print("🤖 Demo ready for ML training data extraction")
        return
    
    if args.training_data:
        training_data = pipeline.get_training_data()
        print(f"🤖 Retrieved {len(training_data)} demos for ML training")
        return
    
    # Process event(s)
    if args.event:
        # Single event
        pipeline.process_event(args.event, args.max_demos)
    elif args.events:
        # Multiple events
        pipeline.process_events(args.events, args.max_demos)


if __name__ == "__main__":
    main()
