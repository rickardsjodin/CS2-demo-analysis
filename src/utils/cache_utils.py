"""
Demo caching utilities for CS2 analysis
"""

import os
import pickle
import hashlib
from pathlib import Path
from awpy import Demo


# Cache settings - always use project root cache directory
def get_project_root():
    """Get the project root directory"""
    current_file = Path(__file__).resolve()
    # Navigate up from src/utils/cache_utils.py to project root
    return current_file.parent.parent.parent

CACHE_DIR = Path("F:\\CS2\\cache")


def get_cache_filename(demo_file):
    """Generate a unique cache filename based on demo file name and modification time"""
    demo_path = Path(demo_file)
    if not demo_path.exists():
        return None
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    # Get file modification time and size for uniqueness
    mtime = demo_path.stat().st_mtime
    size = demo_path.stat().st_size
    
    # Create hash from filename, mtime, and size
    cache_key = f"{demo_path.name}_{mtime}_{size}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    
    return CACHE_DIR / f"{demo_path.stem}_{cache_hash}.pkl"


def load_cached_demo(demo_file):
    """Load demo data from cache if available and valid"""
    cache_file = get_cache_filename(demo_file)
    if cache_file and cache_file.exists():
        try:
            print(f"📦 Loading cached demo data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Create a minimal demo-like object with the cached data
            class CachedDemo:
                def __init__(self, data):
                    self.kills = data.get('kills')
                    self.rounds = data.get('rounds')
                    self.damages = data.get('damages')
                    self.smokes = data.get('smokes')
                    self.flashes = data.get('flashes')
                    self.grenades = data.get('grenades')
                    self.bomb = data.get('bomb')
                    self.frames = data.get('frames')
                    self.ticks = data.get('ticks')
            
            cached_demo = CachedDemo(cache_data)
            print("✅ Cached demo loaded successfully!")
            return cached_demo
            
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(f"⚠️ Cache file is corrupted: {e}")
            print("🗑️ Removing corrupted cache file...")
            try:
                cache_file.unlink()
                print("✅ Corrupted cache file removed")
            except Exception as remove_error:
                print(f"❌ Could not remove cache file: {remove_error}")
        except Exception as e:
            print(f"⚠️ Unexpected error loading cache: {e}")
            print("🗑️ Removing problematic cache file...")
            try:
                cache_file.unlink()
            except:
                pass
    
    return None


def save_demo_to_cache(demo, demo_file):
    """Save parsed demo data to cache (only the essential DataFrames)"""
    cache_file = get_cache_filename(demo_file)
    if cache_file:
        try:
            print(f"💾 Saving demo data to cache: {cache_file}...")
            
            # Extract only the essential data that we actually use
            cache_data = {
                'kills': demo.kills,
                'rounds': demo.rounds,
                'damages': demo.damages if hasattr(demo, 'damages') else None,
                'smokes': demo.smokes if hasattr(demo, 'smokes') else None,
                'flashes': demo.flashes if hasattr(demo, 'flashes') else None,
                'grenades': demo.grenades if hasattr(demo, 'grenades') else None,
                'bomb': demo.bomb if hasattr(demo, 'bomb') else None,
                'ticks': demo.ticks 
            }
            
            # Write to a temporary file first, then rename to avoid corruption
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Only replace the real cache file if the temp file was written successfully
            temp_file.rename(cache_file)
            print("✅ Demo data cached successfully!")
            
        except Exception as e:
            print(f"⚠️ Error saving to cache: {e}")
            # Clean up temporary file if it exists
            temp_file = cache_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass


def load_demo(demo_file, use_cache=True):
    """Load demo either from cache or by parsing"""
    # Try to load from cache first
    if use_cache:
        dem = load_cached_demo(demo_file)
        if dem is not None:
            return dem
    
    # Parse demo if not in cache
    print(f"🎮 Parsing demo: {demo_file}")
    try:
        dem = Demo(demo_file)
        dem.parse()
        print("✅ Demo parsed successfully!")
        
        # Save to cache for next time
        if use_cache:
            save_demo_to_cache(dem, demo_file)
        
        return dem
        
    except Exception as e:
        print(f"❌ Error parsing demo: {e}")
        return None


def clear_cache():
    """Clear all cached demo files"""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("🗑️ Cache cleared successfully!")
    else:
        print("ℹ️ No cache directory found")


def list_cache():
    """List all cached demo files"""
    if not CACHE_DIR.exists():
        print("ℹ️ No cache directory found")
        return
    
    cache_files = [f for f in CACHE_DIR.iterdir() if f.suffix == '.pkl']
    if not cache_files:
        print("ℹ️ No cached files found")
        return
    
    print("📦 Cached demo files:")
    total_size = 0
    for cache_file in cache_files:
        size = cache_file.stat().st_size
        total_size += size
        print(f"  • {cache_file.name} ({size / 1024 / 1024:.1f} MB)")
    
    print(f"Total cache size: {total_size / 1024 / 1024:.1f} MB")
