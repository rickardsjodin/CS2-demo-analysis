#!/usr/bin/env python3
"""
Unzip HLTV demo archives for a given event.

- Finds all .zip files in `demos/<event_id>/`
- For each zip, extracts the .dem file
- Saves it to `demos/<event_id>/<zip_basename>.dem`
"""
import argparse
import py7zr
import zipfile
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

def get_archive_type(file_path: Path) -> str:
    """Read the first few bytes of a file to identify its type."""
    with open(file_path, 'rb') as f:
        signature = f.read(8)
    
    if signature.startswith(b'PK\x03\x04'):
        return 'zip'
    if signature.startswith(b'7z\xbc\xaf\x27\x1c'):
        return '7z'
    # RAR 5.0+ and 1.5-4.x
    if signature.startswith(b'Rar!\x1a\x07\x01\x00') or signature.startswith(b'Rar!\x1a\x07\x00'):
        return 'rar'
    return 'unknown'

def extract_with_7zip(archive_path: Path) -> list[tuple[str, bytes]]:
    """Extract all .dem files from archive using 7-Zip command-line tool."""
    # Common 7-Zip installation paths
    possible_7z_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z.exe"  # If it's in PATH
    ]
    
    # Find 7z.exe
    seven_zip_exe = None
    for path in possible_7z_paths:
        try:
            result = subprocess.run([path, "--help"], capture_output=True, timeout=5)
            if result.returncode == 0:
                seven_zip_exe = path
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if not seven_zip_exe:
        raise RuntimeError("7-Zip not found. Please install 7-Zip or add it to PATH.")
    
    # Use a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract archive to temp directory
        result = subprocess.run([
            seven_zip_exe, "e",  # extract command
            str(archive_path),
            f"-o{temp_dir}",     # output directory
            "*.dem",             # only extract .dem files
            "-y"                 # yes to all prompts
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"7-Zip extraction failed: {result.stderr}")
        
        # Find all extracted .dem files
        temp_path = Path(temp_dir)
        dem_files = list(temp_path.glob("*.dem"))
        
        if not dem_files:
            raise RuntimeError("No .dem file found in archive")
        
        # Read all .dem files and return them with their names
        results = []
        for dem_file in dem_files:
            with open(dem_file, 'rb') as f:
                results.append((dem_file.name, f.read()))
        
        return results

def main():
    ap = argparse.ArgumentParser(description="Unzip HLTV demos for a given event ID.")
    ap.add_argument("--event", type=int, required=True, help="HLTV event id")
    ap.add_argument("--in-dir", type=Path, default=Path("F://CS2/demos_zipped"), help="Input directory for zipped demos")
    ap.add_argument("--out-dir", type=Path, default=Path("F://CS2/demos"), help="Output directory for extracted .dem files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite .dem files if they exist")
    args = ap.parse_args()

    event_in_dir = args.in_dir / str(args.event)
    event_out_dir = args.out_dir / str(args.event)

    if not event_in_dir.exists():
        print(f"Input directory not found: {event_in_dir.resolve()}")
        return

    event_out_dir.mkdir(parents=True, exist_ok=True)

    # The extension might be .zip, but the content could be something else
    archive_files = list(event_in_dir.glob("*.zip"))
    if not archive_files:
        print(f"No .zip files found in {event_in_dir.resolve()}")
        return

    print(f"Identifying archive types in {event_in_dir.resolve()}...")
    file_types = {}
    for file_path in tqdm(archive_files, desc="Identifying files"):
        archive_type = get_archive_type(file_path)
        file_types[file_path] = archive_type
        tqdm.write(f"{file_path.name}: {archive_type}")

    print(f"\nUnzipping {len(archive_files)} demos from {event_in_dir.resolve()} to {event_out_dir.resolve()}")

    for archive_path, archive_type in tqdm(file_types.items(), desc="Unzipping demos"):
        archive_basename = archive_path.stem  # filename without extension
        
        dem_files_data = []
        try:
            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    dem_files = [f for f in zip_ref.namelist() if f.endswith('.dem')]
                    for dem_file in dem_files:
                        dem_content = zip_ref.read(dem_file)
                        dem_files_data.append((Path(dem_file).name, dem_content))
            
            elif archive_type == '7z':
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
                try:
                    dem_files_data = extract_with_7zip(archive_path)
                except RuntimeError as e:
                    tqdm.write(f"Error extracting RAR file {archive_path.name}: {e}")
                    continue
            
            else:
                tqdm.write(f"Skipping unknown file type: {archive_path.name}")
                continue

            if dem_files_data:
                for i, (dem_filename, dem_content) in enumerate(dem_files_data):
                    if len(dem_files_data) == 1:
                        # Single demo file, use archive name
                        output_dem_path = event_out_dir / f"{archive_basename}.dem"
                    else:
                        # Multiple demo files, include original filename
                        clean_dem_name = Path(dem_filename).stem
                        output_dem_path = event_out_dir / f"{archive_basename}_{clean_dem_name}.dem"
                    
                    if output_dem_path.exists() and not args.overwrite:
                        continue
                    
                    with open(output_dem_path, 'wb') as target:
                        target.write(dem_content)
            else:
                tqdm.write(f"Warning: No .dem files found in {archive_path.name}")

        except Exception as e:
            tqdm.write(f"An error occurred with {archive_path.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
