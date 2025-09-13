#!/usr/bin/env python3
"""Unzip HLTV demo archives.

Usage examples:
  - Single event: python unzip_demos.py --event 1234 --in-dir "G:/cs2/demos/archive" --out-dir "G:/cs2/demos"
  - All archives: python unzip_demos.py --all-archive --in-dir "G:/cs2/demos/archive" --out-dir "G:/cs2/demos"

The script finds .zip, .7z and .rar files and extracts any .dem files found.
When using --all-archive the relative folder structure under the input
directory is preserved in the output directory.
"""

from pathlib import Path
import argparse
import zipfile
import subprocess
import tempfile
from typing import List, Tuple

import py7zr
from tqdm import tqdm


def get_archive_type(file_path: Path) -> str:
    """Identify archive type by file signature (magic bytes)."""
    with open(file_path, "rb") as f:
        signature = f.read(8)

    if signature.startswith(b"PK\x03\x04"):
        return "zip"
    if signature.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "7z"
    if signature.startswith(b"Rar!\x1a\x07\x01\x00") or signature.startswith(b"Rar!\x1a\x07\x00"):
        return "rar"
    return "unknown"


def extract_with_7zip(archive_path: Path, output_dir: Path = None, overwrite: bool = False) -> List[Tuple[str, bytes]]:
    """Use 7z.exe to extract .dem files from an archive and return list of (name, bytes).

    This is used as a fallback for RAR archives (or any archive type where native
    python support isn't available).
    """
    if not archive_path.exists() or not archive_path.is_file():
        raise RuntimeError(f"Archive file not found or not a file: {archive_path}")

    # Quick existence check for outputs to avoid re-extracting
    if output_dir and not overwrite:
        base = archive_path.stem
        candidates = [output_dir / f"{base}.dem", *list(output_dir.glob(f"{base}_*.dem"))]
        if any(p.exists() for p in candidates):
            return []

    possible_7z_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z.exe",
    ]

    seven_zip_exe = None
    for p in possible_7z_paths:
        try:
            res = subprocess.run([p, "--help"], capture_output=True, timeout=3)
            if res.returncode == 0:
                seven_zip_exe = p
                break
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    if not seven_zip_exe:
        raise RuntimeError("7-Zip not found. Please install 7-Zip or add it to PATH.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd = [seven_zip_exe, "e", str(archive_path), f"-o{tmp}", "*.dem", "-y"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"7-Zip extraction failed: {proc.stderr}")

        dem_files = list(tmp_path.glob("*.dem"))
        if not dem_files:
            return []

        results: List[Tuple[str, bytes]] = []
        for f in dem_files:
            with open(f, "rb") as fh:
                results.append((f.name, fh.read()))
        return results


def process_archives(archive_files: List[Path], in_root: Path, out_root: Path, overwrite: bool):
    """Process a list of archives and extract .dem files, preserving relative folders.

    - archive_files: list of absolute paths to archives
    - in_root: root input directory (used to compute relative paths)
    - out_root: output root directory
    """
    tqdm.write(f"Identifying archive types for {len(archive_files)} files...")
    file_types = {}
    for p in tqdm(archive_files, desc="Identifying files"):
        t = get_archive_type(p)
        file_types[p] = t
        tqdm.write(f"{p} -> {t}")

    processed = skipped = errors = 0

    for archive_path, archive_type in tqdm(file_types.items(), desc="Unzipping demos"):
        try:
            # compute relative folder and create output directory
            try:
                rel_dir = archive_path.parent.relative_to(in_root)
            except Exception:
                # fallback: use parent name only
                rel_dir = Path(archive_path.parent.name)

            event_out_dir = out_root / rel_dir
            event_out_dir.mkdir(parents=True, exist_ok=True)

            base = archive_path.stem
            existing_candidates = [event_out_dir / f"{base}.dem", *list(event_out_dir.glob(f"{base}_*.dem"))]
            if not overwrite and any(p.exists() for p in existing_candidates):
                skipped += 1
                continue

            dem_files_data: List[Tuple[str, bytes]] = []

            if archive_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as z:
                    dem_names = [n for n in z.namelist() if n.lower().endswith(".dem")]
                    for name in dem_names:
                        dem_files_data.append((Path(name).name, z.read(name)))

            elif archive_type == "7z":
                with py7zr.SevenZipFile(archive_path, "r") as a:
                    all_names = a.getnames()
                    dem_names = [n for n in all_names if n.lower().endswith(".dem")]
                    if dem_names:
                        bio = a.read(targets=dem_names)
                        for name in dem_names:
                            if name in bio:
                                # bio[name] is a file-like object
                                dem_files_data.append((Path(name).name, bio[name].read()))

            elif archive_type == "rar":
                # Use 7z as a fallback for rar
                dem_files_data = extract_with_7zip(archive_path, event_out_dir, overwrite)

            else:
                tqdm.write(f"Skipping unknown file type: {archive_path}")
                continue

            if not dem_files_data:
                tqdm.write(f"Warning: No .dem files found in {archive_path}")
                processed += 1
                continue

            for dem_name, dem_bytes in dem_files_data:
                if len(dem_files_data) == 1:
                    out_path = event_out_dir / f"{base}.dem"
                else:
                    clean = Path(dem_name).stem
                    out_path = event_out_dir / f"{base}_{clean}.dem"

                if out_path.exists() and not overwrite:
                    tqdm.write(f"Skipping existing file: {out_path}")
                    continue

                with open(out_path, "wb") as fh:
                    fh.write(dem_bytes)

            processed += 1

        except Exception as exc:
            tqdm.write(f"Error processing {archive_path}: {exc}")
            errors += 1

    print(f"Done. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


def main():
    ap = argparse.ArgumentParser(description="Unzip HLTV demos for a single event or all archives.")
    ap.add_argument("--event", type=int, help="HLTV event id (optional)")
    ap.add_argument("--in-dir", type=Path, default=Path(r"F:/CS2/demos_zipped"), help="Input directory for zipped demos")
    ap.add_argument("--out-dir", type=Path, default=Path(r"F:/CS2/demos"), help="Output directory for extracted .dem files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite .dem files if they exist")
    ap.add_argument("--all-archive", action="store_true", help="Recursively process all archives under the input directory")
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir

    if args.all_archive:
        archives: List[Path] = []
        archives.extend(list(in_dir.rglob("*.zip")))
        archives.extend(list(in_dir.rglob("*.7z")))
        archives.extend(list(in_dir.rglob("*.rar")))
        if not archives:
            print(f"No archive files found in {in_dir.resolve()}")
            return
        process_archives(archives, in_dir, out_dir, args.overwrite)

    elif args.event is not None:
        event_in = in_dir / str(args.event)
        if not event_in.exists():
            print(f"Input directory not found: {event_in.resolve()}")
            return
        archives = list(event_in.glob("*.zip")) + list(event_in.glob("*.7z")) + list(event_in.glob("*.rar"))
        if not archives:
            print(f"No archive files found in {event_in.resolve()}")
            return
        process_archives(archives, event_in, out_dir / str(args.event), args.overwrite)

    else:
        print("You must specify either --event <id> or --all-archive.")


if __name__ == "__main__":
    main()
