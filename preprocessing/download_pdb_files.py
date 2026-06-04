"""
Script to download PDB/mmCIF files from RCSB PDB.
Uses the list of PDB IDs from NA-MPNN's training splits.
"""

import os
import sys
import json
import urllib.request
import gzip
import shutil
from tqdm import tqdm
import urllib.error


def download_pdb_file(pdb_id, output_dir, file_format="pdb", overwrite=False):
    """
    Download a single PDB file from RCSB PDB and decompress it.
    Gracefully skips obsoleted or oversized structures that return 404.
    """
    pdb_id = pdb_id.lower()

    # Define the output filename WITHOUT .gz, but keep the download URL with .gz
    if file_format == "cif":
        filename = f"{pdb_id}.cif"
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    elif file_format == "pdb":
        filename = f"{pdb_id}.pdb"
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    else:
        raise ValueError(f"Unknown format: {file_format}")

    output_path = os.path.join(output_dir, filename)

    # Skip if already exists
    if os.path.exists(output_path) and not overwrite:
        return output_path

    try:
        # Add User-Agent header so RCSB doesn't block the request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req) as response:
            # Decompress the file stream on the fly and save as plain text!
            with gzip.GzipFile(fileobj=response) as uncompressed:
                with open(output_path, "wb") as out_file:
                    shutil.copyfileobj(uncompressed, out_file)

        return output_path

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # It's obsolete or too big for standard PDB format. Safe to skip.
            return None
        else:
            return None
    except Exception as e:
        return None


def download_pdb_list(pdb_ids, output_dir, file_format="cif", max_workers=4):
    """
    Download multiple PDB files.
    
    Args:
        pdb_ids: List of PDB IDs
        output_dir: Directory to save files
        file_format: "cif" or "pdb"
        max_workers: Not used currently (sequential download)
    
    Returns:
        List of paths to downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_files = []
    failed_ids = []
    
    for pdb_id in tqdm(pdb_ids, desc="Downloading PDB files"):
        path = download_pdb_file(pdb_id, output_dir, file_format)
        if path:
            downloaded_files.append(path)
        else:
            failed_ids.append(pdb_id)
    
    print(f"\nDownloaded {len(downloaded_files)} files")
    if failed_ids:
        print(f"Failed to download {len(failed_ids)} files: {failed_ids}")
    
    return downloaded_files


def load_pdb_ids_from_split(split_file):
    """
    Load PDB IDs from NA-MPNN split file.
    """
    with open(split_file, "r") as f:
        data = json.load(f)

    pdb_ids = []
    # specificity_train.json structure: [ ["1a02", [motifs]], ["scaffold_X", [...]] ]
    for item in data:
        if isinstance(item, list) and len(item) >= 1:
            identifier = str(item[0])
            # Only keep valid 4-character PDB IDs (skips 'scaffold_...' sequences)
            if len(identifier) == 4:
                pdb_ids.append(identifier.lower())

    return list(set(pdb_ids))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download PDB files for NA-MPNN training")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save downloaded files")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path to NA-MPNN split JSON file")
    parser.add_argument("--pdb_ids", type=str, nargs="+", default=None,
                        help="List of PDB IDs to download")
    parser.add_argument("--file_format", type=str, default="pdb", choices=["cif", "pdb"],
                        help="File format (cif or pdb)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to download (for testing)")
    
    args = parser.parse_args()
    
    # Get PDB IDs
    if args.split_file:
        print(f"Loading PDB IDs from {args.split_file}")
        pdb_ids = load_pdb_ids_from_split(args.split_file)
    elif args.pdb_ids:
        pdb_ids = args.pdb_ids
    else:
        print("Error: Must provide either --split_file or --pdb_ids")
        sys.exit(1)
    
    # Limit if requested
    if args.max_files:
        pdb_ids = pdb_ids[:args.max_files]
    
    print(f"Downloading {len(pdb_ids)} PDB files to {args.output_dir}")
    
    # Download
    downloaded_files = download_pdb_list(
        pdb_ids, 
        args.output_dir, 
        args.file_format
    )
    
    # Save list of downloaded files
    list_file = os.path.join(args.output_dir, "downloaded_files.txt")
    with open(list_file, 'w') as f:
        for path in downloaded_files:
            f.write(path + "\n")
    
    print(f"\nList of downloaded files saved to {list_file}")
