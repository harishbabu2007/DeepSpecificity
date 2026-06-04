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


def download_pdb_file(pdb_id, output_dir, file_format="cif", overwrite=False):
    """
    Download a single PDB file from RCSB PDB.
    
    Args:
        pdb_id: 4-character PDB ID (e.g., "1ABC")
        output_dir: Directory to save the file
        file_format: "cif" or "pdb"
        overwrite: Whether to overwrite existing files
    
    Returns:
        Path to downloaded file, or None if failed
    """
    pdb_id = pdb_id.lower()
    
    if file_format == "cif":
        filename = f"{pdb_id}.cif.gz"
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    elif file_format == "pdb":
        filename = f"{pdb_id}.pdb.gz"
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    else:
        raise ValueError(f"Unknown format: {file_format}")
    
    output_path = os.path.join(output_dir, filename)
    
    # Skip if already exists
    if os.path.exists(output_path) and not overwrite:
        return output_path
    
    try:
        # Download file
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(output_path, 'wb') as out_file:
                out_file.write(response.read())
        
        # Decompress if gzipped
        if output_path.endswith('.gz'):
            decompressed_path = output_path[:-3]  # Remove .gz
            with gzip.open(output_path, 'rb') as gz_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(gz_in, f_out)
            os.remove(output_path)  # Remove gz file
            return decompressed_path
        
        return output_path
    
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
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
