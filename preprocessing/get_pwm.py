import os
import requests
import numpy as np
from pyjaspar import jaspardb


def get_metadata_from_pdb(pdb_id):
    """Fetches UniProt IDs and Gene Symbols from the PDBe API."""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    uniprot_ids, gene_symbols = [], []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                pdb_key = list(data.keys())[0]
                uniprot_data = data[pdb_key].get("UniProt", {})
                for up_id, info in uniprot_data.items():
                    uniprot_ids.append(up_id)
                    if "identifier" in info:
                        gene = info["identifier"].split("_")[0]
                        gene_symbols.append(gene.upper())
    except Exception:
        pass
    return list(set(uniprot_ids)), list(set(gene_symbols))


def build_jaspar_index(release="JASPAR2024"):
    """Builds the offline lookup index across ALL collections and ALL historic versions."""
    jdb_obj = jaspardb(release=release)
    all_motifs = jdb_obj.fetch_motifs(all=True, all_versions=True)

    uniprot_to_motifs = {}
    gene_to_motifs = {}

    for motif in all_motifs:
        if motif.acc:
            for up_id in motif.acc:
                uniprot_to_motifs.setdefault(up_id, []).append(motif)
        if motif.name:
            gene_to_motifs.setdefault(motif.name.upper(), []).append(motif)

    return {"by_uniprot": uniprot_to_motifs, "by_gene": gene_to_motifs}


def get_motifs_for_pdb(pdb_id, indices):
    """Finds candidate motifs for a PDB using the pre-built indices."""
    uniprot_ids, gene_symbols = get_metadata_from_pdb(pdb_id)
    matched_motifs = []

    for up_id in uniprot_ids:
        if up_id in indices["by_uniprot"]:
            matched_motifs.extend(indices["by_uniprot"][up_id])

    if not matched_motifs:
        for gene in gene_symbols:
            if gene in indices["by_gene"]:
                matched_motifs.extend(indices["by_gene"][gene])

    unique_motifs = {m.matrix_id: m for m in matched_motifs}.values()
    return list(unique_motifs)


def select_best_motif(motifs, target_length=7):
    """Picks the motif closest to the target length to resolve ties."""
    if not motifs:
        return None
    return sorted(motifs, key=lambda m: (abs(len(m) - target_length), m.matrix_id))[0]


def extract_pwm_matrix(motif):
    """Converts a JASPAR motif into a Numpy Array of shape (L, 4)."""
    try:
        ppm = motif.counts.normalize(pseudocounts=0.5)
        pwm_dict = ppm.log_odds()
        pwm_matrix = np.array(
            [pwm_dict["A"], pwm_dict["C"], pwm_dict["G"], pwm_dict["T"]],
            dtype=np.float32,
        )
        return pwm_matrix.T
    except Exception:
        return None


#fallbacks

def parse_raw_fallback_file(file_path, is_cisbp=False):
    """Parses plain text PWM rows and transforms them into log-odds log(p/0.25) format."""
    matrix = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                if is_cisbp and parts[0].isalpha():
                    continue  # Skip text header row

                vals = [float(x) for x in parts]
                if is_cisbp and len(vals) == 5:
                    matrix.append(vals[1:])  # Drop position tracking index column
                elif not is_cisbp and len(vals) == 4:
                    matrix.append(vals)  # HOCOMOCO layout rows
    except Exception:
        return None

    mat_arr = np.array(matrix, dtype=np.float32)
    if mat_arr.size == 0:
        return None

    # Convert raw counts/probabilities into a uniform Position Probability Matrix (PPM)
    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)

    # Apply a 0.5 pseudocount correction adjustment and convert to standard Log-Odds
    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    log_odds = np.log2(ppm / 0.25)
    return log_odds


def check_local_database_fallbacks(
    pdb_id, hocomoco_dir="../data/motifs/hocomoco", cisbp_dir="../data/motifs/cisbp"
):
    """Scans local folders for file matching by Gene Name or ID prefix."""
    _, gene_symbols = get_metadata_from_pdb(pdb_id)
    if not gene_symbols:
        return None

    # 1. Look inside HOCOMOCO folder (e.g. AHR_HUMAN.H1MO.0.8.pwm)
    if os.path.exists(hocomoco_dir):
        hocomoco_files = os.listdir(hocomoco_dir)
        for gene in gene_symbols:
            for fname in hocomoco_files:
                if fname.upper().startswith(f"{gene}_"):
                    full_path = os.path.join(hocomoco_dir, fname)
                    return parse_raw_fallback_file(full_path, is_cisbp=False)

    # 2. Look inside CIS-BP folder (e.g. M00001_3.10.txt)
    # Without a map file, it matches if a PDB identifier or gene name aligns with target prefixes
    if os.path.exists(cisbp_dir):
        cisbp_files = os.listdir(cisbp_dir)
        # Check if any gene prefix matches the naming variations
        for gene in gene_symbols:
            for fname in cisbp_files:
                if fname.upper().startswith(gene):
                    full_path = os.path.join(cisbp_dir, fname)
                    return parse_raw_fallback_file(full_path, is_cisbp=True)

    return None
