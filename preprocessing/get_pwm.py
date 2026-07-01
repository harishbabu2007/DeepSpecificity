import os
import requests
import numpy as np
from pyjaspar import jaspardb
import sys

from constants import DNA_BACKBONE_ORDER, COORDINATE_SCALE_FACTOR, DNA_FEATURE_SIZE
from dna_features import get_base_letter
from dna_definitions import BASE_HEAVY_ATOMS


def parse_raw_fallback_file(file_path, is_cisbp=False):
    matrix = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"): continue
                parts = line_str.split()
                if is_cisbp and parts[0].isalpha(): continue
                vals = [float(x) for x in parts]
                if is_cisbp and len(vals) == 5: matrix.append(vals[1:])
                elif not is_cisbp and len(vals) == 4: matrix.append(vals)
    except Exception: return None
    mat_arr = np.array(matrix, dtype=np.float32)
    if mat_arr.size == 0: return None
    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)
    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)

def parse_uniprobe_file(file_path):
    matrix_rows = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"): continue
                parts = line_str.split()
                if not parts: continue
                if parts[0].rstrip(':').upper() in ['A', 'C', 'G', 'T']: parts = parts[1:]
                try: matrix_rows.append([float(x) for x in parts if x])
                except ValueError: continue
    except Exception: return None
    mat_arr = np.array(matrix_rows, dtype=np.float32)
    if mat_arr.size == 0: return None
    if mat_arr.shape[0] == 4 and mat_arr.shape[1] != 4: mat_arr = mat_arr.T
    elif mat_arr.shape[0] != 4 and mat_arr.shape[1] == 4: pass
    else: return None 
    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)
    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)

def get_pwm_matrix_from_annotations(pdb_id, annotations, hocomoco_dir="../data/motifs/hocomoco", cisbp_dir="../data/motifs/cisbp", uniprobe_dir="../data/motifs/uniprobe"):
    pdb_id = pdb_id.lower()
    
    if pdb_id not in annotations or not annotations[pdb_id]: return None

    jdb_obj = jaspardb(release="JASPAR2024")
    for site_motifs in annotations[pdb_id]:
        for motif_info in site_motifs:
            if len(motif_info) != 2: continue
            db_name, motif_id = motif_info
            if db_name == "JASPAR":
                try:
                    motif = jdb_obj.fetch_motif_by_id(motif_id.replace(".jaspar", ""))
                    if motif:
                        ppm = motif.counts.normalize(pseudocounts=0.5)
                        pwm_dict = ppm.log_odds()
                        return np.array([pwm_dict["A"], pwm_dict["C"], pwm_dict["G"], pwm_dict["T"]], dtype=np.float32).T
                except Exception: continue
            elif db_name == "HOCOMOCO":
                mat = parse_raw_fallback_file(os.path.join(hocomoco_dir, f"{motif_id}.pwm"), False)
                if mat is not None: return mat
            elif db_name == "CIS-BP":
                mat = parse_raw_fallback_file(os.path.join(cisbp_dir, "pwms", f"{motif_id}.txt"), True)
                if mat is not None: return mat
            elif db_name == "UniPROBE":
                if os.path.exists(uniprobe_dir):
                    for root, _, files in os.walk(uniprobe_dir):
                        for fname in files:
                            if motif_id in fname and fname.upper().endswith(".PWM") and ".RC." not in fname.upper():
                                mat = parse_uniprobe_file(os.path.join(root, fname))
                                if mat is not None: return mat
    return None


def get_metadata_from_pdb(pdb_id):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    uniprot_ids, gene_symbols = [], []
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and response.json():
            for up_id, info in response.json()[list(response.json().keys())[0]].get("UniProt", {}).items():
                uniprot_ids.append(up_id)
                if "identifier" in info: gene_symbols.append(info["identifier"].split("_")[0].upper())
    except Exception: pass
    return list(set(uniprot_ids)), list(set(gene_symbols))

def build_jaspar_index(release="JASPAR2024"):
    jdb_obj = jaspardb(release=release)
    all_motifs = jdb_obj.fetch_motifs(all=True, all_versions=True)
    uniprot_to_motifs, gene_to_motifs = {}, {}
    for motif in all_motifs:
        if motif.acc:
            for up_id in motif.acc: uniprot_to_motifs.setdefault(up_id, []).append(motif)
        if motif.name: gene_to_motifs.setdefault(motif.name.upper(), []).append(motif)
    return {"by_uniprot": uniprot_to_motifs, "by_gene": gene_to_motifs}

def check_local_database_fallbacks(pdb_id, hocomoco_dir="../data/motifs/hocomoco", cisbp_dir="../data/motifs/cisbp", uniprobe_dir="../data/motifs/uniprobe"):
    _, gene_symbols = get_metadata_from_pdb(pdb_id)
    if not gene_symbols: return None
    
    if os.path.exists(hocomoco_dir):
        for gene in gene_symbols:
            for fname in os.listdir(hocomoco_dir):
                if fname.upper().startswith(f"{gene}_"):
                    return parse_raw_fallback_file(os.path.join(hocomoco_dir, fname), False)
                    
    if os.path.exists(cisbp_dir):
        for gene in gene_symbols:
            for fname in os.listdir(cisbp_dir):
                if fname.upper().startswith(gene):
                    return parse_raw_fallback_file(os.path.join(cisbp_dir, fname), True)
                    
    if os.path.exists(uniprobe_dir):
        for root, _, files in os.walk(uniprobe_dir):
            for gene in gene_symbols:
                for fname in files:
                    fname_upper = fname.upper()
                    if ".RC." in fname_upper: continue
                    if fname_upper.startswith(f"{gene}_") or fname_upper.startswith(gene):
                        if fname_upper.endswith(".PWM"):
                            mat = parse_uniprobe_file(os.path.join(root, fname))
                            if mat is not None: return mat
    return None


def get_hybrid_pwm(pdb_id, annotations, jaspar_indices, hoco="../data/motifs/hocomoco", cis="../data/motifs/cisbp", uni="../data/motifs/uniprobe"):
    """Tries strict JSON first. If it fails, hunts down the motif via API & local scan."""
    mat = get_pwm_matrix_from_annotations(pdb_id, annotations, hoco, cis, uni)
    if mat is not None:
        return mat
    
    uniprot_ids, gene_symbols = get_metadata_from_pdb(pdb_id)
    matched_motifs = []
    for up_id in uniprot_ids:
        if up_id in jaspar_indices["by_uniprot"]: matched_motifs.extend(jaspar_indices["by_uniprot"][up_id])
    if not matched_motifs:
        for gene in gene_symbols:
            if gene in jaspar_indices["by_gene"]: matched_motifs.extend(jaspar_indices["by_gene"][gene])
    if matched_motifs:
        best_motif = sorted(matched_motifs, key=lambda m: (abs(len(m) - 18), m.matrix_id))[0]
        try:
            ppm = best_motif.counts.normalize(pseudocounts=0.5)
            pwm_dict = ppm.log_odds()
            return np.array([pwm_dict["A"], pwm_dict["C"], pwm_dict["G"], pwm_dict["T"]], dtype=np.float32).T
        except Exception: pass
        
    return check_local_database_fallbacks(pdb_id, hoco, cis, uni)


def generate_spatial_proximity_mask(
    dna_features,
    protein_features,
    distance_threshold_angstroms=7.0,
):
    """
    Identifies DNA positions that are spatially close to any protein
    side-chain atom.

    A nucleotide is considered proximal if ANY of its representative
    atoms (P, C1', N1/N9) lie within the distance threshold.
    """

    N_d = dna_features.shape[0]

    HALF_DNA_FEATURE_SIZE = DNA_FEATURE_SIZE // 2

    # ------------------------------------------------------------------
    # Forward strand indices
    # ------------------------------------------------------------------

    p_idx = DNA_BACKBONE_ORDER.index("P")
    c1_idx = DNA_BACKBONE_ORDER.index("C1'")

    fwd_p_start = p_idx * 3
    fwd_c1_start = c1_idx * 3

    # N1 / N9 coordinate is stored immediately after backbone coords
    fwd_base_start = len(DNA_BACKBONE_ORDER) * 3

    # ------------------------------------------------------------------
    # Reverse strand indices
    # ------------------------------------------------------------------

    rev_offset = HALF_DNA_FEATURE_SIZE

    rev_p_start = rev_offset + p_idx * 3
    rev_c1_start = rev_offset + c1_idx * 3
    rev_base_start = rev_offset + len(DNA_BACKBONE_ORDER) * 3

    # ------------------------------------------------------------------
    # Extract DNA coordinates
    # ------------------------------------------------------------------

    fwd_p_coords = (
        dna_features[:, fwd_p_start:fwd_p_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    fwd_c1_coords = (
        dna_features[:, fwd_c1_start:fwd_c1_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    fwd_base_coords = (
        dna_features[:, fwd_base_start:fwd_base_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    rev_p_coords = (
        dna_features[:, rev_p_start:rev_p_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    rev_c1_coords = (
        dna_features[:, rev_c1_start:rev_c1_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    rev_base_coords = (
        dna_features[:, rev_base_start:rev_base_start + 3]
        * COORDINATE_SCALE_FACTOR
    )

    # ------------------------------------------------------------------
    # Protein side-chain coordinates
    # ------------------------------------------------------------------

    sidechain_coords_raw = (
        protein_features[:, 32:62]
        .reshape(-1, 10, 3)
        * COORDINATE_SCALE_FACTOR
    )

    sidechain_atoms = sidechain_coords_raw.reshape(-1, 3)

    valid_sidechain_atoms = sidechain_atoms[
        ~np.all(sidechain_atoms == 0.0, axis=1)
    ]

    if len(valid_sidechain_atoms) == 0:
        return (
            np.zeros(N_d, dtype=bool),
            np.zeros(N_d, dtype=bool),
        )

    # ------------------------------------------------------------------
    # Ignore unpaired DNA overhangs
    # ------------------------------------------------------------------

    fwd_valid = ~np.all(fwd_c1_coords == 0.0, axis=1)
    rev_valid = ~np.all(rev_c1_coords == 0.0, axis=1)

    paired_mask = fwd_valid & rev_valid

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

    def check_proximity(dna_coords):
        diff = (
            dna_coords[:, None, :]
            - valid_sidechain_atoms[None, :, :]
        )

        distances = np.linalg.norm(diff, axis=-1)

        return np.any(
            distances < distance_threshold_angstroms,
            axis=1,
        )

    # ------------------------------------------------------------------
    # Forward strand
    # ------------------------------------------------------------------

    fwd_near_p = check_proximity(fwd_p_coords)
    fwd_near_c1 = check_proximity(fwd_c1_coords)
    fwd_near_base = check_proximity(fwd_base_coords)

    # ------------------------------------------------------------------
    # Reverse strand
    # ------------------------------------------------------------------

    rev_near_p = check_proximity(rev_p_coords)
    rev_near_c1 = check_proximity(rev_c1_coords)
    rev_near_base = check_proximity(rev_base_coords)

    # ------------------------------------------------------------------
    # Final masks
    # ------------------------------------------------------------------

    fwd_proximity_mask = (
        fwd_near_p |
        fwd_near_c1 |
        fwd_near_base
    ) & paired_mask

    rev_proximity_mask = (
        rev_near_p |
        rev_near_c1 |
        rev_near_base
    ) & paired_mask

    return (
        fwd_proximity_mask,
        rev_proximity_mask,
    )


def generate_protein_dna_distance_matrix(
    protein_residues,
    dna_pairs,
):
    """
    Computes the minimum heavy-atom distance between every
    DNA base pair and every protein residue.

    DNA atoms considered:
        - DNA_BACKBONE_ORDER
        - BASE_HEAVY_ATOMS (currently N9 for A/G, N1 for C/T)

    Returns
    -------
    distance_matrix : (Nd, Nprotein)
    """

    distance_matrix = []

    for forward_residue, reverse_residue in dna_pairs:

        dna_atoms = []

        # ----------------------------------------------------------
        # Collect the DNA atoms used by the feature extractor
        # ----------------------------------------------------------

        for residue in [forward_residue, reverse_residue]:

            if residue is None:
                continue

            # Backbone atoms
            for atom_name in DNA_BACKBONE_ORDER:

                if atom_name in residue:
                    dna_atoms.append(residue[atom_name].coord.astype(np.float32))

            # Base atoms (N9 / N1 currently)
            base_letter = get_base_letter(residue)

            for atom_name in BASE_HEAVY_ATOMS[base_letter]:

                if atom_name in residue:
                    dna_atoms.append(residue[atom_name].coord.astype(np.float32))

        # ----------------------------------------------------------
        # Compute minimum distance to every protein residue
        # ----------------------------------------------------------

        row = []

        for residue in protein_residues:

            min_distance = float("inf")

            for protein_atom in residue:

                if protein_atom.element == "H":
                    continue

                protein_coord = protein_atom.coord.astype(np.float32)

                for dna_coord in dna_atoms:

                    dist = np.linalg.norm(protein_coord - dna_coord)

                    if dist < min_distance:
                        min_distance = dist

            row.append(min_distance)

        distance_matrix.append(row)

    return np.asarray(
        distance_matrix,
        dtype=np.float32,
    )
