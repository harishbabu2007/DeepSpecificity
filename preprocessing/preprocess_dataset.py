import os
import glob
import traceback
import subprocess
import time
import numpy as np

from tqdm import tqdm
from pdb_parser import load_and_validate, StructureRejected
from coordinate_utils import compute_complex_centroid
from dna_features import generate_dna_features
from protein_features import generate_protein_features
from bond_matrix import generate_bond_matrix
from npz_writer import save_npz, build_output_path
from get_pwm import get_hybrid_pwm, build_jaspar_index, generate_spatial_proximity_mask
from constants import MAX_PROTEIN_LENGTH
from get_shape_features import get_dna_shape_features
import json

from openmm.app import PDBFile
from pdbfixer import PDBFixer
import sys

def load_motif_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = {}
    for item in data:
        if isinstance(item, list) and len(item) >= 2:
            pdb_id = str(item[0]).lower()
            motifs = item[1]
            if len(pdb_id) == 4 and motifs:
                annotations[pdb_id] = motifs
    return annotations

def compute_ic(pwm_col, epsilon=1e-9):
    """Compute normalized Information Content (0.0 to 2.0 bits)."""
    p = np.clip(pwm_col, epsilon, 1.0)
    p = p / np.sum(p)
    entropy = -np.sum(p * np.log2(p))
    return max(0.0, 2.0 - entropy)


def trim_pwm(pwm, ic_threshold=0.5):
    """Trims uninformative flanks where IC < threshold (DeepPBS Supp Section 1)."""
    s = pwm.shape[0]
    start, end = 0, s
    for i in range(s):
        if compute_ic(pwm[i]) >= ic_threshold:
            start = i
            break
    for i in range(s - 1, -1, -1):
        if compute_ic(pwm[i]) >= ic_threshold:
            end = i + 1
            break
    return pwm[start:end] if start < end else pwm


def ic_weighted_pcc(col_pwm, col_dna, epsilon=1e-9):
    """Calculates IC-weighted PCC between a PWM column and 1-hot DNA base."""
    diff_pwm = col_pwm - 0.25
    diff_dna = col_dna - 0.25
    num = np.sum(diff_pwm * diff_dna)
    den = np.sqrt(np.sum(diff_pwm**2) * np.sum(diff_dna**2))
    pcc = num / den if den > epsilon else 0.0
    ic = compute_ic(col_pwm, epsilon)
    return pcc * 0.5 * ic


def ungapped_align(seq_one_hot, pwm, min_overlap=5):
    """Performs an ungapped sliding-window local alignment (Supp Section 2)."""
    l, s = seq_one_hot.shape[0], pwm.shape[0]
    if l < min_overlap or s < min_overlap:
        return 0, 0, 0, -9999.0

    max_score = -9999.0
    opt_i, opt_j, opt_k = 0, 0, 0

    # Precompute pairwise column similarity to prevent intensive looping
    pairwise_scores = np.zeros((s, l))
    for i in range(s):
        for j in range(l):
            pairwise_scores[i, j] = ic_weighted_pcc(pwm[i], seq_one_hot[j])

    # Slide window configurations
    for i in range(s):
        for k in range(min_overlap, s - i + 1):
            for j in range(l - k + 1):
                # Trace diagonal representing un-gapped sequence alignment match
                score = np.sum(pairwise_scores[i : i + k, j : j + k].diagonal())
                if score > max_score:
                    max_score, opt_i, opt_j, opt_k = score, i, j, k

    return opt_i, opt_j, opt_k, max_score


def get_sequence_one_hot(dna_labels):
    """Converts labels to 5'-to-3' forward and reverse one-hot representations."""
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    l = len(dna_labels)
    seq_fwd = np.zeros((l, 4), dtype=np.float32)

    for idx, label in enumerate(dna_labels):
        f_base = label[0]
        if f_base in mapping:
            seq_fwd[idx, mapping[f_base]] = 1.0

    # Reverse complement strand sequence (Flipped and complemented)
    seq_rev = seq_fwd[::-1, [3, 2, 1, 0]]
    return seq_fwd, seq_rev


def build_protein_labels(protein_residues):
    labels = []
    for residue in protein_residues:
        aa = residue.get_resname()
        residue_id = residue.id[1]
        labels.append(f"{aa}{residue_id}")
    return labels


def build_dna_labels(dna_pairs):
    labels = []
    for forward, reverse in dna_pairs:
        forward_name = forward.get_resname().replace("D", "")
        if reverse is None:
            labels.append(f"{forward_name}-")
        else:
            reverse_name = reverse.get_resname().replace("D", "")
            labels.append(f"{forward_name}{reverse_name}")
    return labels


def hydrogenate_pdb_reduce(input_pdb, output_pdb):
    result = subprocess.run(["reduce", input_pdb], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Reduce failed:\n{result.stderr}")
    with open(output_pdb, "w") as f:
        f.write(result.stdout)
    return output_pdb


def hydrogenate_with_pdbfixer(input_pdb: str, output_pdb: str) -> str:
    """
    Hydrogenates and repairs a PDB file using PDBFixer.
    Adds missing heavy atoms, cleans structure drops, and adds hydrogens at pH 7.0.
    """
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    fixer.addMissingHydrogens(pH=7.0)

    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return output_pdb


def process_single_pdb(pdb_path, output_dir, hydrogenated_dir, annotations, jaspar_indices):
    """
    Process one PDB file, including PWM fetching.
    """
    pdb_name = os.path.basename(pdb_path)

    pdb_id = os.path.splitext(pdb_name)[0]

    pwm_matrix = get_hybrid_pwm(pdb_id, annotations, jaspar_indices)

    # Polite delay to prevent PDBe API throttling
    time.sleep(0.1)

    hydrogenated_pdb = os.path.join(hydrogenated_dir, pdb_name)

    if not os.path.exists(hydrogenated_pdb):
        print(f"\nHydrogenating: {pdb_name}")
        hydrogenated_pdb = hydrogenate_pdb_reduce(pdb_path, hydrogenated_pdb)

    try:
        structure, protein_residues, dna_pairs = load_and_validate(hydrogenated_pdb)
        centroid = compute_complex_centroid(protein_residues, dna_pairs)

        dna_features = generate_dna_features(dna_pairs, centroid)
        protein_features = generate_protein_features(protein_residues, centroid)
        bond_matrix = generate_bond_matrix(protein_residues, dna_pairs)

        protein_labels = build_protein_labels(protein_residues)
        dna_labels = build_dna_labels(dna_pairs)

        if len(protein_labels) > MAX_PROTEIN_LENGTH:
            raise StructureRejected(f"\nProtein in {pdb_id} longer than {MAX_PROTEIN_LENGTH}")

        # get dna shape features
        dna_shape_features = get_dna_shape_features(hydrogenated_pdb, dna_pairs)

        # ---- DNA Alignment
        seq_fwd_5to3, seq_rev_5to3 = get_sequence_one_hot(dna_labels)
        N_d = len(dna_labels)

        pwm_present = True

        # pwm_matrix = None
        if pwm_matrix is None:
            pwm_present = False

            target_pwm_forward = seq_fwd_5to3.copy()
            target_pwm_reverse = seq_rev_5to3.copy()

            near_fwd, near_rev = generate_spatial_proximity_mask(dna_features, protein_features, distance_threshold_angstroms=5.0)

            # P - bb
            # C1` -bb
            # N1 - base

            # # If a base isn't close to the protein, assign a uniform 0.25 probability background distribution
            target_pwm_forward[~near_fwd] = 0.25
            target_pwm_reverse[~near_rev] = 0.25

            alignment_mask_forward = np.ones(N_d, dtype=bool)
            alignment_mask_reverse = np.ones(N_d, dtype=bool)
        else:
            # do actual alignment and all
            raw_target_pwm = pwm_matrix
            trimmed_pwm = trim_pwm(raw_target_pwm, ic_threshold=0.5)

            opt_i_fwd, opt_j_fwd, opt_k_fwd, score_fwd = ungapped_align(
                seq_fwd_5to3, trimmed_pwm
            )
            opt_i_rev, opt_j_rev, opt_k_rev, score_rev = ungapped_align(
                seq_rev_5to3, trimmed_pwm
            )

            if max(score_fwd, score_rev) == -9999.0:
                raise StructureRejected(
                    "DNA structural strand or PWM is too short to execute alignment cutoff."
                )

            target_pwm_forward = np.full((N_d, 4), 0.25, dtype=np.float32)
            alignment_mask_forward = np.zeros(N_d, dtype=bool)

            if score_fwd >= score_rev:
                # PWM aligned best directly to the forward structural coordinates
                for col in range(opt_k_fwd):
                    target_pwm_forward[opt_j_fwd + col] = trimmed_pwm[opt_i_fwd + col]
                alignment_mask_forward[opt_j_fwd : opt_j_fwd + opt_k_fwd] = True

                # The reverse strand target is natively the reverse complement of the forward map
                target_pwm_reverse = target_pwm_forward[::-1, [3, 2, 1, 0]]
                alignment_mask_reverse = alignment_mask_forward[::-1]
            else:
                # PWM aligned best to the reverse complement 5'-3' coordinate track
                target_pwm_reverse_5to3 = np.full((N_d, 4), 0.25, dtype=np.float32)
                alignment_mask_reverse_5to3 = np.zeros(N_d, dtype=bool)

                for col in range(opt_k_rev):
                    target_pwm_reverse_5to3[opt_j_rev + col] = trimmed_pwm[opt_i_rev + col]
                alignment_mask_reverse_5to3[opt_j_rev : opt_j_rev + opt_k_rev] = True

                # Reconstruct the forward perspective targets via reverse complement mapping
                target_pwm_forward = target_pwm_reverse_5to3[::-1, [3, 2, 1, 0]]
                alignment_mask_forward = alignment_mask_reverse_5to3[::-1]

                target_pwm_reverse = target_pwm_reverse_5to3
                alignment_mask_reverse = alignment_mask_reverse_5to3

            # alignment_mask_forward = np.ones(N_d, dtype=bool)
            # alignment_mask_reverse = np.ones(N_d, dtype=bool)

        # ---- DNA Alignment

        output_path = build_output_path(pdb_path, output_dir)

        # 2. Save everything including the PWM matrix
        save_npz(
            output_path,
            pdb_id,
            dna_features,
            dna_shape_features,
            protein_features,
            bond_matrix,
            protein_labels,
            dna_labels,
            pwm_present,
            target_pwm_forward,
            alignment_mask_forward,
            target_pwm_reverse,
            alignment_mask_reverse,  # Passed to the writer
        )
    finally:
        pass

    # return {
    #     "pdb_id": pdb_id,
    #     "num_residues": protein_features.shape[0],
    #     "num_base_pairs": dna_features.shape[0],
    #     "num_hbonds": int(bond_matrix.sum()),
    #     "pwm_shape": pwm_matrix.shape,
    # }


def process_directory(pdb_dir, output_dir, hydrogenated_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hydrogenated_dir, exist_ok=True)

    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
    total = len(pdb_files)

    success = 0
    rejected = 0
    failed = 0

    rejection_log = []

    print("Building Offline JASPAR Index for Fallback Search...")
    jaspar_indices = build_jaspar_index()

    # Initialize the JASPAR index ONCE before the loop begins
    print("Loading Motif Annotations from specificity_train.json...")
    annotations = load_motif_annotations("../data/specificity_train.json")
    print(f"Loaded {len(annotations)} annotated structures.\n")

    for pdb_path in tqdm(pdb_files):
        try:
            # Pass the indices down to the single PDB processor
            process_single_pdb(pdb_path, output_dir, hydrogenated_dir, annotations, jaspar_indices)
            success += 1

        except StructureRejected as e:
            rejected += 1
            rejection_log.append((os.path.basename(pdb_path), str(e)))
            print(e)

        except Exception:
            failed += 1
            print(f"\nFAILED: {pdb_path}")
            traceback.print_exc()

    print(f"\nTotal files      : {total}")
    print(f"Successful       : {success}")
    print(f"Rejected         : {rejected}")
    print(f"Failed           : {failed}")

    if rejection_log:
        print("\nRejected structures (First 50):")
        for pdb_name, reason in rejection_log[:50]:
            print(f"{pdb_name} -> {reason}")

    return {"total": total, "success": success, "rejected": rejected, "failed": failed}


if __name__ == "__main__":

    PDB_DIRECTORY = "../data/pdbs"
    OUTPUT_DIRECTORY = "../data/processed"
    HYDROGENATED_DIRECTORY = "../data/hydrogenated"

    process_directory(PDB_DIRECTORY, OUTPUT_DIRECTORY, HYDROGENATED_DIRECTORY)
