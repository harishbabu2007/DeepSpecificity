import os
import numpy as np

from constants import STORE_DTYPE_FEATURES, STORE_DTYPE_BOND_MATRIX


def save_npz(
    output_path,
    pdb_id,
    dna_features,
    dna_shape_features,
    protein_features,
    bond_matrix,
    distance_matrix,
    protein_labels,
    dna_labels,

    pwm_present,

    target_pwm_forward,
    alignment_mask_forward,
    target_pwm_reverse,
    alignment_mask_reverse,
):
    """
    Save single dataset sample including DeepPBS alignment targets and masks.
    """
    np.savez_compressed(
        output_path,
        pdb_id=pdb_id,
        dna_features=dna_features.astype(STORE_DTYPE_FEATURES),
        dna_shape_features=dna_shape_features.astype(STORE_DTYPE_FEATURES),
        protein_features=protein_features.astype(STORE_DTYPE_FEATURES),
        bond_matrix=bond_matrix.astype(STORE_DTYPE_BOND_MATRIX),
        distance_matrix=distance_matrix.astype(STORE_DTYPE_FEATURES),
        protein_labels=np.array(protein_labels, dtype=str),
        dna_labels=np.array(dna_labels, dtype=str),
        pwm_present=np.array(pwm_present, dtype=bool),
        target_pwm_forward=target_pwm_forward.astype(np.float32),
        alignment_mask_forward=alignment_mask_forward.astype(bool),
        target_pwm_reverse=target_pwm_reverse.astype(np.float32),
        alignment_mask_reverse=alignment_mask_reverse.astype(bool),
    )


def build_output_path(pdb_path, output_dir):
    pdb_name = os.path.basename(pdb_path)
    pdb_name = os.path.splitext(pdb_name)[0]
    return os.path.join(output_dir, f"{pdb_name}.npz")
