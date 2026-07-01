import numpy as np

def get_data_from_npz(file_path: str):
    with np.load(file_path) as data:
        return (
            data["pdb_id"],
            data["dna_features"],
            data["dna_shape_features"],
            data["protein_features"],
            data["bond_matrix"],
            data["distance_matrix"],
            data["protein_labels"],
            data["dna_labels"],
            data["pwm_present"],
            data["target_pwm_forward"],
            data["alignment_mask_forward"],
            data["target_pwm_reverse"],
            data["alignment_mask_reverse"],
        )
