from torch.utils.data import Dataset
import numpy as np
import torch
import os
from npz_loader import get_data_from_npz


class PDNADataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_names = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_path = os.path.join(self.data_dir, self.file_names[index])

        (
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
            alignment_mask_reverse,
        ) = get_data_from_npz(file_path)

        return {
            "pdb_id": pdb_id,
            "dna_features": torch.tensor(dna_features, dtype=torch.float32),
            "dna_shape_features": torch.tensor(dna_shape_features, dtype=torch.float32),
            "protein_features": torch.tensor(protein_features, dtype=torch.float32),
            "bond_matrix": torch.tensor(bond_matrix, dtype=torch.uint8),
            "protein_labels": protein_labels,
            "dna_labels": dna_labels,
            "pwm_present": pwm_present.item(),
            "target_pwm_forward": torch.tensor(target_pwm_forward, dtype=torch.float32),
            "alignment_mask_forward": torch.tensor(
                alignment_mask_forward, dtype=torch.bool
            ),
            "target_pwm_reverse": torch.tensor(target_pwm_reverse, dtype=torch.float32),
            "alignment_mask_reverse": torch.tensor(
                alignment_mask_reverse, dtype=torch.bool
            ),
        }
