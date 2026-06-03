import argparse
import os
import subprocess
import tempfile
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.pdb_parser import load_and_validate, StructureRejected
from preprocessing.coordinate_utils import compute_complex_centroid
from preprocessing.dna_features import generate_dna_features
from preprocessing.protein_features import generate_protein_features
from preprocessing.bond_matrix import generate_bond_matrix

def hygrogenate_pdb(input_pdb):
    result = subprocess.run(["reduce", input_pdb], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Reduce failed:\n{result.stderr}")

    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".pdb", delete=False) as f:
        f.write(result.stdout)
        f.flush()

        return f.name


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


def preprocess(pdb_path, device):
    pdb_name = os.path.basename(pdb_path)
    pdb_id = os.path.splitext(pdb_name)[0]

    hydrogenated_pdb = hygrogenate_pdb(pdb_path)

    try:
        structure, protein_residues, dna_pairs = load_and_validate(hydrogenated_pdb)
        centroid = compute_complex_centroid(protein_residues, dna_pairs)

        dna_features = generate_dna_features(dna_pairs, centroid)
        protein_features = generate_protein_features(protein_residues, centroid)
        bond_matrix = generate_bond_matrix(protein_residues, dna_pairs)

        protein_labels = build_protein_labels(protein_residues)
        dna_labels = build_dna_labels(dna_pairs)
    finally:
        if os.path.exists(hydrogenated_pdb):
            os.remove(hydrogenated_pdb)

    return {
        "pdb_id": pdb_id,
        "dna_features": torch.tensor(dna_features, dtype=torch.float32).to(device),
        "protein_features": torch.tensor(protein_features, dtype=torch.float32).to(device),
        "bond_matrix": torch.tensor(bond_matrix, dtype=torch.uint8).to(device),
        "protein_labels": protein_labels,
        "dna_labels": dna_labels
    }

def inference_model(data):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline for DeepSpecificity"
    )

    parser.add_argument("--pdb", type=str, help="path to pdb file")
    parser.add_argument("--out_dir", type=str, help="out dir for pwm png")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = preprocess(args.pdb, device)


if __name__ == "__main__":
    main()
