import argparse
import os
import subprocess
import tempfile
import torch

torch.set_float32_matmul_precision("high")

import sys
from pathlib import Path

# Existing path addition (adds the project root/parent directory)
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Add the preprocessing folder to sys.path so its internal imports can resolve
preprocessing_dir = Path(__file__).resolve().parent / "preprocessing"
sys.path.append(str(preprocessing_dir))

from preprocessing.pdb_parser import load_and_validate, StructureRejected
from preprocessing.coordinate_utils import (
    compute_complex_centroid,
    compute_canonical_rotation,
)
from preprocessing.dna_features import generate_dna_features
from preprocessing.protein_features import generate_protein_features
from preprocessing.bond_matrix import generate_bond_matrix
from preprocessing.get_shape_features import get_dna_shape_features

from architecture.model import DeepSpecificity
from architecture.model_v2_shape import DeepSpecificityWithShape
from architecture.model_v1_shape import DeepSpecificityWithShapeV1
from config import *
from utils import split_dna_features, split_dna_features_no_seq, split_dna_shape_features
from correlation_ploting import interpret_sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

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


def preprocess(pdb_path, device, nohb):
    pdb_name = os.path.basename(pdb_path)
    pdb_id = os.path.splitext(pdb_name)[0]

    if nohb:
        hydrogenated_pdb = pdb_path
    else:
        hydrogenated_pdb = hygrogenate_pdb(pdb_path)

    try:
        structure, protein_residues, dna_pairs = load_and_validate(hydrogenated_pdb)
        centroid  = compute_complex_centroid(protein_residues, dna_pairs)
        rotation  = compute_canonical_rotation(protein_residues, dna_pairs, centroid)
        
        dna_features = generate_dna_features(
            dna_pairs,
            protein_residues,
            centroid,
            rotation=rotation
        )

        protein_features = generate_protein_features(
            protein_residues,
            dna_pairs,
            centroid,
            rotation=rotation
        )
        
        if nohb:
            bond_matrix = torch.zeros((len(protein_residues), len(dna_pairs)))
        else:
            bond_matrix = generate_bond_matrix(protein_residues, dna_pairs)

        protein_labels = build_protein_labels(protein_residues)
        dna_labels = build_dna_labels(dna_pairs)

        dna_shape_features = get_dna_shape_features(hydrogenated_pdb, dna_pairs)
    finally:
        if not nohb:
            if os.path.exists(hydrogenated_pdb):
                os.remove(hydrogenated_pdb)

    return {
        "pdb_id": pdb_id,
        "dna_features": torch.tensor(dna_features, dtype=torch.float32).to(device),
        "dna_shape_features": torch.tensor(dna_shape_features, dtype=torch.float32).to(device),
        "protein_features": torch.tensor(protein_features, dtype=torch.float32).to(
            device
        ),
        "bond_matrix": torch.tensor(bond_matrix, dtype=torch.uint8).to(device),
        "protein_labels": protein_labels,
        "dna_labels": dna_labels,
    }


def inference_model(data, device, checkpoint_path, v2):
    model = DeepSpecificity(
        len_dna_features=DNA_FEATURE_DIM,
        len_prot_features=PROTEIN_FEATURE_DIM,
        d_model=D_MODEL,
        n_head_dna=N_HEAD_DNA,
        n_enc_dna=N_ENC_DNA,
        n_head_prot=N_HEAD_PROT,
        n_enc_prot=N_ENC_PROT,
        n_cross_att_heads=N_CROSS_HEADS,
        n_enc_pwm=N_ENC_PWM,
        n_head_pwm=N_HEAD_PWM,
    ).to(device)
    model = torch.compile(model)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if v2:
        dna_fwd, dna_rc = split_dna_features_no_seq(data["dna_features"])
    else:
        dna_fwd, dna_rc = split_dna_features(data["dna_features"])

    protein_features = data["protein_features"]

    protein_features = protein_features.unsqueeze(0)
    dna_fwd = dna_fwd.unsqueeze(0)
    dna_rc = dna_rc.unsqueeze(0)

    with torch.no_grad():
        pred_fwd = model(dna_fwd, protein_features)
        pred_rc = model(dna_rc, protein_features)

        pred_fwd = pred_fwd.squeeze(0)
        pred_rc = pred_rc.squeeze(0)

        pred_fwd = torch.softmax(pred_fwd, dim=-1).cpu().numpy()
        pred_rc = torch.softmax(pred_rc, dim=-1).cpu().numpy()

    return pred_fwd, pred_rc

def inference_model_shape(data, device, checkpoint_path, amap, is_v2):
    if is_v2:
        model = DeepSpecificityWithShape(
            len_dna_features=DNA_FEATURE_DIM,
            len_prot_features=PROTEIN_FEATURE_DIM,
            len_dna_shape_features=DNA_SHAPE_FEATURES_DIM,
            d_model=D_MODEL,
            n_head_dna=N_HEAD_DNA,
            n_enc_dna=N_ENC_DNA,
            n_head_prot=N_HEAD_PROT,
            n_enc_prot=N_ENC_PROT,
            n_cross_att_heads=N_CROSS_HEADS,
            n_enc_pwm=N_ENC_PWM,
            n_head_pwm=N_HEAD_PWM,
        ).to(device)
    else:
        pass
        model = DeepSpecificityWithShapeV1(
            len_dna_features=DNA_FEATURE_DIM,
            len_prot_features=PROTEIN_FEATURE_DIM,
            len_dna_shape_features=DNA_SHAPE_FEATURES_DIM,
            d_model=D_MODEL,
            n_head_dna=N_HEAD_DNA,
            n_enc_dna=N_ENC_DNA,
            n_head_prot=N_HEAD_PROT,
            n_enc_prot=N_ENC_PROT,
            n_cross_att_heads=N_CROSS_HEADS,
            n_enc_pwm=N_ENC_PWM,
            n_head_pwm=N_HEAD_PWM,
        ).to(device)

    model = torch.compile(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dna_fwd, dna_rc = split_dna_features_no_seq(data["dna_features"])
    dna_shape_features_fwd, dna_shape_features_rev = split_dna_shape_features(
        data["dna_shape_features"]
    )

    protein_features = data["protein_features"]

    protein_features = protein_features.unsqueeze(0)
    dna_fwd = dna_fwd.unsqueeze(0)
    dna_rc = dna_rc.unsqueeze(0)
    dna_shape_features_fwd = dna_shape_features_fwd.unsqueeze(0)
    dna_shape_features_rev = dna_shape_features_rev.unsqueeze(0)

    with torch.no_grad():
        pred_fwd = model(dna_fwd, dna_shape_features_fwd, protein_features)
        pred_rc = model(dna_rc, dna_shape_features_rev, protein_features)

        pred_fwd = pred_fwd.squeeze(0)
        pred_rc = pred_rc.squeeze(0)

        pred_fwd = torch.softmax(pred_fwd, dim=-1).cpu().numpy()
        pred_rc = torch.softmax(pred_rc, dim=-1).cpu().numpy()

    if amap:
        interpret_sample(model, data)

    return pred_fwd, pred_rc


def ppm_to_ic(ppm):
    ppm = np.clip(ppm, 1e-6, 1.0)
    entropy = -np.sum(ppm * np.log2(ppm), axis=1)
    ic = 2.0 - entropy

    return ppm * ic[:, None]


def ppm_to_pwm(ppm, background=0.25):
    ppm = np.clip(ppm, 1e-6, 1.0)
    return np.log2(ppm / background)


def plot_bonded_sequence_logo(
    ppm,
    bond_matrix,
    protein_labels,
    dna_labels,
    title=None
):
    columns = ["A", "C", "G", "T"]
    df = pd.DataFrame(ppm_to_ic(ppm), columns=columns)
    fig, ax = plt.subplots(figsize=(14, 3))

    bond_counts = bond_matrix.sum(axis=0)
    max_bonds = max(bond_counts.max(), 1)

    for idx in range(len(bond_counts)):
        if bond_counts[idx] == 0:
            continue

        alpha = bond_counts[idx] / max_bonds
        alpha = max(alpha, 0.15)

        ax.axvspan(idx - 0.5, idx + 0.5, color="red", alpha=alpha, zorder=0)

    logo = logomaker.Logo(df, ax=ax)
    # print("Maximum stack height:", ppm_to_ic(ppm).sum(axis=1).max())

    ax.set_xlabel("DNA Position")

    ax.set_ylabel("bits")

    if title is not None:

        ax.set_title(title)

    plt.tight_layout()

    print("\nDetected hydrogen bonds")
    print("-" * 50)

    bond_count = 0

    for i in range(bond_matrix.shape[0]):

        for j in range(bond_matrix.shape[1]):

            if bond_matrix[i, j] == 1:

                bond_count += 1

                print(
                    f"[{i:3d}, {j:3d}] "
                    f"{protein_labels[i]} "
                    f"<--> "
                    f"{dna_labels[j]}"
                )

    print("-" * 50)
    print(f"Total bonds: {bond_count}")

    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Inference pipeline for DeepSpecificity"
    )

    parser.add_argument("--pdb", type=str, help="path to pdb file")
    parser.add_argument("--checkpoint", type=str, help="path to the checkpoint file")
    # parser.add_argument("--out_dir", type=str, help="out dir for pwm png")
    parser.add_argument("--v2", action="store_true")
    parser.add_argument("--shape", action="store_true")
    parser.add_argument("--amap", action="store_true")
    parser.add_argument("--nohb", action="store_true")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = preprocess(args.pdb, device, args.nohb)
    if args.shape:
        ppm_fwd, ppm_rc = inference_model_shape(data, device, args.checkpoint, args.amap, args.v2)
    else:
        ppm_fwd, ppm_rc = inference_model(data, device, args.checkpoint, args.v2)

    plot_bonded_sequence_logo(
        ppm_fwd, 
        data["bond_matrix"].cpu().numpy(),
        protein_labels=data["protein_labels"],
        dna_labels=data["dna_labels"],
        title=f"{data['pdb_id']} fwd"
    )
    plot_bonded_sequence_logo(
        ppm_rc,
        data["bond_matrix"].cpu().numpy(),
        protein_labels=data["protein_labels"],
        dna_labels=data["dna_labels"],
        title=f"{data['pdb_id']} rev",
    )
    plt.show()


if __name__ == "__main__":
    main()
