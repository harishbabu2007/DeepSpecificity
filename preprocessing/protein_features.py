import numpy as np

from constants import (
    AA_TO_INDEX,
    PROTEIN_BACKBONE_ORDER,
    MAX_PROTEIN_SIDECHAIN_HEAVY_ATOMS,
)

from residue_definitions import SIDECHAIN_ATOMS

from geometry import pad_coordinate_list

from coordinate_utils import transform_coordinate


def one_hot_residue(residue_name):

    vec = np.zeros(20, dtype=np.float32)

    vec[AA_TO_INDEX[residue_name]] = 1.0

    return vec


def extract_backbone_coordinates(residue, centroid):
    """
    Returns 12 values.
    """

    coords = []

    for atom_name in PROTEIN_BACKBONE_ORDER:

        if atom_name in residue:

            coords.append(
                transform_coordinate(
                    residue[atom_name].coord.astype(np.float32), centroid
                )
            )

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, len(PROTEIN_BACKBONE_ORDER))


def extract_sidechain_coordinates(residue, centroid):
    """
    Returns 30 values.
    """

    coords = []

    residue_name = residue.get_resname().strip()

    atom_order = SIDECHAIN_ATOMS[residue_name]

    for atom_name in atom_order:

        if atom_name in residue:

            coords.append(
                transform_coordinate(
                    residue[atom_name].coord.astype(np.float32), centroid
                )
            )

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, MAX_PROTEIN_SIDECHAIN_HEAVY_ATOMS)


def build_residue_features(residue, centroid):
    """
    Returns 62 values.
    """

    residue_name = residue.get_resname().strip()

    one_hot = one_hot_residue(residue_name)

    backbone = extract_backbone_coordinates(residue, centroid)

    sidechain = extract_sidechain_coordinates(residue, centroid)

    feature_vector = np.concatenate([one_hot, backbone, sidechain])

    assert len(feature_vector) == 62

    return feature_vector


def generate_protein_features(protein_residues, centroid):
    """
    Returns [Np, 62]
    """

    rows = []

    for residue in protein_residues:

        rows.append(build_residue_features(residue, centroid))

    if len(rows) == 0:

        return np.zeros((0, 62), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)
