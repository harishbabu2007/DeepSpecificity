import numpy as np

from constants import (
    AA_TO_INDEX,
    PROTEIN_BACKBONE_ORDER,
    MAX_PROTEIN_SIDECHAIN_HEAVY_ATOMS,
)

from residue_definitions import SIDECHAIN_ATOMS

from geometry import pad_coordinate_list

from coordinate_utils import transform_coordinate, transform_coordinate_canonical


def one_hot_residue(residue_name):

    vec = np.zeros(20, dtype=np.float32)

    vec[AA_TO_INDEX[residue_name]] = 1.0

    return vec


def extract_backbone_coordinates(residue, centroid, rotation=None):
    """
    Returns 12 values.
    If rotation is provided, applies canonical rotation before scaling.
    """

    coords = []

    for atom_name in PROTEIN_BACKBONE_ORDER:

        if atom_name in residue:

            raw = residue[atom_name].coord.astype(np.float32)

            if rotation is not None:
                coords.append(transform_coordinate_canonical(raw, centroid, rotation))
            else:
                coords.append(transform_coordinate(raw, centroid))

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, len(PROTEIN_BACKBONE_ORDER))


def extract_sidechain_coordinates(residue, centroid, rotation=None):
    """
    Returns 30 values.
    If rotation is provided, applies canonical rotation before scaling.
    """

    coords = []

    residue_name = residue.get_resname().strip()

    atom_order = SIDECHAIN_ATOMS[residue_name]

    for atom_name in atom_order:

        if atom_name in residue:

            raw = residue[atom_name].coord.astype(np.float32)

            if rotation is not None:
                coords.append(transform_coordinate_canonical(raw, centroid, rotation))
            else:
                coords.append(transform_coordinate(raw, centroid))

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, MAX_PROTEIN_SIDECHAIN_HEAVY_ATOMS)


def build_residue_features(residue, centroid, rotation=None):
    """
    Returns 62 values.
    """

    residue_name = residue.get_resname().strip()

    one_hot = one_hot_residue(residue_name)

    backbone = extract_backbone_coordinates(residue, centroid, rotation)

    sidechain = extract_sidechain_coordinates(residue, centroid, rotation)

    feature_vector = np.concatenate([one_hot, backbone, sidechain])

    assert len(feature_vector) == 62

    return feature_vector


def generate_protein_features(protein_residues, centroid, rotation=None):
    """
    Returns [Np, 62].
    Pass rotation=R to apply canonical orientation.
    Pass rotation=None (default) to use original behaviour.
    """

    rows = []

    for residue in protein_residues:

        rows.append(build_residue_features(residue, centroid, rotation))

    if len(rows) == 0:

        return np.zeros((0, 62), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)
