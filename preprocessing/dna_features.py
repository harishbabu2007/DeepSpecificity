import numpy as np

from constants import DNA_BACKBONE_ORDER, MAX_DNA_BASE_HEAVY_ATOMS, BASE_TO_INDEX

from dna_definitions import BASE_HEAVY_ATOMS, BASE_NAME_MAP

from geometry import pad_coordinate_list

from coordinate_utils import transform_coordinate


def one_hot_base(base):

    vec = np.zeros(4, dtype=np.float32)

    vec[BASE_TO_INDEX[base]] = 1.0

    return vec


def get_base_letter(residue):

    return BASE_NAME_MAP[residue.get_resname().strip()]


def extract_backbone_coordinates(residue, centroid):
    """
    Returns 33 values.
    """

    coords = []

    for atom_name in DNA_BACKBONE_ORDER:

        if atom_name in residue:

            coords.append(
                transform_coordinate(
                    residue[atom_name].coord.astype(np.float32), centroid
                )
            )

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, len(DNA_BACKBONE_ORDER))


def extract_base_coordinates(residue, base_letter, centroid):
    """
    Returns 33 values.
    """

    coords = []

    atom_order = BASE_HEAVY_ATOMS[base_letter]

    for atom_name in atom_order:

        if atom_name in residue:

            coords.append(
                transform_coordinate(
                    residue[atom_name].coord.astype(np.float32), centroid
                )
            )

        else:

            coords.append(np.zeros(3, dtype=np.float32))

    return pad_coordinate_list(coords, MAX_DNA_BASE_HEAVY_ATOMS)


def build_single_base_features(residue, centroid):

    base = get_base_letter(residue)

    one_hot = one_hot_base(base)

    backbone = extract_backbone_coordinates(residue, centroid)

    base_atoms = extract_base_coordinates(residue, base, centroid)

    return np.concatenate([one_hot, backbone, base_atoms])


def build_paired_base_features(forward_residue, reverse_residue, centroid):
    """
    Returns 140 values.
    """

    forward_features = build_single_base_features(forward_residue, centroid)

    if reverse_residue is None:

        reverse_features = np.zeros(70, dtype=np.float32)

    else:

        reverse_features = build_single_base_features(reverse_residue, centroid)

    feature_vector = np.concatenate([forward_features, reverse_features])

    assert len(feature_vector) == 140

    return feature_vector


def generate_dna_features(dna_pairs, centroid):
    """
    Returns [Nd, 140]
    """

    rows = []

    for forward_residue, reverse_residue in dna_pairs:

        row = build_paired_base_features(forward_residue, reverse_residue, centroid)

        rows.append(row)

    if len(rows) == 0:

        return np.zeros((0, 140), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)
