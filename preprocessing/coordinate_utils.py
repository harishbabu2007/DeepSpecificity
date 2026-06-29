import numpy as np

from constants import COORDINATE_SCALE_FACTOR
from geometry import get_dna_c1_atom

def compute_complex_centroid(protein_residues, dna_pairs):
    """
    Compute centroid using all
    protein and DNA atoms.
    """

    coords = []

    for residue in protein_residues:

        for atom in residue:

            if atom.element == "H":
                continue

            coords.append(atom.coord)

    for forward_residue, reverse_residue in dna_pairs:

        for atom in forward_residue:

            if atom.element == "H":
                continue

            coords.append(atom.coord)

        if reverse_residue is not None:

            for atom in reverse_residue:

                if atom.element == "H":
                    continue

                coords.append(atom.coord)

    if len(coords) == 0:
        raise ValueError("No coordinates found")

    coords = np.asarray(coords, dtype=np.float32)

    return np.mean(coords, axis=0)


def transform_coordinate(coordinate, centroid):
    """
    Center and scale coordinate.
    Original function — untouched.
    """

    return (coordinate - centroid) / COORDINATE_SCALE_FACTOR


def transform_coordinate_list(coordinates, centroid):
    """
    Apply transform to multiple coordinates.
    """

    return [transform_coordinate(coord, centroid) for coord in coordinates]


def zero_coord():
    """
    Standard padding coordinate.
    """

    return np.zeros(3, dtype=np.float32)


def rotation_about_z(theta):

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )


def compute_canonical_rotation(protein_residues, dna_pairs, centroid):
    """
    Computes a canonical 3x3 rotation matrix R that standardises the
    orientation of every protein-DNA complex into the same reference frame:
    """

    c1_coords = []

    for forward_res, reverse_res in dna_pairs:
        for residue in [forward_res, reverse_res]:
            if residue is None:
                continue

            c1_atom = get_dna_c1_atom(residue)

            if c1_atom is not None:
                coord = c1_atom.coord.astype(np.float32)
                c1_coords.append(coord - centroid)

    if len(c1_coords) < 3:
        # Not enough atoms to run PCA
        return np.eye(3, dtype=np.float32)

    c1_coords = np.array(c1_coords, dtype=np.float32)  # shape (N, 3)

    _, _, Vt = np.linalg.svd(c1_coords, full_matrices=False)

    pc1 = Vt[0]  # DNA long axis  (largest variance)
    pc2 = Vt[1]  # perpendicular spread (second largest)

    first_bp = c1_coords[0]
    last_bp = c1_coords[-1]

    dna_direction = last_bp - first_bp

    if np.dot(dna_direction, pc1) < 0:
        pc1 *= -1.0

    pc3 = np.cross(pc1, pc2)
    pc3 = pc3 / (np.linalg.norm(pc3) + 1e-8)

    # new_Z = pc1,  new_X = pc2,  new_Y = pc3
    # R maps original coords to the new frame:  new_coord = R @ old_coord
    # Rows of R = new basis vectors expressed in the original frame
    R = np.stack([pc2, pc3, pc1], axis=0).astype(np.float32)  # shape (3, 3)
    # R[2] *= -1.0
    protein_coords = []

    for residue in protein_residues:

        for atom in residue:

            if atom.element == "H":
                continue

            protein_coords.append(atom.coord.astype(np.float32) - centroid)

    if len(protein_coords) > 0:
        protein_com = np.mean(protein_coords, axis=0)

        protein_com_rotated = R @ protein_com

        theta = np.arctan2(
            protein_com_rotated[1],
            protein_com_rotated[0]
        )

        R = rotation_about_z(-theta) @ R

    return R


def transform_coordinate_canonical(coordinate, centroid, rotation):

    centred = coordinate.astype(np.float32) - centroid
    rotated = rotation @ centred
    return (rotated / COORDINATE_SCALE_FACTOR).astype(np.float32)
