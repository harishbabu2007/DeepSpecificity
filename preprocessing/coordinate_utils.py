import numpy as np

from constants import COORDINATE_SCALE_FACTOR


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
