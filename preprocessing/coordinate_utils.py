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


# ---------------------------------------------------------------------------
# Canonical Rotation — NEW, does not affect any existing code above
# ---------------------------------------------------------------------------


def compute_canonical_rotation(protein_residues, dna_pairs, centroid):
    """
    Computes a canonical 3x3 rotation matrix R that standardises the
    orientation of every protein-DNA complex into the same reference frame:

        - DNA long axis  →  Z axis
        - Protein centre →  positive X side

    Steps
    -----
    1. Collect all DNA C1' heavy-atom coordinates and centre them.
    2. Run PCA on those coordinates.
       PC1 (max variance) captures the DNA long axis  → becomes Z.
       PC2 (2nd variance) captures the perpendicular spread → becomes X.
       PC3 is the remaining orthogonal direction         → becomes Y.
    3. Build rotation matrix  R = [PC2, PC3, PC1]  (rows = new axes).
    4. Disambiguate sign ambiguity: if the protein centre of mass ends up
       on the negative-X side after rotation, flip the X axis.

    Parameters
    ----------
    protein_residues : list of Bio.PDB Residue objects
    dna_pairs        : list of (forward_residue, reverse_residue) tuples
    centroid         : np.ndarray shape (3,) — the complex centroid already
                       computed by compute_complex_centroid()

    Returns
    -------
    R : np.ndarray shape (3, 3), dtype float32
        Rotation matrix.  Apply as:  rotated = R @ (coord - centroid)
    """

    # ------------------------------------------------------------------
    # Step 1 — collect DNA C1' coordinates, centred at the complex centroid
    # ------------------------------------------------------------------
    c1_coords = []

    for forward_res, reverse_res in dna_pairs:

        for residue in [forward_res, reverse_res]:

            if residue is None:
                continue

            if "C1'" in residue:
                coord = residue["C1'"].coord.astype(np.float32)
                c1_coords.append(coord - centroid)  # centre only, no scale yet

    if len(c1_coords) < 3:
        # Not enough atoms to run PCA — return identity (no rotation)
        return np.eye(3, dtype=np.float32)

    c1_coords = np.array(c1_coords, dtype=np.float32)  # shape (N, 3)

    # ------------------------------------------------------------------
    # Step 2 — PCA via SVD on the centred C1' coordinates
    # ------------------------------------------------------------------
    # np.linalg.svd on the data matrix directly gives us the principal axes
    # as the rows of Vt.  Eigenvalues (singular values) are sorted descending.
    _, _, Vt = np.linalg.svd(c1_coords, full_matrices=False)

    # Vt rows are principal components, sorted largest variance first
    pc1 = Vt[0]  # DNA long axis  (largest variance)
    pc2 = Vt[1]  # perpendicular spread (second largest)
    pc3 = np.cross(pc1, pc2)  # guaranteed orthogonal, right-handed system

    # Re-normalise (SVD already gives unit vectors, but cross product may drift)
    pc3 = pc3 / (np.linalg.norm(pc3) + 1e-8)

    # ------------------------------------------------------------------
    # Step 3 — Build rotation matrix
    # We want:   new_Z = pc1,  new_X = pc2,  new_Y = pc3
    # R maps original coords to the new frame:  new_coord = R @ old_coord
    # Rows of R = new basis vectors expressed in the original frame
    # ------------------------------------------------------------------
    R = np.stack([pc2, pc3, pc1], axis=0).astype(np.float32)  # shape (3, 3)
    # R[2] *= -1.0
    # ------------------------------------------------------------------
    # Step 4 — Disambiguate sign: protein should be on positive-X side
    # ------------------------------------------------------------------
    protein_coords = []

    for residue in protein_residues:

        for atom in residue:

            if atom.element == "H":
                continue

            protein_coords.append(atom.coord.astype(np.float32) - centroid)

    if len(protein_coords) > 0:

        protein_com = np.mean(protein_coords, axis=0)  # centre of mass, centred
        protein_com_rotated = R @ protein_com

        if protein_com_rotated[0] < 0.0:
            # Flip the X axis row (row 0) to push protein to positive X
            R[0] *= -1.0

    return R


def transform_coordinate_canonical(coordinate, centroid, rotation):
    """
    NEW function — centre, rotate into canonical frame, then scale.

    Equivalent pipeline:
        centred  = coord - centroid
        rotated  = rotation @ centred
        scaled   = rotated / COORDINATE_SCALE_FACTOR

    Parameters
    ----------
    coordinate : np.ndarray shape (3,)
    centroid   : np.ndarray shape (3,)
    rotation   : np.ndarray shape (3, 3) from compute_canonical_rotation()

    Returns
    -------
    np.ndarray shape (3,), dtype float32
    """

    centred = coordinate.astype(np.float32) - centroid
    rotated = rotation @ centred
    return (rotated / COORDINATE_SCALE_FACTOR).astype(np.float32)
