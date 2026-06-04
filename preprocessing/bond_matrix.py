import numpy as np

from constants import HBOND_MIN_DISTANCE, HBOND_MAX_DISTANCE, HBOND_MIN_ANGLE

from constants import PROTEIN_DONORS, PROTEIN_ACCEPTORS, DNA_DONORS, DNA_ACCEPTORS

from constants import RESIDUE_BASE_CUTOFF

from geometry import distance, calculate_hbond_angle

from hydrogen_builder import get_hydrogen_position


def get_atom_coord(residue, atom_name):

    if atom_name not in residue:
        return None

    return residue[atom_name].coord.astype(np.float32)


def collect_protein_donors(residue):

    residue_name = residue.get_resname().strip()

    donors = []

    if "N" in residue:
        donors.append(("N", residue["N"].coord.astype(np.float32)))

    for atom_name in PROTEIN_DONORS.get(residue_name, []):

        coord = get_atom_coord(residue, atom_name)

        if coord is None:
            continue

        donors.append((atom_name, coord))

    return donors


def collect_protein_acceptors(residue):

    residue_name = residue.get_resname().strip()

    acceptors = []

    if "O" in residue:
        acceptors.append(("O", residue["O"].coord.astype(np.float32)))

    for atom_name in PROTEIN_ACCEPTORS.get(residue_name, []):

        coord = get_atom_coord(residue, atom_name)

        if coord is None:
            continue

        acceptors.append((atom_name, coord))

    return acceptors


def collect_dna_donors(residue, base_letter):

    donors = []

    for atom_name in DNA_DONORS.get(base_letter, []):

        coord = get_atom_coord(residue, atom_name)

        if coord is None:
            continue

        donors.append((atom_name, coord))

    return donors


def collect_dna_acceptors(residue, base_letter):

    acceptors = []

    for atom_name in DNA_ACCEPTORS.get(base_letter, []):

        coord = get_atom_coord(residue, atom_name)

        if coord is None:
            continue

        acceptors.append((atom_name, coord))

    return acceptors


def donor_acceptor_valid(donor_coord, hydrogen_coord, acceptor_coord):

    # ha_distance = distance(
    #     hydrogen_coord,
    #     acceptor_coord
    # )

    da_distance = distance(donor_coord, acceptor_coord)

    if da_distance < HBOND_MIN_DISTANCE:
        return False

    if da_distance > HBOND_MAX_DISTANCE:
        return False

    angle = calculate_hbond_angle(donor_coord, hydrogen_coord, acceptor_coord)

    # if da_distance < 3.2:
    #     print()
    #     print("DIST:", da_distance)
    #     print("ANGLE:", angle)
    #     print("DONOR:", donor_coord)
    #     print("HYDROGEN:", hydrogen_coord)
    #     print("ACCEPTOR:", acceptor_coord)

    if angle < HBOND_MIN_ANGLE:
        return False

    return True


def protein_to_dna_hbond(protein_residue, dna_residue, dna_base):

    donors = collect_protein_donors(protein_residue)

    acceptors = collect_dna_acceptors(dna_residue, dna_base)

    for donor_name, donor_coord in donors:

        hydrogen_coords = get_hydrogen_position(
            protein_residue, donor_name, is_dna=False
        )

        if not hydrogen_coords:
            continue

        for _, acceptor_coord in acceptors:
            for hydrogen_coord in hydrogen_coords:
                if donor_acceptor_valid(donor_coord, hydrogen_coord, acceptor_coord):
                    return True

        # print(
        #     donor_name,
        #     len(hydrogen_coords)
        # )

    return False


def dna_to_protein_hbond(dna_residue, dna_base, protein_residue):

    donors = collect_dna_donors(dna_residue, dna_base)

    acceptors = collect_protein_acceptors(protein_residue)

    for donor_name, donor_coord in donors:

        hydrogen_coords = get_hydrogen_position(dna_residue, donor_name, is_dna=False)

        if not hydrogen_coords:
            continue

        for _, acceptor_coord in acceptors:
            for hydrogen_coord in hydrogen_coords:
                if donor_acceptor_valid(donor_coord, hydrogen_coord, acceptor_coord):
                    return True

    return False


def residue_base_precheck(protein_residue, forward_base, reverse_base):
    """
    Fast distance filter.
    """

    protein_atoms = [atom.coord for atom in protein_residue if atom.element != "H"]

    dna_residues = [forward_base]

    if reverse_base is not None:
        dna_residues.append(reverse_base)

    min_dist = float("inf")

    for dna_residue in dna_residues:

        for p_atom in protein_atoms:

            for d_atom in dna_residue:

                if d_atom.element == "H":
                    continue

                diff = p_atom - d_atom.coord

                dist = np.linalg.norm(diff)

                if dist < min_dist:
                    min_dist = dist

    return min_dist <= RESIDUE_BASE_CUTOFF


def residue_base_hbond(protein_residue, forward_base, reverse_base):
    """
    Binary label.
    """
    if not residue_base_precheck(protein_residue, forward_base, reverse_base):
        return False

    dna_residues = [forward_base]

    if reverse_base is not None:
        dna_residues.append(reverse_base)

    for dna_residue in dna_residues:

        dna_base = dna_residue.get_resname().strip()

        if dna_base.startswith("D"):
            dna_base = dna_base[1:]

        if protein_to_dna_hbond(protein_residue, dna_residue, dna_base):
            return True

        if dna_to_protein_hbond(dna_residue, dna_base, protein_residue):
            return True

    return False


def generate_bond_matrix(protein_residues, dna_pairs):
    """
    Returns [Np, Nd]
    """

    n_protein = len(protein_residues)

    n_dna = len(dna_pairs)

    matrix = np.zeros((n_protein, n_dna), dtype=np.uint8)

    for i, protein_residue in enumerate(protein_residues):

        for j, (forward_base, reverse_base) in enumerate(dna_pairs):

            if residue_base_hbond(protein_residue, forward_base, reverse_base):

                matrix[i, j] = 1

    return matrix
