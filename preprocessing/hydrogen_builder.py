import numpy as np

from geometry import unit_vector


def get_existing_hydrogens(residue, donor_atom_name):

    donor_coord = residue[donor_atom_name].coord

    hydrogens = []

    for atom in residue:

        if atom.element != "H":
            continue

        dist = np.linalg.norm(atom.coord - donor_coord)

        if dist < 1.5:

            hydrogens.append(atom.coord.astype(np.float32))

    return hydrogens


def place_single_hydrogen(donor_coord, reference_coord):
    """
    Place hydrogen opposite
    the bonded heavy atom.
    """

    direction = donor_coord - reference_coord

    direction = unit_vector(direction)

    bond_length = 1.0

    return (donor_coord + direction * bond_length).astype(np.float32)


def get_reference_atom(residue, donor_atom_name):
    """
    Returns heavy atom bonded
    to donor.
    """

    residue_name = residue.get_resname().strip()

    mapping = {
        ("LYS", "NZ"): "CE",
        ("ASN", "ND2"): "CG",
        ("GLN", "NE2"): "CD",
        ("SER", "OG"): "CB",
        ("THR", "OG1"): "CB",
        ("TYR", "OH"): "CZ",
        ("TRP", "NE1"): "CD1",
        ("HIS", "ND1"): "CG",
        ("HIS", "NE2"): "CE1",
        ("ARG", "NE"): "CZ",
        ("ARG", "NH1"): "CZ",
        ("ARG", "NH2"): "CZ",
        ("BACKBONE", "N"): "CA",
    }

    key = (residue_name, donor_atom_name)

    if key not in mapping:
        return None

    atom_name = mapping[key]

    if atom_name not in residue:
        return None

    return residue[atom_name].coord.astype(np.float32)


def build_protein_hydrogen(residue, donor_atom_name):
    """
    HBPLUS-inspired placement.
    """

    existing = get_existing_hydrogens(residue, donor_atom_name)

    return existing

    # if donor_atom_name not in residue:
    #     return None

    # donor_coord = (
    #     residue[
    #         donor_atom_name
    #     ].coord.astype(
    #         np.float32
    #     )
    # )

    # reference_coord = (
    #     get_reference_atom(
    #         residue,
    #         donor_atom_name
    #     )
    # )

    # if reference_coord is None:
    #     return None

    # return place_single_hydrogen(
    #     donor_coord,
    #     reference_coord
    # )


def build_dna_hydrogen(residue, donor_atom_name):
    """
    Simple nucleotide donor placement.
    """

    existing = get_existing_hydrogens(residue, donor_atom_name)

    # if existing is not None:
    #     return existing

    return existing

    # donor_coord = (
    #     residue[
    #         donor_atom_name
    #     ].coord.astype(
    #         np.float32
    #     )
    # )

    # residue_name = (
    #     residue.get_resname()
    #     .strip()
    # )

    # references = {

    #     ("DA", "N6"): "C6",
    #     ("A", "N6"): "C6",

    #     ("DG", "N2"): "C2",
    #     ("G", "N2"): "C2",

    #     ("DC", "N4"): "C4",
    #     ("C", "N4"): "C4",
    # }

    # key = (
    #     residue_name,
    #     donor_atom_name
    # )

    # if key not in references:
    #     return None

    # ref_atom = references[key]

    # if ref_atom not in residue:
    #     return None

    # reference_coord = (
    #     residue[
    #         ref_atom
    #     ].coord.astype(
    #         np.float32
    #     )
    # )

    # return place_single_hydrogen(
    #     donor_coord,
    #     reference_coord
    # )


def get_hydrogen_position(residue, donor_atom_name, is_dna=False):
    """
    Main public API.
    """

    if is_dna:

        return build_dna_hydrogen(residue, donor_atom_name)

    return build_protein_hydrogen(residue, donor_atom_name)
