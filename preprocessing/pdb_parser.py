from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from .constants import (
    DNA_RESIDUES,
    REJECT_RESIDUES,
    REJECT_MULTI_MODEL,
    REJECT_ALTERNATE_LOCATIONS,
)

from .dna_definitions import BASE_NAME_MAP, CANONICAL_PAIRS
from .dna_definitions import WC_PAIR_ATOMS
from .constants import WC_PAIR_DISTANCE_THRESHOLD

from .geometry import distance


class StructureRejected(Exception):
    pass


def load_structure(pdb_path):
    """
    Load structure from PDB file.
    """

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("complex", pdb_path)

    return structure


def resolve_disordered_atoms(model):

    for chain in model:

        for residue in chain:

            for atom in residue:

                if not atom.is_disordered():
                    continue

                occupancies = []

                for alt_atom in atom.child_dict.values():

                    occ = alt_atom.get_occupancy()

                    if occ is None:
                        occ = 0.0

                    occupancies.append((occ, alt_atom.get_altloc()))

                occupancies.sort(reverse=True)

                if len(occupancies) > 1 and occupancies[0][0] == occupancies[1][0]:
                    raise StructureRejected("Ambiguous alternate location occupancy")

                atom.disordered_select(occupancies[0][1])


def validate_structure(structure):
    """
    Apply dataset filtering rules.
    """

    if REJECT_MULTI_MODEL:

        if len(structure) != 1:
            raise StructureRejected("Multiple models present")

    model = structure[0]

    check_modified_residues(model)
    # check_alternate_locations(model)
    # resolve_disordered_atoms(model)

    return True


def check_modified_residues(model):

    for chain in model:

        for residue in chain:

            resname = residue.get_resname().strip()

            if resname in REJECT_RESIDUES:
                raise StructureRejected(f"Modified residue found: {resname}")


def check_alternate_locations(model):
    """
    Accept alternate locations.

    Biopython automatically keeps
    the highest occupancy conformer
    when atom coordinates are accessed.
    """
    return


def extract_protein_chains(model):
    """
    Returns chains containing protein.
    """

    protein_chains = []

    for chain in model:

        residues = list(chain)

        protein_count = 0

        for residue in residues:

            resname = residue.get_resname()

            if is_aa(residue, standard=True):
                protein_count += 1

        if protein_count > 0:
            protein_chains.append(chain)

    return protein_chains


def extract_dna_chains(model):
    """
    Returns DNA chains sorted by size.
    """

    dna_chains = []

    for chain in model:

        dna_residues = []

        for residue in chain:

            resname = residue.get_resname().strip()

            if resname in DNA_RESIDUES:
                dna_residues.append(residue)

        if dna_residues:
            dna_chains.append((chain, len(dna_residues)))

    dna_chains.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in dna_chains]


def extract_protein_residues(model):

    residues = []

    for chain in extract_protein_chains(model):

        for residue in chain:

            if residue.id[0] != " ":
                continue

            if is_aa(residue, standard=True):
                residues.append(residue)

    return residues


def extract_dna_residues(model):

    residues = []

    for chain in extract_dna_chains(model):

        for residue in chain:

            resname = residue.get_resname().strip()

            if resname in DNA_RESIDUES:
                residues.append(residue)

    return residues


def compute_wc_pair_score(residue_a, residue_b):
    """
    Lower score = better Watson-Crick pair.
    """

    base_a = get_base_letter(residue_a)

    base_b = get_base_letter(residue_b)

    pair_key = (base_a, base_b)

    if pair_key not in WC_PAIR_ATOMS:
        return None

    distances = []

    for atom_a, atom_b in WC_PAIR_ATOMS[pair_key]:

        if atom_a not in residue_a:
            return None

        if atom_b not in residue_b:
            return None

        coord_a = residue_a[atom_a].coord

        coord_b = residue_b[atom_b].coord

        distances.append(distance(coord_a, coord_b))

    if not distances:
        return None

    return sum(distances) / len(distances)


def get_base_letter(residue):

    resname = residue.get_resname().strip()

    return BASE_NAME_MAP.get(resname, None)


# def get_base_centroid(residue):
#     """
#     Geometric center of heavy atoms.
#     """

#     coords = []

#     for atom in residue:

#         if atom.element != "H":
#             coords.append(atom.coord)

#     if not coords:
#         return None

#     import numpy as np

#     return np.mean(
#         coords,
#         axis=0
#     )


def find_base_pairs(model):
    """
    Pair DNA bases using Watson-Crick geometry.
    """

    dna_chains = extract_dna_chains(model)

    if len(dna_chains) < 2:

        raise StructureRejected("Less than two DNA chains")

    chain_a = list(dna_chains[0])
    chain_b = list(dna_chains[1])

    used_b = set()

    pairs = []

    for residue_a in chain_a:

        base_a = get_base_letter(residue_a)

        if base_a is None:
            continue

        best_score = float("inf")
        best_match = None

        for idx, residue_b in enumerate(chain_b):

            if idx in used_b:
                continue

            score = compute_wc_pair_score(residue_a, residue_b)

            if score is None:
                continue

            if score < best_score:

                best_score = score
                best_match = (idx, residue_b)

        if best_match is None or best_score > WC_PAIR_DISTANCE_THRESHOLD:

            pairs.append((residue_a, None))

            continue

        idx, residue_b = best_match

        used_b.add(idx)

        pairs.append((residue_a, residue_b))

    paired_count = sum(pair[1] is not None for pair in pairs)

    if paired_count < 5:
        raise StructureRejected("Insufficient paired DNA")

    validate_pairing(pairs)

    return pairs


def validate_pairing(base_pairs):
    """
    Allow terminal overhangs only.
    """

    unpaired = []

    for idx, pair in enumerate(base_pairs):

        if pair[1] is None:
            unpaired.append(idx)

    if not unpaired:
        return

    n = len(base_pairs)

    valid = set()

    i = 0

    while i < n and base_pairs[i][1] is None:
        valid.add(i)
        i += 1

    i = n - 1

    while i >= 0 and base_pairs[i][1] is None:
        valid.add(i)
        i -= 1

    for idx in unpaired:

        if idx not in valid:

            raise StructureRejected("Internal unpaired DNA base")


def load_and_validate(pdb_path):
    """
    Main entry point.
    """

    structure = load_structure(pdb_path)

    validate_structure(structure)

    model = structure[0]

    protein_residues = extract_protein_residues(model)

    dna_pairs = find_base_pairs(model)

    if len(protein_residues) == 0:

        raise StructureRejected("No protein found")

    if len(dna_pairs) == 0:

        raise StructureRejected("No DNA found")

    return (structure, protein_residues, dna_pairs)
