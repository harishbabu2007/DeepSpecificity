import json
import os
import subprocess
import tempfile

import re
import numpy as np

def get_shape_from_x3dna(pdb_file_path):
    abs_pdb_path = os.path.abspath(pdb_file_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dssr_executable = os.path.join(script_dir, "x3dna-dssr")

    # temporary sandbox folder that self-destructs, no need for cleanup
    with tempfile.TemporaryDirectory() as tmpdir:
        command = [dssr_executable, f"-i={abs_pdb_path}", "--json", "--more"]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=tmpdir,
        )

        if result.stdout.strip():
            json_data = json.loads(result.stdout)
        else:
            return None

    return json_data


def parse_dssr_nt(nt_str):
    """
    Parses a DSSR nucleotide string identifier like 'A.DG3' or 'B.DC40'
    Returns: (chain_id, residue_number)
    """
    if not nt_str or "." not in nt_str:
        return None, None
    parts = nt_str.split(".")
    chain = parts[0]
    res_part = parts[1]  # e.g., "DG3" or "DC40"

    # Extract the digit sequence representing the residue sequence number
    match = re.search(r"(-?\d+)", res_part)
    if match:
        return chain, int(match.group(1))
    return chain, None


def extract_dna_shape_features(json_data, dna_pairs):
    """
    Extracts base-pair and step features from DSSR json and aligns them
    perfectly with the sequence order of the dna_pairs.

    Returns:
        bp_features:  (N_dna, 6)
        step_features: (N_dna, 6)
    """
    n_dna = len(dna_pairs)

    bp_features = np.zeros((n_dna, 6), dtype=np.float32)
    step_features = np.zeros((n_dna, 6), dtype=np.float32)

    if json_data is None:
        return bp_features, step_features

    res_to_node_idx = {}
    for idx, (forward_res, reverse_res) in enumerate(dna_pairs):
        f_chain = forward_res.get_parent().id
        f_num = forward_res.id[1]
        res_to_node_idx[(f_chain, f_num)] = idx

        if reverse_res is not None:
            r_chain = reverse_res.get_parent().id
            r_num = reverse_res.id[1]
            res_to_node_idx[(r_chain, r_num)] = idx

    pairs_list = json_data.get("pairs", [])

    for p in pairs_list:
        c1, n1 = parse_dssr_nt(p.get("nt1"))
        c2, n2 = parse_dssr_nt(p.get("nt2"))

        # Find which idx this DSSR pair belongs to
        node_idx = res_to_node_idx.get((c1, n1)) or res_to_node_idx.get((c2, n2))

        if node_idx is not None:
            vals = p.get("bp_params")
            bp_features[node_idx] = vals

    steps_list = []
    for helix in json_data["helices"]:
        steps_list.extend(helix.get("pairs", []))

    for s in steps_list:
        matched_indices = []
        for nt_key in ["nt1", "nt2"]:
            c, n = parse_dssr_nt(s.get(nt_key))
            if (c, n) in res_to_node_idx:
                matched_indices.append(res_to_node_idx[(c, n)])

        if matched_indices:
            # Map step features to the starting node index (i.e. step leaving node i)
            node_idx = min(matched_indices)
            vals = s.get("step_params")
            step_features[node_idx] = vals

    return bp_features, step_features


def get_dna_shape_features(pdb_path, dna_pairs):
    json_dna_shape_data = get_shape_from_x3dna(pdb_path)

    bp_features, step_features = extract_dna_shape_features(json_dna_shape_data, dna_pairs)
    step_features = np.nan_to_num(step_features, nan=0.0)

    bp_combined_feature = np.concatenate((bp_features, step_features), axis=1)

    return bp_combined_feature


if __name__ == "__main__":
    data = get_shape_from_x3dna("../samples/3VD6.pdb")

    # base pair feature - stretch, stagger, buckle, propeller, opening
