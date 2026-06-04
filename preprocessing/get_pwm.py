import os
import numpy as np
from pyjaspar import jaspardb


def parse_raw_fallback_file(file_path, is_cisbp=False):
    """Parses plain text PWM rows into log-odds log(p/0.25) format."""
    matrix = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                if is_cisbp and parts[0].isalpha():
                    continue  # Skip text header row

                vals = [float(x) for x in parts]
                if is_cisbp and len(vals) == 5:
                    matrix.append(vals[1:])  # Drop index column
                elif not is_cisbp and len(vals) == 4:
                    matrix.append(vals)
    except Exception:
        return None

    mat_arr = np.array(matrix, dtype=np.float32)
    if mat_arr.size == 0:
        return None

    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)

    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)


def parse_uniprobe_file(file_path):
    """Parses horizontal UniPROBE format into vertical log-odds format."""
    matrix_rows = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                parts = line_str.split()
                if not parts:
                    continue

                if parts[0].rstrip(":").upper() in ["A", "C", "G", "T"]:
                    parts = parts[1:]

                try:
                    vals = [float(x) for x in parts]
                    if vals:
                        matrix_rows.append(vals)
                except ValueError:
                    continue
    except Exception:
        return None

    mat_arr = np.array(matrix_rows, dtype=np.float32)
    if mat_arr.size == 0:
        return None

    if mat_arr.shape[0] == 4 and mat_arr.shape[1] != 4:
        mat_arr = mat_arr.T
    elif mat_arr.shape[0] != 4 and mat_arr.shape[1] == 4:
        pass
    else:
        return None

    row_sums = mat_arr.sum(axis=1, keepdims=True)
    ppm = np.divide(mat_arr, row_sums, out=np.zeros_like(mat_arr), where=row_sums != 0)

    ppm = (ppm * 100 + 0.5) / (100 + 2.0)
    return np.log2(ppm / 0.25)


def get_pwm_matrix_from_annotations(
    pdb_id,
    annotations,
    hocomoco_dir="data/motifs/hocomoco",
    cisbp_dir="data/motifs/cisbp",
    uniprobe_dir="data/motifs/uniprobe",
):
    """
    Reads the EXACT matrix file specified by specificity_train.json.
    """
    pdb_id = pdb_id.lower()
    if pdb_id not in annotations or not annotations[pdb_id]:
        return None

    motifs_list = annotations[pdb_id]
    jdb_obj = jaspardb(release="JASPAR2024")

    # Data structure: [ [ ["JASPAR", "MA0152.1.jaspar"], ["HOCOMOCO", "NFAC2_HUMAN.H11MO.0.B"] ] ]
    for site_motifs in motifs_list:
        for motif_info in site_motifs:
            if len(motif_info) != 2:
                continue

            db_name, motif_id = motif_info

            if db_name == "JASPAR":
                clean_id = motif_id.replace(".jaspar", "")
                try:
                    motif = jdb_obj.fetch_motif_by_id(clean_id)
                    if motif:
                        ppm = motif.counts.normalize(pseudocounts=0.5)
                        pwm_dict = ppm.log_odds()
                        pwm_matrix = np.array(
                            [
                                pwm_dict["A"],
                                pwm_dict["C"],
                                pwm_dict["G"],
                                pwm_dict["T"],
                            ],
                            dtype=np.float32,
                        )
                        return pwm_matrix.T
                except Exception:
                    continue

            elif db_name == "HOCOMOCO":
                file_path = os.path.join(hocomoco_dir, f"{motif_id}.pwm")
                mat = parse_raw_fallback_file(file_path, is_cisbp=False)
                if mat is not None:
                    return mat

            elif db_name == "CIS-BP":
                file_path = os.path.join(cisbp_dir, "pwms", f"{motif_id}.txt")
                mat = parse_raw_fallback_file(file_path, is_cisbp=True)
                if mat is not None:
                    return mat

            elif db_name == "UniPROBE":
                if os.path.exists(uniprobe_dir):
                    for root, dirs, files in os.walk(uniprobe_dir):
                        for fname in files:
                            if (
                                motif_id in fname
                                and fname.upper().endswith(".PWM")
                                and ".RC." not in fname.upper()
                            ):
                                full_path = os.path.join(root, fname)
                                mat = parse_uniprobe_file(full_path)
                                if mat is not None:
                                    return mat

    return None
