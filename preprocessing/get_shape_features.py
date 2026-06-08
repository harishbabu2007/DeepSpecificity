import json
import os
import subprocess
import tempfile


def get_shape_features(pdb_file_path):
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


if __name__ == "__main__":
    data = get_shape_features("../samples/3VD6.pdb")

    if data:
        with open("./dssr-output.json", "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
    else:
        print("No Data")

    # base pair feature - stretch, stagger, buckle, propeller, opening