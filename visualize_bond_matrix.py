import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def main(npz_path):

    data = np.load(npz_path)

    bond_matrix = data["bond_matrix"]

    protein_labels = data["protein_labels"].astype(str)

    dna_labels = data["dna_labels"].astype(str)

    pdb_id = str(data["pdb_id"])

    rows, cols = bond_matrix.shape

    density = bond_matrix.sum() / bond_matrix.size

    print(f"PDB: {pdb_id}")

    print(f"Shape: {bond_matrix.shape}")

    print(f"Bonds: {int(bond_matrix.sum())}")

    print(f"Density: {density:.6f}")

    print("\nDetected hydrogen bonds")
    print("-" * 50)

    bond_count = 0

    for i in range(bond_matrix.shape[0]):

        for j in range(bond_matrix.shape[1]):

            if bond_matrix[i, j] == 1:

                bond_count += 1

                print(
                    f"[{i:3d}, {j:3d}] "
                    f"{protein_labels[i]} "
                    f"<--> "
                    f"{dna_labels[j]}"
                )

    print("-" * 50)
    print(f"Total bonds: {bond_count}")

    cmap = ListedColormap(["white", "red"])

    fig_width = max(10, cols * 0.5)

    fig_height = max(8, rows * 0.25)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        bond_matrix, cmap=cmap, interpolation="nearest", aspect="auto", vmin=0, vmax=1
    )

    ax.set_xticks(np.arange(cols))

    ax.set_yticks(np.arange(rows))

    ax.set_xticklabels(dna_labels)

    # ax.set_yticklabels(
    #     protein_labels
    # )

    step = max(1, len(protein_labels) // 50)

    ax.set_yticks(np.arange(0, len(protein_labels), step))

    ax.set_yticklabels(protein_labels[::step])

    ##

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.set_xlabel("DNA Base Pairs")

    ax.set_ylabel("Protein Residues")

    ax.set_title(f"{pdb_id} Hydrogen Bond Matrix")

    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)

    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)

    ax.grid(which="minor", color="black", linewidth=0.2)

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.invert_yaxis()

    plt.tight_layout()

    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/{pdb_id}_bond_matrix.png", dpi=300, bbox_inches="tight")

    # plt.show()


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python visualize_bond_matrix.py sample.npz")

        sys.exit(1)

    main(sys.argv[1])
