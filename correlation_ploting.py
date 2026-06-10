import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from utils import split_dna_features_no_seq, split_dna_shape_features

device = "cuda" if torch.cuda.is_available() else "cpu"

DNA_SHAPE_NAMES = [
    "Shear",
    "Stretch",
    "Stagger",
    "Buckle",
    "Propeller",
    "Opening",  # Base pair properties
    "Shift",
    "Slide",
    "Rise",
    "Tilt",
    "Roll",
    "Twist",  # Base pair step properties
]
NUCLEOTIDES = ["A", "C", "G", "T"]


def compute_and_save_strand_attributions(
    model, dna_seq_tensor, dna_shape_tensor, protein_tensor, is_rev, output_dir, steps=100
):
    """
    Computes position-specific Integrated Gradients for a single strand pass.
    Generates Nd images, each containing 4 vertically stacked 12xNd heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure standard dimensions: [1, Nd, Features]
    if len(dna_shape_tensor.shape) == 2:
        dna_seq_tensor = dna_seq_tensor.unsqueeze(0)
        dna_shape_tensor = dna_shape_tensor.unsqueeze(0)
        protein_tensor = protein_tensor.unsqueeze(0)

    Nd = dna_shape_tensor.size(1)

    # Define clean reference baseline (all zeros for raw X3DNA coordinates)
    baseline = torch.zeros_like(dna_shape_tensor)
    delta = dna_shape_tensor - baseline  # Shape: [1, Nd, 12]

    # Create path multiplier coordinates
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)

    # Optimize GPU memory footprint by processing position-by-position,
    # while vectorizing all 4 nucleotide output channels together in a parallel batch.
    print(f"--> Generating position attributions for directory: {output_dir}")
    for i in range(Nd):
        # Accumulator for gradients across the linear path integral: shape [4, Nd, 12]
        grad_sum = torch.zeros(4, Nd, 12, device=device)

        # Expand invariant inputs to match the 4 vectorized channel targets
        dna_seq_batch = dna_seq_tensor.repeat(4, 1, 1)
        protein_batch = protein_tensor.repeat(4, 1, 1)

        # Configure output channel targeting mask
        grad_outputs = torch.zeros(4, Nd, 4, device=device)
        for c in range(4):
            grad_outputs[c, i, c] = 1.0  # Isolate output position i, channel c

        for alpha in alphas:
            # Interpolated coordinate step
            interpolated = baseline + alpha * delta
            interpolated_batch = (
                interpolated.repeat(4, 1, 1).detach().requires_grad_(True)
            )

            # Forward logit generation pass
            pwm_logits = model(
                dna_seq_batch, interpolated_batch, protein_batch
            )  # Shape: [4, Nd, 4]

            # Extract local derivatives
            model.zero_grad()
            grads = torch.autograd.grad(
                outputs=pwm_logits,
                inputs=interpolated_batch,
                grad_outputs=grad_outputs,
                retain_graph=False,
            )[0]

            grad_sum += grads

        # Finalize Riemann integration calculations: scale by baseline displacement
        avg_grads = grad_sum / steps
        ig_attribution = (
            (delta.squeeze(0) * avg_grads).cpu().detach().numpy()
        )  # Shape: [4, Nd, 12]

        if is_rev:
            ig_attribution = np.flip(ig_attribution, axis=1)

        # (1 image -> 4 heatmaps)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        for c in range(4):
            # Transpose array from [Nd, 12] to [12, Nd] to align metrics with canvas layout
            heatmap_data = ig_attribution[c].T

            sns.heatmap(
                heatmap_data,
                cmap="RdBu_r",  # Red = Upwards Logit Push, Blue = Downwards Logit Push
                center=0,
                yticklabels=DNA_SHAPE_NAMES,
                xticklabels=True,
                ax=axes[c],
                cbar_kws={"label": "Attribution Score"},
            )
            axes[c].set_title(
                f"Influence of Shape Features on '{NUCLEOTIDES[c]}' Prediction at This Base",
                fontsize=11,
                fontweight="bold",
            )
            axes[c].set_ylabel("Shape Metrics")

        axes[-1].set_xlabel(
            "Physical Coordinate Location of Shape Features (j)", fontsize=12
        )

        # readable 1-indexed filename layout matching model logit positions
        if is_rev:
            base_num = Nd - i
        else:
            base_num = i + 1
        plt.suptitle(
            f"Positional Interpretation Profile — Output Base Position: {base_num}",
            fontsize=15,
            y=0.99,
            fontweight="bold",
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"base_{base_num:02d}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def interpret_sample(model, data_item, base_save_dir="./attributions"):
    """
    Directly unpacks data items, manages strand flipping transformations,
    and saves output directories for both forward and reverse paths.
    """
    pdb_id = data_item["pdb_id"]
    sample_dir = os.path.join(base_save_dir, pdb_id)

    protein_features = data_item["protein_features"].to(device)
    dna_features = data_item["dna_features"].to(device)
    dna_shape_features = data_item["dna_shape_features"].to(device)

    dna_fwd, dna_rc = split_dna_features_no_seq(dna_features)
    dna_shape_fwd, dna_shape_rev = split_dna_shape_features(dna_shape_features)

    #Forward Strand Run
    fwd_dir = os.path.join(sample_dir, "forward_strand")
    compute_and_save_strand_attributions(
        model=model,
        dna_seq_tensor=dna_fwd,
        dna_shape_tensor=dna_shape_fwd,
        protein_tensor=protein_features,
        is_rev=False,
        output_dir=fwd_dir,
    )

    # Reverse Complement Run
    rev_dir = os.path.join(sample_dir, "reverse_strand")
    compute_and_save_strand_attributions(
        model=model,
        dna_seq_tensor=dna_rc,
        dna_shape_tensor=dna_shape_rev,
        protein_tensor=protein_features,
        is_rev=True,
        output_dir=rev_dir,
    )
    print(f"Successfully made attribution maps for [{pdb_id}].\n")
