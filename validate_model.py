import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from pdna_dataset import PDNADataset
from architecture.model import DeepSpecificityWithShape
from losses import masked_ppm_loss_with_one_hot
from utils import split_dna_features_no_seq, split_dna_shape_features

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


def compute_pcc(pred_ppm, target_ppm):
    """
    Pearson correlation between predicted and target PPM across all positions.
    Both inputs are numpy arrays of shape [Nd, 4].
    Returns a scalar correlation value.
    """
    p = pred_ppm.flatten()
    t = target_ppm.flatten()
    if np.std(p) < 1e-9 or np.std(t) < 1e-9:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def compute_jsd(pred_ppm, target_ppm, eps=1e-9):
    """
    Mean Jensen-Shannon Divergence across positions.
    JSD = 0 means identical distributions, 1 means completely different.
    Both inputs are numpy arrays of shape [Nd, 4].
    Returns a scalar.
    """
    pred_ppm = np.clip(pred_ppm, eps, 1.0)
    target_ppm = np.clip(target_ppm, eps, 1.0)

    m = 0.5 * (pred_ppm + target_ppm)
    kl_pm = np.sum(pred_ppm * np.log2(pred_ppm / m), axis=1)
    kl_qm = np.sum(target_ppm * np.log2(target_ppm / m), axis=1)
    jsd_per_pos = 0.5 * (kl_pm + kl_qm)

    return float(np.mean(jsd_per_pos))


def compute_mae(pred_ppm, target_ppm):
    """
    Mean Absolute Error between predicted and target PPM.

    Inputs:
        pred_ppm   : [Nd,4]
        target_ppm : [Nd,4]

    Returns:
        scalar MAE
    """

    return float(np.mean(np.abs(pred_ppm - target_ppm)))


def compute_motif_recovery(pred_ppm, target_ppm, mask):
    """
    For PWM samples only. At each aligned (masked) position, checks whether
    the predicted dominant base matches the target dominant base.
    Returns accuracy as a float in [0, 1].
    mask is a boolean numpy array of shape [Nd].
    """
    if mask.sum() == 0:
        return float("nan")

    pred_base = np.argmax(pred_ppm[mask], axis=1)
    target_base = np.argmax(target_ppm[mask], axis=1)
    return float(np.mean(pred_base == target_base))


def softmax_np(logits):
    """Numerically stable softmax over last axis for numpy arrays."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def validate(checkpoint_path, val_data_dir):
    val_dataset = PDNADataset(data_dir=val_data_dir)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    print(f"Validation samples : {len(val_dataset)}")

    model = DeepSpecificityWithShape(
        len_dna_features=DNA_FEATURE_DIM,
        len_prot_features=PROTEIN_FEATURE_DIM,
        len_dna_shape_features=DNA_SHAPE_FEATURES_DIM,
        d_model=D_MODEL,
        n_head_dna=N_HEAD_DNA,
        n_enc_dna=N_ENC_DNA,
        n_head_prot=N_HEAD_PROT,
        n_enc_prot=N_ENC_PROT,
        n_cross_att_heads=N_CROSS_HEADS,
        n_enc_pwm=N_ENC_PWM,
        n_head_pwm=N_HEAD_PWM,
    ).to(device)
    model = torch.compile(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    trained_epoch = checkpoint.get("epoch", "?")
    model.eval()
    print(f"Loaded checkpoint  : {checkpoint_path}  (epoch {trained_epoch})\n")

    total_loss = 0.0
    total_loss_pwm = 0.0
    total_loss_nopwm = 0.0
    count_all = 0
    count_pwm = 0
    count_nopwm = 0

    all_pcc = []
    all_jsd = []
    pwm_pcc = []
    pwm_jsd = []
    pwm_recovery = []
    nopwm_pcc = []
    nopwm_jsd = []

    all_mae = []
    pwm_mae = []
    nopwm_mae = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")

        for batch in pbar:
            for item in batch:
                prot = item["protein_features"].to(device)
                dna = item["dna_features"].to(device)
                shape = item["dna_shape_features"].to(device)

                dna_fwd, dna_rc = split_dna_features_no_seq(dna)
                shape_fwd, shape_rev = split_dna_shape_features(shape)

                prot = prot.unsqueeze(0)
                dna_fwd = dna_fwd.unsqueeze(0)
                dna_rc = dna_rc.unsqueeze(0)
                shape_fwd = shape_fwd.unsqueeze(0)
                shape_rev = shape_rev.unsqueeze(0)

                pred_fwd = model(dna_fwd, shape_fwd, prot)
                pred_rc = model(dna_rc, shape_rev, prot)

                target_fwd = item["target_pwm_forward"].to(device)
                target_rev = item["target_pwm_reverse"].to(device)
                mask_fwd = item["alignment_mask_forward"].to(device)
                mask_rev = item["alignment_mask_reverse"].to(device)
                pwm_present = item["pwm_present"]

                loss_fwd = masked_ppm_loss_with_one_hot(
                    pred_fwd, target_fwd, mask_fwd, pwm_present
                )
                loss_rev = masked_ppm_loss_with_one_hot(
                    pred_rc, target_rev, mask_rev, pwm_present
                )
                loss = ((loss_fwd + loss_rev) / 2).item()

                # convert logits PPM (numpy) for metric computation
                pred_fwd_ppm = torch.softmax(pred_fwd.squeeze(0), dim=-1).cpu().numpy()
                pred_rev_ppm = torch.softmax(pred_rc.squeeze(0), dim=-1).cpu().numpy()
                tgt_fwd_np = target_fwd.cpu().numpy()
                tgt_rev_np = target_rev.cpu().numpy()
                mask_fwd_np = mask_fwd.cpu().numpy().astype(bool)
                mask_rev_np = mask_rev.cpu().numpy().astype(bool)

                # average fwd + rev for position-level metrics
                pcc = (
                    compute_pcc(pred_fwd_ppm, tgt_fwd_np)
                    + compute_pcc(pred_rev_ppm, tgt_rev_np)
                ) / 2
                jsd = (
                    compute_jsd(pred_fwd_ppm, tgt_fwd_np)
                    + compute_jsd(pred_rev_ppm, tgt_rev_np)
                ) / 2
                mae = (
                    compute_mae(pred_fwd_ppm, tgt_fwd_np)
                    + compute_mae(pred_rev_ppm, tgt_rev_np)
                ) / 2

                all_pcc.append(pcc)
                all_jsd.append(jsd)
                all_mae.append(mae)
                total_loss += loss
                count_all += 1

                if pwm_present:
                    total_loss_pwm += loss
                    count_pwm += 1
                    pwm_pcc.append(pcc)
                    pwm_jsd.append(jsd)
                    pwm_mae.append(mae)

                    rec_fwd = compute_motif_recovery(
                        pred_fwd_ppm, tgt_fwd_np, mask_fwd_np
                    )
                    rec_rev = compute_motif_recovery(
                        pred_rev_ppm, tgt_rev_np, mask_rev_np
                    )
                    if not (np.isnan(rec_fwd) and np.isnan(rec_rev)):
                        valid_recs = [r for r in [rec_fwd, rec_rev] if not np.isnan(r)]
                        pwm_recovery.append(np.mean(valid_recs))
                else:
                    total_loss_nopwm += loss
                    count_nopwm += 1
                    nopwm_pcc.append(pcc)
                    nopwm_jsd.append(jsd)
                    nopwm_mae.append(mae)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else float("nan")
    
    def safe_median(lst):
        return float(np.median(lst)) if lst else float("nan")

    def safe_min(lst):
        return float(np.min(lst)) if lst else float("nan")

    def safe_max(lst):
        return float(np.max(lst)) if lst else float("nan")

    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)

    print(f"\n  Checkpoint : {checkpoint_path}")
    print(f"  Epoch      : {trained_epoch}")
    print(f"  Samples    : {count_all}  " f"(PWM: {count_pwm} | no-PWM: {count_nopwm})")

    print("\n── Overall ──────────────────────────────────────────────")
    print(f"  Loss (avg)           : {total_loss / max(count_all, 1):.6f}")
    print(
        f"  MAE  (avg)           : {safe_mean(all_mae):.4f}"
    )

    print(
        f"  MAE  (median)        : {safe_median(all_mae):.4f}"
    )

    print(
        f"  MAE  (min)           : {safe_min(all_mae):.4f}"
    )

    print(
        f"  MAE  (max)           : {safe_max(all_mae):.4f}"
    )
    # print(
    #     f"  PCC  (avg)           : {safe_mean(all_pcc):.4f}   "
    #     f"(higher is better, max 1.0)"
    # )
    print(
        f"  JSD  (avg)           : {safe_mean(all_jsd):.4f}   "
        f"(lower is better,  min 0.0)"
    )

    print("\n── PWM samples only ─────────────────────────────────────")
    print(f"  Loss                 : {total_loss_pwm / max(count_pwm, 1):.6f}")
    print(f"  MAE                  : {safe_mean(pwm_mae):.4f}")
    print(f"  MAE (median)         : {safe_median(pwm_mae):.4f}")
    print(f"  MAE (min)            : {safe_min(pwm_mae):.4f}")
    print(f"  MAE (max)            : {safe_max(pwm_mae):.4f}")
    # print(f"  PCC                  : {safe_mean(pwm_pcc):.4f}")
    print(f"  JSD                  : {safe_mean(pwm_jsd):.4f}")
    print(
        f"  Motif Recovery       : {safe_mean(pwm_recovery):.4f}   "
        f"(fraction of aligned positions where top base matches)"
    )

    print("\n── No-PWM samples only ──────────────────────────────────")
    print(f"  Loss                 : {total_loss_nopwm / max(count_nopwm, 1):.6f}")
    print(f"  MAE                  : {safe_mean(nopwm_mae):.4f}")
    print(f"  MAE (median)         : {safe_median(nopwm_mae):.4f}")
    print(f"  MAE (min)            : {safe_min(nopwm_mae):.4f}")
    print(f"  MAE (max)            : {safe_max(nopwm_mae):.4f}")
    # print(f"  PCC                  : {safe_mean(nopwm_pcc):.4f}")
    print(f"  JSD                  : {safe_mean(nopwm_jsd):.4f}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate a DeepSpecificityWithShape checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to the validation NPZ directory",
    )
    args = parser.parse_args()

    validate(args.checkpoint, args.val_dir)
