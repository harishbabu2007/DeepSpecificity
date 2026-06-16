import torch.nn.functional as F
from utils import pwm_to_ppm
import torch


def masked_ppm_loss(logits, target_pwm, mask):
    """
    logits:
        [1, Nd, 4]

    target_pwm:
        [Npwm, 4]

    mask:
        [Nd]
    """
    target_ppm = pwm_to_ppm(target_pwm)

    log_probs = F.log_softmax(logits.squeeze(0), dim=-1)

    target_ppm = target_ppm[mask]
    log_probs = log_probs[mask]

    loss = -(target_ppm * log_probs).sum(dim=-1)

    return loss.mean()

def masked_ppm_loss_with_one_hot(logits, target_pwm, mask, pwm_present, flank_weight=0.5):
    """
    Computes a high-contrast loss function.
    Optimizes the core motif using standard cross-entropy, while aggressively
    forcing flanking/uninformative regions to a perfectly flat uniform distribution (IC=0).
    """
    logits = logits.squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    # pred_probs = F.softmax(logits, dim=-1)

    if pwm_present:
        target_ppm = pwm_to_ppm(target_pwm)

        # Core Motif Region (where mask is True)
        if mask.sum() > 0:
            motif_loss = -(target_ppm[mask] * log_probs[mask]).sum(dim=-1).mean()
        else:
            motif_loss = 0.0

        # Flanking Region (where mask is False) - Flatten background to 0.25
        flank_mask = ~mask
        if flank_mask.sum() > 0:
            uniform_target = torch.full_like(log_probs[flank_mask], 0.25)
            flank_loss = -(uniform_target * log_probs[flank_mask]).sum(dim=-1).mean()
        else:
            flank_loss = 0.0

        return motif_loss + (flank_weight * flank_loss)

    else:
        ce_loss = -(target_pwm * log_probs).sum(dim=-1).mean()

        #  Sharpen unaligned non-contact zones (where target is exactly 0.25)
        is_flank = target_pwm[:, 0] == 0.25
        if is_flank.sum() > 0:
            uniform_target = torch.full_like(log_probs[is_flank], 0.25)
            flank_loss = -(uniform_target * log_probs[is_flank]).sum(dim=-1).mean()
        else:
            flank_loss = 0.0

        return ce_loss + (flank_weight * flank_loss)
