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
    log_probs = F.log_softmax(logits.squeeze(0), dim=-1)  # Shape: [Nd, 4]
    pred_probs = F.softmax(logits.squeeze(0), dim=-1)     # Shape: [Nd, 4]
    target_ppm = pwm_to_ppm(target_pwm)                  # Shape: [Nd, 4]

    if pwm_present:
        # Case A: Experimental PWM exists (Mask marks the true alignment window)
        motif_mask = mask
        flank_mask = ~mask
        
        # Standard Motif Loss (Inside the Alignment Window)
        if motif_mask.sum() > 0:
            motif_loss = -(target_ppm[motif_mask] * log_probs[motif_mask]).sum(dim=-1).mean()
        else:
            motif_loss = 0.0
            
        # Aggressive Flank Flattening (Outside the Alignment Window)
        if flank_mask.sum() > 0:
            uniform_target = torch.full_like(pred_probs[flank_mask], 0.25)
            # MSE provides a tight, direct constraint to force probabilities to exactly 0.25
            flank_loss = F.mse_loss(pred_probs[flank_mask], uniform_target)
        else:
            flank_loss = 0.0
            
        return motif_loss + (flank_weight * flank_loss)

    else:
        # Case B: No Experimental PWM (Target constructed from sequence + proximity mask)
        # Here, the preprocessing script already filled non-contact zones with 0.25.
        # We identify flanks by checking where the target is exactly uniform (0.25).
        is_flank = (target_ppm[:, 0] == 0.25)
        is_contact = ~is_flank
        
        # Optimize actual physical contact points
        if is_contact.sum() > 0:
            contact_loss = -(target_ppm[is_contact] * log_probs[is_contact]).sum(dim=-1).mean()
        else:
            contact_loss = 0.0
            
        # Flatten non-contact zones
        if is_flank.sum() > 0:
            uniform_target = torch.full_like(pred_probs[is_flank], 0.25)
            flank_loss = F.mse_loss(pred_probs[is_flank], uniform_target)
        else:
            flank_loss = 0.0
            
        return contact_loss + (flank_weight * flank_loss)