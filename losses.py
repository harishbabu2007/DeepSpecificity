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


def masked_ppm_loss_with_one_hot(
    logits, target_pwm, mask, pwm_present, flank_weight=0.1, no_pwm_weight=0.3
):
    logits = logits.squeeze(0)
    log_probs = F.log_softmax(logits, dim=-1)

    target_probs = target_pwm

    is_flank = torch.isclose(
        target_probs, torch.tensor(0.25, device=target_probs.device)
    ).all(dim=-1)

    per_position_ce = -(target_probs * log_probs).sum(dim=-1)

    if is_flank.sum() > 0:
        per_position_ce[is_flank] = per_position_ce[is_flank] * flank_weight

    loss = per_position_ce.mean()

    # Down-weight samples that have no real PWM
    if not pwm_present:
        loss = loss * no_pwm_weight

    return loss
