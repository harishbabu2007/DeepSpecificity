import torch.nn.functional as F
from utils import pwm_to_ppm


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
