import torch

def split_dna_features(dna_features):
    dna_fwd = dna_features[:, :70]

    dna_rc = dna_features[:, 70:]
    dna_rc = torch.flip(dna_rc, dims=[0])

    return dna_fwd, dna_rc

def split_dna_shape_features(dna_shape_features):
    dna_shape_features_fwd = dna_shape_features
    dna_shape_features_rev = torch.flip(dna_shape_features, dims=[0])

    return dna_shape_features_fwd, dna_shape_features_rev

def split_dna_features_no_seq(dna_features):
    dna_fwd, dna_rc = split_dna_features(dna_features)

    dna_fwd[:, :4] = 0
    dna_rc[:, :4] = 0

    return dna_fwd, dna_rc


def reverse_complement_pwm(pwm):
    pwm = torch.flip(pwm, dims=[0])
    pwm = pwm[:, [3, 2, 1, 0]]

    return pwm


def pwm_to_ppm(pwm):
    ppm = 0.25 * torch.pow(2.0, pwm)
    ppm = ppm / ppm.sum(dim=-1, keepdim=True)
    return ppm
