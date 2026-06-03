import numpy as np

data = np.load("./data/processed/1an4.npz")


print(data["dna_features"].shape)
print(data["target_pwm_forward"].shape)
print(data["alignment_mask_forward"].shape)
print(data["alignment_mask_forward"].sum())
