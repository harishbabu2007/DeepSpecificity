import os
import wandb
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from pdna_dataset import PDNADataset

from architecture.model_v2_shape import DeepSpecificityWithShape
from losses import masked_ppm_loss_with_one_hot

from utils import split_dna_features_no_seq, split_dna_shape_features

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

os.makedirs("checkpoints", exist_ok=True)

wandb.init(
    project="Deep-Specificity",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    },
)

train_dataset = PDNADataset(data_dir=DATA_RAW_NPZ)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: batch,
)

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

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: " f"{num_params:,}")

best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader)

    for batch in pbar:
        optimizer.zero_grad()

        batch_loss = 0.0

        loss_pwm_total = 0.0
        loss_nopwm_total = 0.0
        pwm_count = 0
        nopwm_count = 0

        for item in batch:
            protein_features = item["protein_features"].to(device)
            dna_features = item["dna_features"].to(device)
            dna_shape_features = item["dna_shape_features"].to(device)

            dna_fwd, dna_rc = split_dna_features_no_seq(dna_features)
            dna_shape_features_fwd, dna_shape_features_rev = split_dna_shape_features(dna_shape_features)

            protein_features = protein_features.unsqueeze(0)
            dna_fwd = dna_fwd.unsqueeze(0)
            dna_rc = dna_rc.unsqueeze(0)
            dna_shape_features_fwd = dna_shape_features_fwd.unsqueeze(0)
            dna_shape_features_rev = dna_shape_features_rev.unsqueeze(0)

            pred_fwd = model(dna_fwd, dna_shape_features_fwd, protein_features)
            pred_rc = model(dna_rc, dna_shape_features_rev, protein_features)

            loss_fwd = masked_ppm_loss_with_one_hot(
                pred_fwd,
                item["target_pwm_forward"].to(device),
                item["alignment_mask_forward"].to(device),
                item["pwm_present"],
            )

            loss_rc = masked_ppm_loss_with_one_hot(
                pred_rc,
                item["target_pwm_reverse"].to(device),
                item["alignment_mask_reverse"].to(device),
                item["pwm_present"],
            )

            loss = (loss_fwd + loss_rc) / 2

            batch_loss += loss

            if item["pwm_present"]:
                loss_pwm_total += loss.item()
                pwm_count += 1
            else:
                loss_nopwm_total += loss.item()
                nopwm_count += 1

        if pwm_count > 0:
            wandb.log({"pwm_loss": loss_pwm_total / pwm_count})
        if nopwm_count > 0:
            wandb.log({"nopwm_loss": loss_nopwm_total / nopwm_count})

        batch_loss = batch_loss / len(batch)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += batch_loss.item()

        pbar.set_description(f"Epoch {epoch+1} Loss {batch_loss.item():.6f}")

    epoch_loss = running_loss / len(train_loader)
    scheduler.step()

    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})
    print(f"Epoch {epoch+1}/{EPOCHS} " f"Loss={epoch_loss:.6f}")

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
        },
        "./checkpoints/latest_model_new_shape.pt",
    )

    if epoch_loss < best_loss:
        best_loss = epoch_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            },
            "./checkpoints/best_model_new_shape.pt",
        )

        print(f"Best model saved " f"(loss={best_loss:.6f})")

wandb.finish()
