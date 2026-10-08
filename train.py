"""Train the colorizer (GAN: U-Net generator + PatchGAN discriminator, LAB color space).

Example (Colab, ~1 hour):
  python download_data.py --out data/flowers
  python train.py --data data/flowers --max_items 4000 --epochs 20
"""
import argparse, os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import ColorDataset
from models import UNetGenerator, PatchDiscriminator, init_weights
from colorspace import lab_norm_to_rgb, gray_rgb


def to_np(t):  # (C,H,W) tensor -> (H,W,C) numpy
    return t.detach().cpu().permute(1, 2, 0).numpy()


def save_samples(G, L, ab_real, path, device):
    """Rows of [grayscale input | colorized by model | original color]."""
    G.eval()
    with torch.no_grad():
        ab_fake = G(L.to(device)).cpu()
    rows = []
    for i in range(L.size(0)):
        l = to_np(L[i])
        rows.append(np.concatenate([gray_rgb(l), lab_norm_to_rgb(l, to_np(ab_fake[i])),
                                    lab_norm_to_rgb(l, to_np(ab_real[i]))], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of color photos")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1_weight", type=float, default=100.0)
    ap.add_argument("--max_items", type=int, default=4000)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    print("Device:", device)

    train_ds = ColorDataset(a.data, "train", max_items=a.max_items)
    val_ds = ColorDataset(a.data, "val")
    print(f"{len(train_ds)} training images, {len(val_ds)} validation images")
    loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch, num_workers=2)

    G = UNetGenerator(in_ch=1, out_ch=2).to(device)       # L (1 channel) -> ab (2 channels)
    D = PatchDiscriminator(in_ch=3).to(device)            # judges (L, ab) pairs
    G.apply(init_weights); D.apply(init_weights)

    adv, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    optG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))

    vis = [val_ds[i] for i in range(min(4, len(val_ds)))]  # fixed images to track progress
    vis_L = torch.stack([v[0] for v in vis]); vis_ab = torch.stack([v[1] for v in vis])
    histD, histG, histV = [], [], []

    for epoch in range(1, a.epochs + 1):
        G.train(); D.train()
        sD = sG = 0.0
        for L, ab in loader:
            L, ab = L.to(device), ab.to(device)
            fake = G(L)

            # Discriminator: real (L, ab) -> 1, fake (L, ab) -> 0
            pr, pf = D(L, ab), D(L, fake.detach())
            lossD = 0.5 * (adv(pr, torch.ones_like(pr)) + adv(pf, torch.zeros_like(pf)))
            optD.zero_grad(); lossD.backward(); optD.step()

            # Generator: fool D + stay close to the real colors (L1)
            pf = D(L, fake)
            lossG = adv(pf, torch.ones_like(pf)) + a.l1_weight * l1(fake, ab)
            optG.zero_grad(); lossG.backward(); optG.step()
            sD += lossD.item(); sG += lossG.item()

        # Validation: average color error (L1 on ab) on photos the model never trained on
        G.eval(); v = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                v += l1(G(L.to(device)), ab.to(device)).item() * L.size(0)
        histD.append(sD / len(loader)); histG.append(sG / len(loader)); histV.append(v / len(val_ds))
        print(f"Epoch {epoch}/{a.epochs}  D={histD[-1]:.3f}  G={histG[-1]:.3f}  val_color_L1={histV[-1]:.4f}")

        save_samples(G, vis_L, vis_ab, f"{a.out}/epoch_{epoch:03d}.png", device)
        torch.save(G.state_dict(), f"{a.out}/generator.pth")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(histD, label="Discriminator"); ax[0].plot(histG, label="Generator")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(histV); ax[1].set_title("Validation color error (lower = better)"); ax[1].set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(f"{a.out}/loss_curve.png", dpi=150)
    print("Done. Weights saved to", f"{a.out}/generator.pth")


if __name__ == "__main__":
    main()
