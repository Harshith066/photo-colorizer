"""Download the Oxford Flowers-102 photos (~8,000 color images) and save them as .jpg files.
Usage: python download_data.py --out data/flowers
(Or skip this and point train.py at ANY folder of color photos.)"""
import argparse, os
from torchvision.datasets import Flowers102

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/flowers")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

n = 0
for split in ("train", "val", "test"):
    ds = Flowers102(root="data/_raw", split=split, download=True)
    for img, _ in ds:
        img.convert("RGB").save(os.path.join(a.out, f"{n:05d}.jpg"), quality=95)
        n += 1
print(f"Saved {n} images to {a.out}")
