"""Dataset of ordinary color photos. Each photo gives us BOTH the input (its lightness L)
and the target (its color channels ab), so no paired data is needed."""
import os, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from colorspace import rgb_to_lab_norm

EXTS = (".jpg", ".jpeg", ".png")


class ColorDataset(Dataset):
    def __init__(self, folder, split="train", size=256, max_items=None, val_every=20):
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(EXTS))
        if not files:
            raise FileNotFoundError(f"No images found in {folder}")
        val = files[::val_every]                      # every 20th image is held out for validation
        train = [f for i, f in enumerate(files) if i % val_every != 0]
        self.files = val if split == "val" else train
        if max_items and split == "train":
            self.files = self.files[:max_items]
        self.size, self.augment = size, split == "train"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        if self.augment:                               # resize to 286, random crop 256, random flip
            big = int(self.size * 1.117)
            img = img.resize((big, big), Image.BICUBIC)
            x, y = random.randint(0, big - self.size), random.randint(0, big - self.size)
            img = img.crop((x, y, x + self.size, y + self.size))
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = img.resize((self.size, self.size), Image.BICUBIC)
        L, ab = rgb_to_lab_norm(np.array(img))
        return torch.from_numpy(L).permute(2, 0, 1), torch.from_numpy(ab).permute(2, 0, 1)
