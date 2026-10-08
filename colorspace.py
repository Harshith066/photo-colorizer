"""RGB <-> LAB helpers (NumPy + OpenCV only).

LAB splits a photo into L (lightness = the black-and-white image) and a,b (color).
The model sees L and predicts a,b. Values are normalized to roughly [-1, 1] for the network.
"""
import numpy as np
import cv2


def rgb_to_lab_norm(rgb_uint8):
    """uint8 HxWx3 RGB -> (L, ab) float32 arrays, L: HxWx1 in [-1,1], ab: HxWx2 in [-1,1]."""
    lab = cv2.cvtColor(rgb_uint8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    L = lab[..., :1] / 50.0 - 1.0
    ab = np.clip(lab[..., 1:] / 110.0, -1.0, 1.0)
    return L.astype(np.float32), ab.astype(np.float32)


def lab_norm_to_rgb(L, ab):
    """L: HxWx1, ab: HxWx2 (normalized) -> uint8 HxWx3 RGB."""
    lab = np.concatenate([(L + 1.0) * 50.0, ab * 110.0], axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def gray_rgb(L):
    """Render just the lightness channel as a gray RGB image."""
    return lab_norm_to_rgb(L, np.zeros(L.shape[:2] + (2,), np.float32))


if __name__ == "__main__":  # self-test: round trip should be nearly lossless
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    L, ab = rgb_to_lab_norm(img)
    back = lab_norm_to_rgb(L, ab)
    err = np.abs(back.astype(int) - img.astype(int)).mean()
    g = gray_rgb(L)
    print(f"L range [{L.min():.2f},{L.max():.2f}] ab range [{ab.min():.2f},{ab.max():.2f}]")
    print(f"round-trip mean abs error: {err:.2f} (of 255)")
    print("gray image is gray:", bool(np.abs(g[..., 0].astype(int) - g[..., 1]).max() <= 2))
