"""Upload a photo (black & white or color) -> get it colorized.
Run:  python app.py --weights outputs/generator.pth [--share]"""
import argparse
import numpy as np
import cv2
import torch
import gradio as gr
from PIL import Image

from models import UNetGenerator
from colorspace import rgb_to_lab_norm, lab_norm_to_rgb, gray_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="outputs/generator.pth")
ap.add_argument("--share", action="store_true", help="create a public link (useful on Colab)")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
G = UNetGenerator(in_ch=1, out_ch=2).to(device)
G.load_state_dict(torch.load(args.weights, map_location=device))
G.eval()


def colorize(img):
    if img is None:
        return None, None
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))                       # keep big uploads manageable
    rgb = np.array(img)
    L_full, _ = rgb_to_lab_norm(rgb)                  # lightness at FULL resolution (keeps detail sharp)

    # The model works at 256x256; predict colors there, then upscale just the color channels.
    L_small = cv2.resize(L_full[..., 0], (256, 256), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(L_small)[None, None].float().to(device)
    with torch.no_grad():
        ab = G(x)[0].permute(1, 2, 0).cpu().numpy()
    ab = cv2.resize(ab, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

    out = lab_norm_to_rgb(L_full, np.clip(ab, -1, 1))
    return Image.fromarray(gray_rgb(L_full)), Image.fromarray(out)


demo = gr.Interface(
    fn=colorize,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=[gr.Image(label="Grayscale input"), gr.Image(label="Colorized by AI")],
    title="AI Photo Colorizer (GAN, PyTorch)",
    description="Upload a black-and-white (or any) photo. Works best on the kind of images the model was trained on.",
)

if __name__ == "__main__":
    demo.launch(share=args.share)
