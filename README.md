# AI Photo Colorizer (GAN, PyTorch)

Turn a black-and-white photo into a color one. A conditional GAN (Pix2Pix-style) is trained in PyTorch on flower photos and served through a small Gradio web app.

## Demo output

Screenshots from the running app (grayscale input and AI-colorized result):

![Demo screenshot 1](images/demo_1.png)
![Demo screenshot 2](images/demo_2.png)
![Demo screenshot 3](images/demo_3.png)
![Demo screenshot 4](images/demo_4.png)
![Demo screenshot 5](images/demo_5.png)

## How it works

Photos are converted from RGB to **LAB color space**, which separates a photo into:

- **L**: lightness (this is exactly the black-and-white photo)
- **a, b**: the two color channels

The model sees **L** and predicts **a, b**. No paired dataset is needed, because every color photo supplies both the input (its L channel) and the answer (its a/b channels).

| Part | Role |
|---|---|
| **Generator** (U-Net) | Takes L (1 channel), outputs ab (2 channels). Skip connections keep fine detail sharp |
| **Discriminator** (PatchGAN) | Looks at (L, ab) pairs and judges patches as real or fake |
| **Loss** | Adversarial loss + 100 x L1 loss against the real colors |

## Training setup

| Setting | Value |
|---|---|
| Dataset | Oxford Flowers-102 (8,189 photos); 4,000 used for training, 410 held out for validation |
| Epochs | 20 |
| Batch size | 16 |
| Optimizer | Adam, learning rate 2e-4, betas (0.5, 0.999) |
| Image size | 256 x 256 |
| Hardware | Google Colab, T4 GPU |

## Results

- Early in training (epoch 2) the colors were pale and tinted: an orange flower came out as a pastel pink-green-yellow mix.
- By epoch 20 the colors were bold and often plausible: the same flower came out bright orange, and foliage was a convincing green.
- Some colors are still wrong. A gray photo does not say whether a flower is pink or blue, so the model has to guess. Some pink flowers came out lavender or yellow-orange.

**Quantitative check.** Mean absolute error on the color channels, on the 410 validation photos (lower is better):

| | Error |
|---|---|
| Model (epoch 20) | 0.156 |
| Gray baseline (predict no color) | 0.172 |

The model beats the baseline by about 9%, which is a real but modest improvement. The validation error was noisy and did not fall steadily across epochs (epoch 1 scored 0.148, better than epoch 20). A GAN is pushed toward bold, committed colors, which can look better yet score worse on L1 error, so this number understates how the results look. The side-by-side images are the better evidence.

## Limitations

- **Trained only on flowers.** Other subjects (landscapes, faces, street scenes) are likely to look muted or odd.
- **Colorization is ambiguous.** Colors are plausible, not guaranteed correct.
- **Resolution.** The model works at 256 x 256; the app predicts color at that size and upscales only the color channels, keeping the original lightness at full resolution.
- **Single validation split, one training run.** No repeated runs or standard image-quality metrics such as FID.

## Run the demo locally

Python with a virtual environment is recommended. The trained model is not stored in this repository (it is about 217 MB), so download it first:

**Trained model:** `<https://drive.google.com/file/d/1E9jmC7MrkzN8yjP6X-PMmf9vb0RrdXGk/view?usp=sharing>`

Place it at `outputs/generator.pth`, then:

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
python app.py --weights outputs/generator.pth
```

Open http://127.0.0.1:7860, upload a photo, and see the grayscale and colorized versions. Press Ctrl+C in the terminal to stop.

## Train it yourself (Google Colab)

Training needs a GPU. In Colab, choose Runtime > Change runtime type > T4 GPU, upload the project files, then:

```bash
!pip install -q gradio opencv-python
!python download_data.py --out data/flowers
!python train.py --data data/flowers --max_items 4000 --epochs 20
```

This saves the trained model, per-epoch sample images, and a loss curve to `outputs/`. Download `outputs/generator.pth` before the Colab session ends, or write it to Google Drive with `--out`.

## Project files

| File | Purpose |
|---|---|
| `colorspace.py` | RGB and LAB conversion helpers (`python colorspace.py` runs a self-test) |
| `models.py` | U-Net generator and PatchGAN discriminator |
| `dataset.py` | Loads photos and builds (L, ab) pairs with augmentation and a train/validation split |
| `download_data.py` | Downloads Oxford Flowers-102 |
| `train.py` | GAN training loop, validation, sample grids, loss curves |
| `app.py` | Gradio demo: upload a photo, get a colorized result |
| `requirements.txt` | Python dependencies |

## Built with

PyTorch, torchvision, OpenCV, NumPy, Matplotlib, Gradio.
