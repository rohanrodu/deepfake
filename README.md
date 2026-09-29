# Deepfake Detector

A simple Real vs Fake image/video classifier built on **EfficientNetV2-S**
(pretrained on ImageNet, fine-tuned with a custom 2-class head).

## Folder Structure

```
deepfake_detector/
├── data/
│   ├── train/
│   │   ├── real/          ← put real training images here
│   │   └── fake/          ← put fake training images here
│   └── val/
│       ├── real/          ← put real validation images here
│       └── fake/          ← put fake validation images here
├── models/                ← trained weights saved here (best_model.pt)
├── src/
│   ├── __init__.py
│   ├── model.py           ← builds the EfficientNetV2-S classifier
│   ├── dataset.py         ← loads images from real/fake folders
│   ├── train.py           ← training loop
│   └── predict.py         ← image/video inference (CLI)
├── app.py                 ← Gradio web demo
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup

```bash
cd deepfake_detector
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your data

Drop images into the matching folders:

```
data/train/real/*.jpg
data/train/fake/*.jpg
data/val/real/*.jpg
data/val/fake/*.jpg
```

An 80/20 real-world split is typical (80% of your images in `train/`,
20% in `val/`). Keep the two classes reasonably balanced if you can.

## 3. Train

```bash
python src/train.py --epochs 5
```

Useful flags:
- `--data data` — root folder containing `train/` and `val/` (default: `data`)
- `--epochs 5` — number of training epochs
- `--batch_size 16`
- `--lr 1e-4`
- `--out models/best_model.pt` — where to save the best checkpoint
- `--freeze_backbone` — only train the new head (faster, needs less data)

The script saves the model with the lowest validation loss to
`models/best_model.pt` automatically.

## 4. Predict (CLI)

```bash
python src/predict.py path/to/image.jpg  --weights models/best_model.pt
python src/predict.py path/to/video.mp4  --weights models/best_model.pt
```

Videos are handled by sampling 8 evenly-spaced frames across the clip and
averaging their predicted probabilities into one final Real/Fake decision.

## 5. Web demo

```bash
python app.py
```

Opens a local Gradio page where you can drag and drop an image or video.

## How it works (short version)

1. **Data**: images are grouped into `real/` and `fake/` folders.
2. **Model**: EfficientNetV2-S (ImageNet-pretrained) with its classification
   head replaced by `Dropout(0.3) -> Linear(features, 2)`.
3. **Training**: standard PyTorch loop — forward pass, `CrossEntropyLoss`,
   backward pass, Adam optimizer. Best model (lowest val loss) is saved.
4. **Inference**:
   - Image -> single forward pass -> softmax -> Real/Fake + confidence.
   - Video -> 8 sampled frames -> softmax each -> average the
     probabilities -> Real/Fake + confidence.

## Notes / next steps if you want to go further

- Add a face-detection/crop step before classification — most manipulation
  artifacts live in the face region, not the whole frame.
- Try `--freeze_backbone` first if your dataset is small, then fine-tune
  the whole network once you have a working baseline.
- For real-world robustness, test the trained model on images with added
  blur / JPEG compression to see how much accuracy drops.
