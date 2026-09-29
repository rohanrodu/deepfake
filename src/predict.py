import os
import sys
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".mov", ".avi")


def load_model(weights_path: str, device: torch.device):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


def predict_image(path: str, model, device) -> tuple:
    image = Image.open(path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    return torch.argmax(probs).item(), probs.cpu().numpy()


def extract_frames(video_path: str, num_frames: int = 8):
    """Uniformly sample `num_frames` frames across the whole video."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return frames

    indexes = set(np.linspace(0, total - 1, num=min(num_frames, total), dtype=int))
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indexes:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def predict_video(path: str, model, device, num_frames: int = 8) -> tuple:
    frames = extract_frames(path, num_frames)
    if not frames:
        raise ValueError("Could not read any frames from video.")

    all_probs = []
    with torch.no_grad():
        for frame in frames:
            tensor = TRANSFORM(frame).unsqueeze(0).to(device)
            all_probs.append(torch.softmax(model(tensor), dim=1))

    # Average probabilities across frames (soft voting) rather than
    # voting on hard labels -> keeps confidence information intact.
    avg_probs = torch.mean(torch.stack(all_probs), dim=0)[0]
    return torch.argmax(avg_probs).item(), avg_probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Deepfake predictor")
    parser.add_argument("path", help="Path to an image or video file")
    parser.add_argument("--weights", default="models/best_model.pt")
    parser.add_argument("--frames", type=int, default=8, help="Frames to sample for video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    ext = os.path.splitext(args.path)[1].lower()
    if ext in IMAGE_EXTS:
        pred, probs = predict_image(args.path, model, device)
    elif ext in VIDEO_EXTS:
        pred, probs = predict_video(args.path, model, device, args.frames)
    else:
        print(f"Unsupported file type: {ext}")
        return

    label = "FAKE" if pred == 1 else "REAL"
    print(f"\nPrediction: {label}")
    print(f"  Real: {probs[0]:.3f}")
    print(f"  Fake: {probs[1]:.3f}")


if __name__ == "__main__":
    main()
