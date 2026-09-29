import mimetypes
import torch
import gradio as gr

from src.model import build_model
from src.predict import TRANSFORM, extract_frames

WEIGHTS_PATH = "models/best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = build_model()
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
MODEL.to(DEVICE).eval()


def classify(file_path):
    if file_path is None:
        return "No file uploaded", ""

    mime, _ = mimetypes.guess_type(file_path)

    with torch.no_grad():
        if mime and mime.startswith("image"):
            from PIL import Image
            img = Image.open(file_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(MODEL(tensor), dim=1)[0]

        elif mime and mime.startswith("video"):
            frames = extract_frames(file_path, num_frames=8)
            if not frames:
                return "Could not read video", ""
            all_probs = []
            for frame in frames:
                tensor = TRANSFORM(frame).unsqueeze(0).to(DEVICE)
                all_probs.append(torch.softmax(MODEL(tensor), dim=1))
            probs = torch.mean(torch.stack(all_probs), dim=0)[0]

        else:
            return "Unsupported file type", ""

    pred = torch.argmax(probs).item()
    label = "🟢 Real" if pred == 0 else "🔴 Fake"
    confidence = f"{probs[pred].item() * 100:.2f}%"
    return label, confidence


demo = gr.Interface(
    fn=classify,
    inputs=gr.File(label="Upload image or video", type="filepath"),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")],
    title="Simple Deepfake Detector",
    description="Real vs Fake classifier.",
)

if __name__ == "__main__":
    demo.launch()
