

import os
from torch.utils.data import Dataset
from PIL import Image

CLASS_MAP = {"real": 0, "fake": 1}
IMG_EXTS = (".jpg", ".jpeg", ".png")


class DeepfakeDataset(Dataset):
    def __init__(self, root: str, transform=None):

        self.transform = transform
        self.samples = []  # list of (filepath, label)

        for class_name, label in CLASS_MAP.items():
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMG_EXTS):
                    self.samples.append((os.path.join(class_dir, fname), label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {root}/real or {root}/fake. "
                "Add some images before training."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
