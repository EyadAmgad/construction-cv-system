"""
PyTorch Dataset for CNN+LSTM training.

Reads dataset/dataset.csv and returns sliding-window sequences of frames
from the same video, each labeled by the class of the last frame in the window.
"""

import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"digging": 0, "loading": 1, "dumping": 2, "waiting": 3}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ExcavatorSequenceDataset(Dataset):
    """
    Sliding window dataset.
    Each sample = (seq_len frames, label_of_last_frame).
    Sequences stay within the same video.
    """

    def __init__(
        self,
        csv_path: str,
        root: str,
        videos: list[str],
        seq_len: int = 16,
        stride: int = 8,
        transform=None,
    ):
        self.root = Path(root)
        self.seq_len = seq_len
        self.transform = transform or VAL_TRANSFORM

        # Load CSV, filter to requested videos
        video_rows: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["video"] in videos:
                    video_rows[row["video"]].append(
                        (int(row["frame"]), row["label"], row["image_path"])
                    )

        # Sort frames per video, build sliding windows
        self.samples: list[list[tuple[str, int]]] = []  # [(img_path, label_idx), ...]
        for video, rows in video_rows.items():
            rows.sort(key=lambda r: r[0])
            for start in range(0, len(rows) - seq_len + 1, stride):
                window = rows[start: start + seq_len]
                self.samples.append(
                    [(r[2], LABEL_MAP[r[1]]) for r in window]
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window = self.samples[idx]
        frames = []
        for img_path, _ in window:
            img = Image.open(self.root / img_path).convert("RGB")
            frames.append(self.transform(img))
        # Label = class of last frame
        label = window[-1][1]
        return torch.stack(frames), torch.tensor(label, dtype=torch.long)
