"""
Train CNN+LSTM model on excavator activity dataset.

Usage:
    python src/train/train.py [options]

Default split: random 70/15/15 across all videos (avoids domain shift).
Video split: train on night videos, val/test on day-B (cross-domain).
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))   # ensure train/ is on path
from dataset_loader import (
    LABEL_MAP,
    IDX_TO_LABEL,
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
    ExcavatorSequenceDataset,
)
from model import CNNLSTM

ROOT = Path(__file__).parent.parent.parent   # src/train → scripts → project root
CSV_PATH = ROOT / "dataset" / "dataset.csv"
CHECKPOINT_DIR = ROOT / "output"

ALL_VIDEOS = [
    "301-250219-011424-011654-night-A",
    "302-250420-004123-004718-night-B",
    "302-250420-110359-110900-day-B",
]


def make_weighted_sampler(dataset):
    """Oversample minority classes to balance training."""
    label_counts = Counter()
    for _, label in dataset:
        label_counts[label.item()] += 1
    total = sum(label_counts.values())
    class_weights = {c: total / count for c, count in label_counts.items()}
    sample_weights = [class_weights[dataset[i][1].item()] for i in range(len(dataset))]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(args):
    if args.split == "video":
        train_videos = [
            "301-250219-011424-011654-night-A",
            "302-250420-004123-004718-night-B",
        ]
        val_videos = ["302-250420-110359-110900-day-B"]
        train_ds = ExcavatorSequenceDataset(
            CSV_PATH, ROOT, train_videos,
            seq_len=args.seq_len, stride=args.stride, transform=TRAIN_TRANSFORM,
        )
        full_val = ExcavatorSequenceDataset(
            CSV_PATH, ROOT, val_videos,
            seq_len=args.seq_len, stride=args.stride, transform=VAL_TRANSFORM,
        )
        val_size = int(0.5 * len(full_val))
        test_size = len(full_val) - val_size
        val_ds, test_ds = random_split(full_val, [val_size, test_size],
                                       generator=torch.Generator().manual_seed(42))
    else:
        full_ds = ExcavatorSequenceDataset(
            CSV_PATH, ROOT, ALL_VIDEOS,
            seq_len=args.seq_len, stride=args.stride, transform=VAL_TRANSFORM,
        )
        n = len(full_ds)
        train_size = int(0.70 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size
        train_ds, val_ds, test_ds = random_split(
            full_ds, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        # Apply augmentation to train subset
        train_ds.dataset.transform = TRAIN_TRANSFORM

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)} sequences")

    sampler = make_weighted_sampler(train_ds) if args.balance else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, shuffle=(sampler is None),
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_class_weights(device):
    """Inverse-frequency weights from full dataset for loss function."""
    counts = Counter()
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            counts[LABEL_MAP[r["label"]]] += 1
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts[i] for i in range(len(LABEL_MAP))],
        dtype=torch.float, device=device,
    )
    return weights / weights.sum() * len(LABEL_MAP)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, clip_grad):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(frames)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def test_report(model, loader, device):
    model.eval()
    correct_per_class = Counter()
    total_per_class = Counter()
    correct, total = 0, 0
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        preds = model(frames).argmax(1)
        for p, l in zip(preds, labels):
            total_per_class[l.item()] += 1
            total += 1
            if p == l:
                correct_per_class[l.item()] += 1
                correct += 1
    print(f"\nTest Accuracy: {correct/total:.4f} ({correct}/{total})")
    for idx, name in IDX_TO_LABEL.items():
        t = total_per_class[idx]
        c = correct_per_class[idx]
        print(f"  {name:10s}: {c}/{t} = {c/t:.4f}" if t else f"  {name:10s}: no samples")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(args)

    model = CNNLSTM(
        num_classes=len(LABEL_MAP),
        feature_dim=512,
        hidden_dim=args.hidden_dim,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    class_weights = get_class_weights(device)
    print(f"Class weights: { {IDX_TO_LABEL[i]: f'{w:.3f}' for i, w in enumerate(class_weights.tolist())} }")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler()

    best_val_acc = 0.0
    patience_counter = 0
    log_path = CHECKPOINT_DIR / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.clip_grad)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  ✅ Saved best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  Early stopping after {epoch} epochs (no improvement for {args.patience} epochs)")
                break

    print("\n--- Test Results (best model) ---")
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pth", weights_only=True))
    test_report(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16, help="Frames per sequence")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride (use = seq_len for no overlap)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (0=disabled)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Use weighted sampler to balance classes (default: True)")
    parser.add_argument("--no-balance", dest="balance", action="store_false")
    parser.add_argument("--split", choices=["video", "random"], default="random")
    args = parser.parse_args()
    main(args)
