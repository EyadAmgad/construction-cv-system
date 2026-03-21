"""
Extract frames from videos and build a complete dataset CSV.

For each expanded labels CSV, extracts every labeled frame as a JPEG image
and writes a dataset CSV with columns: image_path, frame, label, video, split.

Usage:
    python src/build_dataset.py

Output structure:
    dataset/frames/
        <video_stem>/
            frame_00000.jpg
            frame_00001.jpg
            ...
    dataset/dataset.csv   (all videos combined)
"""

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent.parent

# Map each expanded CSV -> its source video
ENTRIES = [
    {
        "csv": ROOT / "train_part_1_labels_expanded.csv",   
        "video": ROOT / "output/train_part_1.mp4",
    },
    {
        "csv": ROOT / "train_part_2_labels_expanded.csv",   
        "video": ROOT / "output/train_part_2.mp4",
    },
    {
        "csv": ROOT / "train_part_4_labels_expanded.csv",     
        "video": ROOT / "output/train_part_4.mp4",
    },
]

FRAMES_DIR = ROOT / "dataset" / "frames"
OUTPUT_CSV = ROOT / "dataset" / "dataset.csv"


def extract_frames(video_path: Path, frame_ids: set[int], out_dir: Path) -> dict[int, Path]:
    """Extract specific frames from a video. Returns {frame_id: image_path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved: dict[int, Path] = {}
    current = 0

    print(f"  Extracting {len(frame_ids)} frames from {video_path.name} ({total} total)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in frame_ids:
            img_path = out_dir / f"frame_{current:06d}.jpg"
            if not img_path.exists():
                cv2.imwrite(str(img_path), frame)
            saved[current] = img_path
            if len(saved) % 500 == 0:
                print(f"    {len(saved)}/{len(frame_ids)} frames saved...")
        current += 1

    cap.release()
    print(f"  ✅ {len(saved)} frames saved to {out_dir}")
    return saved


def main():
    all_rows: list[dict] = []

    for entry in ENTRIES:
        csv_path: Path = entry["csv"]
        video_path: Path = entry["video"]
        video_stem = video_path.stem

        if not csv_path.exists():
            print(f"⚠️  CSV not found: {csv_path}, skipping.")
            continue
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}, skipping.")
            continue

        # Read labels
        rows: list[tuple[int, str]] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((int(row["frame"]), row["label"]))

        frame_ids = {r[0] for r in rows}
        frame_to_label = {r[0]: r[1] for r in rows}

        # Extract frames
        out_dir = FRAMES_DIR / video_stem
        saved = extract_frames(video_path, frame_ids, out_dir)

        # Build rows for dataset CSV
        for frame_id, label in sorted(frame_to_label.items()):
            if frame_id not in saved:
                continue
            img_path = saved[frame_id].relative_to(ROOT)
            all_rows.append({
                "image_path": str(img_path),
                "frame": frame_id,
                "label": label,
                "video": video_stem,
            })

    # Write combined dataset CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "frame", "label", "video"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ Dataset CSV written: {OUTPUT_CSV}  ({len(all_rows)} rows)")
    label_counts = {}
    for r in all_rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    for lbl, cnt in sorted(label_counts.items()):
        print(f"   {lbl}: {cnt}")


if __name__ == "__main__":
    main()
