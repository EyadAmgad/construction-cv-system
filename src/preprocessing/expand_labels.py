"""
Expand sparse keyframe CSV labels to dense per-frame labels.

Rule: each labeled frame's label applies to every frame from that point
up to (but not including) the next labeled frame. The last label extends
to the end of the video (or the last frame in the CSV + step - 1).

Usage:
    python src/expand_labels.py --csv <labels.csv> [--video <video.mp4>]

If --video is provided, the output covers all frames up to the video length.
Otherwise it covers up to the last keyframe's range.

Output: <stem>_expanded.csv  with columns: frame, label
"""

import argparse
import csv
from pathlib import Path


def expand(input_csv: str, total_frames: int | None = None) -> Path:
    rows: list[tuple[int, str]] = []
    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["frame"]), row["label"]))

    rows.sort(key=lambda r: r[0])

    if not rows:
        raise ValueError("CSV is empty")

    out: list[tuple[int, str]] = []

    for i, (frame, label) in enumerate(rows):
        # end of this segment = start of next keyframe (exclusive)
        if i + 1 < len(rows):
            next_frame = rows[i + 1][0]
        elif total_frames is not None:
            next_frame = total_frames
        else:
            # fallback: guess step from previous gap or use 1
            if i > 0:
                step = rows[i][0] - rows[i - 1][0]
            else:
                step = 1
            next_frame = frame + step

        for f in range(frame, next_frame):
            out.append((f, label))

    output_path = Path(input_csv).with_name(Path(input_csv).stem + "_expanded.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "label"])
        writer.writerows(out)

    print(f"✅ {len(out)} frames written to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Sparse keyframe CSV file")
    parser.add_argument("--video", help="Optional: video file to get exact frame count")
    args = parser.parse_args()

    total_frames = None
    if args.video:
        try:
            import cv2
            cap = cv2.VideoCapture(args.video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Video has {total_frames} frames")
        except Exception as e:
            print(f"Warning: could not read video ({e}), using CSV range only")

    expand(args.csv, total_frames)
