"""
Web-based video labeling tool using Gradio (no display required).
Usage:
    python src/label_video.py --video <path_to_video>

Then open the URL shown in the terminal in your browser.

Workflow:
    1. Navigate to a frame → click "Set Start"
    2. Navigate to another frame → click "Set End"
    3. Click a label button → labels every 10th frame in [start, end]
    4. Repeat for each segment
    5. Click "💾 Save CSV" when done
"""

import argparse
import csv
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

LABELS = ["digging", "loading", "dumping", "waiting"]
STEP = 10


def build_app(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    cap.release()

    output_csv = Path(video_path).stem + "_labels.csv"
    frame_labels: dict[int, str] = {}   # frame_id -> label
    intervals: list[dict] = []          # [{start, end, label}]

    def read_frame(idx: int) -> np.ndarray:
        c = cv2.VideoCapture(video_path)
        c.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = c.read()
        c.release()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def intervals_table():
        if not intervals:
            return "No intervals labeled yet."
        rows = "\n".join(
            f"- Frames **{iv['start']}–{iv['end']}** → `{iv['label']}` ({iv['count']} frames)"
            for iv in intervals
        )
        return f"**{len(frame_labels)} total labeled frames**\n\n{rows}"

    def go_to(idx):
        frame = read_frame(idx)
        current = frame_labels.get(int(idx), "—")
        info = f"Frame **{int(idx)}** / {total_frames - 1}  |  Label here: `{current}`"
        return frame, info, intervals_table()

    def apply_interval_label(idx, start_num, end_num, label):
        s, e = int(start_num), int(end_num)
        if s > e:
            s, e = e, s
        count = 0
        for f in range(s, e + 1, STEP):
            frame_labels[f] = label
            count += 1
        intervals.append({"start": s, "end": e, "label": label, "count": count})
        img, info, table = go_to(idx)
        return img, info, table, f"✅ Labeled {count} frames ({s}–{e}) as **{label}**"

    def step(idx, delta):
        new_idx = int(max(0, min(total_frames - 1, int(idx) + delta)))
        img, info, table = go_to(new_idx)
        return new_idx, img, info, table

    def save_csv():
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "label"])
            for fid in sorted(frame_labels):
                writer.writerow([fid, frame_labels[fid]])
        return f"✅ Saved {len(frame_labels)} labels to **{output_csv}**"

    with gr.Blocks(title="Video Labeler") as app:
        gr.Markdown(
            f"## Video Labeler\n`{video_path}`  — {total_frames} frames @ {fps:.0f} fps  |  Step: {STEP} frames per label"
        )

        with gr.Row():
            img = gr.Image(label="Frame", height=480)

        frame_info = gr.Markdown("Frame 0")
        slider = gr.Slider(0, total_frames - 1, step=1, value=0, label="Seek")

        with gr.Row():
            btn_prev = gr.Button("◀ Prev 10")
            btn_next = gr.Button("Next 10 ▶")

        gr.Markdown("### 1. Mark interval")
        with gr.Row():
            num_start = gr.Number(label="Start Frame", value=0, precision=0, minimum=0, maximum=total_frames - 1)
            num_end = gr.Number(label="End Frame", value=0, precision=0, minimum=0, maximum=total_frames - 1)

        gr.Markdown("### 2. Apply label to interval (every 10 frames)")
        with gr.Row():
            label_btns = [gr.Button(lbl) for lbl in LABELS]

        action_status = gr.Markdown("")
        intervals_md = gr.Markdown("No intervals labeled yet.")

        save_btn = gr.Button("💾 Save CSV", variant="primary")
        save_status = gr.Markdown("")

        # Wire up events
        slider.change(go_to, inputs=slider, outputs=[img, frame_info, intervals_md])

        for btn, lbl in zip(label_btns, LABELS):
            btn.click(
                lambda s, e, idx, l=lbl: apply_interval_label(idx, s, e, l),
                inputs=[num_start, num_end, slider],
                outputs=[img, frame_info, intervals_md, action_status],
            )

        btn_prev.click(lambda idx: step(idx, -STEP),
                       inputs=slider, outputs=[slider, img, frame_info, intervals_md])
        btn_next.click(lambda idx: step(idx, STEP),
                       inputs=slider, outputs=[slider, img, frame_info, intervals_md])

        save_btn.click(save_csv, outputs=save_status)
        app.load(lambda: go_to(0), outputs=[img, frame_info, intervals_md])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app = build_app(args.video)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, allowed_paths=["/"])

