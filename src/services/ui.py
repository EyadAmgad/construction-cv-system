"""
Simple UI to run cv_service on a video and display the annotated result.

Usage:
    python src/ui.py
Then open http://localhost:7860
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).parent.parent.parent   # src/services → scripts → project root
SCRIPTS = ROOT / "src"


def run_pipeline(video_file, progress=gr.Progress()):
    if video_file is None:
        return None, None, "⚠️ Please upload a video first."

    out_video = str(ROOT / "output" / "output_annotated.mp4")
    out_csv   = str(ROOT / "output" / "results.csv")

    progress(0, desc="Starting pipeline...")
    cmd = [
        sys.executable, str(SCRIPTS / "services" / "cv_service.py"),
        "--video", video_file,
        "--output", out_csv,
        "--output-video", out_video,
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(ROOT),
    )

    log_lines = []
    total_frames = None
    frame_count = 0

    for line in process.stdout:
        line = line.rstrip()
        # parse total frames from startup line
        if "frames=" in line and total_frames is None:
            try:
                total_frames = int(line.split("frames=")[1].split()[0])
            except Exception:
                pass
        # parse current frame for progress
        if line.startswith("Frame"):
            try:
                frame_count = int(line.split()[1])
                if total_frames:
                    progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
            except Exception:
                pass
        log_lines.append(line)

    process.wait()
    progress(1.0, desc="Done!")

    if process.returncode != 0:
        return None, None, "❌ Pipeline failed:\n" + "\n".join(log_lines[-20:])

    # Parse summary from log
    summary_lines = [l for l in log_lines if "Equipment Summary" in l or l.strip().startswith("EQ-")]
    summary = "\n".join(summary_lines) if summary_lines else "Done."

    # Load CSV for table
    try:
        df = pd.read_csv(out_csv)
        # Keep last result per equipment
        df_summary = (
            df.sort_values("frame")
              .groupby("equipment_id")
              .last()
              .reset_index()[["equipment_id", "label", "confidence", "state", "active_sec", "idle_sec", "utilization_pct"]]
        )
    except Exception:
        df_summary = pd.DataFrame()

    return out_video, df_summary, summary


with gr.Blocks(title="Construction CV System") as app:
    gr.Markdown("# 🏗️ Construction Equipment Activity Monitor")
    gr.Markdown("Upload a site video → YOLO detects equipment → CNN+LSTM classifies activity → annotated video output.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📤 Input Video", height=300)
            run_btn = gr.Button("▶ Run Analysis", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status / Summary", lines=8, interactive=False)

        with gr.Column(scale=2):
            video_output = gr.Video(label="📹 Annotated Output", height=480, autoplay=True)

    gr.Markdown("### Equipment Results")
    results_table = gr.Dataframe(
        headers=["equipment_id", "label", "confidence", "state", "active_sec", "idle_sec", "utilization_pct"],
        label="Final state per equipment",
        interactive=False,
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[video_input],
        outputs=[video_output, results_table, status_box],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[str(ROOT)])
