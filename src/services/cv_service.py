"""
CV Service: per-equipment activity recognition from video.

Pipeline:
  1. MOG2 background subtraction finds large moving objects (excavators)
     YOLO tracks smaller vehicles (trucks, etc.)
  2. Crops are buffered (16 frames) per equipment_id
  3. CNN+LSTM classifies the 16-frame window → activity label + confidence
  4. Time tracker accumulates active/idle seconds → utilization %

Usage:
    python src/services/cv_service.py --video <path_to_video.mp4>
    python src/services/cv_service.py --video <path> --output results.csv
"""

import argparse
import csv
import json
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

try:
    from db import finish_run, insert_detections, start_run
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent   # src/services → scripts → project root
OUTPUT_DIR = ROOT / "output"
CHECKPOINT = OUTPUT_DIR / "best_model.pth"

# ── Kafka ──────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "localhost:9094"
KAFKA_TOPIC     = "equipment-detections"

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ_LEN = 16
MIN_CROP = 32          # skip crops smaller than this in either dimension
NUM_CLASSES = 4

# MOG2 background subtraction tuning
MOG2_HISTORY     = 200   # frames for background model
MOG2_VAR_THRESH  = 40    # sensitivity (lower = more detections)
EXCAVATOR_MIN_AREA = 8000  # px² minimum to be considered an excavator
EXCAVATOR_MAX_COUNT = 3    # keep only the N largest moving regions

LABEL_MAP = {0: "digging", 1: "loading", 2: "dumping", 3: "waiting"}
ACTIVE_LABELS = {"digging", "loading", "dumping"}   # "waiting" → idle

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model loading ──────────────────────────────────────────────────────────────
def load_models(device: torch.device):
    # YOLO — vehicle detection + tracking (trucks, cars, etc.)
    yolo = YOLO(str(OUTPUT_DIR / "yolov8n.pt"))
    yolo.to(device)

    # CNN+LSTM — activity classifier
    sys.path.insert(0, str(ROOT / "src" / "train"))
    from model import CNNLSTM
    cnn_lstm = CNNLSTM(num_classes=NUM_CLASSES).to(device)
    cnn_lstm.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    cnn_lstm.eval()

    # MOG2 — background subtraction for excavator detection
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESH,
        detectShadows=False,
    )

    return yolo, cnn_lstm, mog2


def make_kafka_producer() -> "KafkaProducer | None":
    if not KAFKA_AVAILABLE:
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
        )
        print(f"Kafka producer connected → {KAFKA_BOOTSTRAP}")
        return producer
    except Exception as e:
        print(f"⚠️  Kafka unavailable, continuing without it: {e}")
        return None


def build_payload(frame_id: int, fps: float, equipment_id: str,
                  label: str, state: str,
                  active_sec: float, idle_sec: float,
                  utilization: float) -> dict:
    """Build the standard Kafka JSON payload."""
    total_sec = active_sec + idle_sec
    # Derive equipment class from ID
    equipment_class = "excavator" if equipment_id == "EXCAVATOR" else "truck"
    # Convert frame number to HH:MM:SS.mmm timestamp
    seconds = frame_id / fps if fps else 0
    td = timedelta(seconds=seconds)
    hours, rem = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    timestamp = f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    return {
        "frame_id": frame_id,
        "equipment_id": equipment_id,
        "equipment_class": equipment_class,
        "timestamp": timestamp,
        "utilization": {
            "current_state": state,
            "current_activity": label.upper(),
        },
        "time_analytics": {
            "total_tracked_seconds": round(total_sec, 3),
            "total_active_seconds": round(active_sec, 3),
            "total_idle_seconds": round(idle_sec, 3),
            "utilization_percent": utilization,
        },
    }

# ── Crop helper ────────────────────────────────────────────────────────────────
def safe_crop(frame: np.ndarray, xyxy) -> np.ndarray | None:
    """Crop bbox from frame, clamped to frame bounds. Returns None if too small."""
    h, w = frame.shape[:2]
    x1 = int(max(0, float(xyxy[0])))
    y1 = int(max(0, float(xyxy[1])))
    x2 = int(min(w, float(xyxy[2])))
    y2 = int(min(h, float(xyxy[3])))
    if (x2 - x1) < MIN_CROP or (y2 - y1) < MIN_CROP:
        return None
    return frame[y1:y2, x1:x2]


# ── MOG2 excavator detection ───────────────────────────────────────────────────
def detect_excavators_mog2(frame: np.ndarray, mog2) -> list[tuple[int,int,int,int]]:
    """
    Use background subtraction to find the excavator arm region.
    The camera is mounted looking down at the loading zone — the excavator arm/bucket
    appears at the top of the frame. We search the upper 40% for the largest moving region.
    Returns list of (x1, y1, x2, y2) bounding boxes sorted by area descending.
    """
    h, w = frame.shape[:2]
    # Focus on upper 40% of frame where the excavator arm/bucket is visible
    roi_h = int(h * 0.45)
    roi = frame[:roi_h, :]

    fg_mask = mog2.apply(roi)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < EXCAVATOR_MIN_AREA:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Skip boxes covering >70% of the ROI width (noise/background drift)
        if cw > w * 0.70:
            continue
        # Skip very flat detections
        aspect = cw / ch if ch > 0 else 0
        if aspect > 8 or aspect < 0.1:
            continue
        # y coordinates are in the ROI, no offset needed (already in frame coords since roi starts at 0)
        boxes.append((area, x, y, x + cw, y + ch))

    boxes.sort(key=lambda b: b[0], reverse=True)
    return [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in boxes[:EXCAVATOR_MAX_COUNT]]


@torch.no_grad()
def classify_sequence(crops: list[np.ndarray], model: torch.nn.Module,
                       device: torch.device) -> tuple[str, float]:
    """
    crops: list of 16 BGR numpy arrays
    Returns: (label_string, confidence_float)
    """
    tensors = []
    for crop in crops:
        rgb = crop[..., ::-1].copy()          # BGR → RGB
        pil = Image.fromarray(rgb.astype(np.uint8))
        tensors.append(VAL_TRANSFORM(pil))

    x = torch.stack(tensors).unsqueeze(0).to(device)   # (1, 16, 3, 112, 112)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()
    confidence = round(probs[idx].item(), 2)
    return LABEL_MAP[idx], confidence


# ── Drawing helpers ────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "digging": (0, 165, 255),    # orange
    "loading": (0, 255, 0),      # green
    "dumping": (255, 0, 0),      # blue
    "waiting": (0, 0, 255),      # red
}
DEFAULT_COLOR = (200, 200, 200)


def draw_box(frame: np.ndarray, xyxy, equipment_id: str,
             label: str | None, confidence: float | None,
             utilization: float | None) -> None:
    """Draw bounding box + label overlay on frame in-place."""
    x1 = int(max(0, float(xyxy[0])))
    y1 = int(max(0, float(xyxy[1])))
    x2 = int(min(frame.shape[1], float(xyxy[2])))
    y2 = int(min(frame.shape[0], float(xyxy[3])))

    color = LABEL_COLORS.get(label, DEFAULT_COLOR) if label else DEFAULT_COLOR

    # Excavators get a thicker box; YOLO vehicles get thinner
    thickness = 3 if equipment_id == "EXCAVATOR" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Build overlay text
    if label:
        line1 = f"{equipment_id} | {label} {confidence:.0%}"
        line2 = f"util: {utilization:.1f}%" if utilization is not None else ""
    else:
        line1 = f"{equipment_id} | buffering..."
        line2 = ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1

    for i, text in enumerate([line1, line2]):
        if not text:
            continue
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        ty = y1 - 6 - i * (th + 4)
        tx = x1
        # keep text inside frame
        if ty - th < 0:
            ty = y2 + th + 6 + i * (th + 4)
        cv2.rectangle(frame, (tx, ty - th - 2), (tx + tw + 2, ty + 2), color, -1)
        cv2.putText(frame, text, (tx + 1, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)



def process_video(video_path: str, output_csv: str | None = None,
                  output_video: str | None = None, use_db: bool = False,
                  use_kafka: bool = False, frame_callback=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")
    yolo, cnn_lstm, mog2 = load_models(device)
    print("Models loaded.")

    # ── Kafka producer ───────────────────────────────────────────────────────────
    kafka_producer = make_kafka_producer() if use_kafka else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window_seconds = SEQ_LEN / fps
    print(f"Video: {video_path}  fps={fps:.1f}  frames={total_frames}  {width}x{height}")

    # ── PostgreSQL run record ────────────────────────────────────────────────────
    run_id = None
    if use_db and DB_AVAILABLE:
        try:
            run_id = start_run(video_path, fps, total_frames)
            print(f"DB: started run_id={run_id}")
        except Exception as e:
            print(f"⚠️  DB connection failed, continuing without DB: {e}")
            run_id = None

    # ── Video writer ────────────────────────────────────────────────────────────
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        print(f"Writing annotated video to: {output_video}")

    # Per-equipment state
    frame_buffer: dict[str, deque] = {}
    time_tracker: dict[str, dict[str, float]] = {}
    last_result: dict[str, dict] = {}   # equipment_id → {label, confidence, utilization}

    results_log: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── MOG2: detect large moving objects (excavators) ──────────────────────
        excavator_boxes = detect_excavators_mog2(frame, mog2)

        # ── YOLO: track smaller vehicles (trucks, etc.) ─────────────────────────
        yolo_detections = yolo.track(frame, persist=True, verbose=False)

        # ── Build unified detection list ────────────────────────────────────────
        # Each entry: (equipment_id, xyxy_for_visual_box, crop_for_classifier)
        detections_to_process: list[tuple[str, tuple, np.ndarray | None]] = []

        h_frame, w_frame = frame.shape[:2]

        # ── EXCAVATOR track (always present) ────────────────────────────────────
        # CNN+LSTM was trained on full frames, so always classify the full frame.
        # For the visual bounding box, use the MOG2 detected arm region if available;
        # otherwise fall back to the upper-center area where the arm appears.
        if excavator_boxes:
            exc_xyxy = excavator_boxes[0]   # largest ROI contour
        else:
            # Default: upper-center zone where the excavator arm is typically visible
            exc_xyxy = (w_frame // 4, 0, 3 * w_frame // 4, int(h_frame * 0.45))

        # Always use the full frame for classification (consistent with training data)
        detections_to_process.append(("EXCAVATOR", exc_xyxy, frame.copy()))

        # ── YOLO vehicle detections (trucks, dump trucks, etc.) ─────────────────
        if yolo_detections and yolo_detections[0].boxes is not None:
            for box in yolo_detections[0].boxes:
                if box.id is None:
                    continue
                xyxy = tuple(float(v) for v in box.xyxy[0])
                crop = safe_crop(frame, xyxy)
                detections_to_process.append((f"EQ-{int(box.id)}", xyxy, crop))

        # ── Process each detected equipment ─────────────────────────────────────
        for equipment_id, xyxy, crop in detections_to_process:
            if crop is None:
                continue

            if equipment_id not in frame_buffer:
                frame_buffer[equipment_id] = deque(maxlen=SEQ_LEN)
                time_tracker[equipment_id] = {"active": 0.0, "idle": 0.0}

            frame_buffer[equipment_id].append(crop)

            # ── Classify when buffer is full ───────────────────────────────────
            if len(frame_buffer[equipment_id]) == SEQ_LEN:
                label, confidence = classify_sequence(
                    list(frame_buffer[equipment_id]), cnn_lstm, device
                )
                state = "ACTIVE" if label in ACTIVE_LABELS else "INACTIVE"

                if state == "ACTIVE":
                    time_tracker[equipment_id]["active"] += window_seconds
                else:
                    time_tracker[equipment_id]["idle"] += window_seconds

                active = time_tracker[equipment_id]["active"]
                idle = time_tracker[equipment_id]["idle"]
                total_time = active + idle
                utilization = round(active / total_time * 100, 1) if total_time > 0 else 0.0

                last_result[equipment_id] = {
                    "label": label, "confidence": confidence, "utilization": utilization
                }

                print(
                    f"Frame {frame_idx:6d} | {equipment_id} | "
                    f"{label:10s} ({confidence:.2f}) | {state:8s} | "
                    f"util={utilization:.1f}%"
                )

                results_log.append({
                    "frame": frame_idx,
                    "equipment_id": equipment_id,
                    "label": label,
                    "confidence": confidence,
                    "state": state,
                    "active_sec": round(active, 2),
                    "idle_sec": round(idle, 2),
                    "utilization_pct": utilization,
                })

                # ── Publish to Kafka ───────────────────────────────────────────
                if kafka_producer is not None:
                    payload = build_payload(
                        frame_id=frame_idx,
                        fps=fps,
                        equipment_id=equipment_id,
                        label=label,
                        state=state,
                        active_sec=round(active, 3),
                        idle_sec=round(idle, 3),
                        utilization=utilization,
                    )
                    kafka_producer.send(KAFKA_TOPIC, value=payload)

            # ── Draw on frame ──────────────────────────────────────────────────
            if writer or frame_callback:
                res = last_result.get(equipment_id)
                draw_box(
                    frame, xyxy, equipment_id,
                    res["label"] if res else None,
                    res["confidence"] if res else None,
                    res["utilization"] if res else None,
                )

        # ── Write / stream annotated frame ─────────────────────────────────────
        if writer:
            writer.write(frame)
        if frame_callback:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_callback(buf.tobytes())

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if kafka_producer is not None:
        kafka_producer.flush()
        kafka_producer.close()
        print("Kafka producer flushed and closed.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n── Equipment Summary ──────────────────────────────────────────")
    for eq_id, times in time_tracker.items():
        a, i = times["active"], times["idle"]
        total = a + i
        util = round(a / total * 100, 1) if total > 0 else 0.0
        print(f"  {eq_id}: active={a:.1f}s  idle={i:.1f}s  utilization={util:.1f}%")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if output_csv:
        out = Path(output_csv)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "equipment_id", "label", "confidence",
                "state", "active_sec", "idle_sec", "utilization_pct",
            ])
            writer.writeheader()
            writer.writerows(results_log)
        print(f"\n✅ Results saved to {out}")

    # ── Save to PostgreSQL ─────────────────────────────────────────────────────
    if run_id is not None:
        try:
            insert_detections(run_id, results_log)
            finish_run(run_id)
            print(f"✅ Results saved to PostgreSQL (run_id={run_id})")
        except Exception as e:
            print(f"⚠️  Failed to save to DB: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "results.csv"), help="CSV output path")
    parser.add_argument("--output-video", default=str(OUTPUT_DIR / "output_annotated.mp4"), help="Annotated video output path (.mp4)")
    parser.add_argument("--db", action="store_true", help="Save results directly to PostgreSQL")
    parser.add_argument("--kafka", action="store_true", help="Publish results to Kafka topic")
    args = parser.parse_args()
    process_video(args.video, args.output, args.output_video, use_db=args.db, use_kafka=args.kafka)
