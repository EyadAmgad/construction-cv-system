# Construction Computer Vision System

## 1. Project Overview
This repository contains a prototype demonstrating an end-to-end Computer Vision pipeline and distributed backend architecture for tracking heavy construction equipment. Specifically, it fulfills the requirement to develop a real-time, microservices-based pipeline that processes video clips of construction equipment.

The system tracks utilization states (distinguishing between **ACTIVE** and **INACTIVE** states), classifies specific work activities, and calculates the total time the equipment spends working versus idling. The results are pushed through an Apache Kafka message broker, saved into a PostgreSQL database, and streamed to a real-time Vanilla JS + FastAPI frontend.

---

## 2. Architecture & Setup

### Architecture Overview
The system is built as an event-driven microservices architecture via Docker Compose:
- **CV Pipeline (`cv_service.py`)**: A backend worker handling region-based articulating motion detection, YOLO tracking, and Activity Classification (CNN+LSTM).
- **Apache Kafka**: Acts as the message broker (`equipment-detections` topic). The CV pipeline produces JSON payloads to this broker containing state, activity, bounding boxes, and time counters.
- **Kafka Consumer**: A separate microservice that securely ingests Kafka messages and performs robust SQL `UPSERT` operations into a PostgreSQL database.
- **PostgreSQL**: Serves as the primary data sink (`equipment_events` table) storing utilization and tracking data over time.
- **FastAPI / Uvicorn Server**: Hosts the backend APIs. Processes video uploads, orchestrates the CV pipeline, exposes the UI, and creates a Server-Sent Events (SSE) `/stream` for dynamic frontend dashboard updates.
- **Web Frontend**: A strictly Vanilla JS frontend (served via FastAPI) displaying the processed video frames overlayed with bounding boxes alongside a live Utilization Dashboard (Total Working Time, Total Idle Time, and Utilization percentage).

### Setup Instructions

#### Prerequisites
- Docker & Docker Compose (v2)
- Recommended: At least 8GB of RAM available to Docker.

#### 1. Quick Start
Start the entire stack using Docker Compose. The configuration is built to deploy Kafka, PostgreSQL, PgAdmin, the FastAPI webserver, and the Kafka Consumer securely.

```bash
# Clone the repository
# Make sure to be in the project root directory where docker-compose.yml exists

# Build and Start the microservices
docker compose up -d --build
```
> **Note:** The `docker-compose.yml` restricts Kafka's JVM Heap arguments and utilizes a CPU-only image of PyTorch to ensure the containers can boot reliably on local development machines without crashing due to Out-Of-Memory/disk constraints.

#### 2. Accessing the System
- **Web UI & Dashboard**: Open your browser and navigate to [http://localhost:8000](http://localhost:8000). From here, you can click "Upload Video" to supply a dashcam `.mp4` file and watch the tracking dashboard update dynamically.
- **PgAdmin Dashboard**: Accessible at [http://localhost:5051](http://localhost:5051) (using the credentials located in `.env`). 
- **Graceful Shutdown**:
```bash
docker compose down
```

---

## 3. Technical Write-up: Design Decisions & Trade-offs

### Equipment Utilization Tracking & Articulated Motion Challenge
**The Problem:** Traditional object detection models effectively draw massive boundary boxes around entire vehicles. For standard vehicles (like dump trucks), shifting bounding boxes natively equate to motion. However, articulated equipment like *Excavators* frequently exhibit "arm-only" motion where the tracks and chassis remain completely fixed while the boom digging/swinging happens within the confines of the massive bounding box. Simple centroid-tracking fails here; the machine is actively working but traditional tracking evaluates it as stationary/idling.

**The Solution:**
To satisfy the articulated motion requirement reliably, I paired **MOG2 Background Subtraction** with a temporally-aware **CNN+LSTM architecture**:
1. **Detection & Regional Tracking:** 
   - YOLO handles static bounding bounds tracking for generalized vehicles (trucks). 
   - For articulated machinery (excavators), MOG2 evaluates continuous, dynamic background subtraction frame-by-frame. Even if the cabin is completely rigid, MOG2 catches pixel-level displacement created strictly by moving arms/buckets to flag the region.
2. **Activity Classification (Capturing Temporal Context):**
   - Single-frame algorithms lack context (a bucket at rest looks identical to a bucket moving fast if frozen in an image). Therefore, the pipeline specifically buffers a rolling window of **16 frames**.
   - A **Convolutional Neural Network (CNN)** computes spatial features over the cropped equipment bounding area.
   - A **Long Short-Term Memory (LSTM)** layer receives this 16-frame feature map to evaluate the *temporal dynamic context* of the sequence. It natively recognizes the sequential motion patterns of "Digging", "Swinging/Loading", and "Dumping", as completely distinct from "Waiting" (inactive).

### Dealing with Working vs. Idle Time Calculations
The CV Pipeline locally stores a time-tracker incremented via standard active/inactive boolean flags derived from the CNN+LSTM's confidence predictions.
- **Total Tracked Time** continuously ticks upwards.
- **Total Active Time**: Ticks up if the activity prediction strictly corresponds to an active categorization (Digging, Swinging, Moving).
- **Utilization Percentage**: Calculated strictly as `Total Active Time / Total Tracked Time`.

### Microservices vs. Monolithic Trade-offs
**Decision:** Extracting database write events into an asynchronous Apache Kafka decoupled structure.
- **Trade-off:** Operating Kafka (KRaft) and a dedicated Python Consumer container introduces a memory tax to the local deployment. Orchestrating healthchecks between containers is necessary to prevent premature DB race-conditions.
- **Benefit:** Deep learning video decoding (especially when running on CPU resources for broader Docker portability) is intensely synchronous and computationally bottlenecking. Executing blocking PostgreSQL operations synchronously over every frame parsed would entirely stall the CV framerate throughput. Pushing a lightweight JSON payload of the tracking states onto a Kafka log instantly unblocks the CV Thread; allowing the backend Consumer to bulk-insert metrics behind the scenes smoothly without degrading system stream speed.

---

## 4. Kafka Payload Schema Output
Below is an example of the serialized format shipped via the CV Microservice to the `equipment-detections` Kafka topic exactly as mandated by the requirements:

```json
{
  "timestamp": "2026-03-22T12:00:05.123456",
  "frame_id": 45,
  "equipment_id": "EXCAVATOR",
  "equipment_class": "excavator",
  "bounding_box": [120, 200, 340, 480],
  "state": "ACTIVE",
  "activity": "Digging",
  "metrics": {
    "total_active_seconds": 15.4,
    "total_idle_seconds": 2.1,
    "utilization_percentage": 88.0
  }
}
```

---

## 5. Demo
A working demonstration video showing the system's live UI drawing bounding boxes over the pipeline output and updating the utilization dashboard metrics dynamically based on the Kafka stream:

<video controls src="demo.mp4" title="System Demo" width="100%"></video>

*(If the video does not play in your markdown viewer, you can [click here to view demo.mp4](demo.mp4))*
