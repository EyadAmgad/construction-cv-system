"""
Construction CV — FastAPI Backend

Endpoints:
  GET  /                              health check
  GET  /events                        list equipment events (paginated, filterable)
  GET  /events/{id}                   single event by ID
  GET  /equipment                     list all unique equipment IDs
  GET  /equipment/{equipment_id}      summary + activity breakdown for one equipment
  GET  /equipment/{equipment_id}/timeline  frame-by-frame timeline
  GET  /runs                          list analysis runs
  GET  /runs/{run_id}                 single run + its detections
  GET  /stats                         overall stats (total events, utilization, etc.)

Usage:
    uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import shutil
import threading as _threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .database import close_pool, get_pool
from .models import (
    ActivityBreakdown,
    AnalysisRun,
    Detection,
    EquipmentEvent,
    EquipmentSummary,
)

# ── Frame buffer for live MJPEG stream ────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_OUTPUT_DIR = _ROOT / "output"
_frame_lock = _threading.Lock()
_current_frame_jpeg: bytes = b""
_jobs: dict[str, dict] = {}


def _update_frame(jpeg: bytes) -> None:
    global _current_frame_jpeg
    with _frame_lock:
        _current_frame_jpeg = jpeg


def _read_video_frames(video_path: str, job_id: str) -> None:
    """Background thread: run full cv_service pipeline and push annotated frames to MJPEG buffer."""
    import sys
    sys.path.insert(0, str(_ROOT / "src" / "services"))

    _jobs[job_id]["status"] = "processing"
    try:
        from cv_service import process_video  # noqa: PLC0415
        process_video(
            video_path=video_path,
            frame_callback=_update_frame,
            use_kafka=True,
            use_db=True,
        )
    except Exception as exc:
        print(f"[cv_service] error: {exc}")
    finally:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "done"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_pool()       # warm up connection pool on startup
    yield
    await close_pool()     # clean shutdown


app = FastAPI(
    title="Construction CV API",
    description="Read endpoints for the construction equipment monitoring system.",
    version="1.0.0",
    lifespan=lifespan,
)

# Raise multipart upload limit to 4 GB
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
app.router.on_startup  # touch to ensure router is ready

import starlette.formparsers as _fp  # noqa: E402
_fp.UploadFile.spool_max_size = 4 * 1024 * 1024 * 1024  # 4 GB spool threshold

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles

@app.get("/api/health", tags=["health"])
async def root():
    return {"status": "ok", "service": "construction-cv-api"}


# ── Events ─────────────────────────────────────────────────────────────────────
@app.get("/events", response_model=list[EquipmentEvent], tags=["events"])
async def list_events(
    equipment_id: Optional[str] = Query(None, description="Filter by equipment ID"),
    activity: Optional[str]     = Query(None, description="Filter by activity (DIGGING, LOADING, etc.)"),
    state: Optional[str]        = Query(None, description="Filter by state (ACTIVE / INACTIVE)"),
    limit: int                  = Query(100, ge=1, le=1000),
    offset: int                 = Query(0, ge=0),
):
    pool = await get_pool()
    conditions = []
    params = []
    i = 1
    if equipment_id:
        conditions.append(f"equipment_id = ${i}"); params.append(equipment_id); i += 1
    if activity:
        conditions.append(f"current_activity = ${i}"); params.append(activity.upper()); i += 1
    if state:
        conditions.append(f"current_state = ${i}"); params.append(state.upper()); i += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params += [limit, offset]
    rows = await pool.fetch(
        f"""SELECT * FROM equipment_events {where}
            ORDER BY frame_id, equipment_id
            LIMIT ${i} OFFSET ${i+1}""",
        *params,
    )
    return [dict(r) for r in rows]


@app.get("/events/{event_id}", response_model=EquipmentEvent, tags=["events"])
async def get_event(event_id: int):
    pool = await get_pool()
    row = await pool.fetchrow("SELECT * FROM equipment_events WHERE id = $1", event_id)
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    return dict(row)


# ── Equipment ──────────────────────────────────────────────────────────────────
@app.get("/equipment", tags=["equipment"])
async def list_equipment():
    """Return all unique equipment IDs with their class."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT DISTINCT equipment_id, equipment_class,
                  COUNT(*) as total_events
           FROM equipment_events
           GROUP BY equipment_id, equipment_class
           ORDER BY total_events DESC"""
    )
    return [dict(r) for r in rows]


@app.get("/equipment/{equipment_id}", response_model=EquipmentSummary, tags=["equipment"])
async def get_equipment_summary(equipment_id: str):
    pool = await get_pool()

    base = await pool.fetchrow(
        """SELECT equipment_id, equipment_class,
                  COUNT(*) as total_frames,
                  MAX(total_active_seconds) as total_active_seconds,
                  MAX(total_idle_seconds)   as total_idle_seconds,
                  ROUND(AVG(utilization_percent)::numeric, 2) as avg_utilization_percent
           FROM equipment_events
           WHERE equipment_id = $1
           GROUP BY equipment_id, equipment_class""",
        equipment_id,
    )
    if not base:
        raise HTTPException(status_code=404, detail="Equipment not found")

    activity_rows = await pool.fetch(
        """SELECT current_activity, COUNT(*) as cnt
           FROM equipment_events WHERE equipment_id = $1
           GROUP BY current_activity""",
        equipment_id,
    )
    activities = {r["current_activity"]: r["cnt"] for r in activity_rows}

    return {**dict(base), "activities": activities}


@app.get("/equipment/{equipment_id}/timeline", tags=["equipment"])
async def get_equipment_timeline(
    equipment_id: str,
    limit: int  = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    """Frame-by-frame activity timeline for one equipment."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT frame_id, video_timestamp, current_state,
                  current_activity, utilization_percent
           FROM equipment_events
           WHERE equipment_id = $1
           ORDER BY frame_id
           LIMIT $2 OFFSET $3""",
        equipment_id, limit, offset,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return [dict(r) for r in rows]


# ── Analysis Runs ──────────────────────────────────────────────────────────────
@app.get("/runs", response_model=list[AnalysisRun], tags=["runs"])
async def list_runs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM analysis_runs ORDER BY started_at DESC LIMIT $1 OFFSET $2",
        limit, offset,
    )
    return [dict(r) for r in rows]


@app.get("/runs/{run_id}", tags=["runs"])
async def get_run(run_id: int):
    pool = await get_pool()
    run = await pool.fetchrow("SELECT * FROM analysis_runs WHERE id = $1", run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    detections = await pool.fetch(
        "SELECT * FROM detections WHERE run_id = $1 ORDER BY frame", run_id
    )
    return {"run": dict(run), "detections": [dict(d) for d in detections]}


# ── Stats ──────────────────────────────────────────────────────────────────────
@app.get("/stats", tags=["stats"])
async def get_stats():
    """Overall system statistics."""
    pool = await get_pool()
    totals = await pool.fetchrow(
        """SELECT COUNT(*) as total_events,
                  COUNT(DISTINCT equipment_id) as unique_equipment,
                  COUNT(DISTINCT frame_id) as unique_frames,
                  ROUND(AVG(utilization_percent)::numeric, 2) as avg_utilization
           FROM equipment_events"""
    )
    activity_dist = await pool.fetch(
        """SELECT current_activity, COUNT(*) as count,
                  ROUND(AVG(utilization_percent)::numeric, 2) as avg_util
           FROM equipment_events
           GROUP BY current_activity ORDER BY count DESC"""
    )
    equipment_util = await pool.fetch(
        """SELECT equipment_id, equipment_class,
                  ROUND(AVG(utilization_percent)::numeric, 2) as avg_util
           FROM equipment_events
           GROUP BY equipment_id, equipment_class
           ORDER BY avg_util DESC"""
    )
    return {
        "totals": dict(totals),
        "activity_distribution": [dict(r) for r in activity_dist],
        "equipment_utilization": [dict(r) for r in equipment_util],
    }


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health():
    """Liveness probe."""
    return {"status": "ok"}


# ── Video Upload ───────────────────────────────────────────────────────────────
@app.post("/video/upload", tags=["video"])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a video file. Frames are streamed live via GET /video/feed."""
    # Reset live data on new upload
    try:
        pool = await get_pool()
        await pool.execute("TRUNCATE equipment_events CASCADE")
        await pool.execute("TRUNCATE detections CASCADE")
        await pool.execute("TRUNCATE analysis_runs CASCADE")
    except Exception as e:
        print(f"Warning: Could not wipe prior database run data: {e}")

    global _current_frame_jpeg
    with _frame_lock:
        _current_frame_jpeg = b""

    upload_dir = _OUTPUT_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())[:8]
    dest = upload_dir / f"{job_id}_{file.filename}"

    # Stream to disk in 1 MB chunks — avoids loading the whole file into RAM
    loop = asyncio.get_event_loop()
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await loop.run_in_executor(None, out.write, chunk)
    _jobs[job_id] = {"status": "queued", "video_path": str(dest)}
    background_tasks.add_task(
        lambda: _threading.Thread(
            target=_read_video_frames, args=(str(dest), job_id), daemon=True
        ).start()
    )
    return {"job_id": job_id, "status": "queued", "filename": file.filename}


# ── MJPEG Live Video Feed ──────────────────────────────────────────────────────
async def _mjpeg_generator() -> AsyncGenerator[bytes, None]:
    while True:
        with _frame_lock:
            frame = _current_frame_jpeg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(0.033)  # ~30 fps


@app.get("/video/feed", tags=["video"])
async def video_feed():
    """MJPEG live video stream (annotated frames from the active job)."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Utilization Live ───────────────────────────────────────────────────────────
@app.get("/utilization/live", tags=["utilization"])
async def utilization_live():
    """Latest status snapshot for every tracked equipment."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT DISTINCT ON (equipment_id)
               equipment_id, equipment_class, current_state, current_activity,
               utilization_percent, video_timestamp, frame_id, received_at
           FROM equipment_events
           ORDER BY equipment_id, frame_id DESC"""
    )
    return [dict(r) for r in rows]


# ── Utilization History ────────────────────────────────────────────────────────
@app.get("/utilization/history/{equipment_id}", tags=["utilization"])
async def utilization_history(
    equipment_id: str,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """Frame-by-frame activity history for a single equipment."""
    pool = await get_pool()
    rows = await pool.fetch(
        """SELECT frame_id, video_timestamp, current_state, current_activity,
                  utilization_percent, total_active_seconds, total_idle_seconds
           FROM equipment_events
           WHERE equipment_id = $1
           ORDER BY frame_id
           LIMIT $2 OFFSET $3""",
        equipment_id, limit, offset,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return [dict(r) for r in rows]


# ── SSE Live Stream ────────────────────────────────────────────────────────────
async def _sse_generator() -> AsyncGenerator[str, None]:
    while True:
        pool = await get_pool()
        rows = await pool.fetch(
            """SELECT DISTINCT ON (equipment_id)
                   equipment_id, equipment_class, current_state, current_activity,
                   utilization_percent, total_active_seconds, total_idle_seconds, frame_id
               FROM equipment_events
               ORDER BY equipment_id, frame_id DESC"""
        )
        yield f"data: {json.dumps([dict(r) for r in rows])}\n\n"
        await asyncio.sleep(0.6)


@app.get("/stream", tags=["stream"])
async def sse_stream():
    """Server-Sent Events — pushes live equipment status JSON every ~0.6 s."""
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# Mount the frontend last so it doesn't intercept API routes (like POST /video/upload)
app.mount("/", StaticFiles(directory="src/frontend", html=True), name="frontend")

