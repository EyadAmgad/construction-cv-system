"""Pydantic response models for all API endpoints."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ── equipment_events ──────────────────────────────────────────────────────────
class EquipmentEvent(BaseModel):
    id: int
    received_at: datetime
    frame_id: int
    equipment_id: str
    equipment_class: str
    video_timestamp: str
    current_state: str
    current_activity: str
    total_tracked_seconds: float
    total_active_seconds: float
    total_idle_seconds: float
    utilization_percent: float


# ── analysis_runs ─────────────────────────────────────────────────────────────
class AnalysisRun(BaseModel):
    id: int
    video_path: str
    started_at: datetime
    finished_at: Optional[datetime]
    fps: Optional[float]
    total_frames: Optional[int]


# ── detections ────────────────────────────────────────────────────────────────
class Detection(BaseModel):
    id: int
    run_id: int
    frame: int
    equipment_id: str
    label: str
    confidence: float
    state: str
    active_sec: float
    idle_sec: float
    utilization_pct: float


# ── Summary / aggregations ────────────────────────────────────────────────────
class EquipmentSummary(BaseModel):
    equipment_id: str
    equipment_class: str
    total_frames: int
    total_active_seconds: float
    total_idle_seconds: float
    avg_utilization_percent: float
    activities: dict[str, int]   # activity → frame count


class ActivityBreakdown(BaseModel):
    current_activity: str
    current_state: str
    frame_count: int
    avg_utilization_percent: float
