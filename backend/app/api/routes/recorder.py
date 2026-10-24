import json
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.dataset_recorder import dataset_recorder, DATASET_ROOT

router = APIRouter(prefix="/api/recorder", tags=["Dataset Recorder"])

class StartRecordingRequest(BaseModel):
    sentence_label: str
    gloss_labels: List[str]
    signer_id: Optional[str] = "signer_01"

class SessionControlRequest(BaseModel):
    session_id: str

@router.post("/start")
def start_recording(req: StartRecordingRequest):
    session_id = dataset_recorder.start_session(
        sentence_label=req.sentence_label,
        gloss_labels=req.gloss_labels,
        signer_id=req.signer_id
    )
    return {"status": "started", "session_id": session_id}

@router.post("/pause")
def pause_recording(req: SessionControlRequest):
    dataset_recorder.pause_session(req.session_id)
    return {"status": "paused", "session_id": req.session_id}

@router.post("/resume")
def resume_recording(req: SessionControlRequest):
    dataset_recorder.resume_session(req.session_id)
    return {"status": "resumed", "session_id": req.session_id}

@router.post("/stop")
def stop_recording(req: SessionControlRequest):
    res = dataset_recorder.stop_session(req.session_id)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return {"status": "stopped", "annotations": res}

@router.post("/discard")
def discard_recording(req: SessionControlRequest):
    dataset_recorder.discard_session(req.session_id)
    return {"status": "discarded", "session_id": req.session_id}

@router.get("/stats")
def get_dataset_stats():
    manifest_path = DATASET_ROOT.parent / "dataset_manifest.json"
    if not manifest_path.exists():
        return {
            "total_recordings": 0,
            "total_frames": 0,
            "gloss_distribution": {},
            "sentence_distribution": {},
            "signer_distribution": {},
            "average_confidence": 0.0
        }

    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)

    total_recordings = len(manifest)
    total_frames = sum(m.get("num_frames", 0) for m in manifest)

    gloss_dist = {}
    sentence_dist = {}
    signer_dist = {}

    for m in manifest:
        s_label = m.get("sentence_label", "unknown")
        sentence_dist[s_label] = sentence_dist.get(s_label, 0) + 1

        signer = m.get("signer_id", "unknown")
        signer_dist[signer] = signer_dist.get(signer, 0) + 1

        for g in m.get("gloss_labels", []):
            gloss_dist[g] = gloss_dist.get(g, 0) + 1

    return {
        "total_recordings": total_recordings,
        "total_frames": total_frames,
        "gloss_distribution": gloss_dist,
        "sentence_distribution": sentence_dist,
        "signer_distribution": signer_dist
    }
